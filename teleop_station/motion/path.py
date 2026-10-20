"""
Paths taught by hand: a raw recording of the joints as someone guided the limp arms, turned
into something the executor can replay smoothly.

The raw recording has the hand's tremor, hesitations and uneven speed in it. ``plan_path``:

1. finds the **stops**: every place all joints stayed within ``still_deg`` for ``dwell_s``
   (the person paused there). These are where the repeatability run measures; because the same
   path is replayed every cycle, each stop is always reached from the same side, which keeps
   gearbox backlash out of the measurement without an extra approach move,
2. splits each stretch between two stops wherever the hand nearly stopped (a reversal or a
   sharp turn; below ``via_speed_deg``), and smooths each piece (Gaussian, ``smooth_s``) with
   its ends pinned. The arm passes those points at zero speed but doesn't measure there,
3. keeps the **shape** and discards the timing: the stretch is resampled by distance along
   the path and re-timed with a minimum-jerk profile along it, so it eases out of one stop and
   into the next with no step in acceleration, and lengthened until every joint is under the
   speed and acceleration caps,
4. measures how far the smoothed path strays from what was actually recorded, as a shape
   (the nearest recorded point, not the recorded point at the same moment: smoothing a
   hesitation changes the timing, not where the arm goes). The recording is the known-clear
   path (it was physically traversed), so a smoothed path that strays more than
   ``max_deviation_deg`` from it is refused rather than run.

Everything is DEVICE frame radians, as recorded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .limits import JointLimits
from .trajectory import Trajectory, movej

_PEAK_VEL = 1.875


@dataclass
class RawPath:
    names: list[str]
    t: np.ndarray            # (N,) s
    q: np.ndarray            # (N, n) device rad

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"kind": "raw_path", "names": self.names,
                                          "t": self.t.round(4).tolist(), "q": self.q.round(6).tolist()}) + "\n")

    @classmethod
    def load(cls, path) -> "RawPath":
        d = json.loads(Path(path).read_text())
        return cls(d["names"], np.array(d["t"], float), np.array(d["q"], float))


@dataclass
class PlannedPath:
    names: list[str]
    stops: list[np.ndarray]                  # where each stop is
    segments: list[Trajectory]               # stop k -> stop k+1 (and the loop closure, if any)
    closes_loop: bool                        # the closing move back to stop 1 was never traversed
    deviation_deg: list[float]               # per segment, max over joints, smoothed vs recorded
    problems: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    # Optional: run once from ``rest`` to stops[0] before cycling, and once back after.
    # ``combine_arms`` uses them so cycles stay in the air and the table is touched only at
    # the start and end, along the recorded path.
    rest: np.ndarray | None = None
    lead_in: list[Trajectory] = field(default_factory=list)
    lead_out: list[Trajectory] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "kind": "planned_path", "names": self.names, "params": self.params,
            "stops": [s.tolist() for s in self.stops], "closes_loop": self.closes_loop,
            "deviation_deg": self.deviation_deg, "problems": self.problems,
            "segments": [{"t": s.t.round(4).tolist(), "q": s.q.round(6).tolist()} for s in self.segments],
            "rest": None if self.rest is None else self.rest.round(6).tolist(),
            "lead_in": [{"t": s.t.round(4).tolist(), "q": s.q.round(6).tolist()} for s in self.lead_in],
            "lead_out": [{"t": s.t.round(4).tolist(), "q": s.q.round(6).tolist()} for s in self.lead_out],
        }) + "\n")

    @classmethod
    def load(cls, path) -> "PlannedPath":
        d = json.loads(Path(path).read_text())
        tr = lambda lst: [Trajectory(d["names"], np.array(s["t"]), np.array(s["q"])) for s in lst]
        return cls(d["names"], [np.array(s) for s in d["stops"]], tr(d["segments"]), d["closes_loop"],
                   d["deviation_deg"], d.get("problems", []), d.get("params", {}),
                   None if d.get("rest") is None else np.array(d["rest"]),
                   tr(d.get("lead_in", [])), tr(d.get("lead_out", [])))


def find_dwells(t, q, still_deg: float, dwell_s: float) -> list[tuple[int, int]]:
    """(first, last) sample index of every stretch where all joints stayed within still_deg."""
    still_rad = np.radians(still_deg)
    dwells, i, n = [], 0, len(t)
    while i < n:
        j = i
        lo, hi = q[i].copy(), q[i].copy()
        while j + 1 < n:
            lo2, hi2 = np.minimum(lo, q[j + 1]), np.maximum(hi, q[j + 1])
            if (hi2 - lo2).max() > still_rad:
                break
            lo, hi, j = lo2, hi2, j + 1
        if t[j] - t[i] >= dwell_s:
            dwells.append((i, j))
            i = j + 1
        else:
            i += 1
    return dwells


def _gaussian_smooth(q: np.ndarray, sigma_samples: float) -> np.ndarray:
    """Smooth along time with the ends held (padded with the end values)."""
    if sigma_samples < 0.5 or len(q) < 3:
        return q.copy()
    r = int(np.ceil(3 * sigma_samples))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_samples) ** 2)
    k /= k.sum()
    pad = np.vstack([np.repeat(q[:1], r, 0), q, np.repeat(q[-1:], r, 0)])
    return np.stack([np.convolve(pad[:, j], k, mode="valid") for j in range(q.shape[1])], axis=1)


def _via_points(q: np.ndarray, fs: float, via_speed_deg: float, min_gap_s: float = 0.4) -> list[int]:
    """Indices where the joint-space speed dips to a local minimum below ``via_speed_deg``/s."""
    if len(q) < 5 or via_speed_deg <= 0:
        return []
    speed = np.degrees(np.linalg.norm(np.gradient(q, axis=0), axis=1)) * fs
    k = max(1, int(0.2 * fs))
    sp = np.convolve(speed, np.ones(k) / k, mode="same")
    gap = max(1, int(min_gap_s * fs))
    out = []
    for i in range(gap, len(q) - gap):
        if sp[i] < via_speed_deg and sp[i] == sp[i - gap:i + gap + 1].min():
            if not out or i - out[-1] >= gap:
                out.append(i)
    return out


def _corners(q: np.ndarray, corner_deg: float, step_deg: float = 3.0) -> list[int]:
    """Indices where the path turns by more than ``corner_deg`` within about ``step_deg`` of
    travel, however fast the hand was going: a corner the arm has to stop at."""
    if len(q) < 5 or corner_deg <= 0:
        return []
    d = np.linalg.norm(np.diff(q, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    step = np.radians(step_deg)
    if s[-1] < 3 * step:
        return []
    out = []
    for i in range(1, len(q) - 1):
        a = int(np.searchsorted(s, s[i] - step)); b = int(np.searchsorted(s, s[i] + step))
        if a >= i or b >= len(q) or b <= i:
            continue
        u, v = q[i] - q[a], q[b] - q[i]
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-9 or nv < 1e-9:
            continue
        ang = np.degrees(np.arccos(np.clip(u @ v / (nu * nv), -1, 1)))
        if ang > corner_deg and (not out or s[i] - s[out[-1]] > 2 * step):
            out.append(i)
    return out


def _path_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    """How far apart two paths are as SHAPES, not in time: the largest distance from any
    point of either to the nearest point of the other (Hausdorff, worst joint, degrees).
    A hesitation along the way changes the timing but not the shape, so it doesn't count."""
    def directed(x, y):
        worst = 0.0
        for i in range(0, len(x), 256):
            d = np.abs(x[i:i + 256, None, :] - y[None, :, :]).max(-1)     # Chebyshev, rad
            worst = max(worst, float(d.min(1).max()))
        return worst
    return float(np.degrees(max(directed(a, b), directed(b, a))))


def _retime(names, qs: np.ndarray, vmax, amax, limits: JointLimits | None, dt: float) -> Trajectory:
    """A geometric path (M, n) -> a minimum-jerk-in-distance trajectory under the caps."""
    d = np.linalg.norm(np.diff(qs, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    L = s[-1]
    if L < 1e-6:
        return movej(names, qs[0], qs[-1], vmax, amax, dt)
    # Joint speed = |dq/ds| * ds/dt, and min-jerk peaks ds/dt at 1.875 L / T.
    slope = np.abs(np.diff(qs, axis=0)) / np.maximum(d[:, None], 1e-12)
    T = max(0.3, _PEAK_VEL * L * float(slope.max()) / vmax)
    # Acceleration q'' = q_s s'' + q_ss s'^2: the second term is the path's curvature, at
    # min-jerk's peak speed. Estimate q_ss on a ~1 deg resampling (noise in a dense recording
    # would otherwise look like curvature) and size T so both terms fit under amax.
    n_rs = max(5, int(L / np.radians(1.0)))
    sr = np.linspace(0, L, n_rs)
    qr = np.stack([np.interp(sr, s, qs[:, j]) for j in range(qs.shape[1])], axis=1)
    h = sr[1] - sr[0]
    q_ss = np.abs(np.diff(qr, 2, axis=0)).max() / h ** 2 if n_rs > 2 else 0.0
    T = max(T, float(np.sqrt((5.7735 * L * float(slope.max()) + (_PEAK_VEL * L) ** 2 * q_ss) / amax)))
    for _ in range(40):
        n = max(2, int(np.ceil(T / dt)) + 1)
        tt = np.linspace(0, T, n)
        tau = tt / T
        ss = L * (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5)
        q = np.stack([np.interp(ss, s, qs[:, j]) for j in range(qs.shape[1])], axis=1)
        tr = Trajectory(list(names), tt, q)
        bad = [p for p in (limits.check(tr, vmax, amax) if limits else []) if "peaks" in p]
        if not bad:
            return tr
        T *= 1.15
    return tr


def plan_path(raw: RawPath, limits: JointLimits | None, vmax: float, amax: float, *,
              still_deg: float = 0.5, dwell_s: float = 1.5, smooth_s: float = 0.25,
              min_move_deg: float = 1.0, max_deviation_deg: float = 2.0,
              close_loop_deg: float = 1.0, via_speed_deg: float = 3.0, corner_deg: float = 60.0,
              dt: float = 0.02) -> PlannedPath:
    names, t, q = raw.names, raw.t, raw.q
    params = dict(vmax=vmax, amax=amax, still_deg=still_deg, dwell_s=dwell_s, smooth_s=smooth_s,
                  min_move_deg=min_move_deg, max_deviation_deg=max_deviation_deg, via_speed_deg=via_speed_deg,
                  corner_deg=corner_deg)
    dwells = find_dwells(t, q, still_deg, dwell_s)
    problems = []
    if not dwells:
        return PlannedPath(names, [], [], False, [], ["no pauses found: pause ~2 s at the start, "
                                                      "at each stop and at the end"], params)
    # Merge consecutive dwells that are the same place (a hesitation, not a move).
    stops, spans = [], []
    for a, b in dwells:
        m = q[a:b + 1].mean(0)
        if stops and np.degrees(np.abs(m - stops[-1]).max()) < min_move_deg:
            spans[-1] = (spans[-1][0], b)
            stops[-1] = q[spans[-1][0]:b + 1].mean(0)
        else:
            stops.append(m); spans.append((a, b))
    params["stop_spans_s"] = [[float(t[a]), float(t[b])] for a, b in spans]
    fs = 1.0 / np.median(np.diff(t))
    segments, deviation = [], []
    for k in range(len(stops) - 1):
        a, b = spans[k][1], spans[k + 1][0]                  # leave one stop, reach the next
        seg_raw = np.vstack([stops[k], q[a:b + 1], stops[k + 1]])
        # Split at every point where the hand nearly stopped (a reversal or a turn): those are
        # corners in the path, and a single smooth profile can only get round a corner by
        # crawling the whole stretch. Each piece is one stroke; the arm passes through the
        # split points at zero speed, as the hand did, but doesn't measure there.
        cuts = sorted(set([0] + _via_points(seg_raw, fs, via_speed_deg)
                          + _corners(seg_raw, corner_deg) + [len(seg_raw) - 1]))
        pieces, dev = [], 0.0
        for c0, c1 in zip(cuts, cuts[1:]):
            part = seg_raw[c0:c1 + 1]
            smooth = _gaussian_smooth(part, smooth_s * fs)
            # Pin the piece's ends exactly, blending the correction out along it.
            w = np.linspace(1, 0, len(smooth))[:, None]
            smooth += w * (part[0] - smooth[0]) + (1 - w) * (part[-1] - smooth[-1])
            dev = max(dev, _path_distance_deg(smooth, part))
            pieces.append(_retime(names, smooth, vmax, amax, limits, dt))
        seg = pieces[0]
        for piece in pieces[1:]:
            seg = seg + piece
        deviation.append(dev)
        if dev > max_deviation_deg:
            problems.append(f"segment {k + 1} (stop {k + 1} -> {k + 2}) strays {dev:.1f} deg from the "
                            f"recording (max {max_deviation_deg:.1f}); smooth less or re-record it")
        segments.append(seg)
    closes = False
    if len(stops) > 1:
        # Cycling needs the last stop to lead back to the first. That closing move is planned,
        # not recorded: if it's more than close_loop_deg it was never traversed by hand.
        gap = float(np.degrees(np.abs(stops[-1] - stops[0]).max()))
        if gap > 1e-3:
            segments.append(movej(names, stops[-1], stops[0], vmax, amax, dt))
            deviation.append(float("nan"))
            closes = gap > close_loop_deg
            if closes:
                problems.append(f"NOTE (not an error): the recording ends {gap:.1f} deg from where it "
                                f"started; the closing move back is planned, not recorded. Check it in "
                                f"the preview, or end the recording where it began.")
    if limits:
        for k, s in enumerate(segments):
            problems.extend(f"segment {k + 1}: {p}" for p in limits.check(s, vmax, amax))
        for k, s in enumerate(stops):
            problems.extend(f"stop {k + 1}: {p}" for p in limits.violations(names, s))
    return PlannedPath(names, stops, segments, closes, deviation, problems, params)


def _synchronise(names, parts, dt: float = 0.02, min_time: float = 0.3) -> Trajectory:
    """Run several arms' moves at once, finishing together: each is slowed (time-scaled,
    which keeps its shape and its smooth start and stop) to the longest one's duration.
    ``parts`` is [(column indices, Trajectory | None, hold position)]."""
    T = max([min_time] + [tr.duration for _, tr, _ in parts if tr is not None])
    n = max(2, int(np.ceil(T / dt)) + 1)
    t = np.linspace(0, T, n)
    q = np.zeros((n, len(names)))
    for cols, tr, hold in parts:
        if tr is None:
            q[:, cols] = hold
        else:
            tt = t / T * tr.duration
            q[:, cols] = np.stack([np.interp(tt, tr.t, tr.q[:, j]) for j in range(tr.q.shape[1])], axis=1)
    return Trajectory(list(names), t, q)


def _reverse(tr: Trajectory) -> Trajectory:
    return Trajectory(tr.names, tr.t[-1] - tr.t[::-1], tr.q[::-1].copy())


def arm_clearance(names, trajs, hand_m: float = 0.11, step: int = 5) -> tuple[float, int, float]:
    """Closest the two arms' centre-lines come over some trajectories: (metres, which
    trajectory, at what time). Each arm is its joint chain from the shoulder roll joint out to
    a point ``hand_m`` along the wrist link (the hand), in core's URDF geometry, so both arms
    share the base frame. Compare with the links' thickness, not zero."""
    from humanoid_core.arm.kinematics import ArmChain
    from humanoid_core.layout import LIMB_JOINTS
    chains, cols = {}, {}
    for side in ("left", "right"):
        jn = list(LIMB_JOINTS[f"{side}_arm"])
        if not all(n in names for n in jn):
            return float("inf"), -1, 0.0
        chains[side], cols[side] = ArmChain(jn), [names.index(n) for n in jn]

    def polyline(side, q):
        c = chains[side]
        qu = c.device_to_urdf(q[cols[side]])
        F = c.frames(qu)
        pts = [f[:3, 3] for f in F[1:]] + [F[-1][:3, :3] @ np.array([0, 0, hand_m]) + F[-1][:3, 3]]
        return np.concatenate([np.linspace(a, b, 8) for a, b in zip(pts, pts[1:])])

    best = (float("inf"), -1, 0.0)
    for k, tr in enumerate(trajs):
        for i in range(0, len(tr.t), step):
            a, b = polyline("left", tr.q[i]), polyline("right", tr.q[i])
            d = float(np.linalg.norm(a[:, None] - b[None], axis=2).min())
            if d < best[0]:
                best = (d, k, float(tr.t[i]))
    return best


def combine_arms(raw: RawPath, limits: JointLimits | None, vmax: float, amax: float,
                 sides=("left", "right"), dt: float = 0.02, min_clearance_m: float = 0.15,
                 **kw) -> PlannedPath:
    """Arms recorded one after the other, run together.

    Each arm's part of the recording is planned on its own (``plan_path``: its pauses become
    its stops). Its first stop is where it rested before moving; the rest are in the air,
    except its last, where it was put down. The combined path then:

    - **lead-in**: both arms, together, from rest to their first stop in the air, along the
      recording,
    - **cycle**: stop i of the left arm paired with stop i of the right, both moving at once
      and finishing together (an arm with fewer stops waits at its last). The loop back from
      the last stop to the first is a planned move in the air,
    - **lead-out**: the lead-in in reverse, so the table is only approached along the path
      that was actually taken. Where each arm was put down at the end isn't used.
    """
    # Only the chosen arms' joints are in the plan: a one-arm plan carries no commands for
    # the other arm at all (rather than zeros that would drive it somewhere).
    names = [n for n in raw.names if n.split("_")[0] in sides]
    raw = RawPath(names, raw.t, raw.q[:, [raw.names.index(n) for n in names]])
    per = {}
    for side in sides:
        cols = [i for i, n in enumerate(names) if n.startswith(side)]
        sub = RawPath([names[i] for i in cols], raw.t, raw.q[:, cols])
        plan = plan_path(sub, limits, vmax, amax, dt=dt, **kw)
        n_moves = max(0, len(plan.stops) - 1)
        per[side] = dict(cols=cols, plan=plan, moves=plan.segments[:n_moves],
                         dev=plan.deviation_deg[:n_moves], stops=plan.stops)
    problems = []
    for side, d in per.items():
        if len(d["stops"]) < 3:
            problems.append(f"{side} arm: {len(d['stops'])} stops found; it needs to leave rest, pause "
                            f"somewhere in the air, and come back")
    if problems:
        return PlannedPath(names, [], [], False, [], problems, dict(kw, combined=True))
    rest = np.zeros(len(names))
    for d in per.values():
        rest[d["cols"]] = d["stops"][0]
    # In-air stops: everything between leaving rest and being put down.
    air = {s: d["stops"][1:-1] for s, d in per.items()}
    M = max(len(v) for v in air.values())
    def stop_q(i):
        q = np.zeros(len(names))
        for s, d in per.items():
            q[d["cols"]] = air[s][min(i, len(air[s]) - 1)]
        return q
    stops = [stop_q(i) for i in range(M)]
    lead_in = _synchronise(names, [(d["cols"], d["moves"][0], None) for d in per.values()], dt)
    segments, deviation = [], []
    for i in range(M - 1):
        parts, dev = [], 0.0
        for s, d in per.items():
            k = i + 1                                  # move k goes from stop k to k+1
            if i + 1 < len(air[s]):
                parts.append((d["cols"], d["moves"][k], None)); dev = max(dev, d["dev"][k])
            else:
                parts.append((d["cols"], None, air[s][-1]))
        segments.append(_synchronise(names, parts, dt)); deviation.append(dev)
    closes = False
    if M > 1:
        segments.append(movej(names, stops[-1], stops[0], vmax, amax, dt)); deviation.append(float("nan"))
        closes = float(np.degrees(np.abs(stops[-1] - stops[0]).max())) > 1.0
        if closes:
            problems.append("NOTE (not an error): the loop back from the last stop to the first is a planned "
                            "move in the air, not recorded. Check it in the preview.")
    lead_out = _reverse(lead_in)
    max_dev = kw.get("max_deviation_deg", 2.0)
    for s, d in per.items():
        for k, dev in enumerate(d["dev"][:len(air[s])]):
            if dev > max_dev:
                problems.append(f"{s} arm move {k + 1} strays {dev:.1f} deg from the recording (max {max_dev:.1f})")
    # Each arm's part was safe as recorded because the other arm sat on the table. Run
    # together they can meet, which no one ever tried by hand: check the clearance.
    program = [lead_in] + segments
    d, k, at = arm_clearance(names, program)
    params_clear = round(d, 4)
    if d < min_clearance_m:
        what = "the lead-in (and lead-out)" if k == 0 else f"cycle move {k}"
        problems.append(f"the arms come within {d * 1000:.0f} mm of each other (centre-lines, min "
                        f"{min_clearance_m * 1000:.0f}) during {what} at t={at:.1f} s: they would collide "
                        f"running together. Run one arm at a time (--arms left, then --arms right), or "
                        f"record poses that keep them apart.")
    if limits:
        for label, trs in (("lead-in", [lead_in]), ("cycle move", segments)):
            for k, tr in enumerate(trs):
                problems.extend(f"{label} {k + 1}: {p}" for p in limits.check(tr, vmax, amax))
        for k, q in enumerate(stops):
            problems.extend(f"stop {k + 1}: {p}" for p in limits.violations(names, q))
    params = dict(kw, combined=True, vmax=vmax, amax=amax, clearance_m=params_clear,
                  min_clearance_m=min_clearance_m, stops_per_arm={s: len(v) for s, v in air.items()})
    return PlannedPath(names, stops, segments, closes, deviation, problems, params, rest, [lead_in], [lead_out])


def mirror_arm(plan: PlannedPath, limits: JointLimits | None, vmax: float, amax: float,
               min_clearance_m: float = 0.15) -> PlannedPath:
    """A one-arm plan (the left arm's) run on both arms at once, the right mirroring the left.

    In device frame the mirror is simply right = -left on every joint: with both shoulder
    pitches counting against the URDF axis (core joint_sign -1), core's kinematics put the
    right arm at -q exactly where the left arm at q is, reflected left-right (checked to
    0.00 mm). Physically that holds as far as both arms are zeroed. Then the same checks as
    any two-arm plan: limits (the right arm's own, which aren't an exact mirror) and clearance
    (mirrored arms that both reach in toward the middle meet there).
    """
    left = [n for n in plan.names if n.startswith("left_")]
    if left != plan.names:
        raise ValueError("mirror_arm takes a left-arm-only plan")
    right = ["right_" + n[len("left_"):] for n in left]
    names = left + right

    def both(q):
        q = np.atleast_2d(q)
        return np.hstack([q, -q])

    def tr(t):
        return Trajectory(names, t.t.copy(), both(t.q))

    stops = [both(q)[0] for q in plan.stops]
    out = PlannedPath(names, stops, [tr(s) for s in plan.segments], plan.closes_loop,
                      list(plan.deviation_deg), [p for p in plan.problems if p.startswith("NOTE")],
                      dict(plan.params, mirrored=True),
                      None if plan.rest is None else both(plan.rest)[0],
                      [tr(s) for s in plan.lead_in], [tr(s) for s in plan.lead_out])
    out.problems += [p for p in plan.problems if not p.startswith("NOTE")]
    d, k, at = arm_clearance(names, out.lead_in + out.segments)
    out.params["clearance_m"] = round(d, 4)
    if d < min_clearance_m:
        what = "the lead-in (and lead-out)" if (out.lead_in and k == 0) else f"move {k + (0 if out.lead_in else 1)}"
        out.problems.append(f"mirrored, the arms come within {d * 1000:.0f} mm of each other (centre-lines, min "
                            f"{min_clearance_m * 1000:.0f}) during {what} at t={at:.1f} s: keep the left arm "
                            f"on its own side, out of the middle")
    if limits:
        for label, trs in (("lead-in", out.lead_in), ("move", out.segments)):
            for i, t in enumerate(trs):
                out.problems.extend(f"{label} {i + 1}: {p}" for p in limits.check(t, vmax, amax)
                                    if p.split(" ")[0].startswith("right_"))
        for i, q in enumerate(out.stops):
            out.problems.extend(f"stop {i + 1}: {p}" for p in limits.violations(names, q) if p.startswith("right_"))
    return out
