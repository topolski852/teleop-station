"""
Paths taught by hand: a raw recording of the joints as someone guided the limp arms, turned
into something the executor can replay smoothly.

The raw recording has the hand's tremor, hesitations and uneven speed in it. ``plan_path``:

1. finds the **stops**: every place all joints stayed within ``still_deg`` for ``dwell_s``
   (the person paused there). These are where the repeatability run measures; because the same
   path is replayed every cycle, each stop is always reached from the same side, which keeps
   gearbox backlash out of the measurement without an extra approach move,
2. smooths each stretch between two stops (Gaussian, ``smooth_s``) with the stops pinned,
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
        }) + "\n")

    @classmethod
    def load(cls, path) -> "PlannedPath":
        d = json.loads(Path(path).read_text())
        segs = [Trajectory(d["names"], np.array(s["t"]), np.array(s["q"])) for s in d["segments"]]
        return cls(d["names"], [np.array(s) for s in d["stops"]], segs, d["closes_loop"],
                   d["deviation_deg"], d.get("problems", []), d.get("params", {}))


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
              close_loop_deg: float = 1.0, dt: float = 0.02) -> PlannedPath:
    names, t, q = raw.names, raw.t, raw.q
    params = dict(vmax=vmax, amax=amax, still_deg=still_deg, dwell_s=dwell_s, smooth_s=smooth_s,
                  min_move_deg=min_move_deg, max_deviation_deg=max_deviation_deg)
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
        smooth = _gaussian_smooth(seg_raw, smooth_s * fs)
        # Pin the ends exactly on the stops, blending the correction out over the segment.
        w = np.linspace(1, 0, len(smooth))[:, None]
        smooth += w * (stops[k] - smooth[0]) + (1 - w) * (stops[k + 1] - smooth[-1])
        dev = _path_distance_deg(smooth, seg_raw)
        deviation.append(dev)
        if dev > max_deviation_deg:
            problems.append(f"segment {k + 1} (stop {k + 1} -> {k + 2}) strays {dev:.1f} deg from the "
                            f"recording (max {max_deviation_deg:.1f}); smooth less or re-record it")
        segments.append(_retime(names, smooth, vmax, amax, limits, dt))
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
