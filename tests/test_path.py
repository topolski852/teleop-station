#!/usr/bin/env python3
"""Offline checks of taught-path planning, on a synthetic hand-guided recording. No robot::

    python tests/test_path.py

The recording imitates a person guiding the limp arms: pauses of ~2 s at a few places, uneven
moves between them with hand tremor, and a hesitation mid-move. The planner must find the
pauses, keep the smoothed path close to what was recorded, and produce moves that pass every
check the executor applies before streaming.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from humanoid_core.layout import LIMB_JOINTS                                       # noqa: E402
from teleop_station.motion import (Executor, JointLimits, PlannedPath, RawPath,     # noqa: E402
                                   SimBus, plan_path)

PASS, FAIL = [], []
DEG = np.pi / 180
NAMES = list(LIMB_JOINTS["left_arm"]) + list(LIMB_JOINTS["right_arm"])
LIM = JointLimits.for_arms("/home/nse/humanoid-studio/configs/humanoid_arms.json")
VMAX, AMAX = 15 * DEG, 60 * DEG
RNG = np.random.default_rng(4)


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")


def recording(stops, dwell=2.2, fs=50.0):
    t, q, now = [], [], 0.0
    def add(qs, dur, tremor):
        nonlocal now
        n = int(dur * fs)
        for k in range(n):
            ph = now * 2 * np.pi * 7.0
            q.append(qs(k / max(1, n - 1)) + tremor * DEG * np.sin(ph + np.arange(len(NAMES))) +
                     RNG.normal(0, 0.03 * DEG, len(NAMES)))
            t.append(now); now += 1 / fs
    for i, s in enumerate(stops):
        add(lambda u, s=s: s, dwell, 0.08)                                 # holding still-ish
        if i + 1 < len(stops):
            a, b = s, stops[i + 1]
            # an uneven human move with a hesitation two-thirds of the way
            def move(u, a=a, b=b):
                w = np.clip(u * 1.4, 0, 1) if u < 0.5 else 0.7 + 0.3 * np.clip((u - 0.7) / 0.3, 0, 1)
                w = 3 * w ** 2 - 2 * w ** 3
                return a + (b - a) * w
            add(move, 2.5 + RNG.uniform(0, 1.5), 0.25)
    return RawPath(NAMES, np.array(t), np.array(q))


def main() -> int:
    mid = np.array([(LIM.lower[n] + LIM.upper[n]) / 2 for n in NAMES])
    stops = [mid, mid + np.r_[25, 10, 0, 20, 0, 0, 0, 0, 0, 0] * DEG,
             mid + np.r_[0, 0, 0, 0, 0, -20, -10, 15, -25, 10] * DEG, mid]
    raw = recording(stops)

    print("\n── planning a hand-guided recording ─────────────────────────────")
    plan = plan_path(raw, LIM, VMAX, AMAX)
    check("finds every pause as a stop", len(plan.stops) == len(stops), f"{len(plan.stops)} stops")
    err = max(np.degrees(np.abs(a - b)).max() for a, b in zip(plan.stops, stops)) if len(plan.stops) == len(stops) else 99
    check("stops land where the arm paused (within 0.3 deg)", err < 0.3, f"{err:.2f} deg")
    check("the hesitation mid-move is not a stop", len(plan.stops) == 4)
    check("smoothed path stays within 2 deg of the recording", max(d for d in plan.deviation_deg if d == d) < 2.0,
          f"max {max(d for d in plan.deviation_deg if d == d):.2f} deg")
    check("no problems reported", plan.problems == [], "; ".join(plan.problems))
    joined = all(np.allclose(s.q[0], plan.stops[k]) and np.allclose(s.q[-1], plan.stops[(k + 1) % len(plan.stops)])
                 for k, s in enumerate(plan.segments))
    check("every segment runs stop to stop, and the last closes the loop", joined)
    check("every segment passes the executor's checks",
          all(LIM.check(s, VMAX, AMAX) == [] for s in plan.segments))
    v = max(np.abs(s.velocity()).max() for s in plan.segments)
    check("speed cap respected and actually used", 10 < np.degrees(v) <= 15.75, f"peak {np.degrees(v):.1f} deg/s")
    jerk = max(np.abs(np.diff(s.acceleration(), axis=0)).max() / 0.02 for s in plan.segments)
    jerk_raw = np.abs(np.diff(np.gradient(np.gradient(raw.q, raw.t, axis=0), raw.t, axis=0), axis=0)).max() / 0.02
    check("much smoother than the recording (peak jerk)", jerk < jerk_raw / 10,
          f"{np.degrees(jerk):.0f} vs {np.degrees(jerk_raw):.0f} deg/s^3")

    with tempfile.TemporaryDirectory() as d:
        plan.save(Path(d) / "p.json"); back = PlannedPath.load(Path(d) / "p.json")
        raw.save(Path(d) / "r.json"); rb = RawPath.load(Path(d) / "r.json")
    check("planned and raw paths round-trip through JSON",
          len(back.segments) == len(plan.segments) and np.allclose(back.segments[1].q, plan.segments[1].q, atol=1e-6)
          and np.allclose(rb.q, raw.q, atol=1e-6))

    print("\n── edge cases ───────────────────────────────────────────────────")
    open_end = recording(stops[:3])
    p2 = plan_path(open_end, LIM, VMAX, AMAX)
    check("a recording that ends elsewhere gets a planned closing move, flagged",
          p2.closes_loop and any("NOTE" in p for p in p2.problems) and np.allclose(p2.segments[-1].q[-1], p2.stops[0]))
    p3 = plan_path(RawPath(NAMES, raw.t[:40], raw.q[:40] + np.linspace(0, 0.3, 40)[:, None]), LIM, VMAX, AMAX)
    check("a recording with no pauses is refused, with a hint", p3.segments == [] and "pause" in p3.problems[0])
    far = [mid, mid + np.r_[0, 0, 0, 120, 0, 0, 0, 0, 0, 0] * DEG, mid]
    p4 = plan_path(recording(far), LIM, VMAX, AMAX)
    check("a path past a joint limit is reported", any("limits" in p for p in p4.problems))

    print("\n── streaming one segment to a simulated arm ─────────────────────")
    sim = SimBus(dict(zip(NAMES, plan.stops[0])))
    ex = Executor(sim, NAMES, LIM)
    ex.enable()
    ex.last_target = plan.stops[0]
    ex.run(plan.segments[0], VMAX, AMAX)
    s = ex.settle()
    check("the executor runs a planned segment and settles at the next stop", s["settled"] and
          np.allclose(ex.last_target, plan.stops[1]))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
