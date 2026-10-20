#!/usr/bin/env python3
"""Offline checks of the motion layer against a simulated arm. No robot, no daemon::

    python tests/test_motion.py

The executor is the only thing between a planned move and the motors, and the daemon has no
command watchdog of its own, so each of its safety checks is tested by making it trip.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from humanoid_core.layout import LIMB_JOINTS                                        # noqa: E402
from teleop_station.motion import (Executor, JointLimits, MotionAborted,             # noqa: E402
                                   MotionStopped, SimBus, approach_move, movej)

PASS, FAIL = [], []
DEG = np.pi / 180
NAMES = list(LIMB_JOINTS["left_arm"]) + list(LIMB_JOINTS["right_arm"])
LIM = JointLimits.for_arms("/home/nse/humanoid-studio/configs/humanoid_arms.json")
VMAX, AMAX = 15 * DEG, 60 * DEG


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")


def mid() -> np.ndarray:
    return np.array([(LIM.lower[n] + LIM.upper[n]) / 2 for n in NAMES])


def main() -> int:
    print("\n── planning ─────────────────────────────────────────────────────")
    q0 = mid(); q1 = q0 + np.array([20, -10, 15, 25, 5] * 2) * DEG
    tr = movej(NAMES, q0, q1, VMAX, AMAX)
    v = np.abs(tr.velocity()).max(0)
    check("movej starts and ends exactly", np.allclose(tr.q[0], q0) and np.allclose(tr.q[-1], q1))
    check("every joint finishes together, the slowest at the speed cap",
          abs(np.degrees(v.max()) - 15) < 0.5, f"peak {np.degrees(v.max()):.2f} deg/s")
    check("starts and stops at rest", np.degrees(np.abs(tr.velocity()[[0, -1]])).max() < 0.2)
    check("a planned move passes the checks", LIM.check(tr, VMAX, AMAX) == [], "; ".join(LIM.check(tr, VMAX, AMAX)))
    far = q0.copy(); far[3] = LIM.upper[NAMES[3]] + 10 * DEG
    check("a move past a joint limit is rejected", any("limits" in p for p in LIM.check(movej(NAMES, q0, far, VMAX, AMAX), VMAX, AMAX)))
    check("a move faster than the cap is rejected", any("peaks" in p for p in LIM.check(movej(NAMES, q0, q1, 3 * VMAX, AMAX), VMAX, AMAX)))
    ap = approach_move(NAMES, q0, q1, 3 * DEG, np.ones(len(NAMES)), VMAX, AMAX)
    k_pre = int(np.argmin(np.abs(ap.q[:, 0] - (q1[0] + 3 * DEG))))
    check("approach goes via the point 3 deg beyond the target", np.allclose(ap.q[k_pre], q1 + 3 * DEG, atol=1e-3)
          and np.allclose(ap.end, q1))
    check("the final approach is slower", np.abs(ap.velocity()[k_pre:]).max() < np.abs(ap.velocity()[:k_pre]).max())

    print("\n── executor against a simulated arm ─────────────────────────────")
    logs = []
    sim = SimBus(dict(zip(NAMES, q0)))
    ex = Executor(sim, NAMES, LIM, log=logs.append)
    q_en = ex.enable()
    check("enable puts every joint in POSITION, holding where it is",
          all(s == "ENABLED" for s in sim.state.values()) and np.allclose(q_en, q0, atol=1e-3))
    short = movej(NAMES, q_en, q_en + 4 * DEG, VMAX, AMAX)
    t0 = time.monotonic(); ex.run(short, VMAX, AMAX); dt = time.monotonic() - t0
    check("a move streams in real time", abs(dt - short.duration) < 0.2, f"{dt:.2f} s for a {short.duration:.2f} s move")
    s = ex.settle()
    check("the arm settles on the target", s["settled"], f"{max(map(abs, s['err_deg'])):.2f} deg")
    try:
        ex.run(movej(NAMES, q0, q1, VMAX, AMAX), VMAX, AMAX)
        check("a trajectory that doesn't start at the arm is refused", False)
    except MotionAborted:
        check("a trajectory that doesn't start at the arm is refused", not sim.estopped)
    ex.release(settle_s=0.05)
    check("release ends limp", all(v == "IDLE" for v in sim.state.values()))

    for kind, label in (("stall", "a joint that stops following"), ("stale", "telemetry that stops"),
                        ("error", "a firmware fault")):
        sim = SimBus(dict(zip(NAMES, q0)), faults={NAMES[1]: (0.4, kind)})
        ex = Executor(sim, NAMES, LIM)
        qe = ex.enable()
        try:
            ex.run(movej(NAMES, qe, qe + np.r_[0, 30, 0, 0, 0, 0, 0, 0, 0, 0] * DEG, VMAX, AMAX), VMAX, AMAX)
            check(f"E-stop on {label}", False, "move completed")
        except MotionAborted as e:
            check(f"E-stop on {label}", sim.estopped and all(v == "DAMPING" for v in sim.state.values()), str(e)[:70])

    sim = SimBus(dict(zip(NAMES, q0)))
    ex = Executor(sim, NAMES, LIM)
    qe = ex.enable()
    import threading
    threading.Timer(0.5, ex.request_stop).start()
    try:
        ex.run(movej(NAMES, qe, qe + 20 * DEG, VMAX, AMAX), VMAX, AMAX)
        check("an operator stop holds instead of E-stopping", False, "move completed")
    except MotionStopped:
        check("an operator stop holds instead of E-stopping", not sim.estopped and
              all(v == "ENABLED" for v in sim.state.values()))

    bad = dict(zip(NAMES, q0)); bad[NAMES[0]] = LIM.upper[NAMES[0]] + 5 * DEG
    try:
        Executor(SimBus(bad), NAMES, LIM).enable()
        check("won't enable with a joint outside the planning limits", False)
    except MotionAborted:
        check("won't enable with a joint outside the planning limits", True)
    try:
        Executor(SimBus(dict(zip(NAMES, q0)), state="OFFLINE"), NAMES, LIM).enable()
        check("won't enable offline joints", False)
    except MotionAborted:
        check("won't enable offline joints", True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
