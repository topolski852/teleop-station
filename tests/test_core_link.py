#!/usr/bin/env python3
"""Offline check that the station sees the hardware it is built for, through humanoid-core.

No robot. Run::

    python tests/test_core_link.py

The station is two 5-joint arms and nothing else. If a core change ever shifts the arm joint
catalog or the geometry, this is the first test that should notice, before any station code
plans a move against the wrong arm.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from humanoid_core.arm.kinematics import ArmChain, chain_for     # noqa: E402
from humanoid_core.daemon import DaemonClient                    # noqa: E402
from humanoid_core.layout import LIMB_JOINTS, RobotLayout        # noqa: E402

PASS, FAIL = [], []
ARMS = ("left_arm", "right_arm")


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")


def main() -> int:
    print("\n── the station layout ───────────────────────────────────────────")
    layout = RobotLayout(enabled=ARMS, imu_expected=False)
    check("two arms, no legs", layout.arms == ARMS and layout.legs == ())
    check("ten joints, five per arm", len(layout.joint_order) == 10
          and all(len(LIMB_JOINTS[a]) == 5 for a in ARMS), f"{len(layout.joint_order)} joints")
    check("both arms can be driven in one session", "dual_arm_teleop" in layout.capabilities)
    check("no leg capability leaks in", "walk" not in layout.capabilities)

    print("\n── geometry from core ───────────────────────────────────────────")
    chains = {a: chain_for(a, LIMB_JOINTS[a]) for a in ARMS}
    check("both arm chains build", all(isinstance(c, ArmChain) and c.n == 5
                                       for c in chains.values()))
    mid = {a: (c.limits_lower + c.limits_upper) / 2.0 for a, c in chains.items()}
    tool = {a: chains[a].tool(mid[a]) for a in ARMS}
    check("the arms are mirror images (wrists on opposite sides at mid-range)",
          tool["left_arm"][1] > 0 > tool["right_arm"][1],
          f"y left {tool['left_arm'][1]:+.3f} m, right {tool['right_arm'][1]:+.3f} m")
    lo, hi = chains["left_arm"].reach_bounds()
    check("reach is a plausible arm length", 0.1 < hi < 1.0, f"{lo:.3f}..{hi:.3f} m")
    check("left and right reach agree within 1 mm",
          abs(chains["right_arm"].reach_bounds()[1] - hi) < 1e-3)

    print("\n── daemon client ────────────────────────────────────────────────")
    check("DaemonClient imports from core", callable(DaemonClient))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
