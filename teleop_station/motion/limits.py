"""
Where each joint may go, in DEVICE frame radians, and checking a trajectory against it.

Two tiers exist on this robot. The URDF's soft limits in core (``ArmChain``, URDF frame) are
the ones teleop clamps to; the daemon config's ``position_limits`` are the wider hard tier the
ESC firmware enforces. ``tier="soft"`` (the default) plans inside both, with a margin, so the
firmware clamp is never what stops a move. ``tier="hard"`` plans inside the hard tier only,
with the margin: much more room (yaw about -105..+85 deg against +-45), for poses taught by
hand that the URDF's conservative range excludes. The limits the ESCs actually enforce are
whatever they last had written (humanoid-web's connect writes none, and a zero-offset write
shifts them), so they may differ from this file; treat the hard tier as the planner's bound,
not a promise that the firmware stops there.

The URDF limits are converted with the chain's ``frame_sign`` (both shoulder pitches count
opposite to the URDF axis), which can swap lower and upper.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from humanoid_core.arm.kinematics import ArmChain
from humanoid_core.layout import LIMB_JOINTS

from .trajectory import Trajectory


@dataclass
class JointLimits:
    lower: dict[str, float]
    upper: dict[str, float]

    @classmethod
    def for_arms(cls, daemon_config: str | Path | None = None, arms=("left_arm", "right_arm"),
                 margin_rad: float = np.radians(3.0), tier: str = "soft") -> "JointLimits":
        if tier not in ("soft", "hard"):
            raise ValueError(f"tier must be soft or hard, not {tier!r}")
        if tier == "hard" and not daemon_config:
            raise ValueError("the hard tier comes from the daemon config")
        lower, upper = {}, {}
        hard = {}
        if daemon_config:
            cfg = json.loads(Path(daemon_config).read_text())
            hard = {n: j.get("position_limits") or {} for n, j in cfg["joints"].items()}
        for arm in arms:
            chain = ArmChain(list(LIMB_JOINTS[arm]))
            for k, n in enumerate(chain.joint_names):
                a = chain.urdf_to_device(np.eye(chain.n)[k] * chain.limits_lower[k])[k]
                b = chain.urdf_to_device(np.eye(chain.n)[k] * chain.limits_upper[k])[k]
                lo, hi = (min(a, b), max(a, b)) if tier == "soft" else (-np.inf, np.inf)
                h = hard.get(n, {})
                if h.get("min") is not None:
                    lo = max(lo, float(h["min"]))
                if h.get("max") is not None:
                    hi = min(hi, float(h["max"]))
                lower[n], upper[n] = lo + margin_rad, hi - margin_rad
        return cls(lower, upper)

    def violations(self, names, q) -> list[str]:
        """Joints a single configuration puts outside the limits."""
        out = []
        for n, v in zip(names, np.atleast_1d(q)):
            if n not in self.lower:
                out.append(f"{n}: no limits known")
            elif not (self.lower[n] <= v <= self.upper[n]):
                out.append(f"{n} {np.degrees(v):+.1f} deg outside "
                           f"[{np.degrees(self.lower[n]):+.1f}, {np.degrees(self.upper[n]):+.1f}]")
        return out

    def check(self, traj: Trajectory, vmax: float, amax: float, tol: float = 1.05) -> list[str]:
        """Everything wrong with a trajectory: limits, speed, acceleration, sample spacing."""
        problems = []
        lo = np.array([self.lower.get(n, -np.inf) for n in traj.names])
        hi = np.array([self.upper.get(n, np.inf) for n in traj.names])
        bad = (traj.q < lo) | (traj.q > hi)
        for j in np.where(bad.any(0))[0]:
            k = int(np.argmax(bad[:, j]))
            problems.append(f"{traj.names[j]} leaves its limits at t={traj.t[k]:.2f} s "
                            f"({np.degrees(traj.q[k, j]):+.1f} deg)")
        v = np.abs(traj.velocity()).max(0) if len(traj.t) > 2 else np.zeros(len(traj.names))
        a = np.abs(traj.acceleration()).max(0) if len(traj.t) > 2 else np.zeros(len(traj.names))
        for j, n in enumerate(traj.names):
            if v[j] > vmax * tol:
                problems.append(f"{n} peaks at {np.degrees(v[j]):.1f} deg/s (cap {np.degrees(vmax):.1f})")
            if a[j] > amax * tol * 1.5:          # finite differences overshoot the analytic peak
                problems.append(f"{n} peaks at {np.degrees(a[j]):.0f} deg/s^2 (cap {np.degrees(amax):.0f})")
        if len(traj.t) > 1 and np.diff(traj.t).max() > 0.05:
            problems.append(f"samples {np.diff(traj.t).max() * 1000:.0f} ms apart (max 50)")
        return problems
