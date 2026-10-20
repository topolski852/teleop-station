"""
Joint trajectories: a time grid and the joint angles at each sample, computed before moving.

``movej`` is a synchronised minimum-jerk move: every joint starts and stops together, with
zero velocity and acceleration at both ends, and the duration is set by whichever joint
needs longest under the speed and acceleration caps. Minimum jerk is what makes a move look
smooth rather than mechanical: no step in acceleration anywhere.

``approach_move`` reaches a target via a point ``approach`` beyond it, always from the same
side, finishing slowly. The encoders sit on the motor side of the gearbox, so backlash is
invisible to them; arriving from the same side every time takes it out of the repeatability.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Minimum-jerk s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5 peaks at these multiples of D/T and D/T^2.
_PEAK_VEL = 1.875
_PEAK_ACC = 5.7735


@dataclass
class Trajectory:
    names: list[str]
    t: np.ndarray            # (N,) seconds from the start
    q: np.ndarray            # (N, n_joints) device rad

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    @property
    def end(self) -> np.ndarray:
        return self.q[-1].copy()

    def __add__(self, other: "Trajectory") -> "Trajectory":
        if other.names != self.names:
            raise ValueError("joining trajectories over different joints")
        if len(self.t) == 0:
            return other
        if not np.allclose(self.q[-1], other.q[0], atol=1e-9):
            raise ValueError("trajectories don't join: the second must start where the first ends")
        t2 = other.t[1:] + self.t[-1]
        return Trajectory(self.names, np.concatenate([self.t, t2]), np.vstack([self.q, other.q[1:]]))

    def velocity(self) -> np.ndarray:
        return np.gradient(self.q, self.t, axis=0) if len(self.t) > 2 else np.zeros_like(self.q)

    def acceleration(self) -> np.ndarray:
        return np.gradient(self.velocity(), self.t, axis=0) if len(self.t) > 2 else np.zeros_like(self.q)


def movej(names, q0, q1, vmax: float, amax: float, dt: float = 0.02, min_time: float = 0.3) -> Trajectory:
    """Synchronised minimum-jerk move from q0 to q1. vmax rad/s, amax rad/s^2 per joint."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = np.abs(q1 - q0)
    T = max(min_time, float(np.max(_PEAK_VEL * d / vmax)), float(np.max(np.sqrt(_PEAK_ACC * d / amax))))
    n = max(2, int(np.ceil(T / dt)) + 1)
    t = np.linspace(0.0, T, n)
    tau = t / T
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    return Trajectory(list(names), t, q0 + np.outer(s, q1 - q0))


def approach_move(names, q0, target, approach_rad: float, direction, vmax: float, amax: float,
                  final_vmax: float | None = None, dt: float = 0.02) -> Trajectory:
    """q0 -> target + approach * direction -> target. ``direction`` is +1/-1 per joint and must
    be the same every time this target is visited."""
    target = np.asarray(target, float)
    pre = target + approach_rad * np.asarray(direction, float)
    final_vmax = final_vmax if final_vmax is not None else vmax / 3
    return movej(names, q0, pre, vmax, amax, dt) + movej(names, pre, target, final_vmax, amax, dt)
