"""
Streams precomputed trajectories to the joints and watches them while it does.

The daemon has no command watchdog: an ENABLED joint holds its last target for as long as
the daemon runs, whoever sent it and whether or not they are still alive. So the watching
happens here, every tick, and anything wrong is an E-stop (the daemon puts every joint in
DAMPING on its next control tick):

- a joint's measured position more than ``track_limit`` from what was commanded (blocked,
  stalled, or not following),
- telemetry older than ``stale_s``,
- a joint that reports a firmware error or is no longer ENABLED.

An operator stop (``request_stop``, e.g. from Ctrl-C) is not an E-stop: streaming stops, the
daemon holds the last target (a point on the planned path, within a tick of where the arm
is), and ``MotionStopped`` is raised so the caller can decide what happens next.

If this process is killed outright the same holds: the arm stays at the last target sent.
Closing that gap needs a command timeout in the daemon itself.
"""
from __future__ import annotations

import time

import numpy as np

from .bus import LIVE, JointBus
from .limits import JointLimits
from .trajectory import Trajectory


class MotionAborted(RuntimeError):
    """A safety check tripped; the joints have been E-stopped (DAMPING)."""


class MotionStopped(RuntimeError):
    """The operator asked to stop; the joints are holding where they were."""


class Executor:
    def __init__(self, bus: JointBus, names, limits: JointLimits, rate_hz: float = 50.0,
                 track_limit_deg: float = 8.0, stale_s: float = 0.3, log=None):
        self.bus, self.names, self.limits = bus, list(names), limits
        self.period = 1.0 / rate_hz
        self.track_limit = np.radians(track_limit_deg)
        self.stale_s = stale_s
        self.enabled = False
        self.estopped = False
        self._stop = False
        self.last_target: np.ndarray | None = None
        self.log = log or (lambda msg: None)

    # ── state ────────────────────────────────────────────────────────────────
    def read(self) -> dict:
        return self.bus.read(self.names)

    def positions(self) -> np.ndarray:
        s = self.read()
        return np.array([s[n].position for n in self.names])

    def request_stop(self) -> None:
        self._stop = True

    def estop(self, why: str) -> None:
        self.bus.estop()
        self.estopped, self.enabled = True, False
        self.log(f"E-STOP: {why}")

    def _abort(self, why: str):
        self.estop(why)
        raise MotionAborted(why)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def preflight(self) -> None:
        """Every joint live and fault-free, telemetry fresh. Raises without moving anything."""
        s = self.read()
        bad = {n: (x.state, x.error) for n, x in s.items() if x.state not in LIVE or x.error}
        if bad:
            raise MotionAborted(f"joints not ready: {bad}")
        stale = {n: round(x.age, 2) for n, x in s.items() if x.age > self.stale_s}
        if stale:
            raise MotionAborted(f"stale telemetry: {stale}")
        out = self.limits.violations(self.names, [s[n].position for n in self.names])
        if out:
            raise MotionAborted("joints already outside the planning limits: " + "; ".join(out))

    def enable(self, timeout: float = 2.0) -> np.ndarray:
        """POSITION mode on every joint. The daemon seeds each hold target at the measured
        position, so nothing jumps. Returns where the joints are."""
        self.preflight()
        for n in self.names:
            self.bus.set_mode(n, "POSITION")
        t0 = time.monotonic()
        while True:
            s = self.read()
            if all(s[n].state == "ENABLED" for n in self.names):
                break
            if time.monotonic() - t0 > timeout:
                self._abort(f"not ENABLED after {timeout} s: "
                            f"{ {n: s[n].state for n in self.names if s[n].state != 'ENABLED'} }")
            time.sleep(0.02)
        self.enabled = True
        q = self.positions()
        self.last_target = q
        self.log("enabled: " + ", ".join(f"{n.replace('_joint', '')} {np.degrees(v):+.1f}" for n, v in zip(self.names, q)))
        return q

    def release(self, settle_s: float = 1.0) -> None:
        """DAMPING (the arm sinks slowly, resisted), then IDLE (limp)."""
        for n in self.names:
            self.bus.set_mode(n, "DAMPING")
        time.sleep(settle_s)
        for n in self.names:
            self.bus.set_mode(n, "IDLE")
        self.enabled = False
        self.log("released: DAMPING then IDLE")

    # ── motion ───────────────────────────────────────────────────────────────
    def _check(self, s, commanded) -> None:
        for n, c in zip(self.names, commanded):
            x = s[n]
            if x.age > self.stale_s:
                self._abort(f"{n}: telemetry {x.age:.2f} s old")
            if x.error or x.state != "ENABLED":
                self._abort(f"{n}: state {x.state}, error 0x{x.error:04x}")
            if abs(x.position - c) > self.track_limit:
                self._abort(f"{n}: {np.degrees(x.position - c):+.1f} deg from its command "
                            f"(limit {np.degrees(self.track_limit):.0f})")

    def run(self, traj: Trajectory, vmax: float, amax: float) -> None:
        """Stream a trajectory. It must start where the joints were last commanded."""
        if not self.enabled:
            raise MotionAborted("not enabled")
        if traj.names != self.names:
            raise MotionAborted("trajectory is over different joints")
        problems = self.limits.check(traj, vmax, amax)
        if problems:
            raise MotionAborted("trajectory rejected before moving: " + "; ".join(problems))
        if self.last_target is not None and np.abs(traj.q[0] - self.last_target).max() > np.radians(0.5):
            raise MotionAborted("trajectory doesn't start at the last commanded position")
        t0 = time.monotonic()
        k = 0
        while k < len(traj.t):
            if self._stop:
                self._stop = False
                raise MotionStopped("stopped by the operator; holding")
            now = time.monotonic() - t0
            # Send the sample due now (skip any we fell behind on, never run ahead).
            while k + 1 < len(traj.t) and traj.t[k + 1] <= now:
                k += 1
            target = traj.q[k]
            self._check(self.read(), self.last_target)
            self.bus.send(dict(zip(self.names, target)))
            self.last_target = target
            if k == len(traj.t) - 1:
                break
            time.sleep(max(0.0, min(self.period, traj.t[k + 1] - (time.monotonic() - t0))))

    def settle(self, timeout: float = 3.0, vel_deg_s: float = 0.5, pos_deg: float = 1.0,
               hold_s: float = 0.5) -> dict:
        """Wait (still streaming nothing new; the daemon holds the target) until every joint is
        slow and close to its target for ``hold_s``. Returns how it ended."""
        t0, still_since = time.monotonic(), None
        while True:
            if self._stop:
                self._stop = False
                raise MotionStopped("stopped by the operator; holding")
            s = self.read()
            self._check(s, self.last_target)
            err = np.array([s[n].position for n in self.names]) - self.last_target
            vel = np.array([s[n].velocity for n in self.names])
            ok = np.degrees(np.abs(vel)).max() < vel_deg_s and np.degrees(np.abs(err)).max() < pos_deg
            now = time.monotonic()
            if ok:
                still_since = still_since or now
                if now - still_since >= hold_s:
                    return {"settled": True, "t": now - t0, "err_deg": np.degrees(err).tolist()}
            else:
                still_since = None
            if now - t0 > timeout:
                return {"settled": False, "t": now - t0, "err_deg": np.degrees(err).tolist()}
            time.sleep(self.period)
