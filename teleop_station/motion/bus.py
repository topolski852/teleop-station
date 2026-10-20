"""
The joints as the executor sees them: read state, change mode, send targets, E-stop.

``DaemonBus`` is the real robot through ``humanoid_core.daemon.DaemonClient``. It must be the
**only** daemon client: the daemon sends telemetry to one fixed UDP port, so a second client
(humanoid-web) steals or splits it, and the daemon has no notion of who is in charge. Stop
humanoid-web before opening one.

``SimBus`` is a simulated arm for dry runs: each ENABLED joint follows its target with a
first-order lag, IDLE joints sag, and faults can be injected for tests. Nothing leaves the
process.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

LIVE = {"IDLE", "DAMPING", "ENABLED"}


@dataclass
class JointSample:
    position: float          # device rad
    velocity: float          # rad/s
    state: str               # daemon JointState: OFFLINE, DISABLED, IDLE, DAMPING, ENABLED, FAULT...
    error: int               # firmware error bitmask
    age: float               # seconds since this joint's telemetry was received
    torque: float = float("nan")    # Nm measured at the output (slow-polled by the daemon, ~3 Hz)
    current: float = float("nan")   # A, i_q measured (slow-polled)


class JointBus(Protocol):
    def read(self, names) -> dict[str, JointSample]: ...
    def wake(self) -> None: ...                         # every joint -> IDLE (limp)
    def clear_error(self, name: str) -> None: ...
    def set_mode(self, name: str, mode: str) -> None: ...   # POSITION / DAMPING / IDLE
    def send(self, targets: dict[str, float]) -> None: ...
    def estop(self) -> None: ...                        # every joint -> DAMPING, at once
    def close(self) -> None: ...


class DaemonBus:
    def __init__(self, daemon_config: str, other_clients=("humanoid-web",)):
        import subprocess
        from humanoid_core.daemon import DaemonClient, RobotConfig
        # Both clients would bind :9000 (SO_REUSEADDR) and the kernel would split the
        # telemetry between them without either noticing, so refuse while one is running.
        for unit in other_clients:
            r = subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True)
            if r.stdout.strip() == "active":
                raise RuntimeError(f"{unit} is running and is a daemon client too; stop it first "
                                   f"(sudo systemctl stop {unit})")
        self.client = DaemonClient(RobotConfig.from_json(daemon_config))
        asyncio.run(self.client.start())
        t0 = time.time()
        while not self.client.is_running():
            if time.time() - t0 > 3.0:
                self.client._stop_sync()
                raise RuntimeError("no telemetry from the daemon on :9000. Is humanoid-web still "
                                   "running and holding the port? (systemctl stop humanoid-web)")
            time.sleep(0.05)
    def read(self, names) -> dict[str, JointSample]:
        # The cache keeps the last values forever. Every joint arrives in the same telemetry
        # packet, so the packet's arrival time is each joint's age.
        age = time.monotonic() - self.client._last_tel_time
        out = {}
        for n in names:
            d = self.client.get_cached_joint_state(n) or {}
            tq, cu = d.get("torque"), d.get("current")
            out[n] = JointSample(float(d.get("position", 0.0)), float(d.get("velocity", 0.0)),
                                 d.get("state") or d.get("joint_state") or "UNKNOWN",
                                 int(d.get("error", 0) or 0), age,
                                 float(tq) if tq is not None else float("nan"),
                                 float(cu) if cu is not None else float("nan"))
        return out

    def wake(self) -> None:
        self.client.wake_all("IDLE")

    def clear_error(self, name: str) -> None:
        self.client.clear_error(name)

    def set_mode(self, name: str, mode: str) -> None:
        self.client.set_mode(name, mode)

    def send(self, targets: dict[str, float]) -> None:
        for n, v in targets.items():
            self.client.set_position(n, float(v))

    def estop(self) -> None:
        self.client.estop_all()

    def close(self) -> None:
        self.client._stop_sync()


class SimBus:
    """A simulated arm. ``faults`` maps a joint to (t_seconds, kind): 'stall' freezes it, 'stale'
    stops its telemetry, 'error' raises a firmware error."""

    def __init__(self, start: dict[str, float], tau: float = 0.08, state: str = "IDLE", faults=None):
        self.pos = dict(start)
        self.vel = {n: 0.0 for n in start}
        self.target = dict(start)
        self.state = {n: state for n in start}
        self.tau, self.faults = tau, dict(faults or {})
        self.t0 = self.t_last = time.monotonic()
        self.log: list[tuple[float, str, object]] = []
        self.estopped = False

    def _step(self) -> None:
        now = time.monotonic()
        dt, self.t_last = now - self.t_last, now
        for n in self.pos:
            f = self.faults.get(n)
            if f and now - self.t0 >= f[0] and f[1] == "stall":
                self.vel[n] = 0.0
                continue
            if self.state[n] == "ENABLED":
                new = self.target[n] + (self.pos[n] - self.target[n]) * np.exp(-dt / self.tau)
            elif self.state[n] == "IDLE":
                new = self.pos[n] - 0.05 * dt              # limp: sags
            else:
                new = self.pos[n]                          # DAMPING: holds about where it is
            self.vel[n] = (new - self.pos[n]) / dt if dt > 0 else 0.0
            self.pos[n] = new

    def read(self, names) -> dict[str, JointSample]:
        self._step()
        now = time.monotonic()
        out = {}
        for n in names:
            f = self.faults.get(n)
            fired = f is not None and now - self.t0 >= f[0]
            age = (now - self.t0 - f[0]) if fired and f[1] == "stale" else 0.0
            err = 0x40 if fired and f[1] == "error" else 0
            out[n] = JointSample(self.pos[n], self.vel[n], "FAULT" if err else self.state[n], err, age)
        return out

    def wake(self) -> None:
        self.state = {n: "IDLE" for n in self.state}
        self.log.append((time.monotonic(), "wake", None))

    def clear_error(self, name: str) -> None:
        pass

    def set_mode(self, name: str, mode: str) -> None:
        self.state[name] = {"POSITION": "ENABLED", "ENABLED": "ENABLED"}.get(mode, mode)
        if self.state[name] == "ENABLED":
            self.target[name] = self.pos[name]         # the daemon seeds the hold at measured
        self.log.append((time.monotonic(), "mode", (name, mode)))

    def send(self, targets: dict[str, float]) -> None:
        for n, v in targets.items():
            if self.state[n] == "ENABLED":
                self.target[n] = float(v)

    def estop(self) -> None:
        self.estopped = True
        self.state = {n: "DAMPING" for n in self.state}
        self.log.append((time.monotonic(), "estop", None))

    def close(self) -> None:
        pass
