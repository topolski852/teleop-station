"""
Read-only joint angles for measurement: where the encoders say each joint is.

``WebJointSource`` reads humanoid-web's ``/api/status`` (its cached copy of the daemon
telemetry). It is a stopgap while humanoid-web owns the daemon connection: the daemon sends
telemetry to a single UDP port, so a second ``DaemonClient`` here would steal humanoid-web's
telemetry. Polling ``/api/status`` sends nothing to the daemon or the arms.

Angles are raw device frame (rad), exactly what telemetry reports and what ``SET_POSITION``
takes. No direction signs are applied. They are only meaningful within one power cycle,
since the joint offsets are lost when the motors lose power.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import dataclass

# Joint states whose position is live. OFFLINE positions are 0.0 placeholders, and DISABLED
# joints are silent, so their last position never ages out.
LIVE_STATES = {"IDLE", "DAMPING", "ENABLED"}


@dataclass
class JointReading:
    t: float                          # time.time() when it was fetched
    position: dict[str, float]        # rad, device frame
    velocity: dict[str, float]        # rad/s
    state: dict[str, str]
    error: dict[str, int]

    def live(self, names) -> bool:
        return all(self.state.get(n) in LIVE_STATES for n in names)


class WebJointSource:
    """Polls humanoid-web in a background thread; ``latest()`` is the newest reading."""

    def __init__(self, url: str = "http://127.0.0.1:8000/api/status", hz: float = 20.0):
        self.url, self.period = url, 1.0 / hz
        self._lock = threading.Lock()
        self._latest: JointReading | None = None
        self._history: list[JointReading] = []
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "WebJointSource":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def fetch(self) -> JointReading:
        with urllib.request.urlopen(self.url, timeout=1.0) as r:
            d = json.load(r)["data"]
        js = [j for j in d["joints"] if j.get("limb") in ("left_arm", "right_arm")]
        return JointReading(time.time(),
                            {j["name"]: float(j["position"]) for j in js},
                            {j["name"]: float(j["velocity"]) for j in js},
                            {j["name"]: j["state"] for j in js},
                            {j["name"]: int(j["error"]) for j in js})

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            try:
                r = self.fetch()
                with self._lock:
                    self._latest, self._error = r, None
                    self._history.append(r)
                    del self._history[:-400]            # ~20 s at 20 Hz
            except Exception as e:                     # noqa: BLE001 (report, keep polling)
                with self._lock:
                    self._error = f"{type(e).__name__}: {e}"
            self._stop.wait(max(0.0, self.period - (time.time() - t0)))

    def latest(self) -> JointReading | None:
        with self._lock:
            return self._latest

    def since(self, t: float) -> list[JointReading]:
        with self._lock:
            return [r for r in self._history if r.t >= t]

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error
