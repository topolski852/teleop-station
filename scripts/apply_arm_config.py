#!/usr/bin/env python3
"""Write the arms' limits, torque limits and gains from the daemon config to the ESCs.

    python scripts/apply_arm_config.py            # show what differs, ask, write, read back
    python scripts/apply_arm_config.py --check    # only show what differs

The ESCs keep these in RAM: after a power cycle they're back to whatever their flash holds,
and humanoid-web's Connect writes no config. Run this after every power-up (and after zeroing:
the position limits are stored offset-adjusted, so they're written against the current zero).

Nothing moves. It writes only while EVERY arm joint is IDLE: writing with joints DAMPING left
two of them with a latched watchdog fault on 2026-10-20. Each write is read back; it stops
at the first one that doesn't land. A record goes to runs/config_<time>.json.

Needs to be the only daemon client (humanoid-web stopped): it talks to the daemon's command
port directly and reads states with GET_ALL_STATES.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DAEMON_CONFIG = "/home/nse/humanoid-studio/configs/humanoid_arms.json"
ARM = ("shoulder", "elbow", "wrist")
FIELDS = ("position_limit_min", "position_limit_max", "torque_limit", "position_kp", "velocity_kp")


class Daemon:
    def __init__(self, port: int = 9001):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.settimeout(10)
        self.port = port

    def rpc(self, msg: dict) -> dict:
        rid = str(uuid.uuid4())
        self.s.sendto(json.dumps(dict(msg, id=rid)).encode(), ("127.0.0.1", self.port))
        while True:
            d = json.loads(self.s.recv(65536))
            if d.get("id") == rid:
                return d

    def read(self, joint: str, tries: int = 4) -> dict:
        for _ in range(tries):
            c = self.rpc({"type": "READ_CONFIG", "joint_name": joint}).get("config", {})
            if all(c.get(k) is not None for k in FIELDS + ("position_offset",)):
                return c
            time.sleep(0.3)
        return c


def esc_values(c: dict) -> dict:
    """ESC config as display-frame values (limits are stored offset-adjusted)."""
    off = c["position_offset"]
    return {"position_limit_min": c["position_limit_min"] - off, "position_limit_max": c["position_limit_max"] - off,
            "torque_limit": c["torque_limit"], "position_kp": c["position_kp"], "velocity_kp": c["velocity_kp"]}


def wanted(j: dict) -> dict:
    lim = j.get("position_limits") or {}
    return {"position_limit_min": lim.get("min"), "position_limit_max": lim.get("max"),
            "torque_limit": j.get("torque_limit"), "position_kp": j.get("position_kp"), "velocity_kp": j.get("velocity_kp")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DAEMON_CONFIG)
    ap.add_argument("--check", action="store_true", help="only report differences")
    ap.add_argument("--yes", action="store_true", help="don't ask before writing")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text())["joints"]
    joints = [n for n in cfg if any(k in n for k in ARM)]
    d = Daemon()
    todo = {}
    print(f"{'joint':28s} field              ESC now -> config")
    for n in joints:
        c = d.read(n)
        if any(c.get(k) is None for k in FIELDS + ("position_offset",)):
            print(f"{n}: couldn't read its config: {c}")
            return 1
        have, want = esc_values(c), wanted(cfg[n])
        diff = {k: v for k, v in want.items() if v is not None and abs(have[k] - v) > (1e-3 if "limit_m" not in k else 1e-4)}
        for k, v in diff.items():
            deg = "limit_m" in k
            f = (lambda x: f"{np.degrees(x):+8.2f} deg") if deg else (lambda x: f"{x:8.3f}")
            print(f"{n:28s} {k:18s} {f(have[k])} -> {f(v)}")
        if diff:
            todo[n] = diff
    if not todo:
        print("ESCs already match the config.")
        return 0
    if a.check:
        return 1
    st = d.rpc({"type": "GET_ALL_STATES"})["states"]
    not_idle = {n: st[n].get("joint_state") for n in joints if st[n].get("joint_state") != "IDLE"}
    if not_idle:
        print(f"\nNOT writing: every arm joint must be IDLE (limp) first: {not_idle}")
        return 1
    if not a.yes and input(f"\nWrite these to {len(todo)} ESCs (joints IDLE, nothing moves)? [y/N] ").strip().lower() != "y":
        print("nothing written")
        return 1
    rec = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "config": a.config, "joints": []}
    for n, diff in todo.items():
        st = d.rpc({"type": "GET_ALL_STATES"})["states"]
        if any(st[k].get("joint_state") != "IDLE" for k in joints):
            print("a joint left IDLE; stopping")
            return 1
        before = d.read(n)
        r = d.rpc({"type": "APPLY_CONFIG", "joint_name": n, "config": diff})
        after = d.read(n)
        got = esc_values(after)
        ok = r.get("type") == "ACK" and all(abs(got[k] - v) < 2e-3 for k, v in diff.items()) \
            and abs(after["position_offset"] - before["position_offset"]) < 1e-6
        rec["joints"].append({"joint": n, "wrote": diff, "read_back": {k: got[k] for k in diff}, "ok": ok})
        print(f"  {n:28s} {'OK' if ok else 'DID NOT LAND: ' + json.dumps(r)}")
        if not ok:
            break
    st = d.rpc({"type": "GET_ALL_STATES"})["states"]
    faults = {n: hex(st[n]["error"]) for n in joints if st[n].get("error")}
    print("faults after writing:", faults or "none")
    path = ROOT / "runs" / f"config_{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"record: {path}")
    return 0 if all(j["ok"] for j in rec["joints"]) and not faults else 1


if __name__ == "__main__":
    sys.exit(main())
