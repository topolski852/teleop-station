#!/usr/bin/env python3
"""Show how far the (limp) arms are from a planned path's start, to place them by hand.

    python scripts/match_start.py runs/path_raw_test2.left.planned.json

Reads the encoders only; nothing is commanded. Move the arm until every joint reads within
``--tol`` degrees, then press Enter. The runner's first move goes from wherever the arm is to
the path's start, and that move is planned, not recorded: starting close keeps it short.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from teleop_station.motion import DaemonBus, PlannedPath      # noqa: E402

DAEMON_CONFIG = "/home/nse/humanoid-studio/configs/humanoid_arms.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--tol", type=float, default=3.0)
    ap.add_argument("--config", default=DAEMON_CONFIG)
    a = ap.parse_args()
    plan = PlannedPath.load(a.path)
    start = plan.lead_in[0].q[0] if plan.lead_in else plan.stops[0]
    bus = DaemonBus(a.config)
    done = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.readline(), done.set()), daemon=True).start()
    print(f"Move the arm to the path's start; every joint within {a.tol:.0f} deg = OK. Enter when done.\n")
    try:
        while not done.is_set():
            s = bus.read(plan.names)
            d = np.degrees(np.array([s[n].position for n in plan.names]) - start)
            parts = []
            for n, v in zip(plan.names, d):
                tag = "ok" if abs(v) <= a.tol else ("move +" if v < 0 else "move -")
                parts.append(f"{n.split('_', 1)[1].replace('_joint', '').replace('shoulder_', 'sh_')} {v:+6.1f} {tag:6s}")
            allok = np.abs(d).max() <= a.tol
            print("\r  " + "  ".join(parts) + ("   ALL OK" if allok else "         "), end="", flush=True)
            time.sleep(0.1)
    finally:
        bus.close()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
