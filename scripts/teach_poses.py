#!/usr/bin/env python3
"""Teach poses by hand for the repeatability run: hold an arm in a pose, save its joint angles.

    sudo systemctl stop humanoid-web          # this must be the only daemon client
    python scripts/teach_poses.py --out runs/poses_demo.json

The arms stay limp (IDLE) the whole time, so they can be moved by hand and fall when let go.
Nothing is ever commanded to move. If the joints are asleep (OFFLINE/DISABLED, as after a
power-up) the script asks before waking them to IDLE, which is a mode change, though not a
motion: a limp joint doesn't move when woken.

Commands, typed then Enter:
  rest    save BOTH arms' current angles as the rest pose (arms resting on the table)
  l       save the LEFT arm's current angles as its next pose
  r       save the RIGHT arm's current angles as its next pose
  u       undo the last saved pose
  q       write the file and quit

Right pose i runs together with left pose i in a both-arms run, so teach each arm's poses
on its own side, clear of the other arm's. Angles are device frame radians and only mean
anything until the next power cycle or re-zero (the zero is lost when the motors lose power).
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from humanoid_core.layout import LIMB_JOINTS                     # noqa: E402
from teleop_station.motion import DaemonBus, JointLimits         # noqa: E402
from teleop_station.motion.bus import LIVE                        # noqa: E402

DAEMON_CONFIG = "/home/nse/humanoid-studio/configs/humanoid_arms.json"
ARMS = {"left": list(LIMB_JOINTS["left_arm"]), "right": list(LIMB_JOINTS["right_arm"])}


def fmt(names, s) -> str:
    return "  ".join(f"{n.split('_', 1)[1].replace('_joint', '').replace('shoulder_', 'sh_')} "
                     f"{np.degrees(s[n].position):+6.1f}" for n in names)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "runs" / f"poses_{time.strftime('%Y%m%dT%H%M%S')}.json"))
    ap.add_argument("--limits", default="soft", choices=("soft", "hard"),
                    help="soft: the URDF's range (teleop's); hard: the ESC firmware's wider range")
    ap.add_argument("--config", default=DAEMON_CONFIG, help="the daemon's joint config")
    ap.add_argument("--approach-deg", type=float, default=3.0,
                    help="each pose needs this much room inside the limits for the approach")
    a = ap.parse_args()

    names = ARMS["left"] + ARMS["right"]
    limits = JointLimits.for_arms(a.config, tier=a.limits)
    bus = DaemonBus(a.config)
    try:
        s = bus.read(names)
        asleep = {n: x.state for n, x in s.items() if x.state not in LIVE}
        if asleep:
            print("joints not awake:", {n.replace("_joint", ""): st for n, st in asleep.items()})
            if input("Wake every joint to IDLE (limp, nothing moves)? Arms resting on the table? [y/N] ").strip().lower() != "y":
                print("nothing changed")
                return 1
            bus.wake()
            time.sleep(1.0)
            s = bus.read(names)
            asleep = {n: x.state for n, x in s.items() if x.state not in LIVE}
            if asleep:
                print("still not awake:", asleep)
                return 1
        faults = {n: hex(x.error) for n, x in s.items() if x.error}
        if faults:
            print("joints report errors (clear them in humanoid-web or power-cycle):", faults)
            return 1

        lines: queue.Queue = queue.Queue()
        threading.Thread(target=lambda: [lines.put(l.strip()) for l in sys.stdin], daemon=True).start()
        data = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "daemon_config": a.config,
                "frame": "device rad; valid until the next power cycle or re-zero",
                "joints": ARMS, "rest": None, "poses": {"left": [], "right": []}}
        saved: list[str] = []
        print("\ncommands: rest | l | r | u | q   (then Enter)\n")
        last_print = 0.0
        while True:
            s = bus.read(names)
            if time.time() - last_print > 1.0:
                print(f"L {fmt(ARMS['left'], s)}\nR {fmt(ARMS['right'], s)}   "
                      f"[rest {'set' if data['rest'] else 'NOT set'}, "
                      f"left {len(data['poses']['left'])}, right {len(data['poses']['right'])}]")
                last_print = time.time()
            try:
                cmd = lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if cmd == "q":
                break
            if cmd == "u" and saved:
                what = saved.pop()
                if what == "rest":
                    data["rest"] = None
                else:
                    data["poses"][what].pop()
                print(f"  undone: {what}")
                continue
            if cmd == "rest":
                data["rest"] = {n: s[n].position for n in names}
                saved.append("rest")
                print("  rest saved:", fmt(names, s))
                continue
            side = {"l": "left", "r": "right"}.get(cmd)
            if side is None:
                print("  ? commands: rest | l | r | u | q")
                continue
            q = {n: s[n].position for n in ARMS[side]}
            room = np.radians(a.approach_deg)
            out = [f"{n.replace('_joint', '')} {np.degrees(v):+.1f}" for n, v in q.items()
                   if not (limits.lower[n] + room <= v <= limits.upper[n] - room)]
            if out:
                print(f"  NOT saved: outside the planning limits (with {a.approach_deg:.0f} deg of approach room): {out}")
                continue
            moving = [n for n in ARMS[side] if abs(np.degrees(s[n].velocity)) > 2.0]
            if moving:
                print(f"  NOT saved: still moving ({[n.replace('_joint', '') for n in moving]}); hold it still")
                continue
            data["poses"][side].append({"name": f"{side[0].upper()}{len(data['poses'][side]) + 1}", "q": q})
            saved.append(side)
            print(f"  {side} pose {len(data['poses'][side])} saved:", fmt(ARMS[side], s))
    finally:
        bus.close()
    if data["rest"] is None:
        print("no rest pose saved; the runner needs one ('rest' with both arms on the table)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {a.out}: rest {'yes' if data['rest'] else 'NO'}, "
          f"{len(data['poses']['left'])} left, {len(data['poses']['right'])} right")
    return 0


if __name__ == "__main__":
    sys.exit(main())
