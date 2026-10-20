#!/usr/bin/env python3
"""Record a path by guiding the limp arms by hand. Nothing is commanded to move.

    sudo systemctl stop humanoid-web          # this must be the only daemon client
    python scripts/record_path.py --out runs/path_raw_demo.json

All ten joints are logged at 50 Hz while you move the arms. **Pause for about 2 s** wherever
the run should stop and measure: at the start (arms resting on the table), at each stop, and
at the end. The pauses are found afterwards (``edit_path.py``); the terminal shows "PAUSE" once
you've been still long enough to count, so you never need a free hand during the recording.

- Start and end with the arms resting on the table, in the same place, so the path is a loop.
- Keep the arm you're not moving resting on the table, and set an arm down before letting go:
  a limp arm falls, and a fall would be recorded and replayed.
- Enter starts the recording, Enter (or Ctrl-C) ends it.

If the joints are asleep (as after a power-up) it asks before waking them to IDLE: a mode
change, not a motion. Joints already ENABLED belong to someone else's command; it refuses.
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

from humanoid_core.layout import LIMB_JOINTS                    # noqa: E402
from teleop_station.motion import DaemonBus, RawPath            # noqa: E402
from teleop_station.motion.bus import LIVE                       # noqa: E402

DAEMON_CONFIG = "/home/nse/humanoid-studio/configs/humanoid_arms.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "runs" / f"path_raw_{time.strftime('%Y%m%dT%H%M%S')}.json"))
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--pause-s", type=float, default=1.5, help="still this long = a stop")
    ap.add_argument("--still-deg", type=float, default=0.5, help="every joint within this = still")
    ap.add_argument("--config", default=DAEMON_CONFIG)
    a = ap.parse_args()

    names = list(LIMB_JOINTS["left_arm"]) + list(LIMB_JOINTS["right_arm"])
    bus = DaemonBus(a.config)
    try:
        s = bus.read(names)
        enabled = [n for n in names if s[n].state == "ENABLED"]
        if enabled:
            print("joints are ENABLED (someone is commanding them); not recording:", enabled)
            return 1
        asleep = {n: x.state for n, x in s.items() if x.state not in LIVE}
        if asleep:
            print("joints not awake:", {n.replace("_joint", ""): st for n, st in asleep.items()})
            if input("Wake every joint to IDLE (limp, nothing moves)? Arms resting on the table? [y/N] ").strip().lower() != "y":
                print("nothing changed")
                return 1
            bus.wake()
            time.sleep(1.0)
            s = bus.read(names)
            if any(x.state not in LIVE for x in s.values()):
                print("still not awake:", {n: x.state for n, x in s.items() if x.state not in LIVE})
                return 1
        if any(x.error for x in s.values()):
            print("joints report errors:", {n: hex(x.error) for n, x in s.items() if x.error})
            return 1

        input("Arms resting on the table. Press Enter to START recording (Enter again to stop) ")
        done = threading.Event()
        threading.Thread(target=lambda: (sys.stdin.readline(), done.set()), daemon=True).start()
        t, q = [], []
        t0 = time.monotonic()
        pauses, in_pause, last_show = 0, False, 0.0
        n_win = int(a.pause_s * a.hz)
        try:
            while not done.is_set():
                tick = time.monotonic()
                s = bus.read(names)
                if any(x.age > 0.3 for x in s.values()):
                    print("\ntelemetry stopped; ending the recording")
                    break
                t.append(tick - t0)
                q.append([s[n].position for n in names])
                if len(q) >= n_win:
                    w = np.array(q[-n_win:])
                    still = np.degrees(np.ptp(w, 0).max()) < a.still_deg
                    if still and not in_pause:
                        pauses += 1
                    in_pause = still
                if tick - last_show > 0.5:
                    print(f"\r  {t[-1]:6.1f} s  {'PAUSE (counts as a stop)' if in_pause else 'moving                 '}"
                          f"  stops so far: {pauses}   ", end="", flush=True)
                    last_show = tick
                time.sleep(max(0.0, 1.0 / a.hz - (time.monotonic() - tick)))
        except KeyboardInterrupt:
            pass
        print()
    finally:
        bus.close()
    if len(t) < 10:
        print("nothing recorded")
        return 1
    raw = RawPath(names, np.array(t), np.array(q))
    raw.save(a.out)
    print(f"wrote {a.out}: {t[-1]:.0f} s, {len(t)} samples, ~{pauses} pauses. "
          f"Next: python scripts/edit_path.py {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
