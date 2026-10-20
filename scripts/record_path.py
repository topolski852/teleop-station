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
- The status line shows PAST LIMIT when a joint is beyond where the motors are allowed to drive
  it (``--limits``, the ESC tier by default). A limp arm has no mechanical stops, so the hand
  can take it there, but a replay can't; stay inside. The limits only mean anything once the
  arms are zeroed (the zero is lost at every power-up).
- Pause briefly at every turn and reversal too, not just where you want to measure: the
  planner splits the path at near-stops, and a fast wiggle with no pause can't be smoothed
  without straying from where the hand went.

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
from teleop_station.motion import DaemonBus, JointLimits, RawPath   # noqa: E402
from teleop_station.motion.bus import LIVE                       # noqa: E402

DAEMON_CONFIG = "/home/nse/humanoid-studio/configs/humanoid_arms.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "runs" / f"path_raw_{time.strftime('%Y%m%dT%H%M%S')}.json"))
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--pause-s", type=float, default=1.5, help="still this long = a stop")
    ap.add_argument("--still-deg", type=float, default=0.5, help="every joint within this = still")
    ap.add_argument("--limits", default="hard", choices=("soft", "hard"),
                    help="warn when a joint leaves this range: the motors won't follow it there")
    ap.add_argument("--warn-pitch", type=float, default=65.0,
                    help="warn when a shoulder pitch goes past this many degrees from hanging")
    ap.add_argument("--config", default=DAEMON_CONFIG)
    a = ap.parse_args()

    names = list(LIMB_JOINTS["left_arm"]) + list(LIMB_JOINTS["right_arm"])
    limits = JointLimits.for_arms(a.config, tier=a.limits)
    lo = np.array([limits.lower[n] for n in names]); hi = np.array([limits.upper[n] for n in names])
    bus = DaemonBus(a.config)
    try:
        s = bus.read(names)
        enabled = [n for n in names if s[n].state == "ENABLED"]
        if enabled:
            print("joints are ENABLED (someone is commanding them); not recording:", enabled)
            return 1
        damped = [n.replace("_joint", "") for n in names if s[n].state == "DAMPING"]
        if damped:
            # After an E-stop the joints are DAMPING: they resist being guided by hand.
            print("joints are DAMPING (resist being moved):", damped)
            if input("Switch every joint to IDLE (limp) for guiding by hand? Arms resting on the table? [y/N] ").strip().lower() == "y":
                bus.wake()
                time.sleep(0.5)
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
        over: set[str] = set()
        pitch_high: set[str] = set()
        pitch_cols = [j for j, n in enumerate(names) if n.endswith("shoulder_pitch_joint")]
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
                out = [names[j].replace("_joint", "") for j in np.where((np.array(q[-1]) < lo) | (np.array(q[-1]) > hi))[0]]
                high = [names[j].replace("_joint", "") for j in pitch_cols if abs(np.degrees(q[-1][j])) > a.warn_pitch]
                if high:
                    pitch_high.update(high)
                if out:
                    over.update(out)
                if tick - last_show > 0.3:
                    warn = f"  PAST LIMIT: {', '.join(out)}" if out else ""
                    warn += f"  HIGH PITCH (> {a.warn_pitch:.0f}): {', '.join(high)}" if high else ""
                    print(f"\r  {t[-1]:6.1f} s  {'PAUSE (counts as a stop)' if in_pause else 'moving                 '}"
                          f"  stops so far: {pauses}{warn}".ljust(150), end="", flush=True)
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
    if pitch_high:
        print(f"NOTE: {sorted(pitch_high)} went past {a.warn_pitch:.0f} deg; under the motors the left "
              f"shoulder stopped at ~69 deg with the arm extended (2026-10-20).")
    if over:
        print(f"WARNING: went past the {a.limits} limits on {sorted(over)}: the motors won't follow there, "
              f"so that part can't be replayed. Re-record staying inside.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
