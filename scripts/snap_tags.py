#!/usr/bin/env python3
"""Take stills on Enter and show, for each one, the tags of a body and how far off face-on.

    python scripts/snap_tags.py --body left_hand
    python scripts/spin_survey.py --only-survey --also runs/shots_snap_<time> --also ...

Nothing moves: for posing a body by hand (arm limp) to get views a planned path never gives,
e.g. a cube's end face together with a side face. A pair is usable when both tags are within
``--max-view`` of face-on (the survey's cutoff); the shots stay in runs/shots_snap_<time> to
add to a survey with ``--also``.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from teleop_station.vision import CameraCalibration, TagSet, TagTracker      # noqa: E402
from teleop_station.vision.camera import describe, open_device             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", default="left_hand")
    ap.add_argument("--max-view", type=float, default=65.0)
    ap.add_argument("--camera", default=None, help="camera source (default gphoto2, keeping the shots)")
    ap.add_argument("--calib", default=str(ROOT / "calib" / "a7s3_sigma16_2768x1848.json"))
    a = ap.parse_args()

    calib = CameraCalibration.load(a.calib)
    tags = TagSet.load(ROOT / "config" / "tags.json")
    ids = tags.bodies[a.body]
    tracker = TagTracker(calib, replace(tags, body_layout={}))
    folder = f"runs/shots_snap_{time.strftime('%Y%m%dT%H%M%S')}"
    cam = open_device(a.camera or f"gphoto2:{folder}", *calib.image_size)
    print("camera:", describe(cam))
    print(f"{a.body}: tags {list(ids)}. Pose it, then Enter for a shot; q then Enter to finish.")
    pairs: dict[tuple[int, int], int] = {}
    try:
        while input("> ").strip().lower() != "q":
            ok, f = cam.read()
            if not ok:
                print("  no frame")
                continue
            seen = tracker.detect(f)
            v = {i: seen[i].view_deg for i in ids if i in seen}
            good = sorted(i for i, d in v.items() if d <= a.max_view)
            for x in range(len(good)):
                for y in range(x + 1, len(good)):
                    pairs[(good[x], good[y])] = pairs.get((good[x], good[y]), 0) + 1
            print("  " + ("  ".join(f"{i}: {d:.0f} deg{'' if d <= a.max_view else ' (too edge-on)'}"
                                   for i, d in sorted(v.items())) or "none of its tags seen"))
            print(f"  usable pairs this shot: {[f'{x}-{y}' for x in good for y in good if x < y] or 'none'}"
                  f";  so far: { {f'{x}-{y}': n for (x, y), n in sorted(pairs.items())} }")
    finally:
        cam.release()
    print(f"shots in {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
