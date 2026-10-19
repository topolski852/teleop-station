#!/usr/bin/env python3
"""Live wrist positions in the base tag's frame, and the camera's noise floor.

    python scripts/track_tags.py --source /dev/video0 --calib calib/webcam_1080p.json \\
        --size 0=0.0598 --size 1=0.0297 --size 2=0.0297

``--size ID=METRES`` is each tag's black square edge, measured on the print. Ids default
to 0 = base, 1 = left wrist, 2 = right wrist (``teleop_station.vision.tags.DEFAULT_TAGS``).

A window shows the tags; the terminal prints each wrist's position in the base frame (mm)
and its spread over the last ``--window`` frames. With nothing moving, that spread is the
**noise floor**: the smallest repeatability the setup can resolve. Get it well under the
spread you expect from the arms (aim for ~0.1 mm) before trusting a repeatability run.

``--measure N`` takes one N-frame averaged reading, prints it as JSON and exits. It's the
same reading the repeatability test will take at each stop. ``--no-window`` skips the
preview, for SSH sessions.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from teleop_station.vision import DEFAULT_TAGS, CameraCalibration, TagTracker, mean_pose  # noqa: E402
from teleop_station.vision.camera import describe, latest, open_device                    # noqa: E402
from teleop_station.vision.tags import draw                                               # noqa: E402


def parse_sizes(items: list[str]) -> dict[int, float]:
    sizes = dict(DEFAULT_TAGS.sizes_m)
    for it in items:
        k, v = it.split("=")
        sizes[int(k)] = float(v)
    return sizes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--size", action="append", default=[], metavar="ID=METRES")
    ap.add_argument("--window", type=int, default=30, help="frames in the rolling spread")
    ap.add_argument("--measure", type=int, default=0, metavar="N")
    ap.add_argument("--no-window", action="store_true")
    a = ap.parse_args()

    calib = CameraCalibration.load(a.calib)
    tags = replace(DEFAULT_TAGS, sizes_m=parse_sizes(a.size))
    tracker = TagTracker(calib, tags)
    w, h = calib.image_size
    cap = open_device(a.source, w, h)
    print("camera:", describe(cap), file=sys.stderr)

    hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=max(a.window, a.measure)))
    frames_seen = base_missing = 0
    try:
        while True:
            frame = latest(cap)
            frames_seen += 1
            seen = tracker.detect(frame)
            rel = {}
            if tags.base in seen:
                base_inv = seen[tags.base][0].inverse()
                rel = {n: base_inv @ seen[i][0] for n, i in tags.wrists.items() if i in seen}
            else:
                base_missing += 1
            for name, pose in rel.items():
                hist[name].append(pose)

            if a.measure:
                if frames_seen >= a.measure * 3 and not any(len(q) >= a.measure for q in hist.values()):
                    print(f"base tag seen in {frames_seen - base_missing}/{frames_seen} frames, "
                          f"wrists not enough; nothing measured", file=sys.stderr)
                    return 1
                if hist and all(len(q) >= a.measure for q in hist.values()) and frames_seen >= a.measure:
                    out = {}
                    for name, q in hist.items():
                        mean, std = mean_pose(list(q)[-a.measure:])
                        out[name] = {"pos_mm": (mean.t * 1000).round(3).tolist(),
                                     "std_mm": (std * 1000).round(4).tolist(), "frames": a.measure}
                    print(json.dumps(out, indent=2))
                    return 0
            else:
                line = []
                for name, q in sorted(hist.items()):
                    if len(q) >= 2 and name in rel:
                        _, std = mean_pose(list(q)[-a.window:])
                        p = rel[name].t * 1000
                        line.append(f"{name} [{p[0]:7.2f} {p[1]:7.2f} {p[2]:7.2f}] mm "
                                    f"± {np.linalg.norm(std) * 1000:.3f}")
                status = "  ".join(line) if line else (
                    "base tag not seen" if tags.base not in seen else "no wrist tag seen")
                print("\r" + status.ljust(120), end="", flush=True)

            if not a.no_window:
                show = draw(frame, calib, tags, seen)
                if show.shape[1] > 1600:
                    show = cv2.resize(show, None, fx=1600 / show.shape[1], fy=1600 / show.shape[1])
                cv2.imshow("tags", show)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    return 0
    except KeyboardInterrupt:
        return 0
    finally:
        cap.release()
        print(file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
