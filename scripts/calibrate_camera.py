#!/usr/bin/env python3
"""Solve the camera's intrinsics from the ChArUco board.

Live, from a camera (a window opens; move the board through the view):

    python scripts/calibrate_camera.py --source /dev/video0 --square 0.0279 \\
        --width 1920 --height 1080 --out calib/webcam_1080p.json

From a folder of stills (e.g. D5600 shots pulled with gphoto2):

    python scripts/calibrate_camera.py --source shots/ --square 0.0279 --out calib/d5600.json

``--square`` is one board square's edge in metres, measured on the print with calipers.

Focus, zoom and resolution must be the ones the measurement will use, and must not change
afterwards. In live mode a view is kept automatically when the board is seen well and has
moved since the last kept view; SPACE keeps one by hand, ENTER solves, Q/ESC quits without
saving. Cover the whole image, corners included, and tilt the board (up to ~45 deg) in
several directions: a board held flat-on in the middle gives a confident wrong answer.
Reprojection RMS under ~0.5 px is good; over 1 px, something moved or the print isn't flat.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from teleop_station.vision import calib as C                   # noqa: E402
from teleop_station.vision.camera import describe, frames, latest, open_device   # noqa: E402

TARGET_VIEWS = 30


def from_folder(src: str, board) -> tuple[list, tuple[int, int]]:
    views, size = [], None
    for frame in frames(src):
        s = frame.shape[1::-1]
        if size and s != size:
            sys.exit(f"mixed image sizes in {src}: {size} and {s}")
        size = s
        v = C.detect_board(frame, board)
        print(f"  {'kept' if v else 'no board'}" + (f" ({len(v[0])} corners)" if v else ""))
        if v:
            views.append(v)
    return views, size


def live(src: str, board, width: int, height: int) -> tuple[list, tuple[int, int]] | None:
    cap = open_device(src, width, height)
    print("camera:", describe(cap))
    views, centres, last = [], [], 0.0
    coverage = None
    while True:
        frame = latest(cap)
        size = frame.shape[1::-1]
        if coverage is None:
            coverage = np.zeros(frame.shape[:2], np.uint8)
        v = C.detect_board(frame, board)
        show = frame.copy()
        key = cv2.waitKey(1) & 0xFF
        if v:
            pts = v[1]
            for p in pts:
                cv2.circle(show, tuple(p.astype(int)), 4, (0, 255, 0), -1)
            c = pts.mean(axis=0)
            spread = np.ptp(pts, axis=0).max()
            moved = all(np.linalg.norm(c - q) > 0.08 * max(size) for q in centres[-3:]) or not centres
            auto = len(pts) >= 15 and moved and time.time() - last > 1.0
            if auto or key == ord(" "):
                views.append(v)
                centres.append(c)
                last = time.time()
                cv2.fillConvexPoly(coverage, cv2.convexHull(pts.astype(np.int32)), 255)
                print(f"  view {len(views)}: {len(pts)} corners, span {spread:.0f} px")
        tint = show.copy()
        tint[coverage > 0] = (0.6 * tint[coverage > 0] + 0.4 * np.array([255, 120, 0])).astype(np.uint8)
        cv2.putText(tint, f"views {len(views)}/{TARGET_VIEWS}  ENTER solve  Q quit  "
                    f"(blue = covered; fill the corners too)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("calibrate", tint)
        if key in (13, 10) or len(views) >= TARGET_VIEWS:
            break
        if key in (ord("q"), 27):
            cap.release()
            return None
    cap.release()
    cv2.destroyAllWindows()
    return views, size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="/dev/videoN, N, a video file, or an image folder")
    ap.add_argument("--square", type=float, required=True, help="measured square edge, metres")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--out", required=True, help="where to write the calibration JSON")
    ap.add_argument("--note", default="", help="e.g. 'Sony a7S III, 35mm, f/8, via capture card'")
    a = ap.parse_args()

    board = C.charuco_board(a.square)
    if Path(a.source).is_dir():
        views, size = from_folder(a.source, board)
    else:
        got = live(a.source, board, a.width, a.height)
        if got is None:
            print("quit, nothing saved")
            return 1
        views, size = got
    cal = C.solve(views, size, a.note)
    cal.save(a.out)
    print(f"\nsolved from {cal.n_views} views at {size[0]}x{size[1]}: RMS {cal.rms_px:.3f} px")
    print(f"  fx {cal.K[0,0]:.1f}  fy {cal.K[1,1]:.1f}  cx {cal.K[0,2]:.1f}  cy {cal.K[1,2]:.1f}")
    print(f"  wrote {a.out}")
    if cal.rms_px > 1.0:
        print("  WARNING: RMS over 1 px. Check the print is flat and focus/zoom didn't move.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
