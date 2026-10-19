#!/usr/bin/env python3
"""Print sheets for the camera: the station tags and the ChArUco calibration board.

    python scripts/make_targets.py            # writes targets/tags.png and targets/charuco.png

Both are US-letter pages at 300 dpi. Print at **100% / actual size** (not "fit to page") on
matte paper and glue them to something flat and rigid. Then measure with calipers and use
the measured numbers, not the nominal ones printed on the sheet:
- each tag's black square edge -> ``track_tags.py --size``
- one ChArUco square's edge     -> ``calibrate_camera.py --square``
The 100 mm bar on each sheet tells you at a glance whether the printer scaled the page.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from teleop_station.vision.calib import BOARD_SQUARES, charuco_board     # noqa: E402
from teleop_station.vision.tags import DEFAULT_TAGS, TAG_DICT            # noqa: E402

DPI = 300
PAGE = (int(8.5 * DPI), int(11 * DPI))            # letter, portrait (w, h)


def px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))


def label(img, text, x, y, scale=1.2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, 2, cv2.LINE_AA)


def scale_bar(img, x, y):
    cv2.rectangle(img, (x, y), (x + px(100), y + px(3)), 0, -1)
    label(img, "100 mm: check this before measuring anything else", x, y + px(9), 0.9)


def tag_sheet(out: Path) -> None:
    page = np.full(PAGE[::-1], 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(TAG_DICT)
    cells = 8                                      # 6 data bits + 1 black border each side
    x, y = px(20), px(20)
    for tag_id in [DEFAULT_TAGS.base, *DEFAULT_TAGS.wrists.values()]:
        size_mm = DEFAULT_TAGS.sizes_m[tag_id] * 1000
        side = px(size_mm) // cells * cells        # whole pixels per cell: crisp edges
        img = cv2.aruco.generateImageMarker(d, tag_id, side, borderBits=1)
        if x + side + px(20) > PAGE[0]:
            x, y = px(20), y + prev + px(30)
        page[y:y + side, x:x + side] = img
        label(page, f"36h11 id {tag_id} ({DEFAULT_TAGS.name(tag_id)})", x, y + side + px(6), 0.9)
        label(page, f"nominal {side / DPI * 25.4:.1f} mm, measure it", x, y + side + px(11), 0.8)
        x += side + px(25)
        prev = side
    label(page, "Spare wrist tags: print this page twice. Keep the white margin.",
          px(20), PAGE[1] - px(40), 0.9)
    scale_bar(page, px(20), PAGE[1] - px(25))
    cv2.imwrite(str(out), page)


def board_sheet(out: Path, square_mm: float) -> None:
    page = np.full(PAGE[::-1], 255, np.uint8)
    board = charuco_board(square_mm / 1000)
    w, h = px(square_mm * BOARD_SQUARES[0]), px(square_mm * BOARD_SQUARES[1])
    img = board.generateImage((w, h), marginSize=0, borderBits=1)
    x, y = (PAGE[0] - w) // 2, px(25)
    page[y:y + h, x:x + w] = img
    label(page, f"ChArUco {BOARD_SQUARES[0]}x{BOARD_SQUARES[1]}, DICT_5X5_100, "
          f"nominal square {square_mm:.1f} mm: measure one", x, y + h + px(8), 0.9)
    scale_bar(page, x, y + h + px(20))
    cv2.imwrite(str(out), page)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="targets")
    ap.add_argument("--square-mm", type=float, default=28.0, help="nominal ChArUco square size")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag_sheet(out / "tags.png")
    board_sheet(out / "charuco.png", a.square_mm)
    print(f"wrote {out / 'tags.png'} and {out / 'charuco.png'} (letter, {DPI} dpi, print at 100%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
