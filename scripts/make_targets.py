#!/usr/bin/env python3
"""Print sheets for the camera: the station tags and the ChArUco calibration board.

    python scripts/make_targets.py            # writes targets/charuco.pdf, targets/tags.pdf (+ PNGs)

The tag sheet holds every tag in ``config/tags.json`` at its listed size, for spares. The
board is a 7x5 ChArUco, landscape, 35 mm squares by default (``--square-mm``).

Each sheet is US letter at 300 dpi. Print the **PDF**: it carries its physical size, where a
PNG gets rescaled by most print dialogs. Print at **100% / actual size** (not "fit to page")
on matte paper and glue it to something flat and rigid. Then measure with calipers and use
the measured numbers, not the nominal ones printed on the sheet:
- each tag's black square edge -> ``track_tags.py --size``
- one ChArUco square's edge     -> ``calibrate_camera.py --square``
The 100 mm bar on each sheet tells you at a glance whether the printer scaled the page.
"""
from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from teleop_station.vision.calib import BOARD_SQUARES, charuco_board     # noqa: E402
from teleop_station.vision.tags import TagSet                            # noqa: E402

DPI = 300
PAGE = (int(8.5 * DPI), int(11 * DPI))            # letter, portrait (w, h)
LANDSCAPE = PAGE[::-1]


def px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))


def label(img, text, x, y, scale=1.2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, 2, cv2.LINE_AA)


def scale_bar(img, x, y):
    cv2.rectangle(img, (x, y), (x + px(100), y + px(3)), 0, -1)
    label(img, "100 mm: check this before measuring anything else", x, y + px(9), 0.9)


def write_pdf(path: Path, pages: list[np.ndarray]) -> None:
    """Grayscale pages at DPI into one PDF whose page size is the paper's, so it prints 1:1."""
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", None]
    kids = []
    for img in pages:
        h, w = img.shape
        pw, ph = w / DPI * 72, h / DPI * 72
        n = len(objs) + 1                          # this page's object number
        data = zlib.compress(np.ascontiguousarray(img, np.uint8).tobytes(), 9)
        content = f"q {pw:.3f} 0 0 {ph:.3f} 0 0 cm /Im0 Do Q".encode()
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.3f} {ph:.3f}] "
                    f"/Resources << /XObject << /Im0 {n + 1} 0 R >> >> /Contents {n + 2} 0 R >>".encode())
        objs.append(f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceGray "
                    f"/BitsPerComponent 8 /Filter /FlateDecode /Length {len(data)} >>\nstream\n".encode()
                    + data + b"\nendstream")
        objs.append(f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream")
        kids.append(f"{n} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>".encode()
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for k, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{k} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def tag_sheet(out: Path, tags: TagSet) -> list[Path]:
    """As many pages as the tags need: tags.png, tags_2.png, ..."""
    d = tags.dictionary()
    cells = d.markerSize + 2                       # data bits + 1 black border each side
    pages, page = [], None
    x = y = prev = 0

    def finish():
        label(page, "Keep the white margin around each tag when cutting.",
              px(20), PAGE[1] - px(40), 0.9)
        scale_bar(page, px(20), PAGE[1] - px(25))
        pages.append(page)

    for tag_id in sorted(tags.names):
        side = px(tags.sizes_m[tag_id] * 1000) // cells * cells    # whole pixels per cell
        img = cv2.aruco.generateImageMarker(d, tag_id, side, borderBits=1)
        if page is not None and x + side + px(20) > PAGE[0]:
            x, y = px(20), y + prev + px(30)
        if page is None or y + side + px(60) > PAGE[1]:
            if page is not None:
                finish()
            page = np.full(PAGE[::-1], 255, np.uint8)
            x, y = px(20), px(20)
        page[y:y + side, x:x + side] = img
        label(page, f"{tags.family} id {tag_id} ({tags.name(tag_id)})", x, y + side + px(6), 0.7)
        label(page, f"nominal {side / DPI * 25.4:.1f} mm, measure it", x, y + side + px(11), 0.7)
        x += side + px(25)
        prev = side
    finish()
    paths = [out.with_name(out.stem + (f"_{k + 1}" if k else "") + out.suffix) for k in range(len(pages))]
    for path, pg in zip(paths, pages):
        cv2.imwrite(str(path), pg)
    write_pdf(out.with_suffix(".pdf"), pages)
    return paths


def board_sheet(out: Path, square_mm: float) -> float:
    """The board on a landscape page (it's wider than tall). Returns the nominal square, mm."""
    page = np.full(LANDSCAPE[::-1], 255, np.uint8)
    sq = px(square_mm)                             # whole pixels per square: crisp edges
    w, h = sq * BOARD_SQUARES[0], sq * BOARD_SQUARES[1]
    if w > LANDSCAPE[0] - px(16) or h > LANDSCAPE[1] - px(40):
        sys.exit(f"a {square_mm} mm square board doesn't fit on letter paper")
    nominal = sq / DPI * 25.4
    board = charuco_board(nominal / 1000)
    img = board.generateImage((w, h), marginSize=0, borderBits=1)
    x, y = (LANDSCAPE[0] - w) // 2, px(10)
    page[y:y + h, x:x + w] = img
    label(page, f"ChArUco {BOARD_SQUARES[0]}x{BOARD_SQUARES[1]}, DICT_5X5_100, "
          f"nominal square {nominal:.2f} mm: measure one", x, y + h + px(7), 0.9)
    scale_bar(page, x, y + h + px(12))
    cv2.imwrite(str(out), page)
    write_pdf(out.with_suffix(".pdf"), [page])
    return nominal


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="targets")
    ap.add_argument("--tags", default=str(Path(__file__).resolve().parent.parent / "config" / "tags.json"))
    ap.add_argument("--square-mm", type=float, default=35.0, help="nominal ChArUco square size")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag_pages = tag_sheet(out / "tags.png", TagSet.load(a.tags))
    square = board_sheet(out / "charuco.png", a.square_mm)
    print(f"wrote {out / 'charuco.pdf'} (board, square {square:.2f} mm nominal) and "
          f"{out / 'tags.pdf'} ({len(tag_pages)} pages), plus PNGs. Letter, print at 100%.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
