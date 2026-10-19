#!/usr/bin/env python3
"""Live tag positions in the station's reference frame, and the camera's noise floor.

    python scripts/track_tags.py --source /dev/video0 --calib calib/c310_1280x960.json \\
        --exposure 300

The tag set (family, names, sizes, which tags are the reference) comes from ``--tags``
(default ``config/tags.json``). ``--size ID=METRES`` overrides one tag's size for this run.

Before the camera is calibrated, ``--hfov DEG`` (instead of ``--calib``) uses a rough
pinhole guess: good for placing the camera and checking tracking, not for measuring.

A window shows the tags (reference tags in blue) and a table: each tag's detection rate,
size in pixels, position in the reference frame (mm) and spread over the last ``--window``
frames. With nothing moving, that spread is the **noise floor**: the smallest repeatability
the setup can resolve. Get it well under the spread you expect from the arms (aim for
~0.1 mm) before trusting a repeatability run. In the window, [ and ] step the exposure,
Q/ESC quits.

``--survey N`` averages N frames of the reference tags (table and torso), writes their
layout to ``--layout`` (default ``config/layout.json``, not committed) and exits. Do it after
setup, once the table and torso won't be moved again, and redo it whenever either has been.
Until then only the anchor (the first reference tag) defines the frame. The ``worst``
residual on the reference line is the reference tag that fits the layout worst: a jump of
several pixels means it moved since the survey.

``--measure N`` takes one N-frame averaged reading of every visible tag, prints it as JSON
and exits. It's the same reading the repeatability test will take at each stop. Tags seen
in fewer than N of the first 3N frames are listed under ``missing``.
``--no-window`` skips the preview, for SSH sessions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict, deque
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from teleop_station.vision import CameraCalibration, Pose, TagSet, TagTracker, mean_pose, survey  # noqa: E402
from teleop_station.vision.camera import describe, latest, open_device, set_exposure      # noqa: E402
from teleop_station.vision.tags import draw                                               # noqa: E402

DEFAULT_TAGS_FILE = Path(__file__).resolve().parent.parent / "config" / "tags.json"
DEFAULT_LAYOUT_FILE = DEFAULT_TAGS_FILE.with_name("layout.json")


def load_layout(path: str) -> dict[int, Pose]:
    if not Path(path).exists():
        return {}
    return {int(i): Pose.from_json(p) for i, p in json.loads(Path(path).read_text())["layout"].items()}


def save_layout(path: str, layout: dict[int, Pose], tags: TagSet) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({
        "anchor": tags.anchor, "surveyed": time.strftime("%Y-%m-%d %H:%M:%S"),
        "layout": {str(i): p.to_json() for i, p in sorted(layout.items())}}, indent=2) + "\n")


def parse_sizes(base: dict[int, float], items: list[str]) -> dict[int, float]:
    sizes = dict(base)
    for it in items:
        k, v = it.split("=")
        sizes[int(k)] = float(v)
    return sizes


def table(tags: TagSet, rates: dict[int, float], side: dict[int, float],
          rel: dict, spread: dict) -> list[str]:
    rows = []
    for i in sorted(tags.names):
        n = tags.name(i)
        ref = "*" if i in tags.reference else " "
        row = f"{i:2d}{ref}{n:20s} {rates.get(i, 0) * 100:4.0f}%"
        if i in side:
            row += f" {side[i]:4.0f}px"
        if n in rel:
            p = rel[n].t * 1000
            row += f"  [{p[0]:8.2f} {p[1]:8.2f} {p[2]:8.2f}] mm"
            if n in spread:
                row += f"  ±{spread[n]:.3f}"
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    cal = ap.add_mutually_exclusive_group(required=True)
    cal.add_argument("--calib", help="camera calibration JSON")
    cal.add_argument("--hfov", type=float, help="rough guess instead: horizontal FOV, degrees")
    ap.add_argument("--width", type=int, default=1280, help="with --hfov: capture width")
    ap.add_argument("--height", type=int, default=960, help="with --hfov: capture height")
    ap.add_argument("--tags", default=str(DEFAULT_TAGS_FILE))
    ap.add_argument("--layout", default=str(DEFAULT_LAYOUT_FILE), help="surveyed reference layout")
    ap.add_argument("--size", action="append", default=[], metavar="ID=METRES")
    ap.add_argument("--exposure", type=float, help="manual exposure, driver units (UVC: 100 us)")
    ap.add_argument("--window", type=int, default=30, help="frames in the rolling spread")
    ap.add_argument("--measure", type=int, default=0, metavar="N")
    ap.add_argument("--survey", type=int, default=0, metavar="N")
    ap.add_argument("--no-window", action="store_true")
    a = ap.parse_args()

    calib = (CameraCalibration.load(a.calib) if a.calib
             else CameraCalibration.guess(a.width, a.height, a.hfov))
    tags = TagSet.load(a.tags)
    tags = replace(tags, sizes_m=parse_sizes(tags.sizes_m, a.size), layout=load_layout(a.layout))
    tracker = TagTracker(calib, tags)
    w, h = calib.image_size
    cap = open_device(a.source, w, h, exposure=a.exposure)
    exposure = a.exposure
    print("camera:", describe(cap), file=sys.stderr)
    if a.hfov:
        print(f"calibration: GUESS from hfov {a.hfov} deg; distances are approximate", file=sys.stderr)
    if len(tags.reference) > 1 and not tags.layout:
        print(f"reference: layout not surveyed, using tag {tags.anchor} alone (run --survey)", file=sys.stderr)

    keep = max(a.window, a.measure)
    hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=keep))
    seen_hist: deque = deque(maxlen=a.window)
    survey_frames: list = []
    counts: dict[str, int] = defaultdict(int)
    frames_seen = no_ref = 0
    printed = 0
    try:
        while True:
            frame = latest(cap)
            frames_seen += 1
            seen = tracker.detect(frame)
            seen_hist.append(set(seen))
            rel, fit = tracker.relative(seen)
            if fit is None:
                no_ref += 1
            for name, pose in rel.items():
                hist[name].append(pose)
                counts[name] += 1

            if a.survey:
                if tags.anchor in seen and all(i in seen for i in tags.reference[1:]):
                    survey_frames.append(seen)
                if len(survey_frames) >= a.survey:
                    layout = survey(tags, survey_frames)
                    save_layout(a.layout, layout, tags)
                    for i, p in layout.items():
                        d = np.linalg.norm(p.t) * 1000
                        print(f"tag {i} ({tags.name(i)}) in tag {tags.anchor}'s frame: "
                              f"{(p.t * 1000).round(2).tolist()} mm, {d:.1f} mm apart", file=sys.stderr)
                    missing = set(tags.reference[1:]) - set(layout)
                    print(f"wrote {a.layout}" + (f"; NOT surveyed (never seen with the anchor): "
                                                 f"{sorted(missing)}" if missing else ""), file=sys.stderr)
                    return 0
                if frames_seen >= a.survey * 3 and len(survey_frames) < a.survey // 3:
                    print(f"reference tags {tags.reference} seen together in only "
                          f"{len(survey_frames)}/{frames_seen} frames; nothing surveyed", file=sys.stderr)
                    return 1

            elif a.measure:
                enough = {n for n, c in counts.items() if c >= a.measure}
                expected = {tags.name(i) for i in tags.names}
                if enough >= expected or frames_seen >= a.measure * 3:
                    if not enough:
                        print(f"reference seen in {frames_seen - no_ref}/{frames_seen} frames; "
                              f"nothing measured", file=sys.stderr)
                        return 1
                    out = {}
                    for name in sorted(enough):
                        mean, std = mean_pose(list(hist[name])[-a.measure:])
                        out[name] = {"pos_mm": (mean.t * 1000).round(3).tolist(),
                                     "std_mm": (std * 1000).round(4).tolist()}
                    print(json.dumps({"frames": a.measure, "tags": out,
                                      "missing": sorted(expected - enough)}, indent=2))
                    return 0

            rates = {i: sum(i in s for s in seen_hist) / len(seen_hist) for i in tags.names}
            spread = {}
            for name, q in hist.items():
                recent = list(q)[-a.window:]
                if len(recent) >= 2 and name in rel:
                    spread[name] = float(np.linalg.norm(mean_pose(recent)[1]) * 1000)
            rows = table(tags, rates, {i: s.side_px for i, s in seen.items()}, rel, spread)
            if fit:
                worst = max(fit.per_tag_px, key=fit.per_tag_px.get)
                ref_line = (f"reference: tags {sorted(fit.tags_used)}, fit {fit.rms_px:.2f} px, "
                            f"worst {worst} {fit.per_tag_px[worst]:.2f} px")
            else:
                ref_line = "reference: NOT SEEN, no positions"
            exp_line = f"exposure {exposure:.0f}" if exposure is not None else "exposure auto"

            if not a.measure and not a.survey and frames_seen % 5 == 0 and sys.stderr.isatty():
                lines = [ref_line + "   " + exp_line] + rows
                if printed:
                    print(f"\033[{printed}F", end="", file=sys.stderr)
                print("\n".join(l.ljust(100) for l in lines), file=sys.stderr)
                printed = len(lines)

            if not a.no_window:
                show = draw(frame, calib, tags, seen)
                scale = min(1.0, 1300 / show.shape[0], 1700 / show.shape[1])
                if scale < 1.0:
                    show = cv2.resize(show, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                # The readout goes in its own panel beside the image, so it never hides a tag.
                # Drawn in one pass: OpenCV 5's font spaces letters by stroke weight, so the
                # usual thick-outline-then-thin-text trick draws two misaligned copies.
                lines = [ref_line, exp_line, ""] + rows
                panel = np.zeros((show.shape[0], 760, 3), np.uint8)
                for k, line in enumerate(lines):
                    cv2.putText(panel, line, (12, 30 + 26 * k), cv2.FONT_HERSHEY_PLAIN, 1.3,
                                (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("tags", np.hstack([show, panel]))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return 0
                if key in (ord("["), ord("]")):
                    cur = exposure if exposure is not None else cap.get(cv2.CAP_PROP_EXPOSURE)
                    exposure = max(1.0, cur * (0.8 if key == ord("[") else 1.25))
                    set_exposure(cap, exposure)
    except KeyboardInterrupt:
        return 0
    finally:
        cap.release()


if __name__ == "__main__":
    sys.exit(main())
