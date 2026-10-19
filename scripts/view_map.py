#!/usr/bin/env python3
"""Live check of the camera-to-encoder map: where the encoders say the arms are, drawn on the
camera image, next to where the camera sees the tags.

    python scripts/view_map.py --source gphoto2:runs/shots_view \\
        --calib calib/a7s3_sigma16_2768x1848.json --map calib/arm_map.json

Nothing here commands an arm. Joint angles come from humanoid-web's cached telemetry
(``teleop_station.joint_source``); move the arms by hand while they are limp (IDLE).

On each frame:
- **yellow circle**: a tag where the camera sees it,
- **green**: the encoder angles through the fitted map (``fit_map.py --out``): the arm's joint
  chain, and a cross where each tag should be. A line joins it to the yellow circle, labelled
  with the 3D distance between them (mm),
- **magenta** (N toggles): the same encoder angles through core's kinematics as they stand
  (core's ``joint_sign``, no fitted zero offsets), on the same base. Where it disagrees with
  green, core and the real arm disagree. Before core c72dcc3 it swung the shoulder pitch the
  wrong way; now the two differ only by the zero offsets.

Every frame is logged like ``record_map.py`` (``runs/view_<time>.jsonl``), with the joint
angles read just before and just after the shot, so a session doubles as mapping data. A
large before/after difference means the arm moved during the shot; that frame is marked.
Q/ESC quits.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fit_map import JOINTS, LoadedMap                                        # noqa: E402
from track_tags import DEFAULT_LAYOUT_FILE, DEFAULT_TAGS_FILE, load_layout   # noqa: E402
from teleop_station.joint_source import WebJointSource                       # noqa: E402
from teleop_station.vision import CameraCalibration, TagSet, TagTracker      # noqa: E402
from teleop_station.vision.camera import describe, open_device               # noqa: E402

GREEN, MAGENTA, YELLOW, WHITE = (60, 220, 60), (220, 60, 220), (0, 230, 255), (255, 255, 255)


def project(calib, cam_from_ref, pts_m):
    P = np.atleast_2d(pts_m) @ cam_from_ref.R.T + cam_from_ref.t
    uv, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), calib.K, calib.dist)
    return uv.reshape(-1, 2)


def draw_chain(img, uv, color, s):
    pts = [tuple(int(v) for v in p) for p in uv]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, color, max(2, int(4 * s)), cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, max(4, int(8 * s)), color, -1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--map", default=str(ROOT / "calib" / "arm_map.json"))
    ap.add_argument("--exposure", type=float)
    ap.add_argument("--tags", default=str(DEFAULT_TAGS_FILE))
    ap.add_argument("--layout", default=str(DEFAULT_LAYOUT_FILE))
    ap.add_argument("--joints-url", default="http://127.0.0.1:8000/api/status")
    ap.add_argument("--moving-deg", type=float, default=0.5,
                    help="joint change across a shot that marks the frame as moving")
    ap.add_argument("--out", default=str(ROOT / "runs"))
    a = ap.parse_args()

    calib = CameraCalibration.load(a.calib)
    tags = replace(TagSet.load(a.tags), layout=load_layout(a.layout))
    tracker = TagTracker(calib, tags)
    amap = LoadedMap.load(a.map)
    joints = WebJointSource(a.joints_url).start()
    t_wait = time.time()
    while joints.latest() is None:
        if time.time() - t_wait > 3:
            sys.exit(f"no joint angles from {a.joints_url}: {joints.error}")
        time.sleep(0.05)
    names = [n for side in amap.chains for n in amap.d["arms"][side]["joints"]]

    cap = open_device(a.source, *calib.image_size, exposure=a.exposure)
    print("camera:", describe(cap), file=sys.stderr)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out_path = Path(a.out) / f"view_{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    out = out_path.open("w")
    out.write(json.dumps({"kind": "header", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "calib": a.calib, "map": a.map, "tags": a.tags, "family": tags.family,
                          "sizes_m": tags.sizes_m, "reference": list(tags.reference),
                          "layout": {str(i): p.to_json() for i, p in tags.layout.items()},
                          "joints_url": a.joints_url, "joint_frame": "device rad, no signs",
                          "joints": names}) + "\n")
    print(f"writing {out_path}", file=sys.stderr)

    show_nominal = True
    n_frames = 0
    try:
        while True:
            before = joints.latest()
            ok, frame = cap.read()
            after = joints.latest()
            if not ok:
                raise RuntimeError("camera stopped delivering frames")
            t = time.time()
            n_frames += 1
            q = {n: 0.5 * (before.position[n] + after.position[n]) for n in names}
            moved = max(abs(np.degrees(after.position[n] - before.position[n])) for n in names)
            seen = tracker.detect(frame)
            rel, fit = tracker.relative(seen)
            out.write(json.dumps({
                "kind": "frame", "t": round(t, 4),
                "tags": {n: (p.t * 1000).round(3).tolist() for n, p in rel.items()},
                "joints": {n: round(q[n], 6) for n in names},
                "joints_moved_deg": round(float(moved), 3),
                "ref_px": round(max(fit.per_tag_px.values()), 3) if fit else None}) + "\n")
            out.flush()

            scale = min(1.0, 1300 / frame.shape[0], 1800 / frame.shape[1])
            show = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            lines = [f"frame {n_frames}   " + ("ARM MOVED DURING SHOT" if moved > a.moving_deg else "still"),
                     "green = encoders through fitted map", "magenta = encoders through core, no offsets (N)",
                     "yellow = camera", ""]
            if fit is None:
                lines.append("reference tags NOT SEEN")
            else:
                for side, chain in amap.chains.items():
                    qs = np.array([q[n] for n in amap.d["arms"][side]["joints"]])
                    jp, tp = amap.points(side, qs)
                    draw_chain(show, project(calib, fit.cam_from_ref, jp) * scale, GREEN, scale)
                    if show_nominal:
                        jn, _ = amap.points(side, qs, nominal=True)
                        draw_chain(show, project(calib, fit.cam_from_ref, jn) * scale, MAGENTA, scale)
                    lines.append(f"{side} arm (deg):")
                    lines.append("  " + "  ".join(f"{j.split('_')[-1][:5]} {np.degrees(v):6.1f}"
                                                  for j, v in zip(JOINTS, qs)))
                    for tag, p in tp.items():
                        name = f"{side}_{tag}"
                        c = tuple(int(v) for v in project(calib, fit.cam_from_ref, p)[0] * scale)
                        cv2.drawMarker(show, c, GREEN, cv2.MARKER_CROSS, max(14, int(30 * scale)), 2)
                        if name in rel:
                            d = c2 = None
                            c2 = tuple(int(v) for v in project(calib, fit.cam_from_ref, rel[name].t)[0] * scale)
                            d = np.linalg.norm(rel[name].t - p) * 1000
                            cv2.circle(show, c2, max(8, int(16 * scale)), YELLOW, 2, cv2.LINE_AA)
                            cv2.line(show, c, c2, YELLOW, 1, cv2.LINE_AA)
                            cv2.putText(show, f"{d:.1f}", (c2[0] + 10, c2[1] - 10),
                                        cv2.FONT_HERSHEY_PLAIN, 1.4, YELLOW, 2, cv2.LINE_AA)
                            lines.append(f"    {tag:14s} {d:6.1f} mm")
                        else:
                            lines.append(f"    {tag:14s}   (not seen)")
            lines += ["", "N nominal on/off   Q quit"]
            panel = np.zeros((show.shape[0], 560, 3), np.uint8)
            for k, line in enumerate(lines):
                cv2.putText(panel, line, (12, 30 + 26 * k), cv2.FONT_HERSHEY_PLAIN, 1.3, WHITE, 1, cv2.LINE_AA)
            cv2.imshow("map check", np.hstack([show, panel]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("n"):
                show_nominal = not show_nominal
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        joints.stop()
        out.close()
        print(f"\n{n_frames} frames -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
