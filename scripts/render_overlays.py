#!/usr/bin/env python3
"""Figures for a report: a measured run's own photos with the tags and the arm model drawn on.

    python scripts/render_overlays.py runs/rep_20261020T191207.jsonl runs/shots_rep_20261020T191207 \\
        --arm left --accuracy runs/accuracy_20261020T191207_left.json --stop S2 --cycle 15 \\
        --out runs/figures

Offline: reads the saved stills, moves nothing. For one stop visit it writes

- ``<run>_<stop>_tags.png``: every detected tag outlined with its id (reference tags in cyan, the
  arm's in yellow), the station reference frame's axes, and the hand cube's fitted pose as axes,
- ``<run>_<stop>_model.png``: the arm's joint chain, shoulder to wrist, as the calibrated model
  puts it from the encoder angles (green) and as the controller believes it is, the commanded
  angles with no zero offsets (magenta); a cross where each says the cube is, and a circle where
  the camera sees it. The gap between magenta and the circle is the visit's accuracy error,

each cropped round the arm and scaled to ``--width`` pixels. The stills are matched to stops by
order: the camera's first still, the runner's ``--survey-shots`` (10), then ``--shots`` per stop.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import fit_accuracy as FA                                                         # noqa: E402
from teleop_station.vision import CameraCalibration, Pose, TagSet, TagTracker      # noqa: E402

YELLOW, CYAN, GREEN, MAGENTA, WHITE = (0, 220, 255), (255, 220, 0), (60, 220, 60), (230, 60, 230), (255, 255, 255)


def project(calib, cam_from_ref, P):
    """Reference-frame points (N,3) metres -> pixels (N,2)."""
    rv = cv2.Rodrigues(cam_from_ref.R)[0]
    return cv2.projectPoints(np.asarray(P, float).reshape(-1, 3), rv, cam_from_ref.t, calib.K, calib.dist)[0].reshape(-1, 2)


def label(img, text, xy, color, s):
    x, y = int(xy[0]), int(xy[1])
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (0, 0, 0), int(5 * s), cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, color, int(2 * s), cv2.LINE_AA)


def axes(img, calib, cam_from_x, length, s):
    rv = cv2.Rodrigues(cam_from_x.R)[0]
    cv2.drawFrameAxes(img, calib.K, calib.dist, rv, cam_from_x.t.reshape(3, 1), length, int(3 * s))


def crop(img, pts, pad, width):
    pts = np.asarray(pts)
    h, w = img.shape[:2]
    x0, y0 = np.clip(pts.min(0) - pad, 0, [w, h]).astype(int)
    x1, y1 = np.clip(pts.max(0) + pad, 0, [w, h]).astype(int)
    c = img[y0:y1, x0:x1]
    return cv2.resize(c, (width, int(c.shape[0] * width / c.shape[1])), interpolation=cv2.INTER_AREA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("shots")
    ap.add_argument("--arm", default="left", choices=("left", "right"))
    ap.add_argument("--accuracy", required=True, help="fit_accuracy.py --out of this run")
    ap.add_argument("--stop", default="S2")
    ap.add_argument("--cycle", type=int, default=15, help="which visit (0-based cycle)")
    ap.add_argument("--survey-shots", type=int, default=10)
    ap.add_argument("--calib", default=str(ROOT / "calib" / "a7s3_sigma16_2768x1848.json"))
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--out", default=str(ROOT / "runs" / "figures"))
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.run)]
    hdr = next(r for r in rows if r["kind"] == "header")
    stops = [r for r in rows if r["kind"] == "stop"]
    k = next(i for i, s in enumerate(stops) if s["label"] == a.stop and s["cycle"] == a.cycle)
    visit = stops[k]
    files = sorted(glob.glob(f"{a.shots}/*.jpg"))
    per = hdr["args"]["shots"]
    first = 1 + a.survey_shots + per * k
    frame = cv2.imread(files[first])

    calib = CameraCalibration.load(a.calib)
    tags = TagSet.load(ROOT / "config" / "tags.json")
    layout = {int(i): Pose.from_json(p) for i, p in hdr["camera_layout"].items()}
    tracker = TagTracker(calib, replace(tags, layout=layout))
    seen = tracker.detect(frame)
    rel, fit = tracker.relative(seen)
    if fit is None:
        sys.exit("no reference frame in that still")
    s = frame.shape[1] / 2768
    arm_ids = {i for b, ids in tags.bodies.items() if b.startswith(a.arm) for i in ids}

    # ── tags ────────────────────────────────────────────────────────────────
    img = frame.copy()
    for i, sg in seen.items():
        color = CYAN if i in tags.reference else YELLOW if i in arm_ids else WHITE
        cv2.polylines(img, [sg.corners.astype(np.int32).reshape(-1, 1, 2)], True, color, int(3 * s), cv2.LINE_AA)
        label(img, str(i), sg.corners.mean(0) + [-12 * s, 10 * s], color, s)
    axes(img, calib, fit.cam_from_ref, 0.06, s)
    hand = rel.get(f"{a.arm}_hand")
    if hand is not None:
        axes(img, calib, fit.cam_from_ref @ hand, 0.05, s)
    keep = [sg.corners for i, sg in seen.items() if i in arm_ids or i in tags.reference]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(a.run).stem}_{a.arm}_{a.stop}_c{a.cycle + 1}"
    cv2.imwrite(str(out / f"{stem}_tags.png"), crop(img, np.vstack(keep), 80 * s, a.width))

    # ── model wireframe ─────────────────────────────────────────────────────
    acc = json.loads(Path(a.accuracy).read_text())
    bodies = list(acc["body_on_link"])
    m = FA.Model(a.arm, bodies, offsets=True)
    x = np.r_[acc["base"], np.concatenate([acc["body_on_link"][b] for b in bodies]),
              np.radians([acc["zero_offsets_deg"][FA.JOINTS[j]] for j in FA.FIT_OFFSETS])]
    x0 = x.copy(); x0[m.k_off:] = 0
    idx = [hdr["names"].index(f"{a.arm}_{j}_joint") for j in FA.JOINTS]
    q_enc, q_cmd = np.array(visit["encoder_mean"])[idx], np.array(visit["commanded"])[idx]
    base = FA.T_of(x[:6])
    img = frame.copy()
    pts = []
    for xx, q, color, name in ((x, q_enc, GREEN, "model, encoders"), (x0, q_cmd, MAGENTA, "commanded, no offsets")):
        qq = m.chain.device_to_urdf(q) + m.dq(xx)
        J = m.chain.joint_origins(qq)
        F = m.chain.frames(qq)
        chain = np.vstack([J, F[-1][:3, 3]])
        P = (base[:3, :3] @ chain.T).T + base[:3, 3]
        uv = project(calib, fit.cam_from_ref, P)
        pts.append(uv)
        cv2.polylines(img, [uv.astype(np.int32).reshape(-1, 1, 2)], False, color, int(4 * s), cv2.LINE_AA)
        for p in uv:
            cv2.circle(img, tuple(p.astype(int)), int(7 * s), color, -1, cv2.LINE_AA)
        cube = m.predict(xx, q)["hand"][:3, 3]
        c = project(calib, fit.cam_from_ref, cube)[0]
        cv2.drawMarker(img, tuple(c.astype(int)), color, cv2.MARKER_CROSS, int(30 * s), int(4 * s), cv2.LINE_AA)
        pts.append(c[None])
    if hand is None:
        sys.exit("the hand cube isn't in that still")
    seen_uv = project(calib, fit.cam_from_ref, hand.t)[0]
    cv2.circle(img, tuple(seen_uv.astype(int)), int(16 * s), YELLOW, int(3 * s), cv2.LINE_AA)
    err = {n: np.linalg.norm(m.predict(xx, q)["hand"][:3, 3] - hand.t) * 1000
           for n, xx, q in (("enc", x, q_enc), ("cmd", x0, q_cmd))}
    gap_px = float(np.linalg.norm(pts[3][0] - seen_uv))          # commanded cross to the seen cube
    mm_per_px = 36.7 / seen[next(i for i in tags.bodies[f"{a.arm}_hand"] if i in seen)].side_px
    full = img.copy()
    img = crop(full, np.vstack(pts), 140 * s, a.width)
    # A legend band above the picture, sized to it.
    sc = a.width / 900
    lines = ((f"green: calibrated model, encoder angles (cube {err['enc']:.1f} mm off)", GREEN),
             (f"magenta: commanded angles, no zero offsets (cube {err['cmd']:.1f} mm off)", MAGENTA),
             ("yellow circle: the cube where the camera sees it", YELLOW))
    band = np.zeros((int((30 + 40 * len(lines)) * sc), img.shape[1], 3), np.uint8)
    for k2, (t, color) in enumerate(lines):
        label(band, t, (16 * sc, (38 + 40 * k2) * sc), color, sc)
    cv2.imwrite(str(out / f"{stem}_model.png"), np.vstack([band, img]))
    # The cube close up: the crosses and the circle, 60 mm round it.
    r = 60 / mm_per_px
    cv2.imwrite(str(out / f"{stem}_cube.png"), crop(full, np.vstack([seen_uv - r, seen_uv + r]), 0, 900))
    print(f"wrote {out / (stem + '_tags.png')} and {out / (stem + '_model.png')}; reference fit {fit.rms_px:.2f} px; "
          f"cube error: encoders + model {err['enc']:.1f} mm, as commanded {err['cmd']:.1f} mm; "
          f"on the image the commanded cross is {gap_px * mm_per_px:.1f} mm from the cube ({gap_px:.0f} px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
