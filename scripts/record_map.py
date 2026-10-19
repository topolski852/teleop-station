#!/usr/bin/env python3
"""Record the camera-to-encoder map: move the arms by hand, hold still, a sample is kept.

    python scripts/record_map.py --source /dev/video0 --calib calib/c310_1280x960.json \\
        --exposure 400

With the Sony tethered (a shot plus detection is ~2 s), hold each pose longer:

    python scripts/record_map.py --source gphoto2:runs/shots_<name> \\
        --calib calib/a7s3_sigma16_2768x1848.json --hold 6 --min-frames 3

Nothing here commands an arm. Joint angles are read from humanoid-web's cached telemetry
(``teleop_station.joint_source``). The joints must already be live (IDLE, DAMPING or
ENABLED). In IDLE they are limp and back-drivable, so they can be posed by hand.

Move an arm to a pose and hold it (or rest it on a support). Once every joint and every
visible arm tag has been still for ``--hold`` seconds, a **sample** is written: mean joint
angles, and each tag's mean pose in the reference frame with its spread. The next sample
waits until some joint has moved by ``--min-move`` degrees. SPACE asks for a sample at the
next still moment even without that move; it never keeps one while the arm is moving.
Q/ESC quits.

The samples are a convenience: ``fit_map.py`` finds the still moments in the ``frame``
lines itself.

Survey the reference tags first (``track_tags.py --survey``), with the same exposure.

Every camera frame is also logged with the nearest encoder reading (``frame`` lines), for
the range-of-motion map. Output: ``runs/map_<time>.jsonl`` (one JSON object per line; the
first is a header).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from track_tags import DEFAULT_LAYOUT_FILE, DEFAULT_TAGS_FILE, load_layout   # noqa: E402
from teleop_station.joint_source import WebJointSource                       # noqa: E402
from teleop_station.vision import CameraCalibration, TagSet, TagTracker, mean_pose  # noqa: E402
from teleop_station.vision.camera import describe, open_device               # noqa: E402
from teleop_station.vision.tags import draw                                  # noqa: E402

ARM_TAGS = ("right_hand", "right_wrist", "right_shoulder_roll",
            "left_hand", "left_wrist", "left_shoulder_roll")


def short(joint: str) -> str:
    return joint.replace("_joint", "").replace("shoulder", "sh")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--exposure", type=float, help="manual exposure (UVC: 100 us units)")
    ap.add_argument("--tags", default=str(DEFAULT_TAGS_FILE))
    ap.add_argument("--layout", default=str(DEFAULT_LAYOUT_FILE))
    ap.add_argument("--joints-url", default="http://127.0.0.1:8000/api/status")
    ap.add_argument("--hold", type=float, default=3.0, help="seconds still before a sample")
    ap.add_argument("--still-deg", type=float, default=0.15, help="max joint range while still")
    ap.add_argument("--still-mm", type=float, default=1.5, help="max tag spread while still")
    ap.add_argument("--min-move", type=float, default=2.0, help="degrees between samples")
    ap.add_argument("--min-frames", type=int, default=3,
                    help="camera frames in the hold window (a tethered still camera gives ~0.5/s)")
    ap.add_argument("--arms", default="left,right", help="arms that must be live: left,right")
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--no-window", action="store_true")
    a = ap.parse_args()

    calib = CameraCalibration.load(a.calib)
    tags = replace(TagSet.load(a.tags), layout=load_layout(a.layout))
    if len(tags.reference) > 1 and not tags.layout:
        sys.exit(f"no surveyed layout in {a.layout}: run track_tags.py --survey first")
    tracker = TagTracker(calib, tags)

    joints = WebJointSource(a.joints_url).start()
    t_wait = time.time()
    while joints.latest() is None:
        if time.time() - t_wait > 3:
            sys.exit(f"no joint angles from {a.joints_url}: {joints.error}")
        time.sleep(0.05)
    arms = [f"{x.strip()}_" for x in a.arms.split(",")]
    names = sorted(n for n in joints.latest().position if n.startswith(tuple(arms)))
    if not joints.latest().live(names):
        bad = {n: joints.latest().state[n] for n in names
               if joints.latest().state[n] not in ("IDLE", "DAMPING", "ENABLED")}
        sys.exit(f"joints not live (connect them in humanoid-web first, or narrow --arms): {bad}")
    arm_tags = tuple(n for n in ARM_TAGS if n.startswith(tuple(arms)))

    cap = open_device(a.source, *calib.image_size, exposure=a.exposure)
    print("camera:", describe(cap), file=sys.stderr)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out_path = Path(a.out) / f"map_{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    out = out_path.open("w")
    out.write(json.dumps({"kind": "header", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "calib": a.calib, "calib_rms_px": calib.rms_px, "exposure": a.exposure,
                          "tags": a.tags, "family": tags.family, "sizes_m": tags.sizes_m,
                          "reference": list(tags.reference),
                          "layout": {str(i): p.to_json() for i, p in tags.layout.items()},
                          "joints_url": a.joints_url, "joint_frame": "device rad, no signs",
                          "arms": a.arms, "joints": names}) + "\n")
    print(f"writing {out_path}", file=sys.stderr)

    window: deque = deque()                 # (t, {tag: Pose}) camera frames in the hold window
    want_manual = False                     # SPACE pressed: keep the next still window
    last_sample_q: np.ndarray | None = None
    n_samples = n_frames = 0
    status = ""
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("camera stopped delivering frames")
            t = time.time()
            jr = joints.latest()                    # before detection: closest to the frame
            n_frames += 1
            seen = tracker.detect(frame)
            rel, fit = tracker.relative(seen)
            out.write(json.dumps({
                "kind": "frame", "t": round(t, 4),
                "tags": {n: (p.t * 1000).round(3).tolist() for n, p in rel.items()},
                "joints": {n: round(jr.position[n], 6) for n in names},
                "joint_age_s": round(t - jr.t, 3),
                "ref_px": round(max(fit.per_tag_px.values()), 3) if fit else None}) + "\n")

            window.append((t, rel))
            while window and window[0][0] < t - a.hold:
                window.popleft()
            jhist = joints.since(t - a.hold)
            still, why = False, ""
            if fit is None:
                why = "reference not seen"
            elif not jr.live(names):
                why = "joints not live"
            elif (len(jhist) < 0.6 * a.hold * 20 or len(window) < a.min_frames
                  or t - window[0][0] < 0.6 * a.hold):
                why = "collecting"
            else:
                q = np.array([[r.position[n] for n in names] for r in jhist])
                jrange = np.degrees(np.ptp(q, axis=0).max())
                spread = 0.0
                for n in arm_tags:
                    pts = [r[n].t for _, r in window if n in r]
                    if len(pts) >= 2:
                        spread = max(spread, float(np.ptp(pts, axis=0).max() * 1000))
                if jrange > a.still_deg:
                    why = f"joints moving ({jrange:.2f} deg)"
                elif spread > a.still_mm:
                    why = f"tags moving ({spread:.1f} mm)"
                else:
                    still = True
            qmean = np.array([np.mean([r.position[n] for r in jhist]) for n in names]) if jhist else None
            moved = (last_sample_q is None or qmean is None
                     or np.degrees(np.abs(qmean - last_sample_q).max()) >= a.min_move)

            key = -1
            if not a.no_window:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            if key == ord(" "):
                want_manual = True
            manual = want_manual and still
            if still and (moved or manual):
                want_manual = False
                sample = {"kind": "sample", "t": round(t, 4), "manual": manual,
                          "joints": {n: round(float(v), 6) for n, v in zip(names, qmean)},
                          "joints_range_deg": {n: round(float(np.degrees(np.ptp([r.position[n] for r in jhist]))), 4)
                                               for n in names},
                          "tags": {}, "ref_tags": sorted(fit.tags_used),
                          "ref_px": round(max(fit.per_tag_px.values()), 3)}
                for n in sorted({k for _, r in window for k in r}):
                    poses = [r[n] for _, r in window if n in r]
                    if len(poses) >= min(3, a.min_frames):
                        m, sd = mean_pose(poses)
                        sample["tags"][n] = {"pos_mm": (m.t * 1000).round(3).tolist(),
                                             "R": m.R.round(6).tolist(),
                                             "std_mm": (sd * 1000).round(3).tolist(), "n": len(poses)}
                out.write(json.dumps(sample) + "\n")
                out.flush()
                n_samples += 1
                last_sample_q = qmean
                arm_seen = [n for n in arm_tags if n in sample["tags"]]
                status = f"sample {n_samples}: {len(arm_seen)} arm tags ({', '.join(arm_seen)})"
                print(status, file=sys.stderr)

            if not a.no_window:
                show = draw(frame, calib, tags, seen)
                scale = min(1.0, 1300 / show.shape[0], 1700 / show.shape[1])
                if scale < 1.0:
                    show = cv2.resize(show, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                state = ("STILL - sample kept" if still and not moved else
                         "STILL - sampling" if still else why)
                if want_manual:
                    state += "   (SPACE: waiting for still)"
                lines = [f"samples {n_samples}   frames {n_frames}",
                         f"{state}", status, "",
                         f"{'joint':18s} {'deg':>8s}  state"]
                for n in names:
                    lines.append(f"{short(n):18s} {np.degrees(jr.position[n]):8.2f}  {jr.state[n]}")
                lines += ["", f"arm tags seen: {sum(n in rel for n in arm_tags)}/{len(arm_tags)}",
                          f"reference: {sorted(fit.tags_used) if fit else 'NOT SEEN'}",
                          "SPACE keep sample   Q quit"]
                if joints.error:
                    lines.append(f"joints: {joints.error[:60]}")
                panel = np.zeros((show.shape[0], 620, 3), np.uint8)
                for k, line in enumerate(lines):
                    col = (80, 255, 80) if k == 1 and still else (255, 255, 255)
                    cv2.putText(panel, line, (12, 30 + 26 * k), cv2.FONT_HERSHEY_PLAIN, 1.3,
                                col, 1, cv2.LINE_AA)
                cv2.imshow("map", np.hstack([show, panel]))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        joints.stop()
        out.close()
        print(f"\n{n_samples} samples, {n_frames} frames -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
