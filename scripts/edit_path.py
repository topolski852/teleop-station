#!/usr/bin/env python3
"""Turn a hand-recorded path into a smooth, checked one, and preview it. Moves nothing.

    python scripts/edit_path.py runs/path_raw_demo.json            # -> runs/path_raw_demo.planned.json
    python scripts/edit_path.py runs/path_raw_demo.json --smooth 0.4 --vmax 10
    python scripts/edit_path.py runs/path_raw_demo.json --combine   # arms recorded one after the other

Finds the pauses (the stops), smooths each stretch between them, re-times it with a
minimum-jerk profile under the speed and acceleration caps, and checks it (limits, speed,
how far it strays from what was recorded: see ``teleop_station.motion.path``). Then writes:

- ``<out>.joints.png``: every joint, recorded (grey) against planned (green), with the stops
  and the planning limits,
- ``<out>.preview.mp4``: the planned motion as the arms' joint chain over a camera photo of the
  station (through the fitted map, ``calib/arm_map.json``), with the hand's recorded path
  (magenta) and planned path (green) traced. It uses the newest Sony shot in ``runs/`` unless
  ``--background`` names one, and needs the reference tags in it. Joint zeros change at every
  power-up, so treat the overlay as a guide to the shape of the motion, not to the millimetre.

Watch both before running it: ``run_repeatability.py --path <out>``.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from teleop_station.motion import JointLimits, RawPath, combine_arms, plan_path   # noqa: E402

DEG = np.pi / 180
DAEMON_CONFIG = "/home/nse/humanoid-studio/configs/humanoid_arms.json"
GREEN, MAGENTA, GREY, WHITE, YELLOW = (60, 220, 60), (220, 60, 220), (150, 150, 150), (255, 255, 255), (0, 220, 255)


def joint_plot(raw, plan, limits, path):
    names = raw.names
    W, H, cols, rows = 1800, 1250, 2, len(names) // 2
    img = np.full((H, W, 3), 30, np.uint8)
    pw, ph = W // cols, H // rows
    if "stop_spans_s" in plan.params:
        spans = plan.params["stop_spans_s"]
        t_end = raw.t[-1] + (plan.segments[-1].duration if len(plan.segments) > len(spans) - 1 else 0)
        # planned samples mapped onto the recording's time: segment k fills the gap between stops
        pt, pq = [], []
        for k, seg in enumerate(plan.segments):
            if k + 1 < len(spans):
                a, b = spans[k][1], spans[k + 1][0]
            else:                                       # the closing move, after the recording
                a, b = raw.t[-1], t_end
            pt.append(a + seg.t / max(seg.t[-1], 1e-9) * (b - a)); pq.append(seg.q)
        pt, pq = np.concatenate(pt), np.vstack(pq)
        raw_t = raw.t
    else:
        # Combined arms: the plan runs both at once, so it has its own timeline. Draw recording
        # and plan each across the full width (shapes compare; times don't line up).
        spans = []
        program = plan.lead_in + plan.segments + plan.lead_out
        pt, pq, t0 = [], [], 0.0
        for seg in program:
            pt.append(t0 + seg.t); pq.append(seg.q); t0 += seg.duration
        pt, pq = np.concatenate(pt), np.vstack(pq)
        t_end = raw.t[-1]
        pt = pt / pt[-1] * t_end
        raw_t = raw.t
    for j, n in enumerate(names):
        c, r = (0 if n.startswith("left") else 1), j % rows
        x0, y0 = c * pw + 70, r * ph + 30
        w, h = pw - 100, ph - 60
        vals = np.degrees(np.concatenate([raw.q[:, j], pq[:, j], [limits.lower[n], limits.upper[n]]]))
        lo, hi = vals.min() - 5, vals.max() + 5
        X = lambda tt: (x0 + tt / t_end * w).astype(int)
        Y = lambda v: (y0 + h - (np.degrees(v) - lo) / (hi - lo) * h).astype(int)
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)
        for a, b in spans:
            cv2.rectangle(img, (int(X(np.array(a))), y0), (int(X(np.array(b))), y0 + h), (55, 55, 25), -1)
        for lim in (limits.lower[n], limits.upper[n]):
            yy = int(Y(np.array(lim)))
            for xx in range(x0, x0 + w, 12):
                cv2.line(img, (xx, yy), (xx + 6, yy), (60, 60, 200), 1)
        cv2.polylines(img, [np.c_[X(raw_t), Y(raw.q[:, j])].astype(np.int32)], False, GREY, 1, cv2.LINE_AA)
        cv2.polylines(img, [np.c_[X(pt), Y(pq[:, j])].astype(np.int32)], False, GREEN, 2, cv2.LINE_AA)
        cv2.putText(img, f"{n.replace('_joint', '')}  {lo:.0f}..{hi:.0f} deg", (x0 + 6, y0 + 18),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, "grey recorded, green planned (on the recording's time), bands = stops, red dashes = limits",
                (20, H - 10), cv2.FONT_HERSHEY_PLAIN, 1.3, WHITE, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def preview_video(raw, plan, background, path, fps=25, hold_s=0.8):
    from fit_map import LoadedMap
    from track_tags import DEFAULT_LAYOUT_FILE, DEFAULT_TAGS_FILE, load_layout
    from view_map import draw_chain, project
    from teleop_station.vision import CameraCalibration, TagSet, TagTracker
    calib = CameraCalibration.load(ROOT / "calib" / "a7s3_sigma16_2768x1848.json")
    amap = LoadedMap.load(ROOT / "calib" / "arm_map.json")
    img = cv2.imread(str(background))
    if img is None or img.shape[1::-1] != tuple(calib.image_size):
        return f"background {background} isn't a {calib.image_size[0]}x{calib.image_size[1]} Sony shot"
    tracker = TagTracker(calib, replace(TagSet.load(DEFAULT_TAGS_FILE), layout=load_layout(DEFAULT_LAYOUT_FILE)))
    fit = tracker.fit_reference(tracker.detect(img))
    if fit is None:
        return "no reference tags in the background"
    scale = 1280 / img.shape[1]
    base = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    base = (base * 0.6).astype(np.uint8)
    idx = {side: [plan.names.index(n) for n in amap.d["arms"][side]["joints"]] for side in amap.chains}

    def hand_uv(q):
        out = {}
        for side, ix in idx.items():
            _, tp = amap.points(side, q[ix])
            out[side] = project(calib, fit.cam_from_ref, tp["hand"])[0] * scale
        return out
    for trail, color, Q in (("raw", MAGENTA, raw.q[::5]), ("planned", GREEN, np.vstack([s.q[::3] for s in plan.segments]))):
        for side in idx:
            pts = np.array([hand_uv(q)[side] for q in Q]).astype(np.int32)
            cv2.polylines(base, [pts], False, color, 1 if trail == "raw" else 2, cv2.LINE_AA)
    for k, sq in enumerate(plan.stops):
        for side, uv in hand_uv(sq).items():
            cv2.putText(base, f"S{k + 1}", (int(uv[0]) + 8, int(uv[1]) - 8), cv2.FONT_HERSHEY_PLAIN, 1.4, YELLOW, 2)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (base.shape[1], base.shape[0]))
    frames = []
    def play(seg, label):
        for tt in np.arange(0, seg.duration, 1 / fps):
            j = min(len(seg.t) - 1, int(np.searchsorted(seg.t, tt)))
            frames.append((seg.q[j], f"{label}  t {tt:4.1f}/{seg.duration:.1f} s"))
    for seg in plan.lead_in:
        play(seg, "lead-in from rest")
    for k, seg in enumerate(plan.segments):
        frames += [(seg.q[0], f"stop S{k + 1}")] * int(hold_s * fps)
        play(seg, f"move {k + 1}")
    for seg in plan.lead_out:
        play(seg, "lead-out to rest")
    for q, label in frames:
        f = base.copy()
        for side, ix in idx.items():
            jp, _ = amap.points(side, q[ix])
            draw_chain(f, project(calib, fit.cam_from_ref, jp) * scale, GREEN, scale * 1.5)
        cv2.putText(f, label, (12, 28), cv2.FONT_HERSHEY_PLAIN, 1.6, WHITE, 2, cv2.LINE_AA)
        cv2.putText(f, "magenta = hand as recorded, green = as planned", (12, f.shape[0] - 12),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, WHITE, 1, cv2.LINE_AA)
        vw.write(f)
    vw.release()
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw")
    ap.add_argument("--out", help="planned path JSON (default: <raw>.planned.json)")
    ap.add_argument("--vmax", type=float, default=15.0, help="deg/s per joint")
    ap.add_argument("--amax", type=float, default=60.0, help="deg/s^2 per joint")
    ap.add_argument("--smooth", type=float, default=0.25, help="smoothing, seconds (more = smoother, strays more)")
    ap.add_argument("--pause-s", type=float, default=1.5, help="still this long = a stop")
    ap.add_argument("--still-deg", type=float, default=0.5)
    ap.add_argument("--max-deviation", type=float, default=2.0, help="deg the plan may stray from the recording")
    ap.add_argument("--limits", default="soft", choices=("soft", "hard"))
    ap.add_argument("--combine", action="store_true",
                    help="arms recorded one after the other: plan each, run them together")
    ap.add_argument("--via-speed", type=float, default=8.0, help="split strokes where the hand slowed below this, deg/s")
    ap.add_argument("--background", help="Sony shot of the station for the preview video")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()

    raw = RawPath.load(a.raw)
    limits = JointLimits.for_arms(DAEMON_CONFIG, tier=a.limits)
    kw = dict(still_deg=a.still_deg, dwell_s=a.pause_s, smooth_s=a.smooth, max_deviation_deg=a.max_deviation,
              via_speed_deg=a.via_speed)
    plan = (combine_arms(raw, limits, a.vmax * DEG, a.amax * DEG, **kw) if a.combine
            else plan_path(raw, limits, a.vmax * DEG, a.amax * DEG, **kw))
    plan.params["limits"] = a.limits
    out = Path(a.out or str(Path(a.raw).with_suffix("")) + ".planned.json")
    print(f"{a.raw}: {raw.t[-1]:.0f} s recorded -> {len(plan.stops)} stops, {len(plan.segments)} moves, "
          f"{plan.duration:.0f} s of motion per cycle at {a.vmax:.0f} deg/s"
          + (f"; arms combined (stops per arm {plan.params.get('stops_per_arm')}), lead-in/out "
             f"{plan.lead_in[0].duration:.0f} s each" if plan.lead_in else ""))
    for k, seg in enumerate(plan.segments):
        dev = plan.deviation_deg[k]
        frm, to = k + 1, (k + 1) % len(plan.stops) + 1
        moved = np.degrees(np.abs(seg.q[-1] - seg.q[0]))
        print(f"  move {k + 1}: S{frm} -> S{to}  {seg.duration:5.1f} s, largest joint change {moved.max():5.1f} deg, "
              + (f"strays {dev:.2f} deg from the recording" if dev == dev else "PLANNED (not recorded)"))
    for p in plan.problems:
        print(("  " if p.startswith("NOTE") else "  PROBLEM: ") + p)
    if not plan.segments:
        return 1
    plan.save(out)
    print(f"wrote {out}")
    png = out.with_suffix(".joints.png")
    joint_plot(raw, plan, limits, png)
    print(f"wrote {png}")
    if not a.no_video:
        bg = a.background
        if not bg:
            shots = sorted(glob.glob(str(ROOT / "runs" / "shots_*" / "*.jpg")), key=os.path.getmtime)
            bg = shots[-1] if shots else None
        err = preview_video(raw, plan, bg, out.with_suffix(".preview.mp4")) if bg else "no Sony shot in runs/"
        print(f"preview video: {err}" if err else f"wrote {out.with_suffix('.preview.mp4')} (background {bg})")
    bad = [p for p in plan.problems if not p.startswith("NOTE")]
    if bad:
        print("\nNOT runnable until the problems above are fixed (re-record, or --smooth / --limits).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
