#!/usr/bin/env python3
"""Turn a spin survey's shots into stop records for fit_accuracy.py (more poses, wrist varied).

    python scripts/spin_to_stops.py runs/spin_20261020T174702.json runs/shots_spin_20261020T174702 \\
        --plan runs/path_raw_test3.mirror.planned.json --arm left \\
        --servo-from runs/rep_20261020T185829.jsonl --out runs/spinstops_20261020T174702.jsonl

A spin survey at the stops (``spin_survey.py --at-stops``) logs which stop and wrist step each
shot was taken at, but not the encoders. The commanded angles are the plan's stop with the wrist
turned; the encoder angles are taken as commanded plus the servo error a measured run saw at that
stop (``--servo-from``; at a stop it repeats to ~0.05 deg, it's gravity on a still arm). The
reference frame is surveyed from the session's own shots (only the arm moved).

Each shot becomes one stop record labelled ``S3@+45`` (S3, wrist +45 deg), with the camera
poses in the reference frame as the runner writes them.
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

from teleop_station.motion import PlannedPath                                          # noqa: E402
from teleop_station.vision import CameraCalibration, TagSet, TagTracker, survey        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", help="runs/spin_<time>.json")
    ap.add_argument("shots", help="runs/shots_spin_<time>")
    ap.add_argument("--plan", required=True, help="the planned path the survey ran")
    ap.add_argument("--arm", default="left", choices=("left", "right"))
    ap.add_argument("--servo-from", required=True, metavar="runs/rep_*.jsonl",
                    help="a measured run of the same path that stopped at every stop")
    ap.add_argument("--calib", default=str(ROOT / "calib" / "a7s3_sigma16_2768x1848.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = json.loads(Path(a.record).read_text())["rows"]
    files = sorted(glob.glob(f"{a.shots}/*.jpg"))
    calib = CameraCalibration.load(a.calib)
    tags = TagSet.load(ROOT / "config" / "tags.json")
    det = TagTracker(calib, tags)
    frames = [cv2.imread(f) for f in files]
    seen = [det.detect(f) for f in frames]
    # The camera can take a shot before the first logged one; line them up by the tags seen.
    best = min(range(len(files) - len(rows) + 1),
               key=lambda o: sum(set(rows[k]["seen"]) != set(seen[k + o]) for k in range(len(rows))))
    bad = sum(set(rows[k]["seen"]) != set(seen[k + best]) for k in range(len(rows)))
    print(f"{len(files)} shots, {len(rows)} logged; offset {best}, {bad} with a different set of tags")
    layout = survey(tags, seen)
    tracker = TagTracker(calib, replace(tags, layout=layout))

    plan = PlannedPath.load(a.plan)
    names = [n for n in plan.names if n.startswith(a.arm + "_")]
    keep = [plan.names.index(n) for n in names]
    stop_q = {f"S{k + 1}": np.asarray(q)[keep] for k, q in enumerate(plan.stops)}
    wrist = names.index(f"{a.arm}_wrist_yaw_joint")

    srows = [json.loads(l) for l in open(a.servo_from)]
    shdr = next(r for r in srows if r["kind"] == "header")
    sidx = [shdr["names"].index(n) for n in names]
    servo = {}
    for r in srows:
        if r["kind"] == "stop":
            servo.setdefault(r["label"], []).append(np.array(r["encoder_mean"])[sidx] - np.array(r["commanded"])[sidx])
    servo = {l: np.mean(v, 0) for l, v in servo.items()}
    missing = sorted(set(r["stop"] for r in rows) - set(servo))
    if missing:
        sys.exit(f"--servo-from has no stops {missing}")

    out = [{"kind": "header", "names": names, "source": "spin_to_stops", "record": a.record, "shots": a.shots,
            "servo_from": a.servo_from}]
    for k, r in enumerate(rows):
        q_cmd = stop_q[r["stop"]].copy()
        q_cmd[wrist] += np.radians(r["wrist_offset_deg"])
        rel, fit = tracker.in_reference(frames[k + best])
        if not fit:
            continue
        cam = {n: {"pos_mm": (p.t * 1000).round(4).tolist(), "R": p.R.round(6).tolist()} for n, p in rel.items()}
        out.append({"kind": "stop", "cycle": 0, "label": f"{r['stop']}@{r['wrist_offset_deg']:+.0f}",
                    "commanded": q_cmd.tolist(), "encoder_mean": (q_cmd + servo[r["stop"]]).tolist(),
                    "camera": {"tags": cam, "ref_px_max": round(max(fit.per_tag_px.values()), 3)}})
    Path(a.out).write_text("\n".join(json.dumps(r) for r in out) + "\n")
    print(f"wrote {a.out}: {len(out) - 1} poses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
