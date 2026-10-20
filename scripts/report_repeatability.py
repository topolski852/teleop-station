#!/usr/bin/env python3
"""Report a repeatability run: spread at every pose, drift over the run, encoder vs camera.

    python scripts/report_repeatability.py runs/rep_<time>.jsonl [--map calib/arm_map.json]

Per pose and per arm tag, from the camera (the station reference frame):
- **RP**, pose repeatability as ISO 9283 defines it: l_mean + 3 * S_l, where l is each visit's
  distance from the mean position of all visits,
- the largest distance of any visit from that mean, and the per-axis standard deviation,
- **drift**: the slope of position against time (mm per hour) and how far the last quarter of
  the visits sits from the first quarter.

From the encoders: how far each visit's reading sits from the command (the servo error), and
the spread of the readings across visits (deg).

With ``--map``, also **accuracy**: each tag's camera position against where the encoder
angles put it through the fitted kinematics. Only meaningful if the zero offsets in the map
match this power cycle (re-zero with the camera first).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

ARM_TAGS = {"left": ("left_hand", "left_wrist", "left_shoulder_roll"),
            "right": ("right_hand", "right_wrist", "right_shoulder_roll")}


def iso_rp(P: np.ndarray) -> tuple[float, float]:
    """ISO 9283 pose repeatability and the worst single visit, mm."""
    l = np.linalg.norm(P - P.mean(0), axis=1)
    return float(l.mean() + 3 * l.std(ddof=1)) if len(l) > 1 else float("nan"), float(l.max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--map", help="fitted arm map, for accuracy")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.run)]
    hdr = next(r for r in rows if r["kind"] == "header")
    stops = [r for r in rows if r["kind"] == "stop"]
    ended = [r["msg"] for r in rows if r["kind"] == "event" and r["msg"].startswith("run ended")]
    names = hdr["names"]
    arms = sorted({n.split("_")[0] for n in names})
    cycles = sorted({s["cycle"] for s in stops})
    print(f"{a.run}: {len(stops)} stops, {len(cycles)} cycles, arms {arms}, "
          f"{'DRY RUN, ' if hdr['dry_run'] else ''}{ended[-1] if ended else 'no end record'}")
    if not stops:
        return 1
    hours = (stops[-1]["t"] - stops[0]["t"]) / 3600
    print(f"  {hours * 60:.0f} min, {hdr['args']['vmax']:.0f} deg/s, approach {hdr['args']['approach']:.0f} deg, "
          f"{hdr['args']['shots']} shots per stop")

    amap = None
    if a.map:
        from fit_map import LoadedMap
        amap = LoadedMap.load(a.map)

    for label in dict.fromkeys(s["label"] for s in stops):
        visits = [s for s in stops if s["label"] == label]
        print(f"\n== {label}: {len(visits)} visits")
        cmd = np.array(visits[0]["commanded"])
        E = np.array([v["encoder_mean"] for v in visits])
        err = np.degrees(E - cmd)
        for side in arms:
            idx = [k for k, n in enumerate(names) if n.startswith(side)]
            print(f"  {side} encoders: servo error (reading - command) mean |{np.abs(err[:, idx].mean(0)).max():.2f}| deg "
                  f"worst joint, spread across visits {np.degrees(E[:, idx].std(0)).max():.3f} deg worst joint")
        if visits[0]["camera"] is None:
            print("  (no camera)")
            continue
        t = np.array([v["t"] for v in visits])
        for side in arms:
            for tag in ARM_TAGS[side]:
                P = np.array([v["camera"]["tags"][tag]["pos_mm"] for v in visits if tag in v["camera"]["tags"]])
                tt = np.array([v["t"] for v in visits if tag in v["camera"]["tags"]])
                if len(P) < 2:
                    print(f"  {tag:20s} seen in {len(P)} visits")
                    continue
                rp, worst = iso_rp(P)
                q = max(1, len(P) // 4)
                shift = np.linalg.norm(P[-q:].mean(0) - P[:q].mean(0))
                slope = np.linalg.norm(np.polyfit((tt - tt[0]) / 3600, P, 1)[0]) if len(P) > 2 and tt[-1] > tt[0] else float("nan")
                line = (f"  {tag:20s} RP {rp:6.3f} mm  worst {worst:6.3f}  std xyz {P.std(0, ddof=1).round(3)}  "
                        f"drift {slope:6.2f} mm/h, last-vs-first quarter {shift:5.3f} mm  ({len(P)} visits)")
                if amap is not None:
                    side_names = amap.d["arms"][side]["joints"]
                    acc = []
                    for v in visits:
                        if tag in v["camera"]["tags"]:
                            qv = np.array([v["encoder_mean"][names.index(n)] for n in side_names])
                            _, tp = amap.points(side, qv)
                            acc.append(np.linalg.norm(np.array(v["camera"]["tags"][tag]["pos_mm"]) - tp[tag.split("_", 1)[1]] * 1000))
                    line += f"  accuracy {np.mean(acc):5.2f} mm"
                print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
