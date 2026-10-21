#!/usr/bin/env python3
"""Everything a measured run says about one arm, as numbers for the baseline report.

    python scripts/report_baseline.py runs/rep_<time>.jsonl --arm left \\
        [--accuracy runs/accuracy_<time>.json] [--out runs/baseline_<time>.json]

Per measured stop (the hand cube, in the station reference frame):

- **RP**, ISO 9283 pose repeatability, position: l_mean + 3 S_l over the visits; orientation the
  same on each visit's angle from the mean orientation,
- **drift**: slope against time (mm/h) and the last quarter of the visits against the first,
- **encoder share**: the spread the encoders alone account for (each visit's encoder angles
  through the calibrated model, ``--accuracy``), against the camera's. The encoders are on the
  motor side of the gearbox, so what the camera sees beyond it is backlash, flex or play,
- **servo error**: encoder - command per joint (the sag the gains leave), and its spread,
- **settling**: time to settle and how often the 3 s timeout ran out.

Through the moves (the runner's track lines, 10 a second): per joint, the largest tracking
error, peak torque against its limit, the torque holding each stop (the last line before it),
and the lowest bus voltage.
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

import fit_accuracy as FA                                                  # noqa: E402

DAEMON_CONFIG = "/home/nse/humanoid-studio/configs/humanoid_arms.json"


def iso_rp(l):
    l = np.asarray(l, float)
    return float(l.mean() + 3 * l.std(ddof=1)) if len(l) > 1 else float("nan")


def short(n):
    return n.split("_", 1)[1].replace("_joint", "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--arm", default="left", choices=("left", "right"))
    ap.add_argument("--accuracy", help="fit_accuracy.py --out of this run (for the encoder share)")
    ap.add_argument("--config", default=DAEMON_CONFIG)
    ap.add_argument("--skip-cycles", type=int, default=0,
                    help="leave out the first N cycles' stops (e.g. 1: the first visit comes from the lead-in, "
                         "a joint may approach from the other side)")
    ap.add_argument("--out")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.run)]
    hdr = next(r for r in rows if r["kind"] == "header")
    names = hdr["names"]
    arm_idx = [k for k, n in enumerate(names) if n.startswith(a.arm + "_")]
    anames = [names[k] for k in arm_idx]
    stops = [r for r in rows if r["kind"] == "stop" and r["cycle"] >= a.skip_cycles]
    track = [r for r in rows if r["kind"] == "track"]
    events = [r["msg"] for r in rows if r["kind"] == "event"]
    labels = list(dict.fromkeys(s["label"] for s in stops))
    cycles = len({s["cycle"] for s in stops})
    minutes = (stops[-1]["t"] - stops[0]["t"]) / 60 if stops else 0
    out = {"run": a.run, "arm": a.arm, "cycles": cycles, "skipped_cycles": a.skip_cycles, "minutes": round(minutes, 1),
           "ended": next((e for e in reversed(events) if e.startswith("run ended")), None),
           "args": {k: hdr["args"].get(k) for k in ("path", "vmax", "amax", "shots", "measure", "limits")},
           "stops": {}, "joints": {}}
    print(f"{a.run}: {a.arm} arm, {cycles} cycles over {minutes:.0f} min, stops {labels}; {out['ended']}")

    model = x = None
    if a.accuracy:
        acc = json.loads(Path(a.accuracy).read_text())
        bodies = list(acc["body_on_link"])
        model = FA.Model(a.arm, bodies, offsets=True)
        x = np.r_[acc["base"], np.concatenate([acc["body_on_link"][b] for b in bodies]),
                  [acc["zero_offsets_deg"][FA.JOINTS[j]] for j in FA.FIT_OFFSETS]]
        x[model.k_off:] = np.radians(x[model.k_off:])
        out["accuracy"] = {"calibrated_loo_mm": np.mean([v["hand"][0] for v in acc["budget"]["calibrated (left-one-out)"].values()]),
                           "zero_offsets_mm": np.mean([v["hand"][0] for v in acc["budget"]["+ zero offsets (encoders, dq = 0)"].values()]),
                           "as_commanded_mm": np.mean([v["hand"][0] for v in acc["budget"]["+ servo error (as commanded)"].values()]),
                           "as_commanded_deg": np.mean([v["hand"][1] for v in acc["budget"]["+ servo error (as commanded)"].values()]),
                           "zero_offsets_deg": acc["zero_offsets_deg"]}

    print(f"\n{'stop':5s} {'RP mm':>7s} {'worst':>6s} {'RP deg':>7s} {'drift mm/h':>10s} {'1st-4th q':>9s}"
          f" {'enc RP mm':>9s} {'settle s':>9s} {'timeouts':>8s}   servo error deg (mean; spread)")
    for l in labels:
        v = [s for s in stops if s["label"] == l]
        t = np.array([s["t"] for s in v])
        cam = [(s["t"], s["camera"]["tags"][f"{a.arm}_hand"]) for s in v
               if s.get("camera") and f"{a.arm}_hand" in s["camera"]["tags"]]
        rec = {"visits": len(v)}
        if len(cam) > 1:
            tt = np.array([c[0] for c in cam])
            P = np.array([c[1]["pos_mm"] for c in cam])
            R = [np.array(c[1]["R"]) for c in cam]
            Rm = FA.mean_R(R)
            ang = [np.degrees(np.linalg.norm(FA.rvec(Rm.T @ Ri))) for Ri in R]
            q = max(1, len(P) // 4)
            rec.update({"rp_mm": iso_rp(np.linalg.norm(P - P.mean(0), axis=1)),
                        "worst_mm": float(np.linalg.norm(P - P.mean(0), axis=1).max()),
                        "std_xyz_mm": P.std(0, ddof=1).round(4).tolist(),
                        "rp_deg": iso_rp(ang),
                        "drift_mm_h": float(np.linalg.norm(np.polyfit((tt - tt[0]) / 3600, P, 1)[0])) if tt[-1] > tt[0] else None,
                        "quarter_shift_mm": float(np.linalg.norm(P[-q:].mean(0) - P[:q].mean(0))),
                        "seen": len(cam)})
        E = np.array([np.array(s["encoder_mean"])[arm_idx] for s in v])
        C = np.array(v[0]["commanded"])[arm_idx]
        err = np.degrees(E - C)
        rec["servo_error_deg"] = dict(zip(map(short, anames), err.mean(0).round(3).tolist()))
        rec["servo_spread_deg"] = dict(zip(map(short, anames), err.std(0, ddof=1).round(4).tolist())) if len(v) > 1 else {}
        if model is not None and len(v) > 1:
            Ph = np.array([model.predict(x, e)["hand"][:3, 3] * 1000 for e in E])
            rec["encoder_rp_mm"] = iso_rp(np.linalg.norm(Ph - Ph.mean(0), axis=1))
        st = [s["settle"]["t"] for s in v]
        rec["settle_s"] = {"median": float(np.median(st)), "max": float(np.max(st))}
        rec["settle_timeouts"] = sum(not s["settle"]["settled"] for s in v)
        out["stops"][l] = rec
        f = lambda k, fmt: format(rec[k], fmt) if rec.get(k) is not None else "-"
        print(f"{l:5s} {f('rp_mm', '7.3f')} {f('worst_mm', '6.3f')} {f('rp_deg', '7.3f')} {f('drift_mm_h', '10.2f')}"
              f" {f('quarter_shift_mm', '9.3f')} {f('encoder_rp_mm', '9.3f')} {rec['settle_s']['median']:5.1f}/{rec['settle_s']['max']:3.1f}"
              f" {rec['settle_timeouts']:8d}   " + " ".join(f"{short(n)[:6]} {m:+.2f};{rec['servo_spread_deg'].get(short(n), 0):.3f}"
                                                        for n, m in zip(anames, err.mean(0))))

    lim = {}
    try:
        cfg = json.loads(Path(a.config).read_text())["joints"]
        lim = {n: float(cfg[n]["torque_limit"]) for n in anames if "torque_limit" in cfg.get(n, {})}
    except Exception:
        pass
    if track:
        tt = np.array([r["t"] for r in track])
        Cm = np.array([r["cmd"] for r in track])[:, arm_idx]
        Ms = np.array([r["meas"] for r in track])[:, arm_idx]
        T = np.array([[np.nan if v is None else v for v in r["tau"]] for r in track], float)[:, arm_idx]
        V = np.array([[np.nan if v is None else v for v in r["volt"]] for r in track], float)[:, arm_idx]
        lag = np.degrees(np.abs(Ms - Cm))
        hold = {}
        for l in labels:
            ts = [s["t"] - s["settle"]["t"] for s in stops if s["label"] == l]
            k = [np.searchsorted(tt, t0) - 1 for t0 in ts]
            k = [i for i in k if i >= 0]
            hold[l] = np.nanmean(T[k], 0) if k else np.full(len(anames), np.nan)
        print(f"\n{'joint':16s} {'lag max':>8s} {'p99':>6s} {'peak Nm':>8s} {'of limit':>9s} {'rms Nm':>7s}  holding Nm per stop {labels}  {'min V':>6s}")
        for j, n in enumerate(anames):
            pk = float(np.nanmax(np.abs(T[:, j])))
            rec = {"lag_max_deg": float(lag[:, j].max()), "lag_p99_deg": float(np.percentile(lag[:, j], 99)),
                   "torque_peak_nm": pk, "torque_limit_nm": lim.get(n),
                   "torque_rms_nm": float(np.sqrt(np.nanmean(T[:, j] ** 2))),
                   "holding_nm": {l: float(hold[l][j]) for l in labels}, "volt_min": float(np.nanmin(V[:, j]))}
            out["joints"][short(n)] = rec
            pct = f"{100 * pk / lim[n]:.0f}%" if n in lim else "-"
            print(f"{short(n):16s} {rec['lag_max_deg']:8.2f} {rec['lag_p99_deg']:6.2f} {pk:8.2f} {pct:>9s} {rec['torque_rms_nm']:7.2f}  "
                  + " ".join(f"{hold[l][j]:+5.2f}" for l in labels) + f"  {rec['volt_min']:6.1f}")
    if "accuracy" in out:
        acc = out["accuracy"]
        print(f"\naccuracy (hand cube, mean over stops): calibrated {acc['calibrated_loo_mm']:.2f} mm (left-one-out), "
              f"with zero offsets {acc['zero_offsets_mm']:.2f} mm, as commanded {acc['as_commanded_mm']:.2f} mm / {acc['as_commanded_deg']:.2f} deg")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2, default=float) + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
