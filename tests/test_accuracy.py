#!/usr/bin/env python3
"""Offline checks of scripts/fit_accuracy.py on a synthetic run. No robot, no camera::

    python tests/test_accuracy.py

An arm with known zero offsets (roll +1.0, yaw -2.0, elbow +1.5 deg) and a known gravity sag on
shoulder pitch is "measured" at the 5 stops of the left-arm baseline run, with camera noise
(0.1 mm, 0.05 deg). The calibrated fit must recover the offsets and predict a left-out stop
to well under a millimetre; the uncalibrated ones must show the offsets as error, and the
commanded one the sag on top.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fit_accuracy as F                                            # noqa: E402

PASS, FAIL = [], []
RNG = np.random.default_rng(3)


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")


def main() -> int:
    D = np.radians
    stops = {"S1": [61.3, -10.9, 7.0, 15.7, -13.8], "S2": [47.5, -4.8, 67.3, 68.2, -13.8],
             "S3": [61.3, -4.8, 93.0, 68.0, -13.8], "S7": [56.4, -5.8, 22.0, 30.2, -13.8],
             "S8": [55.1, -6.5, 75.2, 56.6, 9.5]}
    names = [f"left_{j}_joint" for j in F.JOINTS]
    dq_true = D(np.array([0, 1.0, -2.0, 1.5, 0]))
    sag = D(np.array([-2.0, -0.4, 0.1, -0.6, 0.0]))
    m = F.Model("left", list(F.BODY_LINK), offsets=False)
    x = np.zeros(m.n)
    x[:6] = [0.1, -0.2, 0.3, 0.05, -0.30, 0.20]                       # base
    x[6:12] = [0.2, 0.1, -0.3, 0.02, 0.01, -0.04]                     # cube on the wrist link
    x[12:18] = [-0.1, 0.3, 0.2, 0.0, 0.03, 0.02]                      # forearm pair
    x[18:24] = [0.3, 0.0, 0.1, 0.03, 0.0, 0.0]                        # shoulder tag
    rows = [{"kind": "header", "names": names}]
    for c in range(10):
        for label, q in stops.items():
            q_cmd = D(np.array(q))
            q_enc = q_cmd + sag
            P = m.predict(x, q_enc + m.chain.frame_sign * dq_true)    # dq is in the URDF frame
            tags = {}
            for b, T in P.items():
                Rn = F.rmat(RNG.normal(0, D(0.05), 3)) @ T[:3, :3]
                tags[f"left_{b}"] = {"R": Rn.tolist(), "pos_mm": ((T[:3, 3] + RNG.normal(0, 1e-4, 3)) * 1000).tolist()}
            rows.append({"kind": "stop", "cycle": c, "label": label, "commanded": q_cmd.tolist(),
                         "encoder_mean": q_enc.tolist(), "camera": {"tags": tags}})
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    st, _ = F.load_stops(f.name, "left")
    labels = list(st)
    mc, xc = F.fit("left", [st[l] for l in labels], "q_enc", True)
    got = np.degrees(mc.dq(xc))[list(F.FIT_OFFSETS)]
    want = np.degrees(dq_true)[list(F.FIT_OFFSETS)]
    check("calibrated fit recovers the zero offsets to 0.1 deg", np.abs(got - want).max() < 0.1,
          f"got {got.round(2)}, true {want.round(2)}")
    loo = []
    for l in labels:
        mo, xo = F.fit("left", [st[k] for k in labels if k != l], "q_enc", True)
        loo.append(F.errors(mo, xo, st[l], "q_enc")["hand"][0])
    check("calibrated: a left-out stop predicted within 1 mm (0.1 mm camera noise, 4 stops to fit)",
          np.mean(loo) < 1.0, f"{np.mean(loo):.2f} mm")
    # What the offsets and the sag really cost at the cube: the true model with them removed.
    x0 = xc.copy(); x0[mc.k_off:] = 0
    true_off = np.mean([np.linalg.norm(m.predict(x, st[l]["q_enc"])["hand"][:3, 3] - st[l]["bodies"]["hand"][1]) * 1000 for l in labels])
    true_all = np.mean([np.linalg.norm(m.predict(x, st[l]["q_cmd"])["hand"][:3, 3] - st[l]["bodies"]["hand"][1]) * 1000 for l in labels])
    est_off = np.mean([F.errors(mc, x0, st[l], "q_enc")["hand"][0] for l in labels])
    est_all = np.mean([F.errors(mc, x0, st[l], "q_cmd")["hand"][0] for l in labels])
    check("budget: the zero offsets' cost at the cube within 20% of the truth", abs(est_off - true_off) < 0.2 * true_off,
          f"{est_off:.2f} vs true {true_off:.2f} mm")
    check("budget: as commanded (offsets + sag) within 20% of the truth", abs(est_all - true_all) < 0.2 * true_all,
          f"{est_all:.2f} vs true {true_all:.2f} mm")
    Path(f.name).unlink()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
