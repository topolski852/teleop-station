#!/usr/bin/env python3
"""Pose accuracy of one arm from a measured run: where the camera saw it vs where its model puts it.

    python scripts/fit_accuracy.py runs/rep_<time>.jsonl --arm left [--out runs/accuracy_<time>.json]

Per measured stop, every visit's camera reading of the arm's three bodies (the hand cube, the
forearm tag pair, the shoulder_roll tag) is averaged into one pose each (position and
orientation), and so are the commanded and the encoder angles. The model is

    body_in_ref = base @ FK(q + dq)[link] @ body_on_link

with FK from the URDF geometry in core. ``base`` (6) and where each body is mounted on its link
(6 each) aren't errors of the arm but unknowns of the set-up, and they must be fitted. Fitted
with ``dq`` held at zero they soak up most of the arm's errors (on a synthetic arm with 2 deg
zero offsets, a 5-stop fit, and even a 40-pose one, left about 1 mm of the several mm those
offsets cause). So they are fitted once, **with** the zero offsets of roll, yaw and elbow free
(pitch's is a turn of the base; the wrist's turns the cube on its own axis, the same as its
mounting), and then the accuracy is an error budget in that frame:

- **calibrated**: what's left after fitting the zero offsets, for each stop predicted from a fit
  on the *other* stops (leave-one-stop-out): link lengths, flex, backlash, camera,
- **zero offsets**: the same model with the offsets set back to zero, from the encoder angles:
  what the arm's calibrated zeros cost,
- **as commanded**: the same from the commanded angles, the ISO 9283 sense (the arm against its
  own model): adds the servo error (gravity sag, friction), the part the gains decide.

Not counted: a shoulder pitch zero error (it can't be told from where the base is, one arm at a
time), and any error that looks like the base or a mounting being somewhere else.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from humanoid_core.arm.kinematics import ArmChain  # noqa: E402

JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "wrist_yaw")
# Body -> index into ArmChain.frames(q): the link it rides on.
BODY_LINK = {"hand": 4, "wrist": 3, "shoulder_roll": 1}
FIT_OFFSETS = (1, 2, 3)
ROT_MM = 100.0        # a rotation residual of 1 rad weighs as 100 mm (a tag 100 mm out)


def mean_R(Rs):
    U, _, Vt = np.linalg.svd(np.mean(Rs, axis=0))
    return U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt


def rvec(R):
    return cv2.Rodrigues(np.asarray(R, float))[0].ravel()


def rmat(v):
    return cv2.Rodrigues(np.asarray(v, float).reshape(3, 1))[0]


def T_of(v):
    T = np.eye(4); T[:3, :3], T[:3, 3] = rmat(v[:3]), v[3:6]
    return T


def load_stops(path, side, session=0):
    """{label: {"q_cmd", "q_enc", "bodies": {body: (R, t_m)}, "n", "session"}} averaged over the
    visits. A label may carry a variant after '@' (``S3@+45``: S3 with the wrist turned 45 deg);
    it's the same stop for leave-one-out."""
    rows = [json.loads(l) for l in open(path)]
    hdr = next(r for r in rows if r["kind"] == "header")
    names = hdr["names"]
    idx = [names.index(f"{side}_{j}_joint") for j in JOINTS]
    out = {}
    for s in (r for r in rows if r["kind"] == "stop"):
        d = out.setdefault(s["label"], {"cmd": [], "enc": [], "bodies": {}})
        d["cmd"].append(np.array(s["commanded"])[idx]); d["enc"].append(np.array(s["encoder_mean"])[idx])
        for b in BODY_LINK:
            m = (s.get("camera") or {}).get("tags", {}).get(f"{side}_{b}")
            if m:
                d["bodies"].setdefault(b, []).append((np.array(m["R"]), np.array(m["pos_mm"]) / 1000))
    stops = {}
    for label, d in out.items():
        bodies = {b: (mean_R([R for R, _ in v]), np.mean([t for _, t in v], axis=0)) for b, v in d["bodies"].items()
                  if len(v) >= max(1, len(d["cmd"]) // 2)}
        stops[label] = {"q_cmd": np.mean(d["cmd"], 0), "q_enc": np.mean(d["enc"], 0), "bodies": bodies,
                        "n": len(d["cmd"]), "session": session, "stop": label.split("@")[0]}
    return stops, hdr


class Model:
    """x = base per session (6 each) | body_on_link (6 per body) | [zero offsets roll, yaw, elbow].

    A session is one camera set-up (a run, a survey): the reference frame is surveyed afresh
    each time and the torso can shift against the table between them, so each has its own base.
    """

    def __init__(self, side, bodies, offsets, sessions=1):
        self.chain = ArmChain([f"{side}_{j}_joint" for j in JOINTS])
        self.bodies, self.offsets, self.sessions = list(bodies), offsets, sessions
        self.k_mount = 6 * sessions
        self.k_off = self.k_mount + 6 * len(self.bodies)
        self.n = self.k_off + (len(FIT_OFFSETS) if offsets else 0)

    def dq(self, x):
        d = np.zeros(5)
        if self.offsets:
            d[list(FIT_OFFSETS)] = x[self.k_off:]
        return d

    def mount(self, x, k):
        return T_of(x[self.k_mount + 6 * k: self.k_mount + 6 * k + 6])

    def predict(self, x, q, session=0):
        F = self.chain.frames(self.chain.device_to_urdf(q) + self.dq(x))
        B = T_of(x[6 * session: 6 * session + 6])
        return {b: B @ F[BODY_LINK[b]] @ self.mount(x, k) for k, b in enumerate(self.bodies)}

    def residuals(self, x, stops, key):
        out = []
        for s in stops:
            P = self.predict(x, s[key], s["session"])
            for b, (R, t) in s["bodies"].items():
                if b in P:
                    out.append(np.r_[(P[b][:3, 3] - t) * 1000, rvec(P[b][:3, :3].T @ R) * ROT_MM])
        return np.concatenate(out)

    def guess(self, stops, key):
        """Per session, Kabsch on the body positions (each taken at its link's origin); then
        each body's mounting from the first stop that shows it."""
        x = np.zeros(self.n)
        for sess in range(self.sessions):
            A, Bp = [], []
            for s in stops:
                if s["session"] != sess:
                    continue
                F = self.chain.frames(self.chain.device_to_urdf(s[key]))
                for b, (_, t) in s["bodies"].items():
                    A.append(F[BODY_LINK[b]][:3, 3]); Bp.append(t)
            A, Bp = np.array(A), np.array(Bp)
            ca, cb = A.mean(0), Bp.mean(0)
            U, _, Vt = np.linalg.svd((A - ca).T @ (Bp - cb))
            R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
            x[6 * sess: 6 * sess + 3], x[6 * sess + 3: 6 * sess + 6] = rvec(R), cb - R @ ca
        for k, b in enumerate(self.bodies):
            s = next(s for s in stops if b in s["bodies"])
            F = self.chain.frames(self.chain.device_to_urdf(s[key]))
            M = np.linalg.inv(T_of(x[6 * s["session"]: 6 * s["session"] + 6]) @ F[BODY_LINK[b]])
            Rm, tm = s["bodies"][b]
            T = np.eye(4); T[:3, :3], T[:3, 3] = Rm, tm
            M = M @ T
            k0 = self.k_mount + 6 * k
            x[k0: k0 + 3], x[k0 + 3: k0 + 6] = rvec(M[:3, :3]), M[:3, 3]
        return x


def lm(f, x, iters=200):
    lam = 1e-3
    r = f(x)
    for _ in range(iters):
        J = np.empty((r.size, x.size))
        for k in range(x.size):
            h = 1e-7 * max(1.0, abs(x[k]))
            xp = x.copy(); xp[k] += h
            J[:, k] = (f(xp) - r) / h
        A, g = J.T @ J, J.T @ r
        while lam < 1e10:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            r2 = f(x + step)
            if r2 @ r2 < r @ r:
                x, r, lam = x + step, r2, lam / 3
                break
            lam *= 10
        else:
            break
        if np.abs(step).max() < 1e-12:
            break
    return x


def fit(side, stops, key, offsets, sessions=None):
    bodies = sorted({b for s in stops for b in s["bodies"]}, key=list(BODY_LINK).index)
    m = Model(side, bodies, offsets, sessions or 1 + max(s["session"] for s in stops))
    x = lm(lambda x: m.residuals(x, stops, key), m.guess(stops, key))
    return m, x


def errors(m, x, s, key):
    """{body: (position error mm, orientation error deg)} at one stop."""
    P = m.predict(x, s[key], s["session"])
    return {b: (float(np.linalg.norm(P[b][:3, 3] - t) * 1000), float(np.degrees(np.linalg.norm(rvec(P[b][:3, :3].T @ R)))))
            for b, (R, t) in s["bodies"].items() if b in P}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="the measured run: its stops are the ones reported")
    ap.add_argument("--also", action="append", default=[], metavar="JSONL",
                    help="more poses to fit on, each its own session (e.g. spin_to_stops.py output)")
    ap.add_argument("--arm", default="left", choices=("left", "right"))
    ap.add_argument("--out")
    a = ap.parse_args()
    stops, hdr = load_stops(a.run, a.arm)
    labels = [l for l in stops if stops[l]["bodies"]]
    visits = ", ".join(f"{l} ({stops[l]['n']} visits)" for l in labels)
    print(f"{a.run}: {a.arm} arm, stops {visits}")
    extra = []
    for k, f in enumerate(a.also):
        st, _ = load_stops(f, a.arm, session=k + 1)
        extra += [v for v in st.values() if v["bodies"]]
        print(f"  also {f}: {sum(1 for v in st.values() if v['bodies'])} poses at stops "
              f"{sorted({v['stop'] for v in st.values()})}")
    if len(labels) < 4:
        sys.exit("need at least 4 measured stops")
    servo = {l: np.degrees(stops[l]["q_enc"] - stops[l]["q_cmd"]).round(2).tolist() for l in labels}
    result = {"run": a.run, "also": a.also, "arm": a.arm, "stops": labels, "servo_error_deg": servo}
    all_stops = [stops[l] for l in labels] + extra
    m, x = fit(a.arm, all_stops, "q_enc", True)
    x0 = x.copy(); x0[m.k_off:] = 0                                   # offsets back to zero
    loo, loo_offs = {}, []
    for l in labels:
        # Leave out this stop in every session, all its wrist variants with it.
        mo, xo = fit(a.arm, [s for s in all_stops if s["stop"] != l], "q_enc", True, sessions=m.sessions)
        loo[l] = errors(mo, xo, stops[l], "q_enc")
        loo_offs.append(np.degrees(mo.dq(xo))[list(FIT_OFFSETS)])
    budget = {"calibrated (left-one-out)": loo,
              "calibrated (in-sample)": {l: errors(m, x, stops[l], "q_enc") for l in labels},
              "+ zero offsets (encoders, dq = 0)": {l: errors(m, x0, stops[l], "q_enc") for l in labels},
              "+ servo error (as commanded)": {l: errors(m, x0, stops[l], "q_cmd") for l in labels}}
    print(f"\n{'error, mm / deg':34s}" + "".join(f"{l:>14s}" for l in labels) + f"{'mean':>14s}")
    for b in m.bodies:
        print(f"  {b}")
        for name, e in budget.items():
            cells = [f"{e[l][b][0]:6.2f}/{e[l][b][1]:5.2f}" if b in e[l] else f"{'-':>12s}" for l in labels]
            vals = [e[l][b] for l in labels if b in e[l]]
            mean = f"{np.mean([v[0] for v in vals]):6.2f}/{np.mean([v[1] for v in vals]):5.2f}"
            print(f"    {name:32s}" + "".join(f"{c:>14s}" for c in cells) + f"{mean:>14s}")
    if extra:
        ex = [errors(m, x, s, "q_enc") for s in extra]
        h = [e["hand"][0] for e in ex if "hand" in e]
        print(f"  the --also poses, calibrated in-sample: hand {np.mean(h):.2f} mm mean, {np.max(h):.2f} worst ({len(h)} poses)")
        for sess in range(1, m.sessions):
            d = np.linalg.inv(T_of(x[:6])) @ T_of(x[6 * sess: 6 * sess + 6])
            print(f"  session {sess} base vs the run's: {np.linalg.norm(d[:3, 3]) * 1000:.2f} mm, "
                  f"{np.degrees(np.linalg.norm(rvec(d[:3, :3]))):.2f} deg")
    dq = np.degrees(m.dq(x))[list(FIT_OFFSETS)]
    spread = np.ptp(loo_offs, axis=0)
    print("\nzero offsets (camera says joint = encoder + offset, deg; spread over the left-one-out fits):")
    for k, j in enumerate(FIT_OFFSETS):
        print(f"   {JOINTS[j]:15s} {dq[k]:+6.2f}  (spread {spread[k]:.2f})")
    result.update({"budget": budget, "zero_offsets_deg": {JOINTS[j]: float(dq[k]) for k, j in enumerate(FIT_OFFSETS)},
                   "zero_offsets_loo_spread_deg": {JOINTS[j]: float(spread[k]) for k, j in enumerate(FIT_OFFSETS)},
                   "body_on_link": {b: x[m.k_mount + 6 * k: m.k_mount + 6 * k + 6].tolist() for k, b in enumerate(m.bodies)},
                   "base": x[:6].tolist()})
    print("\nservo error at each stop, encoder - command (deg; pitch, roll, yaw, elbow, wrist):")
    for l in labels:
        print(f"   {l:5s} {servo[l]}")
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
