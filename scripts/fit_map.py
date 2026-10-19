#!/usr/bin/env python3
"""Fit the camera-to-encoder map: arm kinematics against the tag positions the camera saw.

    python scripts/fit_map.py runs/map_20261019T182231.jsonl [more runs ...] \\
        --out calib/arm_map.json

Reads the ``frame`` lines of ``record_map.py`` runs, finds the moments each arm held still
(``--still-deg`` over ``--still-s``), averages each into one pose, and fits both arms at once:

- **one** base pose for the torso in the station reference frame (6). Both arms hang off the
  same torso; fitting a base per arm let each arm settle into a mirror-image solution,
- where each tag sits on its link (3 each): shoulder_roll tag on the roll link, wrist tag on
  the elbow link (it doesn't turn with the wrist), hand tag on the wrist link,
- a zero offset per arm for shoulder roll, shoulder yaw and elbow, and the right shoulder
  pitch's zero relative to the left's (a common pitch offset is a rotation of the base about
  the shoulder axis, so the base absorbs it; the wrist's zero needs tag orientation),
- optionally (``--coupling``) a pitch-to-roll coupling per arm: roll_true = roll + c * pitch.

It also tries both directions for each shoulder pitch (the encoder counting with or against
the URDF axis) and keeps the best. On the bench (2026-10-19) **both** came out reversed:
148 mm RMS as the URDF has them, 4.8 mm reversed.

Prediction: ``tag_in_ref = base @ FK(sign * q_encoder + offset)[link] @ tag_on_link``, with FK
from the URDF geometry in core. The residuals are what the encoders plus nominal geometry
can't explain: link-length errors, backlash, flex, camera error. The fitted zero offsets say
how far each joint's calibrated zero is from where the camera says it is.

``--out`` writes the model as JSON for ``scripts/view_map.py`` (``StationMap.load``).
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
# Which link frame each tag rides on: index into ArmChain.frames(q).
TAG_LINK = {"shoulder_roll": 1, "wrist": 3, "hand": 4}
# Joints whose zero offset is fitted. Not pitch (see above), and not the wrist: its zero only
# turns the hand tag about the wrist axis, which a tag-position fit can't tell from where the tag
# sits on the link. It needs the tag's orientation.
FIT_OFFSETS = (1, 2, 3)


def load_frames(paths):
    t, joints, tags = [], [], []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r["kind"] == "frame" and r.get("ref_px") is not None:
                t.append(r["t"]); joints.append(r["joints"]); tags.append(r["tags"])
    return np.array(t), joints, tags


def still_poses(side, t, joints, tags, still_deg, still_s, min_frames, outlier_mm):
    """One averaged pose per still moment of this arm: (q rad (5,), {tag: mm (3,)})."""
    names = [f"{side}_{j}_joint" for j in JOINTS]
    Q = np.array([[j[n] for n in names] for j in joints])
    k0 = 0
    still = np.zeros(len(t), bool)
    for k in range(len(t)):
        while t[k0] < t[k] - still_s:
            k0 += 1
        still[k] = t[k] - t[k0] >= 0.8 * still_s and np.degrees(np.ptp(Q[k0:k + 1], 0).max()) < still_deg
    poses = []
    for seg in np.split(np.arange(len(t)), np.where(np.diff(still.astype(int)) != 0)[0] + 1):
        if not still[seg[0]] or len(seg) < min_frames:
            continue
        obs = {}
        for tag in TAG_LINK:
            name = f"{side}_{tag}"
            P = np.array([tags[k][name] for k in seg if name in tags[k]])
            if len(P) < min_frames:
                continue
            P = P[np.linalg.norm(P - np.median(P, 0), axis=1) < outlier_mm]
            if len(P) >= min_frames:
                obs[tag] = (P.mean(0), P.std(0), len(P))
        if obs:
            poses.append((Q[seg].mean(0), obs, len(seg)))
    # Merge repeats of the same pose (the arm sitting at rest across several segments).
    merged = []
    for q, obs, n in sorted(poses, key=lambda p: -p[2]):
        if all(np.degrees(np.abs(q - m[0])).max() > 1.0 for m in merged):
            merged.append((q, obs, n))
    return merged


def rodrigues(r):
    return cv2.Rodrigues(np.asarray(r, float).reshape(3, 1))[0]


class ArmModel:
    def __init__(self, side):
        self.side = side
        self.chain = ArmChain([f"{side}_{j}_joint" for j in JOINTS])
        self.n = 6 + 3 * len(TAG_LINK) + len(FIT_OFFSETS)

    def unpack(self, x):
        R, t = rodrigues(x[:3]), x[3:6]
        tag_on_link = {tag: x[6 + 3 * k: 9 + 3 * k] for k, tag in enumerate(TAG_LINK)}
        dq = np.zeros(5)
        dq[list(FIT_OFFSETS)] = x[6 + 3 * len(TAG_LINK):]
        return R, t, tag_on_link, dq

    def predict(self, x, q):
        """Every tag's position in the reference frame (mm) for encoder angles q."""
        R, t, on_link, dq = self.unpack(x)
        F = self.chain.frames(self.chain.device_to_urdf(q) + dq)
        return {tag: (R @ (F[i][:3, :3] @ on_link[tag] + F[i][:3, 3]) + t) * 1000
                for tag, i in TAG_LINK.items()}

    def frames_batch(self, Q):
        """ArmChain.frames for many poses at once: (n_joints, N, 4, 4)."""
        c = self.chain
        N = len(Q)
        T = np.broadcast_to(np.eye(4), (N, 4, 4)).copy()
        out = []
        for i in range(c.n):
            step = np.eye(4)
            step[:3, :3], step[:3, 3] = c._R[i], c._xyz[i]
            T = T @ step
            x, y, z = c._axis[i]
            K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            th = Q[:, i][:, None, None]
            rot = np.broadcast_to(np.eye(4), (N, 4, 4)).copy()
            rot[:, :3, :3] = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
            T = T @ rot
            out.append(T)
        return np.stack(out)

    def _index(self, poses):
        key = id(poses)
        if getattr(self, "_idx_key", None) != key:
            self._idx_key = key
            self._Q = np.array([q for q, _, _ in poses])
            self._obs = [(k, tag, p) for k, (_, obs, _) in enumerate(poses) for tag, (p, _, _) in obs.items()]
            self._obs_pose = np.array([k for k, _, _ in self._obs])
            self._obs_link = np.array([TAG_LINK[tag] for _, tag, _ in self._obs])
            self._obs_tag = [tag for _, tag, _ in self._obs]
            self._obs_p = np.array([p for _, _, p in self._obs])

    def residuals(self, x, poses):
        self._index(poses)
        R, t, on_link, dq = self.unpack(x)
        F = self.frames_batch(self.chain.device_to_urdf(self._Q) + dq)      # (5, N, 4, 4)
        Fo = F[self._obs_link, self._obs_pose]                              # (M, 4, 4)
        off = np.array([on_link[tag] for tag in self._obs_tag])             # (M, 3)
        local = np.einsum("mij,mj->mi", Fo[:, :3, :3], off) + Fo[:, :3, 3]
        pred = (local @ R.T + t) * 1000
        return (pred - self._obs_p).ravel()


def initial_guess(model, poses):
    """Kabsch fit of the base, taking each tag to sit at its link's origin."""
    A, B = [], []
    for q, obs, _ in poses:
        F = model.chain.frames(model.chain.device_to_urdf(q))
        for tag, (p, _, _) in obs.items():
            A.append(F[TAG_LINK[tag]][:3, 3]); B.append(p / 1000)
    A, B = np.array(A), np.array(B)
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    x = np.zeros(model.n)
    x[:3] = cv2.Rodrigues(R)[0].ravel()
    x[3:6] = cb - R @ ca
    return x


def levenberg_marquardt(f, x, iters=200, huber_mm=3.0):
    """Robust (Huber, by reweighting) LM with a numeric Jacobian."""
    lam = 1e-2
    for _ in range(iters):
        r = f(x)
        w = np.where(np.abs(r) <= huber_mm, 1.0, huber_mm / np.maximum(np.abs(r), 1e-12))
        sw = np.sqrt(w)
        J = np.empty((len(r), len(x)))
        for k in range(len(x)):
            h = 1e-6 * max(1.0, abs(x[k]))
            xp = x.copy(); xp[k] += h
            J[:, k] = (f(xp) - r) / h
        Jw, rw = J * sw[:, None], r * sw
        A, g = Jw.T @ Jw, Jw.T @ rw
        cost = rw @ rw
        while True:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            r_new = f(x + step)
            w_new = np.where(np.abs(r_new) <= huber_mm, 1.0, huber_mm / np.maximum(np.abs(r_new), 1e-12))
            if (r_new * np.sqrt(w_new)) @ (r_new * np.sqrt(w_new)) < cost:
                x, lam = x + step, max(lam / 3, 1e-9)
                break
            lam *= 4
            if lam > 1e9:
                return x, J
        if np.abs(step).max() < 1e-10:
            break
    return x, J


def report(model, x, J, poses):
    R, t, on_link, dq = model.unpack(x)
    r = model.residuals(x, poses).reshape(-1, 3)
    labels = [tag for _, obs, _ in poses for tag in obs]
    print(f"\n=== {model.side} arm: {len(poses)} still poses, {len(r)} tag observations")
    print(f"  residual RMS {np.sqrt((r ** 2).sum(1).mean()):.2f} mm, median {np.median(np.linalg.norm(r, axis=1)):.2f} mm, "
          f"max {np.linalg.norm(r, axis=1).max():.2f} mm")
    for tag in TAG_LINK:
        rr = np.array([v for v, l in zip(r, labels) if l == tag])
        if len(rr):
            print(f"    {tag:14s} n={len(rr):3d}  RMS {np.sqrt((rr ** 2).sum(1).mean()):6.2f} mm  "
                  f"(x/y/z {np.sqrt((rr ** 2).mean(0)).round(2)})")
    # 1-sigma from the Jacobian, scaled by the residual (a rough guide, not a guarantee).
    dof = max(1, len(r) * 3 - model.n)
    cov = np.linalg.pinv(J.T @ J) * ((r ** 2).sum() / dof)
    sd = np.sqrt(np.diag(cov))
    off_sd = sd[6 + 3 * len(TAG_LINK):]
    print("  zero offsets (camera says joint is at encoder + offset):")
    for k, j in enumerate(FIT_OFFSETS):
        print(f"    {JOINTS[j]:15s} {np.degrees(dq[j]):+7.2f} deg  ± {np.degrees(off_sd[k]):.2f}")
    print("  tag positions on their links (mm):")
    for tag, p in on_link.items():
        print(f"    {tag:14s} {(p * 1000).round(1)}  ({np.linalg.norm(p) * 1000:.0f} mm from the joint)")
    print(f"  arm base in the reference frame: {(t * 1000).round(1)} mm")
    return {"residual_rms_mm": float(np.sqrt((r ** 2).sum(1).mean())),
            "zero_offsets_deg": {JOINTS[j]: float(np.degrees(dq[j])) for j in FIT_OFFSETS},
            "zero_offsets_sd_deg": {JOINTS[j]: float(np.degrees(off_sd[k])) for k, j in enumerate(FIT_OFFSETS)},
            "tag_on_link_mm": {tag: (p * 1000).tolist() for tag, p in on_link.items()},
            "base_R": R.tolist(), "base_t_mm": (t * 1000).tolist(), "n_poses": len(poses)}


SIDES = ("left", "right")
N_ARM = 9 + 3          # 3 tag points + roll/yaw/elbow zero


class StationMap:
    """Both arms on one base. ``x`` = base (6) | left (12) | right (12) | right pitch zero |
    [coupling left, right]."""

    def __init__(self, poses: dict, pitch_sign=(1.0, 1.0), roll_seed=(0.0, 0.0), coupling=False):
        self.poses = poses
        self.arms = {s: ArmModel(s) for s in SIDES}
        for s, sign in zip(SIDES, pitch_sign):
            self.arms[s].chain.frame_sign = np.array([sign, 1, 1, 1, 1.0])
        self.roll_seed = dict(zip(SIDES, roll_seed))
        self.coupling = coupling
        self.n = 6 + 2 * N_ARM + 1 + (2 if coupling else 0)

    def arm_params(self, x, side):
        k = 6 + (0 if side == "left" else N_ARM)
        on = {tag: x[k + 3 * j: k + 3 * j + 3] for j, tag in enumerate(TAG_LINK)}
        dq = np.zeros(5)
        dq[[1, 2, 3]] = x[k + 9: k + 12]
        dq[1] += self.roll_seed[side]
        if side == "right":
            dq[0] = x[6 + 2 * N_ARM]
        c = x[6 + 2 * N_ARM + 1 + SIDES.index(side)] if self.coupling else 0.0
        return on, dq, c

    def joint_angles(self, side, Q, x):
        """Encoder angles (N,5) -> the URDF-frame angles the fitted model uses."""
        arm = self.arms[side]
        on, dq, c = self.arm_params(x, side)
        q = arm.chain.device_to_urdf(np.atleast_2d(Q)).copy()
        q[:, 1] += c * q[:, 0]
        return q + dq

    def residuals(self, x):
        R, t = rodrigues(x[:3]), x[3:6]
        out = []
        for side in SIDES:
            arm = self.arms[side]
            arm._index(self.poses[side])
            on, _, _ = self.arm_params(x, side)
            F = arm.frames_batch(self.joint_angles(side, arm._Q, x))
            Fo = F[arm._obs_link, arm._obs_pose]
            off = np.array([on[g] for g in arm._obs_tag])
            local = np.einsum("mij,mj->mi", Fo[:, :3, :3], off) + Fo[:, :3, 3]
            out.append(((local @ R.T + t) * 1000 - arm._obs_p).ravel())
        return np.concatenate(out)

    def initial_guess(self):
        A, B = [], []
        for side in SIDES:
            arm = self.arms[side]
            for q, obs, _ in self.poses[side]:
                dq = np.zeros(5); dq[1] = self.roll_seed[side]
                F = arm.chain.frames(arm.chain.device_to_urdf(q) + dq)
                for tag, (p, _, _) in obs.items():
                    A.append(F[TAG_LINK[tag]][:3, 3]); B.append(p / 1000)
        A, B = np.array(A), np.array(B)
        ca, cb = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        x = np.zeros(self.n)
        x[:3] = cv2.Rodrigues(R)[0].ravel()
        x[3:6] = cb - R @ ca
        return x

    def to_json(self, x, J, r) -> dict:
        dof = max(1, r.size - self.n)
        sd = np.sqrt(np.diag(np.linalg.pinv(J.T @ J) * ((r ** 2).sum() / dof)))
        out = {"base": {"R": rodrigues(x[:3]).tolist(), "t_m": x[3:6].tolist()},
               "residual_rms_mm": float(np.sqrt((r.reshape(-1, 3) ** 2).sum(1).mean())), "arms": {}}
        for side in SIDES:
            on, dq, c = self.arm_params(x, side)
            k = 6 + (0 if side == "left" else N_ARM)
            sdq = np.zeros(5); sdq[[1, 2, 3]] = sd[k + 9: k + 12]
            if side == "right":
                sdq[0] = sd[6 + 2 * N_ARM]
            out["arms"][side] = {
                "joints": [f"{side}_{j}_joint" for j in JOINTS],
                "frame_sign": self.arms[side].chain.frame_sign.tolist(),
                "zero_offset_rad": dq.tolist(), "zero_offset_sd_rad": sdq.tolist(),
                "pitch_roll_coupling": float(c),
                "tag_on_link_m": {tag: p.tolist() for tag, p in on.items()},
                "tag_link": dict(TAG_LINK)}
        return out


class LoadedMap:
    """A fitted map from JSON: encoder angles -> tag and joint positions in the reference frame."""

    def __init__(self, d: dict):
        self.d = d
        self.R, self.t = np.array(d["base"]["R"]), np.array(d["base"]["t_m"])
        self.chains = {}
        for side, a in d["arms"].items():
            c = ArmChain(a["joints"])
            c.frame_sign = np.array(a["frame_sign"])
            self.chains[side] = c

    @classmethod
    def load(cls, path) -> "LoadedMap":
        return cls(json.loads(Path(path).read_text()))

    def angles(self, side, q_device, nominal=False):
        q = np.asarray(q_device, float)
        if nominal:                                  # the URDF as core has it: no signs, no offsets
            return q.copy()
        a = self.d["arms"][side]
        q = self.chains[side].device_to_urdf(q).copy()
        q[1] += a["pitch_roll_coupling"] * q[0]
        return q + np.array(a["zero_offset_rad"])

    def points(self, side, q_device, nominal=False):
        """(joint origins (6,3) incl. the wrist pivot, {tag: (3,)}) in the reference frame, m."""
        c = self.chains[side]
        q = self.angles(side, q_device, nominal)
        F = c.frames(q)
        joints = np.vstack([c.joint_origins(q), F[-1][:3, 3]])
        a = self.d["arms"][side]
        tags = {tag: F[i][:3, :3] @ np.array(a["tag_on_link_m"][tag]) + F[i][:3, 3]
                for tag, i in a["tag_link"].items()}
        tf = lambda p: self.R @ p + self.t
        return np.array([tf(p) for p in joints]), {k: tf(v) for k, v in tags.items()}


def fit_station(poses, coupling=False):
    """Try both pitch directions per arm and both roll starts; return the best fit."""
    import itertools
    best = None
    for signs in itertools.product((1.0, -1.0), repeat=2):
        for seeds in itertools.product((0.0, np.pi), repeat=2):
            m = StationMap(poses, signs, seeds, coupling)
            x, J = levenberg_marquardt(m.residuals, m.initial_guess(), iters=250)
            r = m.residuals(x)
            rms = float(np.sqrt((r.reshape(-1, 3) ** 2).sum(1).mean()))
            print(f"  pitch signs {tuple(int(v) for v in signs)}, roll start "
                  f"{tuple(int(np.degrees(v)) for v in seeds)}: RMS {rms:7.2f} mm")
            if best is None or rms < best[0]:
                best = (rms, m, x, J, r)
    return best


def report_station(m, x, r):
    k = 0
    for side in SIDES:
        n = sum(len(obs) for _, obs, _ in m.poses[side])
        rr = r.reshape(-1, 3)[k:k + n]; k += n
        labels = [tag for _, obs, _ in m.poses[side] for tag in obs]
        on, dq, c = m.arm_params(x, side)
        print(f"\n=== {side} arm: {len(m.poses[side])} still poses, {n} tag observations, "
              f"pitch sign {int(m.arms[side].chain.frame_sign[0]):+d}")
        print(f"  residual RMS {np.sqrt((rr ** 2).sum(1).mean()):.2f} mm, "
              f"max {np.linalg.norm(rr, axis=1).max():.2f} mm")
        for tag in TAG_LINK:
            v = np.array([e for e, l in zip(rr, labels) if l == tag])
            if len(v):
                print(f"    {tag:14s} n={len(v):3d}  RMS {np.sqrt((v ** 2).sum(1).mean()):6.2f} mm")
        names = ["pitch (vs left)" if side == "right" else None, "roll", "yaw", "elbow"]
        offs = [f"{nm} {((np.degrees(dq[j]) + 180) % 360) - 180:+.2f}" for j, nm in enumerate(names) if nm]
        print("  zero offsets (deg): " + ", ".join(offs) + (f"; coupling c {c:+.4f}" if m.coupling else ""))
        print("  tags from their joint (mm): " +
              ", ".join(f"{tag} {np.linalg.norm(p) * 1000:.0f}" for tag, p in on.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--still-deg", type=float, default=0.3)
    ap.add_argument("--still-s", type=float, default=1.0)
    ap.add_argument("--min-frames", type=int, default=5)
    ap.add_argument("--outlier-mm", type=float, default=3.0)
    ap.add_argument("--coupling", action="store_true", help="also fit pitch-to-roll coupling")
    ap.add_argument("--out", help="write the fitted map as JSON (for view_map.py)")
    a = ap.parse_args()

    t, joints, tags = load_frames(a.runs)
    poses = {side: still_poses(side, t, joints, tags, a.still_deg, a.still_s, a.min_frames, a.outlier_mm)
             for side in SIDES}
    for side in SIDES:
        if len(poses[side]) < 6:
            sys.exit(f"{side}: only {len(poses[side])} still poses, not enough to fit")
    print("fitting both arms on one base:")
    rms, m, x, J, r = fit_station(poses, coupling=a.coupling)
    report_station(m, x, r)
    if a.out:
        d = m.to_json(x, J, r)
        d["runs"] = a.runs
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(d, indent=2) + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
