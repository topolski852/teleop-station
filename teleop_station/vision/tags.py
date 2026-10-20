"""
AprilTag poses in the camera, and every tag expressed in the station's reference frame.

The station's tags are described by a ``TagSet`` (``config/tags.json``): the family, each
tag's name and measured size, and which tags are the **reference**: the tags that stay still
during a test (two on the table, two on the torso) and define the frame everything else is
reported in. The first reference tag is the anchor: the reference frame *is* its tag frame.
The others are tied to it by a layout surveyed at the start of each test (``survey``),
because the table and torso aren't fixed to each other and can shift during setup. At each
frame all visible reference tags are fitted together in one PnP. That matters: a rotation
error in the reference shows up at the arm tags multiplied by their distance from it, and
tags spread across the station pin the rotation far better than one 50 mm tag. It also keeps
the frame alive when a reference tag is blocked.

If a reference tag moves after the survey (the torso bumped on the table), the joint fit
can no longer match it: ``ReferenceFit.per_tag_px`` jumps from a fraction of a pixel to
several. Check it at every stop.

Tag size is the edge of the **black square**, measured on the print with calipers. A 2%
error in it scales every distance by 2%, so don't trust the size you asked the printer for.

A single tag's weakest axis is depth along the camera ray: depth comes from the tag's size
in pixels, so a 0.5 px corner error on a 125 px tag is ~0.4% of the distance. Bigger tags in
the image (closer camera, longer lens) help more than anything else. The noise part of that
averages out over many frames at each stop (``mean_pose``); a fixed bias doesn't, but it is
the same at every return to a pose, so it cancels in repeatability.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .calib import CameraCalibration

FAMILIES = {
    "16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


@dataclass
class Pose:
    """Rigid transform: maps points in the child frame into the parent frame."""
    R: np.ndarray            # 3x3
    t: np.ndarray            # (3,), metres

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R, self.t
        return T

    def inverse(self) -> "Pose":
        return Pose(self.R.T, -self.R.T @ self.t)

    def __matmul__(self, other: "Pose") -> "Pose":
        return Pose(self.R @ other.R, self.R @ other.t + self.t)

    def to_json(self) -> dict:
        return {"R": self.R.round(9).tolist(), "t": self.t.round(9).tolist()}

    @classmethod
    def from_json(cls, d: dict) -> "Pose":
        return cls(np.array(d["R"], float), np.array(d["t"], float))


# The station as built: right arm 0-2, left arm 3-5, torso 6-7, table 8-9. All 50 mm nominal.
_NAMES = {0: "right_hand", 1: "right_wrist", 2: "right_shoulder_roll",
          3: "left_hand", 4: "left_wrist", 5: "left_shoulder_roll",
          6: "torso_top", 7: "torso_front", 8: "table_far_left", 9: "table_near_right"}


@dataclass
class TagSet:
    """Which tag id is which, each one's measured black-square size, and the reference."""
    family: str = "25h9"
    names: dict[int, str] = field(default_factory=lambda: dict(_NAMES))
    sizes_m: dict[int, float] = field(default_factory=lambda: {i: 0.050 for i in _NAMES})
    reference: tuple[int, ...] = (8, 9, 6, 7)
    # Surveyed pose of each non-anchor reference tag in the anchor's frame (``survey``).
    # Per setup, so it is stored separately (scripts/track_tags.py --layout), not by ``save``.
    layout: dict[int, Pose] = field(default_factory=dict)
    note: str = ""
    # Rigid bodies: several tags on one part (the hand cube, a forearm's two faces), fitted
    # together as one target. The first id is the anchor: the body's frame is that tag's frame.
    # A body is reported under its own name in place of its tags'.
    bodies: dict[str, tuple[int, ...]] = field(default_factory=dict)
    # Surveyed pose of each tag of a body in the body's (anchor's) frame (``survey_body``).
    # Fixed by how the part was built, so it survives setups; stored in body_layout.json.
    body_layout: dict[str, dict[int, Pose]] = field(default_factory=dict)

    @property
    def anchor(self) -> int:
        return self.reference[0]

    def name(self, tag_id: int) -> str:
        return self.names.get(tag_id, f"tag{tag_id}")

    def dictionary(self) -> cv2.aruco.Dictionary:
        return cv2.aruco.getPredefinedDictionary(FAMILIES[self.family])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "family": self.family,
            "tags": {str(i): {"name": self.names[i], "size_m": self.sizes_m[i]} for i in sorted(self.names)},
            "reference": list(self.reference),
            "bodies": {k: list(v) for k, v in self.bodies.items()},
            "note": self.note,
        }, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "TagSet":
        d = json.loads(Path(path).read_text())
        tags = {int(i): v for i, v in d["tags"].items()}
        bodies = {k: tuple(v) for k, v in d.get("bodies", {}).items()}
        # A body's layout lives beside the tag file (body_layout.json), surveyed once per build.
        bl_path = Path(path).with_name("body_layout.json")
        body_layout = load_body_layout(bl_path) if bl_path.exists() else {}
        return cls(family=d["family"], names={i: v["name"] for i, v in tags.items()},
                   sizes_m={i: float(v["size_m"]) for i, v in tags.items()},
                   reference=tuple(d["reference"]), note=d.get("note", ""),
                   bodies=bodies, body_layout=body_layout)

    def body_of(self, tag_id: int) -> str | None:
        return next((b for b, ids in self.bodies.items() if tag_id in ids), None)


def load_body_layout(path) -> dict[str, dict[int, Pose]]:
    d = json.loads(Path(path).read_text())
    return {b: {int(i): Pose.from_json(p) for i, p in v.items()} for b, v in d["bodies"].items()}


def save_body_layout(path, layout: dict[str, dict[int, Pose]], info: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"bodies": {b: {str(i): p.to_json() for i, p in sorted(v.items())}
                                                for b, v in layout.items()}, "info": info or {}},
                                     indent=2) + "\n")


DEFAULT_TAGS = TagSet()


def _tag_object_points(size: float) -> np.ndarray:
    # Corner order the detector returns: TL, TR, BR, BL, tag frame x right, y up, z out.
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)


@dataclass
class Sighting:
    """One tag in one frame: its pose in the camera, its image corners, its size in pixels."""
    pose: Pose
    corners: np.ndarray      # 4x2
    side_px: float

    @property
    def view_deg(self) -> float:
        """How far off face-on the camera sees this tag (0 = straight at it, 90 = edge-on)."""
        ray = self.pose.t / np.linalg.norm(self.pose.t)
        return float(np.degrees(np.arccos(min(1.0, abs(float(self.pose.R[:, 2] @ ray))))))


@dataclass
class ReferenceFit:
    """The reference frame in the camera, fitted from every visible reference tag."""
    cam_from_ref: Pose
    tags_used: list[int]
    rms_px: float            # reprojection error of the joint fit
    per_tag_px: dict[int, float]   # each tag's share of it: one that moved stands out


class TagTracker:
    def __init__(self, calib: CameraCalibration, tags: TagSet = DEFAULT_TAGS):
        self.calib = calib
        self.tags = tags
        params = cv2.aruco.DetectorParameters()
        # AprilTag's edge-fit refinement: on synthetic renders it put tags 0.1-0.5 mm from the
        # truth, against 0.3-2.6 mm for CORNER_REFINE_SUBPIX, which reads small tags as smaller
        # (and so farther away) than they are. tests/test_vision.py holds it to that.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        # Read each bit from the middle of its cell with more samples. At ~7-10 px per cell
        # (50 mm tags on a 1280x960 webcam) the defaults let blur from the white margin into
        # the edge cells, and tags were found but failed to decode.
        params.perspectiveRemovePixelPerCell = 10
        params.perspectiveRemoveIgnoredMarginPerCell = 0.3
        self.detector = cv2.aruco.ArucoDetector(tags.dictionary(), params)

    def detect(self, frame: np.ndarray) -> dict[int, Sighting]:
        """Every known tag in the frame, with its pose in the camera."""
        self.calib.check_frame(frame)
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        out: dict[int, Sighting] = {}
        if ids is None:
            return out
        for c, tag_id in zip(corners, ids.ravel()):
            size = self.tags.sizes_m.get(int(tag_id))
            if size is None:
                continue                      # not one of ours
            img = c.reshape(4, 2).astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(_tag_object_points(size), img, self.calib.K,
                                          self.calib.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                side = float(np.mean(np.linalg.norm(img - np.roll(img, 1, axis=0), axis=1)))
                out[int(tag_id)] = Sighting(Pose(cv2.Rodrigues(rvec)[0], tvec.ravel()), img, side)
        return out

    def _ref_from_tag(self, tag_id: int) -> Pose | None:
        if tag_id == self.tags.anchor:
            return Pose(np.eye(3), np.zeros(3))
        return self.tags.layout.get(tag_id)

    def fit_reference(self, seen: dict[int, Sighting]) -> ReferenceFit | None:
        """The reference frame from all visible reference tags, or None if none is usable.

        A reference tag other than the anchor only counts once the layout has been surveyed.
        """
        return self._rigid_fit(seen, self.tags.reference, self._ref_from_tag)

    def fit_body(self, seen: dict[int, Sighting], body: str) -> ReferenceFit | None:
        """A body's frame (its anchor tag's) in the camera, from whichever of its tags are
        visible. Without a surveyed layout only the anchor tag itself can be used."""
        ids = self.tags.bodies[body]
        lay = self.tags.body_layout.get(body, {})
        def from_tag(i):
            return Pose(np.eye(3), np.zeros(3)) if i == ids[0] else lay.get(i)
        return self._rigid_fit(seen, ids, from_tag)

    def _rigid_fit(self, seen, ids, frame_from_tag) -> ReferenceFit | None:
        """One PnP over every corner of every visible tag of a rigid group."""
        usable = [i for i in ids if i in seen and frame_from_tag(i) is not None]
        if not usable:
            return None
        obj, img = [], []
        for i in usable:
            p = frame_from_tag(i)
            obj.append((p.R @ _tag_object_points(self.tags.sizes_m[i]).T).T + p.t)
            img.append(seen[i].corners)
        obj, img = np.vstack(obj).astype(np.float64), np.vstack(img).astype(np.float64)
        # Start from the best-seen tag's own pose, then refine on all corners together.
        first = max(usable, key=lambda i: seen[i].side_px)
        guess = seen[first].pose @ frame_from_tag(first).inverse()
        rvec, tvec = cv2.Rodrigues(guess.R)[0], guess.t.reshape(3, 1).copy()
        if len(usable) > 1:
            ok, rvec, tvec = cv2.solvePnP(obj, img, self.calib.K, self.calib.dist, rvec, tvec,
                                          useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return None
        proj, _ = cv2.projectPoints(obj, rvec, tvec, self.calib.K, self.calib.dist)
        sq = np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1).reshape(len(usable), 4)
        per_tag = {i: float(np.sqrt(sq[k].mean())) for k, i in enumerate(usable)}
        return ReferenceFit(Pose(cv2.Rodrigues(rvec)[0], tvec.ravel()), usable,
                            float(np.sqrt(sq.mean())), per_tag)

    def in_reference(self, frame: np.ndarray) -> tuple[dict[str, Pose], ReferenceFit | None]:
        """Every visible tag's pose in the reference frame, by name. Empty if no reference."""
        seen = self.detect(frame)
        return self.relative(seen)

    def relative(self, seen: dict[int, Sighting]) -> tuple[dict[str, Pose], ReferenceFit | None]:
        """Poses in the reference frame by name: every visible tag under its own name, and every
        body (fitted from all its visible tags) under the body's name. ``self.body_fits`` holds
        each body's fit (which tags it used, residuals) for the last call."""
        fit = self.fit_reference(seen)
        self.body_fits = {}
        if fit is None:
            return {}, None
        ref_inv = fit.cam_from_ref.inverse()
        out = {self.tags.name(i): ref_inv @ s.pose for i, s in seen.items()}
        for body in self.tags.bodies:
            bf = self.fit_body(seen, body)
            if bf is not None:
                self.body_fits[body] = bf
                out[body] = ref_inv @ bf.cam_from_ref
        return out, fit


def survey(tags: TagSet, sightings: list[dict[int, Sighting]], outlier_m: float = 0.005) -> dict[int, Pose]:
    """Each non-anchor reference tag's pose in the anchor frame, averaged over many frames.

    The layout is only as good as the anchor's own single-tag pose: its bias (mostly depth,
    worst for a tag lying flat and seen at a steep angle) becomes a fixed rigid offset of the
    whole reference frame. That offset is identical at every stop, so spreads and drift are
    unaffected; absolute accuracy will need a better-defined frame.

    Frames whose estimate lands more than ``outlier_m`` from the median are dropped: a small
    tag seen nearly head-on can flip to its mirror-ambiguous pose for a frame.
    """
    layout = {}
    for i in tags.reference[1:]:
        rel = [s[tags.anchor].pose.inverse() @ s[i].pose
               for s in sightings if tags.anchor in s and i in s]
        if not rel:
            continue
        med = np.median([p.t for p in rel], axis=0)
        keep = [p for p in rel if np.linalg.norm(p.t - med) < outlier_m]
        if keep:
            layout[i] = mean_pose(keep)[0]
    return layout


def survey_body(ids, sightings: list[dict[int, Sighting]], outlier_m: float = 0.005,
                min_pairs: int = 3, max_view_deg: float = 65.0, calib: CameraCalibration | None = None,
                sizes_m: dict[int, float] | None = None) -> tuple[dict[int, Pose], dict]:
    """Each tag of a rigid body in its anchor's (``ids[0]``) frame, from frames where tags of
    the body are seen together.

    Faces of a cube are only ever seen two or three at a time, so a face may never appear with
    the anchor (the one opposite it, for one). Every pair seen together gives a relative pose
    (averaged, outliers dropped as in ``survey``); the layout is then built outward from the
    anchor along the best-observed pairs. Returns (layout, info) where info says, per tag, how
    it was reached and from how many frames.

    A tag seen more than ``max_view_deg`` off face-on is left out: its own pose gets unreliable
    as it turns edge-on (a cube's end face seen at 70 deg put the left cube 5 mm too wide on
    2026-10-20, where the right cube's at 63 deg was fine).

    Given ``calib`` and ``sizes_m``, the chained layout is then refined on the image corners
    (``refine_body``). Chaining alone carries each tag's own pose error: a single tag's tilt
    from its four corners is biased by up to ~0.7 deg at these viewing angles, and on the left
    cube (2026-10-20) four side faces chained to 362.6 deg round the cube, putting the whole
    2.6 deg on the one pair never measured directly. Refined, both cubes are square to 0.4 deg.
    """
    anchor = ids[0]
    rel: dict[tuple[int, int], list[Pose]] = {}
    for s in sightings:
        vis = [i for i in ids if i in s and s[i].view_deg <= max_view_deg]
        for a in vis:
            for b in vis:
                if a != b:
                    rel.setdefault((a, b), []).append(s[a].pose.inverse() @ s[b].pose)
    pair: dict[tuple[int, int], tuple[Pose, int]] = {}
    for k, ps in rel.items():
        if len(ps) < min_pairs:
            continue
        med = np.median([p.t for p in ps], axis=0)
        keep = [p for p in ps if np.linalg.norm(p.t - med) < outlier_m]
        if len(keep) >= min_pairs:
            pair[k] = (mean_pose(keep)[0], len(keep))
    layout = {anchor: Pose(np.eye(3), np.zeros(3))}
    info = {anchor: {"via": None, "frames": None}}
    while True:
        best = None
        for (a, b), (p, n) in pair.items():
            if a in layout and b not in layout and (best is None or n > best[2]):
                best = (a, b, n, p)
        if best is None:
            break
        a, b, n, p = best
        layout[b] = layout[a] @ p
        info[b] = {"via": a, "frames": n}
    layout.pop(anchor)
    info.pop(anchor)
    if calib is not None and sizes_m is not None and layout:
        layout, fit = refine_body(ids, sightings, layout, calib, sizes_m, max_view_deg)
        info["refined"] = fit
    return layout, info


def refine_body(ids, sightings: list[dict[int, Sighting]], layout: dict[int, Pose],
                calib: CameraCalibration, sizes_m: dict[int, float], max_view_deg: float = 65.0,
                iters: int = 30) -> tuple[dict[int, Pose], dict]:
    """Bundle adjustment of a body layout: every placed tag's pose in the anchor frame and the
    body's pose in each frame together, minimising the corner reprojection error over every
    frame that shows two or more placed tags (Levenberg-Marquardt, numeric Jacobian; small
    problems, a few dozen frames). Returns the refined layout and {"rms_px", "tag_rms_px",
    "frames"}."""
    anchor = ids[0]
    placed = [i for i in ids[1:] if i in layout]
    frames = []
    for s in sightings:
        v = {i: s[i] for i in [anchor] + placed if i in s and s[i].view_deg <= max_view_deg}
        if len(v) >= 2:
            frames.append(v)
    if not frames:
        return layout, {"rms_px": None, "tag_rms_px": {}, "frames": 0}
    obj = {i: _tag_object_points(sizes_m[i]).astype(np.float64) for i in [anchor] + placed}
    K, D = calib.K, calib.dist
    vec = lambda p: np.r_[cv2.Rodrigues(p.R)[0].ravel(), p.t]
    pose = lambda x: Pose(cv2.Rodrigues(x[:3])[0], x[3:6].copy())

    def body_guess(v):
        i = max(v, key=lambda k: v[k].side_px)
        return v[i].pose @ (layout[i].inverse() if i != anchor else Pose(np.eye(3), np.zeros(3)))

    nl = 6 * len(placed)
    x = np.concatenate([vec(layout[i]) for i in placed] + [vec(body_guess(v)) for v in frames])

    def frame_res(x, f):
        lay = {anchor: Pose(np.eye(3), np.zeros(3)), **{i: pose(x[6 * k:6 * k + 6]) for k, i in enumerate(placed)}}
        b = x[nl + 6 * f:nl + 6 * f + 6]
        out = []
        for i, s in frames[f].items():
            P = obj[i] @ lay[i].R.T + lay[i].t
            img = cv2.projectPoints(P, b[:3], b[3:], K, D)[0].reshape(-1, 2)
            out.append((img - s.corners).ravel())
        return np.concatenate(out)

    def residuals(x):
        return [frame_res(x, f) for f in range(len(frames))]

    rows = np.cumsum([0] + [8 * len(v) for v in frames])
    r = residuals(x)
    lam, h = 1e-3, 1e-6
    for _ in range(iters):
        rv = np.concatenate(r)
        J = np.zeros((rv.size, x.size))
        for j in range(x.size):
            dx = np.zeros_like(x); dx[j] = h
            if j < nl:                   # a layout parameter moves every frame
                J[:, j] = (np.concatenate(residuals(x + dx)) - rv) / h
            else:                        # a body pose only its own frame
                f = (j - nl) // 6
                J[rows[f]:rows[f + 1], j] = (frame_res(x + dx, f) - r[f]) / h
        A, g = J.T @ J, J.T @ rv
        while lam < 1e8:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            r2 = residuals(x + step)
            if sum(e @ e for e in r2) < rv @ rv:
                x, r, lam = x + step, r2, lam / 3
                break
            lam *= 10
        else:
            break
        if np.abs(step).max() < 1e-9:
            break
    per: dict[int, list] = {}
    for v, e in zip(frames, r):
        for k, i in enumerate(v):
            per.setdefault(i, []).append(e[8 * k:8 * k + 8])
    rv = np.concatenate(r)
    fit = {"rms_px": float(np.sqrt(np.mean(rv ** 2))), "frames": len(frames),
           "tag_rms_px": {str(i): round(float(np.sqrt(np.mean(np.concatenate(e) ** 2))), 3) for i, e in sorted(per.items())}}
    return {i: pose(x[6 * k:6 * k + 6]) for k, i in enumerate(placed)}, fit


def mean_pose(poses: list[Pose]) -> tuple[Pose, np.ndarray]:
    """Average of several poses of the same tag, and the per-axis position std (metres).

    Rotation: mean of the matrices projected back onto SO(3), which is accurate for the
    small spreads of a stationary tag.
    """
    if not poses:
        raise ValueError("no poses to average")
    t = np.array([p.t for p in poses])
    U, _, Vt = np.linalg.svd(np.mean([p.R for p in poses], axis=0))
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    return Pose(R, t.mean(axis=0)), t.std(axis=0)


def draw(frame: np.ndarray, calib: CameraCalibration, tags: TagSet,
         seen: dict[int, Sighting]) -> np.ndarray:
    """Overlay outlines, axes and names on a copy of the frame."""
    out = frame.copy()
    for tag_id, s in seen.items():
        ref = tag_id in tags.reference
        cv2.polylines(out, [s.corners.astype(np.int32)], True, (255, 160, 0) if ref else (0, 255, 0), 2)
        cv2.drawFrameAxes(out, calib.K, calib.dist, cv2.Rodrigues(s.pose.R)[0], s.pose.t,
                          tags.sizes_m[tag_id] / 2)
        x, y = s.corners.min(axis=0).astype(int)
        cv2.putText(out, f"{tag_id} {tags.name(tag_id)}", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return out
