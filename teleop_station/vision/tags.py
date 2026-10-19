"""
AprilTag (36h11) poses in the camera, and the wrist tags expressed in the base tag's frame.

Tag size is the edge of the **black square**, measured on the print with calipers. A 2%
error in it scales every distance by 2%, so don't trust the size you asked the printer for.

A single tag's weakest axis is depth along the camera ray: depth comes from the tag's size
in pixels, so a 0.5 px corner error on a 125 px tag is ~0.4% of the distance. Bigger tags in
the image (closer camera, longer lens) help more than anything else. The noise part of that
averages out over many frames at each stop (``mean_pose``); a fixed bias doesn't, but it is
the same at every return to a pose, so it cancels in repeatability.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .calib import CameraCalibration

TAG_DICT = cv2.aruco.DICT_APRILTAG_36h11


@dataclass(frozen=True)
class TagSet:
    """Which tag id is which, and each one's measured black-square size (metres)."""
    base: int = 0
    wrists: dict[str, int] = field(default_factory=lambda: {"left_wrist": 1, "right_wrist": 2})
    sizes_m: dict[int, float] = field(default_factory=lambda: {0: 0.060, 1: 0.030, 2: 0.030})

    def name(self, tag_id: int) -> str:
        if tag_id == self.base:
            return "base"
        return next((n for n, i in self.wrists.items() if i == tag_id), f"tag{tag_id}")


DEFAULT_TAGS = TagSet()


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


def _tag_object_points(size: float) -> np.ndarray:
    # Corner order the detector returns: TL, TR, BR, BL, tag frame x right, y up, z out.
    h = size / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)


class TagTracker:
    def __init__(self, calib: CameraCalibration, tags: TagSet = DEFAULT_TAGS):
        self.calib = calib
        self.tags = tags
        params = cv2.aruco.DetectorParameters()
        # AprilTag's edge-fit refinement: on synthetic renders it put tags 0.1-0.5 mm from the
        # truth, against 0.3-2.6 mm for CORNER_REFINE_SUBPIX, which reads small tags as smaller
        # (and so farther away) than they are. tests/test_vision.py holds it to that.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(TAG_DICT), params)

    def detect(self, frame: np.ndarray) -> dict[int, tuple[Pose, np.ndarray]]:
        """Pose of every known tag in the camera frame, plus its image corners (4x2)."""
        self.calib.check_frame(frame)
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        out: dict[int, tuple[Pose, np.ndarray]] = {}
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
                out[int(tag_id)] = (Pose(cv2.Rodrigues(rvec)[0], tvec.ravel()), img)
        return out

    def in_base(self, frame: np.ndarray) -> dict[str, Pose]:
        """Each visible wrist tag's pose in the base tag frame. Empty if the base isn't seen."""
        seen = self.detect(frame)
        if self.tags.base not in seen:
            return {}
        base_inv = seen[self.tags.base][0].inverse()
        return {name: base_inv @ seen[i][0] for name, i in self.tags.wrists.items() if i in seen}


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
         seen: dict[int, tuple[Pose, np.ndarray]]) -> np.ndarray:
    """Overlay outlines, axes and names on a copy of the frame."""
    out = frame.copy()
    for tag_id, (pose, img) in seen.items():
        cv2.polylines(out, [img.astype(np.int32)], True, (0, 255, 0), 2)
        size = tags.sizes_m[tag_id]
        cv2.drawFrameAxes(out, calib.K, calib.dist, cv2.Rodrigues(pose.R)[0], pose.t, size / 2)
        x, y = img[0].astype(int)
        cv2.putText(out, f"{tags.name(tag_id)} {pose.t[2]*1000:.0f}mm", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out
