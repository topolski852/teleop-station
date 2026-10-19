"""
Camera intrinsics, solved from a ChArUco board and stored as JSON.

A calibration is only valid for the focus, zoom and resolution it was taken at. Lock all
three before calibrating (manual focus, no zoom change, fixed capture mode) and recalibrate
if any of them moves. The file records the image size, and loading it against a different
frame size fails rather than silently measuring wrong.

The board uses a 5x5 ArUco dictionary, not the 36h11 family the station tags use, so the
board's markers can never be mistaken for a wrist or base tag that happens to be in view.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Board geometry, in squares. Physical sizes come from measuring the print, not from here.
BOARD_SQUARES = (7, 5)
BOARD_DICT = cv2.aruco.DICT_5X5_100
# Marker side as a fraction of the square side (what make_targets.py prints).
MARKER_FRACTION = 0.75


def charuco_board(square_m: float) -> cv2.aruco.CharucoBoard:
    """The station's ChArUco board, at the square size measured on the print (metres)."""
    return cv2.aruco.CharucoBoard(
        BOARD_SQUARES, square_m, square_m * MARKER_FRACTION,
        cv2.aruco.getPredefinedDictionary(BOARD_DICT))


@dataclass
class CameraCalibration:
    K: np.ndarray            # 3x3 camera matrix
    dist: np.ndarray         # distortion coefficients, OpenCV order
    image_size: tuple[int, int]   # (width, height) it was solved at
    rms_px: float            # reprojection error of the solve
    n_views: int
    note: str = ""

    def check_frame(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if (w, h) != tuple(self.image_size):
            raise ValueError(
                f"frame is {w}x{h} but the calibration was solved at "
                f"{self.image_size[0]}x{self.image_size[1]}; recalibrate or fix the capture mode")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "K": self.K.tolist(), "dist": self.dist.ravel().tolist(),
            "image_size": list(self.image_size), "rms_px": self.rms_px,
            "n_views": self.n_views, "note": self.note,
        }, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "CameraCalibration":
        d = json.loads(Path(path).read_text())
        return cls(K=np.array(d["K"], float), dist=np.array(d["dist"], float),
                   image_size=tuple(d["image_size"]), rms_px=float(d["rms_px"]),
                   n_views=int(d["n_views"]), note=d.get("note", ""))


def detect_board(frame: np.ndarray, board: cv2.aruco.CharucoBoard):
    """ChArUco corners in one frame: (object_points Nx3, image_points Nx2), or None."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ids is None or len(ids) < 6:
        return None
    obj, img = board.matchImagePoints(corners, ids)
    if obj is None or len(obj) < 6:
        return None
    return obj.reshape(-1, 3).astype(np.float32), img.reshape(-1, 2).astype(np.float32)


def solve(views: list[tuple[np.ndarray, np.ndarray]], image_size: tuple[int, int],
          note: str = "") -> CameraCalibration:
    """Intrinsics from board views (each from ``detect_board``)."""
    if len(views) < 8:
        raise ValueError(f"need at least 8 board views, have {len(views)}")
    obj = [v[0] for v in views]
    img = [v[1] for v in views]
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, image_size, None, None)
    return CameraCalibration(K=K, dist=dist, image_size=tuple(image_size), rms_px=float(rms),
                             n_views=len(views), note=note)
