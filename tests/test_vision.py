#!/usr/bin/env python3
"""Offline check of the camera measurement chain against synthetic images. No camera.

    python tests/test_vision.py

Renders the ChArUco board and the station tags through a known pinhole camera, then checks
that calibration recovers that camera and that the wrist tags come back at the positions
they were placed, in the base tag's frame. A failure here means the maths or the tag/board
conventions are wrong, before any real lens, print or lighting gets a chance to confuse it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from teleop_station.vision import DEFAULT_TAGS, CameraCalibration, Pose, TagTracker, mean_pose  # noqa: E402
from teleop_station.vision import calib as C                                                  # noqa: E402
from teleop_station.vision.tags import TAG_DICT                                               # noqa: E402

PASS, FAIL = [], []
W, H = 1920, 1080
K_TRUE = np.array([[1400.0, 0, 955.0], [0, 1400.0, 545.0], [0, 0, 1]])
NO_DIST = np.zeros(5)
RNG = np.random.default_rng(7)


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")


def rot(rx: float, ry: float, rz: float) -> np.ndarray:
    return cv2.Rodrigues(np.array([rx, ry, rz], float))[0]


def paste(scene: np.ndarray, img: np.ndarray, H_img_to_scene: np.ndarray) -> None:
    """Warp a planar pattern into the scene (anti-aliased by supersampling)."""
    s = 4
    # Scene pixel x covers supersampled pixels s*x .. s*x+s-1, centred on s*x + (s-1)/2.
    S = np.array([[s, 0, (s - 1) / 2], [0, s, (s - 1) / 2], [0, 0, 1.0]])
    if scene.ndim == 3 and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    big = cv2.warpPerspective(img, S @ H_img_to_scene, (W * s, H * s), flags=cv2.INTER_LINEAR,
                              borderValue=0)
    mask = cv2.warpPerspective(np.full(img.shape, 255, np.uint8), S @ H_img_to_scene, (W * s, H * s),
                               flags=cv2.INTER_LINEAR, borderValue=0)
    big = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA).astype(float)
    m = cv2.resize(mask, (W, H), interpolation=cv2.INTER_AREA).astype(float) / 255
    scene[:] = (scene * (1 - m) + big * m).astype(np.uint8)


def plane_homography(pose: Pose, pts_img: np.ndarray, pts_plane: np.ndarray) -> np.ndarray:
    """Homography: pattern image pixels -> scene pixels, for a pattern lying on z=0."""
    proj, _ = cv2.projectPoints(np.c_[pts_plane, np.zeros(len(pts_plane))].astype(float),
                                cv2.Rodrigues(pose.R)[0], pose.t, K_TRUE, NO_DIST)
    return cv2.getPerspectiveTransform(pts_img.astype(np.float32), proj.reshape(-1, 2).astype(np.float32))


def render_tag(scene, tag_id: int, size: float, pose: Pose) -> None:
    cell = 40
    side = 8 * cell
    pad = 2 * cell
    m = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(TAG_DICT), tag_id, side, borderBits=1)
    img = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    a, b = pad - 0.5, pad + side - 0.5            # black-square outer edges, pixel-centre coords
    h = size / 2
    Hm = plane_homography(pose, np.array([[a, a], [b, a], [b, b], [a, b]]),
                          np.array([[-h, h], [h, h], [h, -h], [-h, -h]]))
    paste(scene, img, Hm)


def render_board(scene, board, pose: Pose, board_img: np.ndarray, A: np.ndarray) -> None:
    hh, ww = board_img.shape
    corners_img = np.array([[-0.5, -0.5], [ww - 0.5, -0.5], [ww - 0.5, hh - 0.5], [-0.5, hh - 0.5]])
    # A maps board-frame xy (metres) to board-image pixels; invert it for the image corners.
    Ainv = np.linalg.inv(np.vstack([A, [0, 0, 1]]))
    plane = (Ainv @ np.c_[corners_img, np.ones(4)].T).T[:, :2]
    paste(scene, board_img,
          plane_homography(pose, corners_img, plane))


def main() -> int:
    print("\n── camera calibration from synthetic board views ────────────────")
    square = 0.028
    board = C.charuco_board(square)
    bw, bh = C.BOARD_SQUARES
    board_img = board.generateImage((bw * 120, bh * 120), marginSize=0, borderBits=1)
    flat = C.detect_board(cv2.copyMakeBorder(board_img, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255), board)
    check("board detected in its own flat image", flat is not None)
    # Board frame -> board-image pixels, from that flat detection (fixes axis conventions).
    obj, img = flat
    A, *_ = np.linalg.lstsq(np.c_[obj[:, :2], np.ones(len(obj))], img - 60, rcond=None)
    A = A.T
    centre = np.array([bw * square / 2, bh * square / 2, 0])
    views = []
    for _ in range(18):
        # Board frame is x right, y down, like the image, so identity faces the camera unmirrored.
        R = rot(*RNG.uniform(-0.6, 0.6, 2), RNG.uniform(-0.3, 0.3))
        t = np.array([*RNG.uniform(-0.08, 0.08, 2), RNG.uniform(0.35, 0.55)]) - R @ centre
        scene = np.full((H, W), 140, np.uint8)
        render_board(scene, board, Pose(R, t), board_img, A)
        v = C.detect_board(scene, board)
        if v:
            views.append(v)
    check("board found in most synthetic views", len(views) >= 14, f"{len(views)}/18")
    cal = C.solve(views, (W, H), "synthetic")
    check("calibration RMS is sub-pixel", cal.rms_px < 0.5, f"{cal.rms_px:.3f} px")
    f_err = abs(cal.K[0, 0] - K_TRUE[0, 0]) / K_TRUE[0, 0]
    check("focal length recovered within 1%", f_err < 0.01, f"fx {cal.K[0,0]:.1f} vs {K_TRUE[0,0]:.0f}")
    c_err = np.hypot(cal.K[0, 2] - K_TRUE[0, 2], cal.K[1, 2] - K_TRUE[1, 2])
    check("principal point within 10 px", c_err < 10, f"{c_err:.1f} px")

    print("\n── wrist tags in the base frame ─────────────────────────────────")
    tags = DEFAULT_TAGS
    exact = CameraCalibration(K_TRUE, NO_DIST, (W, H), 0.0, 0)
    tracker = TagTracker(exact, tags)
    # Base tag lying on the table, camera looking down at ~45 deg from 0.45 m.
    cam_from_base = Pose(rot(-2.3, 0, 0.1), np.array([0.0, 0.03, 0.45]))
    placed = {"left_wrist": Pose(rot(0.2, -0.3, 0.5), np.array([0.09, 0.05, 0.12])),
              "right_wrist": Pose(rot(-0.1, 0.25, -0.4), np.array([-0.10, 0.06, 0.15]))}
    scene = np.full((H, W, 3), 150, np.uint8)
    render_tag(scene, tags.base, tags.sizes_m[tags.base], cam_from_base)
    for name, p in placed.items():
        i = tags.wrists[name]
        render_tag(scene, i, tags.sizes_m[i], cam_from_base @ p)
    seen = tracker.detect(scene)
    check("base and both wrist tags detected", set(seen) == {0, 1, 2}, f"ids {sorted(seen)}")
    rel = tracker.in_base(scene)
    for name, p in placed.items():
        if name in rel:
            err = np.linalg.norm(rel[name].t - p.t) * 1000
            check(f"{name} position in base frame within 1 mm", err < 1.0, f"{err:.2f} mm")

    no_base = np.full((H, W, 3), 150, np.uint8)
    render_tag(no_base, 1, tags.sizes_m[1], cam_from_base @ placed["left_wrist"])
    check("no base tag -> no base-frame reading (never camera-frame by mistake)",
          tracker.in_base(no_base) == {})

    print("\n── averaging and safety checks ──────────────────────────────────")
    noisy = [Pose(rot(*RNG.normal(0, 0.002, 3)) @ placed["left_wrist"].R,
                  placed["left_wrist"].t + RNG.normal(0, 0.0002, 3)) for _ in range(200)]
    mean, std = mean_pose(noisy)
    check("mean of noisy poses lands on the truth", np.linalg.norm(mean.t - placed["left_wrist"].t) < 5e-5,
          f"{np.linalg.norm(mean.t - placed['left_wrist'].t) * 1e6:.1f} µm off")
    check("reported std matches injected noise", np.allclose(std, 0.0002, rtol=0.25),
          f"{(std * 1000).round(3)} mm")
    check("mean rotation is a proper rotation", abs(np.linalg.det(mean.R) - 1) < 1e-9)
    try:
        tracker.detect(np.zeros((720, 1280, 3), np.uint8))
        check("frame at the wrong resolution is refused", False)
    except ValueError:
        check("frame at the wrong resolution is refused", True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
