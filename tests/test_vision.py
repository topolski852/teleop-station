#!/usr/bin/env python3
"""Offline check of the camera measurement chain against synthetic images. No camera.

    python tests/test_vision.py

Renders the ChArUco board and the station tags through a known pinhole camera, then checks
that calibration recovers that camera, that the reference layout surveys correctly, and that
the arm tags come back at the positions they were placed, in the reference frame. A failure here means the maths or the tag/board
conventions are wrong, before any real lens, print or lighting gets a chance to confuse it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import replace                                                               # noqa: E402

from teleop_station.vision import DEFAULT_TAGS, CameraCalibration, Pose, TagTracker, mean_pose, survey  # noqa: E402
from teleop_station.vision import calib as C                                                          # noqa: E402

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
    d = DEFAULT_TAGS.dictionary()
    cell = 40
    side = (d.markerSize + 2) * cell
    pad = 2 * cell
    m = cv2.aruco.generateImageMarker(d, tag_id, side, borderBits=1)
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

    print("\n── station tags in the reference frame ──────────────────────────")
    tags = DEFAULT_TAGS
    exact = CameraCalibration(K_TRUE, NO_DIST, (W, H), 0.0, 0)
    # Table tag 8 (the anchor) lying flat, camera looking down at ~45 deg from 0.6 m.
    cam_from_ref = Pose(rot(-2.3, 0, 0.1), np.array([0.0, 0.03, 0.60]))
    ref_from_cam = cam_from_ref.inverse()
    Kinv = np.linalg.inv(K_TRUE)

    def facing(u, v, depth, tilt):
        """A tag at pixel (u, v) and this depth, roughly facing the camera, in the ref frame."""
        return ref_from_cam @ Pose(rot(np.pi + tilt[0], tilt[1], tilt[2]), depth * Kinv @ [u, v, 1])

    def on_table(u, v, spin):
        """A tag lying on the table (tag 8's plane) where pixel (u, v) looks at it."""
        ray = ref_from_cam.R @ (Kinv @ [u, v, 1])
        o = ref_from_cam.t
        return Pose(rot(0, 0, spin), o - o[2] / ray[2] * ray)

    # Roughly the real view: torso at the back, right arm on the image left, left arm right.
    true = {9: on_table(700, 950, 0.3),
            6: facing(950, 150, 0.62, (0.3, 0, 0.05)), 7: facing(950, 330, 0.55, (0.1, 0, -0.05))}
    placed = {"right_hand": facing(300, 880, 0.48, (0.3, -0.2, 0.4)),
              "right_wrist": facing(320, 600, 0.52, (0.2, 0.3, 0.1)),
              "right_shoulder_roll": facing(380, 220, 0.58, (-0.2, 0.2, 0)),
              "left_hand": facing(1620, 880, 0.48, (0.2, 0.3, -0.3)),
              "left_wrist": facing(1600, 600, 0.52, (0.3, -0.2, 0)),
              "left_shoulder_roll": facing(1540, 220, 0.58, (0, -0.3, 0.2))}
    ids = {n: i for i, n in tags.names.items()}

    def scene_with(names, refs=(8, 9, 6, 7), shift=None):
        scene = np.full((H, W, 3), 150, np.uint8)
        for i in refs:
            p = Pose(np.eye(3), np.zeros(3)) if i == 8 else true[i]
            if shift and i in shift:
                p = Pose(p.R, p.t + shift[i])
            render_tag(scene, i, tags.sizes_m[i], cam_from_ref @ p)
        for n in names:
            render_tag(scene, ids[n], tags.sizes_m[ids[n]], cam_from_ref @ placed[n])
        return scene

    full = scene_with(placed)
    unsurveyed = TagTracker(exact, tags)
    seen = unsurveyed.detect(full)
    check("all 10 tags detected", set(seen) == set(range(10)), f"ids {sorted(seen)}")
    fit = unsurveyed.fit_reference(seen)
    check("unsurveyed: frame comes from the anchor alone", fit is not None and fit.tags_used == [8])

    layout = survey(tags, [seen] * 3)
    errs = {i: np.linalg.norm(layout[i].t - true[i].t) * 1000 if i in layout else np.inf for i in true}
    check("survey places tags 9, 6, 7 in tag 8's frame within 2 mm", max(errs.values()) < 2.0,
          ", ".join(f"{i}: {e:.2f} mm" for i, e in errs.items()))
    # The joint fit itself, given the true layout: tags land within 0.5 mm.
    rel_t, _ = TagTracker(exact, replace(tags, layout=dict(true))).in_reference(full)
    errs = [np.linalg.norm(rel_t[n].t - p.t) * 1000 if n in rel_t else np.inf for n, p in placed.items()]
    check("four-tag fit, true layout: all six arm tags within 0.5 mm", max(errs) < 0.5,
          f"worst {max(errs):.2f} mm")
    # With the surveyed layout the frame inherits the anchor's own pose error, a fixed rigid
    # offset (here ~1 mm from tag 8 seen at a steep angle). It's the same in every frame, so
    # it doesn't change any spread or drift; it matters only for absolute accuracy.
    tracker = TagTracker(exact, replace(tags, layout=layout))
    rel, fit = tracker.in_reference(full)
    check("surveyed: all four reference tags fitted together",
          fit is not None and sorted(fit.tags_used) == [6, 7, 8, 9],
          f"rms {fit.rms_px:.2f} px" if fit else "")
    errs = [np.linalg.norm(rel[n].t - p.t) * 1000 if n in rel else np.inf for n, p in placed.items()]
    check("surveyed layout: all six arm tags within 2 mm", max(errs) < 2.0, f"worst {max(errs):.2f} mm")

    rel_b, fit_b = tracker.in_reference(scene_with(["left_hand"], refs=(9, 6, 7)))
    err = np.linalg.norm(rel_b["left_hand"].t - placed["left_hand"].t) * 1000 if "left_hand" in rel_b else np.inf
    check("anchor blocked: frame held by 9, 6, 7; left_hand within 1.5 mm",
          fit_b is not None and 8 not in fit_b.tags_used and err < 1.5, f"{err:.2f} mm")
    check("no reference tag -> no reading (never camera-frame by mistake)",
          tracker.in_reference(scene_with(["left_hand"], refs=()))[0] == {})
    _, fit_m = tracker.in_reference(scene_with(["left_hand"], shift={7: np.array([0.005, 0, 0])}))
    worst = max(fit_m.per_tag_px, key=fit_m.per_tag_px.get) if fit_m else None
    still = max(fit.per_tag_px.values())
    check("torso tag moved 5 mm after the survey -> its fit residual stands out",
          worst == 7 and fit_m.per_tag_px[7] > 5 * still,
          f"moved {fit_m.per_tag_px[7]:.2f} px vs still {still:.2f} px" if fit_m else "")

    print("\n── averaging and safety checks ──────────────────────────────────")
    truth = placed["left_hand"]
    noisy = [Pose(rot(*RNG.normal(0, 0.002, 3)) @ truth.R,
                  truth.t + RNG.normal(0, 0.0002, 3)) for _ in range(200)]
    mean, std = mean_pose(noisy)
    check("mean of noisy poses lands on the truth", np.linalg.norm(mean.t - truth.t) < 5e-5,
          f"{np.linalg.norm(mean.t - truth.t) * 1e6:.1f} µm off")
    check("reported std matches injected noise", np.allclose(std, 0.0002, rtol=0.25),
          f"{(std * 1000).round(3)} mm")
    check("mean rotation is a proper rotation", abs(np.linalg.det(mean.R) - 1) < 1e-9)
    from teleop_station.vision import TagSet
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        tracker.tags.save(Path(d) / "tags.json")
        back = TagSet.load(Path(d) / "tags.json")
    check("tag set round-trips through JSON",
          back.names == tags.names and back.reference == tags.reference
          and back.sizes_m == tags.sizes_m and back.family == tags.family)
    try:
        tracker.detect(np.zeros((720, 1280, 3), np.uint8))
        check("frame at the wrong resolution is refused", False)
    except ValueError:
        check("frame at the wrong resolution is refused", True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
