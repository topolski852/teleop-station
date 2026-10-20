#!/usr/bin/env python3
"""Offline checks of rigid multi-tag bodies (the hand cubes), on synthetic renders. No camera::

    python tests/test_bodies.py

A 50 mm cube with a tag on five faces is rendered through a known pinhole camera in 45 random
orientations; about half show only one face, the rest two or three. The survey must recover where every face's tag
sits in the anchor face's frame, including the face opposite the anchor, which never appears
with it; then the body fit must put the cube where it was, from whichever faces are visible.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_vision as V                                                          # noqa: E402
from teleop_station.vision import CameraCalibration, Pose, TagTracker, survey_body   # noqa: E402

PASS, FAIL = [], []
RNG = np.random.default_rng(11)


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")


def cube_faces(edge=0.050, off=0.002):
    """Five faces of a cube, each tag's pose in the cube frame (z out of the face), with each
    tag nudged a little off-centre on its face, as a hand-cut print would be."""
    h = edge / 2
    R = lambda rx, ry, rz: V.rot(rx, ry, rz)
    faces = {10: Pose(np.eye(3), np.array([0, 0, h])),                         # +z (anchor)
             11: Pose(R(0, np.pi / 2, 0), np.array([h, 0, 0])),               # +x
             12: Pose(R(0, -np.pi / 2, 0), np.array([-h, 0, 0])),             # -x
             13: Pose(R(-np.pi / 2, 0, 0), np.array([0, h, 0])),              # +y
             14: Pose(R(np.pi, 0, 0), np.array([0, 0, -h]))}                  # -z, opposite the anchor
    for i, p in faces.items():
        if i != 10:
            p.t = p.t + p.R @ np.r_[RNG.uniform(-off, off, 2), 0]
    return faces


def main() -> int:
    tags = replace(V.DEFAULT_TAGS, names={**V.DEFAULT_TAGS.names, **{i: f"cube_{i}" for i in range(10, 15)}},
                   sizes_m={**V.DEFAULT_TAGS.sizes_m, **{i: 0.0367 for i in range(10, 15)}},
                   bodies={"hand": (10, 11, 12, 13, 14)})
    exact = CameraCalibration(V.K_TRUE, V.NO_DIST, (V.W, V.H), 0.0, 0)
    faces = cube_faces()
    f0 = faces[10]
    rel_true = {i: f0.inverse() @ p for i, p in faces.items()}
    tracker = TagTracker(exact, tags)

    def render(cam_from_cube):
        scene = np.full((V.H, V.W, 3), 150, np.uint8)
        # paint the faces that point at the camera, far ones first
        order = []
        for i, p in faces.items():
            T = cam_from_cube @ p
            facing = T.R[:, 2] @ (-T.t / np.linalg.norm(T.t))
            if facing > 0.25:
                order.append((T.t[2], i, T))
        for _, i, T in sorted(order, reverse=True):
            V.render_tag(scene, i, 0.0367, T)
        return scene, {i for _, i, _ in order}

    print("\n── surveying a 5-face cube from views of 2-3 faces ──────────────")
    sightings, truths = [], []
    for k in range(45):
        R = V.rot(*RNG.uniform(-np.pi, np.pi, 3))
        cam_from_cube = Pose(R, np.array([*RNG.uniform(-0.05, 0.05, 2), RNG.uniform(0.32, 0.42)]))
        scene, vis = render(cam_from_cube)
        seen = tracker.detect(scene)
        sightings.append(seen); truths.append((cam_from_cube, vis))
    together = sum(10 in s and 14 in s for s in sightings)
    check("the face opposite the anchor is never seen with it", together == 0, f"{together} frames")
    layout, info = survey_body((10, 11, 12, 13, 14), sightings, min_pairs=2)
    check("survey reaches every face", sorted(layout) == [11, 12, 13, 14], f"reached {sorted(layout)}; {info}")
    errs = {i: np.linalg.norm(layout[i].t - rel_true[i].t) * 1000 for i in layout}
    check("each face's tag located within 1 mm of where it was glued", max(errs.values()) < 1.0,
          ", ".join(f"{i}: {e:.2f}" for i, e in errs.items()) + " mm")
    check("the opposite face was reached through a neighbour", info.get(14, {}).get("via") not in (None, 10),
          f"via {info.get(14, {}).get('via')}")

    print("\n── fitting the cube from whichever faces are visible ────────────")
    t2 = replace(tags, body_layout={"hand": layout})
    tr2 = TagTracker(exact, t2)
    worst, no_anchor = 0.0, 0
    for k in range(10):
        R = V.rot(*RNG.uniform(-np.pi, np.pi, 3))
        cam_from_cube = Pose(R, np.array([*RNG.uniform(-0.05, 0.05, 2), RNG.uniform(0.32, 0.42)]))
        scene, vis = render(cam_from_cube)
        seen = tr2.detect(scene)
        bf = tr2.fit_body(seen, "hand")
        if bf is None:
            continue
        truth = cam_from_cube @ f0                       # the body frame is the anchor face's
        worst = max(worst, np.linalg.norm(bf.cam_from_ref.t - truth.t) * 1000)
        no_anchor += 10 not in bf.tags_used
    # A view showing one face is a single 37 mm tag: its depth is good to ~1 mm, not better.
    check("cube position within 1.5 mm from any visible faces", worst < 1.5, f"worst {worst:.2f} mm")
    check("the cube is still tracked when its anchor face is hidden", no_anchor > 0, f"{no_anchor} of 10 views")
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
