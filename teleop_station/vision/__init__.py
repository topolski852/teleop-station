"""
Camera measurement for the station: AprilTag poses in a fixed camera, expressed in the frame
of reference tags fixed to the table.

The camera is a measuring instrument here, not a sensor the arms react to. It exists to
answer the accuracy test: where did each wrist actually stop, and how much does that spread
over repeated returns to the same taught pose. Everything is reported relative to the table
tags so a bumped tripod moves the camera, not the numbers.

- ``camera``: open a webcam, an HDMI capture card, a video file or a folder of stills.
- ``calib``: the camera's intrinsics (from a ChArUco board) and how they're stored.
- ``tags``: the station's tag set, detection, the reference-frame fit, and averaging over a stop.
"""
from .calib import CameraCalibration, charuco_board
from .tags import (DEFAULT_TAGS, Pose, ReferenceFit, Sighting, TagSet, TagTracker, load_body_layout,
                   mean_pose, refine_body, save_body_layout, survey, survey_body)

__all__ = [
    "CameraCalibration", "charuco_board",
    "DEFAULT_TAGS", "Pose", "ReferenceFit", "Sighting", "TagSet", "TagTracker", "load_body_layout",
    "mean_pose", "refine_body", "save_body_layout", "survey", "survey_body",
]
