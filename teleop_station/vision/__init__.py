"""
Camera measurement for the station: AprilTag poses in a fixed camera, expressed in the frame
of a reference tag fixed to the station base.

The camera is a measuring instrument here, not a sensor the arms react to. It exists to
answer the accuracy test: where did each wrist actually stop, and how much does that spread
over repeated returns to the same taught pose. Everything is reported relative to the base
tag so a bumped tripod moves the camera, not the numbers.

- ``camera``: open a webcam, an HDMI capture card, a video file or a folder of stills.
- ``calib``: the camera's intrinsics (from a ChArUco board) and how they're stored.
- ``tags``: tag detection, per-tag pose, base-relative pose, and averaging over a stop.
"""
from .calib import CameraCalibration, charuco_board
from .tags import DEFAULT_TAGS, Pose, TagSet, TagTracker, mean_pose

__all__ = [
    "CameraCalibration", "charuco_board",
    "DEFAULT_TAGS", "Pose", "TagSet", "TagTracker", "mean_pose",
]
