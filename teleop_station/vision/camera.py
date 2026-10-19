"""
Frame sources: a V4L2 device (webcam, or an HDMI capture card carrying the Sony's clean
output), a video file, or a folder of stills (the D5600 via gphoto2).

For measurement the camera's automatic modes are the enemy: autofocus or auto-exposure
changing mid-run shifts the corners the pose comes from. ``open_device(lock=True)`` asks the
driver to turn them off and reports what actually stuck; drivers differ, so read the report.
A capture card can't lock anything; set manual focus and exposure on the camera body.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def open_device(source: str, width: int = 0, height: int = 0, lock: bool = True) -> cv2.VideoCapture:
    """Open ``/dev/videoN`` or ``N``. Raises if it doesn't deliver a frame."""
    dev = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {source}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # high res needs MJPG on USB2
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if lock:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)    # V4L2: 1 = manual, 3 = aperture priority
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"camera {source} opened but gave no frame")
    return cap


def describe(cap: cv2.VideoCapture) -> str:
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return (f"{w}x{h} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps, "
            f"autofocus={cap.get(cv2.CAP_PROP_AUTOFOCUS):.0f}, "
            f"auto_exposure={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):.0f}")


def latest(cap: cv2.VideoCapture) -> np.ndarray:
    """A fresh frame: drop whatever the driver buffered while we weren't reading."""
    cap.grab()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("camera stopped delivering frames")
    return frame


def frames(source: str, width: int = 0, height: int = 0, lock: bool = True) -> Iterator[np.ndarray]:
    """Every frame from a device, a video file, or a folder of images (sorted by name)."""
    p = Path(source)
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                img = cv2.imread(str(f))
                if img is not None:
                    yield img
        return
    if p.is_file():
        cap = cv2.VideoCapture(str(p))
    else:
        cap = open_device(source, width, height, lock)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()
