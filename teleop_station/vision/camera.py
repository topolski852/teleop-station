"""
Frame sources: a V4L2 device (webcam, or an HDMI capture card carrying the Sony's clean
output), a tethered still camera over gphoto2 (``gphoto2``: the Sony a7S III in PC Remote,
or the D5600), a video file, or a folder of stills.

For measurement the camera's automatic modes are the enemy: autofocus or auto-exposure
changing mid-run shifts the corners the pose comes from. ``open_device(lock=True)`` asks the
driver to turn off autofocus and auto white balance; drivers differ, so read ``describe``.
Exposure is locked only when you give one (``exposure``, in the driver's units: 100 us on
UVC webcams). Switching to manual without a value leaves whatever the driver last stored,
which on the Logitech C310 is far darker than what auto had chosen. Find a value with the
live view (``track_tags.py``: [ and ] step it), then pass it every run.
A capture card can't lock anything; set manual focus and exposure on the camera body.

A gphoto2 source takes one still per ``read()``: full sensor resolution, no compression
artefacts beyond JPEG, about 1 s per shot on the a7S III over USB 2. Everything is set on
the body: M mode, MF, fixed aperture, shutter, ISO and white balance, SteadyShot off (it
moves the sensor), and PC Remote's save size on Original (the default 2M is 1616x1080).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class GphotoCamera:
    """A tethered still camera with the slice of the ``cv2.VideoCapture`` interface the tools use.

    ``source`` is ``gphoto2`` or ``gphoto2:DIR``; with a DIR every shot is also kept there as
    the camera's JPEG, so a run can be reprocessed later.
    """

    def __init__(self, source: str = "gphoto2"):
        import gphoto2 as gp                      # only needed for this source
        self._gp = gp
        _, _, keep = source.partition(":")
        self.keep_dir = Path(keep) if keep else None
        if self.keep_dir:
            self.keep_dir.mkdir(parents=True, exist_ok=True)
        self.cam = gp.Camera()
        self.cam.init()
        self.size = (0, 0)
        self.shots = 0
        self.model = self.cam.get_summary().text.split("\n")[1:2]

    def _drain(self, ms: int = 200) -> None:
        # Sony bodies refuse the next capture while unread events are queued.
        while self.cam.wait_for_event(ms)[0] != self._gp.GP_EVENT_TIMEOUT:
            pass

    def setting(self, name: str) -> str:
        try:
            return str(self.cam.get_single_config(name).get_value())
        except self._gp.GPhoto2Error:
            return "?"

    def read(self) -> tuple[bool, np.ndarray | None]:
        for attempt in range(3):
            try:
                self._drain()
                path = self.cam.capture(self._gp.GP_CAPTURE_IMAGE)
                cf = self.cam.file_get(path.folder, path.name, self._gp.GP_FILE_TYPE_NORMAL)
                data = bytes(cf.get_data_and_size())     # copy before cf is freed
                break
            except self._gp.GPhoto2Error:
                if attempt == 2:
                    return False, None
                time.sleep(1.0)
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return False, None
        self.shots += 1
        if self.keep_dir:
            (self.keep_dir / f"{time.strftime('%Y%m%dT%H%M%S')}_{self.shots:05d}.jpg").write_bytes(data)
        self.size = (frame.shape[1], frame.shape[0])
        return True, frame

    def grab(self) -> bool:                       # nothing is buffered: every read is a new shot
        return True

    def get(self, prop: int) -> float:
        return {cv2.CAP_PROP_FRAME_WIDTH: self.size[0], cv2.CAP_PROP_FRAME_HEIGHT: self.size[1]}.get(prop, -1)

    def set(self, prop: int, value: float) -> bool:   # settings live on the camera body
        return False

    def describe(self) -> str:
        return (f"gphoto2 {' '.join(self.model)}: {self.size[0]}x{self.size[1]}, "
                f"{self.setting('shutterspeed')} f/{self.setting('f-number').lstrip('f/')} "
                f"ISO {self.setting('iso')}, WB {self.setting('whitebalance')}, "
                f"focus {self.setting('focusmode')}")

    def release(self) -> None:
        try:
            self.cam.exit()
        except self._gp.GPhoto2Error:
            pass


def open_device(source: str, width: int = 0, height: int = 0, lock: bool = True,
                exposure: float | None = None):
    """Open ``/dev/videoN``, ``N``, or ``gphoto2[:DIR]``. Raises if it doesn't deliver a frame.

    For gphoto2, ``width``/``height`` are checked against the shot, not set: the size is
    whatever the camera body produces.
    """
    if source.startswith("gphoto2"):
        cam = GphotoCamera(source)
        ok, frame = cam.read()
        if not ok:
            cam.release()
            raise RuntimeError("gphoto2 camera gave no image (M mode? MF? PC Remote on?)")
        if width and height and cam.size != (width, height):
            cam.release()
            raise RuntimeError(f"camera shoots {cam.size[0]}x{cam.size[1]}, expected {width}x{height} "
                               f"(PC Remote save size on Original? same lens / crop?)")
        return cam
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
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    if exposure is not None:
        set_exposure(cap, exposure)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"camera {source} opened but gave no frame")
    return cap


def set_exposure(cap: cv2.VideoCapture, exposure: float) -> None:
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)        # V4L2: 1 = manual, 3 = aperture priority
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)


def describe(cap) -> str:
    if isinstance(cap, GphotoCamera):
        return cap.describe()
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return (f"{w}x{h} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps, "
            f"autofocus={cap.get(cv2.CAP_PROP_AUTOFOCUS):.0f}, "
            f"auto_exposure={cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):.0f} (1 = manual), "
            f"exposure={cap.get(cv2.CAP_PROP_EXPOSURE):.0f}")


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
    elif source.startswith("gphoto2"):
        cap = open_device(source, width, height)
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
