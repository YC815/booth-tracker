from __future__ import annotations

import math
import time

import cv2

from booth_tracker.types import Frame, Intrinsics

HFOV_DEG = 60.0
REOPEN_INTERVAL_S = 2.0


class WebcamSource:
    name = "webcam"
    has_depth = False

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self._index = index
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None
        self._last_open_attempt = -REOPEN_INTERVAL_S
        self._intrinsics: Intrinsics | None = None

    def _open(self) -> None:
        self._last_open_attempt = time.monotonic()
        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            cap.release()
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap = cap

    def start(self) -> None:
        self._open()

    def read(self) -> Frame | None:
        if self._cap is None:
            if time.monotonic() - self._last_open_attempt < REOPEN_INTERVAL_S:
                time.sleep(0.05)
                return None
            self._open()
            if self._cap is None:
                return None

        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            self._cap.release()
            self._cap = None
            return None

        h, w = bgr.shape[:2]
        if self._intrinsics is None or (self._intrinsics.width, self._intrinsics.height) != (w, h):
            fx = (w / 2) / math.tan(math.radians(HFOV_DEG) / 2)
            self._intrinsics = Intrinsics(fx=fx, fy=fx, cx=w / 2, cy=h / 2, width=w, height=h)
        return Frame(
            rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            depth_m=None,
            intrinsics=self._intrinsics,
            ts=time.monotonic(),
        )

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
