"""包住任一 FrameSource：獨立 thread 一直讀、只留最新一幀。

在 pipeline thread 直接讀時，一步超過 33 ms，Orbbec SDK 的 frameset 佇列就會塞滿
（SDK log：Output frameset queue is full），之後拿到的都是佇列裡最舊的幀，延遲多 ~300 ms。
拆開後讀幀（wait / align / 解碼）和姿勢辨識也能並行。
"""

from __future__ import annotations

import threading
import time

from booth_tracker.perf import PERF
from booth_tracker.sources.base import FrameSource
from booth_tracker.types import Frame

WAIT_S = 1.0  # read() 等新幀的上限，和 FrameSource 契約一致
JOIN_S = 3.0  # stop 等讀幀 thread 的上限（Orbbec wait_for_frames 最多卡 1 s）


class LatestFrameSource:
    def __init__(self, source: FrameSource, wait_s: float = WAIT_S):
        self._source = source
        self.name = source.name
        self.has_depth = source.has_depth
        self._wait_s = wait_s
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._got = 0.0  # 最新幀進來的 time.monotonic()
        self._seq = 0
        self._taken = 0
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._source.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="booth-camera", daemon=True)
        self._thread.start()

    def read(self) -> Frame | None:
        """等比上次新的幀，最多 wait_s 秒；讀幀 thread 的例外在這裡丟出，讓 PipelineRunner 重建。"""
        with self._cond:
            self._cond.wait_for(self._ready, timeout=self._wait_s)
            if self._error is not None:
                err, self._error = self._error, None
                raise err
            if self._seq == self._taken:
                return None
            self._taken = self._seq
            frame, got = self._frame, self._got
        PERF.add("age", (time.monotonic() - got) * 1000)
        return frame

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(JOIN_S)
        self._source.stop()

    def _ready(self) -> bool:
        return self._seq > self._taken or self._error is not None or self._stop.is_set()

    def _run(self) -> None:
        while not self._stop.is_set():
            t = time.perf_counter()
            try:
                frame = self._source.read()
            except Exception as e:
                with self._cond:
                    self._error = e
                    self._cond.notify_all()
                return
            if frame is None:
                self._stop.wait(0.005)  # 不信任每個 source 斷線時都會自己退讓
                continue
            PERF.add("read", (time.perf_counter() - t) * 1000)
            PERF.count("camera")
            with self._cond:
                self._frame, self._got = frame, time.monotonic()
                self._seq += 1
                self._cond.notify_all()
