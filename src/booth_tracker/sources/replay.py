from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from booth_tracker.types import Frame, Intrinsics


class ReplaySource:
    name = "replay"

    def __init__(self, directory: Path, loop: bool = True, realtime: bool = True):
        self._dir = Path(directory)
        self._loop = loop
        self._realtime = realtime
        meta = json.loads((self._dir / "meta.json").read_text())
        self.has_depth: bool = bool(meta["has_depth"])
        self._fps: float = float(meta.get("fps") or 30.0)
        self._intrinsics = Intrinsics(**meta["intrinsics"]) if meta.get("intrinsics") else None
        self._files = sorted((self._dir / "frames").glob("*.npz"))
        if not self._files:
            raise FileNotFoundError(f"{self._dir}/frames 內沒有 .npz")
        self._reset()

    def _reset(self) -> None:
        self._i = 0
        self._ts0: float | None = None  # 錄製的第一個 ts
        self._ts_offset = 0.0  # 每圈累加，讓輸出 ts 單調遞增
        self._last_ts = 0.0
        self._wall0 = 0.0

    def start(self) -> None:
        self._reset()

    def read(self) -> Frame | None:
        if self._i >= len(self._files):
            if not self._loop:
                return None
            self._ts_offset += self._last_ts - (self._ts0 or 0.0) + 1.0 / self._fps
            self._i = 0

        path = self._files[self._i]
        with np.load(path) as z:
            ts = float(z["ts"])
            rgb_bgr = cv2.imdecode(z["rgb_jpeg"], cv2.IMREAD_COLOR)
            depth_mm = z["depth_mm"] if "depth_mm" in z.files else None
        if rgb_bgr is None:
            raise ValueError(f"無法解碼 JPEG：{path}")
        self._i += 1

        if self._ts0 is None:
            self._ts0 = ts
            self._wall0 = time.monotonic()
        self._last_ts = ts
        out_ts = ts + self._ts_offset

        if self._realtime:
            delay = self._wall0 + (out_ts - self._ts0) - time.monotonic()
            if delay > 0:
                time.sleep(delay)

        depth_m = None
        if self.has_depth and depth_mm is not None:
            depth_m = depth_mm.astype(np.float32) / 1000.0
        return Frame(
            rgb=cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB),
            depth_m=depth_m,
            intrinsics=self._intrinsics,
            ts=out_ts,
        )

    def stop(self) -> None:
        pass
