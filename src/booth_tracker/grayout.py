"""偵測之前先用深度把站位區外的人塗灰：背後、側邊的路人進不了姿勢模型。

沒有深度的像素（黑衣、細手臂、髮絲、畫面邊緣）一律保留原色，否則會把玩家自己塗掉。
比站位區近的像素也保留：擋在前面的人要被看見，鎖定層才知道玩家被擋住。
"""

from __future__ import annotations

import numpy as np

from booth_tracker.geometry import CameraPose
from booth_tracker.lock import Zone
from booth_tracker.types import Intrinsics

GRAY = 128


class DepthGate:
    def __init__(self, pose: CameraPose, zones: list[Zone], margin_z: float = 1.0, margin_x: float = 0.8):
        self.pose = pose
        self.x_min = min(z.x - z.r_out for z in zones) - margin_x
        self.x_max = max(z.x + z.r_out for z in zones) + margin_x
        self.z_max = max(z.z + z.r_out for z in zones) + margin_z
        self._maps: tuple[tuple, np.ndarray, np.ndarray] | None = None

    def _forward_maps(self, k: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
        key = (k.width, k.height, k.fx, k.fy, k.cx, k.cy)
        if self._maps is None or self._maps[0] != key:
            self._maps = (key, *self.pose.forward_maps(k))
        return self._maps[1], self._maps[2]

    def outside(self, depth: np.ndarray, k: Intrinsics) -> np.ndarray:
        """HxW bool：有深度、而且在站位區範圍外（太遠或太偏）的像素。"""
        fwd, lat = self._forward_maps(k)
        wz, wx = depth * fwd, depth * lat
        return (depth > 0) & ((wz > self.z_max) | (wx < self.x_min) | (wx > self.x_max))

    def apply(self, rgb: np.ndarray, depth: np.ndarray | None, k: Intrinsics) -> np.ndarray:
        """回傳塗灰後的新影像；沒有深度就原樣回傳。"""
        if depth is None:
            return rgb
        out = rgb.copy()
        out[self.outside(depth, k)] = GRAY
        return out
