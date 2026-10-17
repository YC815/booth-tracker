"""共用資料型別。

座標慣例（世界座標，公尺，以地面對齊）：x 往右、y 往上（地面 = 0）、z 從相機往前（離相機越遠越大）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# MediaPipe Pose 33 個 landmark，索引即 tuple 位置。
LANDMARKS: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

TORSO: tuple[str, ...] = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")


@dataclass
class Person:
    """一幀裡偵測到的一個人。"""

    world: dict[str, np.ndarray]  # 世界座標 (x, y, z)
    visibility: dict[str, float]  # 0..1
    has_depth: bool = True  # 軀幹深度是否真的量到（False = 估出來的，不可信）
    pixels: dict[str, tuple[float, float]] = field(default_factory=dict)  # (u, v) 彩色影像像素
    mask: np.ndarray | None = None  # HxW float32 0..1 人物分割

    def ground(self) -> np.ndarray:
        """軀幹中心投影到地面的 (x, z)。"""
        c = np.mean([self.world[j] for j in TORSO], axis=0)
        return np.array([c[0], c[2]])


# idle   沒有玩家，等人在站位區內舉手
# track  玩家這幀看得到，座標可信
# hold   玩家暫時不可用（被擋、偵測掉了、走出站位區）：凍結判定，player 是最後一次可信的姿勢
# paused hold 太久：提示「回到腳印上」，player 同 hold
Mode = Literal["idle", "track", "hold", "paused"]


@dataclass
class PlayerFrame:
    """PlayerLock.update() 每幀的輸出。"""

    mode: Mode
    lock_id: int | None  # 每次舉手認領 +1；變了就代表換了人
    player: Person | None
    player_index: int | None  # 玩家在這幀 persons 裡的索引；只有 track 時有值
    claim_progress: float = 0.0  # idle 時最接近完成的舉手進度 0..1，給畫面提示用
