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
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class Frame:
    """相機一幀。相機座標：x 往右、y 往下、z 往前（公尺）。"""

    rgb: np.ndarray  # HxWx3 uint8，RGB 順序
    depth_m: np.ndarray | None  # HxW float32，已對齊到 rgb，0 = 無效
    intrinsics: Intrinsics | None
    ts: float  # 秒
    gravity: np.ndarray | None = None  # IMU 量到的重力方向（相機座標，單位向量），沒有就 None


@dataclass
class Detection:
    """姿勢模型對一個人的原始輸出（還沒加深度）。"""

    pixels: dict[str, tuple[float, float]]  # (u, v) 彩色影像像素
    visibility: dict[str, float]
    model_world: dict[str, np.ndarray]  # 模型估的 3D（以髖部中心為原點，x 右、y 下、z 遠離相機，公尺）
    mask: np.ndarray | None = None  # HxW float32 0..1


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


# idle   這一區沒有玩家，等人在區內舉手
# track  玩家這幀看得到，座標可信（可能在區外，看 in_zone）
# hold   玩家暫時看不到（被擋、偵測掉了）：凍結判定，player 是最後一次可信的姿勢
# paused hold 太久：提示「回到腳印上」，player 同 hold
Mode = Literal["idle", "track", "hold", "paused"]


@dataclass
class PlayerFrame:
    """PlayerLock.update() 每幀每一區一個。"""

    zone_id: int
    mode: Mode
    lock_id: int | None  # 每次舉手認領取新號（各區共用計數，不會撞號）；變了就代表換了人
    player: Person | None
    player_index: int | None  # 玩家在這幀 persons 裡的索引；只有 track 時有值
    in_zone: bool = False  # 鎖定玩家此刻在不在自己這一區
    occupied: bool = False  # 這一區現在有人站著（未鎖定的人或本區玩家），給「就位亮格」用
    claim_progress: float = 0.0  # 區內舉手進度 0..1
