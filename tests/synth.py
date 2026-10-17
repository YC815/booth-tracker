"""合成相機畫面：已知的人站在已知位置，算出像素、深度、Detection。"""

from __future__ import annotations

import numpy as np

from booth_tracker.geometry import CameraPose
from booth_tracker.types import Detection, Frame, Intrinsics

K = Intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
POSE = CameraPose.from_angles(height=1.4, pitch_deg=8.0)

# 站姿：相對腳底中心的 (dx, y)，公尺
STANDING = {
    "nose": (0.0, 1.6),
    "left_shoulder": (-0.19, 1.4),
    "right_shoulder": (0.19, 1.4),
    "left_elbow": (-0.22, 1.1),
    "right_elbow": (0.22, 1.1),
    "left_wrist": (-0.24, 0.85),
    "right_wrist": (0.24, 0.85),
    "left_hip": (-0.1, 0.9),
    "right_hip": (0.1, 0.9),
    "left_knee": (-0.1, 0.5),
    "right_knee": (0.1, 0.5),
    "left_ankle": (-0.1, 0.08),
    "right_ankle": (0.1, 0.08),
}


def world_joints(x: float, z: float) -> dict[str, np.ndarray]:
    return {j: np.array([x + dx, y, z]) for j, (dx, y) in STANDING.items()}


def detection(x: float, z: float, pose: CameraPose = POSE, k: Intrinsics = K) -> Detection:
    world = world_joints(x, z)
    names = list(world)
    px = pose.project(np.array([world[n] for n in names]), k)
    pixels = {n: (float(u), float(v)) for n, (u, v) in zip(names, px)}
    hip = (world["left_hip"] + world["right_hip"]) / 2
    # 模型 3D：髖部原點、y 往下、z 遠離相機（這裡人正對相機，z 都是 0）
    model = {n: np.array([w[0] - hip[0], -(w[1] - hip[1]), 0.0]) for n, w in world.items()}
    return Detection(pixels=pixels, visibility={n: 1.0 for n in names}, model_world=model)


def depth_for(people: list[tuple[float, float]], pose: CameraPose = POSE, k: Intrinsics = K,
              background: float = 0.0, radius_px: int = 12) -> np.ndarray:
    """每個人每個關節附近畫一個圓，填真實深度；其餘 = background。"""
    depth = np.full((k.height, k.width), background, dtype=np.float32)
    vv, uu = np.mgrid[0 : k.height, 0 : k.width]
    for x, z in people:
        world = world_joints(x, z)
        for w in world.values():
            cam = pose.to_camera(w[None])[0]
            u, v = pose.project(w[None], k)[0]
            disk = (uu - u) ** 2 + (vv - v) ** 2 <= radius_px**2
            depth[disk] = cam[2]
    return depth


def frame(depth: np.ndarray | None, k: Intrinsics = K, ts: float = 0.0) -> Frame:
    return Frame(rgb=np.full((k.height, k.width, 3), 200, dtype=np.uint8), depth_m=depth, intrinsics=k, ts=ts)
