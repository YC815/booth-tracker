"""2D 關節 + 深度 → 世界座標的 Person。

每個關節取 7x7 鄰域的深度中位數（有 mask 時只取身上的像素），離軀幹深度太遠的當成量錯，
改用軀幹深度加上模型估的前後差。軀幹四點都量不到深度時改用肩寬估距離，並標 has_depth=False，
鎖定層預設不讓這種人參與（避免沒深度的電視人像被當成玩家）。
"""

from __future__ import annotations

import math

import numpy as np

from booth_tracker.geometry import CameraPose, unproject
from booth_tracker.types import TORSO, Detection, Frame, Intrinsics, Person

SHOULDER_WIDTH_M = 0.38
NEIGHBOURHOOD = 3  # 半徑 3 -> 7x7
MAX_DEPTH_DEVIATION_M = 0.6


def estimate_intrinsics(width: int, height: int, hfov_deg: float = 60.0) -> Intrinsics:
    fx = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return Intrinsics(fx=fx, fy=fx, cx=width / 2, cy=height / 2, width=width, height=height)


def _sample_depth(depth: np.ndarray, u: float, v: float, mask: np.ndarray | None) -> float | None:
    """7x7 鄰域有效深度的中位數。關節在輪廓邊緣時鄰域一半是背景，所以有 mask 就只取身上的像素；
    mask 內沒有有效值（細手臂常被 mask 漏掉）才退回整個鄰域。"""
    h, w = depth.shape
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < w and 0 <= vi < h):
        return None
    r = NEIGHBOURHOOD
    rows, cols = slice(max(vi - r, 0), min(vi + r + 1, h)), slice(max(ui - r, 0), min(ui + r + 1, w))
    patch = depth[rows, cols]
    valid = patch > 0
    if mask is not None:
        on_body = valid & (mask[rows, cols] > 0.5)
        if on_body.any():
            valid = on_body
    return float(np.median(patch[valid])) if valid.any() else None


def _with_depth(det: Detection, depth: np.ndarray, k: Intrinsics) -> dict[str, np.ndarray] | None:
    mask = det.mask if det.mask is not None and det.mask.shape == depth.shape else None
    raw = {j: _sample_depth(depth, *det.pixels[j], mask) for j in det.pixels}
    torso = [raw[j] for j in TORSO if raw[j] is not None]
    if not torso:
        return None
    torso_depth = float(np.median(torso))
    torso_model_z = float(np.mean([det.model_world[j][2] for j in TORSO]))
    out = {}
    for j, d in raw.items():
        if d is None or abs(d - torso_depth) > MAX_DEPTH_DEVIATION_M:
            d = torso_depth + (det.model_world[j][2] - torso_model_z)
        out[j] = unproject(*det.pixels[j], d, k)
    return out


def _without_depth(det: Detection, k: Intrinsics) -> dict[str, np.ndarray]:
    """肩寬估距離，髖部中心像素反投影到那個距離，其他關節照模型估的相對位置擺。"""
    (ul, vl), (ur, vr) = det.pixels["left_shoulder"], det.pixels["right_shoulder"]
    d = k.fx * SHOULDER_WIDTH_M / max(math.hypot(ul - ur, vl - vr), 1.0)
    (hlu, hlv), (hru, hrv) = det.pixels["left_hip"], det.pixels["right_hip"]
    hip_cam = unproject((hlu + hru) / 2, (hlv + hrv) / 2, d, k)
    hip_model = (det.model_world["left_hip"] + det.model_world["right_hip"]) / 2
    # 模型 3D 是 x 右、y 下、z 遠離相機，和相機座標同向，直接平移
    return {j: hip_cam + (p - hip_model) for j, p in det.model_world.items()}


def lift(det: Detection, frame: Frame, pose: CameraPose) -> Person:
    h, w = frame.rgb.shape[:2]
    k = frame.intrinsics or estimate_intrinsics(w, h)
    cam = _with_depth(det, frame.depth_m, k) if frame.depth_m is not None else None
    has_depth = cam is not None
    if cam is None:
        cam = _without_depth(det, k)
    names = list(cam)
    world = pose.to_world(np.array([cam[n] for n in names]))
    return Person(
        world=dict(zip(names, world)),
        visibility=dict(det.visibility),
        has_depth=has_depth,
        pixels=det.pixels,
        mask=det.mask,
    )
