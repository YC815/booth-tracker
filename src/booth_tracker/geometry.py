"""相機座標 ↔ 世界座標，以及從深度圖自動找地面。

相機座標：x 右、y 下、z 前。世界座標：x 右、y 上（地面 = 0）、z 水平往前。
換場地不用手量相機高度和俯角：場地淨空時拍一張深度，fit_floor() 用 RANSAC 找出地面平面。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from booth_tracker.types import Intrinsics


def ray_map(k: Intrinsics) -> np.ndarray:
    """HxWx3：每個像素的射線方向，z 分量 = 1（乘上深度就是相機座標）。"""
    u = (np.arange(k.width) - k.cx) / k.fx
    v = (np.arange(k.height) - k.cy) / k.fy
    uu, vv = np.meshgrid(u, v)
    return np.stack([uu, vv, np.ones_like(uu)], axis=-1)


def unproject(u: float, v: float, z: float, k: Intrinsics) -> np.ndarray:
    return np.array([(u - k.cx) / k.fx * z, (v - k.cy) / k.fy * z, z])


@dataclass
class CameraPose:
    axes: np.ndarray  # 3x3，三列依序是世界 x、y、z 軸在相機座標裡的方向
    height: float  # 相機離地高度（公尺）
    inlier_ratio: float | None = None  # fit_floor 的地面點比例，診斷用

    @classmethod
    def from_up(cls, up: np.ndarray, height: float, inlier_ratio: float | None = None) -> CameraPose:
        """up：世界「上方」在相機座標裡的方向。世界 z = 相機光軸投影到地面。"""
        y = up / np.linalg.norm(up)
        z = np.array([0.0, 0.0, 1.0]) - y[2] * y
        z /= np.linalg.norm(z)
        x = np.cross(z, y)  # 相機 y 朝下、世界 y 朝上，左右手系相反
        return cls(axes=np.stack([x, y, z]), height=float(height), inlier_ratio=inlier_ratio)

    @classmethod
    def from_angles(cls, height: float, pitch_deg: float, roll_deg: float = 0.0) -> CameraPose:
        """pitch：往下俯為正。roll：繞光軸，從相機後方看順時針為正。"""
        p, r = math.radians(pitch_deg), math.radians(roll_deg)
        up = np.array([math.sin(r) * math.cos(p), -math.cos(r) * math.cos(p), -math.sin(p)])
        return cls.from_up(up, height)

    @property
    def pitch_deg(self) -> float:
        return math.degrees(math.asin(-self.axes[1][2]))

    @property
    def roll_deg(self) -> float:
        up = self.axes[1]
        return math.degrees(math.atan2(up[0], -up[1]))

    def to_world(self, pts: np.ndarray) -> np.ndarray:
        """Nx3 相機座標 → Nx3 世界座標。"""
        return pts @ self.axes.T + np.array([0.0, self.height, 0.0])

    def to_camera(self, pts: np.ndarray) -> np.ndarray:
        """Nx3 世界座標 → Nx3 相機座標。"""
        return (pts - np.array([0.0, self.height, 0.0])) @ self.axes

    def project(self, pts: np.ndarray, k: Intrinsics) -> np.ndarray:
        """Nx3 世界座標 → Nx2 像素 (u, v)。遊戲在畫面上畫站位區用。點要在相機前方。"""
        cam = self.to_camera(pts)
        return np.stack([cam[:, 0] / cam[:, 2] * k.fx + k.cx, cam[:, 1] / cam[:, 2] * k.fy + k.cy], axis=-1)

    def forward_maps(self, k: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
        """(fwd, lat)：HxW，深度乘上它們就是該像素的世界 z、世界 x。整張圖判斷遠近用。"""
        rays = ray_map(k)
        return rays @ self.axes[2], rays @ self.axes[0]


def fit_floor(
    depth: np.ndarray,
    k: Intrinsics,
    *,
    rng: np.random.Generator | None = None,
    max_tilt_deg: float = 35.0,
    threshold_m: float = 0.03,
    iterations: int = 300,
    min_inlier_ratio: float = 0.15,
    max_points: int = 6000,
) -> CameraPose | None:
    """從深度圖下半部找地面。找不到合理的地面（太斜、太少點、在相機上方）回 None。

    合理 = 法向量和「相機大致朝上」的方向夾角 < max_tilt_deg，而且相機在平面上方。
    牆面常常是最大的平面，靠這個條件排除。
    """
    rng = rng or np.random.default_rng()
    h = depth.shape[0]
    pts = (ray_map(k) * depth[..., None])[h // 2 :]
    d = depth[h // 2 :]
    pts = pts[(d > 0.3) & (d < 6.0)]
    if len(pts) < 100:
        return None
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]

    cos_max = math.cos(math.radians(max_tilt_deg))
    best: np.ndarray | None = None
    for _ in range(iterations):
        plane = _plane(pts[rng.choice(len(pts), 3, replace=False)], cos_max)
        if plane is None:
            continue
        inliers = np.abs(pts @ plane[0] + plane[1]) < threshold_m
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.mean() < min_inlier_ratio:
        return None

    refined = _refine(pts[best], cos_max)
    if refined is None:
        return None
    n, height = refined
    return CameraPose.from_up(n, height, inlier_ratio=float(best.mean()))


def _orient(n: np.ndarray, d: float, cos_max: float) -> tuple[np.ndarray, float] | None:
    """讓相機（原點）在平面正側：n·p + d = 0、d > 0 = 相機高度。不合理回 None。"""
    if d < 0:
        n, d = -n, -d
    if d < 0.2 or n @ np.array([0.0, -1.0, 0.0]) < cos_max:
        return None
    return n, d


def _plane(three: np.ndarray, cos_max: float) -> tuple[np.ndarray, float] | None:
    n = np.cross(three[1] - three[0], three[2] - three[0])
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return None
    n = n / norm
    return _orient(n, float(-n @ three[0]), cos_max)


def _refine(pts: np.ndarray, cos_max: float) -> tuple[np.ndarray, float] | None:
    """最小平方：法向量 = 去中心點雲最小奇異值的方向。"""
    c = pts.mean(axis=0)
    n = np.linalg.svd(pts - c, full_matrices=False)[2][-1]
    return _orient(n, float(-n @ c), cos_max)
