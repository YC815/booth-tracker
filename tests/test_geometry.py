from __future__ import annotations

import math

import numpy as np
import pytest

from booth_tracker.geometry import CameraPose, fit_floor, ray_map
from booth_tracker.types import Intrinsics

K = Intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)


def render_depth(pose: CameraPose, k: Intrinsics = K, wall_z: float | None = None) -> np.ndarray:
    """從相機姿勢算出一張只有地面（和可選的正前方牆面）的深度圖。"""
    rays = ray_map(k)  # HxWx3，z 分量 = 1
    world_rays = rays @ pose.axes.T  # 每條射線在世界座標的方向
    dy = world_rays[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(dy < -1e-6, -pose.height / dy, np.inf)
        t = t_floor
        if wall_z is not None:
            dz = world_rays[..., 2]
            t_wall = np.where(dz > 1e-6, wall_z / dz, np.inf)
            t = np.minimum(t_floor, t_wall)
    depth = np.where(np.isfinite(t) & (t < 8.0), t, 0.0)  # 射線 z 分量是 1，t 就是深度
    return depth.astype(np.float32)


def test_level_camera_maps_optical_axis_point_to_height():
    pose = CameraPose.from_angles(height=1.5, pitch_deg=0.0)
    w = pose.to_world(np.array([[0.0, 0.0, 3.0]]))[0]
    assert w == pytest.approx([0.0, 1.5, 3.0], abs=1e-9)


def test_pitched_down_camera_sees_floor_at_zero_height():
    pose = CameraPose.from_angles(height=1.5, pitch_deg=20.0)
    d = 1.5 / math.sin(math.radians(20))  # 光軸打到地面的距離
    w = pose.to_world(np.array([[0.0, 0.0, d]]))[0]
    assert w[1] == pytest.approx(0.0, abs=1e-9)
    assert w[2] == pytest.approx(1.5 / math.tan(math.radians(20)), abs=1e-9)


def test_camera_y_down_becomes_world_y_up():
    pose = CameraPose.from_angles(height=1.5, pitch_deg=0.0)
    w = pose.to_world(np.array([[0.2, 0.5, 2.0]]))[0]  # 相機座標往下 0.5 m
    assert w == pytest.approx([0.2, 1.0, 2.0], abs=1e-9)


@pytest.mark.parametrize("height,pitch,roll", [(1.2, 5.0, 0.0), (1.6, 15.0, 3.0), (1.0, 0.0, -2.0)])
def test_fit_floor_recovers_pose(height, pitch, roll):
    truth = CameraPose.from_angles(height=height, pitch_deg=pitch, roll_deg=roll)
    rng = np.random.default_rng(0)
    depth = render_depth(truth)
    depth[depth > 0] += rng.normal(0, 0.005, size=int((depth > 0).sum())).astype(np.float32)
    fit = fit_floor(depth, K, rng=rng)
    assert fit is not None
    assert fit.height == pytest.approx(height, abs=0.03)
    assert fit.pitch_deg == pytest.approx(pitch, abs=1.0)
    assert fit.roll_deg == pytest.approx(roll, abs=1.0)


def test_fit_floor_ignores_wall_in_front():
    truth = CameraPose.from_angles(height=1.4, pitch_deg=10.0)
    depth = render_depth(truth, wall_z=4.0)
    fit = fit_floor(depth, K, rng=np.random.default_rng(1))
    assert fit is not None
    assert fit.height == pytest.approx(1.4, abs=0.03)


def test_fit_floor_returns_none_without_floor():
    depth = np.full((K.height, K.width), 3.0, dtype=np.float32)  # 只有一面正對的牆
    assert fit_floor(depth, K, rng=np.random.default_rng(2)) is None


def test_fitted_pose_puts_floor_points_at_zero():
    truth = CameraPose.from_angles(height=1.3, pitch_deg=12.0, roll_deg=2.0)
    depth = render_depth(truth)
    fit = fit_floor(depth, K, rng=np.random.default_rng(3))
    pts = (ray_map(K) * depth[..., None])[depth > 0]
    assert np.abs(fit.to_world(pts)[:, 1]).max() < 0.02


def test_world_forward_factor_matches_full_transform():
    pose = CameraPose.from_angles(height=1.4, pitch_deg=10.0, roll_deg=1.0)
    depth = render_depth(pose)
    fwd, lat = pose.forward_maps(K)
    pts = ray_map(K) * depth[..., None]
    w = pose.to_world(pts.reshape(-1, 3)).reshape(K.height, K.width, 3)
    valid = depth > 0
    assert np.allclose((depth * fwd)[valid], w[..., 2][valid], atol=1e-6)
    assert np.allclose((depth * lat)[valid], w[..., 0][valid], atol=1e-6)
