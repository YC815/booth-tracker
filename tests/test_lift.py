from __future__ import annotations

import numpy as np
import pytest

from booth_tracker.geometry import CameraPose
from booth_tracker.grayout import DepthGate
from booth_tracker.lift import lift
from booth_tracker.lock import Zone
from booth_tracker.plausible import plausible
from tests.synth import K, POSE, depth_for, detection, frame, world_joints


def test_project_and_to_camera_roundtrip():
    w = np.array([[0.3, 1.2, 2.5]])
    cam = POSE.to_camera(w)
    assert POSE.to_world(cam) == pytest.approx(w)
    u, v = POSE.project(w, K)[0]
    assert 0 <= u < K.width and 0 <= v < K.height


# ---- lift ----


def test_lift_with_depth_recovers_world_joints():
    det = detection(0.3, 2.5)
    person = lift(det, frame(depth_for([(0.3, 2.5)], background=6.0)), POSE)
    assert person.has_depth
    truth = world_joints(0.3, 2.5)
    for j in ("nose", "left_shoulder", "right_hip", "left_ankle"):
        assert person.world[j] == pytest.approx(truth[j], abs=0.03)
    assert person.ground() == pytest.approx([0.3, 2.5], abs=0.03)


def test_lift_rejects_joint_depth_far_from_torso():
    """手腕那一圈深度被背景蓋掉（例如手伸出輪廓外），不能拿背景深度。"""
    det = detection(0.0, 2.5)
    depth = depth_for([(0.0, 2.5)], background=0.0)
    u, v = det.pixels["right_wrist"]
    depth[int(v) - 13 : int(v) + 14, int(u) - 13 : int(u) + 14] = 5.5
    person = lift(det, frame(depth), POSE)
    assert person.world["right_wrist"][2] == pytest.approx(2.5, abs=0.1)


def test_lift_without_depth_map_estimates_position_and_flags_it():
    det = detection(0.6, 3.0)
    person = lift(det, frame(None), POSE)
    assert not person.has_depth
    assert person.ground()[0] == pytest.approx(0.6, abs=0.25)  # 不能像 aero 一樣把 x 釘在 0
    assert person.ground()[1] == pytest.approx(3.0, abs=0.5)


def test_lift_with_no_torso_depth_flags_estimated():
    det = detection(1.5, 2.5)
    person = lift(det, frame(np.zeros((K.height, K.width), np.float32)), POSE)
    assert not person.has_depth
    assert person.ground()[0] == pytest.approx(1.5, abs=0.3)


def test_lift_passes_pixels_and_mask_through():
    det = detection(0.0, 2.5)
    det.mask = np.ones((K.height, K.width), np.float32)
    person = lift(det, frame(depth_for([(0.0, 2.5)])), POSE)
    assert person.pixels == det.pixels
    assert person.mask is det.mask


# ---- plausible ----


def test_standing_person_is_plausible():
    person = lift(detection(0.0, 2.5), frame(depth_for([(0.0, 2.5)])), POSE)
    assert plausible(person)


def test_tiny_figure_is_not_plausible():
    person = lift(detection(0.0, 2.5), frame(depth_for([(0.0, 2.5)])), POSE)
    hip = person.world["left_hip"].copy()
    person.world = {j: hip + (p - hip) * 0.3 for j, p in person.world.items()}  # 縮成 30%（例如桌上的公仔）
    assert not plausible(person)


def test_upside_down_is_not_plausible():
    person = lift(detection(0.0, 2.5), frame(depth_for([(0.0, 2.5)])), POSE)
    person.world = {j: p * np.array([1, -1, 1]) for j, p in person.world.items()}
    assert not plausible(person)


def test_estimated_person_skips_size_check():
    person = lift(detection(0.0, 2.5), frame(None), POSE)
    assert plausible(person)


# ---- grayout ----


def test_depth_gate_grays_far_and_side_pixels_but_keeps_holes_and_zone():
    gate = DepthGate(POSE, [Zone(0.0, 2.5)], margin_z=1.0, margin_x=0.2)  # 側邊界 x = ±1.2 m
    depth = np.zeros((K.height, K.width), np.float32)
    rays = np.stack(np.meshgrid((np.arange(K.width) - K.cx) / K.fx, (np.arange(K.height) - K.cy) / K.fy), -1)
    depth[200:280, 300:340] = 2.5  # 站位區裡
    depth[200:280, 100:140] = 6.0  # 背後很遠
    depth[200:280, 600:640] = 2.5  # 2.5 m 遠但在畫面最右邊：世界 x 約 +1.25 m，超出側邊界
    rgb = np.full((K.height, K.width, 3), 200, np.uint8)
    out = gate.apply(rgb, depth, K)
    assert (out[240, 320] == 200).all()  # 區內保留
    assert (out[240, 120] == 128).all()  # 太遠塗灰
    assert (out[10, 10] == 200).all()  # 沒深度保留（黑衣、細手臂）
    x_world = 2.5 * rays[240, 620, 0]
    assert x_world > 1.2 and (out[240, 620] == 128).all()  # 側邊塗灰
    assert (rgb == 200).all()  # 不改原圖


def test_depth_gate_without_depth_is_noop():
    gate = DepthGate(POSE, [Zone(0.0, 2.5)])
    rgb = np.full((K.height, K.width, 3), 200, np.uint8)
    assert gate.apply(rgb, None, K) is rgb
