from __future__ import annotations

import numpy as np
import pytest

from booth_tracker.geometry import CameraPose
from booth_tracker.lock import Zone
from booth_tracker.venue import Venue, calibrate_floor, zone_center
from tests.scene import person
from tests.test_geometry import K, render_depth


def test_save_load_roundtrip(tmp_path):
    v = Venue(camera=CameraPose.from_angles(1.35, 9.0, -1.5), zones=[Zone(-0.7, 2.5, 0.5, 0.7), Zone(0.7, 2.5, 0.5, 0.7)])
    path = tmp_path / "venue.json"
    v.save(path)
    w = Venue.load(path)
    assert w.camera.height == pytest.approx(1.35)
    assert w.camera.pitch_deg == pytest.approx(9.0)
    assert w.camera.roll_deg == pytest.approx(-1.5)
    assert w.zones == v.zones


def test_calibrate_floor_from_several_noisy_frames():
    truth = CameraPose.from_angles(1.5, 12.0, 1.0)
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(10):
        d = render_depth(truth)
        d[rng.random(d.shape) < 0.2] = 0  # 兩成像素沒深度
        frames.append(d)
    pose = calibrate_floor(frames, K)
    assert pose is not None
    assert pose.height == pytest.approx(1.5, abs=0.03)
    assert pose.pitch_deg == pytest.approx(12.0, abs=1.0)


def test_calibrate_floor_fails_on_wall_only():
    frames = [np.full((K.height, K.width), 3.0, np.float32)] * 3
    assert calibrate_floor(frames, K) is None


def test_zone_center_is_median_of_person_nearest_camera():
    frames = [[person(0.1 + 0.01 * (i % 3), 2.4), person(1.5, 4.0)] for i in range(30)]
    x, z = zone_center(frames)
    assert x == pytest.approx(0.11, abs=0.02)
    assert z == pytest.approx(2.4, abs=0.02)


def test_zone_center_needs_someone_most_of_the_time():
    frames = [[person(0.0, 2.5)]] * 5 + [[]] * 25
    assert zone_center(frames) is None
