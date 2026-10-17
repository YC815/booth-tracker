"""指令列冒煙測試：用合成的錄影跑 calibrate 和 eval（eval 用真的 MediaPipe 模型）。"""

from __future__ import annotations

import numpy as np
import pytest

from booth_tracker.cli import main
from booth_tracker.geometry import CameraPose
from booth_tracker.record import record
from booth_tracker.types import Frame
from booth_tracker.venue import Venue
from tests.test_geometry import K, render_depth

TRUTH = CameraPose.from_angles(1.45, 10.0)


class FloorSource:
    """空場地：只有地面的深度，灰色畫面。"""

    name = "fake"
    has_depth = True

    def __init__(self, n: int):
        self._n, self._i = n, 0
        self._depth = render_depth(TRUTH)

    def start(self): ...
    def stop(self): ...

    def read(self):
        if self._i >= self._n:
            return None
        f = Frame(rgb=np.full((K.height, K.width, 3), 90, np.uint8), depth_m=self._depth, intrinsics=K, ts=self._i / 30)
        self._i += 1
        return f


@pytest.fixture
def floor_recording(tmp_path):
    out = tmp_path / "rec" / "empty"
    record(FloorSource(45), out, seconds=0.3)
    return out


def test_calibrate_from_recording_with_manual_zone(floor_recording, tmp_path, capsys):
    venue_path = tmp_path / "venue.json"
    code = main(["calibrate", "--source", "replay", "--replay", str(floor_recording),
                 "--zone-x", "0", "--zone-z", "2.5", "--out", str(venue_path)])
    assert code == 0
    v = Venue.load(venue_path)
    assert v.camera.height == pytest.approx(1.45, abs=0.03)
    assert v.camera.pitch_deg == pytest.approx(10.0, abs=1.0)
    assert len(v.zones) == 1 and v.zones[0].z == 2.5
    assert "離地" in capsys.readouterr().out


def test_calibrate_two_zones(floor_recording, tmp_path):
    venue_path = tmp_path / "venue.json"
    code = main(["calibrate", "--source", "replay", "--replay", str(floor_recording), "--zones", "2",
                 "--radius", "0.5", "--gap", "0.4", "--zone-x", "0", "--zone-z", "2.5", "--out", str(venue_path)])
    assert code == 0
    left, right = Venue.load(venue_path).zones
    assert right.x - left.x == pytest.approx(1.4)


def test_calibrate_manual_camera_without_depth(tmp_path):
    venue_path = tmp_path / "venue.json"
    code = main(["calibrate", "--source", "replay", "--replay", str(tmp_path / "nope"), "--height", "1.2",
                 "--pitch", "5", "--zone-x", "0", "--zone-z", "2.5", "--out", str(venue_path)])
    assert code == 0  # 手動相機 + 手動站位區：不需要讀相機
    assert Venue.load(venue_path).camera.pitch_deg == pytest.approx(5.0)


def test_eval_runs_on_recording_and_writes_video(floor_recording, tmp_path, capsys):
    venue_path = tmp_path / "venue.json"
    main(["calibrate", "--source", "replay", "--replay", str(floor_recording),
          "--zone-x", "0", "--zone-z", "2.5", "--out", str(venue_path)])
    video = tmp_path / "out.mp4"
    code = main(["eval", "--replay", str(floor_recording), "--venue", str(venue_path), "--video", str(video)])
    assert code == 0
    out = capsys.readouterr().out
    assert "zone 0" in out and "認領 0 次" in out
    assert video.exists() and video.stat().st_size > 0
