from __future__ import annotations

import numpy as np
import pytest

from booth_tracker.geometry import CameraPose
from booth_tracker.lock import Zone
from booth_tracker.tracker import BoothTracker
from booth_tracker.venue import Venue
from tests.synth import K, POSE, depth_for, detection, frame

FPS = 30
VENUE = Venue(camera=POSE, zones=[Zone(0.0, 2.5)])


class FakeSource:
    name = "fake"

    def __init__(self, depth: np.ndarray | None, seconds: float):
        self.has_depth = depth is not None
        self._depth = depth
        self._n = round(seconds * FPS)
        self._i = 0

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def read(self):
        if self._i >= self._n:
            return None
        f = frame(self._depth, ts=self._i / FPS)
        self._i += 1
        return f


class FakeDetector:
    def __init__(self, script):
        self.script = script
        self.seen = []

    def detect(self, rgb, ts_ms):
        self.seen.append(rgb)
        return self.script(ts_ms / 1000)


def run_all(tracker: BoothTracker):
    out = []
    while (r := tracker.step()) is not None:
        out.append(r)
    return out


def test_claims_player_end_to_end():
    src = FakeSource(depth_for([(0.0, 2.5)], background=6.0), 1.3)
    det = FakeDetector(lambda t: [detection(0.0, 2.5, hands_up=True)])
    results = run_all(BoothTracker(src, VENUE, detector=det))
    last = results[-1]
    assert last.players[0].mode == "track"
    assert last.players[0].lock_id == 1
    assert len(last.persons) == 1
    assert last.players[0].player.ground() == pytest.approx([0.0, 2.5], abs=0.05)


def test_grays_out_background_before_detection():
    src = FakeSource(depth_for([(0.0, 2.5)], background=6.0), 0.1)
    det = FakeDetector(lambda t: [])
    results = run_all(BoothTracker(src, VENUE, detector=det))
    assert (det.seen[0][5, 5] == 128).all()  # 6 m 遠的背景
    assert (results[0].frame.rgb[5, 5] == 200).all()  # 原始畫面不動


def test_drops_implausible_detections():
    """遠處 7.5 m 的小人，但那些像素量到的深度是 2.5 m（例如牆上海報）：換算出來肩寬只有十幾公分。"""
    wall = np.full((K.height, K.width), 2.5, np.float32)
    src = FakeSource(wall, 0.1)
    det = FakeDetector(lambda t: [detection(0.0, 7.5)])
    results = run_all(BoothTracker(src, VENUE, detector=det))
    assert results[0].persons == []


def test_webcam_without_depth_still_claims_with_estimated_position():
    src = FakeSource(None, 1.3)
    det = FakeDetector(lambda t: [detection(0.0, 2.5, hands_up=True)])
    results = run_all(BoothTracker(src, VENUE, detector=det))
    assert results[-1].players[0].mode == "track"
    assert not results[-1].players[0].player.has_depth


def test_smoothing_reduces_jitter():
    src = FakeSource(None, 3.0)
    # 認領後玩家左右抖 ±3 cm
    det = FakeDetector(lambda t: [detection(0.03 * (-1) ** round(t * FPS) if t > 1.2 else 0.0, 2.5, hands_up=t < 1.2)])
    results = run_all(BoothTracker(src, VENUE, detector=det))
    tracked = [r for r in results[-40:] if r.players[0].mode == "track"]
    raw = np.std([r.persons[0].ground()[0] for r in tracked])
    smooth = np.std([r.players[0].player.ground()[0] for r in tracked])
    assert smooth < raw * 0.6


def test_no_smoothing_returns_raw_person():
    src = FakeSource(None, 1.3)
    det = FakeDetector(lambda t: [detection(0.0, 2.5, hands_up=True)])
    last = run_all(BoothTracker(src, VENUE, detector=det, smoothing=None))[-1]
    assert last.players[0].player is last.persons[0]
