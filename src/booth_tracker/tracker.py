"""整條管線：相機 → 站位區外塗灰 → 姿勢 → 3D 抬升 → 像不像人 → 鎖定 → 平滑。遊戲只要用這個。

    with BoothTracker(make_source("orbbec"), Venue.load()) as tracker:
        while True:
            r = tracker.step()
            if r is None:
                continue
            for p in r.players:  # 每個站位區一個 PlayerFrame
                ...
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from booth_tracker.filter import JointFilter
from booth_tracker.grayout import DepthGate
from booth_tracker.lift import estimate_intrinsics, lift
from booth_tracker.lock import LockConfig, PlayerLock
from booth_tracker.plausible import plausible
from booth_tracker.sources.base import FrameSource
from booth_tracker.types import Detection, Frame, Person, PlayerFrame
from booth_tracker.venue import Venue


class Detector(Protocol):
    def detect(self, rgb: np.ndarray, ts_ms: int) -> list[Detection]: ...


@dataclass
class TrackerFrame:
    frame: Frame  # 原始畫面（沒塗灰）
    persons: list[Person]  # 這幀所有像人的人（世界座標，未平滑）
    players: list[PlayerFrame]  # 每個站位區一個，依 zone_id 排序；player 已平滑


class BoothTracker:
    def __init__(
        self,
        source: FrameSource,
        venue: Venue,
        *,
        lock_config: LockConfig | None = None,
        detector: Detector | None = None,
        smoothing: tuple[float, float] | None = (1.0, 0.05),  # One Euro (min_cutoff, beta)；None = 不平滑
        gray_margin: tuple[float, float] = (1.0, 0.8),  # 站位區外多遠開始塗灰 (前後, 左右)，公尺
    ):
        cfg = lock_config or LockConfig()
        # 沒深度的來源（webcam）每個人都是估的，不能要求深度
        cfg = replace(cfg, zones=venue.zones, require_depth=cfg.require_depth and source.has_depth)
        self.source = source
        self.venue = venue
        self.lock = PlayerLock(cfg)
        self.gate = DepthGate(venue.camera, venue.zones, *gray_margin)
        if detector is None:
            from booth_tracker.pose import PoseDetector

            detector = PoseDetector(num_poses=len(venue.zones) + 2)
        self.detector = detector
        self._smoothing = smoothing
        n = len(venue.zones)
        self._filters: list[JointFilter | None] = [None] * n
        self._filter_lock: list[int | None] = [None] * n
        self._smoothed: list[Person | None] = [None] * n

    def __enter__(self) -> BoothTracker:
        self.source.start()
        return self

    def __exit__(self, *exc) -> None:
        self.source.stop()

    def step(self) -> TrackerFrame | None:
        """讀一幀並處理；來源沒有新幀時回 None。"""
        frame = self.source.read()
        if frame is None:
            return None
        h, w = frame.rgb.shape[:2]
        k = frame.intrinsics or estimate_intrinsics(w, h)
        rgb = self.gate.apply(frame.rgb, frame.depth_m, k)
        dets = self.detector.detect(rgb, int(frame.ts * 1000))
        persons = [p for p in (lift(d, frame, self.venue.camera) for d in dets) if plausible(p)]
        players = self.lock.update(persons, frame.ts)
        if self._smoothing is not None:
            players = [self._smooth(pf, frame.ts) for pf in players]
        return TrackerFrame(frame=frame, persons=persons, players=players)

    def release(self, zone_id: int | None = None) -> None:
        """一局結束時呼叫，見 PlayerLock.release。"""
        self.lock.release(zone_id)

    def _smooth(self, pf: PlayerFrame, ts: float) -> PlayerFrame:
        """換人（lock_id 變了）就重來；hold/paused 沿用最後一個平滑過的姿勢。"""
        k = pf.zone_id
        if pf.lock_id != self._filter_lock[k]:
            self._filters[k] = JointFilter(*self._smoothing)
            self._filter_lock[k], self._smoothed[k] = pf.lock_id, None
        if pf.mode == "track":
            smoothed = replace(pf.player, world=self._filters[k](pf.player.world, ts))
            self._smoothed[k] = smoothed
            return replace(pf, player=smoothed)
        if pf.player is not None and self._smoothed[k] is not None:
            return replace(pf, player=self._smoothed[k])
        return pf
