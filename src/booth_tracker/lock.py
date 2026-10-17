"""玩家鎖定：換人只能靠站位區內舉手認領；追丟只會 hold → paused → idle，絕不自動換人。

每幀把偵測到的人配到「軌跡」（地面 (x, z) 最近鄰），玩家 = 某條軌跡。
- 認領：站位區內、舉手、幾乎不動，持續 claim_s，而且同時只有一個人這樣做。
- 跟隨：玩家軌跡只能配到離上次位置 assoc_radius 內、且沒有比玩家近 occlusion_dz 以上的人；
  兩個候選距離太接近就這幀先不配（hold），不賭。
- 認回：玩家軌跡斷掉後，只有「玩家消失之後才出現」的新軌跡、在原位附近、連續 rejoin_frames 幀，才能接回。
  玩家是自己走出站位區後才消失的，就不認回，要重新舉手。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from booth_tracker.types import Mode, Person, PlayerFrame


@dataclass
class LockConfig:
    zone_x: float = 0.0  # 站位區（膠帶）中心，世界座標
    zone_z: float = 2.5
    zone_r_in: float = 0.7  # 進站位區半徑
    zone_r_out: float = 1.0  # 出站位區半徑（遲滯）
    claim_s: float = 1.0  # 舉手要撐多久
    claim_max_speed: float = 0.3  # m/s，舉手時要幾乎站定
    hands_up_margin: float = 0.1  # 雙腕要高於鼻子多少公尺
    min_visibility: float = 0.5
    require_depth: bool = True  # 沒量到軀幹深度的人完全不參與（webcam 模式設 False）
    assoc_radius: float = 0.4  # 軌跡每次配對最多離上次位置多遠
    ambiguity_margin: float = 0.1  # 玩家兩個候選距離差小於此值就不配
    max_gap_s: float = 0.3  # 軌跡沒配到超過這麼久就刪掉
    occlusion_dz: float = 0.3  # 比玩家近這麼多（z 方向）的人不能是玩家
    rejoin_radius: float = 0.5
    rejoin_frames: int = 8
    hold_s: float = 3.0  # hold 多久變 paused
    pause_s: float = 8.0  # paused 多久放掉鎖定回 idle


@dataclass
class _Track:
    id: int
    pos: np.ndarray  # 地面 (x, z)
    born: float
    seen: float
    person: Person
    in_zone: bool
    speed: float = 0.0  # m/s，EMA
    hits: int = 1  # 連續配到的幀數
    hands_up_since: float | None = None
    det: int | None = None  # 這幀配到的 persons 索引


def _hands_up(p: Person, cfg: LockConfig) -> bool:
    names = ("nose", "left_wrist", "right_wrist")
    if any(n not in p.world or p.visibility.get(n, 0.0) < cfg.min_visibility for n in names):
        return False
    nose_y = p.world["nose"][1]
    return all(p.world[w][1] > nose_y + cfg.hands_up_margin for w in ("left_wrist", "right_wrist"))


class PlayerLock:
    def __init__(self, config: LockConfig | None = None) -> None:
        self.config = config or LockConfig()
        self._tracks: dict[int, _Track] = {}
        self._next_track = 1
        self._lock_count = 0
        self._clear_lock()

    def _clear_lock(self) -> None:
        self._lock_id: int | None = None
        self._player_id: int | None = None
        self._last_person: Person | None = None
        self._last_pos: np.ndarray | None = None
        self._last_seen = 0.0  # 玩家最後一次配到的時間
        self._last_in_zone = False
        self._hold_since: float | None = None

    def release(self) -> None:
        """遊戲一局結束時呼叫：放掉鎖定回 idle。"""
        self._clear_lock()

    # ---- 軌跡 ----

    def _zone_dist(self, pos: np.ndarray) -> float:
        return float(np.hypot(pos[0] - self.config.zone_x, pos[1] - self.config.zone_z))

    def _player_ok(self, tr: _Track, pos: np.ndarray) -> bool:
        """玩家軌跡能不能配到這個位置：不能比玩家近 occlusion_dz 以上（那是擋在前面的人）。"""
        return pos[1] >= tr.pos[1] - self.config.occlusion_dz

    def _player_ambiguous(self, dets: list[np.ndarray]) -> bool:
        player = self._tracks.get(self._player_id) if self._player_id is not None else None
        if player is None:
            return False
        cfg = self.config
        near = sorted(
            float(np.linalg.norm(player.pos - p)) for p in dets if self._player_ok(player, p)
        )
        near = [d for d in near if d <= cfg.assoc_radius]
        return len(near) >= 2 and near[1] - near[0] < cfg.ambiguity_margin

    def _associate(self, persons: list[Person], idx: list[int], now: float) -> None:
        cfg = self.config
        self._tracks = {k: t for k, t in self._tracks.items() if now - t.seen <= cfg.max_gap_s}
        dets = [persons[i].ground() for i in idx]
        ambiguous = self._player_ambiguous(dets)
        pairs = []
        for tr in self._tracks.values():
            tr.det = None
            is_player = tr.id == self._player_id
            if is_player and ambiguous:
                continue
            for j, p in enumerate(dets):
                d = float(np.linalg.norm(tr.pos - p))
                if d <= cfg.assoc_radius and (not is_player or self._player_ok(tr, p)):
                    pairs.append((d, tr.id, j))
        used: set[int] = set()
        for d, tid, j in sorted(pairs):
            tr = self._tracks[tid]
            if tr.det is not None or j in used:
                continue
            self._update_track(tr, persons[idx[j]], dets[j], d, now)
            tr.det = idx[j]
            used.add(j)
        for tr in self._tracks.values():
            if tr.det is None:
                tr.hits = 0
        player = self._tracks.get(self._player_id) if self._player_id is not None else None
        for j, p in enumerate(dets):
            if j in used:
                continue
            if ambiguous and player is not None and np.linalg.norm(player.pos - p) <= cfg.assoc_radius:
                continue  # 玩家附近分不清的人先不開新軌跡
            self._new_track(persons[idx[j]], p, idx[j], now)

    def _update_track(self, tr: _Track, person: Person, pos: np.ndarray, d: float, now: float) -> None:
        cfg = self.config
        dt = now - tr.seen
        v = d / dt if dt > 0 else 0.0
        tr.speed = 0.5 * tr.speed + 0.5 * v
        tr.pos, tr.seen, tr.person = pos, now, person
        tr.hits += 1
        zd = self._zone_dist(pos)
        if zd <= cfg.zone_r_in:
            tr.in_zone = True
        elif zd > cfg.zone_r_out:
            tr.in_zone = False
        self._update_hands(tr, now)

    def _new_track(self, person: Person, pos: np.ndarray, det: int, now: float) -> None:
        tr = _Track(
            id=self._next_track,
            pos=pos,
            born=now,
            seen=now,
            person=person,
            in_zone=self._zone_dist(pos) <= self.config.zone_r_in,
            det=det,
        )
        self._next_track += 1
        self._update_hands(tr, now)
        self._tracks[tr.id] = tr

    def _update_hands(self, tr: _Track, now: float) -> None:
        if not _hands_up(tr.person, self.config) or tr.speed > self.config.claim_max_speed:
            tr.hands_up_since = None
        elif tr.hands_up_since is None:
            tr.hands_up_since = now

    # ---- 認領 / 認回 ----

    def _claim(self, now: float) -> float:
        """處理舉手認領；回傳最高的舉手進度 0..1。"""
        cfg = self.config
        raising = [
            t
            for t in self._tracks.values()
            if t.det is not None and t.in_zone and t.hands_up_since is not None and t.id != self._player_id
        ]
        if not raising:
            return 0.0
        progress = min(1.0, max((now - t.hands_up_since) / cfg.claim_s for t in raising))
        if len(raising) != 1 or progress < 1.0:
            return progress if len(raising) == 1 else 0.0
        tr = raising[0]
        self._clear_lock()
        self._lock_count += 1
        self._lock_id = self._lock_count
        self._player_id = tr.id
        tr.hands_up_since = None
        return 0.0

    def _rejoin(self) -> None:
        cfg = self.config
        if not self._last_in_zone or self._last_pos is None:
            return
        last = self._last_pos
        cands = [
            t
            for t in self._tracks.values()
            if t.det is not None
            and t.born > self._last_seen
            and t.in_zone
            and t.hits >= cfg.rejoin_frames
            and np.linalg.norm(t.pos - last) <= cfg.rejoin_radius
            and t.pos[1] >= last[1] - cfg.occlusion_dz
        ]
        if len(cands) == 1:
            self._player_id = cands[0].id

    # ---- 主流程 ----

    def update(self, persons: list[Person], now: float) -> PlayerFrame:
        cfg = self.config
        idx = [i for i, p in enumerate(persons) if p.has_depth or not cfg.require_depth]
        self._associate(persons, idx, now)

        tracking = self._player_id is not None and self._player_id in self._tracks and self._tracks[self._player_id].det is not None
        progress = 0.0
        if not (tracking and self._tracks[self._player_id].in_zone):
            progress = self._claim(now)

        if self._lock_id is None:
            return PlayerFrame(mode="idle", lock_id=None, player=None, player_index=None, claim_progress=progress)

        if self._player_id not in self._tracks:
            self._rejoin()
        player = self._tracks.get(self._player_id)
        if player is not None and player.det is not None:
            self._last_pos, self._last_seen = player.pos, now
            self._last_in_zone = player.in_zone
            if player.in_zone:
                self._last_person, self._hold_since = player.person, None
                return PlayerFrame(mode="track", lock_id=self._lock_id, player=player.person, player_index=player.det)

        if self._hold_since is None:
            self._hold_since = now
        held = now - self._hold_since
        if held > cfg.hold_s + cfg.pause_s:
            self._clear_lock()
            return PlayerFrame(mode="idle", lock_id=None, player=None, player_index=None, claim_progress=progress)
        mode: Mode = "paused" if held > cfg.hold_s else "hold"
        return PlayerFrame(mode=mode, lock_id=self._lock_id, player=self._last_person, player_index=None, claim_progress=progress)
