"""玩家鎖定：換人只能靠站位區內舉手認領；追丟只會 hold → paused → idle，絕不自動換人。

可以有 N 個站位區，每區一個鎖定槽，各自獨立。
每幀把偵測到的人配到「軌跡」（地面 (x, z) 最近鄰），玩家 = 某條軌跡。
- 認領：區內、舉手、幾乎不動，持續 claim_s，而且區內同時只有一個人這樣做。已是玩家的人不能在任何區認領。
- 跟隨：玩家軌跡只能配到離上次位置 assoc_radius 內、且沒有比玩家近 occlusion_dz 以上的人；
  兩個候選距離太接近就這幀先不配（hold），不賭。
- 出區：玩家看得到就照常 track，in_zone=False；連續在區外超過 hold_s + pause_s 才放掉。
- 認回：玩家軌跡斷掉後，只有「玩家消失之後才出現」的新軌跡、在原位附近、連續 rejoin_frames 幀，才能接回。
  玩家是自己走出區外後才消失的，就不認回，要重新舉手。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from booth_tracker.types import Mode, Person, PlayerFrame


@dataclass
class Zone:
    """站位區：地面上的圓，世界座標。"""

    x: float
    z: float
    r_in: float = 0.7  # 進區半徑
    r_out: float = 1.0  # 出區半徑（遲滯）

    def dist(self, pos: np.ndarray) -> float:
        return float(np.hypot(pos[0] - self.x, pos[1] - self.z))


def side_by_side(center_x: float, z: float, radius: float, gap: float) -> list[Zone]:
    """左右兩區，中間留 gap 公尺空白帶。出區遲滯只延伸到空白帶中線為止（最多 0.2 m）。"""
    off = radius + gap / 2
    r_out = radius + min(0.2, gap / 2)
    return [Zone(center_x - off, z, radius, r_out), Zone(center_x + off, z, radius, r_out)]


@dataclass
class LockConfig:
    zones: list[Zone] = field(default_factory=lambda: [Zone(0.0, 2.5)])
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
    pause_s: float = 8.0  # paused 多久放掉鎖定回 idle（出區太久也用 hold_s + pause_s）


@dataclass
class _Track:
    id: int
    pos: np.ndarray  # 地面 (x, z)
    born: float
    seen: float
    person: Person
    zone: int | None  # 目前在哪一區（含遲滯）
    speed: float = 0.0  # m/s，EMA
    hits: int = 1  # 連續配到的幀數
    hands_up_since: float | None = None
    det: int | None = None  # 這幀配到的 persons 索引


@dataclass
class _Slot:
    """一區的鎖定狀態。"""

    lock_id: int | None = None
    player_id: int | None = None  # 玩家的軌跡 id
    last_person: Person | None = None
    last_pos: np.ndarray | None = None
    last_seen: float = 0.0  # 玩家最後一次配到的時間
    last_in_zone: bool = False
    hold_since: float | None = None  # 看不到玩家的起點
    out_since: float | None = None  # 玩家在區外的起點


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
        self._slots = [_Slot() for _ in self.config.zones]
        self._next_track = 1
        self._lock_count = 0

    def release(self, zone_id: int | None = None) -> None:
        """遊戲一局結束時呼叫：放掉鎖定回 idle。zone_id 省略 = 全部區。"""
        ids = range(len(self._slots)) if zone_id is None else [zone_id]
        for k in ids:
            self._slots[k] = _Slot()

    def _player_ids(self) -> set[int]:
        return {s.player_id for s in self._slots if s.player_id is not None}

    # ---- 軌跡 ----

    def _zone_of(self, pos: np.ndarray, current: int | None) -> int | None:
        zones = self.config.zones
        if current is not None and zones[current].dist(pos) <= zones[current].r_out:
            return current
        return next((k for k, z in enumerate(zones) if z.dist(pos) <= z.r_in), None)

    def _player_ok(self, tr: _Track, pos: np.ndarray) -> bool:
        """玩家軌跡能不能配到這個位置：不能比玩家近 occlusion_dz 以上（那是擋在前面的人）。"""
        return pos[1] >= tr.pos[1] - self.config.occlusion_dz

    def _ambiguous(self, tr: _Track, dets: list[np.ndarray]) -> bool:
        cfg = self.config
        near = sorted(float(np.linalg.norm(tr.pos - p)) for p in dets if self._player_ok(tr, p))
        near = [d for d in near if d <= cfg.assoc_radius]
        return len(near) >= 2 and near[1] - near[0] < cfg.ambiguity_margin

    def _associate(self, persons: list[Person], idx: list[int], now: float) -> None:
        cfg = self.config
        self._tracks = {k: t for k, t in self._tracks.items() if now - t.seen <= cfg.max_gap_s}
        dets = [persons[i].ground() for i in idx]
        players = self._player_ids()
        ambiguous = [t for t in self._tracks.values() if t.id in players and self._ambiguous(t, dets)]
        skip = {t.id for t in ambiguous}
        pairs = []
        for tr in self._tracks.values():
            tr.det = None
            if tr.id in skip:
                continue
            for j, p in enumerate(dets):
                d = float(np.linalg.norm(tr.pos - p))
                if d <= cfg.assoc_radius and (tr.id not in players or self._player_ok(tr, p)):
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
        for j, p in enumerate(dets):
            if j in used:
                continue
            if any(np.linalg.norm(t.pos - p) <= cfg.assoc_radius for t in ambiguous):
                continue  # 玩家附近分不清的人先不開新軌跡
            self._new_track(persons[idx[j]], p, idx[j], now)

    def _update_track(self, tr: _Track, person: Person, pos: np.ndarray, d: float, now: float) -> None:
        dt = now - tr.seen
        v = d / dt if dt > 0 else 0.0
        tr.speed = 0.5 * tr.speed + 0.5 * v
        tr.pos, tr.seen, tr.person = pos, now, person
        tr.hits += 1
        tr.zone = self._zone_of(pos, tr.zone)
        self._update_hands(tr, now)

    def _new_track(self, person: Person, pos: np.ndarray, det: int, now: float) -> None:
        tr = _Track(
            id=self._next_track,
            pos=pos,
            born=now,
            seen=now,
            person=person,
            zone=self._zone_of(pos, None),
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

    def _claim(self, k: int, now: float) -> float:
        """處理 k 區的舉手認領；回傳舉手進度 0..1（區內有兩人舉手時為 0）。"""
        players = self._player_ids()
        raising = [
            t
            for t in self._tracks.values()
            if t.det is not None and t.zone == k and t.hands_up_since is not None and t.id not in players
        ]
        if len(raising) != 1:
            return 0.0
        tr = raising[0]
        progress = min(1.0, (now - tr.hands_up_since) / self.config.claim_s)
        if progress < 1.0:
            return progress
        self._lock_count += 1
        self._slots[k] = _Slot(lock_id=self._lock_count, player_id=tr.id)
        tr.hands_up_since = None
        return 0.0

    def _rejoin(self, k: int) -> None:
        cfg = self.config
        slot = self._slots[k]
        if not slot.last_in_zone or slot.last_pos is None:
            return
        last = slot.last_pos
        players = self._player_ids()
        cands = [
            t
            for t in self._tracks.values()
            if t.det is not None
            and t.id not in players
            and t.born > slot.last_seen
            and t.zone == k
            and t.hits >= cfg.rejoin_frames
            and np.linalg.norm(t.pos - last) <= cfg.rejoin_radius
            and t.pos[1] >= last[1] - cfg.occlusion_dz
        ]
        if len(cands) == 1:
            slot.player_id = cands[0].id

    # ---- 主流程 ----

    def update(self, persons: list[Person], now: float) -> list[PlayerFrame]:
        cfg = self.config
        idx = [i for i, p in enumerate(persons) if p.has_depth or not cfg.require_depth]
        self._associate(persons, idx, now)
        return [self._update_slot(k, now) for k in range(len(self._slots))]

    def _update_slot(self, k: int, now: float) -> PlayerFrame:
        cfg = self.config
        slot = self._slots[k]
        player = self._tracks.get(slot.player_id) if slot.player_id is not None else None
        visible_in_zone = player is not None and player.det is not None and player.zone == k
        progress = 0.0 if visible_in_zone else self._claim(k, now)
        slot = self._slots[k]  # 認領會換掉整個槽
        others = self._player_ids() - {slot.player_id}
        occupied = any(t.det is not None and t.zone == k and t.id not in others for t in self._tracks.values())

        def frame(mode: Mode, **kw) -> PlayerFrame:
            lock_id = self._slots[k].lock_id
            return PlayerFrame(zone_id=k, mode=mode, lock_id=lock_id, occupied=occupied, claim_progress=progress, **kw)

        if slot.lock_id is None:
            return frame("idle", player=None, player_index=None)

        if slot.player_id not in self._tracks:
            self._rejoin(k)
        player = self._tracks.get(slot.player_id)
        if player is not None and player.det is not None:
            return self._track_frame(k, slot, player, now, frame)

        if slot.hold_since is None:
            slot.hold_since = now
        held = now - slot.hold_since
        if held > cfg.hold_s + cfg.pause_s:
            self._slots[k] = _Slot()
            return frame("idle", player=None, player_index=None)
        return frame("paused" if held > cfg.hold_s else "hold", player=slot.last_person, player_index=None)

    def _track_frame(self, k: int, slot: _Slot, player: _Track, now: float, frame) -> PlayerFrame:
        cfg = self.config
        in_zone = player.zone == k
        slot.last_pos, slot.last_seen, slot.last_in_zone = player.pos, now, in_zone
        slot.last_person, slot.hold_since = player.person, None
        if in_zone:
            slot.out_since = None
        elif slot.out_since is None:
            slot.out_since = now
        if slot.out_since is not None and now - slot.out_since > cfg.hold_s + cfg.pause_s:
            self._slots[k] = _Slot()
            return frame("idle", player=None, player_index=None)
        return frame("track", player=player.person, player_index=player.det, in_zone=in_zone)
