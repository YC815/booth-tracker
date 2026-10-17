"""拿錄影跑整條管線：列出鎖定事件、統計各模式幀數，可輸出標註影片。調參和檢查錯誤換鎖用。"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from booth_tracker.lift import estimate_intrinsics
from booth_tracker.tracker import BoothTracker, TrackerFrame
from booth_tracker.types import Intrinsics
from booth_tracker.venue import Venue

BONES = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)
ZONE_COLORS = ((0, 200, 255), (255, 120, 0), (0, 220, 0))  # BGR
GRAY = (150, 150, 150)


@dataclass
class Event:
    t: float
    zone_id: int
    what: str


@dataclass
class ZoneStats:
    claims: int = 0
    frames: Counter = field(default_factory=Counter)
    longest_hold_s: float = 0.0


@dataclass
class Report:
    frames: int
    seconds: float  # 處理耗時（牆鐘）
    events: list[Event]
    zones: list[ZoneStats]

    def text(self) -> str:
        fps = self.frames / self.seconds if self.seconds > 0 else 0.0
        lines = [f"{self.frames} 幀，處理 {self.seconds:.1f} 秒（{fps:.1f} fps）", ""]
        lines += [f"{e.t:8.2f}s  zone {e.zone_id}  {e.what}" for e in self.events]
        lines.append("")
        for k, z in enumerate(self.zones):
            modes = "  ".join(f"{m} {n}" for m, n in sorted(z.frames.items()))
            lines.append(f"zone {k}：認領 {z.claims} 次；{modes}；最長看不到玩家 {z.longest_hold_s:.1f} 秒")
        return "\n".join(lines)


def evaluate(tracker: BoothTracker, on_frame: Callable[[TrackerFrame], None] | None = None) -> Report:
    """跑到來源沒有幀為止（回放請用 loop=False）。"""
    n_zones = len(tracker.venue.zones)
    stats = [ZoneStats() for _ in range(n_zones)]
    events: list[Event] = []
    prev_mode = ["idle"] * n_zones
    prev_lock: list[int | None] = [None] * n_zones
    hold_since: list[float | None] = [None] * n_zones
    frames = 0
    t0 = time.perf_counter()
    while (tf := tracker.step()) is not None:
        frames += 1
        ts = tf.frame.ts
        for pf in tf.players:
            k = pf.zone_id
            stats[k].frames[pf.mode] += 1
            if pf.lock_id != prev_lock[k]:
                what = f"認領 lock_id={pf.lock_id}" if pf.lock_id is not None else f"放掉 lock_id={prev_lock[k]}"
                events.append(Event(ts, k, what))
                stats[k].claims += pf.lock_id is not None
            if pf.mode != prev_mode[k]:
                events.append(Event(ts, k, f"{prev_mode[k]} → {pf.mode}"))
            if pf.mode in ("hold", "paused"):
                hold_since[k] = ts if hold_since[k] is None else hold_since[k]
                stats[k].longest_hold_s = max(stats[k].longest_hold_s, ts - hold_since[k])
            else:
                hold_since[k] = None
            prev_lock[k], prev_mode[k] = pf.lock_id, pf.mode
        if on_frame is not None:
            on_frame(tf)
    return Report(frames=frames, seconds=time.perf_counter() - t0, events=events, zones=stats)


def _draw_skeleton(img: np.ndarray, pixels: dict[str, tuple[float, float]], color, thickness: int) -> None:
    for a, b in BONES:
        if a in pixels and b in pixels:
            pa = tuple(int(round(v)) for v in pixels[a])
            pb = tuple(int(round(v)) for v in pixels[b])
            cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)


def _draw_zone(img: np.ndarray, venue: Venue, k: int, k_intr: Intrinsics, color) -> None:
    z = venue.zones[k]
    a = np.linspace(0, 2 * np.pi, 48)
    pts = np.stack([z.x + z.r_in * np.cos(a), np.zeros_like(a), z.z + z.r_in * np.sin(a)], axis=-1)
    if (venue.camera.to_camera(pts)[:, 2] < 0.1).any():
        return  # 圓有一部分在相機後面，不畫
    cv2.polylines(img, [venue.camera.project(pts, k_intr).astype(np.int32)], True, color, 2, cv2.LINE_AA)


def annotate(tf: TrackerFrame, venue: Venue) -> np.ndarray:
    """回傳 BGR 標註圖（可直接寫進 cv2.VideoWriter）：站位區、所有人灰色骨架、玩家彩色骨架、各區狀態。"""
    img = cv2.cvtColor(tf.frame.rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    k_intr = tf.frame.intrinsics or estimate_intrinsics(w, h)
    for k in range(len(venue.zones)):
        _draw_zone(img, venue, k, k_intr, ZONE_COLORS[k % len(ZONE_COLORS)])
    players = {pf.player_index: pf.zone_id for pf in tf.players if pf.player_index is not None}
    for i, p in enumerate(tf.persons):
        if i in players:
            _draw_skeleton(img, p.pixels, ZONE_COLORS[players[i] % len(ZONE_COLORS)], 4)
        else:
            _draw_skeleton(img, p.pixels, GRAY, 2)
    for row, pf in enumerate(tf.players):
        claim = f" claim {pf.claim_progress:.0%}" if pf.claim_progress > 0 else ""
        text = f"zone {pf.zone_id}: {pf.mode} lock={pf.lock_id}{' OUT' if pf.mode == 'track' and not pf.in_zone else ''}{claim}"
        color = ZONE_COLORS[pf.zone_id % len(ZONE_COLORS)]
        cv2.putText(img, text, (12, 30 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(img, f"t={tf.frame.ts:.2f}", (12, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img
