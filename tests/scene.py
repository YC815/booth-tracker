"""合成場景：用幾個參數造出 Person，按時間餵給 PlayerLock。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from booth_tracker.lock import PlayerLock
from booth_tracker.types import Person, PlayerFrame

FPS = 30
DT = 1 / FPS


def person(x: float, z: float, hands_up: bool = False, height: float = 1.7, has_depth: bool = True) -> Person:
    h = height
    wrist_y = h + 0.15 if hands_up else h * 0.45
    pts = {
        "nose": (0.0, h - 0.1),
        "left_shoulder": (-0.19, h * 0.82),
        "right_shoulder": (0.19, h * 0.82),
        "left_hip": (-0.1, h * 0.53),
        "right_hip": (0.1, h * 0.53),
        "left_wrist": (-0.25, wrist_y),
        "right_wrist": (0.25, wrist_y),
    }
    world = {k: np.array([x + dx, y, z]) for k, (dx, y) in pts.items()}
    return Person(world=world, visibility={k: 1.0 for k in world}, has_depth=has_depth)


Scene = Callable[[float], list[Person]]  # 時間 t（秒）-> 這幀的人


def run(lock: PlayerLock, scene: Scene, seconds: float, t0: float = 0.0) -> list[tuple[float, PlayerFrame, list[Person]]]:
    out = []
    for i in range(round(seconds * FPS)):
        t = t0 + i * DT
        persons = scene(t)
        out.append((t, lock.update(persons, t), persons))
    return out


def player_x(frame: PlayerFrame) -> float | None:
    return None if frame.player is None else float(frame.player.ground()[0])
