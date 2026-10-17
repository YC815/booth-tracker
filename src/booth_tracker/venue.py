"""場地校正：相機姿勢（離地高度、俯角、側傾）和站位區。每換一個場地跑一次 calibrate，存成 venue.json。"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from booth_tracker.geometry import CameraPose, fit_floor
from booth_tracker.lock import Zone
from booth_tracker.paths import DATA_DIR
from booth_tracker.types import Intrinsics, Person

DEFAULT_PATH = DATA_DIR / "venue.json"


@dataclass
class Venue:
    camera: CameraPose
    zones: list[Zone]

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or DEFAULT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "camera": {"height": self.camera.height, "pitch_deg": self.camera.pitch_deg, "roll_deg": self.camera.roll_deg},
            "zones": [asdict(z) for z in self.zones],
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> Venue:
        data = json.loads(Path(path or DEFAULT_PATH).read_text())
        c = data["camera"]
        return cls(
            camera=CameraPose.from_angles(c["height"], c["pitch_deg"], c["roll_deg"]),
            zones=[Zone(**z) for z in data["zones"]],
        )


def calibrate_floor(depths: list[np.ndarray], k: Intrinsics) -> CameraPose | None:
    """多張深度取每像素中位數（忽略無效的 0）再找地面。場地要淨空：站位區附近的地面不能有人。"""
    stack = np.stack(depths).astype(np.float32)
    stack[stack <= 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 整欄都沒深度的像素
        med = np.nanmedian(stack, axis=0)
    return fit_floor(np.nan_to_num(med, nan=0.0), k, rng=np.random.default_rng(0))


def zone_center(frames: list[list[Person]], min_ratio: float = 0.5) -> tuple[float, float] | None:
    """一個人站上膠帶幾秒：取每幀離相機最近的人，地面位置的中位數就是站位區中心。
    有人的幀數不到 min_ratio 回 None。"""
    picks = [min(ps, key=lambda p: p.ground()[1]).ground() for ps in frames if ps]
    if not frames or len(picks) < min_ratio * len(frames):
        return None
    xs, zs = zip(*picks)
    return float(np.median(xs)), float(np.median(zs))
