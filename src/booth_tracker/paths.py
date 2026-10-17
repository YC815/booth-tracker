"""資料目錄：模型、錄影、場地校正。多個遊戲共用同一份（同一台電腦、同一個場地）。"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("BOOTH_DATA_DIR", Path.home() / ".booth-tracker"))
