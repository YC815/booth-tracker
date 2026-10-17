"""姿勢模型偵測到的「人」像不像真人：擋掉公仔、海報上的小人、倒過來的誤判。

有深度時才檢查公尺尺寸；沒深度時尺寸是用肩寬假設估的，檢查了也沒意義。
"""

from __future__ import annotations

import numpy as np

from booth_tracker.types import Person

CORE = ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")
MIN_VISIBILITY = 0.5
SHOULDER_M = (0.2, 0.65)
TORSO_M = (0.25, 0.85)


def plausible(p: Person) -> bool:
    if any(p.visibility.get(j, 0.0) < MIN_VISIBILITY or j not in p.world for j in CORE):
        return False
    shoulder_mid = (p.world["left_shoulder"] + p.world["right_shoulder"]) / 2
    hip_mid = (p.world["left_hip"] + p.world["right_hip"]) / 2
    if not (p.world["nose"][1] > shoulder_mid[1] > hip_mid[1]):
        return False
    if not p.has_depth:
        return True
    shoulder = float(np.linalg.norm(p.world["left_shoulder"] - p.world["right_shoulder"]))
    torso = float(np.linalg.norm(shoulder_mid - hip_mid))
    return SHOULDER_M[0] <= shoulder <= SHOULDER_M[1] and TORSO_M[0] <= torso <= TORSO_M[1]
