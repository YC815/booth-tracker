"""One Euro filter（Casiez et al. 2012），向量版。spec §4.4。"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: np.ndarray | None = None
        self._dx = np.zeros(0)
        self._t = 0.0

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._x is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = t - self._t
        if dt <= 0:  # 時間重複或倒退：不更新，回上一個輸出
            return self._x.copy()
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x.copy()


class JointFilter:
    def __init__(self, min_cutoff: float, beta: float):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self._filters: dict[str, OneEuroFilter] = {}

    def __call__(self, joints: dict[str, np.ndarray], t: float) -> dict[str, np.ndarray]:
        out = {}
        for name, p in joints.items():
            f = self._filters.get(name)
            if f is None:
                f = self._filters[name] = OneEuroFilter(self.min_cutoff, self.beta)
            out[name] = f(p, t)
        return out

    def reset(self) -> None:
        self._filters.clear()
