"""各階段耗時統計：每 BOOTH_PERF_LOG_S 秒（預設 5，0 = 關）寫一行 log，實機找瓶頸用。

  perf 5s | camera 30.0Hz step 29.8Hz sent 29.8Hz | wait 2.1 align 8.3 decode 1.6 depth 0.6 read 12.9 age 3.2 detect 7.8 post 0.4 render 2.0 ms

Hz = 該計數器每秒次數；ms = 該階段中位數。任何 thread 都可以呼叫。
"""

from __future__ import annotations

import logging
import os
import statistics
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

log = logging.getLogger("booth_tracker.perf")


class Perf:
    def __init__(
        self,
        log_s: float,
        clock: Callable[[], float] = time.monotonic,
        emit: Callable[[str], None] = log.info,
    ):
        self.log_s = log_s
        self._clock = clock
        self._emit = emit
        self._lock = threading.Lock()
        self._ms: dict[str, list[float]] = {}
        self._counts: dict[str, int] = {}
        self._since = clock()

    def add(self, stage: str, ms: float) -> None:
        if self.log_s <= 0:
            return
        with self._lock:
            self._ms.setdefault(stage, []).append(ms)
        self._maybe_flush()

    def count(self, name: str) -> None:
        if self.log_s <= 0:
            return
        with self._lock:
            self._counts[name] = self._counts.get(name, 0) + 1
        self._maybe_flush()

    @contextmanager
    def time(self, stage: str) -> Iterator[None]:
        t = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, (time.perf_counter() - t) * 1000)

    def _maybe_flush(self) -> None:
        now = self._clock()
        with self._lock:
            elapsed = now - self._since
            if elapsed < self.log_s:
                return
            ms, counts = self._ms, self._counts
            self._ms, self._counts, self._since = {}, {}, now
        hz = " ".join(f"{k} {v / elapsed:.1f}Hz" for k, v in counts.items())
        stages = " ".join(f"{k} {statistics.median(v):.1f}" for k, v in ms.items())
        parts = [f"perf {elapsed:.0f}s", hz, f"{stages} ms" if stages else ""]
        self._emit(" | ".join(p for p in parts if p))


PERF = Perf(float(os.environ.get("BOOTH_PERF_LOG_S", "5")))
