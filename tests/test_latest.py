import threading
import time

import numpy as np
import pytest

from booth_tracker.perf import Perf
from booth_tracker.sources.latest import LatestFrameSource
from booth_tracker.types import Frame


def _frame(i: int) -> Frame:
    return Frame(rgb=np.zeros((2, 2, 3), np.uint8), depth_m=None, intrinsics=None, ts=float(i))


class CountingSource:
    """每 period 秒產生一幀（ts = 序號），像相機一樣不等消費端。"""

    name = "fake"
    has_depth = False

    def __init__(self, period=0.005, error_at=None):
        self.period, self.error_at = period, error_at
        self.i = 0
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def read(self):
        time.sleep(self.period)
        self.i += 1
        if self.error_at is not None and self.i >= self.error_at:
            raise RuntimeError("相機炸了")
        return _frame(self.i)

    def stop(self):
        self.stopped = True


def test_慢消費端拿到的是最新幀而不是積壓的舊幀():
    inner = CountingSource(period=0.005)
    src = LatestFrameSource(inner)
    src.start()
    try:
        first = src.read()
        time.sleep(0.1)  # 消費端卡住 100 ms，期間產生 ~20 幀
        latest = src.read()
        assert latest.ts - first.ts >= 10
        assert latest.ts >= inner.i - 2  # 跟相機最新的只差一兩幀
    finally:
        src.stop()
    assert inner.started and inner.stopped


def test_同一幀不會讀兩次_沒新幀逾時回None():
    class OneShot(CountingSource):
        def read(self):
            if self.i:
                time.sleep(0.01)
                return None
            self.i = 1
            return _frame(1)

    src = LatestFrameSource(OneShot(), wait_s=0.1)
    src.start()
    try:
        assert src.read().ts == 1.0
        t = time.monotonic()
        assert src.read() is None
        assert time.monotonic() - t >= 0.09
    finally:
        src.stop()


def test_讀幀thread的例外在read丟出():
    src = LatestFrameSource(CountingSource(period=0.001, error_at=3))
    src.start()
    try:
        with pytest.raises(RuntimeError, match="相機炸了"):
            for _ in range(10):
                src.read()
    finally:
        src.stop()


def test_stop會叫醒等待中的read():
    src = LatestFrameSource(CountingSource(period=0.3), wait_s=5)
    src.start()
    out = []
    t = threading.Thread(target=lambda: out.append(src.read()))
    t.start()
    time.sleep(0.05)
    src.stop()
    t.join(1)
    assert not t.is_alive() and out == [None]


def test_perf每段時間印一行中位數與頻率():
    now = [0.0]
    lines = []
    perf = Perf(5, clock=lambda: now[0], emit=lines.append)
    for ms in (1.0, 2.0, 30.0):
        perf.add("detect", ms)
    for _ in range(50):
        perf.count("step")
    assert lines == []
    now[0] = 5.0
    perf.count("step")
    assert lines == ["perf 5s | step 10.2Hz | detect 2.0 ms"]
    perf.add("detect", 9.0)  # 新的一輪重新累計
    now[0] = 10.0
    perf.add("detect", 3.0)
    assert lines[-1] == "perf 5s | detect 6.0 ms"


def test_perf關閉時不記錄():
    lines = []
    perf = Perf(0, clock=lambda: 1e9, emit=lines.append)
    perf.add("x", 1.0)
    perf.count("y")
    assert lines == [] and perf._ms == {} and perf._counts == {}
