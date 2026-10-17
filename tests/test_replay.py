from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from booth_tracker.record import record
from booth_tracker.sources.replay import ReplaySource
from booth_tracker.types import Frame, Intrinsics

W, H = 64, 48
INTR = Intrinsics(fx=50.0, fy=50.0, cx=32.0, cy=24.0, width=W, height=H)


class FakeSource:
    name = "fake"

    def __init__(self, with_depth: bool = True, n: int = 3):
        self.has_depth = with_depth
        self._frames = []
        for i in range(n):
            rgb = np.full((H, W, 3), 40 * (i + 1), dtype=np.uint8)
            depth = np.full((H, W), 1.0 + 0.5 * i, dtype=np.float32) if with_depth else None
            self._frames.append(Frame(rgb=rgb, depth_m=depth, intrinsics=INTR, ts=100.0 + i * 0.01))
        self._i = 0

    def start(self) -> None:
        pass

    def read(self) -> Frame | None:
        if self._i >= len(self._frames):
            return None
        f = self._frames[self._i]
        self._i += 1
        return f

    def stop(self) -> None:
        pass


def _record(tmp_path: Path, with_depth: bool = True) -> Path:
    out = tmp_path / "rec"
    n = record(FakeSource(with_depth), out, seconds=0.3)
    assert n == 3
    return out


def test_record_writes_meta_and_frames(tmp_path: Path):
    out = _record(tmp_path)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["has_depth"] is True
    assert meta["intrinsics"]["fx"] == 50.0
    assert meta["fps"] > 0
    assert sorted(p.name for p in (out / "frames").iterdir()) == ["000000.npz", "000001.npz", "000002.npz"]


def test_roundtrip_no_loop(tmp_path: Path):
    out = _record(tmp_path)
    src = ReplaySource(out, loop=False, realtime=False)
    assert src.has_depth is True
    src.start()
    frames = [src.read() for _ in range(3)]
    assert all(f is not None for f in frames)
    for i, f in enumerate(frames):
        assert f.rgb.shape == (H, W, 3)
        assert abs(int(f.rgb[0, 0, 0]) - 40 * (i + 1)) <= 3  # JPEG 有損
        assert f.depth_m is not None and f.depth_m.dtype == np.float32
        assert np.abs(f.depth_m - (1.0 + 0.5 * i)).max() <= 0.001
        assert f.intrinsics == INTR
    assert src.read() is None
    src.stop()


def test_loop_returns_to_first_frame(tmp_path: Path):
    out = _record(tmp_path)
    src = ReplaySource(out, loop=True, realtime=False)
    src.start()
    firsts = [src.read() for _ in range(4)]
    assert firsts[3] is not None
    assert np.array_equal(firsts[3].rgb, firsts[0].rgb)
    assert firsts[3].ts > firsts[2].ts  # loop 後 ts 仍單調遞增


def test_no_depth(tmp_path: Path):
    out = _record(tmp_path, with_depth=False)
    src = ReplaySource(out, loop=False, realtime=False)
    assert src.has_depth is False
    src.start()
    f = src.read()
    assert f is not None and f.depth_m is None


def test_realtime_sleeps(tmp_path: Path):
    out = _record(tmp_path)
    src = ReplaySource(out, loop=False, realtime=True)
    src.start()
    t0 = time.monotonic()
    for _ in range(3):
        assert src.read() is not None
    assert time.monotonic() - t0 >= 0.015  # 錄的間隔 0.01 s × 2


def test_record_zero_frames_raises_and_writes_no_meta(tmp_path: Path):
    out = tmp_path / "empty"
    with pytest.raises(RuntimeError, match="沒有收到任何幀"):
        record(FakeSource(n=0), out, seconds=0.1)
    assert not (out / "meta.json").exists()


def test_replay_corrupt_jpeg_names_file(tmp_path: Path):
    out = _record(tmp_path)
    bad = out / "frames" / "000000.npz"
    np.savez(bad, rgb_jpeg=np.zeros(10, dtype=np.uint8), ts=100.0)
    src = ReplaySource(out, loop=False, realtime=False)
    src.start()
    with pytest.raises(ValueError, match="000000.npz"):
        src.read()
