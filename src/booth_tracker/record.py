"""錄製 FrameSource 到 data/recordings/<name>/ 格式（meta.json + frames/NNNNNN.npz）。"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from booth_tracker.sources.base import FrameSource


def record(source: FrameSource, out_dir: Path, seconds: float) -> int:
    """錄 `seconds` 秒，回傳幀數。source 的 start()/stop() 由呼叫端負責。"""
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    first_ts = last_ts = 0.0
    intrinsics = None
    t_end = time.monotonic() + seconds
    next_report = time.monotonic() + 1.0

    while time.monotonic() < t_end:
        frame = source.read()
        if frame is None:
            time.sleep(0.005)
            continue
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        if not ok:
            raise RuntimeError("JPEG 編碼失敗")
        data: dict[str, np.ndarray | float] = {"rgb_jpeg": buf.reshape(-1), "ts": float(frame.ts)}
        if frame.depth_m is not None:
            data["depth_mm"] = (frame.depth_m * 1000).astype(np.uint16)
        np.savez(frames_dir / f"{n:06d}.npz", **data)

        if n == 0:
            first_ts = frame.ts
            intrinsics = frame.intrinsics
        last_ts = frame.ts
        n += 1
        if time.monotonic() >= next_report:
            print(f"錄製中：{n} 幀，剩 {max(0.0, t_end - time.monotonic()):.1f} 秒")
            next_report += 1.0

    if n == 0:
        raise RuntimeError(f"錄製失敗：{seconds} 秒內沒有收到任何幀（{source.name}），未寫入 meta.json")

    fps = (n - 1) / (last_ts - first_ts) if n > 1 and last_ts > first_ts else 30.0
    meta = {
        "fps": fps,
        "has_depth": bool(source.has_depth),
        "intrinsics": asdict(intrinsics) if intrinsics else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"錄製完成：{n} 幀，fps={fps:.1f}，輸出 {out_dir}")
    return n
