"""Orbbec Gemini 336L 診斷。逐項印 [PASS] / [FAIL] 原因；第一個 FAIL 之後的項目標示 [SKIP]。

用法（macOS 需 root，原因見 docs/orbbec.md）：
    sudo .venv/bin/booth-tracker doctor
無相機的機器上：uv run --extra orbbec booth-tracker doctor
"""

from __future__ import annotations

import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from booth_tracker.paths import DATA_DIR
from booth_tracker.sources.orbbec import DOC_PATH, SUDO_HINT, START_TIMEOUT_S, OrbbecSdkMissing, OrbbecSource, load_sdk, macos_needs_sudo
from booth_tracker.types import Frame

FPS_FRAMES = 60
FPS_TIMEOUT_S = 30.0
MIN_FPS_RATIO = 0.8
CENTER_HALF = 10  # 21x21
MIN_VALID_RATIO = 0.5
MIN_FIRMWARE = (1, 2, 20)  # Gemini330 / 336L 的最低支援韌體（OrbbecSDK_v2 README）


class Fail(Exception):
    """某一項檢查失敗；訊息即印出的原因。"""


@dataclass
class Ctx:
    sdk: Any = None
    source: OrbbecSource | None = None
    frames: list[Frame] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hung: bool = False  # SDK 呼叫卡死，結束時要 os._exit


def _check_import(ctx: Ctx) -> str:
    try:
        ctx.sdk = load_sdk()
    except OrbbecSdkMissing as e:
        raise Fail(str(e)) from e
    return f"pyorbbecsdk（Orbbec SDK {ctx.sdk.get_version()}）@ {Path(ctx.sdk.__file__).parent}"


def _parse_version(s: str) -> tuple[int, ...] | None:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", s)
    return tuple(int(x) for x in m.groups()) if m else None


def _check_devices(ctx: Ctx) -> str:
    # Context 要活到列舉完（提早釋放 → NULL deviceMgr），但不能活過這個函式（佔住相機 → 第 3 項 uvc_open -3）
    context = ctx.sdk.Context()
    devices = context.query_devices()
    n = devices.get_count()
    if n == 0:
        hints = [
            "SDK 列舉不到任何 Orbbec 裝置。依序檢查：",
            "  1. 線材／埠：必須是 USB 3.x 資料線接主機 USB-C 埠，避開 hub（USB 2.0 辨識可能要 20 秒以上）。",
            "  2. 是否有別的程式佔用相機（OrbbecViewer、上一個沒關的遊戲程序）：先關掉再試。",
            "  3. 拔插相機後等 30 秒再跑一次。",
        ]
        if sys.platform == "darwin":
            hints.append(f"  4. {SUDO_HINT}")
        raise Fail("\n".join(hints))
    lines = []
    for i in range(n):
        info = devices.get_device_by_index(i).get_device_info()
        name, sn, fw = info.get_name(), info.get_serial_number(), info.get_firmware_version()
        conn = info.get_connection_type()
        lines.append(f"[{i}] {name}  SN={sn}  韌體={fw}  PID=0x{info.get_pid():04x}  連線={conn}")
        if "336" not in name:
            ctx.warnings.append(f"裝置名稱「{name}」不是 Gemini 336L，流程仍會嘗試使用第一台裝置。")
        if re.search(r"USB\s*2", conn, re.I):
            ctx.warnings.append(f"連線為 {conn}：頻寬不足，請換 USB 3.x 線與埠。")
        v = _parse_version(fw)
        if v and v < MIN_FIRMWARE:
            ctx.warnings.append(f"韌體 {fw} 低於官方最低支援 {'.'.join(map(str, MIN_FIRMWARE))}，請用 OrbbecViewer 更新。")
    return "\n".join(lines)


def _check_open(ctx: Ctx) -> str:
    src = OrbbecSource()
    ctx.source = src
    src.start()  # 內部對 pipeline.start 有逾時保護
    if src.start_timed_out:
        ctx.hung = True  # 卡住的 SDK thread 會擋住直譯器正常結束
        raise Fail(src.last_error or f"pipeline.start 超過 {START_TIMEOUT_S:g} 秒沒有回應")
    if not src.connected:
        raise Fail(f"開啟 color + depth 失敗：{src.last_error}")
    return f"color={src.profiles['color']}\ndepth={src.profiles['depth']}"


def _check_fps(ctx: Ctx) -> str:
    src = ctx.source
    assert src is not None
    start = time.monotonic()
    times: list[float] = []
    empty = 0
    while len(times) < FPS_FRAMES and time.monotonic() - start < FPS_TIMEOUT_S:
        f = src.read()
        if f is None:
            empty += 1
            continue
        if f.depth_m is None:
            raise Fail("Frame.depth_m 是 None：depth 沒有對齊進來。")
        times.append(time.monotonic())
        ctx.frames.append(f)
        del ctx.frames[:-1]  # 只留最新一幀
    if len(times) < FPS_FRAMES:
        raise Fail(
            f"{FPS_TIMEOUT_S:g} 秒內只收到 {len(times)}/{FPS_FRAMES} 幀（逾時/空幀 {empty} 次）。"
            f"最後錯誤：{src.last_error or '無（wait_for_frames 持續逾時）'}。"
            "常見原因：USB 2.0 線/埠、頻寬被其他裝置占用、macOS 未用 sudo。"
        )
    fps = (len(times) - 1) / (times[-1] - times[0])
    f = ctx.frames[-1]
    detail = (
        f"{fps:.1f} fps（{FPS_FRAMES} 幀，逾時/空幀 {empty} 次）；"
        f"rgb {f.rgb.shape[1]}x{f.rgb.shape[0]}，depth {f.depth_m.shape[1]}x{f.depth_m.shape[0]}"
    )
    if fps < src.fps * MIN_FPS_RATIO:
        raise Fail(f"{detail}\n  fps 低於目標 {src.fps} 的 {MIN_FPS_RATIO:.0%}：確認 USB 3.x、關掉其他吃 CPU 的程式（軟體 D2C 對齊跑在 CPU 上）。")
    return detail


def _check_intrinsics(ctx: Ctx) -> str:
    f = ctx.frames[-1]
    k = f.intrinsics
    if k is None:
        raise Fail("Frame.intrinsics 是 None：get_camera_param / stream profile 取不到 color 內參，或其長寬比與 color 幀不同（不能縮放）。看上方 log。")
    h, w = f.rgb.shape[:2]
    if (k.width, k.height) != (w, h):
        raise Fail(f"內參解析度 {k.width}x{k.height} 與 color 影像 {w}x{h} 不符，反投影會錯。")
    if not (k.fx > 0 and k.fy > 0 and 0 < k.cx < w and 0 < k.cy < h):
        raise Fail(f"內參數值不合理：fx={k.fx} fy={k.fy} cx={k.cx} cy={k.cy}")
    hfov = math.degrees(2 * math.atan(w / (2 * k.fx)))
    return f"fx={k.fx:.2f} fy={k.fy:.2f} cx={k.cx:.2f} cy={k.cy:.2f}（{k.width}x{k.height}，水平 FOV 約 {hfov:.1f}°）"


def _check_center_depth(ctx: Ctx) -> str:
    d = ctx.frames[-1].depth_m
    cy, cx = d.shape[0] // 2, d.shape[1] // 2
    patch = d[cy - CENTER_HALF : cy + CENTER_HALF + 1, cx - CENTER_HALF : cx + CENTER_HALF + 1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    ratio = valid.size / patch.size
    if ratio < MIN_VALID_RATIO:
        med = f"{np.median(valid):.3f} m" if valid.size else "無"
        raise Fail(
            f"畫面中心 21x21 有效深度只有 {ratio:.0%}（中位數 {med}）。"
            "請在鏡頭正前方 0.5–3 m 放人或物體再試；仍為 0 則檢查：太近(<0.3 m)、黑色/反光表面、D2C 對齊是否正常。"
        )
    return f"有效 {ratio:.0%}（{valid.size}/{patch.size}），中位數 {np.median(valid):.3f} m"


def _save_overlay(ctx: Ctx) -> str:
    f = ctx.frames[-1]
    bgr = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR)
    depth = f.depth_m
    vis = cv2.applyColorMap((np.clip(depth / 5.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis[depth <= 0] = 0  # 無效深度顯示黑色
    if vis.shape[:2] != bgr.shape[:2]:
        vis = cv2.resize(vis, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = cv2.addWeighted(bgr, 0.5, vis, 0.5, 0)
    path = Path(DATA_DIR) / "doctor.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), out):
        raise Fail(f"寫入 {path} 失敗")
    sudo_uid = os.environ.get("SUDO_UID")
    if sudo_uid and hasattr(os, "chown") and os.geteuid() == 0:
        os.chown(path, int(sudo_uid), int(os.environ.get("SUDO_GID", sudo_uid)))  # 不要留下 root 擁有的檔案
    return f"{path}（color 與 depth colormap 各 50%，0–5 m，無效深度為黑）"


STEPS: list[tuple[str, Callable[[Ctx], str]]] = [
    ("匯入 pyorbbecsdk", _check_import),
    ("列出裝置", _check_devices),
    ("開啟 color + depth", _check_open),
    (f"連續 {FPS_FRAMES} 幀實測 fps", _check_fps),
    ("color 內參", _check_intrinsics),
    ("中心 21x21 深度", _check_center_depth),
    ("存 doctor.png", _save_overlay),
]


def _emit(tag: str, i: int, name: str, detail: str = "") -> None:
    head = f"[{tag}] {i}. {name}"
    if not detail:
        print(head)
        return
    first, *rest = detail.split("\n")
    print(f"{head}: {first}")
    for line in rest:
        print(f"      {line}")


def run() -> int:
    """0 = 全 PASS；1 = 有項目 FAIL。不會丟 traceback。"""
    ctx = Ctx()
    if macos_needs_sudo():
        print(f"[WARN] 目前不是 root。{SUDO_HINT}")
    failed = False
    try:
        for i, (name, step) in enumerate(STEPS, 1):
            if failed:
                _emit("SKIP", i, name)
                continue
            try:
                _emit("PASS", i, name, step(ctx))
            except Fail as e:
                _emit("FAIL", i, name, str(e))
                failed = True
            except Exception as e:  # SDK 的 OBError 等：只印原因，不印 traceback
                _emit("FAIL", i, name, f"{type(e).__name__}: {e}")
                failed = True
    finally:
        if ctx.source is not None and not ctx.hung:
            ctx.source.stop()
    for w in ctx.warnings:
        print(f"[WARN] {w}")
    print("結果：全部通過" if not failed else f"結果：有項目失敗（排錯文件：{DOC_PATH}）")
    if ctx.hung:  # 卡死的 SDK thread 會擋住直譯器正常結束
        sys.stdout.flush()
        os._exit(1)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
