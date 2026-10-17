"""Orbbec Gemini 336L FrameSource (pyorbbecsdk2 / Orbbec SDK v2).

API 依據與來源見 docs/orbbec.md。重點：
- color：優先 MJPG（SDK 對 Gemini336L 的預設即 1280x720@30 MJPG），轉成 RGB uint8。
- depth：Y16，經 AlignFilter(COLOR_STREAM) 軟體 D2C 對齊到 color，再乘 get_depth_scale() (mm/單位) 轉公尺。
  BOOTH_ORBBEC_ALIGN=hw 改用相機硬體 D2C（省掉 CPU 對齊；這組 color profile 不支援就自動退回軟體）。
  BOOTH_ORBBEC_DEPTH=low 改用同 fps 最小解析度的 Y16（深度只用在關節附近取樣，解析度低軟體對齊比較快）。
- 內參：pipeline.get_camera_param().rgb_intrinsic（color 解析度）。
- 斷線：wait_for_frames 丟例外、連續 MAX_FRAME_ERRORS 幀轉換失敗、或超過 stall_s 沒有完整幀
  → 視為斷線，read() 回 None，每 retry_s 秒重開 pipeline。單一壞幀只丟掉那一幀。
- 內參取不到時 read() 回 None（寧可顯示維修畫面，也不讓 lift3d 默默用 60° FOV 估計），每 intr_retry_s 秒重取。
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from types import ModuleType
from typing import Any

import cv2
import numpy as np

from booth_tracker.paths import REPO_ROOT
from booth_tracker.perf import PERF
from booth_tracker.types import Frame, Intrinsics

log = logging.getLogger("booth_tracker.orbbec")

DOC_PATH = REPO_ROOT / "docs" / "orbbec.md"
WAIT_MS = 1000  # wait_for_frames timeout
START_TIMEOUT_S = 20.0  # pipeline.start 逾時（macOS 沒 root 時社群回報會卡住）
COLOR_ENV = "BOOTH_ORBBEC_COLOR"  # mjpg（預設）| rgb
ALIGN_ENV = "BOOTH_ORBBEC_ALIGN"  # sw（預設，CPU 上的 AlignFilter）| hw（相機硬體 D2C）
DEPTH_ENV = "BOOTH_ORBBEC_DEPTH"  # default（SDK 預設，336L 為 848x480）| low（同 fps 最小解析度的 Y16）
MAX_FRAME_ERRORS = 5  # 連續這麼多幀轉換失敗才當斷線重開
INTR_RETRY_S = 2.0  # 內參取不到時的重試間隔


class OrbbecSdkMissing(ImportError):
    """pyorbbecsdk 匯入失敗。訊息內含處理文件路徑。"""


def load_sdk() -> ModuleType:
    try:
        import pyorbbecsdk
    except ImportError as e:
        raise OrbbecSdkMissing(
            f"無法匯入 pyorbbecsdk（{e}）。安裝：uv sync --extra orbbec（或在遊戲專案依賴 booth-tracker[orbbec]）；"
            f"macOS 無 wheel 時的 source build 步驟見 {DOC_PATH}"
        ) from e
    return pyorbbecsdk


def macos_needs_sudo() -> bool:
    """macOS 的 UVCAssistant 會獨佔 UVC 裝置，沒有 root 時 SDK 常開不了相機。見 docs/orbbec.md。"""
    return sys.platform == "darwin" and hasattr(os, "geteuid") and os.geteuid() != 0


SUDO_HINT = (
    "macOS 上 UVCAssistant 會獨佔 UVC 相機，需用 root 執行："
    "sudo .venv/bin/booth-tracker doctor（不要用 sudo uv run）；詳見 " + str(DOC_PATH)
)


# --------------------------------------------------------------------------
# SDK 呼叫集中在下面這幾個 helper，測試用假 module 替換即可。
# --------------------------------------------------------------------------


def describe_profile(p: Any) -> str:
    """可讀的 profile 描述：1280x720 MJPG @30fps。取不到欄位就退回 str(p)。"""
    try:
        fmt = p.get_format()
        return f"{p.get_width()}x{p.get_height()} {getattr(fmt, 'name', fmt)} @{p.get_fps()}fps"
    except Exception:
        return str(p)


def _pick_profile(profile_list: Any, candidates: list[tuple[int, int, Any, int]]) -> Any:
    """依序嘗試 (w, h, format, fps)，0 表示不限。全部失敗就丟出含可用清單的錯誤。"""
    last: Exception | None = None
    for w, h, fmt, fps in candidates:
        try:
            return profile_list.get_video_stream_profile(w, h, fmt, fps)
        except Exception as e:  # SDK 找不到 profile 時丟 OBError
            last = e
    try:
        available = [describe_profile(profile_list[i]) for i in range(len(profile_list))]
    except Exception:
        available = []
    raise RuntimeError(f"找不到符合的 stream profile（{last}）。可用：{available}")


def _env_choice(env: str, choices: tuple[str, ...]) -> str:
    """環境變數選項，沒設就是第一個。"""
    pref = os.environ.get(env, choices[0]).strip().lower()
    if pref not in choices:
        raise ValueError(f"{env} 只能是 {' 或 '.join(choices)}，收到 {pref!r}")
    return pref


def color_preference() -> str:
    return _env_choice(COLOR_ENV, ("mjpg", "rgb"))


def _video_profiles(profile_list: Any) -> list[Any]:
    out = []
    for i in range(len(profile_list)):
        p = profile_list[i]
        as_video = getattr(p, "as_video_stream_profile", None)
        out.append(as_video() if as_video else p)
    return out


def _y16_by_area(sdk: Any, profiles: list[Any], fps: int) -> list[Any]:
    """fps 相符的 Y16 profile，依解析度由小到大。"""
    y16 = [p for p in profiles if p.get_format() == sdk.OBFormat.Y16 and p.get_fps() == fps]
    return sorted(y16, key=lambda p: p.get_width() * p.get_height())


def _hw_depth_profile(sdk: Any, pipeline: Any, color_profile: Any, fps: int, depth: str) -> Any | None:
    """硬體 D2C 能搭配這組 color profile 的 depth profile；不支援回 None，呼叫端退回軟體對齊。"""
    try:
        listed = pipeline.get_d2c_depth_profile_list(color_profile, sdk.OBAlignMode.HW_MODE)
        y16 = _y16_by_area(sdk, _video_profiles(listed), fps)
    except Exception as e:
        log.warning("取不到硬體 D2C 的 depth profile（%s），改用軟體對齊", e)
        return None
    if not y16:
        log.warning("color %s 沒有可搭配硬體 D2C 的 Y16 @%dfps depth profile，改用軟體對齊", describe_profile(color_profile), fps)
        return None
    return y16[0] if depth == "low" else y16[-1]


def _color_candidates(sdk: Any, w: int, h: int, fps: int, pref: str = "mjpg") -> list[tuple[int, int, Any, int]]:
    F = sdk.OBFormat
    fmts = [F.MJPG, F.RGB] if pref == "mjpg" else [F.RGB, F.MJPG]
    fmts += [F.YUYV, F.UNKNOWN_FORMAT]
    # 先要求指定 fps，找不到再放寬 fps
    return [(w, h, f, fps) for f in fmts] + [(w, h, f, 0) for f in fmts]


def _pick_depth_profile(sdk: Any, depth_list: Any, fps: int, depth: str = "default") -> Any:
    """先用 SDK 預設（336L 為 848x480 Y16；解析度小，軟體 D2C 才跑得動）；預設不是 Y16 才搜尋。
    depth="low"：改挑 fps 相符、解析度最小的 Y16；列舉不到就照預設規則。"""
    if depth == "low":
        try:
            y16 = _y16_by_area(sdk, _video_profiles(depth_list), fps)
        except Exception as e:
            log.warning("列舉 depth profile 失敗：%s", e)
            y16 = []
        if y16:
            return y16[0]
        log.warning("找不到 Y16 @%dfps 的 depth profile，改用預設", fps)
    try:
        default = depth_list.get_default_video_stream_profile()
        if default.get_format() == sdk.OBFormat.Y16:
            return default
    except Exception as e:
        log.warning("取不到預設 depth profile：%s", e)
    try:
        return _pick_profile(depth_list, [(0, 0, sdk.OBFormat.Y16, fps), (0, 0, sdk.OBFormat.Y16, 0)])
    except RuntimeError:
        return depth_list.get_default_video_stream_profile()


def _build_config(
    sdk: Any, pipeline: Any, w: int, h: int, fps: int, pref: str = "mjpg", align: str = "sw", depth: str = "default"
) -> tuple[Any, Any, Any, str]:
    """回傳 (config, color_profile, depth_profile, 實際的 align)。align="hw" 不支援時退回 "sw"。"""
    config = sdk.Config()
    color_list = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
    color_profile = _pick_profile(color_list, _color_candidates(sdk, w, h, fps, pref))
    depth_profile = _hw_depth_profile(sdk, pipeline, color_profile, fps, depth) if align == "hw" else None
    if depth_profile is not None:
        config.set_align_mode(sdk.OBAlignMode.HW_MODE)
    else:
        align = "sw"
        depth_list = pipeline.get_stream_profile_list(sdk.OBSensorType.DEPTH_SENSOR)
        depth_profile = _pick_depth_profile(sdk, depth_list, fps, depth)
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)
    # 每個 frameset 都要同時有 color + depth（官方 03_color_and_depth_aligned.py 的 SW 對齊寫法）
    config.set_frame_aggregate_output_mode(sdk.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    return config, color_profile, depth_profile, align


def _frameset_parts(frames: Any) -> tuple[Any, Any]:
    if not hasattr(frames, "get_color_frame") and hasattr(frames, "as_frame_set"):
        frames = frames.as_frame_set()
    return frames.get_color_frame(), frames.get_depth_frame()


def color_to_rgb(sdk: Any, frame: Any) -> np.ndarray:
    """SDK color frame → HxWx3 uint8 RGB。格式不支援或解碼失敗丟 ValueError。"""
    h, w = frame.get_height(), frame.get_width()
    fmt = frame.get_format()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    F = sdk.OBFormat
    if fmt == F.RGB:
        return data[: h * w * 3].reshape(h, w, 3).copy()
    if fmt == F.BGR:
        return data[: h * w * 3].reshape(h, w, 3)[:, :, ::-1].copy()
    if fmt == F.MJPG:
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("MJPG 解碼失敗")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if fmt == F.YUYV:
        return cv2.cvtColor(data[: h * w * 2].reshape(h, w, 2), cv2.COLOR_YUV2RGB_YUY2)
    if fmt == F.UYVY:
        return cv2.cvtColor(data[: h * w * 2].reshape(h, w, 2), cv2.COLOR_YUV2RGB_UYVY)
    planar = {F.NV12: cv2.COLOR_YUV2RGB_NV12, F.NV21: cv2.COLOR_YUV2RGB_NV21, F.I420: cv2.COLOR_YUV2RGB_I420}
    for f, code in planar.items():
        if fmt == f:
            return cv2.cvtColor(data[: h * w * 3 // 2].reshape(h * 3 // 2, w), code)
    raise ValueError(f"不支援的 color 格式：{fmt}")


def depth_to_meters(frame: Any) -> np.ndarray:
    """Y16 depth frame → HxW float32 公尺，0 = 無效。get_depth_scale() 單位是 mm / 原始值。"""
    h, w = frame.get_height(), frame.get_width()
    raw = np.frombuffer(frame.get_data(), dtype=np.uint16)[: h * w].reshape(h, w)
    return (raw.astype(np.float32) * (float(frame.get_depth_scale()) / 1000.0)).astype(np.float32)


def _scaled(c: Any, w: int, h: int) -> Intrinsics:
    sx, sy = w / c.width, h / c.height
    return Intrinsics(float(c.fx) * sx, float(c.fy) * sy, float(c.cx) * sx, float(c.cy) * sy, w, h)


def read_intrinsics(pipeline: Any, color_frame: Any, verbose: bool = True) -> Intrinsics | None:
    """優先用 pipeline.get_camera_param().rgb_intrinsic，備援 stream profile 內參。

    解析度與 color 幀一致 → 直接用；長寬比相同 → 按比例縮放；長寬比不同 → 回 None（寧可沒有也不給錯的）。
    verbose=False 時失敗不寫 log（重試期間避免洗版）。
    """
    h, w = color_frame.get_height(), color_frame.get_width()
    candidates = []
    try:
        candidates.append(pipeline.get_camera_param().rgb_intrinsic)
    except Exception as e:
        if verbose:
            log.warning("get_camera_param 失敗：%s", e)
    try:
        candidates.append(color_frame.get_stream_profile().as_video_stream_profile().get_intrinsic())
    except Exception as e:
        if verbose:
            log.warning("stream profile 內參取得失敗：%s", e)
    for c in candidates:
        if (int(c.width), int(c.height)) == (w, h):
            return Intrinsics(float(c.fx), float(c.fy), float(c.cx), float(c.cy), w, h)
    for c in candidates:
        if c.width > 0 and c.height > 0 and int(c.width) * h == int(c.height) * w:
            log.warning("內參解析度 %sx%s 與 color 幀 %sx%s 不同，已按比例縮放", c.width, c.height, w, h)
            return _scaled(c, w, h)
    got = [f"{c.width}x{c.height}" for c in candidates] or ["無"]
    if verbose:
        log.error("取不到可用的 color 內參：color 幀 %sx%s，SDK 內參解析度 %s（長寬比不同，不能縮放）", w, h, got
        )
    return None


# --------------------------------------------------------------------------


class OrbbecSource:
    name = "orbbec"
    has_depth = True

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        retry_s: float = 2.0,
        stall_s: float = 3.0,
        start_timeout_s: float = START_TIMEOUT_S,
        color_format: str | None = None,
        align: str | None = None,
        depth: str | None = None,
        intr_retry_s: float = INTR_RETRY_S,
        max_frame_errors: int = MAX_FRAME_ERRORS,
    ):
        self.width, self.height, self.fps = width, height, fps
        self._clock = clock
        self._sleep = sleep
        self._retry_s = retry_s
        self._stall_s = stall_s
        self._start_timeout_s = start_timeout_s
        self._color_pref = (color_format or color_preference()).lower()  # mjpg | rgb，見 BOOTH_ORBBEC_COLOR
        if self._color_pref not in ("mjpg", "rgb"):
            raise ValueError(f"color_format 只能是 mjpg 或 rgb，收到 {self._color_pref!r}")
        self._align_pref = align or _env_choice(ALIGN_ENV, ("sw", "hw"))
        self._depth_pref = depth or _env_choice(DEPTH_ENV, ("default", "low"))
        self._intr_retry_s = intr_retry_s
        self._max_frame_errors = max_frame_errors
        self._intr_failing = False  # 這一輪失敗是否已記錄過（每個失敗期只記一次）
        self._intr_next_try = float("-inf")
        self._frame_errors = 0  # 連續轉換失敗的幀數
        self._pending: tuple[threading.Thread, Any] | None = None  # 逾時而仍卡在 start 裡的 (thread, pipeline)
        self.start_timed_out = False
        self._sdk: Any = None
        self._pipeline: Any = None
        self._align: Any = None  # 軟體對齊的 AlignFilter；硬體 D2C 時為 None
        self._intrinsics: Intrinsics | None = None
        self._last_attempt = float("-inf")
        self._last_ok = 0.0
        self.last_error: str | None = None
        self.profiles: dict[str, str] = {}  # 開啟後的 color / depth profile 描述，給 doctor 印

    @property
    def connected(self) -> bool:
        return self._pipeline is not None

    def start(self) -> None:
        """匯入失敗會丟 OrbbecSdkMissing；相機開不了則記錄在 last_error，由 read() 每 2 秒重試。"""
        self._sdk = load_sdk()
        self._try_open()

    def stop(self) -> None:
        self._close()

    def read(self) -> Frame | None:
        if self._pipeline is None:
            if self._clock() - self._last_attempt < self._retry_s:
                self._sleep(0.1)  # 斷線期間不要讓呼叫端空轉
                return None
            self._try_open()
            if self._pipeline is None:
                return None
        try:
            with PERF.time("wait"):
                frames = self._pipeline.wait_for_frames(WAIT_MS)
        except Exception as e:
            self._drop(f"讀取失敗：{e}")
            return None
        try:
            frame = self._to_frame(frames) if frames is not None else None
            self._frame_errors = 0
        except Exception as e:  # 單一壞幀（buffer 不完整、align 失敗）只丟這一幀
            self._frame_errors += 1
            if self._frame_errors >= self._max_frame_errors:
                self._drop(f"連續 {self._frame_errors} 幀轉換失敗：{e}")
                return None
            log.warning("丟棄壞幀（連續第 %d 次）：%s", self._frame_errors, e)
            frame = None
        now = self._clock()
        if frame is not None:
            self._last_ok = now
        elif now - self._last_ok > self._stall_s:
            self._drop(f"超過 {self._stall_s:g} 秒沒有完整幀")
            return None
        if frame is not None and frame.intrinsics is None:
            return None  # 有幀但沒內參：當作沒有幀，不讓 lift3d 用估計值
        return frame

    # -- internals ---------------------------------------------------------

    def _to_frame(self, frames: Any) -> Frame | None:
        if self._align is None:
            aligned = frames  # 硬體 D2C：相機送來的就是對齊好的
        else:
            with PERF.time("align"):
                aligned = self._align.process(frames)
        if not aligned:
            return None
        color, depth = _frameset_parts(aligned)
        if not color or not depth:
            return None
        try:
            with PERF.time("decode"):
                rgb = color_to_rgb(self._sdk, color)
        except ValueError as e:
            log.warning("丟棄壞幀：%s", e)
            return None
        with PERF.time("depth"):
            depth_m = depth_to_meters(depth)
        if depth_m.shape != rgb.shape[:2]:  # 對齊後理應同尺寸；保險起見最近鄰縮放
            depth_m = cv2.resize(depth_m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        self._ensure_intrinsics(color)
        return Frame(rgb=rgb, depth_m=depth_m, intrinsics=self._intrinsics, ts=self._clock())

    def _ensure_intrinsics(self, color: Any) -> None:
        """內參取不到時每 intr_retry_s 秒重取；每個失敗期只寫一次 log。"""
        if self._intrinsics is not None:
            return
        now = self._clock()
        if now < self._intr_next_try:
            return
        self._intr_next_try = now + self._intr_retry_s
        self._intrinsics = read_intrinsics(self._pipeline, color, verbose=not self._intr_failing)
        if self._intrinsics is None:
            if not self._intr_failing:
                log.warning("color 內參取不到：每 %g 秒重試，期間不輸出幀（相機狀態會顯示未連線）", self._intr_retry_s)
            self._intr_failing = True
            self.last_error = "取不到 color 內參（get_camera_param / stream profile），重試中"
        elif self._intr_failing:
            log.info("color 內參已取得：fx=%.2f fy=%.2f", self._intrinsics.fx, self._intrinsics.fy)
            self._intr_failing = False
            self.last_error = None

    def _start_pipeline(self, pipeline: Any, config: Any) -> bool:
        """在 daemon thread 跑 pipeline.start。逾時回 False（thread 與 pipeline 留在 _pending）；失敗則重新丟出例外。"""
        errors: list[BaseException] = []

        def target() -> None:
            try:
                pipeline.start(config)
            except BaseException as e:
                errors.append(e)

        t = threading.Thread(target=target, daemon=True, name="orbbec-start")
        t.start()
        t.join(self._start_timeout_s)
        if t.is_alive():
            self._pending = (t, pipeline)
            return False
        if errors:
            raise errors[0]
        return True

    def _try_open(self) -> None:
        self._last_attempt = self._clock()
        if self._pending is not None:
            thread, old = self._pending
            if thread.is_alive():  # 上一次 start 還卡著，再開一個 pipeline 只會互搶相機
                self.last_error = "上一次 pipeline.start 仍未返回，暫不重開"
                return
            self._pending = None
            self._close_pipeline(old)  # start 後來才返回：關掉，免得它占著相機
        sdk = self._sdk
        pipeline = None
        try:
            pipeline = sdk.Pipeline()
            config, color_p, depth_p, align = _build_config(
                sdk, pipeline, self.width, self.height, self.fps, self._color_pref, self._align_pref, self._depth_pref
            )
            if not self._start_pipeline(pipeline, config):
                self.start_timed_out = True
                self.last_error = f"pipeline.start 超過 {self._start_timeout_s:g} 秒沒有回應"
                if macos_needs_sudo():
                    self.last_error += f"（{SUDO_HINT}）"
                log.error("Orbbec 開啟失敗：%s", self.last_error)
                return  # 不呼叫 stop：卡住的 pipeline 可能連 stop 也會卡
            self._align = sdk.AlignFilter(align_to_stream=sdk.OBStreamType.COLOR_STREAM) if align == "sw" else None
        except Exception as e:
            self.last_error = str(e) or type(e).__name__
            if macos_needs_sudo():
                self.last_error += f"（{SUDO_HINT}）"
            log.warning("Orbbec 開啟失敗：%s", self.last_error)
            self._close_pipeline(pipeline)
            self._pipeline = None
            return
        self._pipeline = pipeline
        self._intrinsics = None  # 每次重開都重取
        self._intr_failing = False
        self._intr_next_try = float("-inf")
        self._frame_errors = 0
        self.start_timed_out = False
        self._last_ok = self._clock()
        self.last_error = None
        self.profiles = {"color": describe_profile(color_p), "depth": describe_profile(depth_p), "align": align}
        log.info("Orbbec 已開啟：color=%s depth=%s align=%s", self.profiles["color"], self.profiles["depth"], align)

    def _drop(self, reason: str) -> None:
        log.warning("Orbbec 斷線（%s），%g 秒後重試", reason, self._retry_s)
        self.last_error = reason
        self._close()
        self._last_attempt = self._clock()

    def _close(self) -> None:
        p, self._pipeline = self._pipeline, None
        self._close_pipeline(p)

    @staticmethod
    def _close_pipeline(pipeline: Any) -> None:
        if pipeline is None:
            return
        try:
            pipeline.stop()
        except Exception:
            pass  # 裝置已斷線時 stop 也可能丟例外
