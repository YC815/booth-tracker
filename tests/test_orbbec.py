"""OrbbecSource 與 doctor 的測試。以假的 pyorbbecsdk module 注入 sys.modules，不需要真機。

假 SDK 的 API 形狀照 pyorbbecsdk2 2.1.2 內附的 pyorbbecsdk.pyi 與官方 examples 寫成（見 docs/orbbec.md）。
"""

from __future__ import annotations

import enum
import logging
import sys
import threading
import types
import weakref
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytest

from booth_tracker import doctor
from booth_tracker.sources import orbbec
from booth_tracker.sources.orbbec import OrbbecSdkMissing, OrbbecSource


class FakeOBError(Exception):
    pass


class OBFormat(enum.Enum):
    RGB = 1
    BGR = 2
    MJPG = 3
    YUYV = 4
    UYVY = 5
    NV12 = 6
    NV21 = 7
    I420 = 8
    Y16 = 9
    UNKNOWN_FORMAT = 0


class OBSensorType(enum.Enum):
    COLOR_SENSOR = 2
    DEPTH_SENSOR = 3


class OBStreamType(enum.Enum):
    COLOR_STREAM = 2
    DEPTH_STREAM = 3


class OBFrameAggregateOutputMode(enum.Enum):
    FULL_FRAME_REQUIRE = 0


class FakeProfile:
    def __init__(self, w, h, fmt, fps):
        self.w, self.h, self.fmt, self.fps = w, h, fmt, fps

    def get_width(self): return self.w
    def get_height(self): return self.h
    def get_format(self): return self.fmt
    def get_fps(self): return self.fps


class FakeColor:
    def __init__(self, w, h, fmt, data: bytes):
        self.w, self.h, self.fmt, self.data = w, h, fmt, data

    def get_width(self): return self.w
    def get_height(self): return self.h
    def get_format(self): return self.fmt
    def get_data(self): return self.data


class FakeDepth:
    def __init__(self, w, h, raw: np.ndarray, scale: float):
        self.w, self.h, self.raw, self.scale = w, h, raw, scale

    def get_width(self): return self.w
    def get_height(self): return self.h
    def get_data(self): return self.raw.tobytes()
    def get_depth_scale(self): return self.scale


class FakeFrameSet:
    def __init__(self, color, depth):
        self.color, self.depth = color, depth

    def get_color_frame(self): return self.color
    def get_depth_frame(self): return self.depth


@dataclass
class Intr:
    fx: float = 900.0
    fy: float = 901.0
    cx: float = 640.0
    cy: float = 360.0
    width: int = 1280
    height: int = 720


@dataclass
class World:
    """假 SDK 的可操控狀態。"""

    w: int = 1280
    h: int = 720
    color_fmt: OBFormat = OBFormat.RGB
    raw_value: int = 2000
    scale: float = 1.0
    devices: int = 1
    starts: int = 0
    stops: int = 0
    start_error: Exception | None = None
    script: list = field(default_factory=list)  # 每次 wait_for_frames 依序取：Exception 就丟、None 回 None、否則回該 FrameSet
    requested: list = field(default_factory=list)  # color 端 get_video_stream_profile 的呼叫
    depth_searches: list = field(default_factory=list)  # depth（Y16）端的搜尋呼叫；預設 profile 命中時應為空
    depth_default_fmt: OBFormat = OBFormat.Y16
    block_next_start: threading.Event | None = None  # 下一次 pipeline.start 會卡到這個 event 被 set
    ctx_refs: list = field(default_factory=list)
    ctx_alive_at_start: bool | None = None
    camera_param_calls: int = 0
    intr: Intr = field(default_factory=Intr)

    def default_frameset(self):
        if self.color_fmt == OBFormat.MJPG:
            img = np.full((self.h, self.w, 3), (10, 200, 30), np.uint8)  # BGR
            data = cv2.imencode(".jpg", img)[1].tobytes()
        else:
            data = np.full((self.h, self.w, 3), (200, 100, 50), np.uint8).tobytes()
        raw = np.full((self.h, self.w), self.raw_value, np.uint16)
        return FakeFrameSet(
            FakeColor(self.w, self.h, self.color_fmt, data), FakeDepth(self.w, self.h, raw, self.scale)
        )


def make_sdk(world: World) -> types.ModuleType:
    m = types.ModuleType("pyorbbecsdk")
    m.__file__ = "/fake/pyorbbecsdk/__init__.py"
    m.OBError, m.OBFormat, m.OBSensorType = FakeOBError, OBFormat, OBSensorType
    m.OBStreamType, m.OBFrameAggregateOutputMode = OBStreamType, OBFrameAggregateOutputMode
    m.get_version = lambda: "9.9.9"

    class ProfileList:
        def get_video_stream_profile(self, w=0, h=0, fmt=OBFormat.UNKNOWN_FORMAT, fps=0):
            if fmt == OBFormat.Y16:
                world.depth_searches.append((w, h, fmt, fps))
                return FakeProfile(1280, 800, OBFormat.Y16, fps or 30)  # 搜尋會挑到比預設大的解析度
            world.requested.append((w, h, fmt, fps))
            if fmt not in (OBFormat.RGB, OBFormat.MJPG, OBFormat.UNKNOWN_FORMAT):
                raise FakeOBError("no such profile")
            if fmt == OBFormat.UNKNOWN_FORMAT:
                fmt = world.color_fmt
            return FakeProfile(w, h, fmt, fps or 30)

        def get_default_video_stream_profile(self):
            return FakeProfile(848, 480, world.depth_default_fmt, 30)

        def __len__(self): return 1
        def __getitem__(self, i): return FakeProfile(1280, 720, OBFormat.MJPG, 30)

    class Config:
        def enable_stream(self, p): pass
        def set_frame_aggregate_output_mode(self, mode): pass

    class Pipeline:
        def __init__(self): pass
        def get_stream_profile_list(self, sensor): return ProfileList()

        def start(self, config=None):
            world.starts += 1
            world.ctx_alive_at_start = any(r() is not None for r in world.ctx_refs)
            if world.block_next_start is not None:
                ev, world.block_next_start = world.block_next_start, None
                ev.wait(10)
            if world.start_error:
                raise world.start_error

        def stop(self): world.stops += 1

        def wait_for_frames(self, timeout_ms):
            item = world.script.pop(0) if world.script else world.default_frameset()
            if isinstance(item, Exception):
                raise item
            return item

        def get_camera_param(self):
            world.camera_param_calls += 1
            return types.SimpleNamespace(rgb_intrinsic=world.intr)

    class AlignFilter:
        def __init__(self, align_to_stream): assert align_to_stream == OBStreamType.COLOR_STREAM
        def process(self, frames): return frames

    class DeviceInfo:
        def get_name(self): return "Orbbec Gemini 336L"
        def get_serial_number(self): return "CP1234567"
        def get_firmware_version(self): return "1.4.60"
        def get_pid(self): return 0x0803
        def get_connection_type(self): return "USB3.2"

    class Device:
        def get_device_info(self): return DeviceInfo()

    class DeviceList:
        def get_count(self): return world.devices
        def get_device_by_index(self, i): return Device()

    class Context:
        def __init__(self): world.ctx_refs.append(weakref.ref(self))
        def query_devices(self): return DeviceList()

    m.Config, m.Pipeline, m.AlignFilter, m.Context = Config, Pipeline, AlignFilter, Context
    return m


class Clock:
    def __init__(self): self.t = 100.0
    def __call__(self): return self.t


@pytest.fixture
def world(monkeypatch):
    w = World()
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", make_sdk(w))
    return w


@pytest.fixture
def clock():
    return Clock()


def make_source(clock, **kw):
    return OrbbecSource(clock=clock, sleep=lambda s: None, **kw)


# -- 1. depth ---------------------------------------------------------------


def test_depth_raw_2000_scale_1_is_2_meters(world, clock):
    src = make_source(clock)
    src.start()
    f = src.read()
    assert f is not None and f.depth_m is not None
    assert f.depth_m.dtype == np.float32
    assert f.depth_m.shape == (720, 1280)
    assert np.all(f.depth_m == 2.0)


def test_depth_scale_other_than_1_and_zero_stays_invalid(world, clock):
    world.scale = 0.1  # 0.1 mm / 單位：20000 -> 2.0 m
    world.raw_value = 20000
    src = make_source(clock)
    src.start()
    assert np.allclose(src.read().depth_m, 2.0)
    world.raw_value = 0
    assert np.all(src.read().depth_m == 0.0)


# -- 2. color ---------------------------------------------------------------


def test_color_raw_rgb_shape_and_values(world, clock):
    src = make_source(clock)
    src.start()
    f = src.read()
    assert f.rgb.shape == (720, 1280, 3) and f.rgb.dtype == np.uint8
    assert tuple(f.rgb[0, 0]) == (200, 100, 50)


def test_color_mjpg_decoded_to_rgb_order(world, clock):
    world.color_fmt = OBFormat.MJPG
    src = make_source(clock)
    src.start()
    f = src.read()
    assert f.rgb.shape == (720, 1280, 3)
    r, g, b = (int(x) for x in f.rgb[360, 640])
    assert g > r and g > b  # 來源 BGR=(10,200,30) -> RGB 的 G 最大


def test_intrinsics_from_camera_param(world, clock):
    src = make_source(clock)
    src.start()
    k = src.read().intrinsics
    assert (k.fx, k.fy, k.cx, k.cy, k.width, k.height) == (900.0, 901.0, 640.0, 360.0, 1280, 720)


def test_prefers_mjpg_at_requested_resolution(world, clock):
    make_source(clock).start()
    assert world.requested[0][:2] == (1280, 720) and world.requested[0][2] == OBFormat.MJPG


# -- fix round 1 ------------------------------------------------------------


def test_depth_uses_sdk_default_profile_not_search(world, clock):
    src = make_source(clock)
    src.start()
    assert world.depth_searches == []
    assert src.profiles["depth"] == "848x480 Y16 @30fps"


def test_depth_searches_only_when_default_is_not_y16(world, clock):
    world.depth_default_fmt = OBFormat.UNKNOWN_FORMAT
    src = make_source(clock)
    src.start()
    assert world.depth_searches and src.profiles["depth"] == "1280x800 Y16 @30fps"


def test_profile_description_is_readable(world, clock):
    src = make_source(clock)
    src.start()
    assert src.profiles["color"] == "1280x720 MJPG @30fps"


def test_intrinsics_scaled_when_same_aspect(world, clock):
    world.intr = Intr(fx=1350.0, fy=1351.5, cx=960.0, cy=540.0, width=1920, height=1080)
    src = make_source(clock)
    src.start()
    k = src.read().intrinsics
    assert (k.width, k.height) == (1280, 720)
    assert k.fx == pytest.approx(900.0) and k.fy == pytest.approx(901.0)
    assert k.cx == pytest.approx(640.0) and k.cy == pytest.approx(360.0)


def test_intrinsics_missing_yields_no_frame_and_logs_once(world, clock, caplog):
    """I4：有 depth 但沒內參 → read() 回 None（不讓 lift3d 用 60° 估計）；整個失敗期只記一次 ERROR。"""
    world.intr = Intr(width=1280, height=800)  # 16:10 vs 16:9，不能縮放
    src = make_source(clock)
    src.start()
    with caplog.at_level(logging.ERROR, logger="booth_tracker.orbbec"):
        frames = []
        for _ in range(5):
            frames.append(src.read())
            clock.t += 1.0
    assert frames == [None] * 5
    assert src.connected  # 幀照常進來，不算斷線、不重開
    assert world.starts == 1
    assert len([r for r in caplog.records if r.levelno == logging.ERROR]) == 1
    assert "內參" in src.last_error


def test_intrinsics_retried_every_2s_until_available(world, clock):
    world.intr = Intr(width=1280, height=800)
    src = make_source(clock)
    src.start()
    assert src.read() is None and world.camera_param_calls == 1
    clock.t += 1.0
    assert src.read() is None and world.camera_param_calls == 1  # 未滿 2 秒不重取
    world.intr = Intr()  # SDK 恢復
    clock.t += 1.0
    f = src.read()
    assert world.camera_param_calls == 2
    assert f is not None and f.intrinsics is not None and f.intrinsics.fx == 900.0
    assert src.last_error is None
    src.read()
    assert world.camera_param_calls == 2  # 取到之後不再每幀重取


def test_intrinsics_new_failure_episode_logs_again_after_reopen(world, clock, caplog):
    world.intr = Intr(width=1280, height=800)
    src = make_source(clock)
    src.start()
    with caplog.at_level(logging.ERROR, logger="booth_tracker.orbbec"):
        src.read()
        world.script = [FakeOBError("unplugged")]
        src.read()  # 斷線
        clock.t += 2.0
        src.read()  # 重開，新的失敗期
    assert len([r for r in caplog.records if r.levelno == logging.ERROR and "內參" in r.getMessage()]) == 2


def _corrupt_frameset(world):
    """depth buffer 比宣告的尺寸短 → depth_to_meters reshape 丟例外。"""
    good = world.default_frameset()
    short = FakeDepth(world.w, world.h, np.zeros((10, 10), np.uint16), world.scale)
    return FakeFrameSet(good.color, short)


def test_single_corrupt_frame_is_skipped_without_reopen(world, clock):
    """I2：單一壞幀只丟那一幀，不重開 pipeline。"""
    src = make_source(clock)
    src.start()
    world.script = [_corrupt_frameset(world)]
    assert src.read() is None
    assert src.connected and world.stops == 0 and world.starts == 1
    assert src.read() is not None  # 下一幀正常


def test_consecutive_corrupt_frames_reopen_after_max(world, clock):
    src = make_source(clock)
    src.start()
    world.script = [_corrupt_frameset(world) for _ in range(orbbec.MAX_FRAME_ERRORS)]
    for _ in range(orbbec.MAX_FRAME_ERRORS - 1):
        assert src.read() is None and src.connected
    assert src.read() is None
    assert not src.connected and world.stops == 1
    clock.t += 2.0
    assert src.read() is not None and world.starts == 2


def test_corrupt_frame_counter_resets_on_good_frame(world, clock):
    src = make_source(clock)
    src.start()
    bad = [_corrupt_frameset(world) for _ in range(orbbec.MAX_FRAME_ERRORS - 1)]
    world.script = bad + [world.default_frameset()] + list(bad)
    for _ in range(2 * (orbbec.MAX_FRAME_ERRORS - 1) + 1):
        src.read()
    assert src.connected and world.stops == 0


def test_color_format_env_default_is_mjpg_and_rgb_switch(world, clock, monkeypatch):
    make_source(clock).start()
    assert world.requested[0][2] == OBFormat.MJPG
    world.requested.clear()
    monkeypatch.setenv("BOOTH_ORBBEC_COLOR", "rgb")
    src = make_source(clock)
    src.start()
    assert world.requested[0][2] == OBFormat.RGB and src.profiles["color"] == "1280x720 RGB @30fps"


def test_color_format_env_invalid_value_rejected(world, clock, monkeypatch):
    monkeypatch.setenv("BOOTH_ORBBEC_COLOR", "h264")
    with pytest.raises(ValueError):
        make_source(clock)


def test_start_timeout_does_not_block_and_read_retries(world, clock):
    ev = threading.Event()
    world.block_next_start = ev
    src = make_source(clock, start_timeout_s=0.05)
    src.start()  # 不會卡住
    assert not src.connected and src.start_timed_out and "pipeline.start" in src.last_error
    clock.t += 2.0
    assert src.read() is None and world.starts == 1  # 舊的 start 還卡著，不再開第二個 pipeline
    ev.set()  # 讓卡住的 start 返回
    for t in threading.enumerate():
        if t.name == "orbbec-start":
            t.join(2)
    clock.t += 2.0
    assert src.read() is not None  # 重試成功
    assert world.starts == 2 and world.stops == 1  # 舊 pipeline 被關掉
    assert src.connected and not src.start_timed_out


# -- 3. 斷線 / 重連 -----------------------------------------------------------


def test_wait_error_returns_none_then_restarts_after_2s(world, clock):
    src = make_source(clock)
    src.start()
    assert world.starts == 1
    world.script = [FakeOBError("device disconnected")]
    assert src.read() is None
    assert not src.connected and world.stops == 1
    clock.t += 1.9
    assert src.read() is None and world.starts == 1  # 還沒滿 2 秒
    clock.t += 0.2
    f = src.read()  # 滿 2 秒：重開並讀到幀
    assert world.starts == 2 and f is not None


def test_failed_reopen_retries_every_2s(world, clock):
    src = make_source(clock)
    src.start()
    world.script = [FakeOBError("gone")]
    src.read()
    world.start_error = FakeOBError("still gone")
    clock.t += 2.0
    assert src.read() is None and world.starts == 2
    clock.t += 1.0
    assert src.read() is None and world.starts == 2
    clock.t += 1.0
    assert src.read() is None and world.starts == 3
    world.start_error = None
    clock.t += 2.0
    assert src.read() is not None and world.starts == 4


def test_stall_without_frames_counts_as_disconnect(world, clock):
    src = make_source(clock, stall_s=3.0)
    src.start()
    world.script = [None]
    clock.t += 1.0
    assert src.read() is None and src.connected  # 單次逾時不算斷線
    world.script = [None]
    clock.t += 3.0
    assert src.read() is None and not src.connected
    clock.t += 2.0
    assert src.read() is not None and world.starts == 2


def test_start_failure_does_not_raise_and_reports_error(world, clock):
    world.start_error = FakeOBError("uvc_open already opened")
    src = make_source(clock)
    src.start()
    assert not src.connected and "uvc_open" in src.last_error
    assert src.read() is None


def test_missing_sdk_error_mentions_doc_path(monkeypatch, clock):
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", None)  # import 會丟 ImportError
    with pytest.raises(OrbbecSdkMissing) as e:
        make_source(clock).start()
    assert "docs/orbbec.md" in str(e.value)


def test_frame_source_protocol_attrs():
    assert OrbbecSource.name == "orbbec" and OrbbecSource.has_depth is True


# -- doctor -----------------------------------------------------------------


@pytest.fixture
def doctor_env(monkeypatch, tmp_path, world):
    monkeypatch.setattr(doctor, "DATA_DIR", tmp_path)
    monkeypatch.setattr(doctor, "FPS_FRAMES", 60)
    monkeypatch.setattr(orbbec, "macos_needs_sudo", lambda: False)
    monkeypatch.setattr(doctor, "macos_needs_sudo", lambda: False)
    return tmp_path


def test_doctor_no_device_fails_cleanly_at_item_2(doctor_env, world, capsys):
    world.devices = 0
    assert doctor.run() == 1
    out = capsys.readouterr().out
    assert "[PASS] 1." in out
    assert "[FAIL] 2. 列出裝置" in out
    assert "USB 3" in out
    assert "[SKIP] 3." in out and "[SKIP] 7." in out
    assert "Traceback" not in out


def test_doctor_all_pass_with_fake_camera(doctor_env, world, capsys):
    assert doctor.run() == 0
    out = capsys.readouterr().out
    assert out.count("[PASS]") == 7 and "[FAIL]" not in out
    assert "SN=CP1234567" in out and "fx=900.00" in out and "2.000 m" in out
    png = cv2.imread(str(doctor_env / "doctor.png"))
    assert png is not None and png.shape == (720, 1280, 3)


def test_doctor_reports_open_failure_reason(doctor_env, world, capsys):
    world.start_error = FakeOBError("uvc_open already opened")
    assert doctor.run() == 1
    out = capsys.readouterr().out
    assert "[FAIL] 3." in out and "uvc_open already opened" in out and "[SKIP] 4." in out


def test_doctor_fails_when_center_depth_invalid(doctor_env, world, capsys):
    world.raw_value = 0
    assert doctor.run() == 1
    assert "[FAIL] 6." in capsys.readouterr().out


def test_doctor_sdk_import_failure_is_item_1(monkeypatch, doctor_env, capsys):
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", None)
    assert doctor.run() == 1
    out = capsys.readouterr().out
    assert "[FAIL] 1." in out and "orbbec.md" in out and "Traceback" not in out


def test_doctor_releases_enum_context_before_pipeline_start(doctor_env, world):
    # 實機（Gemini 336L、macOS）：列舉用的 Context 沒放掉，pipeline.start 會 uvc_open -3
    assert doctor.run() == 0
    assert world.ctx_alive_at_start is False
