from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from booth_tracker.paths import DATA_DIR
from booth_tracker.pose import MODEL_NAME, MODEL_URL, PoseDetector, ensure_model
from booth_tracker.types import LANDMARKS


def _network_ok() -> bool:
    try:
        urllib.request.urlopen(MODEL_URL, timeout=5).close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def model_ready():
    """模型已在 data/models 就直接用；否則需要網路，連不上就 skip。不在 import 時探測。"""
    if (DATA_DIR / "models" / MODEL_NAME).exists():
        return
    if not _network_ok():
        pytest.skip("模型不存在且無網路")


def test_ensure_model_returns_existing_file_without_download(tmp_path, monkeypatch):
    existing = tmp_path / MODEL_NAME
    existing.write_bytes(b"fake model")

    def boom(*a, **k):
        raise AssertionError("不應該下載")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert ensure_model(tmp_path) == existing
    assert existing.read_bytes() == b"fake model"


def test_ensure_model_downloads_when_missing(tmp_path, monkeypatch):
    dest_dir = tmp_path / "models"
    calls = []

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        return io.BytesIO(b"downloaded")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    p = ensure_model(dest_dir)
    assert calls == [MODEL_URL]
    assert p == dest_dir / MODEL_NAME and p.read_bytes() == b"downloaded"
    assert not list(dest_dir.glob("*.part"))


def test_black_image_returns_empty(model_ready):
    det = PoseDetector(num_poses=3)
    try:
        assert det.detect(np.zeros((480, 640, 3), dtype=np.uint8), 0) == []
    finally:
        det.close()


def test_timestamps_made_strictly_increasing(model_ready):
    det = PoseDetector(num_poses=1)
    try:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # 相同或倒退的 ts_ms 不可丟例外
        for ts in (100, 100, 50, 101):
            assert det.detect(img, ts) == []
    finally:
        det.close()


class _FakeLm:
    def __init__(self, x=0.5, y=0.5):
        self.x, self.y, self.z, self.visibility = x, y, 0.0, 0.9


class _FakeMaskImage:
    def __init__(self, arr):
        self._arr = arr

    def numpy_view(self):
        return self._arr


class _FakeLandmarker:
    def __init__(self, masks):
        self.masks = masks

    def detect_for_video(self, image, ts_ms):
        lms = [_FakeLm() for _ in range(33)]

        class R:
            pose_landmarks = [lms, lms]
            pose_world_landmarks = [lms, lms]
            segmentation_masks = self.masks

        return R

    def close(self): ...


def _detector_with_fake(monkeypatch, masks):
    captured = {}

    def fake_create(options):
        captured["options"] = options
        return _FakeLandmarker(masks)

    monkeypatch.setattr(
        "booth_tracker.pose.vision.PoseLandmarker.create_from_options", staticmethod(fake_create)
    )
    det = PoseDetector(num_poses=2, model_path=Path("unused.task"))
    return det, captured


def test_開啟分割mask(monkeypatch):
    _, captured = _detector_with_fake(monkeypatch, [])
    assert captured["options"].output_segmentation_masks is True


def test_每人帶對應index的mask(monkeypatch):
    m0 = np.zeros((6, 8, 1), np.float32)  # 真 mediapipe 的 numpy_view 是 (H,W,1)
    m1 = np.ones((6, 8, 1), np.float32)
    det, _ = _detector_with_fake(monkeypatch, [_FakeMaskImage(m0), _FakeMaskImage(m1)])
    persons = det.detect(np.zeros((6, 8, 3), np.uint8), 0)
    assert len(persons) == 2
    assert persons[0].mask.dtype == np.float32 and persons[0].mask.shape == (6, 8)
    assert persons[0].mask.sum() == 0 and persons[1].mask.sum() == 48
    assert not np.shares_memory(persons[1].mask, m1)  # 已複製，不指向 MediaPipe 緩衝


def test_沒有mask時為None(monkeypatch):
    det, _ = _detector_with_fake(monkeypatch, [])
    persons = det.detect(np.zeros((6, 8, 3), np.uint8), 0)
    assert [p.mask for p in persons] == [None, None]


def test_每人輸出33個關節_像素與模型3D(monkeypatch):
    det, _ = _detector_with_fake(monkeypatch, [])
    d = det.detect(np.zeros((6, 8, 3), np.uint8), 0)[0]
    assert set(d.pixels) == set(LANDMARKS)
    assert d.pixels["left_ankle"] == (4.0, 3.0)  # 正規化 0.5 x 寬 8、高 6
    assert d.visibility["nose"] == pytest.approx(0.9)
    assert isinstance(d.model_world["nose"], np.ndarray) and d.model_world["nose"].shape == (3,)


class _DelegateLandmarker:
    def __init__(self, delegate, fail_detect=False):
        self.delegate, self.fail_detect, self.closed, self.formats = delegate, fail_detect, False, []

    def detect_for_video(self, image, ts):
        self.formats.append(image.image_format)
        if self.fail_detect:
            raise RuntimeError("GPU 掛了")
        return type("R", (), {"pose_landmarks": [], "pose_world_landmarks": [], "segmentation_masks": None})()

    def close(self):
        self.closed = True


def _fake_create(monkeypatch, fail_create=False, fail_detect=False):
    made = []

    def create(self, delegate):
        if delegate == "gpu" and fail_create:
            raise RuntimeError("no GPU")
        lm = _DelegateLandmarker(delegate, fail_detect=fail_detect and delegate == "gpu")
        made.append(lm)
        return lm

    monkeypatch.setattr(PoseDetector, "_create", create)
    return made


def test_gpu建立失敗退回cpu(monkeypatch, tmp_path):
    made = _fake_create(monkeypatch, fail_create=True)
    det = PoseDetector(model_path=tmp_path / "m.task", delegate="gpu")
    assert det.delegate == "cpu" and [m.delegate for m in made] == ["cpu"]


def test_gpu送SRGBA_推論失敗改用cpu重跑同一幀(monkeypatch, tmp_path):
    import mediapipe as mp

    made = _fake_create(monkeypatch, fail_detect=True)
    det = PoseDetector(model_path=tmp_path / "m.task", delegate="gpu")
    assert det.delegate == "gpu"
    assert det.detect(np.zeros((8, 8, 3), np.uint8), 0) == []
    gpu, cpu = made
    assert gpu.formats == [mp.ImageFormat.SRGBA] and gpu.closed
    assert cpu.formats == [mp.ImageFormat.SRGB] and det.delegate == "cpu"


def test_delegate環境變數(monkeypatch, tmp_path):
    from booth_tracker.pose import DELEGATE_ENV, delegate_preference

    monkeypatch.delenv(DELEGATE_ENV, raising=False)
    assert delegate_preference() == "cpu"
    monkeypatch.setenv(DELEGATE_ENV, " GPU ")
    assert delegate_preference() == "gpu"
    monkeypatch.setenv(DELEGATE_ENV, "tpu")
    with pytest.raises(ValueError):
        delegate_preference()


def test_gpu真模型可跑(model_ready):
    """真的建 GPU delegate 跑一張圖；這台機器沒有 GPU 時會退回 CPU，兩種都不能炸。"""
    det = PoseDetector(num_poses=3, delegate="gpu")
    try:
        assert det.detect(np.zeros((480, 640, 3), np.uint8), 0) == []
    finally:
        det.close()
