"""MediaPipe PoseLandmarker 包裝（Tasks API，VIDEO 模式）。輸出全身 33 個關節和每人的分割 mask。"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from booth_tracker.paths import DATA_DIR
from booth_tracker.types import LANDMARKS, Detection

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
MODEL_NAME = "pose_landmarker_full.task"
# cpu（預設）| gpu。M3 實測 full + segmentation + 3 人：CPU 21 ms → GPU 7.8 ms。
# GPU 建立或推論失敗會自動退回 CPU。GPU 只吃 4 通道影像，所以送 SRGBA。
DELEGATE_ENV = "BOOTH_POSE_DELEGATE"

log = logging.getLogger(__name__)


def delegate_preference() -> str:
    pref = os.environ.get(DELEGATE_ENV, "cpu").strip().lower()
    if pref not in ("cpu", "gpu"):
        raise ValueError(f"{DELEGATE_ENV} 只能是 cpu 或 gpu，收到 {pref!r}")
    return pref


def ensure_model(dest_dir: Path | None = None) -> Path:
    """模型不存在就下載（先寫 .part 再 rename，避免留下半個檔）。"""
    dest = Path(dest_dir if dest_dir is not None else DATA_DIR / "models") / MODEL_NAME
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(MODEL_URL, timeout=60) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    return dest


def _mask_2d(view: np.ndarray) -> np.ndarray:
    """MediaPipe 給 (H,W,1) float32；types.py 契約是 HxW。"""
    m = np.array(view, dtype=np.float32)
    return m[..., 0] if m.ndim == 3 else m


class PoseDetector:
    def __init__(self, num_poses: int = 3, model_path: Path | None = None, delegate: str | None = None):
        self._path = model_path if model_path is not None else ensure_model()
        self._num_poses = num_poses
        self.delegate = "cpu"
        if (delegate or delegate_preference()) == "gpu":
            try:
                self._landmarker = self._create("gpu")
                self.delegate = "gpu"
            except Exception as e:
                log.warning("MediaPipe GPU delegate 建立失敗，改用 CPU：%s", e)
        if self.delegate == "cpu":
            self._landmarker = self._create("cpu")
        log.info("MediaPipe PoseLandmarker：%s", self.delegate)
        self._last_ts = -1

    def _create(self, delegate: str):
        d = BaseOptions.Delegate.GPU if delegate == "gpu" else BaseOptions.Delegate.CPU
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._path), delegate=d),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=self._num_poses,
            min_pose_detection_confidence=0.5,
            output_segmentation_masks=True,
        )
        return vision.PoseLandmarker.create_from_options(options)

    def _image(self, rgb: np.ndarray) -> mp.Image:
        if self.delegate == "gpu":
            return mp.Image(image_format=mp.ImageFormat.SRGBA, data=cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA))
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    def _run(self, rgb: np.ndarray, ts_ms: int):
        try:
            return self._landmarker.detect_for_video(self._image(rgb), ts_ms)
        except Exception:
            if self.delegate != "gpu":
                raise
            log.exception("MediaPipe GPU 推論失敗，改用 CPU")
        try:
            self._landmarker.close()
        except Exception:
            pass
        self._landmarker = self._create("cpu")
        self.delegate = "cpu"
        return self._landmarker.detect_for_video(self._image(rgb), ts_ms)

    def detect(self, rgb: np.ndarray, ts_ms: int) -> list[Detection]:
        ts_ms = int(ts_ms)
        if ts_ms <= self._last_ts:
            ts_ms = self._last_ts + 1
        self._last_ts = ts_ms

        h, w = rgb.shape[:2]
        result = self._run(rgb, ts_ms)

        masks = result.segmentation_masks or []
        out: list[Detection] = []
        for i, (lms, world_lms) in enumerate(zip(result.pose_landmarks, result.pose_world_landmarks)):
            pixels = {name: (lms[j].x * w, lms[j].y * h) for j, name in enumerate(LANDMARKS)}
            visibility = {name: float(lms[j].visibility) for j, name in enumerate(LANDMARKS)}
            model = {name: np.array([world_lms[j].x, world_lms[j].y, world_lms[j].z]) for j, name in enumerate(LANDMARKS)}
            # numpy_view 指向 MediaPipe 內部緩衝，必須複製
            mask = _mask_2d(masks[i].numpy_view()) if i < len(masks) else None
            out.append(Detection(pixels=pixels, visibility=visibility, model_world=model, mask=mask))
        return out

    def close(self) -> None:
        self._landmarker.close()
