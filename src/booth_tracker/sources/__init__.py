from __future__ import annotations

from booth_tracker.sources.base import FrameSource


def make_source(name: str, replay: str | None = None, webcam_index: int = 0) -> FrameSource:
    """Imports lazily so a missing pyorbbecsdk does not break webcam/replay."""
    if name == "webcam":
        from booth_tracker.sources.webcam import WebcamSource

        return WebcamSource(index=webcam_index)
    if name == "orbbec":
        from booth_tracker.sources.orbbec import OrbbecSource

        return OrbbecSource()
    if name == "replay":
        from booth_tracker.paths import DATA_DIR
        from booth_tracker.sources.replay import ReplaySource

        if not replay:
            raise ValueError("--replay NAME is required for source=replay")
        return ReplaySource(DATA_DIR / "recordings" / replay)
    raise ValueError(f"unknown source: {name}")
