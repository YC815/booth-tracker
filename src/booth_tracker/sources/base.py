from __future__ import annotations

from typing import Protocol

from booth_tracker.types import Frame


class FrameSource(Protocol):
    name: str
    has_depth: bool

    def start(self) -> None: ...

    def read(self) -> Frame | None:
        """Block up to ~1 s for the next frame; None on timeout / disconnect."""
        ...

    def stop(self) -> None: ...
