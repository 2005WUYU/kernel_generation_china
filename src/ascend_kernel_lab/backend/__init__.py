"""Execution backends for real Ascend hardware and deterministic tests."""

from typing import TYPE_CHECKING, Any

from .base import Backend, StageResult, StageStatus
from .fake import FakeBackend

if TYPE_CHECKING:
    from .ascend import AscendTritonBackend


def __getattr__(name: str) -> Any:
    if name == "AscendTritonBackend":
        from .ascend import AscendTritonBackend

        return AscendTritonBackend
    raise AttributeError(name)

__all__ = [
    "AscendTritonBackend",
    "Backend",
    "FakeBackend",
    "StageResult",
    "StageStatus",
]
