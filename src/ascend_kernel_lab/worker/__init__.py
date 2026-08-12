"""Durable workers and isolated stage-process helpers."""

from typing import TYPE_CHECKING, Any

from .device_lock import DeviceLock, DeviceLockTimeout
from .stage_runner import (
    ResourceLimits,
    StageProcessResult,
    StageRunner,
    clean_environment,
)

if TYPE_CHECKING:
    from .health import AscendHealthChecker
    from .service import WorkerPayloadError, WorkerService


def __getattr__(name: str) -> Any:
    if name == "AscendHealthChecker":
        from .health import AscendHealthChecker

        return AscendHealthChecker
    if name in {"WorkerPayloadError", "WorkerService"}:
        from .service import WorkerPayloadError, WorkerService

        return {"WorkerPayloadError": WorkerPayloadError, "WorkerService": WorkerService}[name]
    raise AttributeError(name)

__all__ = [
    "AscendHealthChecker",
    "DeviceLock",
    "DeviceLockTimeout",
    "ResourceLimits",
    "StageProcessResult",
    "StageRunner",
    "WorkerPayloadError",
    "WorkerService",
    "clean_environment",
]
