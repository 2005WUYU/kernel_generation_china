"""Controller-side durable evaluation transport errors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ascend_kernel_lab.domain import JobStatus


class QueueBackendError(RuntimeError):
    """Base class for failures in the controller-to-worker queue transport."""


class QueueBackendConfigurationError(QueueBackendError):
    """Raised when a job cannot be tied to immutable controller state."""


class QueueProtocolError(QueueBackendError):
    """Raised when a worker commits a malformed or unsafe result."""


class QueueJobFailed(QueueBackendError):
    """Raised after a durable job reaches the dead-letter state."""

    def __init__(
        self,
        job_id: str,
        *,
        status: JobStatus,
        error: Mapping[str, Any] | None,
    ) -> None:
        self.job_id = job_id
        self.status = status
        self.error = dict(error or {})
        super().__init__(
            f"evaluation job {job_id!r} ended in {status.value}: {self.error}"
        )


class QueueJobCancelled(QueueJobFailed):
    """Raised when an already-cancelled idempotent job is observed."""


class QueueWaitTimeout(QueueBackendError, TimeoutError):
    """The controller stopped waiting; the worker lease may still be active."""

    def __init__(
        self,
        job_id: str,
        *,
        timeout_seconds: float,
        cancelled: bool,
    ) -> None:
        self.job_id = job_id
        self.timeout_seconds = timeout_seconds
        self.cancelled = cancelled
        suffix = "and cancelled it" if cancelled else "without cancelling active work"
        super().__init__(
            f"timed out after {timeout_seconds:g}s waiting for evaluation job "
            f"{job_id!r} {suffix}"
        )
