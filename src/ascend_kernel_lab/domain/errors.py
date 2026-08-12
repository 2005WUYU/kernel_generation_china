"""Domain-specific exceptions.

The controller and worker communicate across a persistence boundary, so callers
need to distinguish invalid input, invalid state transitions, and lost leases
without inspecting error strings.
"""

from __future__ import annotations


class DomainError(ValueError):
    """Base class for invalid domain operations."""


class InvalidTransitionError(DomainError):
    """Raised when an aggregate is moved along an unsupported state edge."""


class ConcurrentUpdateError(DomainError):
    """Raised when optimistic state validation detects a concurrent update."""


class LeaseLostError(DomainError):
    """Raised when a worker no longer owns an evaluation-job lease."""


class IdempotencyConflictError(DomainError):
    """Raised when an idempotency key is reused for a different operation."""


class ArtifactConflictError(DomainError):
    """Raised when immutable artifact contents would be overwritten."""


class ArtifactIntegrityError(DomainError):
    """Raised when artifact bytes no longer match their recorded digest."""
