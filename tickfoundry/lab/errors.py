"""Typed exceptions mirroring the contract's error codes.

Every error carries ``code``, ``message``, ``request_id`` and ``details``
exactly as the engine or control plane sent them, plus the HTTP ``status``.
Client-side conditions (a wait deadline, a transport failure, a missing API
key) use their own codes so a caller can tell them apart from server codes.
"""
from __future__ import annotations

from typing import Any, Mapping

from .types import ErrorInfo


class LabError(Exception):
    """Base class of every SDK error. Unknown server codes raise this class directly."""

    code: str = "UNKNOWN"

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        request_id: str | None = None,
        details: Mapping[str, Any] | None = None,
        status: int | None = None,
    ) -> None:
        self.code = code or type(self).code
        self.message = message or self.code
        self.request_id = request_id
        self.details = dict(details or {})
        self.status = status
        super().__init__(self.__str__())

    def __str__(self) -> str:  # noqa: D105
        parts = [f"{self.code}: {self.message}"]
        if self.status is not None:
            parts.append(f"(http {self.status})")
        if self.request_id:
            parts.append(f"[request {self.request_id}]")
        return " ".join(parts)


# API error codes ----------------------------------------------------------
class NotEntitled(LabError):
    code = "NOT_ENTITLED"


class ScopeTooLarge(LabError):
    code = "SCOPE_TOO_LARGE"


class DataNotPublished(LabError):
    code = "DATA_NOT_PUBLISHED"


class DataQualityBlocked(LabError):
    code = "DATA_QUALITY_BLOCKED"


class QuotaExceeded(LabError):
    code = "QUOTA_EXCEEDED"


class SqlNotAllowed(LabError):
    code = "SQL_NOT_ALLOWED"


class JobTimedOut(LabError):
    """The engine timed the job out (server side; contract code ``JOB_TIMED_OUT``)."""

    code = "JOB_TIMED_OUT"


class SourceChanged(LabError):
    code = "SOURCE_CHANGED"


class ResultExpired(LabError):
    code = "RESULT_EXPIRED"


class InvalidRequest(LabError):
    code = "INVALID_REQUEST"


class NotFound(LabError):
    code = "NOT_FOUND"


class RateLimited(LabError):
    code = "RATE_LIMITED"

    def __init__(self, message: str = "", *, retry_after_s: float | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.retry_after_s = retry_after_s


class EngineUnavailable(LabError):
    code = "ENGINE_UNAVAILABLE"


# Grant and other HTTP-level codes -----------------------------------------
class GrantExpired(LabError):
    """``401 GRANT_EXPIRED``: the client re-mints transparently; surfaced only when re-minting also fails."""

    code = "GRANT_EXPIRED"


class GrantInvalid(LabError):
    code = "GRANT_INVALID"


class FeatureDisabled(LabError):
    """``POST /recipe-runs`` and other features that are not enabled."""

    code = "FEATURE_DISABLED"


class Forbidden(LabError):
    """403 without a contract code (CSRF or origin check on a web route)."""

    code = "FORBIDDEN"


class IdempotencyConflict(LabError):
    """409: the same ``Idempotency-Key`` was reused with a different body."""

    code = "IDEMPOTENCY_CONFLICT"


# Client-side conditions ------------------------------------------------------
class ConfigurationError(LabError):
    code = "CLIENT_CONFIGURATION"


class TransportError(LabError):
    """The request never produced an HTTP response (DNS, connect, read timeout)."""

    code = "CLIENT_TRANSPORT"


class WaitTimeout(LabError):
    """``wait(timeout)`` reached its deadline before the job finished (client side)."""

    code = "CLIENT_WAIT_TIMEOUT"

    def __init__(self, message: str = "", *, job_id: str | None = None, last_state: str | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.job_id = job_id
        self.last_state = last_state


class JobFailed(LabError):
    """A job reached ``failed`` without a contract error code."""

    code = "JOB_FAILED"


class JobCancelled(LabError):
    code = "JOB_CANCELLED"


class UnexpectedResponse(LabError):
    """The body was not what the contract promises (for example JSON where Arrow was requested)."""

    code = "CLIENT_UNEXPECTED_RESPONSE"


ERROR_CLASSES: dict[str, type[LabError]] = {
    cls.code: cls
    for cls in (
        NotEntitled,
        ScopeTooLarge,
        DataNotPublished,
        DataQualityBlocked,
        QuotaExceeded,
        SqlNotAllowed,
        JobTimedOut,
        SourceChanged,
        ResultExpired,
        InvalidRequest,
        NotFound,
        RateLimited,
        EngineUnavailable,
        GrantExpired,
        GrantInvalid,
        FeatureDisabled,
        Forbidden,
        IdempotencyConflict,
        JobFailed,
        JobCancelled,
    )
}

_STATUS_FALLBACK: dict[int, type[LabError]] = {
    400: InvalidRequest,
    401: GrantInvalid,
    403: Forbidden,
    404: NotFound,
    409: IdempotencyConflict,
    413: ScopeTooLarge,
    429: RateLimited,
    502: EngineUnavailable,
    503: EngineUnavailable,
    504: EngineUnavailable,
}


def error_from_info(info: ErrorInfo, *, status: int | None = None, retry_after_s: float | None = None) -> LabError:
    """Build the typed exception for an ``ErrorInfo`` (server code first, HTTP status second)."""
    cls = ERROR_CLASSES.get(info.code)
    if cls is None and status is not None:
        cls = _STATUS_FALLBACK.get(status)
    if cls is None:
        cls = LabError
    kwargs: dict[str, Any] = dict(code=info.code, request_id=info.request_id, details=info.details, status=status)
    if cls is RateLimited:
        return RateLimited(info.message, retry_after_s=retry_after_s, **kwargs)
    return cls(info.message, **kwargs)


__all__ = [
    "ERROR_CLASSES",
    "ConfigurationError",
    "DataNotPublished",
    "DataQualityBlocked",
    "EngineUnavailable",
    "FeatureDisabled",
    "Forbidden",
    "GrantExpired",
    "GrantInvalid",
    "IdempotencyConflict",
    "InvalidRequest",
    "JobCancelled",
    "JobFailed",
    "JobTimedOut",
    "LabError",
    "NotEntitled",
    "NotFound",
    "QuotaExceeded",
    "RateLimited",
    "ResultExpired",
    "ScopeTooLarge",
    "SourceChanged",
    "SqlNotAllowed",
    "TransportError",
    "UnexpectedResponse",
    "WaitTimeout",
    "error_from_info",
]
