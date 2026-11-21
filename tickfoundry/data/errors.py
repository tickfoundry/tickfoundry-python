"""Errors raised by ``tickfoundry.data``.

Every server error keeps the HTTP status, the server's ``error`` text and, when
the API sent one, its machine ``code`` (``history_locked``,
``daily_download_limit``, ``venue_not_covered`` ...), so a caller can branch on
the code and show the text as is.
"""
from __future__ import annotations

from typing import Any, Mapping


class TickFoundryError(Exception):
    """Base class of every error this package raises."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        details: Mapping[str, Any] | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.message = message
        self.status = status
        self.code = code
        self.details = dict(details or {})
        self.retry_after = retry_after
        super().__init__(str(self))

    def __str__(self) -> str:
        bits = [self.message]
        if self.code:
            bits.append(f"[{self.code}]")
        if self.status is not None:
            bits.append(f"(http {self.status})")
        return " ".join(bits)


class MissingApiKey(TickFoundryError):
    """No key was passed and TICKFOUNDRY_API_KEY is not set."""


class AuthError(TickFoundryError):
    """401, or a 403 about the key itself (revoked, no API access, missing scope)."""


class NotEntitled(TickFoundryError):
    """The plan does not cover this venue or layer (402, or 403 ``venue_not_covered``)."""


class HistoryLocked(NotEntitled):
    """The day is before the plan's history window; backfill or a yearly plan unlocks it."""


class RateLimited(TickFoundryError):
    """429 that did not clear within the client's retries."""


class DailyDownloadLimit(RateLimited):
    """The plan's downloads for today are used up (resets 00:00 UTC)."""


class RangeTooLarge(TickFoundryError):
    """One request would exceed the venue's days or bytes per pull."""


class NotFound(TickFoundryError):
    """404: an unknown id, or a venue that is not on sale."""


class JobFailed(TickFoundryError):
    """The server could not build the requested bundle."""


class JobExpired(TickFoundryError):
    """410: the job is past its 30-day access window; request the data again."""


class Timeout(TickFoundryError):
    """A wait (job build or rebuild) passed the caller's deadline."""


def from_response(status: int, body: Any, *, retry_after: float | None = None) -> TickFoundryError:
    """Map an API error response to the matching exception class."""
    if isinstance(body, Mapping):
        raw = body.get("error")
        code = body.get("code") if isinstance(body.get("code"), str) else None
    else:
        raw, code = body, None
    if isinstance(raw, Mapping):  # zod validation errors arrive as an object
        message = "invalid request: " + str(raw)
    else:
        message = str(raw or f"http {status}")
    details = body if isinstance(body, Mapping) else None

    def mk(cls: type[TickFoundryError]) -> TickFoundryError:
        return cls(message, status=status, code=code, details=details, retry_after=retry_after)

    low = message.lower()
    if code == "history_locked":
        return mk(HistoryLocked)
    if code == "daily_download_limit":
        return mk(DailyDownloadLimit)
    if status == 429:
        return mk(RateLimited)
    if status == 401:
        return mk(AuthError)
    if status == 402 or code == "venue_not_covered":
        return mk(NotEntitled)
    if status == 403:
        if "api key" in low or "scope" in low or "api access" in low or "no active plan" in low:
            return mk(AuthError)
        return mk(NotEntitled)
    if status == 404:
        return mk(NotFound)
    if status == 410:
        return mk(JobExpired)
    if status == 400 and ("capped at" in low or "at most" in low or "too long" in low):
        return mk(RangeTooLarge)
    return mk(TickFoundryError)
