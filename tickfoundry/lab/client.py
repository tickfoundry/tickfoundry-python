"""Authenticated client for the hosted Lab API.

Topology: one origin, ``TICKFOUNDRY_LAB_URL``
(default ``https://tickfoundry.com``). Control-plane routes (``/me``,
``/grant``, ``/workspaces*``, ``/saved-analyses*``, ``/usage``, ``/export``)
take the ``tf_live_`` API key as a bearer; every engine route takes a
short-lived grant minted from ``POST /api/lab/v1/grant`` (
Ed25519 JWS, ``exp = iat + 600``). The engine answers ``401 GRANT_EXPIRED``
as a distinct code so the client re-mints once and retries transparently.

Robustness rules: bounded backoff on 429 and
503 honouring ``Retry-After``, an ``Idempotency-Key`` on every POST (scoped
by the server to ``(sub, operation)``; the same key is reused across the
retries of one call so a retry can never double-submit), typed errors, a hard
timeout on every poll loop, never an infinite loop. Request ids are surfaced
on every typed object, on every error and on ``client.last_request_id``.
"""
from __future__ import annotations

import os
import time
import uuid
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterator, Mapping, Sequence

try:  # httpx is a declared dependency; the guard keeps ``LocalReplay`` importable without it.
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

from . import errors
from .types import (
    JOB_TERMINAL_STATES,
    ErrorInfo,
    EventsPage,
    Grant,
    JobInfo,
    ReplayInfo,
    ResultInfo,
    Scope,
    SeriesPage,
    SimulationResult,
    SnapshotResult,
    ns_str,
)

SDK_VERSION = "0.1.0"
DEFAULT_LAB_URL = "https://tickfoundry.com"
API_PREFIX = "/api/lab/v1"
ARROW_MEDIA_TYPE = "application/vnd.apache.arrow.stream"
#: Account routes, authenticated with the API key.
WEB_ROUTES = ("me", "grant", "workspaces", "saved-analyses", "usage", "export")
#: Header the grant route requires; harmless on engine routes.
REQUESTED_WITH = "tickfoundry-lab"
#: Grant TTL assumed when the grant response carries no ``expires_at``.
GRANT_TTL_S = 600.0
#: Statuses retried with bounded backoff.
RETRY_STATUSES = frozenset({429, 503})


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse ``Retry-After`` (delay-seconds or HTTP-date); ``None`` when absent or unparsable."""
    if not value:
        return None
    text = value.strip()
    if text.isdigit():
        return float(text)
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        return None
    return max(0.0, when.timestamp() - time.time())


class LabClient:
    """Synchronous client for ``tickfoundry.com/api/lab/v1``.

    ``sleep``, ``clock`` (monotonic) and ``wall_clock`` are injectable so tests
    run without waiting; ``transport`` accepts an ``httpx`` transport such as
    ``httpx.MockTransport``.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_LAB_URL,
        *,
        timeout: float = 30.0,
        max_retries: int = 4,
        backoff_base_s: float = 0.5,
        backoff_cap_s: float = 8.0,
        retry_after_cap_s: float = 120.0,
        retry_budget_s: float = 90.0,
        grant_refresh_margin_s: float = 30.0,
        transport: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        jitter: Callable[[], float] | None = None,
        user_agent: str | None = None,
    ) -> None:
        if httpx is None:  # pragma: no cover
            raise errors.ConfigurationError("httpx is required for LabClient (pip install httpx)")
        if not api_key or not isinstance(api_key, str):
            raise errors.ConfigurationError("an API key is required (TICKFOUNDRY_API_KEY)")
        if max_retries < 0:
            raise errors.ConfigurationError("max_retries must be >= 0")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self.retry_after_cap_s = retry_after_cap_s
        self.retry_budget_s = retry_budget_s
        self.grant_refresh_margin_s = grant_refresh_margin_s
        self._sleep = sleep
        self._clock = clock
        self._wall_clock = wall_clock
        self._jitter = jitter
        self._grant: Grant | None = None
        self.last_request_id: str | None = None
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            headers={
                "User-Agent": user_agent or f"tickfoundry-sdk/{SDK_VERSION}",
                "X-Lab-Schema": "lab.v1",
            },
        )

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_environment(cls, **overrides: Any) -> "LabClient":
        """Build from ``TICKFOUNDRY_API_KEY`` and ``TICKFOUNDRY_LAB_URL``."""
        api_key = overrides.pop("api_key", None) or os.environ.get("TICKFOUNDRY_API_KEY", "")
        base_url = overrides.pop("base_url", None) or os.environ.get("TICKFOUNDRY_LAB_URL", "") or DEFAULT_LAB_URL
        if not api_key:
            raise errors.ConfigurationError("TICKFOUNDRY_API_KEY is not set")
        return cls(api_key, base_url, **overrides)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "LabClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ----------------------------------------------------------------- grants
    def grant(self, *, force: bool = False) -> Grant:
        """Return a grant that is valid for at least ``grant_refresh_margin_s`` more seconds, minting when needed."""
        current = self._grant
        if current is not None and not force:
            expiry = current.expires_at_epoch_s()
            if expiry is None:
                expiry = current.minted_at_wall_s + GRANT_TTL_S
            if self._wall_clock() + self.grant_refresh_margin_s < expiry:
                return current
        response = self._request("POST", "grant", auth="api_key", json=None)
        body = self._json(response)
        token = body.get("grant") if isinstance(body, Mapping) else None
        if not token:
            raise errors.UnexpectedResponse("grant response has no 'grant' field", request_id=self.last_request_id, status=response.status_code)
        self._grant = Grant(
            token=str(token),
            expires_at=body.get("expires_at"),
            policy_version=body.get("policy_version"),
            request_id=self.last_request_id,
            minted_at_wall_s=self._wall_clock(),
        )
        return self._grant

    def invalidate_grant(self) -> None:
        self._grant = None

    # ---------------------------------------------------------- control plane
    def me(self) -> dict[str, Any]:
        """``GET /api/lab/v1/me``: tier, entitlement snapshot, capabilities, profile, usage."""
        return self._json(self._request("GET", "me", auth="api_key"))

    def usage(self) -> dict[str, Any]:
        return self._json(self._request("GET", "usage", auth="api_key"))

    def workspaces(self) -> Any:
        return self._json(self._request("GET", "workspaces", auth="api_key"))

    # ---------------------------------------------------------------- catalog
    def catalog(self, venue: str, **query: Any) -> dict[str, Any]:
        """``GET /catalog?venue=...`` (``q``, ``family``, ``page``, ``page_size``)."""
        params = {"venue": venue, **{k: v for k, v in query.items() if v is not None}}
        return self._json(self._request("GET", "catalog", params=params))

    def catalog_unit(self, unit_id: str) -> dict[str, Any]:
        return self._json(self._request("GET", f"catalog/units/{unit_id}"))

    def catalog_unit_markets(self, unit_id: str, date: str) -> dict[str, Any]:
        """Markets of a unit-day from the staged file (may be ``202`` with a job)."""
        return self._json(self._request("GET", f"catalog/units/{unit_id}/markets", params={"date": date}))

    def catalog_days(self, instrument_id: str, start: str | None = None, end: str | None = None) -> dict[str, Any]:
        params = {k: v for k, v in (("from", start), ("to", end)) if v is not None}
        return self._json(self._request("GET", f"catalog/instruments/{instrument_id}/days", params=params))

    def health(self) -> dict[str, Any]:
        return self._json(self._request("GET", "health", auth=None))

    # ----------------------------------------------------------------- scopes
    def create_scope(
        self,
        venue: str,
        instrument_ids: Sequence[str],
        start: Any,
        end: Any,
        *,
        products: Sequence[str] = ("book", "trades"),
        depth: int | None = None,
        idempotency_key: str | None = None,
    ) -> Scope:
        """``POST /scopes``. ``start``/``end`` accept RFC 3339 or ns."""
        body: dict[str, Any] = {
            "venue": venue,
            "instrument_ids": list(instrument_ids),
            "start": ns_str(start, name="start"),
            "end": ns_str(end, name="end"),
            "products": list(products),
        }
        if depth is not None:
            body["depth"] = int(depth)
        response = self._request("POST", "scopes", json=body, idempotency_key=idempotency_key)
        return Scope.from_json(self._json(response), request_id=self.last_request_id)

    def get_scope(self, scope_id: str) -> Scope:
        response = self._request("GET", f"scopes/{scope_id}")
        return Scope.from_json(self._json(response), request_id=self.last_request_id)

    # ---------------------------------------------------------------- replays
    def prepare_replay(
        self,
        scope_id: str,
        instrument_id: str,
        *,
        start: Any,
        end: Any,
        products: Sequence[str] = ("book", "trades"),
        seed: str = "authorized_prior_state",
        idempotency_key: str | None = None,
    ) -> "Replay":
        """``POST /replays``: 200 means ready, 202 means staging (``Replay.wait`` polls the job)."""
        body = {
            "scope_id": scope_id,
            "instrument_id": instrument_id,
            "start": ns_str(start, name="start"),
            "end": ns_str(end, name="end"),
            "products": list(products),
            "seed": seed,
        }
        response = self._request("POST", "replays", json=body, idempotency_key=idempotency_key)
        info = ReplayInfo.from_json(self._json(response), request_id=self.last_request_id)
        return Replay(self, info)

    def get_replay(self, replay_id: str) -> "Replay":
        response = self._request("GET", f"replays/{replay_id}")
        return Replay(self, ReplayInfo.from_json(self._json(response), request_id=self.last_request_id))

    # ------------------------------------------------------------ simulations
    def simulate(
        self,
        *,
        instrument_id: str,
        timestamp: Any,
        side: str,
        size_mode: str,
        size: Any,
        replay_id: str | None = None,
        scope_id: str | None = None,
        limit_price: Any = None,
        assumed_delay_ms: int = 0,
        fee_model: str = "excluded",
        fee_model_id: str | None = None,
        explicit_parameters: Mapping[str, Any] | None = None,
        reference: str = "execution_best_quote",
        idempotency_key: str | None = None,
    ) -> "Job":
        """``POST /simulations`` (body). Returns a ``Job``; a warm 200 is already complete."""
        if replay_id is None and scope_id is None:
            raise errors.InvalidRequest("simulate needs replay_id or scope_id")
        if isinstance(size, float) or isinstance(limit_price, float):
            raise errors.InvalidRequest("size and limit_price must be decimal strings, not floats")
        body: dict[str, Any] = {
            "instrument_id": instrument_id,
            "analysis_timestamp_ns": ns_str(timestamp, name="timestamp"),
            "side": side,
            "size_mode": size_mode,
            "size": str(size),
            "assumed_delay_ms": int(assumed_delay_ms),
            "fee_model": fee_model,
            "reference": reference,
        }
        if replay_id is not None:
            body["replay_id"] = replay_id
        if scope_id is not None:
            body["scope_id"] = scope_id
        if limit_price is not None:
            body["limit_price"] = str(limit_price)
        if fee_model_id is not None:
            body["fee_model_id"] = fee_model_id
        if explicit_parameters is not None:
            body["explicit_parameters"] = {k: str(v) for k, v in explicit_parameters.items()}
        response = self._request("POST", "simulations", json=body, idempotency_key=idempotency_key)
        data = self._json(response)
        if response.status_code == 202 or (isinstance(data, Mapping) and "job_id" in data and "status" not in data):
            info = JobInfo.from_json(data, request_id=self.last_request_id)
            return Job(self, info.job_id, info=info, kind="simulation")
        result = SimulationResult.from_json(data, request_id=self.last_request_id)
        return Job(self, str(data.get("job_id") or f"inline-{uuid.uuid4()}"), inline_result=result, kind="simulation")

    # ------------------------------------------------------------------- jobs
    def get_job(self, job_id: str) -> JobInfo:
        response = self._request("GET", f"jobs/{job_id}")
        return JobInfo.from_json(self._json(response), request_id=self.last_request_id)

    def list_jobs(self, *, state: str | None = None, limit: int | None = None) -> list[JobInfo]:
        params = {k: v for k, v in (("state", state), ("limit", limit)) if v is not None}
        data = self._json(self._request("GET", "jobs", params=params))
        rows = data.get("jobs", data) if isinstance(data, Mapping) else data
        return [JobInfo.from_json(r, request_id=self.last_request_id) for r in rows if isinstance(r, Mapping)]

    def cancel_job(self, job_id: str) -> JobInfo | None:
        """``POST /jobs/{id}/cancel`` (202, idempotent). Returns the job when the body carries it."""
        response = self._request("POST", f"jobs/{job_id}/cancel", json=None, idempotency_key=f"cancel-{job_id}")
        data = self._json(response, allow_empty=True)
        if isinstance(data, Mapping) and data:
            return JobInfo.from_json(data, request_id=self.last_request_id)
        return None

    def job(self, job_id: str) -> "Job":
        return Job(self, job_id)

    # ---------------------------------------------------------------- results
    def get_result(self, result_id: str) -> ResultInfo:
        response = self._request("GET", f"results/{result_id}")
        return ResultInfo.from_json(self._json(response), request_id=self.last_request_id)

    def download_result(self, result_id: str, path: str, *, format: str = "parquet", chunk_size: int = 1 << 20) -> int:
        """Stream ``GET /results/{id}/file`` to ``path``; returns the byte count."""
        written = 0
        headers = self._headers(auth="grant", accept="*/*")
        with self._http.stream("GET", self._path(f"results/{result_id}/file"), params={"format": format}, headers=headers) as response:
            self.last_request_id = response.headers.get("X-Request-Id")
            if response.status_code >= 400:
                response.read()
                raise self._error(response)
            with open(path, "wb") as handle:
                for chunk in response.iter_bytes(chunk_size):
                    handle.write(chunk)
                    written += len(chunk)
        return written

    # ------------------------------------------------------------ arrow pages
    def events_page(self, replay_id: str, *, cursor: str | None = None, limit: int | None = None) -> EventsPage:
        """One page of ``GET /replays/{id}/events`` as Arrow record batches."""
        params = {k: v for k, v in (("cursor", cursor), ("limit", limit)) if v is not None}
        response = self._request("GET", f"replays/{replay_id}/events", params=params, accept=ARROW_MEDIA_TYPE)
        batches = self._arrow_batches(response)
        page_events = response.headers.get("X-Lab-Page-Events")
        return EventsPage(
            batches=batches,
            next_cursor=response.headers.get("X-Lab-Next-Cursor") or None,
            page_events=int(page_events) if page_events and page_events.isdigit() else None,
            request_id=self.last_request_id,
        )

    def series(self, replay_id: str, *, res: str = "auto", start: Any = None, end: Any = None) -> SeriesPage:
        params: dict[str, Any] = {"res": res}
        if start is not None:
            params["from"] = ns_str(start, name="from")
        if end is not None:
            params["to"] = ns_str(end, name="to")
        response = self._request("GET", f"replays/{replay_id}/series", params=params, accept=ARROW_MEDIA_TYPE)
        import pyarrow as pa

        batches = self._arrow_batches(response)
        table = pa.Table.from_batches(batches) if batches else pa.table({})
        downsampled = response.headers.get("X-Lab-Downsampled")
        return SeriesPage(
            table=table,
            resolution=response.headers.get("X-Lab-Resolution"),
            downsampled=None if downsampled is None else downsampled.lower() == "true",
            request_id=self.last_request_id,
        )

    def snapshot_at(self, replay_id: str, t: Any) -> SnapshotResult:
        response = self._request("GET", f"replays/{replay_id}/snapshot", params={"t": ns_str(t, name="t")})
        return SnapshotResult.from_json(self._json(response), request_id=self.last_request_id)

    # ------------------------------------------------------------- internals
    @staticmethod
    def _path(route: str) -> str:
        return f"{API_PREFIX}/{route.lstrip('/')}"

    @staticmethod
    def _is_web_route(route: str) -> bool:
        head = route.lstrip("/").split("/", 1)[0].split("?", 1)[0]
        return head in WEB_ROUTES

    def _headers(self, *, auth: str | None, accept: str = "application/json") -> dict[str, str]:
        headers = {"Accept": accept}
        if auth == "api_key":
            headers["Authorization"] = f"Bearer {self._api_key}"
            headers["X-Requested-With"] = REQUESTED_WITH
        elif auth == "grant":
            headers["Authorization"] = f"Bearer {self.grant().token}"
        return headers

    def _backoff(self, attempt: int) -> float:
        delay = min(self.backoff_cap_s, self.backoff_base_s * (2**attempt))
        if self._jitter is not None:
            delay += self._jitter() * self.backoff_base_s
        return delay

    def _request(
        self,
        method: str,
        route: str,
        *,
        auth: str | None = "auto",
        json: Any = None,
        params: Mapping[str, Any] | None = None,
        accept: str = "application/json",
        idempotency_key: str | None = None,
    ) -> Any:
        """Send one logical request with re-mint on ``GRANT_EXPIRED`` and bounded retries on 429/503."""
        if auth == "auto":
            auth = "api_key" if self._is_web_route(route) else "grant"
        if method == "POST" and idempotency_key is None:
            idempotency_key = str(uuid.uuid4())
        started = self._clock()
        attempt = 0
        reminted = False
        while True:
            headers = self._headers(auth=auth, accept=accept)
            if idempotency_key is not None:
                headers["Idempotency-Key"] = idempotency_key
            try:
                response = self._http.request(method, self._path(route), json=json, params=dict(params or {}), headers=headers)
            except httpx.TransportError as exc:
                if attempt < self.max_retries and self._clock() - started < self.retry_budget_s:
                    self._sleep(self._backoff(attempt))
                    attempt += 1
                    continue
                raise errors.TransportError(f"{type(exc).__name__}: {exc}") from exc
            self.last_request_id = response.headers.get("X-Request-Id")
            if response.status_code < 400:
                return response
            info = self._error_info(response)
            if response.status_code == 401 and info.code == "GRANT_EXPIRED" and auth == "grant" and not reminted:
                # A distinct code so clients re-mint; once per call so a bad clock cannot loop.
                self._grant = None
                reminted = True
                continue
            if response.status_code in RETRY_STATUSES and attempt < self.max_retries:
                retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
                delay = self._backoff(attempt) if retry_after is None else min(retry_after, self.retry_after_cap_s)
                elapsed = self._clock() - started
                if elapsed + delay <= self.retry_budget_s:
                    self._sleep(delay)
                    attempt += 1
                    continue
            raise errors.error_from_info(
                info,
                status=response.status_code,
                retry_after_s=_retry_after_seconds(response.headers.get("Retry-After")),
            )

    def _error_info(self, response: Any) -> ErrorInfo:
        try:
            body = response.json()
        except ValueError:
            body = response.text
        return ErrorInfo.from_json(body, status=response.status_code, request_id=response.headers.get("X-Request-Id"))

    def _error(self, response: Any) -> errors.LabError:
        return errors.error_from_info(self._error_info(response), status=response.status_code)

    def _json(self, response: Any, *, allow_empty: bool = False) -> Any:
        if allow_empty and not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise errors.UnexpectedResponse(
                "expected a JSON body", request_id=self.last_request_id, status=response.status_code
            ) from exc

    def _arrow_batches(self, response: Any) -> list[Any]:
        import pyarrow as pa
        import pyarrow.ipc as ipc

        content_type = response.headers.get("Content-Type", "")
        if not response.content:
            return []
        if "json" in content_type:
            raise errors.UnexpectedResponse(
                "expected an Arrow IPC stream, got JSON (send Accept: application/vnd.apache.arrow.stream)",
                request_id=self.last_request_id,
                status=response.status_code,
            )
        try:
            reader = ipc.open_stream(pa.BufferReader(response.content))
            return list(reader)
        except pa.ArrowInvalid as exc:
            raise errors.UnexpectedResponse(f"invalid Arrow stream: {exc}", request_id=self.last_request_id) from exc


class Replay:
    """Handle for one prepared instrument window (``/replays``)."""

    def __init__(self, client: LabClient, info: ReplayInfo) -> None:
        self._client = client
        self.info = info

    @property
    def id(self) -> str:
        return self.info.replay_id

    @property
    def state(self) -> str:
        return self.info.state

    @property
    def request_id(self) -> str | None:
        return self.info.request_id

    def refresh(self) -> "Replay":
        self.info = self._client.get_replay(self.id).info
        return self

    def wait(self, timeout: float = 180.0, *, poll_interval_s: float = 1.0, poll_max_s: float = 10.0) -> "Replay":
        """Block until the replay is ready; a staging replay polls its job with a hard deadline."""
        if self.info.state == "ready":
            return self
        if self.info.state == "failed":
            raise errors.JobFailed(f"replay {self.id} failed", request_id=self.request_id, details=self.info.raw)
        deadline = self._client._clock() + timeout
        if self.info.job_id:
            job = Job(self._client, self.info.job_id, kind="replay")
            job.wait(timeout, poll_interval_s=poll_interval_s, poll_max_s=poll_max_s, fetch_result=False)
        self.refresh()
        polls = 0
        while self.info.state == "staging" and self._client._clock() < deadline and polls < 10_000:
            self._client._sleep(min(poll_interval_s, max(0.0, deadline - self._client._clock())))
            self.refresh()
            polls += 1
        if self.info.state == "ready":
            return self
        if self.info.state == "failed":
            raise errors.JobFailed(f"replay {self.id} failed", request_id=self.request_id, details=self.info.raw)
        raise errors.WaitTimeout(f"replay {self.id} still {self.info.state} after {timeout}s", job_id=self.info.job_id, last_state=self.info.state)

    def pages(self, *, cursor: str | None = None, limit: int | None = None, max_pages: int | None = None) -> Iterator[EventsPage]:
        """Cursor-driven pages; stops when ``X-Lab-Next-Cursor`` is absent (never loops on a repeated cursor)."""
        cursor = cursor if cursor is not None else self.info.first_cursor
        seen: set[str] = set()
        count = 0
        while True:
            page = self._client.events_page(self.id, cursor=cursor, limit=limit)
            yield page
            count += 1
            if not page.next_cursor or (max_pages is not None and count >= max_pages):
                return
            if page.next_cursor in seen or page.next_cursor == cursor:
                raise errors.UnexpectedResponse("engine repeated an events cursor", request_id=page.request_id)
            seen.add(page.next_cursor)
            cursor = page.next_cursor

    def events(self, *, cursor: str | None = None, limit: int | None = None) -> Iterator[Any]:
        """``pyarrow.RecordBatch`` pages of the exact event iterator."""
        for page in self.pages(cursor=cursor, limit=limit):
            yield from page.batches

    def snapshot_at(self, t: Any) -> SnapshotResult:
        return self._client.snapshot_at(self.id, t)

    def series(self, *, res: str = "auto", start: Any = None, end: Any = None) -> SeriesPage:
        return self._client.series(self.id, res=res, start=start, end=end)

    def simulate(self, **kwargs: Any) -> "Job":
        kwargs.setdefault("replay_id", self.id)
        return self._client.simulate(**kwargs)


class Job:
    """A job handle ("Jobs"). ``wait`` polls with bounded backoff and a hard deadline."""

    def __init__(
        self,
        client: LabClient,
        job_id: str,
        *,
        info: JobInfo | None = None,
        inline_result: Any = None,
        kind: str | None = None,
    ) -> None:
        self._client = client
        self.id = job_id
        self.info = info
        self.kind = kind or (info.kind if info else None)
        self._inline_result = inline_result

    @property
    def state(self) -> str:
        if self._inline_result is not None:
            return "succeeded"
        return self.info.state if self.info else "unknown"

    @property
    def done(self) -> bool:
        return self._inline_result is not None or (self.info is not None and self.info.terminal)

    @property
    def request_id(self) -> str | None:
        if self.info is not None:
            return self.info.request_id
        return getattr(self._inline_result, "request_id", None)

    def refresh(self) -> JobInfo:
        self.info = self._client.get_job(self.id)
        if self.kind is None:
            self.kind = self.info.kind
        return self.info

    def cancel(self) -> JobInfo | None:
        info = self._client.cancel_job(self.id)
        if info is not None:
            self.info = info
        return info

    def wait(
        self,
        timeout: float = 180.0,
        *,
        poll_interval_s: float = 1.0,
        poll_max_s: float = 10.0,
        fetch_result: bool = True,
    ) -> Any:
        """Poll ``GET /jobs/{id}`` until terminal or ``timeout`` seconds (monotonic) have passed.

        The delay grows from ``poll_interval_s`` to ``poll_max_s``, honours a
        ``Retry-After`` sent with the job body, and is clipped to the deadline;
        reaching the deadline raises ``WaitTimeout`` (client side) which is
        distinct from the engine's ``JOB_TIMED_OUT``. A failed job raises the
        typed error of its ``error.code``; ``cancelled`` raises ``JobCancelled``.
        """
        if self._inline_result is not None:
            return self._inline_result
        if timeout <= 0:
            raise errors.WaitTimeout("timeout must be positive", job_id=self.id)
        clock = self._client._clock
        deadline = clock() + timeout
        polls = 0
        delay = poll_interval_s
        info = self.info if self.info is not None and self.info.terminal else self.refresh()
        while not info.terminal:
            polls += 1
            remaining = deadline - clock()
            if remaining <= 0 or polls > 100_000:
                raise errors.WaitTimeout(
                    f"job {self.id} still {info.state} after {timeout}s", job_id=self.id, last_state=info.state, request_id=info.request_id
                )
            retry_after = _retry_after_seconds(info.raw.get("retry_after")) if isinstance(info.raw.get("retry_after"), str) else None
            step = delay if retry_after is None else max(retry_after, 0.0)
            self._client._sleep(min(step, remaining))
            delay = min(poll_max_s, delay * 2)
            info = self.refresh()
        return self._outcome(info, fetch_result=fetch_result)

    def _outcome(self, info: JobInfo, *, fetch_result: bool) -> Any:
        if info.state == "succeeded":
            if not fetch_result:
                return info
            return self.result(info)
        if info.state == "cancelled":
            raise errors.JobCancelled(f"job {self.id} was cancelled", request_id=info.request_id)
        if info.state == "timed_out":
            message = info.error.message if info.error else f"job {self.id} timed out on the engine"
            raise errors.JobTimedOut(message, request_id=info.request_id, details=info.error.details if info.error else {})
        if info.error is not None:
            raise errors.error_from_info(info.error)
        raise errors.JobFailed(f"job {self.id} failed", request_id=info.request_id, details=info.raw)

    def result(self, info: JobInfo | None = None) -> Any:
        """Fetch the job's result: a ``SimulationResult`` for simulations, else ``ResultInfo``."""
        info = info or self.info or self.refresh()
        if info.state != "succeeded":
            raise errors.InvalidRequest(f"job {self.id} is {info.state}, not succeeded", request_id=info.request_id)
        if not info.result_id:
            if (self.kind or info.kind) == "replay" and info.replay_id:
                return info
            return info
        result = self._client.get_result(info.result_id)
        kind = self.kind or info.kind or result.kind
        body = result.raw
        looks_like_simulation = isinstance(body, Mapping) and (
            "fills" in body or (isinstance(body.get("preview"), Mapping) and "fills" in body["preview"]) or (isinstance(body.get("result"), Mapping) and "fills" in body["result"])
        )
        if kind == "simulation" or looks_like_simulation:
            return SimulationResult.from_json(body, request_id=result.request_id)
        return result


__all__ = ["ARROW_MEDIA_TYPE", "DEFAULT_LAB_URL", "Job", "LabClient", "Replay", "SDK_VERSION"]
