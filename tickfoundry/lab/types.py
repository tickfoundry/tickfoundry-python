"""Contract types of ``lab.v1`` as plain dataclasses.

Everything that crosses the JSON boundary keeps the API's conventions:
nanosecond timestamps and exact numbers (prices, sizes,
notionals, ids) are strings, dates are ``YYYY-MM-DD``, instants are RFC 3339
UTC. The dataclasses are tolerant readers: unknown keys are kept in ``raw``
so a newer engine never breaks an older SDK, and missing keys become
``None`` rather than raising.

Timestamps a caller passes to the SDK may be RFC 3339 strings, decimal
nanosecond strings, ints (nanoseconds) or ``datetime`` objects; ``to_ns``
turns them into an exact int64 without ever going through a float.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "lab.v1"
NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_RFC3339 = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[Tt ](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?\s*(Z|z|[+-]\d{2}:?\d{2})?$"
)
_INT = re.compile(r"^-?\d+$")


def to_ns(value: Any, *, name: str = "timestamp") -> int:
    """Convert a caller-supplied instant to epoch nanoseconds (exact).

    Accepts an int (already nanoseconds), a decimal-digit string (nanoseconds
    as the contract carries them), an RFC 3339 string with up to nine
    fractional digits (``2026-08-19T20:15:00Z``, ``...+00:00``), a
    ``datetime`` (naive means UTC) or a ``numpy.datetime64``. Floats are
    refused because a float cannot carry a full-precision int64.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name}: bool is not a timestamp")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        raise TypeError(f"{name}: floats cannot carry a full-precision int64; pass ns as int or str")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        delta: timedelta = value - _EPOCH
        return (delta.days * 86_400 + delta.seconds) * NS_PER_S + delta.microseconds * 1_000
    if isinstance(value, str):
        text = value.strip()
        if _INT.match(text):
            return int(text)
        match = _RFC3339.match(text)
        if not match:
            raise ValueError(f"{name}: not an RFC 3339 instant or a nanosecond string: {value!r}")
        year, month, day, hour, minute, second, frac, tz = match.groups()
        base = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), tzinfo=timezone.utc)
        offset = 0
        if tz and tz not in ("Z", "z"):
            sign = 1 if tz[0] == "+" else -1
            digits = tz[1:].replace(":", "")
            offset = sign * (int(digits[:2]) * 3600 + int(digits[2:]) * 60)
        seconds = (base - _EPOCH).days * 86_400 + (base - _EPOCH).seconds - offset
        frac_ns = int((frac or "0").ljust(9, "0"))
        return seconds * NS_PER_S + frac_ns
    try:  # numpy is a dependency of pyarrow-based callers but not of this module
        import numpy as np  # noqa: WPS433

        if isinstance(value, np.datetime64):
            return int(value.astype("datetime64[ns]").astype("int64"))
        if isinstance(value, np.integer):
            return int(value)
    except ImportError:  # pragma: no cover
        pass
    raise TypeError(f"{name}: unsupported timestamp type {type(value).__name__}")


def ns_to_rfc3339(ns: int) -> str:
    """Epoch nanoseconds to an RFC 3339 UTC string with nanosecond precision."""
    seconds, rem = divmod(int(ns), NS_PER_S)
    instant = _EPOCH + timedelta(seconds=seconds)
    return instant.strftime("%Y-%m-%dT%H:%M:%S") + f".{rem:09d}Z"


def ns_str(value: Any, *, name: str = "timestamp") -> str:
    """Contract form of a timestamp: a decimal int64 string."""
    return str(to_ns(value, name=name))


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    return int(value)


def _opt_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


# ---------------------------------------------------------------------------
# Errors and grants
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ErrorInfo:
    """The ``error`` object of an error response."""

    code: str
    message: str
    request_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, body: Any, *, status: int | None = None, request_id: str | None = None) -> "ErrorInfo":
        err = body.get("error") if isinstance(body, Mapping) else None
        if not isinstance(err, Mapping):
            code = "HTTP_ERROR" if status is None else f"HTTP_{status}"
            message = body if isinstance(body, str) else "non-JSON error response"
            return cls(code=code, message=str(message)[:500], request_id=request_id)
        return cls(
            code=str(err.get("code") or (f"HTTP_{status}" if status else "UNKNOWN")),
            message=str(err.get("message") or ""),
            request_id=_opt_str(err.get("request_id")) or request_id,
            details=_dict(err.get("details")),
        )


@dataclass(frozen=True, slots=True)
class Grant:
    """A minted grant (``POST /api/lab/v1/grant``). ``token`` is the JWS."""

    token: str
    expires_at: str | None
    policy_version: int | None
    request_id: str | None = None
    minted_at_wall_s: float = 0.0

    def expires_at_epoch_s(self) -> float | None:
        if not self.expires_at:
            return None
        try:
            return to_ns(self.expires_at, name="expires_at") / NS_PER_S
        except (ValueError, TypeError):
            return None


# ---------------------------------------------------------------------------
# Scopes and replays
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Fragment:
    fragment_id: str
    instrument_id: str | None
    date: str | None
    product: str | None
    bytes: int | None
    row_count: int | None
    depth: int | None
    quality_status: str | None
    cached: bool | None

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Fragment":
        return cls(
            fragment_id=str(data.get("fragment_id")),
            instrument_id=_opt_str(data.get("instrument_id")),
            date=_opt_str(data.get("date")),
            product=_opt_str(data.get("product")),
            bytes=_opt_int(data.get("bytes")),
            row_count=_opt_int(data.get("row_count")),
            depth=_opt_int(data.get("depth")),
            quality_status=_opt_str(data.get("quality_status")),
            cached=data.get("cached"),
        )


@dataclass(frozen=True, slots=True)
class Scope:
    """``POST /scopes`` / ``GET /scopes/{id}`` response."""

    id: str
    expires_at: str | None
    fragments: list[Fragment]
    estimate: dict[str, Any]
    effective_depth: int | None
    request_id: str | None
    raw: dict[str, Any]

    @classmethod
    def from_json(cls, data: Mapping[str, Any], *, request_id: str | None = None) -> "Scope":
        return cls(
            id=str(data.get("scope_id") or data.get("id")),
            expires_at=_opt_str(data.get("expires_at")),
            fragments=[Fragment.from_json(f) for f in _list(data.get("fragments")) if isinstance(f, Mapping)],
            estimate=_dict(data.get("estimate")),
            effective_depth=_opt_int(data.get("effective_depth")),
            request_id=request_id or _opt_str(data.get("request_id")),
            raw=dict(data),
        )


@dataclass(frozen=True, slots=True)
class ReplayInfo:
    """``POST /replays`` / ``GET /replays/{id}`` response."""

    replay_id: str
    state: str
    job_id: str | None
    coverage: dict[str, Any]
    event_count: int | None
    first_cursor: str | None
    resolutions: list[str]
    depth: int | None
    clock_kind: str | None
    seed_state: str | None
    expires_at: str | None
    data_quality: dict[str, Any]
    request_id: str | None
    raw: dict[str, Any]

    @classmethod
    def from_json(cls, data: Mapping[str, Any], *, request_id: str | None = None) -> "ReplayInfo":
        return cls(
            replay_id=str(data.get("replay_id") or data.get("id")),
            state=str(data.get("state") or "unknown"),
            job_id=_opt_str(data.get("job_id")),
            coverage=_dict(data.get("coverage")),
            event_count=_opt_int(data.get("event_count")),
            first_cursor=_opt_str(data.get("first_cursor")),
            resolutions=[str(r) for r in _list(data.get("resolutions"))],
            depth=_opt_int(data.get("depth")),
            clock_kind=_opt_str(data.get("clock_kind")),
            seed_state=_opt_str(data.get("seed_state")),
            expires_at=_opt_str(data.get("expires_at")),
            data_quality=_dict(data.get("data_quality")),
            request_id=request_id or _opt_str(data.get("request_id")),
            raw=dict(data),
        )


# ---------------------------------------------------------------------------
# Books and snapshots (``/snapshot``)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Level:
    price: str
    quantity: str
    order_count: int | None = None

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Level":
        return cls(str(data["price"]), str(data["quantity"]), _opt_int(data.get("order_count")))


@dataclass(frozen=True, slots=True)
class Snapshot:
    """The ``snapshot`` object of ``GET /replays/{id}/snapshot``."""

    analysis_time_ns: str
    observed_time_ns: str | None
    source_time_ms: str | None
    clock_kind: str | None
    bids: list[Level]
    asks: list[Level]
    visible_depth: int | None
    full_depth_known: bool | None
    quality_flags: list[str]

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Snapshot":
        return cls(
            analysis_time_ns=str(data.get("analysis_time_ns")),
            observed_time_ns=_opt_str(data.get("observed_time_ns")),
            source_time_ms=_opt_str(data.get("source_time_ms")),
            clock_kind=_opt_str(data.get("clock_kind")),
            bids=[Level.from_json(l) for l in _list(data.get("bids"))],
            asks=[Level.from_json(l) for l in _list(data.get("asks"))],
            visible_depth=_opt_int(data.get("visible_depth")),
            full_depth_known=data.get("full_depth_known"),
            quality_flags=[str(f) for f in _list(data.get("quality_flags"))],
        )

    @property
    def best_bid(self) -> Level | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Level | None:
        return self.asks[0] if self.asks else None


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    t_ns: str
    snapshot: Snapshot | None
    age_ns: str | None
    state: str
    ordinal: int | None
    request_id: str | None
    raw: dict[str, Any]

    @classmethod
    def from_json(cls, data: Mapping[str, Any], *, request_id: str | None = None) -> "SnapshotResult":
        snap = data.get("snapshot")
        return cls(
            t_ns=str(data.get("t_ns")),
            snapshot=Snapshot.from_json(snap) if isinstance(snap, Mapping) else None,
            age_ns=_opt_str(data.get("age_ns")),
            state=str(data.get("state") or "unknown"),
            ordinal=_opt_int(data.get("ordinal")),
            request_id=request_id or _opt_str(data.get("request_id")),
            raw=dict(data),
        )


# ---------------------------------------------------------------------------
# Simulations (inputs and outputs of ``POST /simulations``)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Fill:
    price: str
    quantity: str
    notional: str

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Fill":
        return cls(str(data["price"]), str(data["quantity"]), str(data["notional"]))


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Simulation outputs as strings; ``raw`` keeps every field the engine sent."""

    status: str
    requested_quantity: str | None
    filled_quantity: str | None
    residual_quantity: str | None
    residual_budget: str | None
    vwap: str | None
    notional: str | None
    fees: str | None
    fee_asset: str | None
    fee_model: str | None
    fills: list[Fill]
    levels_consumed: int | None
    best_quote_reference: str | None
    mid_reference: str | None
    reference_price: str | None
    reference_kind: str | None
    adverse_price_difference: str | None
    relative_slippage_bps: str | None
    price_unit: str | None
    quantity_unit: str | None
    quote_currency: str | None
    snapshot_age_ns: str | None
    decision_snapshot_ordinal: int | None
    execution_snapshot_ordinal: int | None
    visible_depth: int | None
    full_depth_known: bool | None
    quality_flags: list[str]
    warnings: list[Any]
    versions: dict[str, Any]
    request_id: str | None
    raw: dict[str, Any]

    @classmethod
    def from_json(cls, data: Mapping[str, Any], *, request_id: str | None = None) -> "SimulationResult":
        # A result fetched through /results/{id} may nest the simulation body.
        body: Mapping[str, Any] = data
        for key in ("result", "preview"):
            nested = data.get(key)
            if "status" not in data and isinstance(nested, Mapping) and "status" in nested:
                body = nested
                break
        return cls(
            status=str(body.get("status") or "unknown"),
            requested_quantity=_opt_str(body.get("requested_quantity")),
            filled_quantity=_opt_str(body.get("filled_quantity")),
            residual_quantity=_opt_str(body.get("residual_quantity")),
            residual_budget=_opt_str(body.get("residual_budget")),
            vwap=_opt_str(body.get("vwap")),
            notional=_opt_str(body.get("notional")),
            fees=_opt_str(body.get("fees")),
            fee_asset=_opt_str(body.get("fee_asset")),
            fee_model=_opt_str(body.get("fee_model")),
            fills=[Fill.from_json(f) for f in _list(body.get("fills")) if isinstance(f, Mapping)],
            levels_consumed=_opt_int(body.get("levels_consumed")),
            best_quote_reference=_opt_str(body.get("best_quote_reference")),
            mid_reference=_opt_str(body.get("mid_reference")),
            reference_price=_opt_str(body.get("reference_price")),
            reference_kind=_opt_str(body.get("reference_kind")),
            adverse_price_difference=_opt_str(body.get("adverse_price_difference")),
            relative_slippage_bps=_opt_str(body.get("relative_slippage_bps")),
            price_unit=_opt_str(body.get("price_unit")),
            quantity_unit=_opt_str(body.get("quantity_unit")),
            quote_currency=_opt_str(body.get("quote_currency")),
            snapshot_age_ns=_opt_str(body.get("snapshot_age_ns")),
            decision_snapshot_ordinal=_opt_int(body.get("decision_snapshot_ordinal")),
            execution_snapshot_ordinal=_opt_int(body.get("execution_snapshot_ordinal")),
            visible_depth=_opt_int(body.get("visible_depth")),
            full_depth_known=body.get("full_depth_known"),
            quality_flags=[str(f) for f in _list(body.get("quality_flags"))],
            warnings=_list(body.get("warnings")),
            versions=_dict(body.get("versions")),
            request_id=request_id or _opt_str(data.get("request_id")),
            raw=dict(data),
        )


# ---------------------------------------------------------------------------
# Jobs and results ("Jobs", "Results")
# ---------------------------------------------------------------------------
JOB_TERMINAL_STATES = frozenset({"succeeded", "failed", "timed_out", "cancelled"})
JOB_STATES = frozenset(
    {"queued", "staging", "running", "publishing", "succeeded", "failed", "timed_out", "cancelling", "cancelled"}
)


@dataclass(frozen=True, slots=True)
class JobProgress:
    phase: str | None
    bytes_staged: int | None
    bytes_total_estimate: int | None
    queue_position: int | None

    @classmethod
    def from_json(cls, data: Any) -> "JobProgress":
        d = _dict(data)
        return cls(
            phase=_opt_str(d.get("phase")),
            bytes_staged=_opt_int(d.get("bytes_staged")),
            bytes_total_estimate=_opt_int(d.get("bytes_total_estimate")),
            queue_position=_opt_int(d.get("queue_position")),
        )


@dataclass(frozen=True, slots=True)
class JobUsage:
    cpu_s: float | None
    peak_rss_bytes: int | None
    bytes_fetched: int | None
    bytes_staged: int | None

    @classmethod
    def from_json(cls, data: Any) -> "JobUsage":
        d = _dict(data)
        cpu = d.get("cpu_s")
        return cls(
            cpu_s=None if cpu is None else float(cpu),
            peak_rss_bytes=_opt_int(d.get("peak_rss_bytes")),
            bytes_fetched=_opt_int(d.get("bytes_fetched")),
            bytes_staged=_opt_int(d.get("bytes_staged")),
        )


@dataclass(frozen=True, slots=True)
class JobInfo:
    job_id: str
    kind: str | None
    state: str
    progress: JobProgress
    created_at: str | None
    started_at: str | None
    finished_at: str | None
    error: ErrorInfo | None
    result_id: str | None
    replay_id: str | None
    usage: JobUsage
    request_id: str | None
    raw: dict[str, Any]

    @classmethod
    def from_json(cls, data: Mapping[str, Any], *, request_id: str | None = None) -> "JobInfo":
        err = data.get("error")
        return cls(
            job_id=str(data.get("job_id") or data.get("id")),
            kind=_opt_str(data.get("kind")),
            state=str(data.get("state") or "unknown"),
            progress=JobProgress.from_json(data.get("progress")),
            created_at=_opt_str(data.get("created_at")),
            started_at=_opt_str(data.get("started_at")),
            finished_at=_opt_str(data.get("finished_at")),
            error=ErrorInfo.from_json({"error": err}, request_id=request_id) if isinstance(err, Mapping) else None,
            result_id=_opt_str(data.get("result_id")),
            replay_id=_opt_str(data.get("replay_id")),
            usage=JobUsage.from_json(data.get("usage")),
            request_id=request_id or _opt_str(data.get("request_id")),
            raw=dict(data),
        )

    @property
    def terminal(self) -> bool:
        return self.state in JOB_TERMINAL_STATES


@dataclass(frozen=True, slots=True)
class ResultInfo:
    """``GET /results/{id}``: metadata plus a bounded preview."""

    result_id: str
    kind: str | None
    bytes: int | None
    rows: int | None
    expires_at: str | None
    schema: Any
    preview: Any
    request_id: str | None
    raw: dict[str, Any]

    @classmethod
    def from_json(cls, data: Mapping[str, Any], *, request_id: str | None = None) -> "ResultInfo":
        return cls(
            result_id=str(data.get("result_id") or data.get("id")),
            kind=_opt_str(data.get("kind")),
            bytes=_opt_int(data.get("bytes")),
            rows=_opt_int(data.get("rows")),
            expires_at=_opt_str(data.get("expires_at")),
            schema=data.get("schema"),
            preview=data.get("preview"),
            request_id=request_id or _opt_str(data.get("request_id")),
            raw=dict(data),
        )


# ---------------------------------------------------------------------------
# Arrow pages (``/events`` and ``/series``)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EventsPage:
    """One ``GET /replays/{id}/events`` page: Arrow batches plus the cursor headers."""

    batches: list[Any]
    next_cursor: str | None
    page_events: int | None
    request_id: str | None


@dataclass(frozen=True, slots=True)
class SeriesPage:
    """One ``GET /replays/{id}/series`` response as a ``pyarrow.Table``."""

    table: Any
    resolution: str | None
    downsampled: bool | None
    request_id: str | None


def event_schema(*, with_order_counts: bool = False):
    """The Arrow schema of the exact event iterator (``/events``).

    ``bid_n`` / ``ask_n`` exist only on Hyperliquid pages; pass
    ``with_order_counts=True`` to include them. Built lazily because pyarrow is
    optional for pure type consumers.
    """
    import pyarrow as pa

    fields = [
        pa.field("ordinal", pa.int64(), nullable=False),
        pa.field("t_ns", pa.int64(), nullable=False),
        pa.field("kind", pa.dictionary(pa.int8(), pa.string()), nullable=False),
        pa.field("source_time_ms", pa.int64()),
        pa.field("source_seq", pa.int64()),
        pa.field("ordering_uncertain", pa.bool_(), nullable=False),
        pa.field("best_bid", pa.float64()),
        pa.field("best_bid_size", pa.float64()),
        pa.field("best_ask", pa.float64()),
        pa.field("best_ask_size", pa.float64()),
        pa.field("bid_px", pa.list_(pa.float64())),
        pa.field("bid_sz", pa.list_(pa.float64())),
        pa.field("ask_px", pa.list_(pa.float64())),
        pa.field("ask_sz", pa.list_(pa.float64())),
    ]
    if with_order_counts:
        fields += [pa.field("bid_n", pa.list_(pa.int32())), pa.field("ask_n", pa.list_(pa.int32()))]
    fields += [
        pa.field("trade_price", pa.float64()),
        pa.field("trade_size", pa.float64()),
        pa.field("trade_side", pa.dictionary(pa.int8(), pa.string())),
        pa.field("trade_id", pa.string()),
        pa.field("quality_flags", pa.list_(pa.string()), nullable=False),
    ]
    return pa.schema(fields)


__all__ = [
    "JOB_STATES",
    "JOB_TERMINAL_STATES",
    "NS_PER_MS",
    "NS_PER_S",
    "SCHEMA_VERSION",
    "ErrorInfo",
    "EventsPage",
    "Fill",
    "Fragment",
    "Grant",
    "JobInfo",
    "JobProgress",
    "JobUsage",
    "Level",
    "ReplayInfo",
    "ResultInfo",
    "Scope",
    "SeriesPage",
    "SimulationResult",
    "Snapshot",
    "SnapshotResult",
    "event_schema",
    "ns_str",
    "ns_to_rfc3339",
    "to_ns",
]
