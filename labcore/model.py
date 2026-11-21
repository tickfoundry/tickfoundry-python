"""Canonical internal data model for TickFoundry Lab.

Implements the canonical records (``Instrument``, ``BookSnapshot``,
``Trade``), the precision rules (int64 nanoseconds as Python
ints, ``Decimal`` prices/quantities, NaN/Infinity to ``null`` at JSON
boundaries, large IDs as strings), the total ordering key and the
level validation/normalization rules.

Nothing here reads venue files; adapters build these records and the rest of
labcore consumes them.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields, is_dataclass
from decimal import Decimal, InvalidOperation, localcontext
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .numeric import DECIMAL_CONTEXT

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
#: Largest integer JavaScript can hold exactly; larger ints are JSON strings.
JS_SAFE_INT = (1 << 53) - 1

# ---------------------------------------------------------------------------
# Quality flags
# ---------------------------------------------------------------------------
CROSSED_BOOK = "crossed_book"
MISSING_BID_SIDE = "missing_bid_side"
MISSING_ASK_SIDE = "missing_ask_side"
EMPTY_BOOK = "empty_book"
INVALID_LEVEL = "invalid_level"
TRUNCATED_DEPTH = "truncated_depth"
FLOAT_CONVERTED = "float_converted"

#: Flags that make a snapshot unusable as a book state ("last valid
#: snapshot"). A one-sided book is still a valid state; a crossed or empty
#: book is not.
BLOCKING_BOOK_FLAGS = frozenset({CROSSED_BOOK, EMPTY_BOOK})


class ClockKind(str, Enum):
    """Which clock ``analysis_time_ns`` was derived from."""

    collector_observed = "collector_observed"
    exchange_time = "exchange_time"
    reconstructed_block = "reconstructed_block"


class BookSide(str, Enum):
    bid = "bid"
    ask = "ask"


class AggressorSide(str, Enum):
    buy = "buy"
    sell = "sell"


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------
#: (analysis_time_ns, key_rank, int_key, str_key, stable_row_ordinal).
#: ``source_order_key`` may be an int, a str or absent; the rank keeps mixed
#: key types totally ordered without comparing int to str.
OrderingKey = tuple[int, int, int, str, int]


def ordering_key(
    analysis_time_ns: int,
    source_order_key: int | str | None,
    stable_row_ordinal: int,
) -> OrderingKey:
    """Total ordering key ``(analysis_time_ns, source_order_key, stable_row_ordinal)``.

    Every adapter defines a deterministic total ordering from native
    provenance plus a stable row ordinal. ``None`` keys sort before int keys,
    which sort before str keys; the ordinal breaks any remaining tie.
    """
    if source_order_key is None:
        return (analysis_time_ns, 0, 0, "", stable_row_ordinal)
    if isinstance(source_order_key, bool):
        raise TypeError("source_order_key must be int, str or None")
    if isinstance(source_order_key, int):
        return (analysis_time_ns, 1, source_order_key, "", stable_row_ordinal)
    return (analysis_time_ns, 2, 0, str(source_order_key), stable_row_ordinal)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Level:
    """One visible price level. ``quantity`` is in the instrument's quantity unit."""

    price: Decimal
    quantity: Decimal
    order_count: int | None = None


@dataclass(slots=True)
class Instrument:
    """Instrument descriptor.

    ``tick_size`` / ``lot_increment`` are the values resolved for the analysis
    time from the history references; ``contract_multiplier`` defaults to 1
    (outcome shares, spot base quantity).
    """

    venue: str
    product_type: str
    instrument_id: str
    native_symbol: str
    quantity_unit: str
    price_unit: str
    entitlement_unit_ids: list[str] = field(default_factory=list)
    outcome_id: str | None = None
    condition_id: str | None = None
    event_id: str | None = None
    base_asset: str | None = None
    quote_asset: str | None = None
    tick_size_history_ref: str | None = None
    lot_size_history_ref: str | None = None
    contract_multiplier: Decimal = Decimal(1)
    tick_size: Decimal | None = None
    lot_increment: Decimal | None = None


@dataclass(slots=True)
class BookSnapshot:
    """A visible order-book snapshot.

    ``analysis_time_ns`` is a normalized ordering coordinate, not a claim of
    nanosecond observation accuracy. ``full_depth_known`` is ``True`` when the
    venue guarantees the ladder is complete, ``False`` when it is known to be
    capped, ``None`` when unknown.
    """

    instrument_id: str
    analysis_time_ns: int
    bids: list[Level]
    asks: list[Level]
    clock_kind: ClockKind = ClockKind.collector_observed
    observed_time_ns: int | None = None
    source_time_ms: int | None = None
    source_order_key: int | str | None = None
    stable_row_ordinal: int = 0
    visible_depth: int | None = None
    full_depth_known: bool | None = None
    quality_flags: list[str] = field(default_factory=list)
    dataset_version: str | None = None
    source_fragment_id: str | None = None

    def ordering_key(self) -> OrderingKey:
        return ordering_key(self.analysis_time_ns, self.source_order_key, self.stable_row_ordinal)


@dataclass(slots=True)
class Trade:
    """A tape event. ``aggressor_side`` only when the source verifies it."""

    instrument_id: str
    analysis_time_ns: int
    price: Decimal
    quantity: Decimal
    clock_kind: ClockKind = ClockKind.collector_observed
    source_order_key: int | str | None = None
    stable_row_ordinal: int = 0
    aggressor_side: AggressorSide | None = None
    native_trade_id: str | None = None
    native_transaction_id: str | None = None
    fee: Decimal | None = None
    fee_asset: str | None = None
    quality_flags: list[str] = field(default_factory=list)
    dataset_version: str | None = None

    def ordering_key(self) -> OrderingKey:
        return ordering_key(self.analysis_time_ns, self.source_order_key, self.stable_row_ordinal)


# ---------------------------------------------------------------------------
# Validation and normalization
# ---------------------------------------------------------------------------
def coerce_decimal(value: Any) -> tuple[Decimal | None, bool]:
    """Best-effort conversion of a raw level value to a finite ``Decimal``.

    Returns ``(decimal_or_None, came_from_float)``. NaN, Infinity, ``None``,
    bools and unparsable strings become ``None`` (never replace a
    missing price or size with zero). Floats are converted through ``repr``;
    the caller records ``float_converted`` because upstream precision loss
    cannot be undone.
    """
    if value is None or isinstance(value, bool):
        return None, False
    if isinstance(value, Decimal):
        return (value if value.is_finite() else None), False
    if isinstance(value, int):
        return Decimal(value), False
    if isinstance(value, float):
        if not math.isfinite(value):
            return None, True
        return Decimal(repr(value)), True
    if isinstance(value, str):
        try:
            parsed = Decimal(value)
        except InvalidOperation:
            return None, False
        return (parsed if parsed.is_finite() else None), False
    return None, False


def normalize_levels(levels: Iterable[Any], side: BookSide) -> tuple[list[Level], int, bool]:
    """Drop invalid levels and sort: asks ascending, bids descending.

    Accepts ``Level`` objects, ``(price, quantity[, order_count])`` tuples or
    mappings with those keys. A level is invalid when its price or quantity is
    missing, non-finite, zero or negative. Returns ``(levels, dropped_count,
    any_float_converted)``. Equal prices keep their input order (stable sort).
    """
    out: list[Level] = []
    dropped = 0
    float_seen = False
    for raw in levels:
        if isinstance(raw, Level):
            price, quantity, order_count = raw.price, raw.quantity, raw.order_count
        elif isinstance(raw, Mapping):
            price, quantity, order_count = raw.get("price"), raw.get("quantity"), raw.get("order_count")
        else:
            price, quantity, *rest = raw
            order_count = rest[0] if rest else None
        p, fp = coerce_decimal(price)
        q, fq = coerce_decimal(quantity)
        float_seen = float_seen or fp or fq
        if p is None or q is None or p <= 0 or q <= 0:
            dropped += 1
            continue
        out.append(Level(p, q, order_count))
    out.sort(key=lambda lvl: lvl.price, reverse=(side is BookSide.bid))
    return out, dropped, float_seen


@dataclass(slots=True)
class ValidatedBook:
    """A snapshot after ``validate_book``: normalized ladders plus quality flags."""

    bids: list[Level]
    asks: list[Level]
    quality_flags: list[str]
    visible_depth: int
    full_depth_known: bool | None
    dropped_levels: int

    @property
    def is_valid(self) -> bool:
        """False when a blocking flag (crossed or empty) is present."""
        return not (BLOCKING_BOOK_FLAGS & set(self.quality_flags))

    @property
    def best_bid(self) -> Level | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Level | None:
        return self.asks[0] if self.asks else None

    @property
    def mid(self) -> Decimal | None:
        """``(best_bid + best_ask) / 2`` when both sides are valid."""
        if not self.bids or not self.asks or not self.is_valid:
            return None
        return (self.bids[0].price + self.asks[0].price) / Decimal(2)


def validate_book(snapshot: BookSnapshot, *, depth_cap: int | None = None) -> ValidatedBook:
    """Normalize a snapshot and compute its quality flags.

    Flags: ``invalid_level`` (a null/zero/negative price or size was dropped),
    ``empty_book`` (no valid level on either side), ``missing_bid_side`` /
    ``missing_ask_side`` (one side has no valid level), ``crossed_book``
    (best bid >= best ask), ``truncated_depth`` (``visible_depth`` reached
    ``depth_cap`` and ``full_depth_known`` is not ``True``: seeing exactly the
    maximum number of delivered levels does not prove there is no deeper
    liquidity), ``float_converted`` (a source value was a float). Flags the
    adapter already attached are preserved.
    """
    bids, dropped_bids, float_bids = normalize_levels(snapshot.bids, BookSide.bid)
    asks, dropped_asks, float_asks = normalize_levels(snapshot.asks, BookSide.ask)
    flags: list[str] = list(dict.fromkeys(snapshot.quality_flags))

    def add(flag: str) -> None:
        if flag not in flags:
            flags.append(flag)

    dropped = dropped_bids + dropped_asks
    if dropped:
        add(INVALID_LEVEL)
    if float_bids or float_asks:
        add(FLOAT_CONVERTED)
    if not bids and not asks:
        add(EMPTY_BOOK)
    else:
        if not bids:
            add(MISSING_BID_SIDE)
        if not asks:
            add(MISSING_ASK_SIDE)
    if bids and asks and bids[0].price >= asks[0].price:
        add(CROSSED_BOOK)

    delivered_depth = max(len(snapshot.bids), len(snapshot.asks))
    visible_depth = snapshot.visible_depth if snapshot.visible_depth is not None else delivered_depth
    if depth_cap is not None and visible_depth >= depth_cap and snapshot.full_depth_known is not True:
        add(TRUNCATED_DEPTH)

    return ValidatedBook(
        bids=bids,
        asks=asks,
        quality_flags=flags,
        visible_depth=visible_depth,
        full_depth_known=snapshot.full_depth_known,
        dropped_levels=dropped,
    )


def is_valid_snapshot(snapshot: BookSnapshot, *, depth_cap: int | None = None) -> bool:
    """True when the snapshot is a usable book state (not crossed, not empty)."""
    return validate_book(snapshot, depth_cap=depth_cap).is_valid


# ---------------------------------------------------------------------------
# JSON boundary
# ---------------------------------------------------------------------------
#: Integer fields with these suffixes are epoch nanoseconds / milliseconds and
#: always serialize as decimal strings, whatever their magnitude.
_TIME_FIELD_SUFFIXES = ("_ns", "_ms")


def to_jsonable(obj: Any, *, field_name: str | None = None) -> Any:
    """Convert labcore values to JSON-safe primitives.

    Rules: ``Decimal`` -> shortest exact decimal string, never
    exponent notation, no trailing zeros (non-finite -> ``null``); ints in
    ``*_ns`` / ``*_ms`` fields and any int outside the JavaScript safe range
    -> string; float NaN/Infinity -> ``null``; enums -> value; dataclasses ->
    objects; ``Fraction`` is refused (convert at the source).
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Decimal):
        if not obj.is_finite():
            return None
        with localcontext(DECIMAL_CONTEXT):
            return format(obj.normalize(), "f")
    if isinstance(obj, int):
        as_time = field_name is not None and field_name.endswith(_TIME_FIELD_SUFFIXES)
        if as_time or abs(obj) > JS_SAFE_INT:
            return str(obj)
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name), field_name=f.name) for f in fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v, field_name=str(k)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v, field_name=field_name) for v in obj]
    raise TypeError(f"cannot serialize {type(obj).__name__} to JSON")


def json_dumps(obj: Any, **kwargs: Any) -> str:
    """``json.dumps`` with labcore conversions; NaN can never leak (``allow_nan=False``)."""
    kwargs.setdefault("separators", (",", ":"))
    return json.dumps(to_jsonable(obj), allow_nan=False, **kwargs)


def parse_int64(value: Any, *, name: str = "value") -> int:
    """Parse an int64 from a JSON string or int; refuse floats and out-of-range values."""
    if isinstance(value, bool):
        raise TypeError(f"{name}: bool is not an integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        parsed = int(value.strip())
    elif isinstance(value, float):
        raise TypeError(f"{name}: floats cannot carry a full-precision int64")
    else:
        raise TypeError(f"{name}: unsupported type {type(value).__name__}")
    if not INT64_MIN <= parsed <= INT64_MAX:
        raise ValueError(f"{name}: {parsed} is outside int64")
    return parsed


def parse_optional_int64(value: Any, *, name: str = "value") -> int | None:
    return None if value is None else parse_int64(value, name=name)


def parse_decimal(value: Any, *, name: str = "value") -> Decimal | None:
    """Parse a JSON decimal string (or int) to ``Decimal``; ``null`` stays ``None``."""
    if value is None:
        return None
    if isinstance(value, float):
        raise TypeError(f"{name}: floats are not accepted at the JSON boundary")
    parsed, _ = coerce_decimal(value)
    if parsed is None:
        raise ValueError(f"{name}: not a finite decimal: {value!r}")
    return parsed


def _parse_order_key(value: Any) -> int | str | None:
    if value is None or isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    raise TypeError("source_order_key must be int, str or null")


def level_from_json(data: Mapping[str, Any]) -> Level:
    price = parse_decimal(data.get("price"), name="price")
    quantity = parse_decimal(data.get("quantity"), name="quantity")
    if price is None or quantity is None:
        raise ValueError("level requires price and quantity")
    order_count = data.get("order_count")
    return Level(price, quantity, None if order_count is None else int(order_count))


def book_snapshot_from_json(data: Mapping[str, Any]) -> BookSnapshot:
    """Inverse of ``to_jsonable`` for ``BookSnapshot`` (full-precision ints)."""
    return BookSnapshot(
        instrument_id=str(data["instrument_id"]),
        analysis_time_ns=parse_int64(data["analysis_time_ns"], name="analysis_time_ns"),
        bids=[level_from_json(lvl) for lvl in data.get("bids", [])],
        asks=[level_from_json(lvl) for lvl in data.get("asks", [])],
        clock_kind=ClockKind(data.get("clock_kind", ClockKind.collector_observed.value)),
        observed_time_ns=parse_optional_int64(data.get("observed_time_ns"), name="observed_time_ns"),
        source_time_ms=parse_optional_int64(data.get("source_time_ms"), name="source_time_ms"),
        source_order_key=_parse_order_key(data.get("source_order_key")),
        stable_row_ordinal=int(data.get("stable_row_ordinal", 0)),
        visible_depth=None if data.get("visible_depth") is None else int(data["visible_depth"]),
        full_depth_known=data.get("full_depth_known"),
        quality_flags=list(data.get("quality_flags", [])),
        dataset_version=data.get("dataset_version"),
        source_fragment_id=data.get("source_fragment_id"),
    )


def trade_from_json(data: Mapping[str, Any]) -> Trade:
    """Inverse of ``to_jsonable`` for ``Trade``."""
    price = parse_decimal(data.get("price"), name="price")
    quantity = parse_decimal(data.get("quantity"), name="quantity")
    if price is None or quantity is None:
        raise ValueError("trade requires price and quantity")
    aggressor = data.get("aggressor_side")
    return Trade(
        instrument_id=str(data["instrument_id"]),
        analysis_time_ns=parse_int64(data["analysis_time_ns"], name="analysis_time_ns"),
        price=price,
        quantity=quantity,
        clock_kind=ClockKind(data.get("clock_kind", ClockKind.collector_observed.value)),
        source_order_key=_parse_order_key(data.get("source_order_key")),
        stable_row_ordinal=int(data.get("stable_row_ordinal", 0)),
        aggressor_side=None if aggressor is None else AggressorSide(aggressor),
        native_trade_id=data.get("native_trade_id"),
        native_transaction_id=data.get("native_transaction_id"),
        fee=parse_decimal(data.get("fee"), name="fee"),
        fee_asset=data.get("fee_asset"),
        quality_flags=list(data.get("quality_flags", [])),
        dataset_version=data.get("dataset_version"),
    )


def instrument_from_json(data: Mapping[str, Any]) -> Instrument:
    """Inverse of ``to_jsonable`` for ``Instrument``."""
    multiplier = parse_decimal(data.get("contract_multiplier"), name="contract_multiplier")
    return Instrument(
        venue=str(data["venue"]),
        product_type=str(data["product_type"]),
        instrument_id=str(data["instrument_id"]),
        native_symbol=str(data["native_symbol"]),
        quantity_unit=str(data["quantity_unit"]),
        price_unit=str(data["price_unit"]),
        entitlement_unit_ids=[str(x) for x in data.get("entitlement_unit_ids", [])],
        outcome_id=data.get("outcome_id"),
        condition_id=data.get("condition_id"),
        event_id=data.get("event_id"),
        base_asset=data.get("base_asset"),
        quote_asset=data.get("quote_asset"),
        tick_size_history_ref=data.get("tick_size_history_ref"),
        lot_size_history_ref=data.get("lot_size_history_ref"),
        contract_multiplier=Decimal(1) if multiplier is None else multiplier,
        tick_size=parse_decimal(data.get("tick_size"), name="tick_size"),
        lot_increment=parse_decimal(data.get("lot_increment"), name="lot_increment"),
    )


def sort_snapshots(snapshots: Sequence[BookSnapshot]) -> list[BookSnapshot]:
    """Sort by the total ordering key (stable, idempotent)."""
    return sorted(snapshots, key=BookSnapshot.ordering_key)


def sort_trades(trades: Sequence[Trade]) -> list[Trade]:
    return sorted(trades, key=Trade.ordering_key)
