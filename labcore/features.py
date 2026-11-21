"""Per-snapshot book features and time-weighted bin aggregation.

Bins are half-open UTC intervals ``[start, end)``. Book features are
time-weighted means over the duration for which a valid book state is
known; a prior state is carried forward only until the next snapshot, a
known gap, an invalidating snapshot, the configured staleness bound or the
window end, and integration is clipped at bin boundaries. A bin never uses
data recorded at or after its end (no look-ahead).

Trade aggregates respect coverage: an empty bin with complete capture
coverage is zero trades, an empty bin during missing coverage is ``null``.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from enum import Enum
from fractions import Fraction
from typing import Iterable, Sequence

from .model import (
    BLOCKING_BOOK_FLAGS,
    BookSnapshot,
    Instrument,
    Trade,
    sort_snapshots,
    sort_trades,
    validate_book,
)
from .asof import merge_intervals
from .numeric import DECIMAL_CONTEXT, VALUE_PRECISION_DP, fraction_to_decimal, quantize_fraction
from .version import ENGINE_VERSION, SCHEMA_VERSION

# Bin flags.
BELOW_COVERAGE_THRESHOLD = "below_coverage_threshold"
PARTIAL_WINDOW_BIN = "partial_window_bin"
KNOWN_GAP_OVERLAP = "known_gap_overlap"
NO_VALID_STATE = "no_valid_state"
STALE_STATE_CUTOFF = "stale_state_cutoff"
INCOMPLETE_BIN = "incomplete_bin"
TRADES_UNKNOWN_MISSING_COVERAGE = "trades_unknown_missing_coverage"
VALUES_NULLED_BELOW_THRESHOLD = "values_nulled_below_threshold"


class SegmentEnd(str, Enum):
    """Why a carried-forward state stopped contributing."""

    next_snapshot = "next_snapshot"
    known_gap = "known_gap"
    stale = "stale"
    window_end = "window_end"
    open = "open"


@dataclass(frozen=True, slots=True)
class FeaturePolicy:
    """Aggregation policy, echoed in result metadata.

    ``coverage_threshold`` is the valid-duration fraction below which a bin is
    flagged (default 0.90); ``max_state_age_ns`` bounds carry-forward;
    ``null_below_threshold`` additionally nulls book features of flagged bins.
    """

    depth_levels: int = 1
    coverage_threshold: Decimal = Decimal("0.90")
    max_state_age_ns: int | None = None
    null_below_threshold: bool = False
    depth_cap: int | None = None
    value_precision_dp: int = VALUE_PRECISION_DP


@dataclass(frozen=True, slots=True)
class BookState:
    """Features of one validated snapshot.

    ``valid`` is ``False`` for crossed/empty books; such a snapshot stops the
    carry-forward of the previous state and contributes nothing itself.
    """

    time_ns: int
    valid: bool
    mid: Decimal | None
    spread: Decimal | None
    microprice: Decimal | None
    depth_imbalance: Decimal | None
    bid_quantity: Decimal | None
    ask_quantity: Decimal | None
    bid_depth_notional: Decimal | None
    ask_depth_notional: Decimal | None
    visible_quote_depth: Decimal | None
    depth_levels: int
    quality_flags: tuple[str, ...]


def book_state(
    snapshot: BookSnapshot,
    *,
    depth_levels: int = 1,
    contract_multiplier: Decimal = Decimal(1),
    depth_cap: int | None = None,
    precision_dp: int = VALUE_PRECISION_DP,
) -> BookState:
    """Compute the book features of one snapshot.

    Mid and spread need both sides; microprice is ``None`` on a zero
    denominator; depth imbalance uses the top ``depth_levels`` of each side
    (the caller must pass the permitted depth); visible quote depth is
    ``sum(price * quantity * multiplier)`` per side in the quote currency.
    """
    if depth_levels < 1:
        raise ValueError("depth_levels must be >= 1")
    book = validate_book(snapshot, depth_cap=depth_cap)
    valid = book.is_valid
    with localcontext(DECIMAL_CONTEXT):
        bids = book.bids[:depth_levels]
        asks = book.asks[:depth_levels]
        bid_qty = sum((lvl.quantity for lvl in bids), Decimal(0)) if bids else None
        ask_qty = sum((lvl.quantity for lvl in asks), Decimal(0)) if asks else None
        bid_notional = sum((lvl.price * lvl.quantity * contract_multiplier for lvl in bids), Decimal(0)) if bids else None
        ask_notional = sum((lvl.price * lvl.quantity * contract_multiplier for lvl in asks), Decimal(0)) if asks else None
        visible_depth = None
        if bid_notional is not None or ask_notional is not None:
            visible_depth = (bid_notional or Decimal(0)) + (ask_notional or Decimal(0))

        mid = spread = microprice = imbalance = None
        if valid and bids and asks:
            best_bid, best_ask = bids[0], asks[0]
            mid = (best_bid.price + best_ask.price) / Decimal(2)
            spread = best_ask.price - best_bid.price
            denominator = best_bid.quantity + best_ask.quantity
            if denominator > 0:
                micro = (Fraction(best_ask.price) * Fraction(best_bid.quantity) + Fraction(best_bid.price) * Fraction(best_ask.quantity)) / Fraction(denominator)
                microprice, _ = fraction_to_decimal(micro, precision_dp)
        if valid and (bid_qty or ask_qty):
            total = (bid_qty or Decimal(0)) + (ask_qty or Decimal(0))
            if total > 0:
                imb = (Fraction(bid_qty or 0) - Fraction(ask_qty or 0)) / Fraction(total)
                imbalance, _ = fraction_to_decimal(imb, precision_dp)
        if not valid:
            bid_qty = ask_qty = bid_notional = ask_notional = visible_depth = None

    return BookState(
        time_ns=snapshot.analysis_time_ns,
        valid=valid,
        mid=mid,
        spread=spread,
        microprice=microprice,
        depth_imbalance=imbalance,
        bid_quantity=bid_qty,
        ask_quantity=ask_qty,
        bid_depth_notional=bid_notional,
        ask_depth_notional=ask_notional,
        visible_quote_depth=visible_depth,
        depth_levels=depth_levels,
        quality_flags=tuple(book.quality_flags),
    )


@dataclass(frozen=True, slots=True)
class Bin:
    """Half-open interval ``[start_ns, end_ns)`` in UTC epoch nanoseconds."""

    start_ns: int
    end_ns: int

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


def make_bins(start_ns: int, end_ns: int, width_ns: int) -> list[Bin]:
    """Grid-aligned bins covering ``[start_ns, end_ns)``.

    The first bin starts at ``floor(start / width) * width`` so edge bins are
    partial when the window is not aligned; their coverage reflects that.
    """
    if width_ns <= 0:
        raise ValueError("width_ns must be positive")
    if end_ns <= start_ns:
        return []
    bins: list[Bin] = []
    cursor = (start_ns // width_ns) * width_ns
    while cursor < end_ns:
        bins.append(Bin(cursor, cursor + width_ns))
        cursor += width_ns
    return bins


@dataclass(slots=True)
class BinFeatures:
    """Aggregated features for one bin."""

    start_ns: int
    end_ns: int
    duration_ns: int
    valid_duration_ns: int
    coverage_fraction: Decimal
    capture_coverage_fraction: Decimal
    coverage_ok: bool
    snapshot_count: int
    state_age_at_end_ns: int | None
    mid: Decimal | None
    spread: Decimal | None
    microprice: Decimal | None
    depth_imbalance: Decimal | None
    bid_depth_notional: Decimal | None
    ask_depth_notional: Decimal | None
    visible_quote_depth: Decimal | None
    trade_count: int | None
    trade_quantity: Decimal | None
    trade_notional: Decimal | None
    vwap: Decimal | None
    complete: bool
    flags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FeatureRun:
    """Bins plus the metadata needed to interpret them."""

    bins: list[BinFeatures]
    weighting: str
    interval_convention: str
    coverage_threshold: Decimal
    max_state_age_ns: int | None
    null_below_threshold: bool
    depth_levels: int
    value_precision_dp: int
    price_unit: str | None
    quantity_unit: str | None
    quote_currency: str | None
    contract_multiplier: Decimal
    window_start_ns: int | None
    window_end_ns: int | None
    known_gaps: list[tuple[int, int]]
    engine_version: str
    schema_version: str
    dataset_versions: list[str]


@dataclass(frozen=True, slots=True)
class _Segment:
    start_ns: int
    end_ns: int
    state: BookState | None
    end_reason: SegmentEnd


_MEAN_FEATURES = (
    "mid",
    "spread",
    "microprice",
    "depth_imbalance",
    "bid_depth_notional",
    "ask_depth_notional",
    "visible_quote_depth",
)


def _build_segments(
    states: Sequence[BookState],
    gaps: Sequence[tuple[int, int]],
    max_state_age_ns: int | None,
    window_end_ns: int | None,
) -> list[_Segment]:
    """Carry each state forward until the next snapshot, a gap, staleness or window end."""
    gap_starts = [g[0] for g in gaps]
    segments: list[_Segment] = []
    for i, state in enumerate(states):
        start = state.time_ns
        end = states[i + 1].time_ns if i + 1 < len(states) else None
        reason = SegmentEnd.next_snapshot if end is not None else SegmentEnd.open
        if max_state_age_ns is not None:
            stale_at = start + max_state_age_ns
            if end is None or stale_at < end:
                end, reason = stale_at, SegmentEnd.stale
        gi = bisect_right(gap_starts, start)
        if gi < len(gap_starts):
            gap_at = gap_starts[gi]
            if end is None or gap_at < end:
                end, reason = gap_at, SegmentEnd.known_gap
        if window_end_ns is not None and (end is None or window_end_ns < end):
            end, reason = window_end_ns, SegmentEnd.window_end
        if end is None:
            continue  # open-ended state without a window end: nothing to integrate against
        if end > start:
            segments.append(_Segment(start, end, state if state.valid else None, reason))
    return segments


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def aggregate_bins(
    snapshots: Sequence[BookSnapshot],
    trades: Sequence[Trade],
    bins: Sequence[Bin],
    *,
    instrument: Instrument | None = None,
    policy: FeaturePolicy | None = None,
    window_start_ns: int | None = None,
    window_end_ns: int | None = None,
    known_gaps: Iterable[tuple[int, int]] = (),
    now_ns: int | None = None,
) -> FeatureRun:
    """Time-weighted aggregation of book and trade features per bin.

    ``snapshots`` before ``window_start_ns`` act as an authorized seed (the
    caller decides which earlier rows it may use); their contribution is
    clipped to the window. ``known_gaps`` are half-open capture gaps: a state
    stops at a gap start and resumes only at the next snapshot. ``now_ns``
    marks bins ending after it as incomplete. Bins may be in any order; the
    output preserves the input order.
    """
    policy = policy or FeaturePolicy()
    multiplier = instrument.contract_multiplier if instrument is not None else Decimal(1)
    gaps = merge_intervals(known_gaps)
    ordered_snapshots = sort_snapshots(snapshots)
    ordered_trades = sort_trades(trades)
    dataset_versions = sorted({s.dataset_version for s in ordered_snapshots if s.dataset_version} | {t.dataset_version for t in ordered_trades if t.dataset_version})

    # Effective window end for open-ended carry-forward: the latest bin end.
    effective_window_end = window_end_ns
    if effective_window_end is None and bins:
        effective_window_end = max(b.end_ns for b in bins)

    with localcontext(DECIMAL_CONTEXT):
        states = [
            book_state(s, depth_levels=policy.depth_levels, contract_multiplier=multiplier, depth_cap=policy.depth_cap, precision_dp=policy.value_precision_dp)
            for s in ordered_snapshots
        ]
        # Equal-time snapshots: the last in total order is the effective state.
        deduped: list[BookState] = []
        for st in states:
            if deduped and deduped[-1].time_ns == st.time_ns:
                deduped[-1] = st
            else:
                deduped.append(st)
        segments = _build_segments(deduped, gaps, policy.max_state_age_ns, effective_window_end)
        segment_starts = [seg.start_ns for seg in segments]
        snapshot_times = [s.analysis_time_ns for s in ordered_snapshots]
        trade_times = [t.analysis_time_ns for t in ordered_trades]

        out: list[BinFeatures] = []
        for b in bins:
            if b.end_ns <= b.start_ns:
                raise ValueError("bin end must be after start")
            lo, hi = b.start_ns, b.end_ns
            flags: list[str] = []
            if window_start_ns is not None:
                lo = max(lo, window_start_ns)
            if window_end_ns is not None:
                hi = min(hi, window_end_ns)
            in_window = max(0, hi - lo)
            if in_window < b.duration_ns:
                flags.append(PARTIAL_WINDOW_BIN)

            gap_overlap = sum(_overlap(lo, hi, gs, ge) for gs, ge in gaps) if in_window else 0
            if gap_overlap:
                flags.append(KNOWN_GAP_OVERLAP)
            capture_ns = in_window - gap_overlap
            capture_fraction = quantize_fraction(Fraction(capture_ns, b.duration_ns), 6)

            # Book state integration, clipped to [lo, hi).
            valid_ns = 0
            sums: dict[str, Fraction] = {name: Fraction(0) for name in _MEAN_FEATURES}
            durations: dict[str, int] = {name: 0 for name in _MEAN_FEATURES}
            state_at_end: BookState | None = None
            if in_window:
                si = bisect_right(segment_starts, lo) - 1
                if si < 0:
                    si = 0
                while si < len(segments) and segments[si].start_ns < hi:
                    seg = segments[si]
                    dt = _overlap(lo, hi, seg.start_ns, seg.end_ns)
                    if dt > 0:
                        if seg.state is not None:
                            valid_ns += dt
                            for name in _MEAN_FEATURES:
                                value = getattr(seg.state, name)
                                if value is not None:
                                    sums[name] += Fraction(value) * dt
                                    durations[name] += dt
                        if seg.end_reason is SegmentEnd.stale and lo < seg.end_ns < hi:
                            flags.append(STALE_STATE_CUTOFF)
                        if seg.start_ns < hi <= seg.end_ns and seg.state is not None:
                            state_at_end = seg.state
                    si += 1
            coverage = quantize_fraction(Fraction(valid_ns, b.duration_ns), 6)
            coverage_ok = coverage >= policy.coverage_threshold
            if not coverage_ok:
                flags.append(BELOW_COVERAGE_THRESHOLD)
            if valid_ns == 0:
                flags.append(NO_VALID_STATE)

            means: dict[str, Decimal | None] = {}
            for name in _MEAN_FEATURES:
                if durations[name] > 0:
                    means[name], _ = fraction_to_decimal(sums[name] / durations[name], policy.value_precision_dp)
                else:
                    means[name] = None
            if policy.null_below_threshold and not coverage_ok and any(v is not None for v in means.values()):
                means = {name: None for name in _MEAN_FEATURES}
                flags.append(VALUES_NULLED_BELOW_THRESHOLD)

            # Trades in [lo, hi).
            t_lo = bisect_left(trade_times, lo)
            t_hi = bisect_left(trade_times, hi)
            count = 0
            quantity = Decimal(0)
            notional = Decimal(0)
            for trade in ordered_trades[t_lo:t_hi]:
                if trade.price <= 0 or trade.quantity <= 0:
                    continue
                count += 1
                quantity += trade.quantity
                notional += trade.price * trade.quantity * multiplier
            trade_count: int | None
            trade_quantity: Decimal | None
            trade_notional: Decimal | None
            vwap: Decimal | None = None
            if count > 0:
                trade_count, trade_quantity, trade_notional = count, quantity, notional
                vwap, _ = fraction_to_decimal(Fraction(notional) / (Fraction(quantity) * Fraction(multiplier)), policy.value_precision_dp)
            elif capture_fraction >= policy.coverage_threshold:
                trade_count, trade_quantity, trade_notional = 0, Decimal(0), Decimal(0)
            else:
                trade_count = trade_quantity = trade_notional = None
                flags.append(TRADES_UNKNOWN_MISSING_COVERAGE)

            complete = now_ns is None or b.end_ns <= now_ns
            if not complete:
                flags.append(INCOMPLETE_BIN)
            flags = list(dict.fromkeys(flags))

            out.append(
                BinFeatures(
                    start_ns=b.start_ns,
                    end_ns=b.end_ns,
                    duration_ns=b.duration_ns,
                    valid_duration_ns=valid_ns,
                    coverage_fraction=coverage,
                    capture_coverage_fraction=capture_fraction,
                    coverage_ok=coverage_ok,
                    snapshot_count=bisect_left(snapshot_times, hi) - bisect_left(snapshot_times, lo) if in_window else 0,
                    state_age_at_end_ns=None if state_at_end is None else hi - state_at_end.time_ns,
                    mid=means["mid"],
                    spread=means["spread"],
                    microprice=means["microprice"],
                    depth_imbalance=means["depth_imbalance"],
                    bid_depth_notional=means["bid_depth_notional"],
                    ask_depth_notional=means["ask_depth_notional"],
                    visible_quote_depth=means["visible_quote_depth"],
                    trade_count=trade_count,
                    trade_quantity=trade_quantity,
                    trade_notional=trade_notional,
                    vwap=vwap,
                    complete=complete,
                    flags=flags,
                )
            )

    return FeatureRun(
        bins=out,
        weighting="time_weighted",
        interval_convention="[start, end) UTC",
        coverage_threshold=policy.coverage_threshold,
        max_state_age_ns=policy.max_state_age_ns,
        null_below_threshold=policy.null_below_threshold,
        depth_levels=policy.depth_levels,
        value_precision_dp=policy.value_precision_dp,
        price_unit=None if instrument is None else instrument.price_unit,
        quantity_unit=None if instrument is None else instrument.quantity_unit,
        quote_currency=None if instrument is None else instrument.quote_asset,
        contract_multiplier=multiplier,
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
        known_gaps=gaps,
        engine_version=ENGINE_VERSION,
        schema_version=SCHEMA_VERSION,
        dataset_versions=dataset_versions,
    )


__all__ = [
    "BELOW_COVERAGE_THRESHOLD",
    "Bin",
    "BinFeatures",
    "BookState",
    "FeaturePolicy",
    "FeatureRun",
    "INCOMPLETE_BIN",
    "KNOWN_GAP_OVERLAP",
    "NO_VALID_STATE",
    "PARTIAL_WINDOW_BIN",
    "STALE_STATE_CUTOFF",
    "SegmentEnd",
    "TRADES_UNKNOWN_MISSING_COVERAGE",
    "aggregate_bins",
    "book_state",
    "make_bins",
    "BLOCKING_BOOK_FLAGS",
]
