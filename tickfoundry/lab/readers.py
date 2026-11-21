"""Venue file readers for ``LocalReplay``.

These are the SDK's own minimal readers (pyarrow only): the venue rules
(ordering, clocks, flags, depth) are implemented here directly. The hosted engine reads the same files through its own adapters;
tests hold the two to the same results. The swap point for a shared reader is
``open_bundle`` and the two ``*Columns`` containers below.

Design: files are read with pyarrow into columnar numpy arrays once per
instrument (the same footprint as the published
https://tickfoundry.com/backtest/hl_slippage.py), sorted by the venue's total
order, and ``labcore`` records are
materialised lazily per row. Nothing here changes labcore semantics; the
reader only decides what goes into a ``BookSnapshot`` / ``Trade``.
"""
from __future__ import annotations

import csv
import hashlib
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from labcore.model import AggressorSide, BookSnapshot, ClockKind, Instrument, Level, Trade

POLYMARKET = "polymarket"
HYPERLIQUID = "hyperliquid"
KALSHI = "kalshi"
VENUE_DEPTH = 25  # both venues deliver at most 25 levels

# Wire quality flags emitted by the readers.
CROSSED = "crossed"
LOCKED = "locked"
ONE_SIDED = "one_sided"
EMPTY = "empty"
NO_BOOK = "no_book"
TOP_DISAGREES_WITH_FEED = "top_disagrees_with_feed"
SOURCE_TIME_MISSING = "source_time_missing"
NEGATIVE_LATENCY = "negative_latency"
SYNTHESIZED_FILL = "synthesized_fill"
REPLAY_ARTIFACT = "replay_artifact"
LADDER_HOLE = "ladder_hole"
NO_SEED = "no_seed"
TICK_SIZE_CHANGED = "tick_size_changed"
RECEIVE_TIME_STEPPED_BACK = "receive_time_stepped_back"
LATE_SNAPSHOTS_IN_ARRIVAL_ORDER = "late_snapshots_in_arrival_order"
#: Kalshi: venue order is (hour of receipt, msg_seq); late hourly snapshots were applied in arrival
#: order before 2026-11-08 15:00 UTC; trade/book pairs closer than 30 ms cannot be ordered.
KS_HOUR_NS = 3_600_000_000_000
KS_ARRIVAL_ORDER_UNTIL_NS = 1_794_150_000_000_000_000
KS_UNCERTAINTY_WINDOW_NS = 30_000_000
#: Attributed depth flags: whose limitation the missing depth is.
PLAN_DEPTH_CAP = "plan_depth_cap"
VENUE_BOOK_DEEPER = "venue_book_deeper"
TOP_OF_BOOK_ONLY = "top_of_book_only"
CLOCK_RESOLUTION_MS = "clock_resolution_ms"

_NS_PER_MS = 1_000_000
_LEVEL_RE = re.compile(r"^(bid|ask)_px_(\d+)$")


def _float_column(table: pa.Table, name: str, n: int) -> np.ndarray:
    """Float64 column with nulls as NaN (NaN and NULL padding are the same thing)."""
    if name not in table.column_names:
        return np.full(n, np.nan, dtype=np.float64)
    return table[name].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)


def _int_column(table: pa.Table, name: str, n: int, *, fill: int = 0) -> np.ndarray:
    if name not in table.column_names:
        return np.full(n, fill, dtype=np.int64)
    col = table[name]
    if col.null_count:
        col = pc.fill_null(col, fill)
    return col.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


def _str_column(table: pa.Table, name: str, n: int) -> np.ndarray:
    if name not in table.column_names:
        return np.full(n, None, dtype=object)
    return np.array(table[name].to_pylist(), dtype=object)


def _matrix(table: pa.Table, prefix: str, depth: int, n: int, *, dtype: str = "float") -> np.ndarray:
    if depth == 0:
        return np.empty((n, 0), dtype=np.float64 if dtype == "float" else np.int32)
    if dtype == "float":
        cols = [_float_column(table, f"{prefix}_{i}", n) for i in range(1, depth + 1)]
        return np.column_stack(cols) if cols else np.empty((n, 0))
    cols = [_int_column(table, f"{prefix}_{i}", n, fill=-1).astype(np.int32) for i in range(1, depth + 1)]
    return np.column_stack(cols)


def ladder_depth_of(names: Sequence[str]) -> int:
    """Number of ladder columns in a file (25, 10 on a 10-level plan, or 0 for l1)."""
    depth = 0
    for name in names:
        match = _LEVEL_RE.match(name)
        if match:
            depth = max(depth, int(match.group(2)))
    return depth


def decimal_from_float(value: float) -> Decimal:
    """Exact decimal of the stored double via ``repr`` (labcore's ``coerce_decimal`` rule)."""
    return Decimal(repr(float(value)))


def infer_tick_size(prices: Iterable[Decimal], candidates: Sequence[Decimal]) -> Decimal | None:
    """Coarsest candidate grid every price sits on (inferred from the observed grid)."""
    prices = list(prices)
    if not prices:
        return None
    for tick in candidates:
        if all((p / tick) == (p / tick).to_integral_value() for p in prices):
            return tick
    return None


PM_TICK_CANDIDATES = (Decimal("0.01"), Decimal("0.0025"), Decimal("0.001"))
HL_TICK_CANDIDATES = tuple(Decimal(10) ** k for k in range(3, -7, -1))
KS_TICK_CANDIDATES = (Decimal("0.01"), Decimal("0.005"), Decimal("0.001"), Decimal("0.0001"))


_PM_TICK_GRIDS = (Decimal("0.01"), Decimal("0.0025"), Decimal("0.001"))


def _pm_tick_refinement_ns(cols: "BookColumns", reference_tick: Decimal | None = None) -> int | None:
    """Time of the first tick-size refinement, the same rule as the engine's Polymarket
    ``tick_history``: each row is classed by the finest grid its prices
    need (1 c, 0.25 c, 0.1 c); the first class seeds the history and a finer class starts a
    new entry. A market's reference tick (``markets.tick_size``, 4 dp), when it is one of the
    three grids, seeds the history instead, as the hosted engine's catalog reference does.
    Returns the second entry's time, or None when the tick never changed."""
    if cols.n == 0 or cols.ladder_depth == 0:
        return None
    px = np.concatenate([cols.bid_px, cols.ask_px], axis=1)
    finite = np.isfinite(px)
    with np.errstate(invalid="ignore"):
        off_cent = finite & (np.abs(px * 100 - np.round(px * 100)) > 1e-6)
        off_quarter = finite & (np.abs(px * 400 - np.round(px * 400)) > 1e-6)
    grid = np.where(off_quarter.any(axis=1), 2, np.where(off_cent.any(axis=1), 1, 0))
    start = _PM_TICK_GRIDS.index(reference_tick) if reference_tick in _PM_TICK_GRIDS else int(grid[0])
    finer = np.flatnonzero(grid > start)
    return int(cols.t_ns[finer[0]]) if len(finer) else None


# ---------------------------------------------------------------------------
# Columnar containers
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class BookColumns:
    """One instrument's book rows, sorted by the venue's total order."""

    venue: str
    instrument_id: str
    n: int
    ladder_depth: int
    t_ns: np.ndarray
    seq: np.ndarray
    src_ms: np.ndarray
    src_row: np.ndarray
    bid_px: np.ndarray
    bid_sz: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray
    bid_n: np.ndarray | None
    ask_n: np.ndarray | None
    bid_levels: np.ndarray | None
    ask_levels: np.ndarray | None
    reconciles: np.ndarray | None
    event_kind: np.ndarray | None
    n_bids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    n_asks: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    crossed: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    locked: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    one_sided: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    empty: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    no_book: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    hole: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    valid: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    full_depth_known: list[bool | None] = field(default_factory=list)
    clock_kind: ClockKind = ClockKind.collector_observed
    ms_clock: bool = False
    #: The rows came from top-of-book columns (an l1 file): one level a side, nothing deeper on file.
    top_of_book: bool = False
    #: The venue's book exceeded the 25 stored levels, from the counters BEFORE any depth-cap clamp.
    venue_deeper: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    #: Polymarket: first row on a finer tick grid than the day's first rows (a price off the coarser
    #: grid starts a new tick, a coarser one never ends it, as the hosted Lab rules); rows from it on
    #: carry ``tick_size_changed``.
    tick_changed_from_ns: int | None = None
    #: Kalshi: raw receive times (``t_ns`` holds their running maximum in venue order) and the flags.
    t_raw: np.ndarray | None = None
    recv_back: np.ndarray | None = None

    # ------------------------------------------------------------- derived
    def finish(self) -> "BookColumns":
        """Derive validity, sides and completeness."""
        self.tick_changed_from_ns = _pm_tick_refinement_ns(self) if self.venue == POLYMARKET else None
        n, d = self.n, self.ladder_depth
        pop_b = ~np.isnan(self.bid_px) & ~np.isnan(self.bid_sz) if d else np.zeros((n, 0), dtype=bool)
        pop_a = ~np.isnan(self.ask_px) & ~np.isnan(self.ask_sz) if d else np.zeros((n, 0), dtype=bool)

        def prefix_len(pop: np.ndarray) -> np.ndarray:
            if pop.shape[1] == 0:
                return np.zeros(n, dtype=np.int64)
            all_true = pop.all(axis=1)
            first_false = pop.argmin(axis=1)
            return np.where(all_true, pop.shape[1], first_false).astype(np.int64)

        def has_hole(pop: np.ndarray) -> np.ndarray:
            if pop.shape[1] == 0:
                return np.zeros(n, dtype=bool)
            after_gap = np.cumsum(~pop, axis=1) > 0
            return (after_gap & pop).any(axis=1)

        self.n_bids = prefix_len(pop_b)
        self.n_asks = prefix_len(pop_a)
        # A hole (a missing level between populated ones) is never compacted.
        self.hole = has_hole(pop_b) | has_hole(pop_a)
        top_b = self.bid_px[:, 0] if d else np.full(n, np.nan)
        top_a = self.ask_px[:, 0] if d else np.full(n, np.nan)
        has_b = self.n_bids > 0
        has_a = self.n_asks > 0
        both = has_b & has_a
        self.crossed = both & (top_b > top_a)
        self.locked = both & (top_b == top_a)
        self.one_sided = has_b ^ has_a
        neither = ~has_b & ~has_a
        if self.venue == HYPERLIQUID:
            # The coin-day placeholder: no levels and zero counters.
            counters_zero = np.ones(n, dtype=bool)
            if self.bid_levels is not None and self.ask_levels is not None:
                counters_zero = (self.bid_levels == 0) & (self.ask_levels == 0)
            self.no_book = neither & counters_zero
            self.empty = neither & ~self.no_book
        else:
            self.no_book = np.zeros(n, dtype=bool)
            self.empty = neither
        self.valid = ~self.crossed & ~self.empty & ~self.no_book & ~self.hole
        if len(self.venue_deeper) != n:
            self.venue_deeper = np.zeros(n, dtype=bool)
        if self.venue == HYPERLIQUID and not self.top_of_book:
            # A side showing all 25 stored levels means "at least 25" (counters are capped).
            self.venue_deeper = self.venue_deeper | (self.n_bids >= VENUE_DEPTH) | (self.n_asks >= VENUE_DEPTH)
        self.full_depth_known = self._completeness()
        return self

    def _completeness(self) -> list[bool | None]:
        """``full_depth_known`` per row."""
        n, d = self.n, self.ladder_depth
        if self.top_of_book:
            # One level a side and nothing about the rest: known incomplete, not unknown (the engine says the same).
            return [False] * n
        if d == 0:
            return [None] * n
        if self.venue == HYPERLIQUID:
            # Counters are capped at 25: a side showing 25 proves nothing.
            capped = (self.n_bids >= VENUE_DEPTH) | (self.n_asks >= VENUE_DEPTH)
            if self.bid_levels is not None and self.ask_levels is not None:
                capped |= (self.bid_levels >= VENUE_DEPTH) | (self.ask_levels >= VENUE_DEPTH)
            return [False if c else True for c in capped]
        if self.bid_levels is None or self.ask_levels is None:
            return [None] * n
        if d >= VENUE_DEPTH:
            # Polymarket counters report the whole reconstructed book.
            known = (self.bid_levels <= VENUE_DEPTH) & (self.ask_levels <= VENUE_DEPTH)
            return [bool(k) for k in known]
        # Depth-capped file: the plan's depth cap clamped the counters, so a counter equal to the cap is unknown.
        at_cap = (self.bid_levels >= d) | (self.ask_levels >= d)
        return [None if c else True for c in at_cap]

    # --------------------------------------------------------------- rows
    def visible_depth(self, i: int) -> int:
        return int(max(self.n_bids[i], self.n_asks[i]))

    def wire_flags(self, i: int, *, depth_cap: int | None = None) -> list[str]:
        """Wire flags of row ``i`` as the reader can determine them."""
        flags: list[str] = []
        if self.crossed[i]:
            flags.append(CROSSED)
        elif self.locked[i]:
            flags.append(LOCKED)
        if self.one_sided[i]:
            flags.append(ONE_SIDED)
        if self.empty[i]:
            flags.append(EMPTY)
        if self.no_book[i]:
            flags.append(NO_BOOK)
        if self.hole[i]:
            flags.append(LADDER_HOLE)
        if self.venue == KALSHI:
            return flags + self._kalshi_flags(i, depth_cap=depth_cap)
        if self.venue == POLYMARKET:
            if self.reconciles is not None and self.event_kind is not None:
                if self.event_kind[i] == "update" and self.reconciles[i] is False:
                    flags.append(TOP_DISAGREES_WITH_FEED)
            src = int(self.src_ms[i])
            if src == 0:
                flags.append(SOURCE_TIME_MISSING)
            elif int(self.t_ns[i]) < src * _NS_PER_MS:
                flags.append(NEGATIVE_LATENCY)
            if self.ms_clock:
                flags.append(CLOCK_RESOLUTION_MS)
        elif self.crossed[i]:
            flags.append(REPLAY_ARTIFACT)
        # Depth attribution: never a generic "truncated" word. A top-of-book
        # day names its source; otherwise the plan cap is flagged where the clamped counters reach it
        # (completeness unknown), and the venue's own depth beyond the 25 stored levels separately.
        if self.top_of_book:
            flags.append(TOP_OF_BOOK_ONLY)
        else:
            cap = depth_cap if depth_cap is not None else self.ladder_depth
            if cap and cap < VENUE_DEPTH and self.visible_depth(i) >= cap and self.full_depth_known[i] is not True:
                flags.append(PLAN_DEPTH_CAP)
            if self.venue_deeper[i]:
                flags.append(VENUE_BOOK_DEEPER)
        if self.tick_changed_from_ns is not None and int(self.t_ns[i]) >= self.tick_changed_from_ns:
            flags.append(TICK_SIZE_CHANGED)
        return flags

    def _kalshi_flags(self, i: int, *, depth_cap: int | None = None) -> list[str]:
        """After the structural flags, in the hosted engine's order: depth attribution, then
        source_time_missing / negative_latency, receive_time_stepped_back, the arrival-order era."""
        flags: list[str] = []
        cap = depth_cap if depth_cap is not None else self.ladder_depth
        if self.top_of_book:
            flags.append(TOP_OF_BOOK_ONLY)
        else:
            if cap and cap < VENUE_DEPTH and self.visible_depth(i) >= cap and self.full_depth_known[i] is not True:
                flags.append(PLAN_DEPTH_CAP)
            if self.venue_deeper[i]:
                flags.append(VENUE_BOOK_DEEPER)
        raw = int(self.t_raw[i]) if self.t_raw is not None else int(self.t_ns[i])
        src = int(self.src_ms[i])
        if src == 0:
            flags.append(SOURCE_TIME_MISSING)
        elif raw < src * _NS_PER_MS:
            flags.append(NEGATIVE_LATENCY)
        if self.recv_back is not None and self.recv_back[i]:
            flags.append(RECEIVE_TIME_STEPPED_BACK)
        if raw < KS_ARRIVAL_ORDER_UNTIL_NS:
            flags.append(LATE_SNAPSHOTS_IN_ARRIVAL_ORDER)
        return flags

    def levels(self, i: int, side: str) -> list[Level]:
        px, sz = (self.bid_px, self.bid_sz) if side == "bid" else (self.ask_px, self.ask_sz)
        counts = self.bid_n if side == "bid" else self.ask_n
        count = int(self.n_bids[i] if side == "bid" else self.n_asks[i])
        out: list[Level] = []
        for k in range(count):
            order_count = None
            if counts is not None:
                c = int(counts[i, k])
                order_count = None if c < 0 else c
            out.append(Level(decimal_from_float(px[i, k]), decimal_from_float(sz[i, k]), order_count))
        return out

    def snapshot(self, i: int, *, dataset_version: str | None = None, depth_cap: int | None = None) -> BookSnapshot:
        """Materialise row ``i`` as a labcore ``BookSnapshot``."""
        src = int(self.src_ms[i])
        source_time_ms: int | None = src
        observed: int | None = int(self.t_ns[i])
        if self.venue == POLYMARKET and src == 0:
            source_time_ms = None  # ts_src_ms == 0 means missing
        if self.venue == HYPERLIQUID:
            observed = None  # nothing was observed; never derive latency
        return BookSnapshot(
            instrument_id=self.instrument_id,
            analysis_time_ns=int(self.t_ns[i]),
            bids=self.levels(i, "bid"),
            asks=self.levels(i, "ask"),
            clock_kind=self.clock_kind,
            observed_time_ns=observed,
            source_time_ms=source_time_ms,
            source_order_key=int(self.seq[i]),
            stable_row_ordinal=int(i),
            visible_depth=self.visible_depth(i),
            full_depth_known=self.full_depth_known[i],
            quality_flags=self.wire_flags(i, depth_cap=depth_cap),
            dataset_version=dataset_version,
            source_fragment_id=None,
        )

    def block_of(self, i: int) -> int | None:
        return int(self.seq[i]) if self.venue == HYPERLIQUID else None


@dataclass(slots=True)
class TradeColumns:
    """One instrument's tape rows, sorted by the venue's total order."""

    venue: str
    instrument_id: str
    n: int
    t_ns: np.ndarray
    seq: np.ndarray
    src_ms: np.ndarray
    src_row: np.ndarray
    price: np.ndarray
    size: np.ndarray
    side: np.ndarray
    tx_hash: np.ndarray
    tid: np.ndarray | None
    oid: np.ndarray | None
    fee: np.ndarray | None
    fee_token: np.ndarray | None
    source_event_type: np.ndarray | None
    clock_kind: ClockKind = ClockKind.collector_observed

    def wire_flags(self, i: int) -> list[str]:
        flags: list[str] = []
        if self.venue == HYPERLIQUID:
            if self.source_event_type is not None and self.source_event_type[i] == "l3_taker_status":
                flags.append(SYNTHESIZED_FILL)
        else:
            src = int(self.src_ms[i])
            if src == 0:
                flags.append(SOURCE_TIME_MISSING)
            elif int(self.t_ns[i]) < src * _NS_PER_MS:
                flags.append(NEGATIVE_LATENCY)
        return flags

    def trade_id(self, i: int) -> str | None:
        """Identity: PM ``transaction_hash``; HL ``tid`` (null before 2025-03-22)."""
        if self.venue == HYPERLIQUID:
            if self.tid is not None and self.tid[i] is not None:
                return str(int(self.tid[i]))
            return None
        tx = self.tx_hash[i]
        return None if tx is None else str(tx)

    def trade(self, i: int, *, dataset_version: str | None = None) -> Trade:
        side_raw = self.side[i]
        side = None
        if isinstance(side_raw, str):
            lowered = side_raw.lower()
            if lowered in ("buy", "sell"):
                side = AggressorSide(lowered)
            elif self.venue == KALSHI and lowered in ("yes", "no"):
                # the taker's outcome side: buying YES lifts the YES asks, buying NO hits the YES bids
                side = AggressorSide("buy" if lowered == "yes" else "sell")
        fee: Decimal | None = None
        fee_asset: str | None = None
        if self.venue == HYPERLIQUID and self.fee is not None:
            raw_fee = self.fee[i]
            if raw_fee is not None and not (isinstance(raw_fee, float) and math.isnan(raw_fee)):
                fee = decimal_from_float(raw_fee)
                fee_asset = None if self.fee_token is None else self.fee_token[i]
        native_trade_id = None
        if self.venue == HYPERLIQUID and self.tid is not None and self.tid[i] is not None:
            native_trade_id = str(int(self.tid[i]))
        elif self.venue == KALSHI and self.tx_hash[i] is not None:
            native_trade_id = str(self.tx_hash[i])
        tx = self.tx_hash[i]
        return Trade(
            instrument_id=self.instrument_id,
            analysis_time_ns=int(self.t_ns[i]),
            price=decimal_from_float(self.price[i]),
            quantity=decimal_from_float(self.size[i]),
            clock_kind=self.clock_kind,
            source_order_key=int(self.seq[i]),
            stable_row_ordinal=int(i),
            aggressor_side=side,
            native_trade_id=native_trade_id,
            native_transaction_id=None if tx is None else str(tx),
            fee=fee,
            fee_asset=fee_asset,
            quality_flags=self.wire_flags(i),
            dataset_version=dataset_version,
        )


# ---------------------------------------------------------------------------
# Bundle discovery
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class BundleFiles:
    venue: str
    root: Path
    books: list[Path]  # l2 files (preferred) or l1 files
    books_product: str  # "l2" | "l1"
    trades: list[Path]
    reference_dir: Path | None
    quality: dict[str, Path]  # HL: coin -> quality json


def _pm_date_of(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("date="):
            return part[5:]
    return None


def _classify(path: Path) -> tuple[str | None, str] | None:
    """(venue hint, product) of a parquet file by name: bundle names and the fixtures' flat ``pm_*``/``hl_*`` names."""
    name = path.name
    if name in ("l1.parquet", "l2.parquet", "trades.parquet"):
        hint = HYPERLIQUID if any(part in ("hl_l1", "hl_l2", "hl_trades") or part.startswith("coin=") for part in path.parts) else None
        if any(part in ("kalshi_l1", "kalshi_l2", "kalshi_trades") for part in path.parts):
            hint = KALSHI
        return hint, name[:-8]
    for venue, prefix in ((POLYMARKET, "pm_"), (HYPERLIQUID, "hl_")):
        for product in ("l1", "l2", "trades"):
            if name == f"{prefix}{product}.parquet":
                return venue, product
    return None


def discover(root: Path, *, dates: Sequence[str] | None = None, venue: str | None = None) -> BundleFiles:
    """Find venue files under ``root``.

    Polymarket bundles: ``l1.parquet`` / ``l2.parquet`` / ``trades.parquet`` at
    the root or under ``date=D`` (and deeper ``series=``/``event=``) folders,
    with ``reference/*.csv``. Hyperliquid pulls: ``hl_l2/date=D/<COIN>/l2.parquet``
    and siblings (the archive layout ``l2/coin=<dir>/l2.parquet`` is accepted too).
    The fixtures' flat ``pm_l2.parquet`` / ``hl_l2.parquet`` names are recognised;
    a directory holding both venues needs ``venue``. Variant files such as
    ``pm_l2_nan.parquet`` are never picked up implicitly (use ``from_files``).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"bundle directory not found: {root}")
    found: dict[str, dict[str, list[Path]]] = {v: {"l1": [], "l2": [], "trades": []} for v in (POLYMARKET, HYPERLIQUID, KALSHI)}
    for path in sorted(root.rglob("*.parquet")):
        classified = _classify(path)
        if classified is None:
            continue
        hint, product = classified
        if dates and _pm_date_of(path) is not None and _pm_date_of(path) not in set(dates):
            continue
        if hint is None:
            names = pq.ParquetFile(path).schema_arrow.names
            hint = HYPERLIQUID if "coin" in names else KALSHI if "market_ticker" in names else POLYMARKET
        found[hint][product].append(path)
    present = [v for v in (POLYMARKET, HYPERLIQUID, KALSHI) if any(found[v].values())]
    if venue is not None:
        if venue not in found:
            raise ValueError(f"unknown venue {venue!r}")
        if venue not in present:
            raise FileNotFoundError(f"no {venue} parquet files under {root}")
        chosen = venue
    elif len(present) == 1:
        chosen = present[0]
    elif not present:
        raise FileNotFoundError(f"no l1/l2/trades parquet files under {root}")
    else:
        raise ValueError(f"{root} holds files of several venues; pass venue='polymarket', 'hyperliquid' or 'kalshi'")
    files = found[chosen]
    books, product = (files["l2"], "l2") if files["l2"] else (files["l1"], "l1")
    reference = root / "reference"
    quality: dict[str, Path] = {}
    if chosen == HYPERLIQUID:
        for q in root.rglob("*.json"):
            if "quality" in q.parts or "scores" in q.parts:
                quality[q.stem] = q
    return BundleFiles(chosen, root, books, product, files["trades"], reference if reference.is_dir() else None, quality)


def bundle_from_files(venue: str, *, books: Sequence[Path] = (), trades: Sequence[Path] = (), root: Path | None = None, reference_dir: Path | None = None) -> BundleFiles:
    """Explicit file lists (fixture variants such as ``pm_l2_depth10.parquet``)."""
    if venue not in (POLYMARKET, HYPERLIQUID, KALSHI):
        raise ValueError(f"unknown venue {venue!r}")
    books = [Path(p) for p in books]
    trades = [Path(p) for p in trades]
    if not books and not trades:
        raise ValueError("at least one book or trade file is required")
    product = "l2" if any(ladder_depth_of(pq.ParquetFile(p).schema_arrow.names) > 0 for p in books) else "l1"
    root = Path(root) if root is not None else Path((books or trades)[0]).parent
    return BundleFiles(venue, root, books, product, trades, reference_dir, {})


def dataset_token(files: Iterable[Path], root: Path) -> str:
    """``local:`` plus sha256 of (relative path, size) pairs, 16 hex chars (the hosted Lab computes its own, different token)."""
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"|")
        digest.update(str(path.stat().st_size).encode())
        digest.update(b"\n")
    return "local:" + digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Polymarket reference tables
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class PmReference:
    tokens: dict[str, dict[str, str]]  # token_id -> row
    markets: dict[str, dict[str, str]]  # condition_id -> row
    series: dict[str, str] | None
    events: dict[str, dict[str, str]]

    @classmethod
    def load(cls, reference_dir: Path | None) -> "PmReference":
        if reference_dir is None:
            return cls({}, {}, None, {})

        def read(name: str) -> list[dict[str, str]]:
            path = reference_dir / f"{name}.csv"
            if not path.is_file():
                return []
            with path.open(newline="", encoding="utf-8") as handle:
                return [dict(row) for row in csv.DictReader(handle)]

        tokens = {row["token_id"]: row for row in read("tokens") if row.get("token_id")}
        markets = {row["condition_id"]: row for row in read("markets") if row.get("condition_id")}
        events = {row["event_id"]: row for row in read("events") if row.get("event_id")}
        series_rows = read("series")
        return cls(tokens, markets, series_rows[0] if series_rows else None, events)

    @staticmethod
    def tick_size(raw: str | None) -> Decimal | None:
        """``markets.tick_size`` is float-polluted: parse with Decimal and round to 4 dp."""
        if not raw:
            return None
        try:
            return Decimal(raw).quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN).normalize()
        except Exception:  # noqa: BLE001
            return None


def pm_instrument(token_id: str, condition_id: str | None, reference: PmReference) -> tuple[Instrument, dict[str, Any]]:
    """Build the labcore ``Instrument`` of a Polymarket outcome token."""
    token = reference.tokens.get(token_id, {})
    condition = condition_id or token.get("condition_id")
    market = reference.markets.get(condition or "", {})
    event_id = market.get("event_id") or token.get("event_id")
    unit_ids: list[str] = []
    if reference.series and reference.series.get("series_id"):
        unit_ids.append(f"series:{reference.series['series_id']}")
    elif event_id:
        unit_ids.append(f"event:{event_id}")
    outcome = token.get("outcome_code") or token.get("outcome_label")
    if not outcome and market:
        outcome = "YES" if market.get("yes_token_id") == token_id else ("NO" if market.get("no_token_id") == token_id else None)
    instrument = Instrument(
        venue=POLYMARKET,
        product_type="prediction_market",
        instrument_id=f"pm:{token_id}",
        native_symbol=token_id,
        quantity_unit="shares",
        price_unit="probability",
        entitlement_unit_ids=unit_ids,
        outcome_id=token_id,
        condition_id=condition,
        event_id=event_id,
        base_asset=None,
        quote_asset="USDC",
        tick_size_history_ref="observed_price_grid" if market.get("tick_size") else None,
        lot_size_history_ref=None,
        contract_multiplier=Decimal(1),
        tick_size=None,  # a history, never one value per day; inferred per execution book
        lot_increment=None,
    )
    meta = {
        "reference_tick": PmReference.tick_size(market.get("tick_size")),
        "question": market.get("question"),
        "outcome": outcome,
        "end_time": market.get("end_time"),
        "market_id": market.get("market_id") or token.get("market_id"),
        "reference_tick_size": str(PmReference.tick_size(market.get("tick_size"))) if market.get("tick_size") else None,
        "reference_tick_size_raw": market.get("tick_size"),
    }
    return instrument, meta


def hl_family(coin: str) -> str:
    """perp, spot or hip3 from the native coin name (the delivery's own family rule)."""
    if ":" in coin:
        return "hip3"
    if coin.startswith("@") or "/" in coin:
        return "spot"
    return "perp"


def hl_instrument(coin: str) -> tuple[Instrument, dict[str, Any]]:
    family = hl_family(coin)
    quote: str | None = "USDC"
    base = coin
    if family == "spot":
        if "/" in coin:
            base, quote = coin.split("/", 1)
        else:
            quote = None  # an @index spot coin's quote token is unknown without coins.json
    elif family == "hip3":
        base = coin.split(":", 1)[1]
    instrument = Instrument(
        venue=HYPERLIQUID,
        product_type=family,
        instrument_id=f"hl:{coin}",
        native_symbol=coin,
        quantity_unit=base,
        price_unit=quote or "quote",
        entitlement_unit_ids=[HYPERLIQUID],
        base_asset=base,
        quote_asset=quote,
        tick_size_history_ref="observed_price_grid",
        contract_multiplier=Decimal(1),
    )
    return instrument, {"family": family}


# ---------------------------------------------------------------------------
# Reading rows
# ---------------------------------------------------------------------------
_PM_BOOK_BASE = [
    "ts_src_ms", "ts_recv_ns", "condition_id", "token_id", "event_kind", "source_event_type", "msg_seq",
    "bid_levels", "ask_levels", "best_bid", "best_ask", "best_bid_size", "best_ask_size", "bid_ask_reconciles",
]
_KS_BOOK_BASE = ["ts_src_ms", "ts_recv_ns", "market_ticker", "event_kind", "bid_levels", "ask_levels", "best_bid", "best_ask", "best_bid_size", "best_ask_size", "source_event_type", "msg_seq"]
_HL_BOOK_BASE = ["ts_src_ms", "ts_recv_ns", "coin", "event_kind", "bid_levels", "ask_levels", "best_bid", "best_ask", "best_bid_size", "best_ask_size", "source_event_type", "msg_seq"]


def _read_concat(paths: Sequence[Path], wanted: Sequence[str], filt: Any = None) -> tuple[pa.Table, np.ndarray, np.ndarray]:
    """Read the wanted columns of every file and concatenate; returns (table, file_index, row_in_file)."""
    tables: list[pa.Table] = []
    file_index: list[np.ndarray] = []
    row_in_file: list[np.ndarray] = []
    for k, path in enumerate(paths):
        names = pq.ParquetFile(path).schema_arrow.names
        columns = [c for c in wanted if c in names]
        table = pq.read_table(path, columns=columns, filters=filt)
        if filt is not None:
            # Row offsets inside the file are lost once a filter is applied; recover them with an index read.
            full = pq.read_table(path, columns=[filt[0][0]])
            mask = pc.equal(full[filt[0][0]], filt[0][2])
            rows = np.flatnonzero(mask.to_numpy(zero_copy_only=False))
        else:
            rows = np.arange(table.num_rows)
        tables.append(table)
        file_index.append(np.full(table.num_rows, k, dtype=np.int64))
        row_in_file.append(rows.astype(np.int64))
    if not tables:
        return pa.table({}), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if len(tables) == 1:
        table = tables[0]
    else:
        common = [c for c in wanted if all(c in t.column_names for t in tables)]
        table = pa.concat_tables([t.select(common) for t in tables])
    return table, np.concatenate(file_index), np.concatenate(row_in_file)


def read_books(bundle: BundleFiles, instrument_key: str, *, depth: int | None = None) -> BookColumns:
    """Read one instrument's book rows (``token_id`` for PM, ``coin`` for HL), sorted by the venue order."""
    venue = bundle.venue
    if venue == POLYMARKET:
        paths = bundle.books
        filt = [("token_id", "=", instrument_key)]
        base = _PM_BOOK_BASE
    elif venue == KALSHI:
        paths = bundle.books
        filt = [("market_ticker", "=", instrument_key)]
        base = _KS_BOOK_BASE
    else:
        paths = [p for p in bundle.books if _hl_coin_of(p) == instrument_key]
        filt = None
        base = _HL_BOOK_BASE
    if not paths:
        raise KeyError(instrument_key)
    file_depth = max(ladder_depth_of(pq.ParquetFile(p).schema_arrow.names) for p in paths)
    ladder = file_depth
    if depth is not None:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        ladder = min(file_depth, depth) if file_depth else 0
    level_cols = [f"{s}_{k}_{i}" for s in ("bid", "ask") for k in ("px", "sz") for i in range(1, ladder + 1)]
    count_cols = [f"{s}_n_{i}" for s in ("bid", "ask") for i in range(1, ladder + 1)] if venue == HYPERLIQUID else []
    table, file_index, row_in_file = _read_concat(paths, [*base, *level_cols, *count_cols], filt)
    n = table.num_rows
    t_ns = _int_column(table, "ts_recv_ns", n)
    seq = _int_column(table, "msg_seq", n)
    src_ms = _int_column(table, "ts_src_ms", n)
    # Total order: PM (ts_recv_ns, msg_seq, row); HL (msg_seq, ts_recv_ns, row); Kalshi (hour, msg_seq, row).
    stable = file_index * (1 << 40) + row_in_file
    if venue == POLYMARKET:
        order = np.lexsort((stable, seq, t_ns))
    elif venue == KALSHI:
        order = np.lexsort((stable, seq, t_ns // KS_HOUR_NS))      # (hour, msg_seq): the venue order
    else:
        order = np.lexsort((stable, t_ns, seq))
    if n and not np.array_equal(order, np.arange(n)):
        table = table.take(pa.array(order))
        t_ns, seq, src_ms, stable = t_ns[order], seq[order], src_ms[order], stable[order]
    t_raw = recv_back = None
    if venue == KALSHI and n:
        # Analysis time = running maximum of the receive time in venue order (the capture's own rule).
        t_raw = t_ns
        t_ns = np.maximum.accumulate(t_raw)
        recv_back = t_raw < t_ns
    if file_depth == 0:
        # l1 fallback (Basic bundles): one level per side from the top-of-book columns.
        bid_px = _float_column(table, "best_bid", n)[:, None]
        bid_sz = _float_column(table, "best_bid_size", n)[:, None]
        ask_px = _float_column(table, "best_ask", n)[:, None]
        ask_sz = _float_column(table, "best_ask_size", n)[:, None]
        ladder = 1
        bid_n = ask_n = None
        bid_levels = ask_levels = None
        top_of_book = True
        venue_deeper = np.zeros(n, dtype=bool)
    else:
        bid_px = _matrix(table, "bid_px", ladder, n)
        bid_sz = _matrix(table, "bid_sz", ladder, n)
        ask_px = _matrix(table, "ask_px", ladder, n)
        ask_sz = _matrix(table, "ask_sz", ladder, n)
        bid_n = _matrix(table, "bid_n", ladder, n, dtype="int") if venue == HYPERLIQUID else None
        ask_n = _matrix(table, "ask_n", ladder, n, dtype="int") if venue == HYPERLIQUID else None
        bid_levels = _int_column(table, "bid_levels", n) if "bid_levels" in table.column_names else None
        ask_levels = _int_column(table, "ask_levels", n) if "ask_levels" in table.column_names else None
        top_of_book = False
        venue_deeper = np.zeros(n, dtype=bool)
        if bid_levels is not None and ask_levels is not None:
            # Polymarket counters describe the whole book (above 25 is deeper than stored); Hyperliquid
            # counters are capped at 25 (25 means at least 25). Read before the cap emulation clamps them.
            if venue in (POLYMARKET, KALSHI):
                venue_deeper = (bid_levels > VENUE_DEPTH) | (ask_levels > VENUE_DEPTH)
            else:
                venue_deeper = (bid_levels >= VENUE_DEPTH) | (ask_levels >= VENUE_DEPTH)
        if depth is not None and bid_levels is not None and ask_levels is not None and ladder < file_depth:
            # Emulate the plan's depth cap: drop deeper levels and clamp the counters.
            bid_levels = np.minimum(bid_levels, ladder)
            ask_levels = np.minimum(ask_levels, ladder)
    reconciles = None
    event_kind = None
    if venue == POLYMARKET:
        if "bid_ask_reconciles" in table.column_names:
            reconciles = np.array(table["bid_ask_reconciles"].to_pylist(), dtype=object)
        event_kind = _str_column(table, "event_kind", n)
    instrument_id = {POLYMARKET: "pm:", KALSHI: "ks:"}.get(venue, "hl:") + instrument_key
    # Files of 2026-02-21 to 2026-05-10 insert ``ts_recv_iso`` and stamp ``ts_recv_ns`` at millisecond
    # precision; both marks are required so whole-second synthetic rows never match.
    has_iso = any("ts_recv_iso" in pq.ParquetFile(p).schema_arrow.names for p in paths)
    ms_clock = bool(n) and venue == POLYMARKET and has_iso and bool(np.all(t_ns % _NS_PER_MS == 0))
    cols = BookColumns(
        venue=venue,
        instrument_id=instrument_id,
        n=n,
        ladder_depth=ladder,
        t_ns=t_ns,
        seq=seq,
        src_ms=src_ms,
        src_row=stable,
        bid_px=bid_px,
        bid_sz=bid_sz,
        ask_px=ask_px,
        ask_sz=ask_sz,
        bid_n=bid_n,
        ask_n=ask_n,
        bid_levels=bid_levels,
        ask_levels=ask_levels,
        reconciles=reconciles,
        event_kind=event_kind,
        clock_kind=ClockKind.reconstructed_block if venue == HYPERLIQUID else ClockKind.collector_observed,
        ms_clock=ms_clock,
        top_of_book=top_of_book,
        venue_deeper=venue_deeper,
        t_raw=t_raw,
        recv_back=recv_back,
    )
    return cols.finish()


_PM_TRADE_COLS = ["ts_src_ms", "ts_recv_ns", "condition_id", "token_id", "price", "size", "side", "fee_rate_bps", "transaction_hash", "msg_seq"]
_KS_TRADE_COLS = ["ts_src_ms", "ts_recv_ns", "market_ticker", "price", "size", "side", "trade_id", "msg_seq"]
_HL_TRADE_COLS = ["ts_src_ms", "ts_recv_ns", "coin", "price", "size", "side", "transaction_hash", "tid", "oid", "dir", "fee", "fee_token", "source_event_type", "msg_seq"]


def read_trades(bundle: BundleFiles, instrument_key: str) -> TradeColumns:
    venue = bundle.venue
    if venue == POLYMARKET:
        paths = bundle.trades
        filt = [("token_id", "=", instrument_key)]
        wanted = _PM_TRADE_COLS
    elif venue == KALSHI:
        paths = bundle.trades
        filt = [("market_ticker", "=", instrument_key)]
        wanted = _KS_TRADE_COLS
    else:
        paths = [p for p in bundle.trades if _hl_coin_of(p) == instrument_key]
        filt = None
        wanted = _HL_TRADE_COLS
    instrument_id = {POLYMARKET: "pm:", KALSHI: "ks:"}.get(venue, "hl:") + instrument_key
    clock = ClockKind.reconstructed_block if venue == HYPERLIQUID else ClockKind.collector_observed
    if not paths:
        empty = np.empty(0, dtype=np.int64)
        return TradeColumns(venue, instrument_id, 0, empty, empty, empty, empty, np.empty(0), np.empty(0), np.empty(0, dtype=object), np.empty(0, dtype=object), None, None, None, None, None, clock)
    table, file_index, row_in_file = _read_concat(paths, wanted, filt)
    n = table.num_rows
    t_ns = _int_column(table, "ts_recv_ns", n)
    seq = _int_column(table, "msg_seq", n)
    src_ms = _int_column(table, "ts_src_ms", n)
    stable = file_index * (1 << 40) + row_in_file
    order = np.lexsort((stable, seq, t_ns)) if venue in (POLYMARKET, KALSHI) else np.lexsort((stable, t_ns, seq))
    if n and not np.array_equal(order, np.arange(n)):
        table = table.take(pa.array(order))
        t_ns, seq, src_ms, stable = t_ns[order], seq[order], src_ms[order], stable[order]
    tid = oid = fee = fee_token = source_event_type = None
    if venue == HYPERLIQUID:
        tid = np.array(table["tid"].to_pylist(), dtype=object) if "tid" in table.column_names else None
        oid = np.array(table["oid"].to_pylist(), dtype=object) if "oid" in table.column_names else None
        fee = _float_column(table, "fee", n) if "fee" in table.column_names else None
        fee_token = _str_column(table, "fee_token", n)
        source_event_type = _str_column(table, "source_event_type", n)
    return TradeColumns(
        venue=venue,
        instrument_id=instrument_id,
        n=n,
        t_ns=t_ns,
        seq=seq,
        src_ms=src_ms,
        src_row=stable,
        price=_float_column(table, "price", n),
        size=_float_column(table, "size", n),
        side=_str_column(table, "side", n),
        tx_hash=_str_column(table, "trade_id" if venue == KALSHI else "transaction_hash", n),
        tid=tid,
        oid=oid,
        fee=fee,
        fee_token=fee_token,
        source_event_type=source_event_type,
        clock_kind=clock,
    )


def _hl_coin_of(path: Path) -> str:
    """The coin of an HL file: the ``coin`` column's first value (directories are not invertible)."""
    cached = _HL_COIN_CACHE.get(path)
    if cached is not None:
        return cached
    table = pq.read_table(path, columns=["coin"])
    coin = str(table["coin"][0].as_py()) if table.num_rows else path.parent.name.removeprefix("coin=")
    _HL_COIN_CACHE[path] = coin
    return coin


_HL_COIN_CACHE: dict[Path, str] = {}


def pm_tokens(bundle: BundleFiles) -> list[tuple[str, str | None]]:
    """Distinct ``(token_id, condition_id)`` pairs across the bundle's book and trade files."""
    seen: dict[str, str | None] = {}
    for path in [*bundle.books, *bundle.trades]:
        names = pq.ParquetFile(path).schema_arrow.names
        cols = [c for c in ("token_id", "condition_id") if c in names]
        if "token_id" not in cols:
            continue
        table = pq.read_table(path, columns=cols)
        pairs = table.group_by(cols).aggregate([]).to_pylist()
        for row in pairs:
            tok = str(row["token_id"])
            seen.setdefault(tok, row.get("condition_id"))
    return sorted(seen.items())


def kalshi_markets(bundle: BundleFiles) -> list[str]:
    """Distinct market tickers across the bundle's book and trade files."""
    seen: set[str] = set()
    for path in [*bundle.books, *bundle.trades]:
        if "market_ticker" in pq.ParquetFile(path).schema_arrow.names:
            seen.update(str(v) for v in pc.unique(pq.read_table(path, columns=["market_ticker"]).column(0)).to_pylist() if v is not None)
    return sorted(seen)


def kalshi_instrument(ticker: str) -> tuple[Instrument, dict[str, Any]]:
    """A Kalshi market: YES-side book in dollars, $1 contracts, size step 0.01."""
    instrument = Instrument(
        venue=KALSHI,
        product_type="prediction_market",
        instrument_id=f"ks:{ticker}",
        native_symbol=ticker,
        quantity_unit="contracts",
        price_unit="probability",
        entitlement_unit_ids=[],
        outcome_id=ticker,
        event_id=ticker.rsplit("-", 1)[0] if "-" in ticker else None,
        base_asset=None,
        quote_asset="USD",
        tick_size_history_ref="observed_price_grid",
        lot_size_history_ref=None,
        contract_multiplier=Decimal(1),
        tick_size=None,
        lot_increment=Decimal("0.01"),
    )
    return instrument, {"market_ticker": ticker}


def hl_coins(bundle: BundleFiles) -> list[str]:
    coins = {_hl_coin_of(p) for p in [*bundle.books, *bundle.trades]}
    return sorted(coins)


__all__ = [
    "BookColumns",
    "BundleFiles",
    "HL_TICK_CANDIDATES",
    "HYPERLIQUID",
    "KALSHI",
    "KS_TICK_CANDIDATES",
    "PM_TICK_CANDIDATES",
    "POLYMARKET",
    "PmReference",
    "TradeColumns",
    "VENUE_DEPTH",
    "bundle_from_files",
    "dataset_token",
    "decimal_from_float",
    "discover",
    "hl_coins",
    "hl_family",
    "hl_instrument",
    "infer_tick_size",
    "kalshi_instrument",
    "kalshi_markets",
    "ladder_depth_of",
    "pm_instrument",
    "pm_tokens",
    "read_books",
    "read_trades",
]
