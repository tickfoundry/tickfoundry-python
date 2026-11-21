"""``LocalReplay``: replay, as-of lookup and visible-book simulation over downloaded files.

No server is involved (local operation does not require hosted
compute once the data is acquired). The semantics are labcore's (``asof``,
``simulate``) fed by the venue readers of ``tickfoundry.lab.readers``, so a
local result must equal the hosted engine's for the same inputs, source
version and parameters (a parity test holds the two to the same results).

Conventions carried from the contract:

- ``snapshot_at(t)`` is the last row at or before ``t`` (the
  ``searchsorted(side="right") - 1`` rule). With the
  default policy an invalid row (crossed, empty, ladder hole) is returned with
  state ``blocked``; ``policy="last_valid"`` walks back over invalid rows the
  way labcore's ``snapshot_at`` does and reports ``valid_snapshot_age_ns``.
- Hyperliquid adds ``snapshot_at_block(n)`` (last book row with block
  ``<= n``) and the simulator's ``execution_rule="block_n_minus_1"`` (a
  decision at block n executes against the book after block n-1, the public
  guide's convention).
- Quality flags use the hosted API's names; labcore's core names are
  translated at the JSON boundary by ``wire_quality_flags``.
- Locked books (bid == ask) warn and simulate; crossed books block by default
 . labcore 0.1.0 flags both as ``crossed_book``, so the
  adapter distinguishes them from the columns and lifts the block for a
  locked book without touching labcore.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa

from labcore import asof as _asof
from labcore import series as _series
from labcore import fees as _fees
from labcore.cost import NO_VALID_BOOK as _CTM_NO_VALID_BOOK, CostToMove, cost_to_move as _cost_to_move
from labcore.model import CROSSED_BOOK, EMPTY_BOOK, FLOAT_CONVERTED, TRUNCATED_DEPTH, BookSnapshot, Instrument, Trade, to_jsonable
from labcore.simulate import FeeModel, QualityPolicy, SimulationResult, simulate_execution
from labcore.version import ENGINE_VERSION, SCHEMA_VERSION

from . import readers as R
from .types import event_schema, ns_to_rfc3339, to_ns

ADAPTER_VERSION = "tickfoundry-sdk-local/0.1.0"

#: labcore flag -> the hosted API's wire flag. ``float_converted`` is core-only; so is
#: ``truncated_depth``: the book row's attributed flags (``plan_depth_cap``, ``venue_book_deeper``,
#: ``top_of_book_only``) name the cause on the wire instead.
WIRE_FLAG_MAP: dict[str, str | None] = {
    CROSSED_BOOK: R.CROSSED,
    EMPTY_BOOK: R.EMPTY,
    FLOAT_CONVERTED: None,
    TRUNCATED_DEPTH: None,
}
#: The attributed depth flags, in wire order.
DEPTH_ATTRIBUTION_FLAGS = (R.PLAN_DEPTH_CAP, R.VENUE_BOOK_DEEPER, R.TOP_OF_BOOK_ONLY)
#: Flags that are warnings rather than blocks (plus labcore's simulator warnings).
WARNING_FLAGS = frozenset(
    {
        R.LOCKED, R.ONE_SIDED, R.TOP_DISAGREES_WITH_FEED, R.SOURCE_TIME_MISSING, R.NEGATIVE_LATENCY,
        R.REPLAY_ARTIFACT, R.PLAN_DEPTH_CAP, R.VENUE_BOOK_DEEPER, R.TOP_OF_BOOK_ONLY, R.NO_SEED, R.SYNTHESIZED_FILL, R.CLOCK_RESOLUTION_MS,
        "fees_excluded", "lot_increment_assumed", "requested_quantity_rounded_to_lot",
        "visible_depth_may_be_incomplete", "vwap_rounded", "limit_price_off_tick", "reference_unavailable",
        "decision_book_unavailable", "missing_bid_side", "missing_ask_side", "tick_size_changed", "late_snapshots_in_arrival_order",
    }
)

STATE_VALID = "valid"
STATE_BLOCKED = "blocked"
STATE_UNAVAILABLE = "unavailable_before_first_snapshot"


def _plain(value: Any) -> Any:
    """Replace str-mixin Enum members with their values so JSON bodies hold plain ``str``.

    ``labcore.model.to_jsonable`` returns a ``str``-subclass Enum unchanged (it is a
    ``str``); ``json.dumps`` prints its value, but dict consumers would see the Enum.
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def wire_quality_flags(flags: Sequence[str], *, locked: bool = False) -> list[str]:
    """Translate labcore flag names to the hosted API's names, dropping core-only ones.

    A locked book (bid == ask) is ``crossed_book`` to labcore 0.1.0 but
    ``locked`` on the wire; the caller says which it is.
    """
    out: list[str] = []
    for flag in flags:
        if flag == CROSSED_BOOK and locked:
            mapped: str | None = R.LOCKED
        else:
            mapped = WIRE_FLAG_MAP.get(flag, flag)
        if mapped is not None and mapped not in out:
            out.append(mapped)
    return out


class _LazyBooks(Sequence[BookSnapshot]):
    """Sequence view over ``BookColumns`` that materialises rows on demand (for labcore ``snapshot_at``)."""

    def __init__(self, cols: R.BookColumns, dataset_version: str | None) -> None:
        self._cols = cols
        self._dv = dataset_version

    def __len__(self) -> int:
        return self._cols.n

    def __getitem__(self, index: Any) -> BookSnapshot:  # type: ignore[override]
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]  # type: ignore[return-value]
        if index < 0:
            index += len(self)
        return self._cols.snapshot(index, dataset_version=self._dv)


@dataclass(slots=True)
class LocalSnapshot:
    """Result of ``LocalReplay.snapshot_at`` / ``snapshot_at_block`` (``/snapshot``)."""

    instrument_id: str
    t_ns: int | None
    state: str
    ordinal: int | None
    snapshot: BookSnapshot | None
    age_ns: int | None
    valid_snapshot_age_ns: int | None
    skipped_invalid: int
    quality_flags: list[str]
    conditions: list[str]
    block: int | None = None
    query_block: int | None = None
    source_row: int | None = None
    policy: str = "default"

    @property
    def valid(self) -> bool:
        return self.state == STATE_VALID

    @property
    def locked(self) -> bool:
        return R.LOCKED in self.quality_flags

    def to_json(self) -> dict[str, Any]:
        snap = None
        if self.snapshot is not None:
            snap = to_jsonable(self.snapshot)
            snap["quality_flags"] = wire_quality_flags(self.snapshot.quality_flags, locked=self.locked)
            for key in ("instrument_id", "source_order_key", "stable_row_ordinal", "dataset_version", "source_fragment_id"):
                snap.pop(key, None)
        body: dict[str, Any] = {
            "t_ns": None if self.t_ns is None else str(self.t_ns),
            "snapshot": snap,
            "age_ns": None if self.age_ns is None else str(self.age_ns),
            "state": self.state,
            "ordinal": self.ordinal,
            "quality_flags": list(self.quality_flags),
            "conditions": list(self.conditions),
            "policy": self.policy,
        }
        if self.valid_snapshot_age_ns is not None and self.valid_snapshot_age_ns != self.age_ns:
            body["valid_snapshot_age_ns"] = str(self.valid_snapshot_age_ns)
        if self.skipped_invalid:
            body["skipped_invalid"] = self.skipped_invalid
        if self.block is not None:
            body["block"] = self.block
        if self.query_block is not None:
            body["query_block"] = self.query_block
        return _plain(body)


@dataclass(frozen=True, slots=True)
class LocalEvent:
    """One event of the merged replay stream (deterministic order)."""

    ordinal: int
    kind: str
    t_ns: int
    record: BookSnapshot | Trade
    ordering_uncertain: bool
    quality_flags: list[str]
    stream_index: int

    @property
    def is_book(self) -> bool:
        return self.kind == "book"


@dataclass(slots=True)
class LocalSimulation:
    """A local simulation: the labcore result plus the as-of context the engine also reports."""

    result: SimulationResult
    decision: LocalSnapshot
    execution: LocalSnapshot
    instrument: Instrument
    execution_rule: str
    block: int | None
    dataset_version: str
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return self.result.status.value

    @property
    def quality_flags(self) -> list[str]:
        wire = wire_quality_flags(self.result.quality_flags, locked=self.execution.locked)
        if TRUNCATED_DEPTH not in self.result.quality_flags:
            return wire
        # The core said the visible depth hit the cap: the execution row's attributed flags say why
        # (they normally already travel with the row; this guarantees it), ahead of the consequence.
        missing = [f for f in DEPTH_ATTRIBUTION_FLAGS if f in self.execution.quality_flags and f not in wire]
        if not missing:
            return wire
        at = wire.index("visible_depth_may_be_incomplete") if "visible_depth_may_be_incomplete" in wire else len(wire)
        return wire[:at] + missing + wire[at:]

    def fee_estimate(self, *, schedule: Mapping[str, Any] | None = None, fee_multiplier: Any = 1, fee_type: str = "quadratic", taker: bool = True) -> Decimal:
        """ESTIMATED venue fee for this walk from the venue's published rule (``labcore.fees``).

        Polymarket: pass the market's ``fee_schedule``; Kalshi: the series' ``fee_multiplier`` and ``fee_type``.
        Hyperliquid fees are recorded on every fill, so there is no estimate.
        The recorded fee of a real fill is always authoritative.
        """
        return _fees.walk_fee_estimate(self.result.fills, venue=self.instrument.venue, schedule=schedule, fee_multiplier=fee_multiplier, fee_type=fee_type, taker=taker)

    def to_json(self) -> dict[str, Any]:
        """The ``POST /simulations`` response shape (plus extras)."""
        body = to_jsonable(self.result)
        # Fills are {price, quantity, notional}; the fee joins only under an explicit model.
        body["fills"] = [
            {"price": f["price"], "quantity": f["quantity"], "notional": f["notional"], **({"fee": f["fee"]} if f.get("fee") is not None else {})}
            for f in body["fills"]
        ]
        body["quality_flags"] = self.quality_flags
        # The hosted engine's order: the warning flags as they stand in quality_flags, then any extra.
        body["warnings"] = [f for f in body["quality_flags"] if f in WARNING_FLAGS]
        body["warnings"] += [w for w in dict.fromkeys(self.warnings) if w not in body["warnings"]]
        body["decision_snapshot_ordinal"] = self.decision.ordinal
        body["execution_snapshot_ordinal"] = self.execution.ordinal
        body["execution_rule"] = self.execution_rule
        if self.block is not None:
            body["decision_block"] = self.block
            body["execution_block"] = self.execution.block
        body["versions"] = {"engine": ENGINE_VERSION, "adapter": ADAPTER_VERSION, "dataset": self.dataset_version}
        body["schema_version"] = SCHEMA_VERSION
        body["engine_version"] = ENGINE_VERSION
        body["request_id"] = None
        return _plain(body)


@dataclass(slots=True)
class InstrumentInfo:
    instrument: Instrument
    key: str  # token_id or coin
    meta: dict[str, Any]

    @property
    def instrument_id(self) -> str:
        return self.instrument.instrument_id


@dataclass(slots=True)
class LocalCostToMove:
    """``LocalReplay.cost_to_move``: the book in force (``snapshot``) and the cost to move it (``result``)."""

    snapshot: LocalSnapshot
    result: CostToMove

    @property
    def status(self) -> str:
        return self.result.status


class LocalReplay:
    """Replay and simulate over a downloaded bundle."""

    def __init__(self, bundle: R.BundleFiles, *, depth: int | None = None, dataset_version: str | None = None) -> None:
        self._bundle = bundle
        self.venue = bundle.venue
        self.root = bundle.root
        self.depth = depth
        self.dataset_version = dataset_version or R.dataset_token([*bundle.books, *bundle.trades], bundle.root)
        self._books: dict[str, R.BookColumns] = {}
        self._trades: dict[str, R.TradeColumns] = {}
        self._instruments: dict[str, InstrumentInfo] = {}
        self.warnings: list[str] = []
        self._reference = R.PmReference.load(bundle.reference_dir) if bundle.venue == R.POLYMARKET else None
        self._discover_instruments()

    # ----------------------------------------------------------- construction
    @classmethod
    def from_bundle(cls, path: str | Path, *, venue: str | None = None, depth: int | None = None, dates: Sequence[str] | None = None, dataset_version: str | None = None) -> "LocalReplay":
        """Open a downloaded Polymarket bundle or a Hyperliquid pull directory.

        ``depth`` emulates a grant's depth cap (levels beyond it are dropped
        and the counters clamped); ``dates`` restricts
        which ``date=`` folders are loaded (the scope of a multi-day bundle);
        ``venue`` picks one when a directory holds files of both.
        """
        return cls(R.discover(Path(path), dates=dates, venue=venue), depth=depth, dataset_version=dataset_version)

    @classmethod
    def from_files(cls, venue: str, *, books: Sequence[str | Path] = (), trades: Sequence[str | Path] = (), reference_dir: str | Path | None = None, depth: int | None = None, dataset_version: str | None = None) -> "LocalReplay":
        """Open explicit files (for example a fixture variant such as ``pm_l2_depth10.parquet``)."""
        bundle = R.bundle_from_files(venue, books=[Path(p) for p in books], trades=[Path(p) for p in trades], reference_dir=None if reference_dir is None else Path(reference_dir))
        return cls(bundle, depth=depth, dataset_version=dataset_version)

    def _discover_instruments(self) -> None:
        if self.venue == R.POLYMARKET:
            assert self._reference is not None
            for token_id, condition_id in R.pm_tokens(self._bundle):
                instrument, meta = R.pm_instrument(token_id, condition_id, self._reference)
                self._instruments[instrument.instrument_id] = InstrumentInfo(instrument, token_id, meta)
        elif self.venue == R.KALSHI:
            for ticker in R.kalshi_markets(self._bundle):
                instrument, meta = R.kalshi_instrument(ticker)
                self._instruments[instrument.instrument_id] = InstrumentInfo(instrument, ticker, meta)
        else:
            for coin in R.hl_coins(self._bundle):
                instrument, meta = R.hl_instrument(coin)
                quality = self._bundle.quality.get(coin)
                if quality is not None:
                    meta["quality_file"] = str(quality)
                self._instruments[instrument.instrument_id] = InstrumentInfo(instrument, coin, meta)

    # ---------------------------------------------------------------- catalog
    def instrument_ids(self) -> list[str]:
        return sorted(self._instruments)

    def instrument(self, instrument_id: str) -> Instrument:
        return self._info(instrument_id).instrument

    def describe(self, instrument_id: str) -> dict[str, Any]:
        info = self._info(instrument_id)
        books = self.books(instrument_id)
        trades = self.trades(instrument_id)
        return {
            "instrument_id": instrument_id,
            "venue": self.venue,
            **info.meta,
            "book_rows": books.n,
            "trade_rows": trades.n,
            "ladder_depth": books.ladder_depth,
            "first_t_ns": None if not books.n else str(int(books.t_ns[0])),
            "last_t_ns": None if not books.n else str(int(books.t_ns[-1])),
            "first_block": books.block_of(0) if books.n else None,
            "last_block": books.block_of(books.n - 1) if books.n else None,
            "clock_kind": books.clock_kind.value,
            "dataset_version": self.dataset_version,
        }

    def _info(self, instrument_id: str) -> InstrumentInfo:
        try:
            return self._instruments[instrument_id]
        except KeyError:
            raise KeyError(f"unknown instrument {instrument_id!r}; known: {self.instrument_ids()[:10]}") from None

    def books(self, instrument_id: str) -> R.BookColumns:
        if instrument_id not in self._books:
            info = self._info(instrument_id)
            try:
                cols = R.read_books(self._bundle, info.key, depth=self.depth)
            except KeyError:
                cols = _empty_books(self.venue, instrument_id)
            ref_tick = info.meta.get("reference_tick")
            if cols.venue == R.POLYMARKET and ref_tick is not None:
                cols.tick_changed_from_ns = R._pm_tick_refinement_ns(cols, ref_tick)
            self._books[instrument_id] = cols
        return self._books[instrument_id]

    def trades(self, instrument_id: str) -> R.TradeColumns:
        if instrument_id not in self._trades:
            info = self._info(instrument_id)
            self._trades[instrument_id] = R.read_trades(self._bundle, info.key)
        return self._trades[instrument_id]

    # ------------------------------------------------------------------ as-of
    def snapshot_at(self, instrument_id: str, t: Any, *, policy: str = "default", window_start: Any = None) -> LocalSnapshot:
        """Last book row at or before ``t``; see the module docstring for policies."""
        t_ns = to_ns(t, name="t")
        cols = self.books(instrument_id)
        idx = int(np.searchsorted(cols.t_ns, np.int64(t_ns), side="right")) - 1 if cols.n else -1
        return self._resolve(instrument_id, cols, idx, t_ns=t_ns, policy=policy, window_start=window_start)

    def snapshot_at_block(self, instrument_id: str, block: int, *, policy: str = "default") -> LocalSnapshot:
        """Hyperliquid: last book row with block height ``<= block``."""
        cols = self.books(instrument_id)
        if cols.venue != R.HYPERLIQUID:
            raise ValueError("snapshot_at_block is defined for Hyperliquid instruments only")
        idx = int(np.searchsorted(cols.seq, np.int64(block), side="right")) - 1 if cols.n else -1
        t_ns = int(cols.t_ns[idx]) if idx >= 0 else None
        out = self._resolve(instrument_id, cols, idx, t_ns=t_ns, policy=policy, window_start=None)
        out.query_block = int(block)
        return out

    def _resolve(self, instrument_id: str, cols: R.BookColumns, idx: int, *, t_ns: int | None, policy: str, window_start: Any) -> LocalSnapshot:
        if policy not in ("default", "last_valid"):
            raise ValueError("policy must be 'default' or 'last_valid'")
        flags: list[str] = []
        if cols.n == 0:
            flags.append(R.NO_BOOK)
        if idx < 0:
            flags.append(R.NO_SEED)
            return LocalSnapshot(instrument_id, t_ns, STATE_UNAVAILABLE, None, None, None, None, 0, flags, [_asof.NO_SNAPSHOT_AT_OR_BEFORE], policy=policy)
        start_ns = None if window_start is None else to_ns(window_start, name="window_start")
        if start_ns is not None and int(cols.t_ns[idx]) < start_ns:
            # The last state precedes the scope: no seed, never an invented opening book.
            flags.append(R.NO_SEED)
            return LocalSnapshot(instrument_id, t_ns, STATE_UNAVAILABLE, None, None, None, None, 0, flags, [_asof.NO_SNAPSHOT_AT_OR_BEFORE, "state_precedes_window"], policy=policy)
        query_t = int(cols.t_ns[idx]) if t_ns is None else t_ns
        raw_age = query_t - int(cols.t_ns[idx])
        books = _LazyBooks(cols, self.dataset_version)
        valid_mask = cols.valid if policy == "last_valid" else np.ones(cols.n, dtype=bool)
        # labcore does the tie handling, age and conditions; the mask decides which rows count as valid.
        res = _asof.snapshot_at(books, query_t, times=cols.t_ns, valid_mask=valid_mask, window_start_ns=start_ns)
        if res.snapshot is None or res.index is None:
            flags.append(R.NO_SEED)
            return LocalSnapshot(instrument_id, t_ns, STATE_UNAVAILABLE, None, None, raw_age, None, res.skipped_invalid, flags, list(res.conditions), policy=policy)
        i = res.index
        snap = res.snapshot
        state = STATE_VALID if cols.valid[i] else STATE_BLOCKED
        wire = list(dict.fromkeys(flags + list(snap.quality_flags)))
        return LocalSnapshot(
            instrument_id=instrument_id,
            t_ns=t_ns,
            state=state,
            ordinal=int(i),
            snapshot=snap,
            age_ns=raw_age,
            valid_snapshot_age_ns=res.age_ns,
            skipped_invalid=res.skipped_invalid,
            quality_flags=wire,
            conditions=list(res.conditions),
            block=cols.block_of(i),
            source_row=int(cols.src_row[i] & ((1 << 40) - 1)),
            policy=policy,
        )

    # ----------------------------------------------------------- cost to move
    def cost_to_move(
        self,
        instrument_id: str,
        t: Any = None,
        *,
        cents: Decimal | str | int | None = None,
        odds: Decimal | str | int | None = None,
        bps: Decimal | str | int | None = None,
        block: int | None = None,
        policy: str = "default",
    ) -> LocalCostToMove:
        """Cost to move the book in force at ``t`` (or Hyperliquid ``block``) by ``cents``, ``odds`` or ``bps``.

        The book is ``snapshot_at`` / ``snapshot_at_block`` (the Lab's as-of rule); the number is
        ``labcore.cost.cost_to_move``, the per-book figure of the TickFoundry cost-to-move index.
        Polymarket books are binary (``cents`` / ``odds`` / ``bps``); Hyperliquid books take ``bps``.
        A blocked or unavailable book has no cost (``status = "no_valid_book"``).
        """
        if block is not None:
            snap = self.snapshot_at_block(instrument_id, block, policy=policy)
        else:
            if t is None:
                raise ValueError("pass t (or block for Hyperliquid)")
            snap = self.snapshot_at(instrument_id, t, policy=policy)
        cols = self.books(instrument_id)
        binary = cols.venue != R.HYPERLIQUID
        if snap.state != STATE_VALID or snap.snapshot is None or snap.ordinal is None:
            # Validate the request the same way, then report the missing book.
            probe = BookSnapshot(instrument_id, 0, [], [])
            res = _cost_to_move(probe, cents=cents, odds=odds, bps=bps, binary=binary)
            res.status = _CTM_NO_VALID_BOOK
            return LocalCostToMove(snap, res)
        i = int(snap.ordinal)
        totals: list[int | None] = [None, None]
        for k, arr in enumerate((cols.bid_levels, cols.ask_levels)):
            if arr is None:
                continue
            visible = len(snap.snapshot.bids if k == 0 else snap.snapshot.asks)
            count = int(arr[i])
            if self.depth is not None and visible >= self.depth:
                continue                      # counters were clamped to the emulated plan depth: unknown
            if cols.venue == R.HYPERLIQUID and count >= R.VENUE_DEPTH:
                continue                      # Hyperliquid counters stop at 25: a side at 25 may go on
            totals[k] = count
        res = _cost_to_move(snap.snapshot, cents=cents, odds=odds, bps=bps, binary=binary,
                            bid_levels_total=totals[0], ask_levels_total=totals[1])
        return LocalCostToMove(snap, res)

    # ----------------------------------------------------------------- series
    RESOLUTIONS = {"1s": 1, "10s": 10, "1m": 60, "5m": 300, "10m": 600, "15m": 900, "1h": 3600}

    def _day_window(self, cols: R.BookColumns, start: Any, end: Any) -> tuple[int, int]:
        if start is not None and end is not None:
            return to_ns(start, name="start"), to_ns(end, name="end")
        if cols.n == 0:
            raise ValueError("no book rows: pass start and end")
        day = int(cols.t_ns[0]) // (86_400 * 1_000_000_000) * 86_400 * 1_000_000_000
        s0 = to_ns(start, name="start") if start is not None else day
        e0 = to_ns(end, name="end") if end is not None else day + 86_400 * 1_000_000_000
        return s0, e0

    def _series_events(self, instrument_id: str) -> pa.Table:
        """The columns the series code reads, straight from the columnar arrays: book rows in the
        venue's total order with each side's ladder up to its first hole (the hosted index keeps the
        same prefix), then the trades."""
        cols = self.books(instrument_id)
        tr = self.trades(instrument_id)

        def ladder(px: np.ndarray, counts: np.ndarray) -> pa.Array:
            counts = np.asarray(counts, dtype=np.int64)
            offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
            if cols.n and px.shape[1]:
                mask = np.arange(px.shape[1])[None, :] < counts[:, None]
                flat = px[mask]
            else:
                flat = np.zeros(0)
            return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), pa.array(flat, type=pa.float64()))

        nb, nt = cols.n, tr.n
        book = pa.table({
            "t_ns": pa.array(cols.t_ns, type=pa.int64()),
            "kind": pa.array(["book"] * nb, type=pa.string()),
            "bid_px": ladder(cols.bid_px, cols.n_bids), "bid_sz": ladder(cols.bid_sz, cols.n_bids),
            "ask_px": ladder(cols.ask_px, cols.n_asks), "ask_sz": ladder(cols.ask_sz, cols.n_asks),
            "trade_price": pa.nulls(nb, type=pa.float64()), "trade_size": pa.nulls(nb, type=pa.float64()),
        })
        empty = pa.ListArray.from_arrays(pa.array(np.zeros(nt + 1, dtype=np.int32)), pa.array([], type=pa.float64()))
        trades = pa.table({
            "t_ns": pa.array(tr.t_ns, type=pa.int64()),
            "kind": pa.array(["trade"] * nt, type=pa.string()),
            "bid_px": empty, "bid_sz": empty, "ask_px": empty, "ask_sz": empty,
            "trade_price": pa.array(np.asarray(tr.price, dtype=np.float64), type=pa.float64()),
            "trade_size": pa.array(np.asarray(tr.size, dtype=np.float64), type=pa.float64()),
        })
        return pa.concat_tables([book, trades])

    def series(
        self,
        instrument_id: str,
        *,
        start: Any = None,
        end: Any = None,
        res: str | int = "1m",
        micro: bool = True,
        band: tuple[str, Any] | None = None,
        imbalance_levels: int | None = None,
        ohlc: bool = False,
        gaps: Sequence[tuple[Any, Any]] = (),
    ) -> pa.Table:
        """Time-weighted bins of the book and the tape, the hosted Lab's ``/series`` table (same code).

        Standard columns: ``t_ns`` (bin start), ``best_bid``, ``best_ask``, ``mid``, ``spread``, each
        side's depth quantity and notional, ``trade_count``, ``trade_qty``, ``vwap``, ``coverage`` and
        ``flags``. A book value is the mean over the time a valid state was in force inside the bin;
        a state is carried forward until the next book row, a known gap or the window end, and never
        from data at or after the bin's end. ``micro`` adds ``microprice`` and ``depth_imbalance``;
        ``band=("cents", "0.02")`` or ``("bps", 10)`` adds the quantity and notional resting within
        that distance of the mid on each side; ``ohlc`` adds trade candles. ``start`` / ``end``
        default to the UTC day of the data; ``gaps`` are known capture gaps ``(start, end)``.
        """
        width = self.RESOLUTIONS[res] if isinstance(res, str) else int(res)
        if width <= 0:
            raise ValueError("res must be positive")
        cols = self.books(instrument_id)
        start_ns, end_ns = self._day_window(cols, start, end)
        if end_ns <= start_ns:
            raise ValueError("end must be after start")
        info = self._info(instrument_id)
        mult = info.instrument.contract_multiplier if info.instrument.contract_multiplier is not None else Decimal(1)
        band_f = None if band is None else (band[0], float(Decimal(str(band[1]))))
        gap_ns = [(to_ns(a, name="gap start"), to_ns(b, name="gap end")) for a, b in gaps]
        out = _series.build_series_fast(
            self._series_events(instrument_id), multiplier=mult, depth=cols.ladder_depth, start_ns=start_ns, end_ns=end_ns,
            gaps=_asof.merge_intervals(gap_ns), resolutions={"r": width * 1_000_000_000}, max_bins=10_000_000,
            micro=micro, band=band_f, ohlc=ohlc, imbalance_levels=imbalance_levels,
        )
        return out["r"]

    def candles(self, instrument_id: str, *, start: Any = None, end: Any = None, res: str | int = "1m", gaps: Sequence[tuple[Any, Any]] = ()) -> pa.Table:
        """Trade candles from the tape per bin: open / high / low / close (first, highest, lowest, last
        trade price), ``trade_count``, ``trade_qty``, ``trade_notional``, ``vwap``. A bin without trades
        has null prices; its count is 0 when capture covered it and null when it did not."""
        t = self.series(instrument_id, start=start, end=end, res=res, micro=False, ohlc=True, gaps=gaps)
        return t.select(["t_ns", "open", "high", "low", "close", "trade_count", "trade_qty", "trade_notional", "vwap", "coverage", "flags"])

    def snapshots(self, instrument_id: str, *, start: Any, end: Any, every: str | int = "1m", policy: str = "default") -> pa.Table:
        """The book in force at fixed times ``start, start + every, ... < end`` (the ``snapshot_at`` rule
        at each point): ``state``, best bid / ask and their sizes, ``mid``, ``spread``, ``age_ns`` and the
        row ``ordinal``. A blocked row (crossed, empty, ladder hole) has state ``blocked`` and no mid;
        ``policy="last_valid"`` walks back to the last valid row instead."""
        step = (self.RESOLUTIONS[every] if isinstance(every, str) else int(every)) * 1_000_000_000
        s0, e0 = to_ns(start, name="start"), to_ns(end, name="end")
        if step <= 0 or e0 <= s0:
            raise ValueError("every must be positive and end after start")
        grid = np.arange(s0, e0, step, dtype=np.int64)
        recs = [self.snapshot_at(instrument_id, int(t), policy=policy) for t in grid]

        def top(r: LocalSnapshot, side: str, k: str):
            if r.snapshot is None:
                return None
            levels = r.snapshot.bids if side == "bid" else r.snapshot.asks
            if not levels:
                return None
            return float(levels[0].price if k == "px" else levels[0].quantity)

        bb = [top(r, "bid", "px") for r in recs]
        ba = [top(r, "ask", "px") for r in recs]
        ok = [r.state == STATE_VALID and b is not None and a is not None and b < a for r, b, a in zip(recs, bb, ba)]
        return pa.table({
            "t_ns": pa.array(grid, type=pa.int64()),
            "state": pa.array([r.state for r in recs], type=pa.string()),
            "best_bid": pa.array(bb, type=pa.float64()),
            "best_bid_size": pa.array([top(r, "bid", "sz") for r in recs], type=pa.float64()),
            "best_ask": pa.array(ba, type=pa.float64()),
            "best_ask_size": pa.array([top(r, "ask", "sz") for r in recs], type=pa.float64()),
            "mid": pa.array([(b + a) / 2 if o else None for b, a, o in zip(bb, ba, ok)], type=pa.float64()),
            "spread": pa.array([a - b if o else None for b, a, o in zip(bb, ba, ok)], type=pa.float64()),
            "age_ns": pa.array([r.age_ns for r in recs], type=pa.int64()),
            "ordinal": pa.array([r.ordinal for r in recs], type=pa.int64()),
        })

    # ---------------------------------------------------------------- events
    def _window(self, cols_t: np.ndarray, cols_src_ms: np.ndarray, start: Any, end: Any) -> tuple[int, int]:
        """Row range [lo, hi) of a time window. HL filters on block time ``ts_src_ms``."""
        if self.venue == R.HYPERLIQUID:
            keys = cols_src_ms * 1_000_000
        else:
            keys = cols_t
        lo = 0 if start is None else int(np.searchsorted(keys, np.int64(to_ns(start, name="start")), side="left"))
        hi = len(keys) if end is None else int(np.searchsorted(keys, np.int64(to_ns(end, name="end")), side="left"))
        return lo, max(lo, hi)

    def _merged_index(self, instrument_id: str, start: Any, end: Any, products: Sequence[str]) -> tuple[R.BookColumns, R.TradeColumns, np.ndarray, np.ndarray, np.ndarray]:
        """Deterministic merge of books and trades on labcore's key without materialising records.

        Order: ``(analysis_time_ns, source_order_key, stable_row_ordinal)`` then book before
        trade on a full-key tie (``labcore.asof.merge_books_and_trades``). Ties on
        ``analysis_time_ns`` across streams are ``ordering_uncertain`` on Polymarket; on
        Hyperliquid the block height is a shared native sequence, so only full-key ties are.
        """
        books = self.books(instrument_id)
        trades = self.trades(instrument_id)
        want_books = "book" in products
        want_trades = "trades" in products or "trade" in products
        b_lo, b_hi = self._window(books.t_ns, books.src_ms, start, end) if want_books else (0, 0)
        t_lo, t_hi = self._window(trades.t_ns, trades.src_ms, start, end) if want_trades else (0, 0)
        b_idx = np.arange(b_lo, b_hi, dtype=np.int64)
        t_idx = np.arange(t_lo, t_hi, dtype=np.int64)
        times = np.concatenate([books.t_ns[b_lo:b_hi], trades.t_ns[t_lo:t_hi]])
        seqs = np.concatenate([books.seq[b_lo:b_hi], trades.seq[t_lo:t_hi]])
        ordinals = np.concatenate([b_idx, t_idx])
        kinds = np.concatenate([np.zeros(len(b_idx), dtype=np.int8), np.ones(len(t_idx), dtype=np.int8)])
        order = np.lexsort((kinds, ordinals, seqs, times))
        # ordering_uncertain per labcore's rule.
        uncertain = np.zeros(len(order), dtype=bool)
        if len(order) and len(b_idx) and len(t_idx):
            st, sk, ss, so = times[order], kinds[order], seqs[order], ordinals[order]
            if self.venue == R.HYPERLIQUID:
                full = np.stack([st, ss, so], axis=1)
                _, inverse, counts = np.unique(full, axis=0, return_inverse=True, return_counts=True)
                inverse = inverse.reshape(-1)
                # a full-key group containing both kinds
                has_book = np.zeros(len(counts), dtype=bool)
                has_trade = np.zeros(len(counts), dtype=bool)
                has_book[inverse[sk == 0]] = True
                has_trade[inverse[sk == 1]] = True
                uncertain = (has_book & has_trade)[inverse]
            else:
                _, inverse = np.unique(st, return_inverse=True)
                inverse = inverse.reshape(-1)
                has_book = np.zeros(inverse.max() + 1, dtype=bool)
                has_trade = np.zeros(inverse.max() + 1, dtype=bool)
                has_book[inverse[sk == 0]] = True
                has_trade[inverse[sk == 1]] = True
                uncertain = (has_book & has_trade)[inverse]
            if self.venue == R.KALSHI:
                # Trades and book deltas arrive on different connections: within 30 ms they cannot be ordered.
                bt, tt = st[sk == 0], st[sk == 1]

                def near(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
                    j = np.searchsorted(ref, x)
                    prev = (j > 0) & (x - ref[np.maximum(j - 1, 0)] <= R.KS_UNCERTAINTY_WINDOW_NS)
                    nxt = (j < len(ref)) & (ref[np.minimum(j, len(ref) - 1)] - x <= R.KS_UNCERTAINTY_WINDOW_NS)
                    return prev | nxt

                uncertain = uncertain.copy()
                uncertain[sk == 1] |= near(tt, bt)
                uncertain[sk == 0] |= near(bt, tt)
        return books, trades, kinds[order], ordinals[order], uncertain

    def seed_state(self, instrument_id: str, start: Any = None, *, window_start: Any = None) -> str:
        """Contract ``seed_state`` of a window: is a valid state known at ``start``?

        ``seeded_from_prior_rows`` when the bundle holds a valid book at or before
        ``start`` (inside the same day, or a prior day the caller loaded);
        ``unavailable_until_first_snapshot`` otherwise (the first events carry
        ``no_seed``); ``not_requested`` for an unbounded replay.
        """
        if start is None:
            return "not_requested"
        seed = self.snapshot_at(instrument_id, start, policy="last_valid", window_start=window_start)
        return "seeded_from_prior_rows" if seed.state == STATE_VALID else "unavailable_until_first_snapshot"

    def window_info(self, instrument_id: str, *, start: Any = None, end: Any = None, products: Sequence[str] = ("book", "trades"), window_start: Any = None) -> dict[str, Any]:
        """What ``POST /replays`` reports for a window: event count, seed state, clock kind, depth."""
        books, _trades, kinds, _ordinals, _unc = self._merged_index(instrument_id, start, end, products)
        return {
            "instrument_id": instrument_id,
            "event_count": int(len(kinds)),
            "book_events": int((kinds == 0).sum()),
            "trade_events": int((kinds == 1).sum()),
            "seed_state": self.seed_state(instrument_id, start, window_start=window_start),
            "clock_kind": books.clock_kind.value,
            "depth": int(books.ladder_depth),
            "dataset_version": self.dataset_version,
        }

    def events(self, instrument_id: str, *, start: Any = None, end: Any = None, products: Sequence[str] = ("book", "trades"), window_start: Any = None) -> Iterator[LocalEvent]:
        """Yield the merged event stream of one instrument in the deterministic total order (records built lazily).

        When no valid state exists at ``start`` the first book event carries ``no_seed``
        (contract ``POST /replays`` seeding rule); the state before it is unknown, never invented.
        """
        books, trades, kinds, ordinals, uncertain = self._merged_index(instrument_id, start, end, products)
        need_seed_flag = self.seed_state(instrument_id, start, window_start=window_start) == "unavailable_until_first_snapshot"
        for n, (kind, i, unc) in enumerate(zip(kinds.tolist(), ordinals.tolist(), uncertain.tolist())):
            if kind == 0:
                record: BookSnapshot | Trade = books.snapshot(i, dataset_version=self.dataset_version)
                if need_seed_flag:
                    record.quality_flags.append(R.NO_SEED)
                    need_seed_flag = False
                yield LocalEvent(n, "book", int(record.analysis_time_ns), record, unc, list(record.quality_flags), i)
            else:
                trade = trades.trade(i, dataset_version=self.dataset_version)
                yield LocalEvent(n, "trade", int(trade.analysis_time_ns), trade, unc, list(trade.quality_flags), i)

    def event_batches(self, instrument_id: str, *, start: Any = None, end: Any = None, products: Sequence[str] = ("book", "trades"), page_events: int = 10_000) -> Iterator[pa.RecordBatch]:
        """The contract's ``/events`` Arrow pages built from the columnar arrays (no Decimal materialisation)."""
        if page_events < 1:
            raise ValueError("page_events must be >= 1")
        books, trades, kinds, ordinals, uncertain = self._merged_index(instrument_id, start, end, products)
        hl = self.venue == R.HYPERLIQUID
        schema = event_schema(with_order_counts=hl)
        need_seed_flag = self.seed_state(instrument_id, start) == "unavailable_until_first_snapshot"
        total = len(kinds)
        for lo in range(0, total, page_events):
            hi = min(total, lo + page_events)
            rows = _EventRows(hl)
            for n in range(lo, hi):
                i = int(ordinals[n])
                if kinds[n] == 0:
                    rows.add_book(n, books, i, bool(uncertain[n]))
                    if need_seed_flag:
                        rows.cols["quality_flags"][-1] = [*rows.cols["quality_flags"][-1], R.NO_SEED]
                        need_seed_flag = False
                else:
                    rows.add_trade(n, trades, i, bool(uncertain[n]))
            yield rows.batch(schema)

    # -------------------------------------------------------------- simulate
    def simulate(
        self,
        instrument_id: str,
        timestamp: Any = None,
        *,
        side: str,
        size_mode: str,
        size: Decimal | str | int,
        limit_price: Decimal | str | None = None,
        assumed_delay_ms: int = 0,
        fee_model: str | FeeModel = "excluded",
        explicit_parameters: Mapping[str, Any] | None = None,
        fee_model_id: str | None = None,
        reference: str = "execution_best_quote",
        block: int | None = None,
        execution_rule: str = "as_of_time",
        book_policy: str = "default",
        window_start: Any = None,
        lot_increment: Decimal | str | None = None,
        allow_crossed: bool = False,
        max_snapshot_age_ns: int | None = None,
    ) -> LocalSimulation:
        """Visible-book execution analysis at ``timestamp`` (inputs) through ``labcore.simulate_execution``.

        ``execution_rule="block_n_minus_1"`` (Hyperliquid) executes a decision at
        ``block`` n against the book after block n-1; ``as_of_time`` executes against
        the last book at or before ``timestamp + assumed_delay_ms``. Each call is an
        independent scenario; nothing is consumed.
        """
        info = self._info(instrument_id)
        cols = self.books(instrument_id)
        warnings: list[str] = []
        if execution_rule not in ("as_of_time", "block_n_minus_1"):
            raise ValueError("execution_rule must be 'as_of_time' or 'block_n_minus_1'")
        if execution_rule == "block_n_minus_1":
            if cols.venue != R.HYPERLIQUID:
                raise ValueError("block_n_minus_1 is a Hyperliquid rule")
            if block is None:
                raise ValueError("block_n_minus_1 needs the decision block")
            t_ns = self._block_time(instrument_id, int(block)) if timestamp is None else to_ns(timestamp, name="timestamp")
            if t_ns is None:
                raise ValueError(f"block {block} is not in the data; pass timestamp explicitly")
            decision = self.snapshot_at_block(instrument_id, int(block) - 1, policy=book_policy)
            execution = decision
            if execution.snapshot is not None and execution.snapshot.analysis_time_ns > t_ns:
                # Two blocks inside one millisecond: keep no-look-ahead by moving the decision
                # instant to the book row's own time.
                t_ns = execution.snapshot.analysis_time_ns
                warnings.append("same_millisecond_blocks")
            delay = 0
        else:
            if timestamp is None:
                raise ValueError("timestamp is required")
            t_ns = to_ns(timestamp, name="timestamp")
            delay = int(assumed_delay_ms)
            decision = self.snapshot_at(instrument_id, t_ns, policy=book_policy, window_start=window_start)
            execution = decision if delay == 0 else self.snapshot_at(instrument_id, t_ns + delay * 1_000_000, policy=book_policy, window_start=window_start)
        exec_book = execution.snapshot if execution.state != STATE_UNAVAILABLE else None
        decision_book = decision.snapshot if decision.state != STATE_UNAVAILABLE else None
        instrument = replace(info.instrument)
        if lot_increment is not None:
            instrument.lot_increment = Decimal(str(lot_increment))
        if exec_book is not None:
            instrument.tick_size = self._tick_in_force(exec_book)
        model = fee_model if isinstance(fee_model, FeeModel) else self._fee_model(fee_model, explicit_parameters, fee_model_id)
        blocking = set(QualityPolicy().blocking_flags) | {R.LADDER_HOLE}
        if allow_crossed or (execution.locked and not (exec_book is not None and R.CROSSED in exec_book.quality_flags)):
            # A locked book warns and simulates; labcore 0.1.0 would block it as crossed.
            blocking.discard(CROSSED_BOOK)
        policy = QualityPolicy(blocking_flags=frozenset(blocking), depth_cap=cols.ladder_depth or None, max_snapshot_age_ns=max_snapshot_age_ns)
        result = simulate_execution(
            instrument,
            exec_book,
            side,
            size_mode,
            size,
            limit_price=limit_price,
            fee_model=model,
            reference=reference,
            decision_book=decision_book if decision_book is not exec_book else None,
            analysis_timestamp_ns=t_ns,
            assumed_delay_ms=delay,
            quality_policy=policy,
        )
        # No book at or before the execution time: the result says so (status no_valid_book); like the
        # hosted engine, the simulation adds no ``no_seed`` warning (the snapshot itself still carries it).
        return LocalSimulation(result, decision, execution, instrument, execution_rule, block, self.dataset_version, warnings)

    def _block_time(self, instrument_id: str, block: int) -> int | None:
        """Block time of ``block`` in ns from its book row, else from its fills (``ts_src_ms * 1e6``)."""
        cols = self.books(instrument_id)
        j = int(np.searchsorted(cols.seq, np.int64(block), side="left"))
        if j < cols.n and int(cols.seq[j]) == block:
            return int(cols.src_ms[j]) * 1_000_000
        trades = self.trades(instrument_id)
        j = int(np.searchsorted(trades.seq, np.int64(block), side="left"))
        if j < trades.n and int(trades.seq[j]) == block:
            return int(trades.src_ms[j]) * 1_000_000
        return None

    def _tick_in_force(self, book: BookSnapshot) -> Decimal | None:
        """Tick size inferred from the execution book's price grid (lifecycle feed absent)."""
        prices = [lvl.price for lvl in [*book.bids, *book.asks]]
        candidates = {R.POLYMARKET: R.PM_TICK_CANDIDATES, R.KALSHI: R.KS_TICK_CANDIDATES}.get(self.venue, R.HL_TICK_CANDIDATES)
        return R.infer_tick_size(prices, candidates)

    @staticmethod
    def _fee_model(kind: str, params: Mapping[str, Any] | None, model_id: str | None) -> FeeModel:
        if kind == "excluded":
            return FeeModel.excluded()
        if kind == "explicit_model":
            params = dict(params or {})
            return FeeModel.explicit(
                bps=params.get("bps"),
                fixed_per_unit=params.get("fixed_per_unit"),
                fee_asset=params.get("fee_asset"),
                model_id=model_id,
                included_in_budget=bool(params.get("included_in_budget", False)),
            )
        raise ValueError("fee_model must be 'excluded' or 'explicit_model'")

    # -------------------------------------------------------------- helpers
    def coverage(self, instrument_id: str) -> dict[str, Any]:
        """First/last analysis times and blocks of the loaded rows (the window the data really covers)."""
        cols = self.books(instrument_id)
        if not cols.n:
            return {"book_rows": 0}
        return {
            "book_rows": int(cols.n),
            "first_t_ns": str(int(cols.t_ns[0])),
            "last_t_ns": str(int(cols.t_ns[-1])),
            "first_iso": ns_to_rfc3339(int(cols.t_ns[0])),
            "last_iso": ns_to_rfc3339(int(cols.t_ns[-1])),
            "first_block": cols.block_of(0),
            "last_block": cols.block_of(cols.n - 1),
            "invalid_rows": int((~cols.valid).sum()),
        }


def _empty_books(venue: str, instrument_id: str) -> R.BookColumns:
    from labcore.model import ClockKind

    empty = np.empty(0, dtype=np.int64)
    cols = R.BookColumns(
        venue=venue, instrument_id=instrument_id, n=0, ladder_depth=0, t_ns=empty, seq=empty, src_ms=empty, src_row=empty,
        bid_px=np.empty((0, 0)), bid_sz=np.empty((0, 0)), ask_px=np.empty((0, 0)), ask_sz=np.empty((0, 0)),
        bid_n=None, ask_n=None, bid_levels=None, ask_levels=None, reconciles=None, event_kind=None,
        clock_kind=ClockKind.reconstructed_block if venue == R.HYPERLIQUID else ClockKind.collector_observed,
    )
    return cols.finish()


class _EventRows:
    """Column accumulator for one Arrow page in the contract's ``/events`` schema."""

    def __init__(self, hl: bool) -> None:
        self.hl = hl
        self.cols: dict[str, list[Any]] = {
            k: [] for k in (
                "ordinal", "t_ns", "kind", "source_time_ms", "source_seq", "ordering_uncertain", "best_bid", "best_bid_size",
                "best_ask", "best_ask_size", "bid_px", "bid_sz", "ask_px", "ask_sz", "bid_n", "ask_n", "trade_price",
                "trade_size", "trade_side", "trade_id", "quality_flags",
            )
        }

    def add_book(self, n: int, books: R.BookColumns, i: int, uncertain: bool) -> None:
        c = self.cols
        nb, na = int(books.n_bids[i]), int(books.n_asks[i])
        src = int(books.src_ms[i])
        c["ordinal"].append(n)
        c["t_ns"].append(int(books.t_ns[i]))
        c["kind"].append("book")
        c["source_time_ms"].append(None if (books.venue in (R.POLYMARKET, R.KALSHI) and src == 0) else src)
        c["source_seq"].append(int(books.seq[i]))
        c["ordering_uncertain"].append(uncertain)
        c["best_bid"].append(float(books.bid_px[i, 0]) if nb else None)
        c["best_bid_size"].append(float(books.bid_sz[i, 0]) if nb else None)
        c["best_ask"].append(float(books.ask_px[i, 0]) if na else None)
        c["best_ask_size"].append(float(books.ask_sz[i, 0]) if na else None)
        c["bid_px"].append(books.bid_px[i, :nb].tolist())
        c["bid_sz"].append(books.bid_sz[i, :nb].tolist())
        c["ask_px"].append(books.ask_px[i, :na].tolist())
        c["ask_sz"].append(books.ask_sz[i, :na].tolist())
        if self.hl:
            c["bid_n"].append(None if books.bid_n is None else [None if v < 0 else int(v) for v in books.bid_n[i, :nb].tolist()])
            c["ask_n"].append(None if books.ask_n is None else [None if v < 0 else int(v) for v in books.ask_n[i, :na].tolist()])
        c["trade_price"].append(None)
        c["trade_size"].append(None)
        c["trade_side"].append(None)
        c["trade_id"].append(None)
        c["quality_flags"].append(books.wire_flags(i))

    def add_trade(self, n: int, trades: R.TradeColumns, i: int, uncertain: bool) -> None:
        c = self.cols
        src = int(trades.src_ms[i])
        side = trades.side[i]
        c["ordinal"].append(n)
        c["t_ns"].append(int(trades.t_ns[i]))
        c["kind"].append("trade")
        c["source_time_ms"].append(None if (trades.venue in (R.POLYMARKET, R.KALSHI) and src == 0) else src)
        c["source_seq"].append(int(trades.seq[i]))
        c["ordering_uncertain"].append(uncertain)
        for key in ("best_bid", "best_bid_size", "best_ask", "best_ask_size"):
            c[key].append(None)
        for key in ("bid_px", "bid_sz", "ask_px", "ask_sz"):
            c[key].append(None)
        if self.hl:
            c["bid_n"].append(None)
            c["ask_n"].append(None)
        c["trade_price"].append(float(trades.price[i]))
        c["trade_size"].append(float(trades.size[i]))
        if isinstance(side, str) and trades.venue == R.KALSHI:
            side = {"yes": "buy", "no": "sell"}.get(side.lower(), side)   # the taker's outcome side, in YES-book terms
        c["trade_side"].append(None if not isinstance(side, str) else side.lower())
        c["trade_id"].append(trades.trade_id(i))
        c["quality_flags"].append(trades.wire_flags(i))

    def batch(self, schema: pa.Schema) -> pa.RecordBatch:
        arrays = []
        for fld in schema:
            values = self.cols[fld.name]
            if pa.types.is_dictionary(fld.type):
                arrays.append(pa.array(values, type=pa.string()).dictionary_encode().cast(fld.type))
            else:
                arrays.append(pa.array(values, type=fld.type))
        return pa.RecordBatch.from_arrays(arrays, schema=schema)


__all__ = [
    "ADAPTER_VERSION",
    "InstrumentInfo",
    "LocalEvent",
    "LocalReplay",
    "LocalSimulation",
    "LocalSnapshot",
    "STATE_BLOCKED",
    "STATE_UNAVAILABLE",
    "STATE_VALID",
    "WARNING_FLAGS",
    "WIRE_FLAG_MAP",
    "wire_quality_flags",
]
