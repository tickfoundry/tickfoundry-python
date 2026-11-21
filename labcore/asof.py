"""As-of snapshot lookup and deterministic book/trade merge.

``snapshot_at`` returns the last *valid* snapshot at or before ``t`` (never
the closest on either side) together with its age and the gap/seed/staleness
conditions the caller must surface. ``merge_books_and_trades`` produces a
deterministic replay order and labels ties whose cross-stream order is not
established by the source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Iterator, Sequence

import numpy as np

from .model import (
    BookSnapshot,
    ClockKind,
    OrderingKey,
    Trade,
    is_valid_snapshot,
)

# Conditions reported by snapshot_at.
NO_SNAPSHOT_AT_OR_BEFORE = "no_snapshot_at_or_before"
NO_VALID_SNAPSHOT_AT_OR_BEFORE = "no_valid_snapshot_at_or_before"
INVALID_SNAPSHOTS_SKIPPED = "invalid_snapshots_skipped"
KNOWN_GAP_SINCE_SNAPSHOT = "known_gap_since_snapshot"
AGE_EXCEEDS_MAX = "age_exceeds_max"
SEEDED_BEFORE_WINDOW = "seeded_before_window"
EXACT_TIMESTAMP_MATCH = "exact_timestamp_match"
TIMESTAMP_TIE_RESOLVED_BY_ORDINAL = "timestamp_tie_resolved_by_ordinal"

ORDERING_NOTE = (
    "last snapshot with analysis_time_ns <= t in the total order "
    "(analysis_time_ns, source_order_key, stable_row_ordinal); equal "
    "timestamps are resolved by the key and ordinal, not by observation order"
)


class AsOfStatus(str, Enum):
    """Whether the returned state may be used as the book at ``t``."""

    available = "available"
    stale = "stale"
    gap_since_snapshot = "gap_since_snapshot"
    unavailable = "unavailable"


@dataclass(slots=True)
class AsOfResult:
    """Result of ``snapshot_at`` (age, clock kind, ordering, gap/seed)."""

    query_time_ns: int
    snapshot: BookSnapshot | None
    index: int | None
    age_ns: int | None
    status: AsOfStatus
    conditions: list[str] = field(default_factory=list)
    skipped_invalid: int = 0
    clock_kind: ClockKind | None = None
    ordering_note: str = ORDERING_NOTE

    @property
    def usable(self) -> bool:
        return self.status is AsOfStatus.available


def snapshot_times(snapshots: Sequence[BookSnapshot]) -> np.ndarray:
    """int64 array of ``analysis_time_ns`` for a sorted snapshot sequence.

    Raises ``ValueError`` when the sequence is not sorted by the total ordering key.
    """
    times = np.fromiter((s.analysis_time_ns for s in snapshots), dtype=np.int64, count=len(snapshots))
    if len(snapshots) > 1:
        if np.any(np.diff(times) < 0):
            raise ValueError("snapshots must be sorted by analysis_time_ns")
        previous = snapshots[0].ordering_key()
        for snap in snapshots[1:]:
            current = snap.ordering_key()
            if current < previous:
                raise ValueError("snapshots must be sorted by the total ordering key")
            previous = current
    return times


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and merge half-open ``[start, end)`` intervals; empty ones are dropped."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted((int(s), int(e)) for s, e in intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def snapshot_at(
    snapshots_sorted: Sequence[BookSnapshot],
    t_ns: int,
    *,
    times: np.ndarray | None = None,
    valid_mask: Sequence[bool] | np.ndarray | None = None,
    max_age_ns: int | None = None,
    known_gaps: Iterable[tuple[int, int]] = (),
    window_start_ns: int | None = None,
    depth_cap: int | None = None,
) -> AsOfResult:
    """Return the last valid snapshot at or before ``t_ns``.

    ``snapshots_sorted`` must be sorted by the total ordering key. The lookup
    is a binary search over the int64 ``analysis_time_ns`` array (``times``,
    computed when omitted); among equal timestamps the last element in the
    total order wins, i.e. the tie is broken by ``source_order_key`` and then
    ``stable_row_ordinal``. Invalid (crossed/empty) snapshots are skipped
    backwards and counted in ``skipped_invalid``.

    Conditions: ``known_gap_since_snapshot`` when a known capture
    gap starts after the snapshot and at or before ``t`` (the state at ``t``
    is unknown, status ``gap_since_snapshot``); ``age_exceeds_max`` when
    ``max_age_ns`` is given and exceeded (status ``stale``; age alone is not
    proof of an outage); ``seeded_before_window`` when the snapshot precedes
    ``window_start_ns`` (a preceding seed was used, the caller must have
    checked the entitlement rule); no snapshot at or before ``t`` gives
    status ``unavailable`` and never an opening book invented from later data.
    """
    if times is None:
        times = snapshot_times(snapshots_sorted)
    if len(times) != len(snapshots_sorted):
        raise ValueError("times must align with snapshots_sorted")

    if len(times) == 0:
        return AsOfResult(t_ns, None, None, None, AsOfStatus.unavailable, [NO_SNAPSHOT_AT_OR_BEFORE])

    idx = int(np.searchsorted(times, np.int64(t_ns), side="right")) - 1
    if idx < 0:
        return AsOfResult(t_ns, None, None, None, AsOfStatus.unavailable, [NO_SNAPSHOT_AT_OR_BEFORE])

    candidate_idx = idx
    skipped = 0
    while candidate_idx >= 0:
        if valid_mask is not None:
            valid = bool(valid_mask[candidate_idx])
        else:
            valid = is_valid_snapshot(snapshots_sorted[candidate_idx], depth_cap=depth_cap)
        if valid:
            break
        skipped += 1
        candidate_idx -= 1
    if candidate_idx < 0:
        return AsOfResult(
            t_ns, None, None, None, AsOfStatus.unavailable,
            [NO_VALID_SNAPSHOT_AT_OR_BEFORE, INVALID_SNAPSHOTS_SKIPPED],
            skipped_invalid=skipped,
        )

    snapshot = snapshots_sorted[candidate_idx]
    snap_time = snapshot.analysis_time_ns
    age_ns = t_ns - snap_time
    conditions: list[str] = []
    status = AsOfStatus.available

    if skipped:
        conditions.append(INVALID_SNAPSHOTS_SKIPPED)
    if age_ns == 0:
        conditions.append(EXACT_TIMESTAMP_MATCH)
    if candidate_idx > 0 and int(times[candidate_idx - 1]) == snap_time:
        conditions.append(TIMESTAMP_TIE_RESOLVED_BY_ORDINAL)

    for gap_start, _gap_end in merge_intervals(known_gaps):
        if snap_time < gap_start <= t_ns:
            conditions.append(KNOWN_GAP_SINCE_SNAPSHOT)
            status = AsOfStatus.gap_since_snapshot
            break

    if max_age_ns is not None and age_ns > max_age_ns:
        conditions.append(AGE_EXCEEDS_MAX)
        if status is AsOfStatus.available:
            status = AsOfStatus.stale

    if window_start_ns is not None and snap_time < window_start_ns:
        conditions.append(SEEDED_BEFORE_WINDOW)

    return AsOfResult(
        query_time_ns=t_ns,
        snapshot=snapshot,
        index=candidate_idx,
        age_ns=age_ns,
        status=status,
        conditions=conditions,
        skipped_invalid=skipped,
        clock_kind=snapshot.clock_kind,
    )


# ---------------------------------------------------------------------------
# Deterministic merge
# ---------------------------------------------------------------------------
class EventKind(str, Enum):
    book = "book"
    trade = "trade"


@dataclass(frozen=True, slots=True)
class MergedEvent:
    """One event of the merged replay stream.

    ``ordering_uncertain`` is ``True`` when this event ties on
    ``analysis_time_ns`` with an event of the other stream and the source
    cannot establish which came first; the emitted order is then a
    synthetic tie-break, not a causal sequence.
    """

    sequence: int
    kind: EventKind
    analysis_time_ns: int
    record: BookSnapshot | Trade
    ordering_uncertain: bool


def merge_books_and_trades(
    books: Sequence[BookSnapshot],
    trades: Sequence[Trade],
    *,
    cross_stream_keys_comparable: bool = False,
) -> Iterator[MergedEvent]:
    """Merge two individually sorted streams into one deterministic order.

    Events are ordered by the total ordering key; when a book and a trade share the
    full key the book is emitted first. Because a sequence number is not
    assumed to be globally unique across streams, any book/trade pair with
    equal ``analysis_time_ns`` is labelled ``ordering_uncertain`` unless
    ``cross_stream_keys_comparable`` is set by an adapter that guarantees a
    shared native sequence, in which case only ties on the full key are
    labelled.
    """
    keyed: list[tuple[OrderingKey, int, EventKind, BookSnapshot | Trade]] = []
    keyed.extend((b.ordering_key(), 0, EventKind.book, b) for b in books)
    keyed.extend((t.ordering_key(), 1, EventKind.trade, t) for t in trades)
    keyed.sort(key=lambda item: (item[0], item[1]))

    sequence = 0
    run_start = 0
    total = len(keyed)
    while run_start < total:
        run_time = keyed[run_start][0][0]
        run_end = run_start
        while run_end < total and keyed[run_end][0][0] == run_time:
            run_end += 1
        run = keyed[run_start:run_end]
        kinds = {item[2] for item in run}
        if len(kinds) < 2:
            uncertain = [False] * len(run)
        elif not cross_stream_keys_comparable:
            uncertain = [True] * len(run)
        else:
            book_keys = {item[0] for item in run if item[2] is EventKind.book}
            trade_keys = {item[0] for item in run if item[2] is EventKind.trade}
            shared = book_keys & trade_keys
            uncertain = [item[0] in shared for item in run]
        for (key, _rank, kind, record), flag in zip(run, uncertain):
            yield MergedEvent(sequence, kind, key[0], record, flag)
            sequence += 1
        run_start = run_end
