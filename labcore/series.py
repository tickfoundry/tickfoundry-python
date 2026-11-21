"""Vectorized downsampled series (the hosted ``/series`` table;
time-weighted bins) with the semantics of ``labcore.features``.

``labcore.features.aggregate_bins`` is the reference: exact Decimal and
Fraction arithmetic over labcore records. Building the three resolutions of a
whole-day replay through it costs tens of seconds (every call recomputes a
``book_state`` per snapshot and walks segments per bin), so the hosted Lab and
the SDK use this module: numpy over the sorted arrays of the event table. The rules are
the reference's, restated:

* a snapshot is a valid state when it is neither crossed (``best_bid >=
  best_ask``, which includes a locked book) nor empty (no level on either
  side); a valid one-sided book contributes its present side only;
* equal-time snapshots collapse to the last in total order; a state is
  carried forward until the next snapshot, the first known-gap start after
  it, or the window end, and is clipped at bin edges (``[start, end)``);
* a bin's book features are time-weighted means over the duration for which
  that feature was present; coverage is the valid duration over the bin
  duration, quantized half-even to six decimals;
* trades are counted inside ``[start, end)``; an empty bin with capture
  coverage at or above the threshold is zero, otherwise unknown (null).

Means are rounded to labcore's twelve decimals at the boundary. A parity test
holds this module to the reference within 1e-9 on real windows, capture gaps
and Hyperliquid included.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .features import (
    BELOW_COVERAGE_THRESHOLD,
    KNOWN_GAP_OVERLAP,
    NO_VALID_STATE,
    PARTIAL_WINDOW_BIN,
    TRADES_UNKNOWN_MISSING_COVERAGE,
    Bin,
)

#: ``FeaturePolicy.coverage_threshold`` at the six-decimal quantum.
COVERAGE_THRESHOLD_Q6 = 900_000
COVERAGE_DP = 6
VALUE_DP = 12
INT64_MAX = np.iinfo(np.int64).max

# Columns time-weighted the labcore way (present only when the book is valid
# and the feature is defined) and the four engine columns that follow
# ``_time_weighted_means`` (valid and finite).
_MEAN_COLUMNS = ("best_bid", "best_ask", "mid", "spread", "bid_depth_qty", "ask_depth_qty", "bid_depth_notional", "ask_depth_notional")
_ROUNDED_COLUMNS = ("mid", "spread", "bid_depth_notional", "ask_depth_notional")


def ladder_matrix(column: pa.ChunkedArray | pa.Array, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dense ``n x depth`` float matrix (NaN past each row's levels) and the
    per-row level count for a ``list<double>`` ladder column at ``indices``."""
    arr = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    taken = arr.take(pa.array(indices, type=pa.int64()))
    lengths = np.asarray(pc.fill_null(pc.list_value_length(taken), 0).to_numpy(zero_copy_only=False), dtype=np.int64)
    flat = np.asarray(pc.list_flatten(taken).to_numpy(zero_copy_only=False), dtype=np.float64)
    depth = int(lengths.max()) if len(lengths) else 0
    out = np.full((len(lengths), max(depth, 1)), np.nan)
    if depth:
        mask = np.arange(depth)[None, :] < lengths[:, None]
        out[:, :depth][mask] = flat
    return out, lengths


def _side_stats(px: np.ndarray, sz: np.ndarray, count: np.ndarray, *, best: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(has_levels, best price, total quantity, price*quantity sum) per row."""
    has = count > 0
    best_px = np.full(len(count), np.nan)
    qty = np.full(len(count), np.nan)
    notional = np.full(len(count), np.nan)
    if has.any():
        reducer = np.nanmax if best == "max" else np.nanmin
        best_px[has] = reducer(px[has], axis=1)
        qty[has] = np.nansum(sz[has], axis=1)
        notional[has] = np.nansum(px[has] * sz[has], axis=1)
    return has, best_px, qty, notional


#: Optional per-snapshot columns (``book_features(..., micro=True, band=...)``), time-weighted like
#: the mean columns and appended after the standard ones. The hosted ``/series`` asks for none.
MICRO_COLUMNS = ("microprice", "depth_imbalance")
BAND_COLUMNS = ("bid_band_qty", "ask_band_qty", "bid_band_notional", "ask_band_notional")
OHLC_COLUMNS = ("open", "high", "low", "close", "trade_notional")


def book_features(table: pa.Table, book_rows: np.ndarray, *, multiplier: float, micro: bool = False,
                  band: tuple[str, float] | None = None, imbalance_levels: int | None = None) -> dict[str, np.ndarray]:
    """Per-snapshot values in book-row order: labcore's ``book_state`` for the
    mean columns plus the engine's raw best/size columns, and ``valid``.

    ``micro`` adds labcore's ``microprice`` (top-level size-weighted, valid
    two-sided books) and ``depth_imbalance`` (``(bid - ask) / (bid + ask)`` over
    the top ``imbalance_levels`` of each side, every visible level by default,
    as the Lab's feature policy does; valid books). ``band=("cents", D)`` or ``("bps", N)`` adds the
    quantity and notional resting within ``mid +- D`` (or ``mid x N / 10000``)
    on each side, on valid two-sided books."""
    n = len(book_rows)
    if n == 0:
        empty = np.zeros(0)
        return {"t": np.zeros(0, dtype=np.int64), "valid": np.zeros(0, dtype=bool), **{c: empty for c in _MEAN_COLUMNS}}
    t = np.asarray(table.column("t_ns").take(pa.array(book_rows, type=pa.int64())).to_numpy(zero_copy_only=False), dtype=np.int64)
    bpx, bcnt = ladder_matrix(table.column("bid_px"), book_rows)
    bsz, _ = ladder_matrix(table.column("bid_sz"), book_rows)
    apx, acnt = ladder_matrix(table.column("ask_px"), book_rows)
    asz, _ = ladder_matrix(table.column("ask_sz"), book_rows)
    has_b, bb, bq, bn = _side_stats(bpx, bsz, bcnt, best="max")
    has_a, ba, aq, an = _side_stats(apx, asz, acnt, best="min")
    both = has_b & has_a
    with np.errstate(invalid="ignore"):
        crossed = both & (bb >= ba)
    empty = ~has_b & ~has_a
    valid = ~(crossed | empty)
    nan = np.nan
    mid = np.where(valid & both, (bb + ba) / 2.0, nan)
    spread = np.where(valid & both, ba - bb, nan)
    extra: dict[str, np.ndarray] = {}
    if micro:
        rows = np.arange(n)
        top_b = np.zeros(n)
        top_a = np.zeros(n)
        if has_b.any():
            ib = np.nanargmax(np.where(np.isfinite(bpx), bpx, -np.inf), axis=1)
            top_b = np.where(has_b, np.nan_to_num(bsz[rows, ib]), 0.0)
        if has_a.any():
            ia = np.nanargmin(np.where(np.isfinite(apx), apx, np.inf), axis=1)
            top_a = np.where(has_a, np.nan_to_num(asz[rows, ia]), 0.0)
        den = top_b + top_a
        if imbalance_levels is None:
            ib_q = np.where(has_b, np.nan_to_num(bq), 0.0)
            ia_q = np.where(has_a, np.nan_to_num(aq), 0.0)
        else:
            k = int(imbalance_levels)
            ib_q = np.where(has_b, np.nansum(bsz[:, :k], axis=1), 0.0)
            ia_q = np.where(has_a, np.nansum(asz[:, :k], axis=1), 0.0)
        tot = ib_q + ia_q
        with np.errstate(invalid="ignore", divide="ignore"):
            extra["microprice"] = np.where(valid & both & (den > 0), (ba * top_b + bb * top_a) / np.where(den > 0, den, 1.0), nan)
            extra["depth_imbalance"] = np.where(valid & (tot > 0), (ib_q - ia_q) / np.where(tot > 0, tot, 1.0), nan)
    if band is not None:
        kind, amount = band
        if kind not in ("cents", "bps") or not amount > 0:
            raise ValueError("band must be ('cents', D) or ('bps', N) with a positive amount")
        width = np.full(n, float(amount)) if kind == "cents" else mid * float(amount) / 10_000.0
        ok = valid & both
        with np.errstate(invalid="ignore"):
            b_in = np.isfinite(bpx) & (bpx >= (mid - width - 1e-12)[:, None])
            a_in = np.isfinite(apx) & (apx <= (mid + width + 1e-12)[:, None])
        band_bq = np.nansum(np.where(b_in, bsz, 0.0), axis=1)
        band_aq = np.nansum(np.where(a_in, asz, 0.0), axis=1)
        extra["bid_band_qty"] = np.where(ok, band_bq, nan)
        extra["ask_band_qty"] = np.where(ok, band_aq, nan)
        extra["bid_band_notional"] = np.where(ok, np.nansum(np.where(b_in, bpx * bsz, 0.0), axis=1) * multiplier, nan)
        extra["ask_band_notional"] = np.where(ok, np.nansum(np.where(a_in, apx * asz, 0.0), axis=1) * multiplier, nan)
    return {
        **extra,
        "t": t,
        "valid": valid,
        # Engine columns: the raw top level and the whole-ladder quantity
        # (``s.bids[0].price`` and ``sum(l.quantity for l in s.bids)``).
        "best_bid": np.where(has_b, bpx[:, 0], nan),
        "best_ask": np.where(has_a, apx[:, 0], nan),
        "bid_depth_qty": np.where(has_b, bq, nan),
        "ask_depth_qty": np.where(has_a, aq, nan),
        # labcore book_state columns (present only on a valid book).
        "mid": mid,
        "spread": spread,
        "bid_depth_notional": np.where(valid & has_b, bn * multiplier, nan),
        "ask_depth_notional": np.where(valid & has_a, an * multiplier, nan),
    }


def segments(times: np.ndarray, gaps: list[tuple[int, int]], window_end_ns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``labcore.features._build_segments`` over sorted times: returns
    ``(state_index, start, end)`` of the carried-forward states (equal times
    collapse to the last one; ends at the next state, the first gap start
    after the start, or the window end; empty segments dropped)."""
    if len(times) == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    last_of_time = np.flatnonzero(np.diff(times, append=INT64_MAX) != 0)
    start = times[last_of_time]
    end = np.append(start[1:], INT64_MAX)
    gap_starts = np.array(sorted(int(g[0]) for g in gaps), dtype=np.int64)
    if len(gap_starts):
        nxt = np.searchsorted(gap_starts, start, side="right")
        has = nxt < len(gap_starts)
        cut = np.where(has, gap_starts[np.minimum(nxt, len(gap_starts) - 1)], INT64_MAX)
        end = np.minimum(end, cut)
    end = np.minimum(end, np.int64(window_end_ns))
    keep = end > start
    return last_of_time[keep], start[keep], end[keep]


def _round_half_even_div(numerator: np.ndarray, denominator: int) -> np.ndarray:
    """``round(Fraction(numerator, denominator))`` (half to even) on int64 arrays."""
    q, r = np.divmod(numerator, denominator)
    twice = 2 * r
    up = (twice > denominator) | ((twice == denominator) & (q % 2 == 1))
    return q + up.astype(np.int64)


def aggregate(
    features: dict[str, np.ndarray],
    trades_t: np.ndarray,
    trades_price: np.ndarray,
    trades_size: np.ndarray,
    bins: list[Bin],
    *,
    multiplier: float,
    window_start_ns: int,
    window_end_ns: int,
    gaps: list[tuple[int, int]],
    extra_columns: tuple[str, ...] = (),
    ohlc: bool = False,
) -> pa.Table:
    """One resolution: the wire table of the hosted ``/series``, plus any
    ``extra_columns`` of ``features`` (time-weighted the same way) and, with
    ``ohlc``, trade candles (first / highest / lowest / last trade price in the
    bin and its traded notional), appended after the standard columns."""
    n_bins = len(bins)
    if n_bins == 0:
        return _empty_table()
    width = bins[0].duration_ns
    bin0 = bins[0].start_ns
    bin_start = np.array([b.start_ns for b in bins], dtype=np.int64)
    bin_end = bin_start + width
    lo = np.maximum(bin_start, window_start_ns)
    hi = np.minimum(bin_end, window_end_ns)
    in_window = np.maximum(hi - lo, 0)

    # Known-gap overlap per bin (a handful of gaps at most).
    gap_overlap = np.zeros(n_bins, dtype=np.int64)
    for gs, ge in gaps:
        gap_overlap += np.maximum(np.minimum(hi, ge) - np.maximum(lo, gs), 0)
    gap_overlap = np.where(in_window > 0, gap_overlap, 0)
    capture_ns = in_window - gap_overlap
    capture_q6 = _round_half_even_div(capture_ns * 10**COVERAGE_DP, width)

    # Pieces: the window split at every bin edge and segment boundary; each
    # piece lies in exactly one bin and at most one carried-forward state.
    state_idx, seg_start, seg_end = segments(features["t"], gaps, window_end_ns)
    points = np.concatenate([bin_start, bin_end[-1:], seg_start, seg_end])
    points = np.unique(np.clip(points, window_start_ns, window_end_ns))
    a = points[:-1]
    b = points[1:]
    dt = (b - a).astype(np.float64)
    piece_bin = (a - bin0) // width
    # A window with no book snapshot at all (a start/end that misses every
    # book row, or a trades-only window) has no carried-forward state: every
    # piece is inactive, every bin has zero valid duration and null means.
    # The case is explicit because the segment arrays are empty then.
    has_states = len(seg_start) > 0
    if has_states:
        k = np.searchsorted(seg_start, a, side="right") - 1
        active = (k >= 0) & (seg_end[np.maximum(k, 0)] > a)
        state = np.where(active, state_idx[np.maximum(k, 0)], 0)
        valid_piece = active & features["valid"][state]
    else:
        active = np.zeros(len(a), dtype=bool)
        state = np.zeros(len(a), dtype=np.int64)
        valid_piece = active

    valid_ns = np.bincount(piece_bin[valid_piece], weights=dt[valid_piece], minlength=n_bins).astype(np.int64)
    coverage_q6 = _round_half_even_div(valid_ns * 10**COVERAGE_DP, width)

    means: dict[str, np.ndarray] = {}
    for name in (*_MEAN_COLUMNS, *extra_columns):
        if not has_states:
            means[name] = np.full(n_bins, np.nan)
            continue
        values = features[name][state]
        present = valid_piece & np.isfinite(values)
        total = np.bincount(piece_bin[present], weights=values[present] * dt[present], minlength=n_bins)
        duration = np.bincount(piece_bin[present], weights=dt[present], minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(duration > 0, total / np.where(duration > 0, duration, 1.0), np.nan)
        if name in _ROUNDED_COLUMNS or name in extra_columns:
            mean = np.round(mean, VALUE_DP)
        means[name] = mean

    # Trades inside [lo, hi) with a positive price and quantity.
    ok = np.isfinite(trades_price) & np.isfinite(trades_size) & (trades_price > 0) & (trades_size > 0)
    ok &= (trades_t >= window_start_ns) & (trades_t < window_end_ns)
    tb = (trades_t[ok] - bin0) // width
    inside = (tb >= 0) & (tb < n_bins)
    tb = tb[inside]
    tp = trades_price[ok][inside]
    ts = trades_size[ok][inside]
    count = np.bincount(tb, minlength=n_bins).astype(np.int64)
    qty = np.bincount(tb, weights=ts, minlength=n_bins)
    notional = np.bincount(tb, weights=tp * ts * multiplier, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = np.where(count > 0, np.round(notional / (qty * multiplier), VALUE_DP), np.nan)
    capture_ok = capture_q6 >= COVERAGE_THRESHOLD_Q6
    trades_known = (count > 0) | capture_ok
    trade_count = pa.array(np.where(trades_known, count, 0).astype(np.int32), mask=~trades_known)
    trade_qty = pa.array(np.where(trades_known, qty, 0.0), mask=~trades_known)

    coverage_ok = coverage_q6 >= COVERAGE_THRESHOLD_Q6
    partial = in_window < width
    gap_flag = gap_overlap > 0
    no_state = valid_ns == 0
    trades_unknown = ~trades_known
    flags = [
        ",".join(
            f
            for f, on in (
                (PARTIAL_WINDOW_BIN, partial[i]),
                (KNOWN_GAP_OVERLAP, gap_flag[i]),
                (BELOW_COVERAGE_THRESHOLD, not coverage_ok[i]),
                (NO_VALID_STATE, no_state[i]),
                (TRADES_UNKNOWN_MISSING_COVERAGE, trades_unknown[i]),
            )
            if on
        )
        for i in range(n_bins)
    ]

    def nullable(values: np.ndarray) -> pa.Array:
        return pa.array(values, type=pa.float64(), mask=~np.isfinite(values))

    appended: dict[str, pa.Array] = {name: nullable(means[name]) for name in extra_columns}
    if ohlc:
        o = np.full(n_bins, np.nan)
        h = np.full(n_bins, np.nan)
        lo_ = np.full(n_bins, np.nan)
        c = np.full(n_bins, np.nan)
        if len(tb):
            tt = trades_t[ok][inside]
            order = np.lexsort((np.arange(len(tb)), tt))       # time order, file order inside a tie
            tb_s, tp_s = tb[order], tp[order]
            first = np.unique(tb_s, return_index=True)[1]
            last = len(tb_s) - 1 - np.unique(tb_s[::-1], return_index=True)[1]
            bins_hit = tb_s[first]
            o[bins_hit] = tp_s[first]
            c[bins_hit] = tp_s[last]
            np.fmax.at(h, tb_s, tp_s)
            np.fmin.at(lo_, tb_s, tp_s)
        known_notional = np.where(trades_known, notional, np.nan)
        appended |= {"open": nullable(o), "high": nullable(h), "low": nullable(lo_), "close": nullable(c),
                     "trade_notional": nullable(known_notional)}

    return pa.table(
        {
            "t_ns": pa.array(bin_start, type=pa.int64()),
            "best_bid": nullable(means["best_bid"]),
            "best_ask": nullable(means["best_ask"]),
            "mid": nullable(means["mid"]),
            "spread": nullable(means["spread"]),
            "bid_depth_qty": nullable(means["bid_depth_qty"]),
            "ask_depth_qty": nullable(means["ask_depth_qty"]),
            "bid_depth_notional": nullable(means["bid_depth_notional"]),
            "ask_depth_notional": nullable(means["ask_depth_notional"]),
            "trade_count": trade_count,
            "trade_qty": trade_qty,
            "vwap": nullable(vwap),
            "coverage": pa.array((coverage_q6 / 10.0**COVERAGE_DP).astype(np.float32), type=pa.float32()),
            "flags": pa.array(flags, type=pa.string()),
            **appended,
        }
    )


def _empty_table() -> pa.Table:
    return pa.table(
        {
            "t_ns": pa.array([], type=pa.int64()),
            **{c: pa.array([], type=pa.float64()) for c in _MEAN_COLUMNS},
            "trade_count": pa.array([], type=pa.int32()),
            "trade_qty": pa.array([], type=pa.float64()),
            "vwap": pa.array([], type=pa.float64()),
            "coverage": pa.array([], type=pa.float32()),
            "flags": pa.array([], type=pa.string()),
        }
    )


def trade_arrays(table: pa.Table, trade_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = pa.array(trade_rows, type=pa.int64())
    t = np.asarray(table.column("t_ns").take(idx).to_numpy(zero_copy_only=False), dtype=np.int64)
    price = np.asarray(pc.fill_null(table.column("trade_price").take(idx), np.nan).to_numpy(zero_copy_only=False), dtype=np.float64)
    size = np.asarray(pc.fill_null(table.column("trade_size").take(idx), np.nan).to_numpy(zero_copy_only=False), dtype=np.float64)
    return t, price, size


def build_series_fast(table: pa.Table, *, multiplier: Any, depth: int, start_ns: int, end_ns: int, gaps: list[tuple[int, int]], resolutions: dict[str, int], max_bins: int = 200_000,
                      micro: bool = False, band: tuple[str, float] | None = None, ohlc: bool = False,
                      imbalance_levels: int | None = None) -> dict[str, pa.Table]:
    """All resolutions from an event index table (book rows in venue order,
    trade rows anywhere), mirroring the labcore reference. ``micro``, ``band``
    and ``ohlc`` add the optional columns (see ``book_features`` / ``aggregate``)."""
    from .features import make_bins

    kinds = table.column("kind")
    is_book = np.asarray(pc.equal(kinds.cast(pa.string()), "book").to_numpy(zero_copy_only=False), dtype=bool)
    book_rows = np.flatnonzero(is_book)
    trade_rows = np.flatnonzero(~is_book)
    mult = float(multiplier)
    features = book_features(table, book_rows, multiplier=mult, micro=micro, band=band, imbalance_levels=imbalance_levels)
    extra = (MICRO_COLUMNS if micro else ()) + (BAND_COLUMNS if band is not None else ())
    # The as-of view is time-sorted; a stable sort keeps the venue order
    # inside equal timestamps, which is labcore's total order.
    order = np.argsort(features["t"], kind="stable")
    features = {name: values[order] for name, values in features.items()}
    trades_t, trades_price, trades_size = trade_arrays(table, trade_rows)
    out: dict[str, pa.Table] = {}
    for res, width in resolutions.items():
        bins = make_bins(start_ns, end_ns, width)
        if len(bins) > max_bins:
            continue
        out[res] = aggregate(features, trades_t, trades_price, trades_size, bins, multiplier=mult, window_start_ns=start_ns, window_end_ns=end_ns, gaps=gaps,
                             extra_columns=extra, ohlc=ohlc)
    return out


__all__ = ["build_series_fast", "book_features", "segments", "aggregate", "ladder_matrix", "MICRO_COLUMNS", "BAND_COLUMNS", "OHLC_COLUMNS"]
