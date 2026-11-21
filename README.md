# tickfoundry

Pull historical order-book data for Polymarket, Kalshi and Hyperliquid into pandas, and price executions against the recorded book with the TickFoundry Lab's own code.

Status: alpha.

The code is MIT-licensed (see LICENSE). The data you download is covered by your TickFoundry plan's data licence, not by this licence.

## Install

The package is not on PyPI yet, so `pip install tickfoundry` does not install it. Install it from this repository instead. You need Python 3.11 or later, and git on your PATH (pip uses git to fetch the code).

```
pip install "git+https://github.com/tickfoundry/tickfoundry-python"
```

With the optional Polars and DuckDB support (or just one of the two):

```
pip install "tickfoundry[polars,duckdb] @ git+https://github.com/tickfoundry/tickfoundry-python"
```

To fix a version, pin a commit. Version 0.2.1 is commit `1edfcaa`:

```
pip install "git+https://github.com/tickfoundry/tickfoundry-python@1edfcaa5ff89998d9f6c70a94a5dedb2dccbed0f"
```

To upgrade to the latest version later:

```
pip install --upgrade "git+https://github.com/tickfoundry/tickfoundry-python"
```

## Quick start

```python
import tickfoundry as tf

# The public Polymarket sample day (BTC 4-hour, 2026-08-19): no key, no account.
d = tf.sample()
d.l2.head()

# With an API key (any paid plan; create one at https://tickfoundry.com/account/api).
# The key comes from TICKFOUNDRY_API_KEY, or pass it first: tf.pull(api_key, ...).
tf.search("bitcoin up or down", venue="polymarket")        # ids for pull
tf.plan_pull("KXBTC15M", "2026-11-01", "2026-11-30")       # what a pull would cost, requests nothing
l2 = tf.pull("KXBTC15M", "2026-11-12", layers="l2")        # one layer -> one DataFrame
d = tf.pull("BTC", "2026-11-01", "2026-11-03", venue="hyperliquid")
d.l1, d.l2, d.trades, d.known_gaps
```

## What `pull` does

1. Splits the range into requests the venue accepts: up to 31 days or 10 GiB on Polymarket and Kalshi, up to 92 days or 10 GiB on Hyperliquid, each one bundle. A request the server refuses as too large is split in half and retried.
2. Reuses what you already have: the local cache (`~/.cache/tickfoundry`, or `TICKFOUNDRY_CACHE`), then a ready job from your account's job list, and only then requests the rest. An identical request returns the same job and does not count as a new download.
3. Waits for the jobs (one list call per poll when many are pending), downloads with resume, extracts, and reads the parquet into pandas with a `date` column (plus `series` on Kalshi).

Days are whole UTC days. A `start` or `end` with a time of day cuts the rows after loading, on `ts_recv_ns` (or the venue time where a table has no receive time, as Kalshi trades).

Limits are your plan's: requests per minute and per day, downloads per day, and the history window. `plan_pull` shows them before you start; errors carry the server's own text and code (`HistoryLocked`, `DailyDownloadLimit`, `NotEntitled`, `RangeTooLarge`, `JobFailed`, `JobExpired`).

## Execution cost on the recorded book

`tf.LocalReplay` replays a downloaded day and prices orders against it with the TickFoundry Lab's own code, at the version this release was built from.

```python
d = tf.sample()
r = tf.LocalReplay.from_bundle(d.folders[0])           # one replay per bundle day
inst = max(r.instrument_ids(), key=lambda i: r.books(i).n)
t = "2026-08-19T21:00:00Z"

s = r.simulate(inst, t, side="buy", size_mode="quote_budget", size="1000")
s.result.vwap, s.result.relative_slippage_bps, s.result.levels_consumed

r.cost_to_move(inst, t, cents="0.02").result           # dollars to move the price 2 cents up and down
tf.round_trip(r, inst, t, size="500", hold_ms=60_000)  # buy, then sell a minute later
tf.sliced_order(r, inst, t, "2026-08-19T21:10:00Z", side="buy", total_size="2000", slices=5)
```

`LocalSimulation.fee_estimate(...)` applies a venue's published fee rule to a walk, as an estimate; the fee recorded on a real fill is always the authoritative one. `examples/sample_day.ipynb` runs all of this on the free sample day.

Every number walks the book as it was recorded. Your order never touched that book, so nothing models your own market impact, the book refilling after you, queue position or hidden liquidity; a sliced order's cost is a lower bound.

Kalshi replay and simulation run in this package on downloaded Kalshi files. The hosted Lab adds Kalshi by the end of November 2026.

## The book over time

`LocalReplay.series` gives the book as time series, computed by the Lab's own series code on the same as-of rule as `simulate`:

```python
ser = r.series(inst, start="2026-08-19T20:00:00Z", end="2026-08-20T00:00:00Z", res="1m",
               band=("cents", "0.02"), ohlc=True)        # a pyarrow Table, one row per minute
ser.to_pandas()[["mid", "spread", "microprice", "depth_imbalance", "bid_band_notional", "ask_band_notional"]]

r.candles(inst, res="1m")                                # OHLC from the trades
r.snapshots(inst, every="15m", start=..., end=...)       # the book in force on a time grid
```

`series` columns: `t_ns`, `best_bid`, `best_ask`, `mid`, `spread`, bid and ask depth (quantity and notional), `trade_count`, `trade_qty`, `vwap`, `coverage` and `flags`; with `micro=True` (the default) `microprice` and `depth_imbalance`; with `band=("cents", x)` or `("bps", x)` the depth within that band of the mid; with `ohlc=True` `open`, `high`, `low`, `close` and `trade_notional`.

## Tests

```
pip install -e ".[dev]"
pytest -q
```

The tests run against a fake API; no network or account is needed.
