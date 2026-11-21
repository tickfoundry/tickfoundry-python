"""``Client``: the data side of the SDK (pull, search, me, plan_pull, known_gaps, sample).

    import tickfoundry as tf
    c = tf.Client()                      # TICKFOUNDRY_API_KEY from the environment
    ids = c.search("bitcoin up or down", venue="polymarket")
    d = c.pull(ids.id[0], "2026-11-01", "2026-11-03")
    d.l2.head()

How a pull works (the API documentation recommends "list, reuse, then request only
what is missing"):
1. Split the range into requests the venue accepts: up to 31 days / 10 GiB on
   Polymarket and Kalshi, up to 92 days / 10 GiB on Hyperliquid.
2. Use the local cache first; then, for Polymarket, a ready job from the
   account's job list; then request the rest (the server also returns the same
   job, without counting a download, for an identical request).
3. Wait for every job with one list call per poll, download with resume,
   extract into the cache, read the parquet into pandas.
Whole UTC days are fetched; a ``start`` / ``end`` with a time of day cuts the
rows after loading.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from . import catalog, errors
from .bundle import Data, cut_to_window, read_hyperliquid, read_kalshi, read_polymarket, safe_extract
from .http import Http
from .venues import GIB, HYPERLIQUID, KALSHI, POLYMARKET, SPECS, infer_venue, normalize_venue

SAMPLE_URL = "https://tickfoundry.com/samples/polymarket-btc-4h-2026-08-19-day.zip"
POLL_SECONDS = 5.0


def default_cache_dir() -> Path:
    import os

    base = os.environ.get("TICKFOUNDRY_CACHE") or os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "tickfoundry" if not os.environ.get("TICKFOUNDRY_CACHE") else Path(base)


# -- dates ---------------------------------------------------------------------
def _parse_bound(v: Any, *, end: bool) -> tuple[dt.date, int | None]:
    """A date bound plus, when it carries a time of day, the exact ns to cut at."""
    if isinstance(v, pd.Timestamp):
        v = v.to_pydatetime()
    if isinstance(v, dt.datetime):
        ts = v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)
        ts = ts.astimezone(dt.timezone.utc)
        ns = int(ts.timestamp()) * 1_000_000_000 + ts.microsecond * 1000
        midnight = ts.hour == ts.minute == ts.second == ts.microsecond == 0
        if end and midnight:  # an exclusive midnight end: the day before is the last whole day
            return (ts - dt.timedelta(days=1)).date(), ns
        return ts.date(), (None if midnight else ns)
    if isinstance(v, dt.date):
        return v, None
    if isinstance(v, str):
        s = v.strip()
        if len(s) == 10:
            return dt.date.fromisoformat(s), None
        return _parse_bound(pd.Timestamp(s).to_pydatetime(), end=end)
    raise TypeError(f"expected a date, datetime or ISO string, got {type(v).__name__}")


def _days(a: dt.date, b: dt.date) -> list[dt.date]:
    return [a + dt.timedelta(days=i) for i in range((b - a).days + 1)]


def chunk_range(venue: str, start: dt.date, end: dt.date, avg_bytes_per_day: int | None = None,
                per_day: bool = False) -> list[tuple[dt.date, dt.date]]:
    """Split [start, end] into requests the venue accepts (kept under 80% of the byte cap)."""
    spec = SPECS[venue]
    per = 1 if per_day else spec.max_days
    if spec.max_bytes and avg_bytes_per_day:
        per = max(1, min(per, int(0.8 * spec.max_bytes // max(1, avg_bytes_per_day))))
    days = _days(start, end)
    return [(days[i], days[min(i + per, len(days)) - 1]) for i in range(0, len(days), per)]


@dataclass
class PullPlan:
    """What ``pull`` would do, without requesting anything (``Client.plan_pull``)."""

    venue: str
    series_id: str
    start: dt.date
    end: dt.date
    requests: list[tuple[dt.date, dt.date]]
    cached: int
    est_bytes: int | None
    downloads_left_today: int | None
    history_from: str | None
    warnings: list[str] = field(default_factory=list)

    @property
    def new_requests(self) -> int:
        return len(self.requests) - self.cached

    def __str__(self) -> str:
        size = f"~{self.est_bytes / 1e9:.1f} GB" if self.est_bytes else "size unknown"
        lines = [
            f"{self.venue} {self.series_id} {self.start} to {self.end}: {len(self.requests)} request(s), "
            f"{self.cached} already cached, {self.new_requests} to request, {size}",
        ]
        if self.downloads_left_today is not None:
            lines.append(f"downloads left today: {self.downloads_left_today}")
        lines += [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines)


class Client:
    """The data client. ``api_key`` defaults to ``TICKFOUNDRY_API_KEY``."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        cache_dir: str | Path | None = None,
        transport: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        poll_seconds: float = POLL_SECONDS,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.http = Http(api_key, base_url=base_url, transport=transport, sleep=sleep)
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self.sleep = sleep
        self.poll_seconds = poll_seconds
        self.progress = progress or (lambda msg: None)
        self._pm_single_day = False  # set when the server only takes one Polymarket day per request

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- account and catalog ------------------------------------------------
    def me(self) -> dict[str, Any]:
        """Plan, limits, usage today and history window (``GET /api/v1/me``)."""
        return self.http.get("/api/v1/me")

    def search(self, q: str = "", *, venue: str = POLYMARKET, kind: str | None = None, limit: int = 50,
               category: str | None = None) -> pd.DataFrame:
        return catalog.search(self.http, q, venue=normalize_venue(venue) or POLYMARKET, kind=kind, limit=limit,
                              category=category)

    def jobs(self, *, limit: int = 200, state: str | None = None) -> list[dict[str, Any]]:
        """The account's download jobs, newest first (``GET /api/v1/downloads``)."""
        params: dict[str, Any] = {"limit": limit}
        if state:
            params["state"] = state
        return list(self.http.get("/api/v1/downloads", params).get("jobs", []))

    # -- planning ----------------------------------------------------------
    def _venue_for(self, series_id: str, venue: str | None) -> str:
        v = normalize_venue(venue) or infer_venue(series_id)
        if v is None:
            raise ValueError(
                f"cannot tell the venue of {series_id!r}: pass venue='polymarket', 'kalshi' or 'hyperliquid'"
            )
        return v

    def _layers(self, venue: str, layers: str | Sequence[str]) -> list[str]:
        names = [layers] if isinstance(layers, str) else list(layers)
        spec = SPECS[venue]
        bad = [x for x in names if x not in spec.datasets]
        if bad:
            raise ValueError(f"{venue} has no layer {bad[0]!r}; choose from {', '.join(spec.datasets)}")
        return names

    def plan_pull(self, series_id: str, start: Any, end: Any | None = None, *, venue: str | None = None,
                  layers: str | Sequence[str] = ("l1", "l2", "trades")) -> PullPlan:
        """Dry run of ``pull``: requests, cache hits, estimated size and what would block. Requests nothing."""
        v = self._venue_for(series_id, venue)
        names = self._layers(v, layers)
        d0, _ = _parse_bound(start, end=False)
        d1, _ = _parse_bound(end if end is not None else start, end=True)
        if d1 < d0:
            raise ValueError("end is before start")
        avg = catalog.avg_bytes_per_day(self.http, v, series_id, [SPECS[v].datasets[n] for n in names]) if v in (KALSHI, HYPERLIQUID) else None
        reqs = chunk_range(v, d0, d1, avg, per_day=(v == POLYMARKET and self._pm_single_day))
        cached = sum(1 for a, b in reqs if self._cached_folder(self._sig(v, series_id, a, b, names)) is not None)
        warnings: list[str] = []
        me = self.me()
        limits, usage, hist = me.get("limits", {}), me.get("usage", {}), me.get("history", {})
        per_day = limits.get("downloads_per_day")
        left = None if per_day is None else max(0, int(per_day) - int(usage.get("downloads_today") or 0))
        if left is not None and len(reqs) - cached > left:
            warnings.append(
                f"{len(reqs) - cached} new requests but {left} downloads left today; the rest waits for 00:00 UTC"
            )
        hfrom = hist.get("from")
        if hfrom and not hist.get("unlimited") and str(d0) < hfrom:
            warnings.append(f"days before {hfrom} are outside the plan's history window (backfill or a yearly plan)")
        est = avg * len(_days(d0, d1)) if avg else None
        return PullPlan(v, series_id, d0, d1, reqs, cached, est, left, hfrom, warnings)

    # -- the pull ------------------------------------------------------------
    def pull(
        self,
        series_id: str,
        start: Any,
        end: Any | None = None,
        *,
        venue: str | None = None,
        layers: str | Sequence[str] = ("l1", "l2", "trades"),
        timeout: float = 3600.0,
    ) -> Data | pd.DataFrame:
        """Download (or reuse) and load one series over a range of UTC days.

        ``series_id``: a Polymarket series or event id (UUID from ``search``), a
        Kalshi series ticker, or a Hyperliquid coin. ``end`` defaults to
        ``start``; both are inclusive dates, or datetimes to cut rows at.
        ``layers`` as one name ("l2") returns that DataFrame; a list returns ``Data``.
        """
        v = self._venue_for(series_id, venue)
        names = self._layers(v, layers)
        d0, start_ns = _parse_bound(start, end=False)
        if end is None:  # one day: the day of ``start``, cut from its time of day if it has one
            d1, end_ns = d0, None
        else:
            d1, end_ns = _parse_bound(end, end=True)
        if d1 < d0:
            raise ValueError("end is before start")
        avg = catalog.avg_bytes_per_day(self.http, v, series_id, [SPECS[v].datasets[n] for n in names]) if v in (KALSHI, HYPERLIQUID) else None
        reqs = chunk_range(v, d0, d1, avg, per_day=(v == POLYMARKET and self._pm_single_day))
        folders = self._fetch_all(v, series_id, reqs, names, timeout=timeout)
        if v == POLYMARKET:
            data = read_polymarket([f for _, f in folders], names, [str(a) for (a, _), _ in folders])
        elif v == KALSHI:
            data = read_kalshi([f for _, f in folders])
        else:
            data = read_hyperliquid([f for _, f in folders])
        data = cut_to_window(data, start_ns, end_ns)
        if isinstance(layers, str):
            return data[layers]
        return data

    # Request bodies -----------------------------------------------------------
    def _body(self, venue: str, series_id: str, a: dt.date, b: dt.date, names: list[str]) -> dict[str, Any]:
        spec = SPECS[venue]
        datasets = [spec.datasets[n] for n in names]
        if venue == POLYMARKET:
            if self._pm_single_day:
                return {"unit_id": series_id, "date": str(a), "layers": datasets}
            return {"unit_id": series_id, "start": str(a), "end": str(b), "layers": datasets}
        return {"start": str(a), "end": str(b), "datasets": datasets, spec.id_field or "series": [series_id]}

    # Cache ----------------------------------------------------------------------
    @staticmethod
    def _sig(venue: str, series_id: str, a: dt.date, b: dt.date, names: Iterable[str]) -> str:
        raw = json.dumps([venue, series_id, str(a), str(b), sorted(names)])
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _index_path(self) -> Path:
        return self.cache_dir / "index.json"

    def _index(self) -> dict[str, Any]:
        try:
            return json.loads(self._index_path().read_text())
        except (OSError, ValueError):
            return {}

    def _remember(self, sig: str, entry: dict[str, Any]) -> None:
        idx = self._index()
        idx[sig] = {**idx.get(sig, {}), **entry}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._index_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(idx, indent=1, sort_keys=True))
        tmp.replace(self._index_path())

    def _cached_folder(self, sig: str) -> Path | None:
        e = self._index().get(sig)
        if e and e.get("folder") and Path(e["folder"]).is_dir():
            return Path(e["folder"])
        return None

    # Fetching ------------------------------------------------------------------
    def _fetch_all(self, venue: str, series_id: str, reqs: list[tuple[dt.date, dt.date]], names: list[str],
                   *, timeout: float) -> list[tuple[tuple[dt.date, dt.date], Path]]:
        done: dict[tuple[dt.date, dt.date], Path] = {}
        pending: dict[str, tuple[dt.date, dt.date]] = {}  # job id -> range
        known_urls: dict[str, str] = {}  # ready jobs found in the account's list: no poll needed
        queue = list(reqs)
        listed: list[dict[str, Any]] | None = None
        while queue:
            a, b = queue.pop(0)
            sig = self._sig(venue, series_id, a, b, names)
            folder = self._cached_folder(sig)
            if folder is not None:
                done[(a, b)] = folder
                continue
            job_id = self._index().get(sig, {}).get("job")
            if job_id is None and venue == POLYMARKET and a == b:  # a listed job is one series·day
                if listed is None:
                    listed = self.jobs(state="ready")
                job_id, url = _match_polymarket_job(listed, series_id, a, names)
                if job_id and url:
                    known_urls[job_id] = url
            if job_id is None:
                try:
                    resp = self.http.post(SPECS[venue].request_path, self._body(venue, series_id, a, b, names))
                except errors.TickFoundryError as e:
                    if venue == POLYMARKET and not self._pm_single_day and _single_day_only(e):
                        # An older server: one series·day per request. Re-plan this range by day.
                        self._pm_single_day = True
                        queue[:0] = [(d, d) for d in _days(a, b)]
                        self.progress("this server takes one Polymarket day per request; requesting day by day")
                        continue
                    if not isinstance(e, errors.RangeTooLarge):
                        raise
                    if a == b:
                        raise
                    mid = a + dt.timedelta(days=(b - a).days // 2)
                    queue[:0] = [(a, mid), (mid + dt.timedelta(days=1), b)]
                    self.progress(f"{a}..{b} is over the per-request cap; split in two")
                    continue
                job_id = resp["id"]
                self.progress(f"requested {venue} {series_id} {a}..{b} (job {job_id})")
            self._remember(sig, {"job": job_id, "venue": venue, "series": series_id, "start": str(a), "end": str(b)})
            pending[job_id] = (a, b)
        if pending:
            ready = self._wait(pending, timeout=timeout, known=known_urls)
            for job_id, (a, b) in pending.items():
                done[(a, b)] = self._download_extract(job_id, ready[job_id], venue, series_id, a, b, names)
        return sorted(done.items(), key=lambda kv: kv[0][0])

    def _wait(self, pending: dict[str, tuple[dt.date, dt.date]], *, timeout: float,
              known: dict[str, str] | None = None) -> dict[str, str]:
        """Poll until every job is ready. Returns job id -> file URL. One list call per round when many are pending."""
        deadline = time.monotonic() + timeout
        urls: dict[str, str] = {j: u for j, u in (known or {}).items() if j in pending}
        while True:
            open_ids = [j for j in pending if j not in urls]
            if not open_ids:
                return urls
            if len(open_ids) > 3:
                by_id = {j.get("id"): j for j in self.jobs(limit=200)}
                states = {j: by_id.get(j) for j in open_ids}
            else:
                states = {}
                for j in open_ids:
                    status, body = self.http.get_raw(f"/api/v1/downloads/{j}")
                    if status == 410:
                        raise errors.JobExpired(str(body.get("error", "expired")), status=410)
                    if status >= 400:
                        raise errors.from_response(status, body)
                    states[j] = {"id": j, "state": body.get("status"), "url": body.get("url"), "error": body.get("error")}
            for j, row in states.items():
                if row is None:
                    continue
                state = row.get("state")
                if state == "ready" and row.get("url"):
                    urls[j] = row["url"]
                elif state == "failed":
                    raise errors.JobFailed(f"job {j} failed: {row.get('error') or 'staging failed'}")
            if len(urls) < len(pending):
                if time.monotonic() > deadline:
                    raise errors.Timeout(f"{len(pending) - len(urls)} job(s) not ready after {timeout:.0f} s; "
                                         "pull again later to pick them up")
                self.sleep(self.poll_seconds)

    def _download_extract(self, job_id: str, url: str, venue: str, series_id: str, a: dt.date, b: dt.date,
                          names: list[str]) -> Path:
        sig = self._sig(venue, series_id, a, b, names)
        zpath = self.cache_dir / "zips" / f"{job_id}.zip"
        if not zpath.exists():
            self.progress(f"downloading job {job_id}")
            self.http.download(url, zpath)
        folder = safe_extract(zpath, self.cache_dir / "data" / venue / job_id)
        zpath.unlink(missing_ok=True)  # keep the extracted copy only
        self._remember(sig, {"folder": str(folder)})
        return folder

    # -- known gaps ----------------------------------------------------------
    def known_gaps(self, series_id: str, start: Any, end: Any | None = None, *, venue: str | None = None) -> pd.DataFrame:
        """Gaps and caveats for a series over a range.

        Polymarket: the bundles' ``known_gaps.csv`` (pulls l1 if not cached).
        Hyperliquid: the bundles' per-coin quality files. Kalshi: the
        exchange-wide caveats the API publishes, cut to the range.
        """
        v = self._venue_for(series_id, venue)
        d0, _ = _parse_bound(start, end=False)
        d1, _ = _parse_bound(end if end is not None else start, end=True)
        if v == KALSHI:
            rows = [c for c in catalog.kalshi_caveats(self.http)
                    if str(c.get("first_date") or "") <= str(d1) and str(c.get("last_date") or "9999") >= str(d0)]
            return pd.DataFrame(rows)
        data = self.pull(series_id, d0, d1, venue=v, layers=["l1"])
        assert isinstance(data, Data)
        return data.known_gaps


def _single_day_only(e: errors.TickFoundryError) -> bool:
    """A 400 from a server whose Polymarket route still requires ``date`` (no range support)."""
    if e.status != 400:
        return False
    err = e.details.get("error") if isinstance(e.details, dict) else None
    fields = err.get("fieldErrors") if isinstance(err, dict) else None
    return isinstance(fields, dict) and "date" in fields


def _match_polymarket_job(jobs: list[dict[str, Any]], unit_id: str, day: dt.date,
                          names: list[str]) -> tuple[str | None, str | None]:
    """A ready job of the account for this series·day carrying every asked layer: (id, file url)."""
    want = set(SPECS[POLYMARKET].datasets[n] for n in names)
    for j in jobs:
        if j.get("unit_id") == unit_id and str(j.get("date") or "")[:10] == str(day) and not j.get("date_end") \
                and j.get("state") == "ready" and want <= set(j.get("layers") or []):
            return j.get("id"), j.get("url")
    return None, None


def sample(*, cache_dir: str | Path | None = None, url: str = SAMPLE_URL, transport: Any = None) -> Data:
    """The public Polymarket sample day (BTC 4-hour, 2026-08-19) with no key and no account."""
    cache = Path(cache_dir) if cache_dir else default_cache_dir()
    folder = cache / "sample" / Path(url).stem
    if not folder.is_dir():
        http = Http(None, require_key=False, transport=transport)
        try:
            zpath = cache / "sample" / Path(url).name
            http.download(url, zpath)
            safe_extract(zpath, folder)
            zpath.unlink(missing_ok=True)
        finally:
            http.close()
    return read_polymarket([folder], ["l1", "l2", "trades"], ["2026-08-19"])


__all__ = ["Client", "Data", "PullPlan", "chunk_range", "sample", "GIB"]
