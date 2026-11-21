"""tickfoundry.data against a fake API (httpx.MockTransport): no network, no account."""
from __future__ import annotations

import datetime as dt
import io
import json
import re
import zipfile
from pathlib import Path

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import tickfoundry as tf
from tickfoundry.data import errors
from tickfoundry.data.bundle import safe_extract
from tickfoundry.data.client import Client, chunk_range

UNIT = "0b1c2d3e-4f50-6172-8394-a5b6c7d8e9f0"
NS_DAY = 86_400 * 1_000_000_000


def _pq(table: pa.Table) -> bytes:
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def _day_ns(day: str, hour: int = 0) -> int:
    d = dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp()) * 1_000_000_000 + hour * 3600 * 1_000_000_000


def pm_zip(day: str) -> bytes:
    base = f"polymarket-btc-{day}-l2"
    t = pa.table({"ts_recv_ns": [_day_ns(day, 1), _day_ns(day, 13)], "msg_seq": [1, 2], "token_id": ["t", "t"],
                  "bid_px_1": [0.4, 0.41], "ask_px_1": [0.42, 0.43]})
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr(f"{base}/README.md", "readme")
        z.writestr(f"{base}/l1.parquet", _pq(t))
        z.writestr(f"{base}/l2.parquet", _pq(t))
        z.writestr(f"{base}/trades.parquet", _pq(pa.table({"ts_recv_ns": [_day_ns(day, 2)], "price": [0.42]})))
        z.writestr(f"{base}/reference/markets.csv", "condition_id,yes_token_id\nc1,t\n")
        z.writestr(f"{base}/known_gaps.csv", "start,end,reason\n01:00,01:05,reconnect\n")
    return out.getvalue()


def pm_range_zip(days: list[str], slug: str = "btc-4h") -> bytes:
    """The account page's range bundle: <base>/<slug>/date=D/ holding each day's files."""
    base = f"polymarket-{slug}-{days[0]}-to-{days[-1]}-l2"
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr(f"{base}/README.md", "bundle readme")
        for day in days:
            t = pa.table({"ts_recv_ns": [_day_ns(day, 1), _day_ns(day, 13)], "msg_seq": [1, 2], "token_id": ["t", "t"],
                          "bid_px_1": [0.4, 0.41], "ask_px_1": [0.42, 0.43]})
            d = f"{base}/{slug}/date={day}"
            z.writestr(f"{d}/README.md", "day readme")
            z.writestr(f"{d}/l1.parquet", _pq(t))
            z.writestr(f"{d}/l2.parquet", _pq(t))
            z.writestr(f"{d}/trades.parquet", _pq(pa.table({"ts_recv_ns": [_day_ns(day, 2)], "price": [0.42]})))
            # a market added on the last day, and a renamed question for c1: the latest day must win
            q = "renamed" if day == days[-1] else "original"
            extra = f"c{len(days)},t2,new\n" if day == days[-1] else ""
            z.writestr(f"{d}/reference/markets.csv", f"condition_id,yes_token_id,question\nc1,t,{q}\n{extra}")
            z.writestr(f"{d}/known_gaps.csv", "dataset,start_utc,end_utc,severity,attribution,note\nl2,2026-11-02T01:00,2026-11-02T01:05,minor,capture,reconnect\n")
    return out.getvalue()


def kalshi_zip(days: list[str], series: str = "KXTEST", clock_from: str = "9999-12-31") -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for d in days:
            cols = {"ts_src_ms": [_day_ns(d, 3) // 1_000_000], "ts_recv_ns": [_day_ns(d, 3)], "market_ticker": [f"{series}-A"],
                    "best_bid": [0.5], "msg_seq": [7]}
            if d >= clock_from:
                cols["venue_clock"] = [123]
            z.writestr(f"kalshi-{series}-x/kalshi_l1/date={d}/{series}/l1.parquet", _pq(pa.table(cols)))
            z.writestr(f"kalshi-{series}-x/kalshi_trades/date={d}/{series}/trades.parquet",
                       _pq(pa.table({"ts_src_ms": [_day_ns(d, 4) // 1_000_000], "ts_recv_ns": pa.array([None], pa.int64()),
                                     "price": [0.5]})))
    return out.getvalue()


class FakeApi:
    """A small stand-in for /api/v1 with programmable job states and files."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.jobs: dict[str, dict] = {}
        self.files: dict[str, bytes] = {}
        self.polls_until_ready = 1
        self.listed_ready: list[dict] = []
        self.too_large_over_days: int | None = None
        self.next_id = 0
        self.me = {"limits": {"downloads_per_day": 50}, "usage": {"downloads_today": 48},
                   "history": {"from": "2026-11-01", "unlimited": False}}
        self.kalshi_series = [{"series": "KXTEST", "title": "Test", "avg_bytes_per_day": {"kalshi_l1": 1000, "kalshi_l2": 9000}}]
        self.fail_first_429 = False
        self.pm_single_day_only = False  # an older server without Polymarket ranges

    def _new_job(self, payload: bytes, **meta) -> dict:
        self.next_id += 1
        jid = f"job{self.next_id}"
        self.jobs[jid] = {"id": jid, "state": "staging", "polls": 0, **meta}
        self.files[jid] = payload
        return self.jobs[jid]

    def handler(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        self.calls.append((method, path))
        if self.fail_first_429:
            self.fail_first_429 = False
            return httpx.Response(429, headers={"Retry-After": "1"}, json={"error": "slow down"})
        if path == "/api/v1/me":
            return httpx.Response(200, json=self.me)
        if path == "/api/v1/kalshi" and method == "GET":
            return httpx.Response(200, json={"series": self.kalshi_series, "known_caveats": [
                {"first_date": "2026-11-08", "last_date": "2026-11-08", "title": "hole", "note": "n"}]})
        if path == "/api/v1/catalog":
            return httpx.Response(200, json={"units": [{"id": UNIT, "title": "BTC 4h", "kind": "series", "first_date": "2026-01-01"}]})
        if path == "/api/v1/downloads" and method == "GET":
            rows = list(self.listed_ready)
            for j in self.jobs.values():
                ready = j["polls"] >= self.polls_until_ready
                j["polls"] += 1
                rows.append({"id": j["id"], "state": "ready" if ready else "staging",
                             "url": f"https://x.test/api/v1/downloads/{j['id']}/file" if ready else None})
            return httpx.Response(200, json={"jobs": rows})
        if path == "/api/v1/downloads" and method == "POST":
            body = json.loads(request.content)
            if "start" in body:
                if self.pm_single_day_only:
                    return httpx.Response(400, json={"error": {"formErrors": [], "fieldErrors": {"date": ["Required"]}}})
                a, b = dt.date.fromisoformat(body["start"]), dt.date.fromisoformat(body["end"])
                days = [str(a + dt.timedelta(days=i)) for i in range((b - a).days + 1)]
                j = self._new_job(pm_range_zip(days))
                return httpx.Response(200, json={"status": "staging", "id": j["id"], "poll": f"/api/v1/downloads/{j['id']}"})
            j = self._new_job(pm_zip(body["date"]), unit_id=body["unit_id"], date=body["date"])
            return httpx.Response(200, json={"status": "staging", "id": j["id"], "poll": f"/api/v1/downloads/{j['id']}"})
        if path == "/api/v1/kalshi/downloads":
            body = json.loads(request.content)
            a, b = dt.date.fromisoformat(body["start"]), dt.date.fromisoformat(body["end"])
            n = (b - a).days + 1
            if self.too_large_over_days and n > self.too_large_over_days:
                return httpx.Response(400, json={"error": "this selection is 12 GB; one pull is capped at 10 GB"})
            days = [str(a + dt.timedelta(days=i)) for i in range(n)]
            j = self._new_job(kalshi_zip(days, clock_from="2026-10-29"))
            return httpx.Response(200, json={"status": "staging", "id": j["id"]})
        m = re.match(r"^/api/v1/downloads/(job\d+)$", path)
        if m:
            j = self.jobs[m.group(1)]
            ready = j["polls"] >= self.polls_until_ready
            j["polls"] += 1
            if ready:
                return httpx.Response(200, json={"status": "ready", "id": j["id"], "url": f"https://x.test/api/v1/downloads/{j['id']}/file"})
            return httpx.Response(200, json={"status": "staging", "id": j["id"], "retry_after": 5})
        m = re.match(r"^/api/v1/downloads/(job\d+|old\d+)/file$", path)
        if m:
            data = self.files.get(m.group(1)) or pm_zip("2026-11-02")
            return httpx.Response(200, content=data)
        return httpx.Response(404, json={"error": "not found"})


@pytest.fixture
def api():
    return FakeApi()


@pytest.fixture
def client(api, tmp_path):
    return Client("tf_live_x", base_url="https://x.test", cache_dir=tmp_path / "cache",
                  transport=httpx.MockTransport(api.handler), sleep=lambda s: None)


def test_chunking():
    d = dt.date(2026, 11, 1)
    assert len(chunk_range("polymarket", d, d + dt.timedelta(days=29))) == 1
    assert len(chunk_range("polymarket", d, d + dt.timedelta(days=29), per_day=True)) == 30
    k = chunk_range("kalshi", d, d + dt.timedelta(days=39))
    assert [(a.day, b.day) for a, b in k] == [(1, 1 + 30), (2, 10)] or (len(k) == 2 and (k[0][1] - k[0][0]).days == 30)
    # 1 GiB a day: 8 days fit under 80% of 10 GiB
    big = chunk_range("kalshi", d, d + dt.timedelta(days=30), avg_bytes_per_day=1024**3)
    assert (big[0][1] - big[0][0]).days + 1 == 8
    assert len(chunk_range("hyperliquid", d, d + dt.timedelta(days=364))) == 4


def test_polymarket_range_is_one_request_one_bundle(api, client):
    d = client.pull(UNIT, "2026-11-02", "2026-11-04")
    posts = [c for c in api.calls if c == ("POST", "/api/v1/downloads")]
    assert len(posts) == 1
    assert sorted(d.l2["date"].unique()) == ["2026-11-02", "2026-11-03", "2026-11-04"]
    assert set(d.l2["slug"]) == {"btc-4h"} and len(d.l2) == 6 and len(d.known_gaps) == 3
    m = d.reference["markets"]
    assert len(m) == 2 and m.set_index("condition_id").loc["c1", "question"] == "renamed"
    assert len(d.folders) == 3 and all(f.name.startswith("date=") for f in d.folders)
    n = len(api.calls)
    assert len(client.pull(UNIT, "2026-11-02", "2026-11-04").l1) == 6 and len(api.calls) == n  # cached


def test_polymarket_older_server_falls_back_to_days_and_reuses_a_listed_job(api, client):
    api.pm_single_day_only = True
    api.listed_ready = [{"id": "old1", "unit_id": UNIT, "date": "2026-11-02", "date_end": None, "state": "ready",
                         "layers": ["l1", "l2", "trades"], "url": "https://x.test/api/v1/downloads/old1/file"}]
    api.files["old1"] = pm_zip("2026-11-02")
    d = client.pull(UNIT, "2026-11-02", "2026-11-04")
    posts = [c for c in api.calls if c == ("POST", "/api/v1/downloads")]
    assert len(posts) == 3  # the refused range, then 11-03 and 11-04; 11-02 reused from the job list
    assert sorted(d.l2["date"].unique()) == ["2026-11-02", "2026-11-03", "2026-11-04"]
    assert len(d.l2) == 6 and "markets" in d.reference and len(d.known_gaps) == 3
    n = len(api.calls)
    again = client.pull(UNIT, "2026-11-02", "2026-11-04")
    assert len(api.calls) == n  # fully served from the cache
    assert len(again.l2) == 6


def test_single_layer_returns_a_dataframe_and_time_cut(api, client):
    df = client.pull(UNIT, dt.datetime(2026, 11, 3, 12, tzinfo=dt.timezone.utc), "2026-11-03T23:59:59Z", layers="l2")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1 and df["ts_recv_ns"].iloc[0] == _day_ns("2026-11-03", 13)


def test_kalshi_range_is_split_and_venue_clock_survives(api, client):
    d = client.pull("KXTEST", "2026-10-20", "2026-11-25", layers=["l1", "trades"])
    posts = [c for c in api.calls if c == ("POST", "/api/v1/kalshi/downloads")]
    assert len(posts) == 2  # 37 days -> 31 + 6
    assert len(d.l1) == 37 and "venue_clock" in d.l1.columns
    assert d.l1["venue_clock"].isna().sum() == 9  # 10-20 .. 10-28 have no clock
    assert set(d.l1["series"]) == {"KXTEST"}
    assert len(d.trades) == 37  # trades have no receive time; kept by the venue time


def test_server_cap_splits_the_request(api, client):
    api.too_large_over_days = 5
    d = client.pull("KXTEST", "2026-11-01", "2026-11-10", layers="l1")
    assert len(d) == 10
    posts = [c for c in api.calls if c == ("POST", "/api/v1/kalshi/downloads")]
    assert len(posts) >= 3


def test_retry_after_429_then_success(api, client):
    api.fail_first_429 = True
    assert client.me()["limits"]["downloads_per_day"] == 50


def test_daily_download_limit_is_raised_not_retried(tmp_path):
    calls = []

    def handler(req):
        calls.append(req)
        if req.method == "GET":
            return httpx.Response(200, json={"jobs": []})
        return httpx.Response(429, headers={"Retry-After": "40000"}, json={"error": "daily limit", "code": "daily_download_limit"})

    c = Client("tf_live_x", base_url="https://x.test", cache_dir=tmp_path, transport=httpx.MockTransport(handler), sleep=lambda s: None)
    with pytest.raises(errors.DailyDownloadLimit):
        c.pull(UNIT, "2026-11-02")
    assert len([r for r in calls if r.method == "POST"]) == 1


def test_error_mapping():
    assert isinstance(errors.from_response(402, {"error": "Kalshi is its own subscription"}), errors.NotEntitled)
    assert isinstance(errors.from_response(403, {"error": "x", "code": "history_locked"}), errors.HistoryLocked)
    assert isinstance(errors.from_response(401, {"error": "invalid API key"}), errors.AuthError)
    assert isinstance(errors.from_response(400, {"error": "a Kalshi pull covers at most 31 days"}), errors.RangeTooLarge)
    assert isinstance(errors.from_response(410, {"error": "expired"}), errors.JobExpired)


def test_failed_job_raises(api, client):
    def handler(req):
        if req.method == "POST":
            return httpx.Response(200, json={"status": "staging", "id": "job9"})
        if req.url.path == "/api/v1/downloads/job9":
            return httpx.Response(200, json={"status": "failed", "id": "job9", "error": "no data"})
        return httpx.Response(200, json={"jobs": []})

    client.http._client._transport = httpx.MockTransport(handler)
    with pytest.raises(errors.JobFailed):
        client.pull(UNIT, "2026-11-02")


def test_download_resumes_with_range_and_waits_out_rebuild(tmp_path):
    payload = b"0123456789" * 1000
    seen = []

    def handler(req):
        seen.append(req.headers.get("range"))
        if len(seen) == 1:
            return httpx.Response(202, headers={"Retry-After": "1"}, json={"status": "rebuilding"})
        rng = req.headers.get("range")
        if rng:
            start = int(rng.split("=")[1].rstrip("-"))
            return httpx.Response(206, content=payload[start:])
        return httpx.Response(200, content=payload)

    c = Client("tf_live_x", base_url="https://x.test", cache_dir=tmp_path, transport=httpx.MockTransport(handler), sleep=lambda s: None)
    dest = tmp_path / "f.zip"
    part = tmp_path / "f.zip.part"
    part.write_bytes(payload[:4000])
    c.http.download("https://x.test/api/v1/downloads/j/file", dest)
    assert dest.read_bytes() == payload
    assert seen[-1] == "bytes=4000-"


def test_zip_slip_is_refused(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../../escape.txt", "x")
    with pytest.raises(errors.TickFoundryError):
        safe_extract(z, tmp_path / "out")
    assert not (tmp_path.parent / "escape.txt").exists()


def test_plan_pull_warns_on_cap_and_history(api, client):
    api.me["usage"]["downloads_today"] = 50  # none left today
    p = client.plan_pull(UNIT, "2026-10-28", "2026-11-03")
    assert len(p.requests) == 1 and p.cached == 0
    assert any("downloads left today" in w for w in p.warnings)
    assert any("history window" in w for w in p.warnings)
    assert "1 request(s)" in str(p)


def test_kalshi_known_gaps_are_the_published_caveats(api, client):
    g = client.known_gaps("KXTEST", "2026-11-01", "2026-11-10")
    assert list(g["title"]) == ["hole"]
    assert client.known_gaps("KXTEST", "2026-11-09", "2026-11-10").empty


def test_venue_needed_when_ambiguous(client):
    with pytest.raises(ValueError, match="venue"):
        client.pull("BTC", "2026-11-01")


def test_top_level_pull_takes_the_key_first(monkeypatch, api, tmp_path):
    made = {}

    class Stub:
        def __init__(self, key):
            made["key"] = key

        def pull(self, *a, **k):
            return ("pulled", a, k)

    monkeypatch.setattr(tf, "Client", Stub)
    tf._clients.clear()
    out = tf.pull("tf_live_abc", "KXTEST", "2026-11-01", "2026-11-02")
    assert made["key"] == "tf_live_abc" and out[1] == ("KXTEST", "2026-11-01", "2026-11-02")


def test_missing_key_message(monkeypatch, tmp_path):
    monkeypatch.delenv("TICKFOUNDRY_API_KEY", raising=False)
    with pytest.raises(errors.MissingApiKey, match="account/api"):
        Client(cache_dir=tmp_path)


def test_sample_without_a_key(tmp_path):
    def handler(req):
        assert "authorization" not in req.headers
        return httpx.Response(200, content=pm_zip("2026-08-19"))

    d = tf.sample(cache_dir=tmp_path, url="https://x.test/samples/polymarket-btc-4h-2026-08-19-day.zip",
                  transport=httpx.MockTransport(handler))
    assert len(d.l2) == 2 and set(d.l2["date"]) == {"2026-08-19"}


def test_avg_bytes_sum_only_the_asked_datasets():
    from tickfoundry.data.catalog import _sum_avg
    assert _sum_avg({"kalshi_l1": 10, "kalshi_l2": 90}, ["kalshi_l1"]) == 10
    assert _sum_avg({"kalshi_l1": 10, "kalshi_l2": 90}, None) == 100
    assert _sum_avg(55, ["x"]) == 55 and _sum_avg(None, None) is None
