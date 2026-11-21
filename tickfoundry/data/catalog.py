"""Find the ids ``pull`` takes: Polymarket series or events, Kalshi series, Hyperliquid coins."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .http import Http
from .venues import HYPERLIQUID, KALSHI, POLYMARKET

COLUMNS = ["venue", "id", "title", "first_date", "last_date"]


def search(http: Http, q: str = "", *, venue: str = POLYMARKET, kind: str | None = None, limit: int = 50,
           category: str | None = None) -> pd.DataFrame:
    """Search one venue's catalog. Returns a DataFrame whose ``id`` column is what ``pull`` takes.

    Polymarket: ``GET /api/v1/catalog`` (title search; ``kind`` = "series" or "event").
    Kalshi: ``GET /api/v1/kalshi`` (series search; ``category`` optional).
    Hyperliquid: ``GET /api/v1/hyperliquid`` lists every coin; filtered here on ``q``.
    """
    rows: list[dict[str, Any]]
    if venue == POLYMARKET:
        params: dict[str, Any] = {"limit": max(1, min(200, limit))}
        if q:
            params["q"] = q[:64]
        if kind:
            params["kind"] = kind
        body = http.get("/api/v1/catalog", params)
        rows = [
            {"venue": POLYMARKET, "id": u.get("id"), "title": u.get("title"), "first_date": u.get("first_date"),
             "last_date": u.get("last_date"), "kind": u.get("kind"), "slug": u.get("slug"),
             "vertical": u.get("vertical"), "market_count": u.get("market_count")}
            for u in body.get("units", [])
        ]
    elif venue == KALSHI:
        params = {"page_size": max(1, min(500, limit))}
        if q:
            params["q"] = q
        if category:
            params["category"] = category
        body = http.get("/api/v1/kalshi", params)
        rows = [
            {"venue": KALSHI, "id": s.get("series"), "title": s.get("title"), "first_date": s.get("first_date"),
             "last_date": s.get("last_date"), "category": s.get("category"), "frequency": s.get("frequency"),
             "days": s.get("days"), "avg_bytes_per_day": s.get("avg_bytes_per_day")}
            for s in body.get("series", [])
        ]
    elif venue == HYPERLIQUID:
        body = http.get("/api/v1/hyperliquid")
        ql = q.lower()
        rows = [
            {"venue": HYPERLIQUID, "id": c.get("coin"), "title": c.get("coin"), "first_date": c.get("first_date"),
             "last_date": c.get("last_date"), "family": c.get("family"), "days": c.get("days"),
             "avg_bytes_per_day": c.get("avg_bytes_per_day")}
            for c in body.get("coins", [])
            if not ql or ql in str(c.get("coin", "")).lower()
        ][:limit]
    else:  # pragma: no cover - normalize_venue guards this
        raise ValueError(f"unknown venue {venue!r}")
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    return df


def _sum_avg(avg: Any, datasets: list[str] | None) -> int | None:
    """The API gives average bytes per day per dataset ({"kalshi_l2": n, ...}); sum the ones asked for."""
    if isinstance(avg, (int, float)):
        return int(avg)
    if isinstance(avg, dict):
        keys = datasets if datasets else list(avg)
        vals = [avg.get(k) for k in keys]
        return int(sum(v for v in vals if isinstance(v, (int, float)))) or None
    return None


def avg_bytes_per_day(http: Http, venue: str, series_id: str, datasets: list[str] | None = None) -> int | None:
    """The catalog's average bytes per day for one Kalshi series or HL coin, over ``datasets`` (None when unknown)."""
    if venue == KALSHI:
        body = http.get("/api/v1/kalshi", {"q": series_id, "page_size": 50})
        for s in body.get("series", []):
            if s.get("series") == series_id:
                return _sum_avg(s.get("avg_bytes_per_day"), datasets)
    elif venue == HYPERLIQUID:
        body = http.get("/api/v1/hyperliquid")
        for c in body.get("coins", []):
            if c.get("coin") == series_id:
                return _sum_avg(c.get("avg_bytes_per_day"), datasets)
    return None


def kalshi_caveats(http: Http) -> list[dict[str, Any]]:
    """Exchange-wide Kalshi caveats the API publishes (first_date, last_date, title, note)."""
    body = http.get("/api/v1/kalshi", {"page_size": 1})
    return list(body.get("known_caveats", []))
