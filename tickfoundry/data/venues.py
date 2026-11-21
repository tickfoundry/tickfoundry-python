"""Per-venue facts the client needs: request routes, layer names and per-request bounds.

The bounds mirror the API's documented limits: a Polymarket, Kalshi or
Hyperliquid request takes a range of days as one bundle (31, 31 and 92 days, up
to 10 GiB). The server stays authoritative: a request it refuses as too large is
split in half and retried, and a server that only takes single Polymarket days
gets one request per day (see client.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

GIB = 1024**3

POLYMARKET = "polymarket"
KALSHI = "kalshi"
HYPERLIQUID = "hyperliquid"
VENUES = (POLYMARKET, KALSHI, HYPERLIQUID)

ALIASES = {"pm": POLYMARKET, "poly": POLYMARKET, "polymarket": POLYMARKET, "kalshi": KALSHI, "ks": KALSHI,
           "hl": HYPERLIQUID, "hyperliquid": HYPERLIQUID}

# Layer names a caller uses -> the dataset ids each venue's request takes.
LAYERS = ("l1", "l2", "trades")


@dataclass(frozen=True)
class VenueSpec:
    name: str
    request_path: str
    max_days: int
    max_bytes: int | None
    datasets: dict[str, str]  # layer -> request dataset id
    id_field: str | None  # body field carrying the ids (None: one unit per request)


SPECS: dict[str, VenueSpec] = {
    POLYMARKET: VenueSpec(POLYMARKET, "/api/v1/downloads", 31, 10 * GIB, {"l1": "l1", "l2": "l2", "trades": "trades"}, "unit_ids"),
    KALSHI: VenueSpec(KALSHI, "/api/v1/kalshi/downloads", 31, 10 * GIB,
                      {"l1": "kalshi_l1", "l2": "kalshi_l2", "trades": "kalshi_trades", "ticker": "kalshi_ticker",
                       "cfbenchmarks": "kalshi_cfbenchmarks"}, "series"),
    HYPERLIQUID: VenueSpec(HYPERLIQUID, "/api/v1/hyperliquid/downloads", 92, 10 * GIB,
                           {"l1": "hl_l1", "l2": "hl_l2", "trades": "hl_trades"}, "coins"),
}

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def normalize_venue(venue: str | None) -> str | None:
    if venue is None:
        return None
    v = ALIASES.get(venue.strip().lower())
    if v is None:
        raise ValueError(f"unknown venue {venue!r}: use one of {', '.join(VENUES)}")
    return v


def is_uuid(s: str) -> bool:
    return bool(_UUID.match(s))


def infer_venue(series_id: str) -> str | None:
    """Best guess from the id's shape; None when it cannot tell.

    A Polymarket series or event is a catalog UUID. Kalshi series tickers are
    upper case and nearly all start with KX. Anything else is ambiguous
    (a Hyperliquid coin like BTC, or an older Kalshi ticker) and needs
    ``venue=``.
    """
    if is_uuid(series_id):
        return POLYMARKET
    if series_id.startswith("KX") and series_id.upper() == series_id:
        return KALSHI
    return None
