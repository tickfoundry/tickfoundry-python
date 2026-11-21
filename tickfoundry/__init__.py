"""TickFoundry Python SDK.

    import tickfoundry as tf

    d = tf.sample()                                          # public Polymarket sample day, no key
    tf.search("bitcoin up or down", venue="polymarket")      # find ids (TICKFOUNDRY_API_KEY set)
    l2 = tf.pull("KXBTC15M", "2026-11-12", layers="l2")      # one layer -> one DataFrame
    d = tf.pull(api_key, series_id, start_dt, end_dt)        # the key may also be the first argument

``tickfoundry.data`` holds the download client. ``tickfoundry.lab`` (the Lab
client and local replay) and ``tickfoundry.execution`` (round trips and sliced
orders on the Lab's simulator) are optional at import time.
"""
from __future__ import annotations

from typing import Any

from .data import Client, Data, PullPlan, TickFoundryError, errors, sample

try:  # execution cost and local replay (the Lab's own code); absent in a data-only build
    from .execution import RoundTrip, SlicedOrder, round_trip, sliced_order
    from .lab import LabClient, LocalReplay
except ImportError:  # pragma: no cover
    RoundTrip = SlicedOrder = round_trip = sliced_order = LabClient = LocalReplay = None  # type: ignore[assignment,misc]

__version__ = "0.2.1"

_clients: dict[str | None, Client] = {}


def _split_key(args: tuple[Any, ...], kw: dict[str, Any]) -> tuple[Client, tuple[Any, ...]]:
    key = kw.pop("api_key", None)
    if key is None and args and isinstance(args[0], str) and args[0].startswith(("tf_live_", "tf_test_")):
        key, args = args[0], args[1:]
    client = _clients.get(key)
    if client is None:
        client = _clients[key] = Client(key)
    return client, args


def pull(*args: Any, **kw: Any) -> Data | Any:
    """``tf.pull([api_key,] series_id, start, end=None, venue=None, layers=("l1","l2","trades"))``. See ``Client.pull``."""
    client, args = _split_key(args, kw)
    return client.pull(*args, **kw)


def search(*args: Any, **kw: Any) -> Any:
    """``tf.search(q, venue="polymarket", kind=None, limit=50)``. See ``Client.search``."""
    client, args = _split_key(args, kw)
    return client.search(*args, **kw)


def me(*args: Any, **kw: Any) -> dict[str, Any]:
    """Plan, limits and usage today for the key. See ``Client.me``."""
    client, args = _split_key(args, kw)
    return client.me(*args, **kw)


def plan_pull(*args: Any, **kw: Any) -> PullPlan:
    """Dry run of a pull: requests, cache hits, size, what would block. See ``Client.plan_pull``."""
    client, args = _split_key(args, kw)
    return client.plan_pull(*args, **kw)


def known_gaps(*args: Any, **kw: Any) -> Any:
    """Known gaps and caveats for a series over a range. See ``Client.known_gaps``."""
    client, args = _split_key(args, kw)
    return client.known_gaps(*args, **kw)


__all__ = ["Client", "Data", "LabClient", "LocalReplay", "PullPlan", "RoundTrip", "SlicedOrder", "TickFoundryError",
           "__version__", "errors", "known_gaps", "me", "plan_pull", "pull", "round_trip", "sample", "search",
           "sliced_order"]
