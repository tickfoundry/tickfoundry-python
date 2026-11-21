"""Fee ESTIMATES from the venues' published fee rules.

These are estimates, never the fee actually charged. Where a fill's real fee
is recorded it is authoritative: Polymarket's on-chain ``fee_usdc`` in
``trades_onchain``, Hyperliquid's ``fee`` / ``fee_token`` on every fill. Use
these functions to price a hypothetical order, such as a simulated walk.

- Polymarket: a taker pays ``shares x rate x price x (1 - price)`` with the
  market's ``fee_schedule`` (``rate``, ``exponent``, ``takerOnly``,
  ``rebateRate``), published when the market is created. On one series-day
  we measured, 95.4% of 4,874 fee-paying fills matched this estimate within
  2%; on Protocol V2 markets the fee is set when the order is matched. Only
  the ``exponent = 1`` schedule is implemented; any other raises.
- Kalshi: UNVERIFIED. ``0.07 x contracts x price x (1 - price)`` for a taker
  and ``0.0175 x ...`` for a maker on series whose ``fee_type`` charges
  makers, times the series' ``fee_multiplier``. These coefficients, the way
  the multiplier applies and Kalshi's rounding (per fill, then per order) have
  not been checked against Kalshi's published fee schedule; the estimate is
  the unrounded amount and warns on use (``KALSHI_RULE_VERIFIED = False``).
- Hyperliquid: no estimate here; the fee is on every recorded fill.

Rebates and volume tiers are not modelled.
"""
from __future__ import annotations

import warnings
from decimal import Decimal, localcontext
from typing import Any, Iterable, Mapping

from .numeric import DECIMAL_CONTEXT, as_decimal


def _dec(value: Any, name: str) -> Decimal:
    if isinstance(value, float):
        value = repr(value)
    return as_decimal(value, name=name)


def polymarket_fee_estimate(quantity: Any, price: Any, schedule: Mapping[str, Any] | None, *, taker: bool = True) -> Decimal:
    """Estimated fee in USDC for ``quantity`` shares filled at ``price`` under a market's ``fee_schedule``.

    No schedule (a fee-free market) or a maker fill under a ``takerOnly``
    schedule is 0.
    """
    with localcontext(DECIMAL_CONTEXT):
        q = _dec(quantity, "quantity")
        p = _dec(price, "price")
        if q < 0 or not (0 <= p <= 1):
            raise ValueError("quantity must be >= 0 and price within [0, 1]")
        if not schedule:
            return Decimal(0)
        if not taker and schedule.get("takerOnly", True):
            return Decimal(0)
        exponent = schedule.get("exponent", 1)
        if Decimal(str(exponent)) != 1:
            raise ValueError(f"fee_schedule exponent {exponent} is not implemented (only 1)")
        rate = _dec(schedule.get("rate", 0), "rate")
        return q * rate * p * (1 - p)


KALSHI_TAKER_RATE = Decimal("0.07")
KALSHI_MAKER_RATE = Decimal("0.0175")
#: The Kalshi coefficients, multiplier rule and rounding are not yet checked against Kalshi's schedule.
KALSHI_RULE_VERIFIED = False
KALSHI_UNVERIFIED = ("Kalshi fee estimate: the coefficients (0.07 taker, 0.0175 maker), the fee_multiplier rule and "
                     "Kalshi's rounding are not verified against Kalshi's published fee schedule")


def kalshi_fee_estimate(contracts: Any, price: Any, *, fee_multiplier: Any = 1, fee_type: str = "quadratic", taker: bool = True) -> Decimal:
    """UNVERIFIED estimate in dollars for ``contracts`` filled at YES ``price`` (dollars), before Kalshi's rounding.

    A maker pays only on series whose ``fee_type`` contains ``maker``. Warns (``UserWarning``) while
    ``KALSHI_RULE_VERIFIED`` is False.
    """
    if not KALSHI_RULE_VERIFIED:
        warnings.warn(KALSHI_UNVERIFIED, UserWarning, stacklevel=2)
    with localcontext(DECIMAL_CONTEXT):
        c = _dec(contracts, "contracts")
        p = _dec(price, "price")
        m = _dec(fee_multiplier, "fee_multiplier")
        if c < 0 or not (0 <= p <= 1) or m < 0:
            raise ValueError("contracts and fee_multiplier must be >= 0 and price within [0, 1]")
        if taker:
            rate = KALSHI_TAKER_RATE
        elif "maker" in (fee_type or ""):
            rate = KALSHI_MAKER_RATE
        else:
            return Decimal(0)
        return rate * c * p * (1 - p) * m


def walk_fee_estimate(fills: Iterable[Any], *, venue: str, schedule: Mapping[str, Any] | None = None, fee_multiplier: Any = 1,
                      fee_type: str = "quadratic", taker: bool = True) -> Decimal:
    """Estimated fee for a simulated walk (``SimulationResult.fills``): the sum over the levels it consumed.

    Polymarket takes the market's ``fee_schedule``; Kalshi the series' ``fee_multiplier`` and ``fee_type``.
    """
    fills = list(fills)
    with localcontext(DECIMAL_CONTEXT):
        if venue == "polymarket":
            return sum((polymarket_fee_estimate(f.quantity, f.price, schedule, taker=taker) for f in fills), Decimal(0))
        if venue == "kalshi":
            return sum((kalshi_fee_estimate(f.quantity, f.price, fee_multiplier=fee_multiplier, fee_type=fee_type, taker=taker) for f in fills), Decimal(0))
        if venue == "hyperliquid":
            raise ValueError("hyperliquid fees are recorded on every fill (trades.fee); no estimate")
        raise ValueError(f"unknown venue {venue!r}")
