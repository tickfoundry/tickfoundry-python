"""Thin wrappers on the Lab's own simulator (``tickfoundry.lab.LocalReplay.simulate``).

Every number comes from ``labcore.simulate_execution``, the Lab's own simulation
code; these functions only call it more than once and add the results up.

Read the limits before using a number:
- Each call walks the book AS RECORDED. Our order never touched that book, so
  nothing models our own impact, the book refilling after us, queue position or
  hidden liquidity.
- A round trip's exit leg sees the recorded book at the exit time, not a book
  that our entry already ate into.
- A sliced order's later slices see books our earlier slices never moved. Its
  cost is a LOWER BOUND on what working the order would really have cost.
- Time moves by the as-of rule (the last book at or before each time). On
  Hyperliquid that is not the Lab's block rule (decide at block n, execute after
  n-1); call ``LocalReplay.simulate(..., block=n, execution_rule="block_n_minus_1")``
  directly for that. These wrappers refuse ``block`` / ``execution_rule``.
- Mids: each leg's ``mid_reference`` is the mid of the book it executed against
  (under the default reference), i.e. at ``t + assumed_delay_ms`` when a delay is passed.

Fees: ``model_fees`` sums the fees of an explicit ``fee_model`` passed through to
``simulate`` (none by default). ``estimated_fees`` is filled only when you pass
``fee_schedule`` (Polymarket: the market's fee_schedule) or ``fee_multiplier``
(Kalshi: the series' multiplier, with ``fee_type``); it is the venue's published
rule applied to each leg's fills, an ESTIMATE. A real fill's recorded fee
(on-chain on Polymarket, on the tape on Hyperliquid) is always authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping

BPS = Decimal(10_000)
FILLED = "filled_on_visible_book"
_REFUSED = ("block", "execution_rule")


def _ns(t: Any) -> int:
    from .lab import to_ns  # the Lab's local replay

    return to_ns(t, name="timestamp")


def _dec(v: Any) -> Decimal | None:
    return None if v is None else Decimal(str(v))


def _check_kw(kw: Mapping[str, Any]) -> None:
    bad = [k for k in _REFUSED if k in kw]
    if bad:
        raise ValueError(
            f"{', '.join(bad)} is not supported here: these wrappers move in time (as-of rule). For the Hyperliquid "
            "block rule call LocalReplay.simulate(..., block=n, execution_rule='block_n_minus_1') per leg."
        )


class _Fees:
    """Collects the two kinds of fee over several legs."""

    def __init__(self, schedule: Mapping[str, Any] | None, multiplier: Any, fee_type: str) -> None:
        self.estimate = schedule is not None or multiplier is not None
        self.schedule, self.multiplier, self.fee_type = schedule, multiplier, fee_type
        self.model: Decimal | None = None
        self.estimated: Decimal | None = Decimal(0) if self.estimate else None

    def add(self, sim: Any) -> None:
        if sim.result.fees is not None:
            self.model = (self.model or Decimal(0)) + Decimal(str(sim.result.fees))
        if self.estimate and self.estimated is not None:
            kw: dict[str, Any] = {"schedule": self.schedule, "fee_type": self.fee_type}
            if self.multiplier is not None:
                kw["fee_multiplier"] = self.multiplier
            self.estimated += Decimal(str(sim.fee_estimate(**kw)))


@dataclass
class RoundTrip:
    """Enter at ``entry_time``, exit the filled quantity ``hold_ms`` later."""

    entry: Any  # LocalSimulation
    exit: Any | None  # LocalSimulation, None when the entry filled nothing
    requested: Decimal | None  # the entry's requested quantity (None for a quote budget)
    entry_filled: Decimal
    exit_filled: Decimal
    cost: Decimal | None  # quote currency, before fees: what the trip lost against doing nothing, on the exited part
    cost_per_unit: Decimal | None
    cost_bps_of_mid: Decimal | None  # cost per unit against the entry leg's mid
    model_fees: Decimal | None
    estimated_fees: Decimal | None
    note: str = "recorded books; the exit book never saw the entry"

    def to_dict(self) -> dict[str, Any]:
        def s(v: Any) -> str | None:
            return None if v is None else str(v)

        return {
            "requested": s(self.requested), "entry_filled": s(self.entry_filled), "exit_filled": s(self.exit_filled),
            "entry_vwap": s(self.entry.result.vwap), "exit_vwap": s(self.exit.result.vwap) if self.exit else None,
            "entry_status": self.entry.status, "exit_status": self.exit.status if self.exit else None,
            "cost": s(self.cost), "cost_per_unit": s(self.cost_per_unit), "cost_bps_of_mid": s(self.cost_bps_of_mid),
            "model_fees": s(self.model_fees), "estimated_fees": s(self.estimated_fees), "note": self.note,
        }


def round_trip(replay: Any, instrument_id: str, entry_time: Any, *, size: Any, size_mode: str = "base_quantity",
               first: str = "buy", hold_ms: int = 0, fee_schedule: Mapping[str, Any] | None = None,
               fee_multiplier: Any = None, fee_type: str = "quadratic", **simulate_kw: Any) -> RoundTrip:
    """Buy then sell (or ``first="sell"``: sell then buy back) the same quantity ``hold_ms`` later.

    ``size_mode="quote_budget"`` works for a buy-first trip: the exit sells
    whatever quantity the budget bought. Cost, per unit and bps are None when
    either leg could not trade.
    """
    _check_kw(simulate_kw)
    if first not in ("buy", "sell"):
        raise ValueError("first must be 'buy' or 'sell'")
    fees = _Fees(fee_schedule, fee_multiplier, fee_type)
    t0 = _ns(entry_time)
    entry = replay.simulate(instrument_id, t0, side=first, size_mode=size_mode, size=size, **simulate_kw)
    fees.add(entry)
    requested = _dec(entry.result.requested_quantity)
    qty = Decimal(str(entry.result.filled_quantity))
    if qty <= 0:
        return RoundTrip(entry, None, requested, qty, Decimal(0), None, None, None, fees.model, fees.estimated,
                         f"the entry could not trade ({entry.status})")
    back = "sell" if first == "buy" else "buy"
    exit_ = replay.simulate(instrument_id, t0 + int(hold_ms) * 1_000_000, side=back, size_mode="base_quantity", size=qty,
                            **simulate_kw)
    fees.add(exit_)
    out_qty = Decimal(str(exit_.result.filled_quantity))
    if out_qty <= 0:
        return RoundTrip(entry, exit_, requested, qty, out_qty, None, None, None, fees.model, fees.estimated,
                         f"the exit could not trade ({exit_.status}); no cost computed")
    n_in, n_out = Decimal(str(entry.result.notional)), Decimal(str(exit_.result.notional))
    notes = [RoundTrip.note]
    if requested is not None and qty < requested:
        notes.append(f"the entry filled {qty} of {requested} requested")
    if out_qty < qty:
        notes.append(f"the exit filled {out_qty} of {qty}; cost covers the exited part only")
        n_in = n_in * out_qty / qty
    cost = (n_in - n_out) if first == "buy" else (n_out - n_in)
    per = cost / out_qty
    mid = _dec(entry.result.mid_reference)
    bps = (per / mid * BPS) if mid else None
    return RoundTrip(entry, exit_, requested, qty, out_qty, cost, per, bps, fees.model, fees.estimated, "; ".join(notes))


@dataclass
class SlicedOrder:
    """One parent order worked as N equal child orders spread over a window."""

    slices: list[Any] = field(default_factory=list)  # LocalSimulation per child
    total: Decimal = Decimal(0)  # the parent size (quantity, or budget in quote_budget mode)
    filled_quantity: Decimal = Decimal(0)
    unfilled: Decimal | None = None  # total - filled_quantity in base_quantity mode
    slices_not_filled: int = 0  # children whose status is not filled_on_visible_book
    notional: Decimal = Decimal(0)
    vwap: Decimal | None = None
    arrival_mid: Decimal | None = None  # mid at the first slice's execution
    shortfall_bps: Decimal | None = None  # vwap against the arrival mid, signed so a cost is positive
    single_shot: Any | None = None  # the whole size at the start, for comparison
    single_shot_shortfall_bps: Decimal | None = None
    model_fees: Decimal | None = None
    estimated_fees: Decimal | None = None
    note: str = "LOWER BOUND: each slice walks the recorded book; no impact or refill from earlier slices is modelled"


def sliced_order(replay: Any, instrument_id: str, start: Any, end: Any, *, side: str, total_size: Any, slices: int,
                 size_mode: str = "base_quantity", fee_schedule: Mapping[str, Any] | None = None,
                 fee_multiplier: Any = None, fee_type: str = "quadratic", **simulate_kw: Any) -> SlicedOrder:
    """Split ``total_size`` into ``slices`` equal children at even times in [start, end)."""
    _check_kw(simulate_kw)
    if slices < 1:
        raise ValueError("slices must be at least 1")
    t0, t1 = _ns(start), _ns(end)
    if t1 <= t0:
        raise ValueError("end must be after start")
    total = Decimal(str(total_size))
    child = (total / slices).quantize(Decimal("0.00000001"))
    sizes = [child] * (slices - 1) + [total - child * (slices - 1)]
    step = (t1 - t0) // slices
    out = SlicedOrder(total=total)
    fees = _Fees(fee_schedule, fee_multiplier, fee_type)
    sign = Decimal(1) if side == "buy" else Decimal(-1)
    for i, sz in enumerate(sizes):
        sim = replay.simulate(instrument_id, t0 + i * step, side=side, size_mode=size_mode, size=sz, **simulate_kw)
        out.slices.append(sim)
        fees.add(sim)
        if i == 0:
            out.arrival_mid = _dec(sim.result.mid_reference)
        if sim.status != FILLED:
            out.slices_not_filled += 1
        out.filled_quantity += Decimal(str(sim.result.filled_quantity))
        out.notional += Decimal(str(sim.result.notional))
    if size_mode == "base_quantity":
        out.unfilled = total - out.filled_quantity
    if out.filled_quantity > 0:
        out.vwap = out.notional / out.filled_quantity
    if out.vwap is not None and out.arrival_mid:
        out.shortfall_bps = sign * (out.vwap - out.arrival_mid) / out.arrival_mid * BPS
    out.model_fees, out.estimated_fees = fees.model, fees.estimated
    if out.slices_not_filled:
        out.note += f"; {out.slices_not_filled} of {slices} slices did not fill on the visible book"
    whole = replay.simulate(instrument_id, t0, side=side, size_mode=size_mode, size=total, **simulate_kw)
    out.single_shot = whole
    w_vwap = _dec(whole.result.vwap)
    if w_vwap is not None and out.arrival_mid:
        out.single_shot_shortfall_bps = sign * (w_vwap - out.arrival_mid) / out.arrival_mid * BPS
    return out


__all__ = ["RoundTrip", "SlicedOrder", "round_trip", "sliced_order"]
