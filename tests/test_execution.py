"""Round trip and sliced order: on the Lab's Polymarket fixture day (folder in TF_LAB_FIXTURES), and on a stub replay.

The fixture tests are skipped when the fixture folder or the Lab package is absent.
"""
from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path

import pytest

lab = pytest.importorskip("tickfoundry.lab", reason="the Lab package is not installed")
from tickfoundry.execution import round_trip, sliced_order  # noqa: E402

def _fixtures() -> Path:
    return Path(os.environ.get("TF_LAB_FIXTURES") or "/nonexistent")


FIXTURE = _fixtures() / "polymarket/2026-08-19/btc-up-or-down-4h"



@pytest.fixture(scope="module")
def replay():
    if not FIXTURE.is_dir():
        pytest.skip("Lab fixture day not available (set TF_LAB_FIXTURES)")
    return lab.LocalReplay.from_bundle(FIXTURE)


@pytest.fixture(scope="module")
def busy(replay):
    """The instrument with the most book rows, and a time in the middle of its day."""
    best = max(replay.instrument_ids(), key=lambda i: replay.books(i).n)
    cols = replay.books(best)
    return best, int(cols.t_ns[cols.n // 2])


def test_round_trip_equals_two_lab_simulations(replay, busy):
    inst, t = busy
    rt = round_trip(replay, inst, t, size="100", hold_ms=0)
    a = replay.simulate(inst, t, side="buy", size_mode="base_quantity", size="100")
    b = replay.simulate(inst, t, side="sell", size_mode="base_quantity", size=a.result.filled_quantity)
    assert rt.entry.result.vwap == a.result.vwap and rt.exit.result.vwap == b.result.vwap
    if rt.exit.result.filled_quantity == a.result.filled_quantity:
        assert rt.cost == Decimal(str(a.result.notional)) - Decimal(str(b.result.notional))
        assert rt.cost >= 0  # an immediate round trip pays at least the spread
        assert rt.cost_bps_of_mid is not None
    assert "never saw" in rt.note or "filled only" in rt.note


def test_round_trip_with_budget_and_hold(replay, busy):
    inst, t = busy
    rt = round_trip(replay, inst, t, size="250", size_mode="quote_budget", hold_ms=60_000)
    assert rt.entry.result.size_mode.value == "quote_budget"
    assert rt.exit is None or rt.exit.result.side.value == "sell"


def test_sliced_order_adds_up_its_children_and_says_it_is_a_lower_bound(replay, busy):
    inst, t = busy
    s = sliced_order(replay, inst, t, t + 10 * 60 * 1_000_000_000, side="buy", total_size="1000", slices=5)
    assert len(s.slices) == 5
    assert s.filled_quantity == sum(Decimal(str(x.result.filled_quantity)) for x in s.slices)
    assert s.notional == sum(Decimal(str(x.result.notional)) for x in s.slices)
    assert sum(Decimal(str(x.result.requested_quantity)) for x in s.slices) == Decimal("1000")
    assert "LOWER BOUND" in s.note
    assert s.single_shot is not None and s.single_shot.result.requested_quantity == Decimal("1000")


def test_bad_arguments(replay, busy):
    inst, t = busy
    with pytest.raises(ValueError):
        sliced_order(replay, inst, t, t, side="buy", total_size="10", slices=2)
    with pytest.raises(ValueError):
        round_trip(replay, inst, t, size="10", first="hold")


# -- edge cases on a stub replay -------------------------------------
class _R:
    def __init__(self, filled, notional, status, mid="0.5", requested=None, fees=None):
        from types import SimpleNamespace as NS
        self.result = NS(filled_quantity=Decimal(filled), notional=Decimal(notional), vwap=None, mid_reference=Decimal(mid),
                         requested_quantity=None if requested is None else Decimal(requested), fees=fees)
        self.status = status

    def fee_estimate(self, **kw):
        return Decimal("0.01") * self.result.filled_quantity


class StubReplay:
    def __init__(self, legs):
        self.legs, self.calls = list(legs), []

    def simulate(self, inst, t, **kw):
        self.calls.append((t, kw))
        return self.legs.pop(0)


def test_round_trip_exit_that_cannot_trade_reports_no_cost():
    r = StubReplay([_R("100", "50", "filled_on_visible_book", requested="100"), _R("0", "0", "no_valid_book")])
    rt = round_trip(r, "pm:x", 1_000_000_000, size="100", hold_ms=1000)
    assert rt.cost is None and rt.cost_per_unit is None and rt.cost_bps_of_mid is None
    assert "exit could not trade" in rt.note and rt.entry_filled == Decimal("100") and rt.exit_filled == 0


def test_round_trip_partial_entry_is_visible():
    r = StubReplay([_R("60", "30", "partially_filled_on_visible_book", requested="100"), _R("60", "29.4", "filled_on_visible_book")])
    rt = round_trip(r, "pm:x", 1_000_000_000, size="100")
    assert rt.requested == Decimal("100") and rt.entry_filled == Decimal("60")
    assert "filled 60 of 100" in rt.note and rt.cost == Decimal("0.6")


def test_sliced_order_counts_unfilled_and_estimates_fees_per_leg():
    legs = [_R("200", "100", "filled_on_visible_book"), _R("50", "26", "partially_filled_on_visible_book"),
            _R("200", "104", "filled_on_visible_book"), _R("400", "210", "partially_filled_on_visible_book")]
    s = sliced_order(StubReplay(legs), "pm:x", 0, 3_000_000_000, side="buy", total_size="600", slices=3,
                     fee_multiplier="1")
    assert s.filled_quantity == Decimal("450") and s.unfilled == Decimal("150") and s.slices_not_filled == 1
    assert s.estimated_fees == Decimal("4.50") and "1 of 3 slices" in s.note


def test_block_rule_is_refused():
    with pytest.raises(ValueError, match="block"):
        round_trip(StubReplay([]), "hl:BTC", 0, size="1", block=5)
    with pytest.raises(ValueError, match="execution_rule"):
        sliced_order(StubReplay([]), "hl:BTC", 0, 10, side="buy", total_size="1", slices=1, execution_rule="block_n_minus_1")
