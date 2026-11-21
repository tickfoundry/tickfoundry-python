"""Cost to move: what a taker spends to push the visible book's price by a
given amount, on one recorded book.

For a book with best bid ``b`` and best ask ``a`` (``b < a``) the mid is
``(a + b) / 2``. The *up* move sweeps every ask at or under the up target
(cost ``price x quantity``); the *down* move sweeps every bid at or over the
down target. On a binary outcome book (prices in dollars per $1 share) the
down move is bought on the other outcome, at ``(1 - bid) x quantity``; on any
other book it is the notional sold, ``bid x quantity``. ``mean_cost`` is the
average of the two directions.

Moves:

- ``cents=D``: targets ``mid + D`` and ``mid - D`` (binary books, dollars).
- ``odds=F``: the odds ``mid / (1 - mid)`` multiplied (up) or divided (down)
  by ``F``; the same amount of evidence at every price (binary books).
- ``bps=N``: targets ``mid x (1 +- N / 10000)`` (any book).

This is the per-book number behind the TickFoundry cost-to-move index, in
exact decimal arithmetic. A side is ``capped`` when its last visible level is
still inside the band and the venue's book on that side may be deeper than
the levels visible here; the cost is then a lower bound. A locked or crossed
book, or a missing side, has no mid and no cost (``status = no_valid_book``).
It does not model queue position, hidden liquidity, refills or impact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, localcontext

from .model import BookSnapshot, Level, validate_book
from .numeric import DECIMAL_CONTEXT, as_decimal

OK = "ok"
NO_VALID_BOOK = "no_valid_book"
#: The visible ladder ended inside the band and the book may continue.
DEPTH_MAY_BE_INCOMPLETE = "visible_depth_may_be_incomplete"


@dataclass(frozen=True, slots=True)
class MoveSide:
    """One direction of the move. ``cost`` is in the quote currency, ``quantity`` in the instrument's unit."""

    target: Decimal
    cost: Decimal
    quantity: Decimal
    levels: int
    capped: bool | None


@dataclass(slots=True)
class CostToMove:
    status: str
    move: str
    amount: Decimal
    binary: bool
    mid: Decimal | None = None
    spread: Decimal | None = None
    up: MoveSide | None = None
    down: MoveSide | None = None
    mean_cost: Decimal | None = None
    flags: list[str] = field(default_factory=list)


def _targets(mid: Decimal, *, cents: Decimal | None, odds: Decimal | None, bps: Decimal | None) -> tuple[Decimal, Decimal]:
    if cents is not None:
        return mid + cents, mid - cents
    if odds is not None:
        o = mid / (1 - mid)
        up, dn = o * odds, o / odds
        return up / (1 + up), dn / (1 + dn)
    assert bps is not None
    step = mid * bps / Decimal(10_000)
    return mid + step, mid - step


def _side(levels: list[Level], target: Decimal, *, ask: bool, binary: bool, multiplier: Decimal, total_levels: int | None, complete: bool | None) -> MoveSide:
    cost = Decimal(0)
    qty = Decimal(0)
    used = 0
    for level in levels:
        inside = level.price <= target if ask else level.price >= target
        if not inside:
            break
        unit = level.price if ask or not binary else (1 - level.price)
        cost += unit * level.quantity * multiplier
        qty += level.quantity
        used += 1
    capped: bool | None = False
    if levels and used == len(levels):
        # The whole visible side sits inside the band: deeper levels could too.
        if complete is True:
            capped = False
        elif total_levels is not None:
            capped = total_levels > len(levels)
        else:
            capped = None
    return MoveSide(target=target, cost=cost, quantity=qty, levels=used, capped=capped)


def cost_to_move(
    book: BookSnapshot,
    *,
    cents: Decimal | str | int | None = None,
    odds: Decimal | str | int | None = None,
    bps: Decimal | str | int | None = None,
    binary: bool = True,
    multiplier: Decimal | str | int = 1,
    bid_levels_total: int | None = None,
    ask_levels_total: int | None = None,
    bid_complete: bool | None = None,
    ask_complete: bool | None = None,
) -> CostToMove:
    """Cost to move one book by ``cents``, ``odds`` or ``bps`` (exactly one).

    ``bid_levels_total`` / ``ask_levels_total`` are the venue's level counts on
    each side when the source carries them (``bid_levels`` / ``ask_levels``);
    ``*_complete`` says a side is known to end at its last visible level. With
    neither, a side whose visible levels all sit inside the band is reported
    ``capped = None`` (unknown).
    """
    given = [name for name, v in (("cents", cents), ("odds", odds), ("bps", bps)) if v is not None]
    if len(given) != 1:
        raise ValueError("pass exactly one of cents, odds or bps")
    move = given[0]
    with localcontext(DECIMAL_CONTEXT):
        raw = {"cents": cents, "odds": odds, "bps": bps}[move]
        assert raw is not None
        amount = as_decimal(repr(raw) if isinstance(raw, float) else raw, name=move)
        if amount <= 0:
            raise ValueError(f"{move} must be positive")
        if move == "odds" and amount <= 1:
            raise ValueError("odds must be greater than 1")
        if move in ("cents", "odds") and not binary:
            raise ValueError(f"{move} moves are defined for binary outcome books; use bps")
        mult = as_decimal(multiplier, name="multiplier")
        out = CostToMove(status=NO_VALID_BOOK, move=move, amount=amount, binary=binary)
        valid = validate_book(book)
        bids, asks = valid.bids, valid.asks
        if not bids or not asks or bids[0].price >= asks[0].price:
            return out
        mid = (bids[0].price + asks[0].price) / 2
        if binary and not (0 < mid < 1):
            return out
        up_t, dn_t = _targets(mid, cents=amount if move == "cents" else None, odds=amount if move == "odds" else None, bps=amount if move == "bps" else None)
        up = _side(asks, up_t, ask=True, binary=binary, multiplier=mult, total_levels=ask_levels_total, complete=ask_complete)
        dn = _side(bids, dn_t, ask=False, binary=binary, multiplier=mult, total_levels=bid_levels_total, complete=bid_complete)
        out.status = OK
        out.mid = mid
        out.spread = asks[0].price - bids[0].price
        out.up, out.down = up, dn
        out.mean_cost = (up.cost + dn.cost) / 2
        if up.capped is not False or dn.capped is not False:
            out.flags.append(DEPTH_MAY_BE_INCOMPLETE)
        return out
