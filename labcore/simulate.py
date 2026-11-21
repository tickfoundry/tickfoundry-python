"""Visible-book execution analysis.

Answers: *what price and quantity does the recorded visible book imply for an
immediate order, subject to these explicit assumptions?*. It does not
model queue position, hidden liquidity, cancellations between observation
and arrival, replenishment, market impact or margin. Each call is an
independent scenario that does not consume the historical book.

Money and quantities are ``Decimal`` under a 60-digit context (exact for the
products involved); ratios (VWAP, basis points) are exact ``Fraction`` values
quantized only at the output boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from enum import Enum
from fractions import Fraction

from .model import (
    CROSSED_BOOK,
    EMPTY_BOOK,
    MISSING_ASK_SIDE,
    MISSING_BID_SIDE,
    BookSnapshot,
    ClockKind,
    Instrument,
    Level,
    ValidatedBook,
    validate_book,
)
from .numeric import (
    BPS_PRECISION_DP,
    DECIMAL_CONTEXT,
    VALUE_PRECISION_DP,
    as_decimal,
    floor_to_increment,
    fraction_to_decimal,
    is_multiple_of,
    quantize_fraction,
)
from .version import ENGINE_VERSION, SCHEMA_VERSION

#: Quantity granularity assumed for quote-budget buys when the instrument has
#: no lot increment. Reported via the ``lot_increment_assumed`` flag.
DEFAULT_QUANTITY_INCREMENT = Decimal("0.00000001")

# Simulator-specific flags (in addition to book quality flags).
FEES_EXCLUDED = "fees_excluded"
LOT_INCREMENT_ASSUMED = "lot_increment_assumed"
REQUESTED_QUANTITY_ROUNDED = "requested_quantity_rounded_to_lot"
DECISION_BOOK_UNAVAILABLE = "decision_book_unavailable"
REFERENCE_UNAVAILABLE = "reference_unavailable"
VWAP_ROUNDED = "vwap_rounded"
LIMIT_PRICE_OFF_TICK = "limit_price_off_tick"
SNAPSHOT_TOO_OLD = "snapshot_too_old"
NO_SNAPSHOT = "no_snapshot"
NO_FILL_WITHIN_LIMIT = "no_fill_within_limit"
BUDGET_BELOW_ONE_LOT = "budget_below_one_lot"
DEPTH_MAY_BE_INCOMPLETE = "visible_depth_may_be_incomplete"


class Side(str, Enum):
    buy = "buy"
    sell = "sell"


class SizeMode(str, Enum):
    base_quantity = "base_quantity"
    quote_budget = "quote_budget"


class ReferenceKind(str, Enum):
    decision_best_quote = "decision_best_quote"
    decision_mid = "decision_mid"
    execution_best_quote = "execution_best_quote"


class FeeModelKind(str, Enum):
    excluded = "excluded"
    explicit_model = "explicit_model"


class SimulationStatus(str, Enum):
    """Simulation status values. A partial result is never labelled "filled"."""

    filled_on_visible_book = "filled_on_visible_book"
    partially_filled_on_visible_book = "partially_filled_on_visible_book"
    no_valid_book = "no_valid_book"
    blocked_by_quality_policy = "blocked_by_quality_policy"


class WalkStop(str, Enum):
    """Why the ladder walk ended; explains a residual quantity or budget."""

    order_filled = "order_filled"
    budget_exhausted = "budget_exhausted"
    budget_below_lot = "budget_below_lot"
    visible_depth_exhausted = "visible_depth_exhausted"
    limit_price_reached = "limit_price_reached"
    no_levels = "no_levels"
    not_walked = "not_walked"


class SimulationError(ValueError):
    """Base class for typed simulator input errors."""


class InvalidSimulationInput(SimulationError):
    """A request parameter is missing, non-positive or inconsistent."""


class QuoteBudgetSellUnsupported(SimulationError):
    """Quote-budget sizing is restricted to buys."""


class LookAheadError(SimulationError):
    """The execution book is later than the execution time (no look-ahead)."""


@dataclass(frozen=True, slots=True)
class FeeModel:
    """Fee assumption. ``excluded`` yields ``fees=None`` plus a warning flag.

    ``bps`` is charged on notional, ``fixed_per_unit`` per quantity unit. With
    ``included_in_budget`` a quote-budget buy reserves the fee from the budget.
    """

    kind: FeeModelKind = FeeModelKind.excluded
    model_id: str | None = None
    bps: Decimal | None = None
    fixed_per_unit: Decimal | None = None
    fee_asset: str | None = None
    included_in_budget: bool = False

    @classmethod
    def excluded(cls) -> "FeeModel":
        return cls()

    @classmethod
    def explicit(
        cls,
        *,
        bps: Decimal | str | int | None = None,
        fixed_per_unit: Decimal | str | int | None = None,
        fee_asset: str | None = None,
        model_id: str | None = None,
        included_in_budget: bool = False,
    ) -> "FeeModel":
        if bps is None and fixed_per_unit is None:
            raise InvalidSimulationInput("explicit fee model needs bps and/or fixed_per_unit")
        return cls(
            kind=FeeModelKind.explicit_model,
            model_id=model_id,
            bps=None if bps is None else as_decimal(bps, name="bps"),
            fixed_per_unit=None if fixed_per_unit is None else as_decimal(fixed_per_unit, name="fixed_per_unit"),
            fee_asset=fee_asset,
            included_in_budget=included_in_budget,
        )

    def fee_for(self, notional: Decimal, quantity: Decimal) -> Decimal | None:
        """Fee for one fill, ``None`` when fees are excluded."""
        if self.kind is FeeModelKind.excluded:
            return None
        fee = Decimal(0)
        if self.bps is not None:
            fee += notional * self.bps / Decimal(10_000)
        if self.fixed_per_unit is not None:
            fee += quantity * self.fixed_per_unit
        return fee

    def fee_per_unit_at(self, unit_cost: Decimal) -> Decimal:
        """Fee per quantity unit at a given unit cost (used for budget reservation)."""
        fee = Decimal(0)
        if self.kind is FeeModelKind.excluded:
            return fee
        if self.bps is not None:
            fee += unit_cost * self.bps / Decimal(10_000)
        if self.fixed_per_unit is not None:
            fee += self.fixed_per_unit
        return fee


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    """Which book conditions block a simulation ("quality policy").

    ``blocking_flags`` block with status ``blocked_by_quality_policy``;
    ``depth_cap`` is the venue's maximum delivered level count (used for the
    ``truncated_depth`` flag); ``max_snapshot_age_ns`` blocks a book older
    than the allowed age relative to the execution time.
    """

    blocking_flags: frozenset[str] = frozenset({CROSSED_BOOK})
    depth_cap: int | None = None
    max_snapshot_age_ns: int | None = None


@dataclass(frozen=True, slots=True)
class Fill:
    """One level consumed by the walk. ``notional = quantity * price * multiplier``."""

    level_index: int
    price: Decimal
    quantity: Decimal
    notional: Decimal
    fee: Decimal | None = None


@dataclass(slots=True)
class SimulationResult:
    """Simulation outputs with explicit units."""

    status: SimulationStatus
    side: Side
    size_mode: SizeMode
    requested_quantity: Decimal | None
    effective_requested_quantity: Decimal | None
    rounding_residual_quantity: Decimal | None
    requested_budget: Decimal | None
    filled_quantity: Decimal
    residual_quantity: Decimal | None
    residual_budget: Decimal | None
    vwap: Decimal | None
    notional: Decimal
    fees: Decimal | None
    fee_asset: str | None
    fee_model: FeeModelKind
    fee_model_id: str | None
    fills: list[Fill]
    levels_consumed: int
    walk_stop: WalkStop
    best_quote_reference: Decimal | None
    mid_reference: Decimal | None
    reference_price: Decimal | None
    reference_kind: ReferenceKind
    execution_best_quote: Decimal | None
    execution_mid: Decimal | None
    adverse_price_difference: Decimal | None
    relative_slippage_bps: Decimal | None
    price_unit: str
    quantity_unit: str
    quote_currency: str | None
    notional_unit: str | None
    adverse_price_difference_unit: str
    relative_slippage_unit: str
    contract_multiplier: Decimal
    lot_increment: Decimal | None
    limit_price: Decimal | None
    assumed_delay_ms: int
    decision_time_ns: int | None
    execution_time_ns: int | None
    execution_book_time_ns: int | None
    decision_book_time_ns: int | None
    snapshot_age_ns: int | None
    clock_kind: ClockKind | None
    visible_depth: int | None
    full_depth_known: bool | None
    quality_flags: list[str]
    scenario: str
    engine_version: str
    schema_version: str
    dataset_version: str | None
    source_fragment_id: str | None

    @property
    def unfilled_quantity(self) -> Decimal | None:
        """Alias of ``residual_quantity`` for base-quantity orders."""
        return self.residual_quantity


@dataclass(slots=True)
class _Walk:
    fills: list[Fill] = field(default_factory=list)
    filled: Decimal = Decimal(0)
    notional: Decimal = Decimal(0)
    fees: Decimal = Decimal(0)
    remaining_quantity: Decimal | None = None
    remaining_budget: Decimal | None = None
    stop: WalkStop = WalkStop.not_walked
    flags: list[str] = field(default_factory=list)


def _limit_blocks(side: Side, price: Decimal, limit_price: Decimal | None) -> bool:
    if limit_price is None:
        return False
    return price > limit_price if side is Side.buy else price < limit_price


def _walk_base_quantity(
    levels: list[Level],
    side: Side,
    quantity: Decimal,
    multiplier: Decimal,
    lot: Decimal | None,
    limit_price: Decimal | None,
    fee_model: FeeModel,
) -> _Walk:
    """Base-quantity walk: ``filled_i = min(remaining, visible_i)``."""
    remaining: Decimal = quantity
    walk = _Walk(remaining_quantity=remaining)
    if not levels:
        walk.stop = WalkStop.no_levels
        return walk
    for index, level in enumerate(levels):
        if _limit_blocks(side, level.price, limit_price):
            walk.stop = WalkStop.limit_price_reached
            return walk
        available = floor_to_increment(level.quantity, lot) if lot is not None else level.quantity
        fill_qty = min(remaining, available)
        if fill_qty <= 0:
            continue
        notional = fill_qty * level.price * multiplier
        fee = fee_model.fee_for(notional, fill_qty)
        walk.fills.append(Fill(index, level.price, fill_qty, notional, fee))
        walk.filled += fill_qty
        walk.notional += notional
        if fee is not None:
            walk.fees += fee
        remaining -= fill_qty
        walk.remaining_quantity = remaining
        if remaining == 0:
            walk.stop = WalkStop.order_filled
            return walk
    walk.stop = WalkStop.visible_depth_exhausted
    return walk


def _walk_quote_budget(
    levels: list[Level],
    budget: Decimal,
    multiplier: Decimal,
    lot: Decimal,
    limit_price: Decimal | None,
    fee_model: FeeModel,
) -> _Walk:
    """Quote-budget buy: quantity limited by remaining budget and lots.

    Fees are debited from the budget only when the model says they are
    included; otherwise they are reported alongside and the budget buys
    notional only.
    """
    remaining: Decimal = budget
    walk = _Walk(remaining_budget=remaining)
    if not levels:
        walk.stop = WalkStop.no_levels
        return walk
    last_effective_unit_cost: Decimal | None = None
    for index, level in enumerate(levels):
        if _limit_blocks(Side.buy, level.price, limit_price):
            walk.stop = WalkStop.limit_price_reached
            return walk
        unit_cost = level.price * multiplier
        effective_unit_cost = unit_cost
        if fee_model.included_in_budget:
            effective_unit_cost = unit_cost + fee_model.fee_per_unit_at(unit_cost)
        last_effective_unit_cost = effective_unit_cost
        affordable = floor_to_increment(Fraction(remaining) / Fraction(effective_unit_cost), lot)
        if affordable <= 0:
            walk.stop = WalkStop.budget_exhausted if remaining == 0 else WalkStop.budget_below_lot
            return walk
        available = floor_to_increment(level.quantity, lot)
        fill_qty = min(available, affordable)
        if fill_qty <= 0:
            continue
        notional = fill_qty * unit_cost
        fee = fee_model.fee_for(notional, fill_qty)
        walk.fills.append(Fill(index, level.price, fill_qty, notional, fee))
        walk.filled += fill_qty
        walk.notional += notional
        remaining -= notional
        if fee is not None:
            walk.fees += fee
            if fee_model.included_in_budget:
                remaining -= fee
        walk.remaining_budget = remaining
        if remaining == 0:
            walk.stop = WalkStop.budget_exhausted
            return walk
    # Ladder exhausted: decide whether budget or depth was the binding constraint.
    if last_effective_unit_cost is not None and remaining < last_effective_unit_cost * lot:
        walk.stop = WalkStop.budget_below_lot
    else:
        walk.stop = WalkStop.visible_depth_exhausted
    return walk


def _reference_quotes(book: ValidatedBook | None, side: Side) -> tuple[Decimal | None, Decimal | None]:
    """(best quote on the side the order hits, mid) from a validated book."""
    if book is None or not book.is_valid:
        return None, None
    best = book.best_ask if side is Side.buy else book.best_bid
    return (best.price if best else None), book.mid


def simulate_execution(
    instrument: Instrument,
    execution_book: BookSnapshot | None,
    side: Side | str,
    size_mode: SizeMode | str,
    size: Decimal | str | int,
    *,
    limit_price: Decimal | str | None = None,
    fee_model: FeeModel | None = None,
    reference: ReferenceKind | str = ReferenceKind.execution_best_quote,
    decision_book: BookSnapshot | None = None,
    analysis_timestamp_ns: int | None = None,
    assumed_delay_ms: int = 0,
    quality_policy: QualityPolicy | None = None,
) -> SimulationResult:
    """Run one visible-book execution analysis.

    ``execution_book`` is the recorded book at or before
    ``analysis_timestamp_ns + assumed_delay_ms`` (selected by the caller with
    ``snapshot_at``); ``decision_book`` is the book at the decision time and
    only feeds the ``decision_*`` references (keeps them separate).
    Fills come exclusively from the execution-time book. Depth is never
    invented: when the visible ladder runs out the remainder is reported
    unfilled, and ``truncated_depth`` says whether deeper liquidity may exist.
    """
    side = Side(side)
    size_mode = SizeMode(size_mode)
    reference = ReferenceKind(reference)
    fee_model = fee_model or FeeModel.excluded()
    policy = quality_policy or QualityPolicy()

    with localcontext(DECIMAL_CONTEXT):
        size_dec = as_decimal(size, name="size")
        if size_dec <= 0:
            raise InvalidSimulationInput("size must be positive")
        if size_mode is SizeMode.quote_budget and side is Side.sell:
            raise QuoteBudgetSellUnsupported(
                "quote_budget sizing is restricted to buys; a sell must specify base quantity"
            )
        if assumed_delay_ms < 0:
            raise InvalidSimulationInput("assumed_delay_ms must be nonnegative")
        limit_dec = None if limit_price is None else as_decimal(limit_price, name="limit_price")
        if limit_dec is not None and limit_dec <= 0:
            raise InvalidSimulationInput("limit_price must be positive")
        multiplier = instrument.contract_multiplier if instrument.contract_multiplier is not None else Decimal(1)
        if multiplier <= 0:
            raise InvalidSimulationInput("contract_multiplier must be positive")
        lot = instrument.lot_increment
        if lot is not None and lot <= 0:
            raise InvalidSimulationInput("lot_increment must be positive")

        flags: list[str] = []
        if fee_model.kind is FeeModelKind.excluded:
            flags.append(FEES_EXCLUDED)
        if limit_dec is not None and instrument.tick_size is not None and not is_multiple_of(limit_dec, instrument.tick_size):
            flags.append(LIMIT_PRICE_OFF_TICK)

        decision_time_ns = analysis_timestamp_ns
        execution_time_ns = None if analysis_timestamp_ns is None else analysis_timestamp_ns + assumed_delay_ms * 1_000_000

        # Sizing bookkeeping ------------------------------------------------
        requested_quantity: Decimal | None = None
        effective_quantity: Decimal | None = None
        rounding_residual: Decimal | None = None
        requested_budget: Decimal | None = None
        walk_lot = lot
        if size_mode is SizeMode.base_quantity:
            requested_quantity = size_dec
            effective_quantity = floor_to_increment(size_dec, lot) if lot is not None else size_dec
            rounding_residual = size_dec - effective_quantity
            if rounding_residual != 0:
                flags.append(REQUESTED_QUANTITY_ROUNDED)
            if effective_quantity <= 0:
                raise InvalidSimulationInput("size is below one lot increment")
        else:
            requested_budget = size_dec
            if walk_lot is None:
                walk_lot = DEFAULT_QUANTITY_INCREMENT
                flags.append(LOT_INCREMENT_ASSUMED)

        # Validate books --------------------------------------------------
        exec_valid = None if execution_book is None else validate_book(execution_book, depth_cap=policy.depth_cap)
        decision_valid = None if decision_book is None else validate_book(decision_book, depth_cap=policy.depth_cap)
        if exec_valid is not None:
            for flag in exec_valid.quality_flags:
                if flag not in flags:
                    flags.append(flag)
        snapshot_age_ns = None
        if execution_book is not None and execution_time_ns is not None:
            snapshot_age_ns = execution_time_ns - execution_book.analysis_time_ns
            if snapshot_age_ns < 0:
                raise LookAheadError("execution_book is later than the execution time")

        def build(status: SimulationStatus, walk: _Walk) -> SimulationResult:
            return _build_result(
                status=status,
                walk=walk,
                instrument=instrument,
                side=side,
                size_mode=size_mode,
                requested_quantity=requested_quantity,
                effective_quantity=effective_quantity,
                rounding_residual=rounding_residual,
                requested_budget=requested_budget,
                fee_model=fee_model,
                reference=reference,
                exec_valid=exec_valid,
                decision_valid=decision_valid,
                execution_book=execution_book,
                decision_book=decision_book,
                assumed_delay_ms=assumed_delay_ms,
                decision_time_ns=decision_time_ns,
                execution_time_ns=execution_time_ns,
                snapshot_age_ns=snapshot_age_ns,
                multiplier=multiplier,
                lot=lot,
                limit_price=limit_dec,
                flags=flags,
            )

        if execution_book is None or exec_valid is None:
            flags.append(NO_SNAPSHOT)
            return build(SimulationStatus.no_valid_book, _Walk(remaining_quantity=effective_quantity, remaining_budget=requested_budget))

        blocking = policy.blocking_flags & set(exec_valid.quality_flags)
        if blocking:
            return build(SimulationStatus.blocked_by_quality_policy, _Walk(remaining_quantity=effective_quantity, remaining_budget=requested_budget))
        if policy.max_snapshot_age_ns is not None and snapshot_age_ns is not None and snapshot_age_ns > policy.max_snapshot_age_ns:
            flags.append(SNAPSHOT_TOO_OLD)
            return build(SimulationStatus.blocked_by_quality_policy, _Walk(remaining_quantity=effective_quantity, remaining_budget=requested_budget))

        levels = exec_valid.asks if side is Side.buy else exec_valid.bids
        if not levels:
            # empty_book / missing side is already in flags from validate_book.
            return build(SimulationStatus.no_valid_book, _Walk(remaining_quantity=effective_quantity, remaining_budget=requested_budget, stop=WalkStop.no_levels))

        # Walk ------------------------------------------------------------
        if size_mode is SizeMode.base_quantity:
            if effective_quantity is None:
                raise InvalidSimulationInput("base_quantity sizing needs a quantity")
            walk = _walk_base_quantity(levels, side, effective_quantity, multiplier, lot, limit_dec, fee_model)
            status = SimulationStatus.filled_on_visible_book if walk.stop is WalkStop.order_filled else SimulationStatus.partially_filled_on_visible_book
        else:
            if requested_budget is None or walk_lot is None:
                raise InvalidSimulationInput("quote_budget sizing needs a budget and a lot increment")
            walk = _walk_quote_budget(levels, requested_budget, multiplier, walk_lot, limit_dec, fee_model)
            budget_bound = walk.stop in (WalkStop.budget_exhausted, WalkStop.budget_below_lot)
            # A budget that cannot buy one lot at the best price fills nothing;
            # that is not a "filled" outcome.
            status = SimulationStatus.filled_on_visible_book if budget_bound and walk.filled > 0 else SimulationStatus.partially_filled_on_visible_book
            if walk.filled == 0 and walk.stop is WalkStop.budget_below_lot:
                flags.append(BUDGET_BELOW_ONE_LOT)
        if walk.filled == 0 and walk.stop is WalkStop.limit_price_reached:
            flags.append(NO_FILL_WITHIN_LIMIT)
        if status is SimulationStatus.partially_filled_on_visible_book and walk.stop is WalkStop.visible_depth_exhausted and exec_valid.full_depth_known is not True:
            flags.append(DEPTH_MAY_BE_INCOMPLETE)
        return build(status, walk)


def _build_result(
    *,
    status: SimulationStatus,
    walk: _Walk,
    instrument: Instrument,
    side: Side,
    size_mode: SizeMode,
    requested_quantity: Decimal | None,
    effective_quantity: Decimal | None,
    rounding_residual: Decimal | None,
    requested_budget: Decimal | None,
    fee_model: FeeModel,
    reference: ReferenceKind,
    exec_valid: ValidatedBook | None,
    decision_valid: ValidatedBook | None,
    execution_book: BookSnapshot | None,
    decision_book: BookSnapshot | None,
    assumed_delay_ms: int,
    decision_time_ns: int | None,
    execution_time_ns: int | None,
    snapshot_age_ns: int | None,
    multiplier: Decimal,
    lot: Decimal | None,
    limit_price: Decimal | None,
    flags: list[str],
) -> SimulationResult:
    """Assemble the simulation output, including references and slippage."""
    # References: decision-time book for decision_* kinds, execution book otherwise.
    execution_best, execution_mid = _reference_quotes(exec_valid, side)
    if reference is ReferenceKind.execution_best_quote:
        best_ref, mid_ref = execution_best, execution_mid
    elif decision_valid is not None:
        best_ref, mid_ref = _reference_quotes(decision_valid, side)
    elif assumed_delay_ms == 0:
        # With no delay the decision book is the execution book.
        best_ref, mid_ref = execution_best, execution_mid
    else:
        best_ref, mid_ref = None, None
        flags.append(DECISION_BOOK_UNAVAILABLE)
    reference_price = mid_ref if reference is ReferenceKind.decision_mid else best_ref

    vwap: Decimal | None = None
    vwap_frac: Fraction | None = None
    if walk.filled > 0:
        vwap_frac = Fraction(walk.notional) / (Fraction(walk.filled) * Fraction(multiplier))
        vwap, exact = fraction_to_decimal(vwap_frac, VALUE_PRECISION_DP)
        if not exact:
            flags.append(VWAP_ROUNDED)

    adverse: Decimal | None = None
    slippage_bps: Decimal | None = None
    if vwap_frac is not None and reference_price is not None and reference_price > 0:
        ref_frac = Fraction(reference_price)
        adverse_frac = vwap_frac - ref_frac if side is Side.buy else ref_frac - vwap_frac
        adverse, _ = fraction_to_decimal(adverse_frac, VALUE_PRECISION_DP)
        slippage_bps = quantize_fraction(adverse_frac / ref_frac * 10_000, BPS_PRECISION_DP)
    elif vwap_frac is not None and reference_price is None:
        flags.append(REFERENCE_UNAVAILABLE)

    fees: Decimal | None
    fee_asset: str | None
    if fee_model.kind is FeeModelKind.excluded:
        fees, fee_asset = None, None
    else:
        fees = walk.fees
        fee_asset = fee_model.fee_asset or instrument.quote_asset

    residual_quantity = walk.remaining_quantity if size_mode is SizeMode.base_quantity else None
    residual_budget = walk.remaining_budget if size_mode is SizeMode.quote_budget else None

    return SimulationResult(
        status=status,
        side=side,
        size_mode=size_mode,
        requested_quantity=requested_quantity,
        effective_requested_quantity=effective_quantity,
        rounding_residual_quantity=rounding_residual,
        requested_budget=requested_budget,
        filled_quantity=walk.filled,
        residual_quantity=residual_quantity,
        residual_budget=residual_budget,
        vwap=vwap,
        notional=walk.notional,
        fees=fees,
        fee_asset=fee_asset,
        fee_model=fee_model.kind,
        fee_model_id=fee_model.model_id,
        fills=list(walk.fills),
        levels_consumed=len(walk.fills),
        walk_stop=walk.stop,
        best_quote_reference=best_ref,
        mid_reference=mid_ref,
        reference_price=reference_price,
        reference_kind=reference,
        execution_best_quote=execution_best,
        execution_mid=execution_mid,
        adverse_price_difference=adverse,
        relative_slippage_bps=slippage_bps,
        price_unit=instrument.price_unit,
        quantity_unit=instrument.quantity_unit,
        quote_currency=instrument.quote_asset,
        notional_unit=instrument.quote_asset,
        adverse_price_difference_unit=instrument.price_unit,
        relative_slippage_unit="bps",
        contract_multiplier=multiplier,
        lot_increment=lot,
        limit_price=limit_price,
        assumed_delay_ms=assumed_delay_ms,
        decision_time_ns=decision_time_ns,
        execution_time_ns=execution_time_ns,
        execution_book_time_ns=None if execution_book is None else execution_book.analysis_time_ns,
        decision_book_time_ns=None if decision_book is None else decision_book.analysis_time_ns,
        snapshot_age_ns=snapshot_age_ns,
        clock_kind=None if execution_book is None else execution_book.clock_kind,
        visible_depth=None if exec_valid is None else exec_valid.visible_depth,
        full_depth_known=None if exec_valid is None else exec_valid.full_depth_known,
        quality_flags=list(dict.fromkeys(flags)),
        scenario="independent",
        engine_version=ENGINE_VERSION,
        schema_version=SCHEMA_VERSION,
        dataset_version=None if execution_book is None else execution_book.dataset_version,
        source_fragment_id=None if execution_book is None else execution_book.source_fragment_id,
    )


__all__ = [
    "BUDGET_BELOW_ONE_LOT",
    "DEFAULT_QUANTITY_INCREMENT",
    "FeeModel",
    "FeeModelKind",
    "Fill",
    "InvalidSimulationInput",
    "LookAheadError",
    "QualityPolicy",
    "QuoteBudgetSellUnsupported",
    "ReferenceKind",
    "Side",
    "SimulationError",
    "SimulationResult",
    "SimulationStatus",
    "SizeMode",
    "WalkStop",
    "simulate_execution",
    "EMPTY_BOOK",
    "MISSING_ASK_SIDE",
    "MISSING_BID_SIDE",
]
