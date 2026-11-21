"""Exact numeric helpers shared by the simulator and the feature layer.

Money and quantities are ``Decimal``; ratios that may not terminate (VWAP,
basis points, time-weighted means) are computed as exact ``Fraction`` values
and only converted to ``Decimal`` at the output boundary (decimal
arithmetic for execution maths, never floats).
"""
from __future__ import annotations

from decimal import Context, Decimal, InvalidOperation, localcontext
from fractions import Fraction

#: Working precision for Decimal arithmetic inside labcore. Products of
#: prices, quantities and multipliers are exact well within 60 digits.
DECIMAL_CONTEXT = Context(prec=60)

#: Output precision for ratios that do not terminate (VWAP, means).
VALUE_PRECISION_DP = 12
#: Output precision for relative slippage in basis points.
BPS_PRECISION_DP = 6


def as_decimal(value: Decimal | str | int, *, name: str = "value") -> Decimal:
    """Coerce a user-supplied number to ``Decimal`` without going through float.

    Floats are refused on purpose: a float already lost precision upstream
    and the caller must convert it explicitly and record that limitation.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name}: bool is not a valid number")
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, int):
        result = Decimal(value)
    elif isinstance(value, str):
        try:
            result = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"{name}: not a decimal string: {value!r}") from exc
    elif isinstance(value, float):
        raise TypeError(f"{name}: floats are not accepted; pass a decimal string")
    else:
        raise TypeError(f"{name}: unsupported numeric type {type(value).__name__}")
    if not result.is_finite():
        raise ValueError(f"{name}: must be finite, got {result}")
    return result


def fraction_to_decimal(value: Fraction, max_dp: int = VALUE_PRECISION_DP) -> tuple[Decimal, bool]:
    """Convert a fraction to ``Decimal``.

    Returns ``(decimal, exact)``. When the fraction terminates within
    ``max_dp`` decimal places the conversion is exact; otherwise the value is
    rounded half-even to ``max_dp`` places and ``exact`` is ``False`` so the
    caller can flag the rounding.
    """
    numerator, denominator = value.numerator, value.denominator
    reduced = denominator
    twos = fives = 0
    while reduced % 2 == 0:
        reduced //= 2
        twos += 1
    while reduced % 5 == 0:
        reduced //= 5
        fives += 1
    with localcontext(DECIMAL_CONTEXT):
        if reduced == 1 and max(twos, fives) <= max_dp:
            places = max(twos, fives)
            scaled = numerator * (10**places) // denominator
            return Decimal(scaled).scaleb(-places), True
        scaled = round(value * (10**max_dp))
        return Decimal(scaled).scaleb(-max_dp), False


def quantize_fraction(value: Fraction, places: int) -> Decimal:
    """Round a fraction half-even to a fixed number of decimal places."""
    with localcontext(DECIMAL_CONTEXT):
        return Decimal(round(value * (10**places))).scaleb(-places)


def floor_to_increment(value: Decimal | Fraction, increment: Decimal) -> Decimal:
    """Largest multiple of ``increment`` that is ``<= value`` (exact)."""
    if increment <= 0:
        raise ValueError("increment must be positive")
    ratio = Fraction(value) / Fraction(increment)
    steps = ratio.numerator // ratio.denominator
    with localcontext(DECIMAL_CONTEXT):
        return Decimal(steps) * increment


def is_multiple_of(value: Decimal, increment: Decimal) -> bool:
    """True when ``value`` is an exact integer multiple of ``increment``."""
    ratio = Fraction(value) / Fraction(increment)
    return ratio.denominator == 1
