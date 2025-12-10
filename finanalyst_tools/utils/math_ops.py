# finanalyst_tools/utils/math_ops.py
"""
Safe mathematical operations for financial calculations.
Uses Decimal for precision and handles edge cases.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import TypeVar

from ..config import DECIMAL_PLACES, DEFAULT_ROUNDING


Numeric = TypeVar("Numeric", int, float, Decimal)


class CalculationError(Exception):
    """Raised when a calculation cannot be performed."""
    pass


class DivisionByZeroError(CalculationError):
    """Raised when division by zero is attempted."""
    pass


class InvalidInputError(CalculationError):
    """Raised when input values are invalid."""
    pass


def to_decimal(value: Numeric | str | None, default: Decimal = Decimal("0")) -> Decimal:
    """
    Safely convert a value to Decimal.
    
    Args:
        value: The value to convert
        default: Default value if conversion fails
        
    Returns:
        Decimal representation of the value
    """
    if value is None:
        return default
    
    if isinstance(value, Decimal):
        return value
    
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def safe_divide(
    numerator: Numeric,
    denominator: Numeric,
    default: Decimal | None = None,
    precision: int = DECIMAL_PLACES["ratio"],
    raise_on_zero: bool = False
) -> Decimal | None:
    """
    Safely divide two numbers, handling division by zero.
    
    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return if division by zero (None means return None)
        precision: Decimal places for rounding
        raise_on_zero: If True, raise exception instead of returning default
        
    Returns:
        Result of division or default value
        
    Raises:
        DivisionByZeroError: If raise_on_zero is True and denominator is zero
    """
    num = to_decimal(numerator)
    denom = to_decimal(denominator)
    
    if denom == 0:
        if raise_on_zero:
            raise DivisionByZeroError(
                f"Cannot divide {numerator} by zero"
            )
        return default
    
    result = num / denom
    return round_decimal(result, precision)


def round_decimal(
    value: Decimal | float,
    precision: int = 2,
    rounding: str = ROUND_HALF_UP
) -> Decimal:
    """
    Round a decimal value to specified precision.
    
    Args:
        value: The value to round
        precision: Number of decimal places
        rounding: Rounding mode
        
    Returns:
        Rounded Decimal value
    """
    dec_value = to_decimal(value)
    quantize_str = "0." + "0" * precision if precision > 0 else "0"
    return dec_value.quantize(Decimal(quantize_str), rounding=rounding)


def calculate_percentage(
    part: Numeric,
    whole: Numeric,
    precision: int = DECIMAL_PLACES["percentage"]
) -> Decimal | None:
    """
    Calculate percentage (part/whole * 100).
    
    Args:
        part: The numerator
        whole: The denominator (total)
        precision: Decimal places for result
        
    Returns:
        Percentage value or None if whole is zero
    """
    result = safe_divide(part, whole, precision=precision + 2)
    if result is None:
        return None
    return round_decimal(result * 100, precision)


def calculate_growth_rate(
    current: Numeric,
    previous: Numeric,
    precision: int = DECIMAL_PLACES["growth_rate"]
) -> Decimal | None:
    """
    Calculate growth rate ((current - previous) / previous * 100).
    
    Args:
        current: Current period value
        previous: Previous period value
        precision: Decimal places for result
        
    Returns:
        Growth rate as percentage or None if previous is zero
    """
    curr = to_decimal(current)
    prev = to_decimal(previous)
    
    if prev == 0:
        return None
    
    growth = ((curr - prev) / prev) * 100
    return round_decimal(growth, precision)


def calculate_average(*values: Numeric, precision: int = 2) -> Decimal:
    """
    Calculate arithmetic mean of values.
    
    Args:
        *values: Values to average
        precision: Decimal places for result
        
    Returns:
        Average value
        
    Raises:
        InvalidInputError: If no values provided
    """
    if not values:
        raise InvalidInputError("Cannot calculate average of empty sequence")
    
    decimals = [to_decimal(v) for v in values]
    total = sum(decimals)
    return round_decimal(total / len(decimals), precision)


def calculate_cagr(
    beginning_value: Numeric,
    ending_value: Numeric,
    periods: int,
    precision: int = DECIMAL_PLACES["growth_rate"]
) -> Decimal | None:
    """
    Calculate Compound Annual Growth Rate.
    
    CAGR = (Ending Value / Beginning Value)^(1/n) - 1
    
    Args:
        beginning_value: Starting value
        ending_value: Ending value
        periods: Number of periods
        precision: Decimal places for result
        
    Returns:
        CAGR as percentage or None if calculation not possible
    """
    if periods <= 0:
        return None
    
    begin = to_decimal(beginning_value)
    end = to_decimal(ending_value)
    
    if begin <= 0 or end <= 0:
        return None
    
    # Calculate (end/begin)^(1/periods) - 1
    ratio = float(end / begin)
    cagr = (ratio ** (1 / periods)) - 1
    
    return round_decimal(Decimal(str(cagr * 100)), precision)


def calculate_weighted_average(
    values: list[Numeric],
    weights: list[Numeric],
    precision: int = 2
) -> Decimal:
    """
    Calculate weighted average.
    
    Args:
        values: List of values
        weights: List of weights (must match length of values)
        precision: Decimal places for result
        
    Returns:
        Weighted average
        
    Raises:
        InvalidInputError: If lists are empty or different lengths
    """
    if not values or not weights:
        raise InvalidInputError("Values and weights cannot be empty")
    
    if len(values) != len(weights):
        raise InvalidInputError("Values and weights must have same length")
    
    dec_values = [to_decimal(v) for v in values]
    dec_weights = [to_decimal(w) for w in weights]
    
    total_weight = sum(dec_weights)
    if total_weight == 0:
        raise InvalidInputError("Total weight cannot be zero")
    
    weighted_sum = sum(v * w for v, w in zip(dec_values, dec_weights))
    return round_decimal(weighted_sum / total_weight, precision)
