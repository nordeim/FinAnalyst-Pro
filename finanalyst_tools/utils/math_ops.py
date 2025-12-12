# File: finanalyst_tools/utils/math_ops.py
"""
Safe mathematical operations for financial calculations.

This module provides Decimal-based arithmetic functions that:
- Ensure precision (no floating-point errors)
- Handle edge cases gracefully (division by zero, None values)
- Support configurable rounding
- Provide statistical functions

All monetary and ratio calculations should use these functions
to ensure consistency and accuracy.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Sequence, Any
import math

from finanalyst_tools.config import (
    DECIMAL_PLACES,
    DEFAULT_ROUNDING,
    ZERO_THRESHOLD,
    RoundingMode,
)
from finanalyst_tools.exceptions import (
    DivisionByZeroError,
    InvalidInputError,
)


# Type alias for numeric types
Numeric = int | float | Decimal


# ============================================================================
# TYPE CONVERSION
# ============================================================================

def to_decimal(
    value: Numeric | str | None,
    default: Decimal | None = None,
    precision: int | None = None,
) -> Decimal:
    """
    Safely convert a value to Decimal.
    
    Handles various input types and returns a default value
    for None or unconvertible inputs.
    
    Args:
        value: Value to convert (int, float, str, Decimal, or None)
        default: Default value if conversion fails (default: Decimal("0"))
        precision: Optional precision to round to
        
    Returns:
        Decimal representation of the value
        
    Examples:
        >>> to_decimal(100)
        Decimal('100')
        >>> to_decimal("123.45")
        Decimal('123.45')
        >>> to_decimal(None, default=Decimal("0"))
        Decimal('0')
    """
    if default is None:
        default = Decimal("0")
    
    if value is None:
        return default
    
    try:
        if isinstance(value, Decimal):
            result = value
        elif isinstance(value, float):
            # Use string conversion to avoid float precision issues
            result = Decimal(str(value))
        else:
            result = Decimal(str(value))
        
        if precision is not None:
            result = round_decimal(result, precision)
        
        return result
    except (InvalidOperation, ValueError, TypeError):
        return default


def is_effectively_zero(value: Numeric | None, threshold: float = ZERO_THRESHOLD) -> bool:
    """
    Check if a value is effectively zero (within threshold).
    
    Args:
        value: Value to check
        threshold: Threshold for zero comparison
        
    Returns:
        True if value is None or within threshold of zero
    """
    if value is None:
        return True
    return abs(float(value)) < threshold


# ============================================================================
# SAFE ARITHMETIC
# ============================================================================

def safe_divide(
    numerator: Numeric | None,
    denominator: Numeric | None,
    default: Decimal | None = None,
    precision: int = DECIMAL_PLACES["ratio"],
    raise_on_zero: bool = False,
) -> Decimal:
    """
    Safely divide two numbers with zero handling.
    
    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return if division is impossible (default: Decimal("0"))
        precision: Decimal places to round result to
        raise_on_zero: If True, raise DivisionByZeroError instead of returning default
        
    Returns:
        Result of division, or default if denominator is zero/None
        
    Raises:
        DivisionByZeroError: If raise_on_zero is True and denominator is zero
        
    Examples:
        >>> safe_divide(100, 4)
        Decimal('25.0000')
        >>> safe_divide(100, 0)
        Decimal('0')
        >>> safe_divide(100, 0, raise_on_zero=True)
        DivisionByZeroError: Cannot divide 100 by zero
    """
    if default is None:
        default = Decimal("0")
    
    if numerator is None:
        return default
    
    num = to_decimal(numerator)
    denom = to_decimal(denominator)
    
    if is_effectively_zero(denom):
        if raise_on_zero:
            raise DivisionByZeroError(
                numerator=float(num),
                denominator_name="denominator",
            )
        return default
    
    result = num / denom
    return round_decimal(result, precision)


def round_decimal(
    value: Numeric | None,
    precision: int = 2,
    rounding: RoundingMode = DEFAULT_ROUNDING,
) -> Decimal:
    """
    Round a Decimal value to specified precision.
    
    Args:
        value: Value to round
        precision: Number of decimal places
        rounding: Rounding mode to use
        
    Returns:
        Rounded Decimal value
    """
    if value is None:
        return Decimal("0")
    
    dec_value = to_decimal(value)
    quantize_str = "0." + "0" * precision if precision > 0 else "1"
    return dec_value.quantize(Decimal(quantize_str), rounding=rounding.get_decimal_rounding())


def clamp_value(
    value: Numeric,
    min_value: Numeric | None = None,
    max_value: Numeric | None = None,
) -> Decimal:
    """
    Clamp a value within a range.
    
    Args:
        value: Value to clamp
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        
    Returns:
        Clamped value as Decimal
    """
    result = to_decimal(value)
    
    if min_value is not None:
        min_dec = to_decimal(min_value)
        if result < min_dec:
            result = min_dec
    
    if max_value is not None:
        max_dec = to_decimal(max_value)
        if result > max_dec:
            result = max_dec
    
    return result


# ============================================================================
# PERCENTAGE & GROWTH CALCULATIONS
# ============================================================================

def calculate_percentage(
    part: Numeric | None,
    whole: Numeric | None,
    precision: int = DECIMAL_PLACES["percentage"],
) -> Decimal:
    """
    Calculate percentage: (part / whole) × 100.
    
    Args:
        part: The numerator
        whole: The denominator
        precision: Decimal places for result
        
    Returns:
        Percentage value (e.g., 25.00 for 25%)
    """
    if part is None or whole is None:
        return Decimal("0")
    
    ratio = safe_divide(part, whole, precision=precision + 2)
    return round_decimal(ratio * 100, precision)


def calculate_growth_rate(
    current: Numeric | None,
    previous: Numeric | None,
    precision: int = DECIMAL_PLACES["percentage"],
) -> Decimal | None:
    """
    Calculate period-over-period growth rate.
    
    Formula: ((current - previous) / |previous|) × 100
    
    Args:
        current: Current period value
        previous: Previous period value
        precision: Decimal places for result
        
    Returns:
        Growth rate as percentage, or None if calculation impossible
    """
    if current is None or previous is None:
        return None
    
    curr = to_decimal(current)
    prev = to_decimal(previous)
    
    if is_effectively_zero(prev):
        return None
    
    change = curr - prev
    rate = (change / abs(prev)) * 100
    return round_decimal(rate, precision)


def calculate_cagr(
    beginning_value: Numeric | None,
    ending_value: Numeric | None,
    periods: int,
    precision: int = DECIMAL_PLACES["percentage"],
) -> Decimal | None:
    """
    Calculate Compound Annual Growth Rate (CAGR).
    
    Formula: ((ending / beginning) ^ (1/periods) - 1) × 100
    
    Args:
        beginning_value: Value at start
        ending_value: Value at end
        periods: Number of periods (years)
        precision: Decimal places for result
        
    Returns:
        CAGR as percentage, or None if calculation impossible
    """
    if beginning_value is None or ending_value is None or periods <= 0:
        return None
    
    begin = to_decimal(beginning_value)
    end = to_decimal(ending_value)
    
    if is_effectively_zero(begin) or begin < 0 or end < 0:
        return None
    
    try:
        ratio = float(end / begin)
        cagr = (ratio ** (1 / periods) - 1) * 100
        return round_decimal(Decimal(str(cagr)), precision)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

def calculate_average(
    *values: Numeric | None,
    precision: int = DECIMAL_PLACES["ratio"],
) -> Decimal:
    """
    Calculate arithmetic mean of values.
    
    Args:
        *values: Values to average
        precision: Decimal places for result
        
    Returns:
        Arithmetic mean, or Decimal("0") if no valid values
    """
    valid_values = [to_decimal(v) for v in values if v is not None]
    
    if not valid_values:
        return Decimal("0")
    
    total = sum(valid_values)
    return round_decimal(total / len(valid_values), precision)


def calculate_weighted_average(
    values: Sequence[Numeric | None],
    weights: Sequence[Numeric | None],
    precision: int = DECIMAL_PLACES["ratio"],
) -> Decimal | None:
    """
    Calculate weighted average of values.
    
    Args:
        values: Values to average
        weights: Corresponding weights
        precision: Decimal places for result
        
    Returns:
        Weighted average, or None if calculation impossible
    """
    if len(values) != len(weights):
        return None
    
    pairs = [
        (to_decimal(v), to_decimal(w))
        for v, w in zip(values, weights)
        if v is not None and w is not None
    ]
    
    if not pairs:
        return None
    
    weighted_sum = sum(v * w for v, w in pairs)
    total_weight = sum(w for _, w in pairs)
    
    if is_effectively_zero(total_weight):
        return None
    
    return round_decimal(weighted_sum / total_weight, precision)


def calculate_variance(
    values: Sequence[Numeric | None],
    population: bool = True,
) -> Decimal | None:
    """
    Calculate variance of values.
    
    Args:
        values: Values to calculate variance for
        population: If True, use population variance (N); else sample variance (N-1)
        
    Returns:
        Variance, or None if insufficient values
    """
    valid_values = [to_decimal(v) for v in values if v is not None]
    n = len(valid_values)
    
    if n < 2:
        return None
    
    mean = sum(valid_values) / n
    squared_diffs = [(v - mean) ** 2 for v in valid_values]
    
    divisor = n if population else (n - 1)
    return round_decimal(sum(squared_diffs) / divisor, 6)


def calculate_std_dev(
    values: Sequence[Numeric | None],
    population: bool = True,
) -> Decimal | None:
    """
    Calculate standard deviation of values.
    
    Args:
        values: Values to calculate std dev for
        population: If True, use population std dev; else sample std dev
        
    Returns:
        Standard deviation, or None if insufficient values
    """
    variance = calculate_variance(values, population)
    if variance is None:
        return None
    
    std_dev = Decimal(str(math.sqrt(float(variance))))
    return round_decimal(std_dev, 4)


def calculate_min_max(
    values: Sequence[Numeric | None],
) -> tuple[Decimal, Decimal] | None:
    """
    Calculate minimum and maximum of values.
    
    Args:
        values: Values to find min/max for
        
    Returns:
        Tuple of (min, max), or None if no valid values
    """
    valid_values = [to_decimal(v) for v in values if v is not None]
    
    if not valid_values:
        return None
    
    return (min(valid_values), max(valid_values))
