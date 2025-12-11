# finanalyst_tools/utils/math_ops.py
"""
Safe mathematical operations for financial calculations.

This module provides Decimal-based arithmetic functions that:
- Ensure precision (no floating-point errors)
- Handle edge cases gracefully (division by zero, None values)
- Support configurable rounding
- Provide statistical functions for trend analysis

All monetary and ratio calculations should use these functions
to ensure consistency and accuracy throughout the package.

Author: FinAnalyst-Pro Team
Version: 1.0.0
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Sequence, Any, overload
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
    CalculationError,
)


# Type alias for numeric values
Numeric = int | float | Decimal | str


# ============================================================================
# TYPE CONVERSION
# ============================================================================

def to_decimal(
    value: Numeric | None,
    default: Decimal | None = None,
    precision: int | None = None,
) -> Decimal:
    """
    Safely convert a value to Decimal.
    
    Handles various input types (int, float, str, Decimal) and returns
    a default value for None or unconvertible inputs.
    
    Args:
        value: Value to convert (int, float, str, Decimal, or None)
        default: Default value if conversion fails (default: Decimal("0"))
        precision: Optional decimal places for rounding
        
    Returns:
        Decimal representation of the value
        
    Raises:
        InvalidInputError: If value cannot be converted and no default provided
        
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
    
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, (int, float)):
        try:
            # Use str() for floats to avoid floating-point precision issues
            result = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return default
    elif isinstance(value, str):
        try:
            # Remove common formatting characters
            cleaned = value.strip().replace(",", "").replace("$", "").replace("S$", "")
            # Handle percentage strings
            if cleaned.endswith("%"):
                cleaned = cleaned[:-1]
                result = Decimal(cleaned)
            else:
                result = Decimal(cleaned)
        except (InvalidOperation, ValueError):
            return default
    else:
        return default
    
    if precision is not None:
        result = round_decimal(result, precision)
    
    return result


def is_effectively_zero(value: Numeric | None, threshold: float = ZERO_THRESHOLD) -> bool:
    """
    Check if a value is effectively zero (within threshold).
    
    Args:
        value: Value to check
        threshold: Threshold for considering value as zero
        
    Returns:
        True if value is None or within threshold of zero
    """
    if value is None:
        return True
    
    try:
        decimal_value = to_decimal(value)
        return abs(float(decimal_value)) < threshold
    except (ValueError, InvalidOperation):
        return True


# ============================================================================
# SAFE ARITHMETIC OPERATIONS
# ============================================================================

def safe_divide(
    numerator: Numeric | None,
    denominator: Numeric | None,
    default: Decimal | None = None,
    precision: int | None = None,
    raise_on_zero: bool = False,
    denominator_name: str = "denominator",
) -> Decimal:
    """
    Safely divide two numbers, handling zero denominators.
    
    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return if division impossible (default: Decimal("0"))
        precision: Decimal places for result (default: uses ratio precision)
        raise_on_zero: If True, raise exception instead of returning default
        denominator_name: Name of denominator for error messages
        
    Returns:
        Result of division or default value
        
    Raises:
        DivisionByZeroError: If raise_on_zero=True and denominator is zero
        
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
    if precision is None:
        precision = DECIMAL_PLACES["ratio"]
    
    num = to_decimal(numerator)
    denom = to_decimal(denominator)
    
    if is_effectively_zero(denom):
        if raise_on_zero:
            raise DivisionByZeroError(
                numerator=num,
                denominator_name=denominator_name,
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
    Round a Decimal to specified precision.
    
    Args:
        value: Value to round
        precision: Number of decimal places
        rounding: Rounding mode to use
        
    Returns:
        Rounded Decimal value
        
    Examples:
        >>> round_decimal(Decimal("1.2345"), 2)
        Decimal('1.23')
        >>> round_decimal(Decimal("1.235"), 2, RoundingMode.BANKERS)
        Decimal('1.24')
    """
    if value is None:
        return Decimal("0")
    
    decimal_value = to_decimal(value)
    quantize_str = "1." + "0" * precision if precision > 0 else "1"
    
    return decimal_value.quantize(
        Decimal(quantize_str),
        rounding=rounding.get_decimal_rounding()
    )


# ============================================================================
# PERCENTAGE & GROWTH CALCULATIONS
# ============================================================================

def calculate_percentage(
    part: Numeric | None,
    whole: Numeric | None,
    precision: int = 2,
    multiply_by_100: bool = True,
) -> Decimal:
    """
    Calculate percentage: (part / whole) × 100.
    
    Args:
        part: The numerator (part of the whole)
        whole: The denominator (the whole)
        precision: Decimal places for result
        multiply_by_100: If True, return as percentage (e.g., 25.00 for 25%)
        
    Returns:
        Percentage value
        
    Examples:
        >>> calculate_percentage(25, 100)
        Decimal('25.00')
        >>> calculate_percentage(1, 4, multiply_by_100=False)
        Decimal('0.25')
    """
    result = safe_divide(part, whole, precision=precision + 2)
    
    if multiply_by_100:
        result = result * Decimal("100")
    
    return round_decimal(result, precision)


def calculate_growth_rate(
    current: Numeric | None,
    previous: Numeric | None,
    precision: int = 2,
) -> Decimal | None:
    """
    Calculate period-over-period growth rate: ((current - previous) / previous) × 100.
    
    Args:
        current: Current period value
        previous: Previous period value
        precision: Decimal places for result
        
    Returns:
        Growth rate as percentage, or None if cannot be calculated
        
    Examples:
        >>> calculate_growth_rate(110, 100)
        Decimal('10.00')
        >>> calculate_growth_rate(80, 100)
        Decimal('-20.00')
    """
    curr = to_decimal(current)
    prev = to_decimal(previous)
    
    if is_effectively_zero(prev):
        return None  # Cannot calculate growth from zero
    
    growth = ((curr - prev) / prev) * Decimal("100")
    return round_decimal(growth, precision)


def calculate_cagr(
    beginning_value: Numeric | None,
    ending_value: Numeric | None,
    periods: int,
    precision: int = 2,
) -> Decimal | None:
    """
    Calculate Compound Annual Growth Rate (CAGR).
    
    Formula: ((Ending / Beginning) ^ (1/periods) - 1) × 100
    
    Args:
        beginning_value: Starting value
        ending_value: Ending value
        periods: Number of periods
        precision: Decimal places for result
        
    Returns:
        CAGR as percentage, or None if cannot be calculated
        
    Examples:
        >>> calculate_cagr(100, 200, 5)  # Doubled in 5 years
        Decimal('14.87')
    """
    if periods <= 0:
        return None
    
    begin = to_decimal(beginning_value)
    end = to_decimal(ending_value)
    
    if is_effectively_zero(begin) or begin < 0 or end < 0:
        return None
    
    try:
        # Use float for power calculation, then convert back
        ratio = float(end / begin)
        if ratio <= 0:
            return None
        
        cagr = (math.pow(ratio, 1 / periods) - 1) * 100
        return round_decimal(Decimal(str(cagr)), precision)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

def calculate_average(
    *values: Numeric | None,
    precision: int = 2,
    exclude_none: bool = True,
) -> Decimal:
    """
    Calculate arithmetic mean of values.
    
    Args:
        *values: Values to average
        precision: Decimal places for result
        exclude_none: If True, skip None values; if False, treat as 0
        
    Returns:
        Average value
        
    Examples:
        >>> calculate_average(10, 20, 30)
        Decimal('20.00')
        >>> calculate_average(10, None, 30, exclude_none=True)
        Decimal('20.00')
    """
    if not values:
        return Decimal("0")
    
    if exclude_none:
        valid_values = [to_decimal(v) for v in values if v is not None]
    else:
        valid_values = [to_decimal(v) for v in values]
    
    if not valid_values:
        return Decimal("0")
    
    total = sum(valid_values, Decimal("0"))
    count = Decimal(str(len(valid_values)))
    
    return round_decimal(total / count, precision)


def calculate_weighted_average(
    values: Sequence[Numeric | None],
    weights: Sequence[Numeric | None],
    precision: int = 2,
) -> Decimal:
    """
    Calculate weighted average.
    
    Args:
        values: Sequence of values
        weights: Sequence of weights (must match length of values)
        precision: Decimal places for result
        
    Returns:
        Weighted average
        
    Raises:
        InvalidInputError: If values and weights have different lengths
        
    Examples:
        >>> calculate_weighted_average([10, 20, 30], [1, 2, 1])
        Decimal('20.00')  # (10*1 + 20*2 + 30*1) / (1+2+1)
    """
    if len(values) != len(weights):
        raise InvalidInputError(
            "Values and weights must have the same length",
            field_name="weights",
            expected=f"sequence of length {len(values)}",
        )
    
    if not values:
        return Decimal("0")
    
    weighted_sum = Decimal("0")
    weight_sum = Decimal("0")
    
    for value, weight in zip(values, weights):
        if value is not None and weight is not None:
            v = to_decimal(value)
            w = to_decimal(weight)
            weighted_sum += v * w
            weight_sum += w
    
    if is_effectively_zero(weight_sum):
        return Decimal("0")
    
    return round_decimal(weighted_sum / weight_sum, precision)


def calculate_variance(
    values: Sequence[Numeric | None],
    population: bool = True,
    precision: int = 4,
) -> Decimal:
    """
    Calculate variance of values.
    
    Args:
        values: Sequence of values
        population: If True, population variance; if False, sample variance
        precision: Decimal places for result
        
    Returns:
        Variance value
        
    Examples:
        >>> calculate_variance([2, 4, 4, 4, 5, 5, 7, 9])
        Decimal('4.0000')  # Population variance
    """
    valid_values = [to_decimal(v) for v in values if v is not None]
    n = len(valid_values)
    
    if n < 2:
        return Decimal("0")
    
    mean = calculate_average(*valid_values, precision=precision + 2)
    
    squared_diffs = [(v - mean) ** 2 for v in valid_values]
    sum_squared = sum(squared_diffs, Decimal("0"))
    
    divisor = n if population else (n - 1)
    variance = sum_squared / Decimal(str(divisor))
    
    return round_decimal(variance, precision)


def calculate_std_dev(
    values: Sequence[Numeric | None],
    population: bool = True,
    precision: int = 4,
) -> Decimal:
    """
    Calculate standard deviation of values.
    
    Args:
        values: Sequence of values
        population: If True, population std dev; if False, sample std dev
        precision: Decimal places for result
        
    Returns:
        Standard deviation
        
    Examples:
        >>> calculate_std_dev([2, 4, 4, 4, 5, 5, 7, 9])
        Decimal('2.0000')
    """
    variance = calculate_variance(values, population=population, precision=precision + 2)
    
    if variance == 0:
        return Decimal("0")
    
    # Use float for square root, then convert back
    std_dev = Decimal(str(math.sqrt(float(variance))))
    return round_decimal(std_dev, precision)


def calculate_min_max(
    values: Sequence[Numeric | None],
) -> tuple[Decimal, Decimal]:
    """
    Calculate minimum and maximum of values.
    
    Args:
        values: Sequence of values
        
    Returns:
        Tuple of (minimum, maximum)
        
    Examples:
        >>> calculate_min_max([5, 2, 8, 1, 9])
        (Decimal('1'), Decimal('9'))
    """
    valid_values = [to_decimal(v) for v in values if v is not None]
    
    if not valid_values:
        return (Decimal("0"), Decimal("0"))
    
    return (min(valid_values), max(valid_values))


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def compare_values(
    value_a: Numeric | None,
    value_b: Numeric | None,
    tolerance: float = ZERO_THRESHOLD,
) -> int:
    """
    Compare two values with tolerance.
    
    Args:
        value_a: First value
        value_b: Second value
        tolerance: Tolerance for equality comparison
        
    Returns:
        -1 if a < b, 0 if a ≈ b (within tolerance), 1 if a > b
        
    Examples:
        >>> compare_values(100, 100.0001)
        0  # Within tolerance
        >>> compare_values(100, 200)
        -1
    """
    a = to_decimal(value_a)
    b = to_decimal(value_b)
    
    diff = float(a - b)
    
    if abs(diff) <= tolerance:
        return 0
    elif diff < 0:
        return -1
    else:
        return 1


def is_within_range(
    value: Numeric | None,
    min_value: Numeric | None,
    max_value: Numeric | None,
) -> bool:
    """
    Check if value is within range [min_value, max_value].
    
    Args:
        value: Value to check
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        True if value is within range (inclusive)
        
    Examples:
        >>> is_within_range(50, 0, 100)
        True
        >>> is_within_range(150, 0, 100)
        False
    """
    if value is None:
        return False
    
    v = to_decimal(value)
    
    if min_value is not None:
        if v < to_decimal(min_value):
            return False
    
    if max_value is not None:
        if v > to_decimal(max_value):
            return False
    
    return True


# ============================================================================
# FINANCIAL-SPECIFIC CALCULATIONS
# ============================================================================

def calculate_average_balance(
    beginning: Numeric | None,
    ending: Numeric | None,
    precision: int = 2,
) -> Decimal:
    """
    Calculate average balance from beginning and ending values.
    
    Common in financial calculations like ROA, ROE where average
    balance sheet values are used.
    
    Args:
        beginning: Beginning period balance
        ending: Ending period balance
        precision: Decimal places for result
        
    Returns:
        Average of beginning and ending
        
    Examples:
        >>> calculate_average_balance(1000, 1200)
        Decimal('1100.00')
    """
    return calculate_average(beginning, ending, precision=precision)


def annualize_rate(
    periodic_rate: Numeric | None,
    periods_per_year: int,
    precision: int = 2,
) -> Decimal:
    """
    Annualize a periodic rate.
    
    Args:
        periodic_rate: Rate for one period (as percentage)
        periods_per_year: Number of periods in a year
        precision: Decimal places for result
        
    Returns:
        Annualized rate as percentage
        
    Examples:
        >>> annualize_rate(2.5, 4)  # 2.5% quarterly
        Decimal('10.00')  # Simple annualization
    """
    if periods_per_year <= 0:
        return Decimal("0")
    
    rate = to_decimal(periodic_rate)
    annual = rate * Decimal(str(periods_per_year))
    
    return round_decimal(annual, precision)
