# finanalyst_tools/utils/__init__.py
```py
# File: finanalyst_tools/utils/__init__.py
"""
Utility functions for FinAnalyst-Pro Agent Tools.

This module exports:
- Math operations (safe division, decimal handling, statistics)
- Formatting functions (numbers, currency, percentages)
- Currency utilities (SGD-specific, GST calculations)
"""

from finanalyst_tools.utils.math_ops import (
    to_decimal,
    safe_divide,
    round_decimal,
    calculate_percentage,
    calculate_growth_rate,
    calculate_cagr,
    calculate_average,
    calculate_weighted_average,
    calculate_variance,
    calculate_std_dev,
    calculate_min_max,
    is_effectively_zero,
    clamp_value,
)

from finanalyst_tools.utils.formatting import (
    format_number,
    format_currency,
    format_percentage,
    format_ratio,
    format_change,
    format_large_number,
    format_trend_indicator,
    format_status_indicator,
    format_markdown_table,
    format_value_with_unit,
)

from finanalyst_tools.utils.currency import (
    get_currency_symbol,
    get_currency_decimals,
    get_currency_name,
    is_supported_currency,
    format_sgd,
    calculate_gst_exclusive,
    calculate_gst_inclusive,
    calculate_gst_amount,
    is_sfrs_small_entity,
    SGD_GST_RATE,
    CURRENCY_INFO,
)


__all__ = [
    # Math operations
    "to_decimal",
    "safe_divide",
    "round_decimal",
    "calculate_percentage",
    "calculate_growth_rate",
    "calculate_cagr",
    "calculate_average",
    "calculate_weighted_average",
    "calculate_variance",
    "calculate_std_dev",
    "calculate_min_max",
    "is_effectively_zero",
    "clamp_value",
    # Formatting
    "format_number",
    "format_currency",
    "format_percentage",
    "format_ratio",
    "format_change",
    "format_large_number",
    "format_trend_indicator",
    "format_status_indicator",
    "format_markdown_table",
    "format_value_with_unit",
    # Currency
    "get_currency_symbol",
    "get_currency_decimals",
    "get_currency_name",
    "is_supported_currency",
    "format_sgd",
    "calculate_gst_exclusive",
    "calculate_gst_inclusive",
    "calculate_gst_amount",
    "is_sfrs_small_entity",
    "SGD_GST_RATE",
    "CURRENCY_INFO",
]

```

# finanalyst_tools/utils/math_ops.py
```py
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

```

# finanalyst_tools/utils/formatting.py
```py
# File: finanalyst_tools/utils/formatting.py
"""
Formatting utilities for financial data display.

Provides consistent formatting for:
- Numbers with thousands separators
- Currency values with symbols
- Percentages and ratios
- Trend indicators
- Markdown tables

All functions handle None gracefully with configurable fallback values.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Sequence

from finanalyst_tools.config import (
    DECIMAL_PLACES,
    CURRENCY_SYMBOLS,
    DEFAULT_CURRENCY,
    TREND_SYMBOLS,
    STATUS_SYMBOLS,
    LARGE_NUMBER_SUFFIXES,
    METRIC_UNITS,
)
from finanalyst_tools.utils.math_ops import to_decimal


# ============================================================================
# NUMBER FORMATTING
# ============================================================================

def format_number(
    value: float | Decimal | int | None,
    precision: int = 2,
    use_thousands_sep: bool = True,
    fallback: str = "N/A",
) -> str:
    """
    Format a number with optional thousands separator.
    
    Args:
        value: Number to format
        precision: Decimal places
        use_thousands_sep: Whether to include thousand separators
        fallback: String to return if value is None
        
    Returns:
        Formatted string
    """
    if value is None:
        return fallback
    
    dec_value = to_decimal(value, precision=precision)
    
    if use_thousands_sep:
        return f"{float(dec_value):,.{precision}f}"
    return f"{float(dec_value):.{precision}f}"


def format_currency(
    value: float | Decimal | int | None,
    currency_code: str = DEFAULT_CURRENCY,
    precision: int = DECIMAL_PLACES["currency"],
    show_symbol: bool = True,
    fallback: str = "N/A",
) -> str:
    """
    Format a monetary value with currency symbol.
    
    Args:
        value: Amount to format
        currency_code: ISO currency code
        precision: Decimal places
        show_symbol: Whether to include currency symbol
        fallback: String to return if value is None
        
    Returns:
        Formatted currency string (e.g., "S$1,234.56")
    """
    if value is None:
        return fallback
    
    dec_value = to_decimal(value, precision=precision)
    formatted = f"{float(dec_value):,.{precision}f}"
    
    if show_symbol:
        symbol = CURRENCY_SYMBOLS.get(currency_code, currency_code)
        # Handle negative values
        if dec_value < 0:
            return f"-{symbol}{formatted.lstrip('-')}"
        return f"{symbol}{formatted}"
    
    return formatted


def format_percentage(
    value: float | Decimal | None,
    precision: int = DECIMAL_PLACES["percentage"],
    show_symbol: bool = True,
    show_sign: bool = False,
    fallback: str = "N/A",
) -> str:
    """
    Format a percentage value.
    
    Args:
        value: Percentage value (e.g., 25.5 for 25.5%)
        precision: Decimal places
        show_symbol: Whether to include % symbol
        show_sign: Whether to show + for positive values
        fallback: String to return if value is None
        
    Returns:
        Formatted percentage string
    """
    if value is None:
        return fallback
    
    dec_value = to_decimal(value, precision=precision)
    
    sign = ""
    if show_sign and dec_value > 0:
        sign = "+"
    
    formatted = f"{float(dec_value):.{precision}f}"
    
    if show_symbol:
        return f"{sign}{formatted}%"
    return f"{sign}{formatted}"


def format_ratio(
    value: float | Decimal | None,
    precision: int = 2,
    suffix: str = "x",
    fallback: str = "N/A",
) -> str:
    """
    Format a ratio value.
    
    Args:
        value: Ratio value (e.g., 1.5 for 1.5x)
        precision: Decimal places
        suffix: Suffix to append (default: "x")
        fallback: String to return if value is None
        
    Returns:
        Formatted ratio string (e.g., "1.50x")
    """
    if value is None:
        return fallback
    
    dec_value = to_decimal(value, precision=precision)
    return f"{float(dec_value):.{precision}f}{suffix}"


def format_change(
    value: float | Decimal | None,
    precision: int = 2,
    show_sign: bool = True,
    fallback: str = "N/A",
) -> str:
    """
    Format a change value with +/- sign.
    
    Args:
        value: Change value
        precision: Decimal places
        show_sign: Whether to show + for positive values
        fallback: String to return if value is None
        
    Returns:
        Formatted change string
    """
    if value is None:
        return fallback
    
    dec_value = to_decimal(value, precision=precision)
    
    if show_sign and dec_value > 0:
        return f"+{float(dec_value):.{precision}f}"
    return f"{float(dec_value):.{precision}f}"


def format_large_number(
    value: float | Decimal | int | None,
    precision: int = 1,
    fallback: str = "N/A",
) -> str:
    """
    Format large numbers with K/M/B/T suffixes.
    
    Args:
        value: Number to format
        precision: Decimal places
        fallback: String to return if value is None
        
    Returns:
        Formatted string (e.g., "1.5M", "500K")
    """
    if value is None:
        return fallback
    
    num = float(to_decimal(value))
    
    if num == 0:
        return "0"
    
    sign = "-" if num < 0 else ""
    num = abs(num)
    
    for threshold, suffix in LARGE_NUMBER_SUFFIXES:
        if num >= threshold:
            formatted = num / threshold
            return f"{sign}{formatted:.{precision}f}{suffix}"
    
    return f"{sign}{num:.{precision}f}"


# ============================================================================
# INDICATOR FORMATTING
# ============================================================================

def format_trend_indicator(
    direction: str,
    include_text: bool = False,
) -> str:
    """
    Format a trend direction as an indicator.
    
    Args:
        direction: One of "increasing", "decreasing", "stable", "volatile"
        include_text: Whether to include text after symbol
        
    Returns:
        Trend indicator symbol (e.g., "↑", "↓")
    """
    symbol = TREND_SYMBOLS.get(direction.lower(), "?")
    
    if include_text:
        return f"{symbol} {direction.capitalize()}"
    return symbol


def format_status_indicator(
    status: str,
    include_text: bool = False,
) -> str:
    """
    Format a status as an indicator.
    
    Args:
        status: One of "good", "warning", "error", "info", "unknown"
        include_text: Whether to include text after symbol
        
    Returns:
        Status indicator symbol (e.g., "✅", "⚠️")
    """
    symbol = STATUS_SYMBOLS.get(status.lower(), STATUS_SYMBOLS["unknown"])
    
    if include_text:
        return f"{symbol} {status.capitalize()}"
    return symbol


def format_value_with_unit(
    value: float | Decimal | None,
    metric_name: str,
    precision: int | None = None,
    currency_code: str = DEFAULT_CURRENCY,
) -> str:
    """
    Format a value with appropriate unit based on metric name.
    
    Args:
        value: Value to format
        metric_name: Name of the metric (used to determine unit)
        precision: Override precision (uses default for unit type if None)
        currency_code: Currency code for currency values
        
    Returns:
        Formatted string with appropriate unit
    """
    if value is None:
        return "N/A"
    
    unit = METRIC_UNITS.get(metric_name.lower(), "number")
    
    if unit == "percentage":
        prec = precision if precision is not None else DECIMAL_PLACES["percentage"]
        return format_percentage(value, precision=prec)
    elif unit == "ratio":
        prec = precision if precision is not None else DECIMAL_PLACES["ratio"]
        return format_ratio(value, precision=prec)
    elif unit == "currency":
        prec = precision if precision is not None else DECIMAL_PLACES["currency"]
        return format_currency(value, currency_code=currency_code, precision=prec)
    elif unit == "days":
        prec = precision if precision is not None else 0
        return f"{int(value)} days"
    else:
        prec = precision if precision is not None else 2
        return format_number(value, precision=prec)


# ============================================================================
# TABLE FORMATTING
# ============================================================================

def format_markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    alignment: Sequence[str] | None = None,
) -> str:
    """
    Format data as a Markdown table.
    
    Args:
        headers: Column headers
        rows: Table rows (each row is a sequence of cell values)
        alignment: Column alignments ("left", "center", "right") per column
        
    Returns:
        Markdown table string
    """
    if not headers or not rows:
        return ""
    
    # Convert all values to strings
    str_headers = [str(h) for h in headers]
    str_rows = [[str(cell) for cell in row] for row in rows]
    
    # Calculate column widths
    col_widths = [len(h) for h in str_headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))
    
    # Build alignment row
    if alignment is None:
        alignment = ["left"] * len(headers)
    
    align_row = []
    for i, align in enumerate(alignment):
        width = col_widths[i] if i < len(col_widths) else 3
        if align == "center":
            align_row.append(":" + "-" * (width - 2) + ":")
        elif align == "right":
            align_row.append("-" * (width - 1) + ":")
        else:  # left
            align_row.append(":" + "-" * (width - 1))
    
    # Build table
    lines = []
    
    # Header row
    header_cells = [h.ljust(col_widths[i]) for i, h in enumerate(str_headers)]
    lines.append("| " + " | ".join(header_cells) + " |")
    
    # Alignment row
    lines.append("| " + " | ".join(align_row) + " |")
    
    # Data rows
    for row in str_rows:
        cells = []
        for i, cell in enumerate(row):
            width = col_widths[i] if i < len(col_widths) else len(cell)
            cells.append(cell.ljust(width))
        lines.append("| " + " | ".join(cells) + " |")
    
    return "\n".join(lines)

```

# finanalyst_tools/utils/serialization.py
```py
from __future__ import annotations

from dataclasses import is_dataclass
from decimal import Decimal
from enum import Enum
from typing import Any


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]

    if is_dataclass(value):
        return to_jsonable(value.__dict__)

    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump(by_alias=False))
        except TypeError:
            return to_jsonable(value.model_dump())

    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        return to_jsonable(value.to_dict())

    return str(value)

```

# finanalyst_tools/utils/currency.py
```py
# File: finanalyst_tools/utils/currency.py
"""
Currency handling utilities with Singapore focus.

Provides:
- Currency information (symbols, decimal places, names)
- SGD-specific formatting
- GST calculations (Singapore 9% rate)
- SFRS for Small Entities qualification checks
"""

from __future__ import annotations

from decimal import Decimal
from typing import Final

from finanalyst_tools.config import (
    SingaporeConstants,
    SUPPORTED_CURRENCIES,
    CURRENCY_SYMBOLS,
    ZERO_DECIMAL_CURRENCIES,
    DECIMAL_PLACES,
)
from finanalyst_tools.utils.math_ops import to_decimal, round_decimal


# ============================================================================
# SINGAPORE CONSTANTS
# ============================================================================

SGD_GST_RATE: Final[Decimal] = SingaporeConstants.GST_RATE

# SFRS for Small Entities thresholds (qualify if meet 2 of 3)
SFRS_THRESHOLDS: Final[dict[str, int]] = {
    "revenue": SingaporeConstants.SFRS_SMALL_ENTITY_REVENUE,
    "total_assets": SingaporeConstants.SFRS_SMALL_ENTITY_ASSETS,
    "employees": SingaporeConstants.SFRS_SMALL_ENTITY_EMPLOYEES,
}


# ============================================================================
# CURRENCY INFORMATION
# ============================================================================

CURRENCY_INFO: Final[dict[str, dict[str, str | int]]] = {
    "SGD": {"symbol": "S$", "decimals": 2, "name": "Singapore Dollar"},
    "USD": {"symbol": "$", "decimals": 2, "name": "US Dollar"},
    "EUR": {"symbol": "€", "decimals": 2, "name": "Euro"},
    "GBP": {"symbol": "£", "decimals": 2, "name": "British Pound"},
    "JPY": {"symbol": "¥", "decimals": 0, "name": "Japanese Yen"},
    "CNY": {"symbol": "¥", "decimals": 2, "name": "Chinese Yuan"},
    "HKD": {"symbol": "HK$", "decimals": 2, "name": "Hong Kong Dollar"},
    "AUD": {"symbol": "A$", "decimals": 2, "name": "Australian Dollar"},
    "MYR": {"symbol": "RM", "decimals": 2, "name": "Malaysian Ringgit"},
    "IDR": {"symbol": "Rp", "decimals": 0, "name": "Indonesian Rupiah"},
    "THB": {"symbol": "฿", "decimals": 2, "name": "Thai Baht"},
    "INR": {"symbol": "₹", "decimals": 2, "name": "Indian Rupee"},
    "KRW": {"symbol": "₩", "decimals": 0, "name": "South Korean Won"},
    "NZD": {"symbol": "NZ$", "decimals": 2, "name": "New Zealand Dollar"},
    "PHP": {"symbol": "₱", "decimals": 2, "name": "Philippine Peso"},
    "VND": {"symbol": "₫", "decimals": 0, "name": "Vietnamese Dong"},
}


# ============================================================================
# CURRENCY UTILITIES
# ============================================================================

def get_currency_symbol(currency_code: str) -> str:
    """
    Get the symbol for a currency code.
    
    Args:
        currency_code: ISO currency code (e.g., "SGD")
        
    Returns:
        Currency symbol (e.g., "S$")
    """
    info = CURRENCY_INFO.get(currency_code.upper())
    if info:
        return str(info["symbol"])
    return currency_code


def get_currency_decimals(currency_code: str) -> int:
    """
    Get the standard decimal places for a currency.
    
    Args:
        currency_code: ISO currency code
        
    Returns:
        Number of decimal places (0 for JPY, KRW, etc.)
    """
    info = CURRENCY_INFO.get(currency_code.upper())
    if info:
        return int(info["decimals"])
    return 2  # Default


def get_currency_name(currency_code: str) -> str:
    """
    Get the full name of a currency.
    
    Args:
        currency_code: ISO currency code
        
    Returns:
        Currency name (e.g., "Singapore Dollar")
    """
    info = CURRENCY_INFO.get(currency_code.upper())
    if info:
        return str(info["name"])
    return currency_code


def is_supported_currency(currency_code: str) -> bool:
    """
    Check if a currency is supported.
    
    Args:
        currency_code: ISO currency code
        
    Returns:
        True if supported
    """
    return currency_code.upper() in SUPPORTED_CURRENCIES


def format_sgd(
    amount: float | Decimal | int | None,
    show_symbol: bool = True,
    precision: int = 2,
) -> str:
    """
    Format an amount in Singapore Dollars.
    
    Args:
        amount: Amount to format
        show_symbol: Whether to include S$ symbol
        precision: Decimal places
        
    Returns:
        Formatted SGD string (e.g., "S$1,234.56")
    """
    if amount is None:
        return "N/A"
    
    dec_amount = to_decimal(amount, precision=precision)
    formatted = f"{float(dec_amount):,.{precision}f}"
    
    if show_symbol:
        if dec_amount < 0:
            return f"-S${formatted.lstrip('-')}"
        return f"S${formatted}"
    return formatted


# ============================================================================
# GST CALCULATIONS
# ============================================================================

def calculate_gst_exclusive(gst_inclusive: float | Decimal) -> Decimal:
    """
    Convert GST-inclusive amount to GST-exclusive.
    
    Args:
        gst_inclusive: Amount including GST
        
    Returns:
        Amount excluding GST
        
    Example:
        >>> calculate_gst_exclusive(109)
        Decimal('100.00')
    """
    amount = to_decimal(gst_inclusive)
    result = amount / (1 + SGD_GST_RATE)
    return round_decimal(result, DECIMAL_PLACES["currency"])


def calculate_gst_inclusive(gst_exclusive: float | Decimal) -> Decimal:
    """
    Convert GST-exclusive amount to GST-inclusive.
    
    Args:
        gst_exclusive: Amount excluding GST
        
    Returns:
        Amount including GST
        
    Example:
        >>> calculate_gst_inclusive(100)
        Decimal('109.00')
    """
    amount = to_decimal(gst_exclusive)
    result = amount * (1 + SGD_GST_RATE)
    return round_decimal(result, DECIMAL_PLACES["currency"])


def calculate_gst_amount(base_amount: float | Decimal) -> Decimal:
    """
    Calculate the GST portion on a GST-exclusive amount.
    
    Args:
        base_amount: GST-exclusive amount
        
    Returns:
        GST amount
        
    Example:
        >>> calculate_gst_amount(100)
        Decimal('9.00')
    """
    amount = to_decimal(base_amount)
    result = amount * SGD_GST_RATE
    return round_decimal(result, DECIMAL_PLACES["currency"])


# ============================================================================
# SFRS QUALIFICATION
# ============================================================================

def is_sfrs_small_entity(
    annual_revenue: float | Decimal | None = None,
    total_assets: float | Decimal | None = None,
    num_employees: int | None = None,
) -> tuple[bool, list[str]]:
    """
    Check if a company qualifies as a Small Entity under SFRS.
    
    A company qualifies if it meets at least 2 of the 3 criteria:
    - Annual revenue ≤ S$10M
    - Total assets ≤ S$10M
    - Employees ≤ 50
    
    Args:
        annual_revenue: Annual revenue in SGD
        total_assets: Total assets in SGD
        num_employees: Number of employees
        
    Returns:
        Tuple of (qualifies, list of met criteria)
    """
    criteria_met = []
    
    if annual_revenue is not None:
        if to_decimal(annual_revenue) <= SFRS_THRESHOLDS["revenue"]:
            criteria_met.append("revenue")
    
    if total_assets is not None:
        if to_decimal(total_assets) <= SFRS_THRESHOLDS["total_assets"]:
            criteria_met.append("total_assets")
    
    if num_employees is not None:
        if num_employees <= SFRS_THRESHOLDS["employees"]:
            criteria_met.append("employees")
    
    qualifies = len(criteria_met) >= 2
    return (qualifies, criteria_met)

```

# finanalyst_tools/validation/reconciliation.py
```py
# File: finanalyst_tools/validation/reconciliation.py
"""
Cross-statement reconciliation validation.

Verifies consistency between values that should match across
different financial statements:
- Net income (IS vs CF)
- Cash balance (BS vs CF)
- Retained earnings rollforward
- Balance sheet equation
- Working capital consistency
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from finanalyst_tools.config import ReconciliationTolerances
from finanalyst_tools.models.validation import (
    ReconciliationCheck,
    ReconciliationResult,
)
from finanalyst_tools.models.financial_statements import (
    IncomeStatementData,
    BalanceSheetData,
    CashFlowStatementData,
    FinancialStatementSet,
)
from finanalyst_tools.utils.math_ops import to_decimal, is_effectively_zero


def _create_check(
    check_name: str,
    statement_a: str,
    value_a: Decimal,
    statement_b: str,
    value_b: Decimal,
    tolerance_level: str = "normal",
) -> ReconciliationCheck:
    """
    Create a reconciliation check result.
    
    Args:
        check_name: Name of the check
        statement_a: Source of first value
        value_a: First value
        statement_b: Source of second value
        value_b: Second value
        tolerance_level: Tolerance level ("strict", "normal", "loose")
        
    Returns:
        ReconciliationCheck with pass/fail result
    """
    tolerance = ReconciliationTolerances.get_tolerance(tolerance_level)
    difference = abs(value_a - value_b)
    
    # Calculate if within tolerance
    passed = ReconciliationTolerances.is_within_tolerance(
        float(value_a),
        float(value_b),
        tolerance,
    )
    
    if passed:
        message = f"Values match within {tolerance:.1%} tolerance"
    else:
        pct_diff = (difference / max(abs(value_a), abs(value_b), Decimal("1"))) * 100
        message = f"Values differ by {difference:,.2f} ({pct_diff:.1f}%), exceeds {tolerance:.1%} tolerance"
    
    return ReconciliationCheck(
        check_name=check_name,
        statement_a=statement_a,
        value_a=value_a,
        statement_b=statement_b,
        value_b=value_b,
        difference=difference,
        tolerance=tolerance,
        passed=passed,
        message=message,
    )


def reconcile_net_income(
    income_statement: IncomeStatementData,
    cash_flow_statement: CashFlowStatementData,
) -> ReconciliationCheck:
    """
    Verify net income matches between Income Statement and Cash Flow Statement.
    
    Args:
        income_statement: Income statement data
        cash_flow_statement: Cash flow statement data
        
    Returns:
        ReconciliationCheck result
    """
    is_net_income = income_statement.calculated_net_income
    cf_net_income = cash_flow_statement.net_income
    
    return _create_check(
        check_name="Net Income Reconciliation",
        statement_a="Income Statement",
        value_a=is_net_income,
        statement_b="Cash Flow Statement",
        value_b=cf_net_income,
        tolerance_level="strict",
    )


def reconcile_cash_balance(
    balance_sheet: BalanceSheetData,
    cash_flow_statement: CashFlowStatementData,
) -> ReconciliationCheck | None:
    """
    Verify ending cash balance matches between Balance Sheet and Cash Flow Statement.
    
    Args:
        balance_sheet: Balance sheet data
        cash_flow_statement: Cash flow statement data
        
    Returns:
        ReconciliationCheck result or None if ending_cash not provided
    """
    if cash_flow_statement.ending_cash is None:
        return None
    
    bs_cash = balance_sheet.cash_and_equivalents
    cf_ending_cash = cash_flow_statement.ending_cash
    
    return _create_check(
        check_name="Cash Balance Reconciliation",
        statement_a="Balance Sheet",
        value_a=bs_cash,
        statement_b="Cash Flow (Ending)",
        value_b=cf_ending_cash,
        tolerance_level="strict",
    )


def reconcile_retained_earnings(
    current_balance_sheet: BalanceSheetData,
    prior_balance_sheet: BalanceSheetData | None,
    income_statement: IncomeStatementData,
    dividends_paid: Decimal | None = None,
) -> ReconciliationCheck | None:
    """
    Verify retained earnings rollforward.
    
    Formula: Prior RE + Net Income - Dividends = Current RE
    
    Args:
        current_balance_sheet: Current period balance sheet
        prior_balance_sheet: Prior period balance sheet
        income_statement: Current period income statement
        dividends_paid: Dividends paid during period (optional)
        
    Returns:
        ReconciliationCheck result or None if prior BS not provided
    """
    if prior_balance_sheet is None:
        return None
    
    prior_re = prior_balance_sheet.retained_earnings
    net_income = income_statement.calculated_net_income
    dividends = dividends_paid or Decimal("0")
    
    expected_re = prior_re + net_income - dividends
    actual_re = current_balance_sheet.retained_earnings
    
    return _create_check(
        check_name="Retained Earnings Rollforward",
        statement_a="Calculated (Prior RE + NI - Div)",
        value_a=expected_re,
        statement_b="Balance Sheet",
        value_b=actual_re,
        tolerance_level="normal",
    )


def reconcile_balance_sheet_equation(
    balance_sheet: BalanceSheetData,
) -> ReconciliationCheck:
    """
    Verify the fundamental accounting equation: Assets = Liabilities + Equity.
    
    Args:
        balance_sheet: Balance sheet data
        
    Returns:
        ReconciliationCheck result
    """
    total_assets = balance_sheet.calculated_total_assets
    total_liab_equity = (
        balance_sheet.calculated_total_liabilities +
        balance_sheet.calculated_total_equity
    )
    
    return _create_check(
        check_name="Balance Sheet Equation",
        statement_a="Total Assets",
        value_a=total_assets,
        statement_b="Liabilities + Equity",
        value_b=total_liab_equity,
        tolerance_level="strict",
    )


def reconcile_working_capital(
    balance_sheet: BalanceSheetData,
) -> ReconciliationCheck:
    """
    Verify working capital calculation consistency.
    
    Args:
        balance_sheet: Balance sheet data
        
    Returns:
        ReconciliationCheck result
    """
    current_assets = balance_sheet.calculated_current_assets
    current_liabilities = balance_sheet.calculated_current_liabilities
    calculated_wc = current_assets - current_liabilities
    
    # Compare with the property calculation
    property_wc = balance_sheet.working_capital
    
    return _create_check(
        check_name="Working Capital Consistency",
        statement_a="CA - CL Calculation",
        value_a=calculated_wc,
        statement_b="Working Capital Property",
        value_b=property_wc,
        tolerance_level="strict",
    )


def run_all_reconciliations(
    statement_set: FinancialStatementSet,
    prior_balance_sheet: BalanceSheetData | None = None,
    dividends_paid: Decimal | None = None,
) -> ReconciliationResult:
    """
    Run all applicable reconciliation checks.
    
    Args:
        statement_set: Complete set of financial statements
        prior_balance_sheet: Prior period balance sheet (optional)
        dividends_paid: Dividends paid during period (optional)
        
    Returns:
        ReconciliationResult with all check results
    """
    result = ReconciliationResult()
    
    # Balance sheet equation (always run)
    bs_equation = reconcile_balance_sheet_equation(statement_set.balance_sheet)
    result.add_check(bs_equation)
    
    # Working capital consistency (always run)
    wc_check = reconcile_working_capital(statement_set.balance_sheet)
    result.add_check(wc_check)
    
    # Net income reconciliation (if cash flow available)
    if statement_set.cash_flow_statement:
        ni_check = reconcile_net_income(
            statement_set.income_statement,
            statement_set.cash_flow_statement,
        )
        result.add_check(ni_check)
        
        # Cash balance reconciliation
        cash_check = reconcile_cash_balance(
            statement_set.balance_sheet,
            statement_set.cash_flow_statement,
        )
        if cash_check:
            result.add_check(cash_check)
    
    # Retained earnings rollforward (if prior BS available)
    if prior_balance_sheet:
        re_check = reconcile_retained_earnings(
            statement_set.balance_sheet,
            prior_balance_sheet,
            statement_set.income_statement,
            dividends_paid,
        )
        if re_check:
            result.add_check(re_check)
    
    return result

```

# finanalyst_tools/validation/__init__.py
```py
# File: finanalyst_tools/validation/__init__.py
"""
Validation functions for FinAnalyst-Pro Agent Tools.

This package provides:
- Schema validation for financial statements
- Cross-statement reconciliation
- Plausibility checks for calculated metrics
"""

from finanalyst_tools.validation.schema_validator import (
    validate_income_statement_schema,
    validate_balance_sheet_schema,
    validate_cash_flow_schema,
    validate_financial_data_completeness,
    validate_statement_set,
    REQUIRED_FIELDS,
    FIELD_ALIASES,
)

from finanalyst_tools.validation.reconciliation import (
    reconcile_net_income,
    reconcile_cash_balance,
    reconcile_retained_earnings,
    reconcile_balance_sheet_equation,
    reconcile_working_capital,
    run_all_reconciliations,
)

from finanalyst_tools.validation.plausibility import (
    check_plausibility,
    check_all_plausibility,
    PlausibilityChecker,
)


__all__ = [
    # Schema validation
    "validate_income_statement_schema",
    "validate_balance_sheet_schema",
    "validate_cash_flow_schema",
    "validate_financial_data_completeness",
    "validate_statement_set",
    "REQUIRED_FIELDS",
    "FIELD_ALIASES",
    # Reconciliation
    "reconcile_net_income",
    "reconcile_cash_balance",
    "reconcile_retained_earnings",
    "reconcile_balance_sheet_equation",
    "reconcile_working_capital",
    "run_all_reconciliations",
    # Plausibility
    "check_plausibility",
    "check_all_plausibility",
    "PlausibilityChecker",
]

```

# finanalyst_tools/validation/utils.py
```py
# finanalyst_tools/validation/utils.py
"""
Utility functions for validation system integration.

Provides:
- Exception to ValidationResult conversion
- ValidationResult to reasoning block formatting
- Cross-system type mapping utilities
- Enhanced error handling and recovery
"""

from __future__ import annotations

from typing import Any, cast
from decimal import Decimal

from finanalyst_tools.models.validation import (
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
)
from finanalyst_tools.exceptions import (
    FinAnalystError,
    CalculationError,
    ValidationError,
    DataError,
    ToolError,
    DivisionByZeroError,
    InvalidInputError,
    SchemaValidationError,
    DataCompletenessError,
    ReconciliationError,
    PlausibilityError,
    DataParsingError,
    MissingDataError,
    ToolNotFoundError,
    ToolExecutionError,
    ToolParameterError,
)


def exception_to_validation_result(
    exc: Exception,
    field: str = "unknown",
    context: str = "general operation"
) -> ValidationResult:
    """
    Convert any exception to a ValidationResult object.
    
    This is the central integration point between the exception hierarchy
    and the validation system.
    
    Args:
        exc: Exception to convert
        field: Field name for validation issue
        context: Context description for error message
        
    Returns:
        ValidationResult with the error
    """
    result = ValidationResult()
    
    # Handle different exception types with appropriate severity mapping
    severity_map = {
        DivisionByZeroError: ValidationSeverity.ERROR,
        InvalidInputError: ValidationSeverity.ERROR,
        SchemaValidationError: ValidationSeverity.ERROR,
        DataCompletenessError: ValidationSeverity.ERROR,
        ReconciliationError: ValidationSeverity.ERROR,
        ToolNotFoundError: ValidationSeverity.ERROR,
        ToolExecutionError: ValidationSeverity.ERROR,
        ToolParameterError: ValidationSeverity.ERROR,
        PlausibilityError: ValidationSeverity.WARNING,
        DataParsingError: ValidationSeverity.ERROR,
        MissingDataError: ValidationSeverity.WARNING,
        ValueError: ValidationSeverity.ERROR,
        TypeError: ValidationSeverity.ERROR,
    }
    
    severity = severity_map.get(type(exc), ValidationSeverity.ERROR)
    
    # Get exception details
    details = {}
    if hasattr(exc, 'details'):
        details = getattr(exc, 'details', {})
    elif hasattr(exc, '__dict__'):
        details = exc.__dict__
    
    # Create validation issue
    issue = ValidationIssue(
        field=field,
        message=str(exc),
        severity=severity,
        actual_value=str(details.get('actual_value', 'unknown')),
        expected=str(details.get('expected', 'valid value')),
        suggestion=get_exception_suggestion(exc, context)
    )
    
    result.add_issue(issue)
    result.context["error_type"] = type(exc).__name__
    result.context["context"] = context
    
    return result


def get_exception_suggestion(exc: Exception, context: str = "general operation") -> str:
    """
    Get a helpful suggestion for resolving an exception.
    
    Args:
        exc: Exception to get suggestion for
        context: Context of the error
        
    Returns:
        Suggestion string
    """
    if isinstance(exc, DivisionByZeroError):
        return "Check denominator values are non-zero before calculation"
    elif isinstance(exc, InvalidInputError):
        if hasattr(exc, 'expected'):
            return f"Provide a value that is: {exc.expected}"
        return "Verify input data format and values"
    elif isinstance(exc, SchemaValidationError):
        return "Check that your data matches the expected schema structure"
    elif isinstance(exc, DataCompletenessError):
        if hasattr(exc, 'missing_fields'):
            return f"Provide the missing fields: {', '.join(exc.missing_fields)}"
        return "Ensure all required data fields are provided"
    elif isinstance(exc, ReconciliationError):
        return "Verify data accuracy across financial statements"
    elif isinstance(exc, PlausibilityError):
        return "Review input data for accuracy - values may be outside normal ranges"
    elif isinstance(exc, ToolNotFoundError):
        if hasattr(exc, 'available_tools'):
            return f"Use one of the available tools: {', '.join(exc.available_tools[:3])}"
        return "Check that the tool name is correct and available"
    elif isinstance(exc, ToolExecutionError):
        return "Check tool parameters and try again"
    elif isinstance(exc, ToolParameterError):
        if hasattr(exc, 'expected_type'):
            return f"Provide a valid {exc.expected_type} value for this parameter"
        return "Check parameter requirements and provide valid values"
    elif isinstance(exc, DataParsingError):
        return "Verify source data format and encoding"
    elif isinstance(exc, MissingDataError):
        if hasattr(exc, 'missing_fields'):
            return f"Provide the missing fields: {', '.join(exc.missing_fields)}"
        return "Ensure all required data is available"
    
    return f"Review the error and try again. Contact support if the issue persists."


def result_to_reasoning_block(result: ValidationResult) -> str:
    """
    Convert ValidationResult to formatted reasoning block.
    
    Args:
        result: Validation result to format
        
    Returns:
        Formatted markdown block
    """
    lines = [
        f"### Validation Result for {result.context.get('context', 'analysis')}",
        "",
        "**Summary**:",
        f"- Status: {'✅ Valid' if result.is_valid else '❌ Invalid'}",
        f"- Errors: {result.error_count}",
        f"- Warnings: {result.warning_count}",
        f"- Info: {result.info_count}",
        "",
    ]
    
    if not result.is_valid:
        lines.append("**Errors**:")
        for issue in result.issues:
            error_icon = "❌ " if issue.severity == ValidationSeverity.ERROR else "⚠️ "
            lines.append(f"  - {error_icon}{issue.field}: {issue.message}")
            if issue.actual_value is not None:
                lines.append(f"    Actual: {issue.actual_value}, Expected: {issue.expected or 'valid value'}")
            if issue.suggestion:
                lines.append(f"    Suggestion: {issue.suggestion}")
        lines.append("")
    
    if result.warning_count > 0:
        lines.append("**Warnings**:")
        for issue in result.warnings:
            lines.append(f"  - ⚠️ {issue.field}: {issue.message}")
            if issue.suggestion:
                lines.append(f"    Suggestion: {issue.suggestion}")
        lines.append("")
    
    if result.info_count > 0:
        lines.append("**Information**:")
        for issue in result.info:
            lines.append(f"  - ℹ️ {issue.field}: {issue.message}")
        lines.append("")
    
    lines.append("**Recommendation**:")
    if result.can_proceed:
        lines.append("✅ Analysis can proceed with the provided data.")
        if result.warning_count > 0:
            lines.append("⚠️ However, please review the warnings for potential data quality issues.")
    else:
        lines.append("❌ Analysis cannot proceed due to validation errors.")
        lines.append("Please correct the errors before continuing.")
    
    return "\n".join(lines)


def merge_validation_results(results: list[ValidationResult]) -> ValidationResult:
    """
    Merge multiple ValidationResult objects into one.
    
    Args:
        results: List of validation results to merge
        
    Returns:
        Merged ValidationResult
    """
    merged = ValidationResult()
    
    for result in results:
        merged = merged.merge(result)
    
    return merged


def is_validation_successful(result: ValidationResult) -> bool:
    """
    Check if validation was successful (no errors).
    
    Args:
        result: ValidationResult to check
        
    Returns:
        True if no errors, False otherwise
    """
    return result.can_proceed


def get_validation_summary(result: ValidationResult) -> dict[str, Any]:
    """
    Get a summary of validation results.
    
    Args:
        result: ValidationResult to summarize
        
    Returns:
        Summary dictionary
    """
    return {
        "is_valid": result.is_valid,
        "can_proceed": result.can_proceed,
        "error_count": result.error_count,
        "warning_count": result.warning_count,
        "info_count": result.info_count,
        "total_issues": result.total_issue_count,
    }

```

# finanalyst_tools/validation/schema_validator.py
```py
# File: finanalyst_tools/validation/schema_validator.py
"""
Schema validation for financial statement data.

Provides validation functions to verify:
- Required fields are present
- Field types are correct
- Data is complete for requested analysis type
"""

from __future__ import annotations

from typing import Any
from decimal import Decimal

from finanalyst_tools.models.validation import (
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
)
from finanalyst_tools.models.financial_statements import (
    IncomeStatementData,
    BalanceSheetData,
    CashFlowStatementData,
    FinancialStatementSet,
)


# ============================================================================
# FIELD DEFINITIONS
# ============================================================================

REQUIRED_FIELDS: dict[str, dict[str, list[str]]] = {
    "profitability": {
        "income_statement": ["total_revenue", "cost_of_goods_sold"],
        "balance_sheet": ["total_assets", "total_shareholders_equity"],
    },
    "liquidity": {
        "balance_sheet": [
            "current_assets", "current_liabilities",
            "cash_and_equivalents", "inventory"
        ],
    },
    "solvency": {
        "balance_sheet": [
            "total_liabilities", "total_shareholders_equity",
            "total_assets", "long_term_debt"
        ],
        "income_statement": ["interest_expense"],
    },
    "efficiency": {
        "income_statement": ["total_revenue", "cost_of_goods_sold"],
        "balance_sheet": ["inventory", "accounts_receivable", "accounts_payable", "total_assets"],
    },
    "comprehensive": {
        "income_statement": [
            "total_revenue", "cost_of_goods_sold", "net_income"
        ],
        "balance_sheet": [
            "total_assets", "total_liabilities", "total_shareholders_equity",
            "current_assets", "current_liabilities", "cash_and_equivalents"
        ],
    },
}

FIELD_ALIASES: dict[str, list[str]] = {
    "total_revenue": ["revenue", "net_revenue", "net_sales", "sales", "total_sales"],
    "cost_of_goods_sold": ["cogs", "cost_of_sales", "cost_of_revenue"],
    "net_income": ["net_profit", "net_earnings", "profit_after_tax"],
    "total_assets": ["assets"],
    "total_liabilities": ["liabilities"],
    "total_shareholders_equity": ["shareholders_equity", "equity", "total_equity", "stockholders_equity"],
    "current_assets": ["total_current_assets"],
    "current_liabilities": ["total_current_liabilities"],
    "cash_and_equivalents": ["cash", "cash_and_cash_equivalents"],
    "accounts_receivable": ["ar", "trade_receivables", "receivables"],
    "accounts_payable": ["ap", "trade_payables", "payables"],
    "inventory": ["inventories", "stock"],
    "long_term_debt": ["lt_debt", "non_current_debt"],
    "interest_expense": ["interest_cost", "finance_cost"],
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_field_value(
    data: dict[str, Any],
    canonical_name: str,
) -> tuple[Any, str | None]:
    """
    Find a field value by canonical name or any of its aliases.
    
    Args:
        data: Dictionary of data fields
        canonical_name: The canonical field name to look for
        
    Returns:
        Tuple of (value, found_field_name) or (None, None) if not found
    """
    # Check canonical name first
    if canonical_name in data and data[canonical_name] is not None:
        return data[canonical_name], canonical_name
    
    # Check aliases
    aliases = FIELD_ALIASES.get(canonical_name, [])
    for alias in aliases:
        if alias in data and data[alias] is not None:
            return data[alias], alias
    
    # Check case-insensitive
    data_lower = {k.lower(): (v, k) for k, v in data.items()}
    if canonical_name.lower() in data_lower:
        value, original_key = data_lower[canonical_name.lower()]
        if value is not None:
            return value, original_key
    
    for alias in aliases:
        if alias.lower() in data_lower:
            value, original_key = data_lower[alias.lower()]
            if value is not None:
                return value, original_key
    
    return None, None


def is_numeric(value: Any) -> bool:
    """Check if a value is numeric."""
    if value is None:
        return False
    if isinstance(value, (int, float, Decimal)):
        return True
    if isinstance(value, str):
        try:
            Decimal(value)
            return True
        except:
            return False
    return False


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_income_statement_schema(
    data: dict[str, Any] | IncomeStatementData,
) -> ValidationResult:
    """
    Validate income statement data structure.
    
    Args:
        data: Income statement data (dict or model)
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult()
    
    # Convert model to dict if needed
    if isinstance(data, IncomeStatementData):
        data_dict = data.model_dump(by_alias=True)
    else:
        data_dict = data
    
    # Required fields for basic income statement
    required = ["total_revenue", "cost_of_goods_sold"]
    
    for field in required:
        value, found_name = find_field_value(data_dict, field)
        if value is None:
            result.add_error(
                field=field,
                message=f"Required field '{field}' is missing",
                expected="Numeric value",
                suggestion=f"Provide {field} value. Accepted aliases: {FIELD_ALIASES.get(field, [])}",
            )
        elif not is_numeric(value):
            result.add_error(
                field=found_name or field,
                message=f"Field '{found_name or field}' must be numeric",
                actual_value=value,
                expected="Numeric value (int, float, or Decimal)",
            )
    
    # Validate revenue > 0 (usually)
    revenue, _ = find_field_value(data_dict, "total_revenue")
    if revenue is not None and is_numeric(revenue):
        if Decimal(str(revenue)) < 0:
            result.add_warning(
                field="total_revenue",
                message="Revenue is negative, which is unusual",
                actual_value=float(revenue),
                expected="Typically positive value",
            )
    
    return result


def validate_balance_sheet_schema(
    data: dict[str, Any] | BalanceSheetData,
) -> ValidationResult:
    """
    Validate balance sheet data structure.
    
    Args:
        data: Balance sheet data (dict or model)
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult()
    
    # Convert model to dict if needed
    if isinstance(data, BalanceSheetData):
        data_dict = data.model_dump(by_alias=True)
    else:
        data_dict = data
    
    # Required fields for basic balance sheet
    required = ["cash_and_equivalents"]
    
    for field in required:
        value, found_name = find_field_value(data_dict, field)
        if value is None:
            result.add_error(
                field=field,
                message=f"Required field '{field}' is missing",
                expected="Numeric value",
                suggestion=f"Provide {field} value. Accepted aliases: {FIELD_ALIASES.get(field, [])}",
            )
        elif not is_numeric(value):
            result.add_error(
                field=found_name or field,
                message=f"Field '{found_name or field}' must be numeric",
                actual_value=value,
                expected="Numeric value",
            )
    
    # Validate cash >= 0
    cash, _ = find_field_value(data_dict, "cash_and_equivalents")
    if cash is not None and is_numeric(cash):
        if Decimal(str(cash)) < 0:
            result.add_error(
                field="cash_and_equivalents",
                message="Cash cannot be negative",
                actual_value=float(cash),
                expected="Non-negative value",
            )
    
    # Check balance sheet equation if all components present
    assets, _ = find_field_value(data_dict, "total_assets")
    liabilities, _ = find_field_value(data_dict, "total_liabilities")
    equity, _ = find_field_value(data_dict, "total_shareholders_equity")
    
    if all(v is not None and is_numeric(v) for v in [assets, liabilities, equity]):
        assets_dec = Decimal(str(assets))
        liab_equity = Decimal(str(liabilities)) + Decimal(str(equity))
        diff = abs(assets_dec - liab_equity)
        
        # Allow 1% tolerance
        if assets_dec != 0:
            tolerance = abs(assets_dec) * Decimal("0.01")
            if diff > tolerance:
                result.add_warning(
                    field="balance_sheet_equation",
                    message="Assets ≠ Liabilities + Equity",
                    actual_value=f"Assets={assets}, L+E={liab_equity}, Diff={diff}",
                    expected="Assets = Liabilities + Equity (within 1%)",
                )
    
    return result


def validate_cash_flow_schema(
    data: dict[str, Any] | CashFlowStatementData,
) -> ValidationResult:
    """
    Validate cash flow statement data structure.
    
    Args:
        data: Cash flow statement data (dict or model)
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult()
    
    # Convert model to dict if needed
    if isinstance(data, CashFlowStatementData):
        data_dict = data.model_dump(by_alias=True)
    else:
        data_dict = data
    
    # Required field
    if "net_income" not in data_dict or data_dict["net_income"] is None:
        result.add_error(
            field="net_income",
            message="Required field 'net_income' is missing from cash flow statement",
            expected="Numeric value matching income statement",
        )
    
    return result


def validate_financial_data_completeness(
    income_statement: dict[str, Any] | IncomeStatementData | None,
    balance_sheet: dict[str, Any] | BalanceSheetData | None,
    cash_flow: dict[str, Any] | CashFlowStatementData | None,
    analysis_type: str,
) -> ValidationResult:
    """
    Validate that all required data is present for the requested analysis type.
    
    Args:
        income_statement: Income statement data
        balance_sheet: Balance sheet data
        cash_flow: Cash flow statement data
        analysis_type: Type of analysis requested
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult()
    
    # Get required fields for this analysis type
    requirements = REQUIRED_FIELDS.get(analysis_type.lower(), {})
    
    if not requirements:
        result.add_warning(
            field="analysis_type",
            message=f"Unknown analysis type: {analysis_type}",
            suggestion="Using default comprehensive requirements",
        )
        requirements = REQUIRED_FIELDS.get("comprehensive", {})
    
    # Check income statement requirements
    if "income_statement" in requirements:
        if income_statement is None:
            result.add_error(
                field="income_statement",
                message=f"Income statement is required for {analysis_type} analysis",
                expected="Complete income statement data",
            )
        else:
            is_dict = income_statement.model_dump(by_alias=True) if isinstance(income_statement, IncomeStatementData) else income_statement
            for field in requirements["income_statement"]:
                value, _ = find_field_value(is_dict, field)
                if value is None:
                    result.add_error(
                        field=field,
                        message=f"Field '{field}' is required for {analysis_type} analysis",
                        expected="Numeric value",
                    )
    
    # Check balance sheet requirements
    if "balance_sheet" in requirements:
        if balance_sheet is None:
            result.add_error(
                field="balance_sheet",
                message=f"Balance sheet is required for {analysis_type} analysis",
                expected="Complete balance sheet data",
            )
        else:
            bs_dict = balance_sheet.model_dump(by_alias=True) if isinstance(balance_sheet, BalanceSheetData) else balance_sheet
            for field in requirements["balance_sheet"]:
                value, _ = find_field_value(bs_dict, field)
                if value is None:
                    result.add_warning(
                        field=field,
                        message=f"Field '{field}' is recommended for {analysis_type} analysis",
                        suggestion="Some calculations may be skipped",
                    )
    
    # Check cash flow requirements
    if "cash_flow" in requirements:
        if cash_flow is None:
            result.add_warning(
                field="cash_flow",
                message=f"Cash flow statement is recommended for {analysis_type} analysis",
                suggestion="Cash flow metrics will be skipped",
            )
    
    return result


def validate_statement_set(
    statement_set: FinancialStatementSet,
    analysis_type: str = "comprehensive",
) -> ValidationResult:
    """
    Validate a complete financial statement set.
    
    Args:
        statement_set: Complete set of financial statements
        analysis_type: Type of analysis to validate for
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult()
    
    # Validate individual statements
    is_result = validate_income_statement_schema(statement_set.income_statement)
    result.merge(is_result)
    
    bs_result = validate_balance_sheet_schema(statement_set.balance_sheet)
    result.merge(bs_result)
    
    if statement_set.cash_flow_statement:
        cf_result = validate_cash_flow_schema(statement_set.cash_flow_statement)
        result.merge(cf_result)
    
    # Validate completeness for analysis type
    completeness_result = validate_financial_data_completeness(
        statement_set.income_statement,
        statement_set.balance_sheet,
        statement_set.cash_flow_statement,
        analysis_type,
    )
    result.merge(completeness_result)
    
    return result

```

# finanalyst_tools/validation/plausibility.py
```py
# File: finanalyst_tools/validation/plausibility.py
"""
Plausibility checking for calculated financial metrics.

Verifies that calculated values fall within reasonable ranges
based on typical business metrics and industry norms.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from finanalyst_tools.config import PlausibilityRanges
from finanalyst_tools.models.validation import (
    ValidationSeverity,
    PlausibilityCheck,
    PlausibilityResult,
)
from finanalyst_tools.models.analysis_results import CalculationResult


def check_plausibility(
    metric_name: str,
    value: Decimal | float | None,
    custom_range: tuple[float, float] | None = None,
) -> PlausibilityCheck:
    """
    Check if a metric value is within plausible range.
    
    Args:
        metric_name: Name of the metric
        value: The calculated value
        custom_range: Optional custom range to use instead of default
        
    Returns:
        PlausibilityCheck result
    """
    if value is None:
        return PlausibilityCheck(
            metric_name=metric_name,
            value=Decimal("0"),
            plausible_range=(0, 0),
            is_plausible=True,
            assessment="not_calculated",
            severity=ValidationSeverity.INFO,
            message="Value not calculated",
        )
    
    dec_value = Decimal(str(value))
    float_value = float(dec_value)
    
    # Get range
    if custom_range:
        plausible_range = custom_range
    else:
        plausible_range = PlausibilityRanges.get_range(metric_name)
    
    if plausible_range is None:
        return PlausibilityCheck(
            metric_name=metric_name,
            value=dec_value,
            plausible_range=(float("-inf"), float("inf")),
            is_plausible=True,
            assessment="no_range_defined",
            severity=ValidationSeverity.INFO,
            message=f"No plausibility range defined for {metric_name}",
        )
    
    # Check against range
    min_val, max_val = plausible_range
    is_plausible = min_val <= float_value <= max_val
    
    if float_value < min_val:
        assessment = "below_range"
        message = f"{metric_name} of {float_value:.2f} is below typical range ({min_val:.1f} to {max_val:.1f})"
        severity = ValidationSeverity.WARNING
    elif float_value > max_val:
        assessment = "above_range"
        message = f"{metric_name} of {float_value:.2f} is above typical range ({min_val:.1f} to {max_val:.1f})"
        severity = ValidationSeverity.WARNING
    else:
        assessment = "within_range"
        message = f"{metric_name} of {float_value:.2f} is within typical range"
        severity = ValidationSeverity.INFO
    
    return PlausibilityCheck(
        metric_name=metric_name,
        value=dec_value,
        plausible_range=plausible_range,
        is_plausible=is_plausible,
        assessment=assessment,
        severity=severity,
        message=message,
    )


def check_all_plausibility(
    metrics: list[CalculationResult],
) -> PlausibilityResult:
    """
    Check plausibility for a list of calculation results.
    
    Args:
        metrics: List of calculation results to check
        
    Returns:
        PlausibilityResult with all check results
    """
    result = PlausibilityResult()
    
    for metric in metrics:
        check = check_plausibility(
            metric_name=metric.metric_name,
            value=metric.value,
            custom_range=metric.plausibility_range,
        )
        result.add_check(check)
        
        # Update the metric's plausibility status
        if not check.is_plausible:
            metric.is_plausible = False
            metric.add_warning(check.message)
    
    return result


class PlausibilityChecker:
    """
    Class-based plausibility checker with customization options.
    """
    
    def __init__(
        self,
        custom_ranges: dict[str, tuple[float, float]] | None = None,
        strict_mode: bool = False,
    ):
        """
        Initialize the plausibility checker.
        
        Args:
            custom_ranges: Dictionary of custom ranges by metric name
            strict_mode: If True, implausible values raise errors instead of warnings
        """
        self.custom_ranges = custom_ranges or {}
        self.strict_mode = strict_mode
    
    def get_range(self, metric_name: str) -> tuple[float, float] | None:
        """Get the range for a metric, checking custom ranges first."""
        if metric_name in self.custom_ranges:
            return self.custom_ranges[metric_name]
        return PlausibilityRanges.get_range(metric_name)
    
    def check(
        self,
        metric_name: str,
        value: Decimal | float | None,
    ) -> PlausibilityCheck:
        """Check a single metric."""
        custom_range = self.custom_ranges.get(metric_name)
        result = check_plausibility(metric_name, value, custom_range)
        
        # Upgrade to error if strict mode
        if self.strict_mode and not result.is_plausible:
            result.severity = ValidationSeverity.ERROR
        
        return result
    
    def check_all(
        self,
        metrics: list[CalculationResult],
    ) -> PlausibilityResult:
        """Check multiple metrics."""
        result = PlausibilityResult()
        
        for metric in metrics:
            check = self.check(metric.metric_name, metric.value)
            result.add_check(check)
            
            if not check.is_plausible:
                metric.is_plausible = False
                metric.add_warning(check.message)
        
        return result
    
    def add_custom_range(
        self,
        metric_name: str,
        min_value: float,
        max_value: float,
    ) -> None:
        """Add or update a custom range."""
        self.custom_ranges[metric_name] = (min_value, max_value)

```

# finanalyst_tools/orchestration/pipeline.py
```py
# finanalyst_tools/orchestration/pipeline.py
"""
Analysis pipeline implementing the mandatory 5-phase workflow.

Phases:
1. VALIDATE - Schema validation, completeness check
2. ANALYZE - Identify what to calculate based on data
3. CALCULATE - Execute calculations with audit trail
4. INTERPRET - Add context and insights
5. VERIFY - Pre-delivery checks

This ensures consistent, auditable analysis execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from decimal import Decimal

from finanalyst_tools.models.financial_statements import (
    FinancialStatementSet,
    IncomeStatementData,
    BalanceSheetData,
    CashFlowStatementData,
)
from finanalyst_tools.models.analysis_results import (
    MetricCategory,
    MetricCollection,
    ComprehensiveAnalysisResult,
    CalculationResult,
)
from finanalyst_tools.models.validation import (
    ValidationResult,
    ReconciliationResult,
    PlausibilityResult,
)
from finanalyst_tools.validation.schema_validator import (
    validate_statement_set,
    validate_financial_data_completeness,
)
from finanalyst_tools.validation.reconciliation import run_all_reconciliations
from finanalyst_tools.validation.plausibility import check_all_plausibility
from finanalyst_tools.calculations.profitability import calculate_all_profitability_metrics
from finanalyst_tools.calculations.liquidity import calculate_all_liquidity_metrics
from finanalyst_tools.orchestration.confidence_scorer import calculate_confidence_level
from finanalyst_tools.exceptions import FinAnalystError


class AnalysisPhase(str, Enum):
    """Phases of the analysis pipeline."""
    VALIDATE = "validate"
    ANALYZE = "analyze"
    CALCULATE = "calculate"
    INTERPRET = "interpret"
    VERIFY = "verify"


@dataclass
class AnalysisRequest:
    """
    Request for financial analysis.
    
    Attributes:
        statement_set: Financial statements to analyze
        prior_statement_set: Prior period statements (optional)
        analysis_type: Type of analysis requested
        include_trends: Whether to include trend analysis
        currency: Currency for reporting
    """
    statement_set: FinancialStatementSet
    prior_statement_set: FinancialStatementSet | None = None
    analysis_type: str = "comprehensive"
    include_trends: bool = False
    currency: str = "SGD"


@dataclass
class PipelineState:
    """
    Internal state of the pipeline during execution.
    """
    current_phase: AnalysisPhase = AnalysisPhase.VALIDATE
    validation_result: ValidationResult | None = None
    reconciliation_result: ReconciliationResult | None = None
    plausibility_result: PlausibilityResult | None = None
    metric_collections: list[MetricCollection] = field(default_factory=list)
    all_metrics: list[CalculationResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    phase_completed: dict[AnalysisPhase, bool] = field(default_factory=dict)


class AnalysisPipeline:
    """
    Pipeline for executing financial analysis.
    
    Implements the mandatory 5-phase workflow:
    REQUEST → [1.VALIDATE] → [2.ANALYZE] → [3.CALCULATE] → [4.INTERPRET] → [5.VERIFY] → DELIVER
    """
    
    def __init__(self):
        """Initialize the pipeline."""
        self.state: PipelineState | None = None

    def _require_phase(self, phase: AnalysisPhase) -> None:
        if self.state is None:
            raise FinAnalystError("Pipeline has not been initialized")

        if not self.state.phase_completed.get(phase, False):
            raise FinAnalystError(f"Phase '{phase.value}' is required before continuing")

    def _set_phase_completed(self, phase: AnalysisPhase) -> None:
        if self.state is None:
            raise FinAnalystError("Pipeline has not been initialized")
        self.state.phase_completed[phase] = True
    
    def execute(self, request: AnalysisRequest) -> ComprehensiveAnalysisResult:
        """
        Execute the full analysis pipeline.
        
        Args:
            request: Analysis request with financial data
            
        Returns:
            ComprehensiveAnalysisResult with all analysis outputs
        """
        # Initialize state
        self.state = PipelineState()
        
        try:
            # Phase 1: VALIDATE
            self._phase_validate(request)
            if not self.state.validation_result or not self.state.validation_result.can_proceed:
                return self._create_error_result(request, "Validation failed")

            # Phase 2: ANALYZE
            analysis_plan = self._phase_analyze(request)

            # Phase 3: CALCULATE
            self._phase_calculate(request, analysis_plan)

            # Phase 4: INTERPRET
            self._phase_interpret(request)

            # Phase 5: VERIFY
            self._phase_verify(request)

            # Create final result
            return self._create_result(request)
        except Exception as e:
            if self.state is not None:
                self.state.errors.append(f"Pipeline error in phase '{self.state.current_phase.value}': {str(e)}")
            return self._create_error_result(request, f"Pipeline execution failed: {str(e)}")
    
    def _phase_validate(self, request: AnalysisRequest) -> None:
        """
        Phase 1: VALIDATE
        
        - Schema validation
        - Data completeness check
        - Cross-statement reconciliation
        """
        self.state.current_phase = AnalysisPhase.VALIDATE
        
        # Schema validation
        validation = validate_statement_set(
            request.statement_set,
            request.analysis_type,
        )
        self.state.validation_result = validation
        
        if not validation.can_proceed:
            self.state.errors.append("Schema validation failed")
            self._set_phase_completed(AnalysisPhase.VALIDATE)
            return
        
        # Reconciliation (if cash flow available)
        prior_bs = None
        if request.prior_statement_set:
            prior_bs = request.prior_statement_set.balance_sheet
        
        reconciliation = run_all_reconciliations(
            request.statement_set,
            prior_balance_sheet=prior_bs,
        )
        self.state.reconciliation_result = reconciliation
        
        if not reconciliation.all_passed:
            for check in reconciliation.failed_checks:
                self.state.warnings.append(f"Reconciliation: {check.message}")

        self._set_phase_completed(AnalysisPhase.VALIDATE)
    
    def _phase_analyze(self, request: AnalysisRequest) -> dict[str, bool]:
        """
        Phase 2: ANALYZE
        
        Determine what calculations to perform based on:
        - Analysis type requested
        - Data available
        
        Returns:
            Dictionary of metric categories to calculate
        """
        self._require_phase(AnalysisPhase.VALIDATE)
        if not self.state.validation_result or not self.state.validation_result.can_proceed:
            raise FinAnalystError("Cannot analyze because validation did not pass")

        self.state.current_phase = AnalysisPhase.ANALYZE
        
        analysis_plan = {
            "profitability": False,
            "liquidity": False,
            "solvency": False,
            "efficiency": False,
        }
        
        analysis_type = request.analysis_type.lower()
        
        if analysis_type in ("profitability", "comprehensive"):
            analysis_plan["profitability"] = True
        
        if analysis_type in ("liquidity", "comprehensive"):
            analysis_plan["liquidity"] = True
        
        if analysis_type in ("solvency", "comprehensive"):
            analysis_plan["solvency"] = True
        
        if analysis_type in ("efficiency", "comprehensive"):
            analysis_plan["efficiency"] = True
        
        self._set_phase_completed(AnalysisPhase.ANALYZE)
        return analysis_plan
    
    def _phase_calculate(
        self,
        request: AnalysisRequest,
        analysis_plan: dict[str, bool],
    ) -> None:
        """
        Phase 3: CALCULATE
        
        Execute all planned calculations.
        """
        self._require_phase(AnalysisPhase.ANALYZE)
        self.state.current_phase = AnalysisPhase.CALCULATE
        
        prior_bs = None
        if request.prior_statement_set:
            prior_bs = request.prior_statement_set.balance_sheet
        
        # Profitability metrics
        if analysis_plan.get("profitability"):
            profitability = calculate_all_profitability_metrics(
                income_statement=request.statement_set.income_statement,
                balance_sheet=request.statement_set.balance_sheet,
                prior_balance_sheet=prior_bs,
            )
            self.state.metric_collections.append(profitability)
            self.state.all_metrics.extend(profitability.metrics)
        
        # Liquidity metrics
        if analysis_plan.get("liquidity"):
            liquidity = calculate_all_liquidity_metrics(
                balance_sheet=request.statement_set.balance_sheet,
            )
            self.state.metric_collections.append(liquidity)
            self.state.all_metrics.extend(liquidity.metrics)
        
        # Note: Solvency and Efficiency calculations would be added in Phase 2
        
        self._set_phase_completed(AnalysisPhase.CALCULATE)
    
    def _phase_interpret(self, request: AnalysisRequest) -> None:
        """
        Phase 4: INTERPRET
        
        Add context and insights to calculated metrics.
        """
        self._require_phase(AnalysisPhase.CALCULATE)
        self.state.current_phase = AnalysisPhase.INTERPRET
        
        # Plausibility checks on all metrics
        plausibility = check_all_plausibility(self.state.all_metrics)
        self.state.plausibility_result = plausibility
        
        # Add warnings for implausible values
        for check in plausibility.implausible_checks:
            self.state.warnings.append(f"Plausibility: {check.message}")
        
        self._set_phase_completed(AnalysisPhase.INTERPRET)
    
    def _phase_verify(self, request: AnalysisRequest) -> None:
        """
        Phase 5: VERIFY
        
        Pre-delivery checks:
        - Ensure all requested calculations completed
        - Verify no critical errors
        - Final quality check
        """
        self._require_phase(AnalysisPhase.INTERPRET)
        self.state.current_phase = AnalysisPhase.VERIFY
        
        # Check that calculations were performed
        if not self.state.metric_collections:
            self.state.warnings.append("No metrics were calculated")
        
        # Check for any uncalculable metrics
        uncalculable = [m for m in self.state.all_metrics if m.value is None]
        if uncalculable:
            for m in uncalculable:
                self.state.warnings.append(f"Could not calculate: {m.metric_name}")
        
        self._set_phase_completed(AnalysisPhase.VERIFY)
    
    def _create_result(self, request: AnalysisRequest) -> ComprehensiveAnalysisResult:
        """Create the final analysis result."""
        
        # Calculate confidence
        data_completeness = 1.0
        if self.state.validation_result:
            total_issues = self.state.validation_result.total_issue_count
            data_completeness = max(0.0, 1.0 - (total_issues * 0.1))
        
        confidence = calculate_confidence_level(
            validation_result=self.state.validation_result,
            plausibility_result=self.state.plausibility_result,
            reconciliation_result=self.state.reconciliation_result,
            data_completeness=data_completeness,
        )
        
        # Build result
        result = ComprehensiveAnalysisResult(
            analysis_type=request.analysis_type,
            period=str(request.statement_set.period),
            currency=request.currency,
            metric_collections=self.state.metric_collections,
            confidence=confidence,
        )

        uncalculable_metrics = [m.metric_name for m in self.state.all_metrics if m.value is None]
        result.is_partial = len(uncalculable_metrics) > 0
        result.uncalculable_metrics = uncalculable_metrics
        result.pipeline_errors = list(self.state.errors)
        result.pipeline_warnings = list(self.state.warnings)
        
        # Add validation summary
        if self.state.validation_result:
            result.validation_summary = self.state.validation_result.to_dict()
        
        # Add reconciliation summary
        if self.state.reconciliation_result:
            result.reconciliation_summary = self.state.reconciliation_result.to_dict()
        
        # Add recommendations based on findings
        result.recommendations = self._generate_recommendations()
        
        return result
    
    def _create_error_result(
        self,
        request: AnalysisRequest,
        error_message: str,
    ) -> ComprehensiveAnalysisResult:
        """Create an error result when pipeline fails."""
        result = ComprehensiveAnalysisResult(
            analysis_type=request.analysis_type,
            period=str(request.statement_set.period),
            currency=request.currency,
        )

        if self.state is not None:
            result.pipeline_errors = list(self.state.errors)
            result.pipeline_warnings = list(self.state.warnings)

            if self.state.validation_result:
                result.validation_summary = self.state.validation_result.to_dict()

            if self.state.reconciliation_result:
                result.reconciliation_summary = self.state.reconciliation_result.to_dict()

        if error_message:
            result.pipeline_errors.append(error_message)

        return result
    
    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on analysis findings."""
        recommendations = []
        
        # Based on profitability
        for collection in self.state.metric_collections:
            if collection.category == MetricCategory.PROFITABILITY:
                npm = collection.get_metric("Net Profit Margin")
                if npm and npm.value is not None:
                    if npm.value < Decimal("5"):
                        recommendations.append(
                            "Net profit margin is low (<5%). Consider reviewing cost structure."
                        )
                    elif npm.value > Decimal("20"):
                        recommendations.append(
                            "Strong net profit margin (>20%). Consider reinvestment opportunities."
                        )
        
        # Based on liquidity
        for collection in self.state.metric_collections:
            if collection.category == MetricCategory.LIQUIDITY:
                cr = collection.get_metric("Current Ratio")
                if cr and cr.value is not None:
                    if cr.value < Decimal("1"):
                        recommendations.append(
                            "Current ratio below 1.0 indicates liquidity risk. Review working capital management."
                        )
        
        # Default recommendation if none generated
        if not recommendations:
            recommendations.append(
                "Financial metrics are within normal ranges. Continue monitoring key indicators."
            )
        
        return recommendations

```

# finanalyst_tools/orchestration/__init__.py
```py
# finanalyst_tools/orchestration/__init__.py
"""
Orchestration layer for FinAnalyst-Pro Agent Tools.

This package provides:
- Analysis pipeline with 5-phase workflow
- Confidence scoring for analysis results
- Report generation with mandatory template
"""

from finanalyst_tools.orchestration.pipeline import (
    AnalysisPipeline,
    AnalysisRequest,
    AnalysisPhase,
)

from finanalyst_tools.orchestration.confidence_scorer import (
    calculate_confidence_level,
    ConfidenceScorer,
)

from finanalyst_tools.orchestration.report_generator import (
    generate_financial_report,
    ReportGenerator,
    ReportFormat,
)


__all__ = [
    # Pipeline
    "AnalysisPipeline",
    "AnalysisRequest",
    "AnalysisPhase",
    # Confidence
    "calculate_confidence_level",
    "ConfidenceScorer",
    # Reporting
    "generate_financial_report",
    "ReportGenerator",
    "ReportFormat",
]

```

# finanalyst_tools/orchestration/report_generator.py
```py
# finanalyst_tools/orchestration/report_generator.py
"""
Report generator for financial analysis results.

Generates reports in the mandatory format specified in system prompt:
- Financial Analysis Report header
- Data Validation Summary
- Key Findings
- Detailed Metrics
- Calculation Audit Trail
- Recommendations
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
from datetime import datetime

from finanalyst_tools.models.analysis_results import (
    ComprehensiveAnalysisResult,
    MetricCollection,
    CalculationResult,
    ConfidenceAssessment,
)
from finanalyst_tools.utils.formatting import (
    format_currency,
    format_percentage,
    format_ratio,
    format_markdown_table,
    format_value_with_unit,
)


class ReportFormat(str, Enum):
    """Available report formats."""
    MARKDOWN = "markdown"
    JSON = "json"
    TEXT = "text"


def generate_financial_report(
    analysis_result: ComprehensiveAnalysisResult,
    format: ReportFormat = ReportFormat.MARKDOWN,
    include_audit_trail: bool = True,
) -> str:
    """
    Generate a financial analysis report.
    
    Args:
        analysis_result: Complete analysis result
        format: Output format
        include_audit_trail: Whether to include detailed calculation steps
        
    Returns:
        Formatted report string
    """
    generator = ReportGenerator(
        include_audit_trail=include_audit_trail,
    )
    
    if format == ReportFormat.MARKDOWN:
        return generator.generate_markdown(analysis_result)
    elif format == ReportFormat.JSON:
        return analysis_result.to_json()
    else:
        return generator.generate_text(analysis_result)


class ReportGenerator:
    """
    Generator for financial analysis reports.
    """
    
    def __init__(
        self,
        include_audit_trail: bool = True,
        include_warnings: bool = True,
        company_name: str | None = None,
    ):
        """
        Initialize the report generator.
        
        Args:
            include_audit_trail: Whether to include calculation steps
            include_warnings: Whether to include warning messages
            company_name: Optional company name for report header
        """
        self.include_audit_trail = include_audit_trail
        self.include_warnings = include_warnings
        self.company_name = company_name
    
    def generate_markdown(
        self,
        result: ComprehensiveAnalysisResult,
    ) -> str:
        """
        Generate report in Markdown format.
        
        Follows the mandatory template from system prompt.
        """
        sections = []
        
        # Header
        sections.append(self._generate_header(result))
        
        # Executive Summary with Confidence
        sections.append(self._generate_summary(result))
        
        # Data Validation Summary
        sections.append(self._generate_validation_section(result))
        
        # Key Findings
        sections.append(self._generate_key_findings(result))
        
        # Detailed Metrics
        sections.append(self._generate_metrics_section(result))
        
        # Audit Trail (optional)
        if self.include_audit_trail:
            sections.append(self._generate_audit_trail(result))
        
        # Recommendations
        sections.append(self._generate_recommendations(result))
        
        # Footer
        sections.append(self._generate_footer())
        
        return "\n\n".join(sections)
    
    def generate_text(
        self,
        result: ComprehensiveAnalysisResult,
    ) -> str:
        """Generate report in plain text format."""
        # Simplified version of markdown
        md = self.generate_markdown(result)
        # Remove markdown formatting
        text = md.replace("# ", "").replace("## ", "").replace("### ", "")
        text = text.replace("**", "").replace("*", "")
        text = text.replace("|", " ")
        return text
    
    def _generate_header(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate report header."""
        lines = ["# Financial Analysis Report"]
        
        if self.company_name:
            lines.append(f"\n**Company**: {self.company_name}")
        
        lines.append(f"**Analysis Type**: {result.analysis_type.title()}")
        lines.append(f"**Data Period**: {result.period}")
        lines.append(f"**Currency**: {result.currency}")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        return "\n".join(lines)
    
    def _generate_summary(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate executive summary with confidence level."""
        lines = ["## Executive Summary"]
        
        # Confidence level
        if result.confidence:
            lines.append(f"\n**Confidence Level**: {result.confidence.to_display()}")
        
        # Quick stats
        lines.append(f"\n**Metrics Calculated**: {result.total_metrics}")

        if result.is_partial:
            lines.append("**Result Status**: ⚠️ Partial")
            if result.uncalculable_metrics:
                lines.append(f"**Uncalculable Metrics**: {len(result.uncalculable_metrics)}")

        # Count warnings
        warning_count = len(result.all_warnings)
        if warning_count > 0:
            lines.append(f"**Warnings**: {warning_count}")

        if result.pipeline_warnings:
            lines.append(f"**Pipeline Warnings**: {len(result.pipeline_warnings)}")

        if result.pipeline_errors:
            lines.append(f"**Pipeline Errors**: {len(result.pipeline_errors)}")
        
        return "\n".join(lines)
    
    def _generate_validation_section(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate data validation summary."""
        lines = ["## 1. Data Validation Summary"]
        
        if result.validation_summary:
            is_valid = result.validation_summary.get("is_valid", True)
            error_count = result.validation_summary.get("error_count", 0)
            warning_count = result.validation_summary.get("warning_count", 0)
            
            status = "✅ Passed" if is_valid else "❌ Failed"
            lines.append(f"\n**Status**: {status}")
            
            if error_count > 0:
                lines.append(f"- Errors: {error_count}")
            if warning_count > 0:
                lines.append(f"- Warnings: {warning_count}")
            
            if is_valid and error_count == 0 and warning_count == 0:
                lines.append("- All validation checks passed")
        else:
            lines.append("\n- No validation summary available")

        return "\n".join(lines)

    def _generate_key_findings(self, result: ComprehensiveAnalysisResult) -> str:
        lines = ["## 2. Key Findings"]

        if not result.metric_collections:
            lines.append("\n- No metrics calculated")
            return "\n".join(lines)

        lines.append("")
        shown = 0
        for collection in result.metric_collections:
            if not collection.metrics:
                continue

            lines.append(f"### {collection.category.value.title()}")
            for metric in collection.metrics:
                plausible = "✅" if metric.is_plausible else "⚠️"
                warning_count = len(metric.warnings)
                warning_suffix = f" ({warning_count} warning(s))" if warning_count > 0 else ""
                lines.append(f"- {plausible} **{metric.metric_name}**: {metric.formatted_value}{warning_suffix}")
                shown += 1
                if shown >= 8:
                    break

            if shown >= 8:
                break

        if shown == 0:
            lines.append("\n- No metrics calculated")

        return "\n".join(lines)

    def _generate_metrics_section(self, result: ComprehensiveAnalysisResult) -> str:
        lines = ["## 3. Detailed Metrics"]

        if not result.metric_collections:
            lines.append("\n- No metrics available")
            return "\n".join(lines)

        for collection in result.metric_collections:
            lines.append("")
            lines.append(collection.to_table())

        return "\n".join(lines)

    def _generate_audit_trail(self, result: ComprehensiveAnalysisResult) -> str:
        lines = ["## 4. Calculation Audit Trail"]

        if not result.all_metrics:
            lines.append("\n- No calculations available")
            return "\n".join(lines)

        for metric in result.all_metrics:
            lines.append("")
            lines.append(f"### {metric.metric_name}")
            if metric.calculation_steps:
                lines.append("**Calculation Steps**:")
                for step in metric.calculation_steps:
                    lines.append(f"- {step}")
            if metric.inputs:
                lines.append("**Inputs Used**:")
                for k, v in metric.inputs.items():
                    lines.append(f"- {k}: {v}")
            if metric.warnings:
                lines.append("**Warnings**:")
                for w in metric.warnings:
                    lines.append(f"- {w}")

        return "\n".join(lines)

    def _generate_recommendations(self, result: ComprehensiveAnalysisResult) -> str:
        lines = ["## 5. Recommendations"]

        if result.recommendations:
            lines.append("")
            for rec in result.recommendations:
                lines.append(f"- {rec}")
            return "\n".join(lines)

        lines.append("\n- No recommendations available")
        return "\n".join(lines)

    def _generate_footer(self) -> str:
        return f"---\nReport generated by FinAnalyst-Pro on {datetime.now().strftime('%Y-%m-%d %H:%M')}"

```

