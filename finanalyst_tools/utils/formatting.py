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
