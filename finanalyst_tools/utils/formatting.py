# finanalyst_tools/utils/formatting.py
"""
Number, currency, and output formatting utilities.

This module provides consistent formatting functions for:
- Currency values with proper symbols and locale conventions
- Percentages with configurable precision
- Ratios and financial metrics
- Trend indicators and status symbols
- Markdown tables for reports

All functions handle None/invalid inputs gracefully,
returning "N/A" or appropriate placeholder text.

Author: FinAnalyst-Pro Team
Version: 1.0.0
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Sequence

from finanalyst_tools.config import (
    DECIMAL_PLACES,
    CURRENCY_SYMBOLS,
    ZERO_DECIMAL_CURRENCIES,
    LARGE_NUMBER_SUFFIXES,
    TREND_SYMBOLS,
    STATUS_SYMBOLS,
    DEFAULT_CURRENCY,
)
from finanalyst_tools.utils.math_ops import to_decimal, is_effectively_zero


# ============================================================================
# NUMBER FORMATTING
# ============================================================================

def format_number(
    value: int | float | Decimal | None,
    precision: int = 2,
    use_thousands_sep: bool = True,
    na_value: str = "N/A",
) -> str:
    """
    Format a number with optional thousands separator.
    
    Args:
        value: Number to format
        precision: Decimal places
        use_thousands_sep: Whether to include thousand separators
        na_value: String to return for None values
        
    Returns:
        Formatted number string
        
    Examples:
        >>> format_number(1234567.89)
        '1,234,567.89'
        >>> format_number(None)
        'N/A'
    """
    if value is None:
        return na_value
    
    try:
        decimal_value = to_decimal(value)
        
        if precision == 0:
            formatted = f"{int(decimal_value):,}" if use_thousands_sep else str(int(decimal_value))
        else:
            if use_thousands_sep:
                formatted = f"{float(decimal_value):,.{precision}f}"
            else:
                formatted = f"{float(decimal_value):.{precision}f}"
        
        return formatted
    except (ValueError, TypeError, InvalidOperation):
        return na_value


def format_large_number(
    value: int | float | Decimal | None,
    precision: int = 1,
    na_value: str = "N/A",
) -> str:
    """
    Format large numbers with K/M/B/T suffixes.
    
    Args:
        value: Number to format
        precision: Decimal places after suffix
        na_value: String to return for None values
        
    Returns:
        Formatted string with suffix
        
    Examples:
        >>> format_large_number(1500000)
        '1.5M'
        >>> format_large_number(2500000000)
        '2.5B'
    """
    if value is None:
        return na_value
    
    try:
        num = float(to_decimal(value))
        
        # Handle negative numbers
        sign = "-" if num < 0 else ""
        num = abs(num)
        
        # Find appropriate suffix
        for threshold, suffix in LARGE_NUMBER_SUFFIXES:
            if num >= threshold:
                formatted = f"{sign}{num / threshold:.{precision}f}{suffix}"
                return formatted
        
        # No suffix needed for small numbers
        return format_number(value, precision=precision)
    except (ValueError, TypeError):
        return na_value


# ============================================================================
# CURRENCY FORMATTING
# ============================================================================

def format_currency(
    value: int | float | Decimal | None,
    currency_code: str = DEFAULT_CURRENCY,
    precision: int | None = None,
    show_symbol: bool = True,
    show_code: bool = False,
    na_value: str = "N/A",
) -> str:
    """
    Format a monetary value with currency symbol.
    
    Args:
        value: Amount to format
        currency_code: ISO currency code (default: SGD)
        precision: Decimal places (auto-detected based on currency if None)
        show_symbol: Whether to show currency symbol
        show_code: Whether to show currency code (e.g., "SGD")
        na_value: String to return for None values
        
    Returns:
        Formatted currency string
        
    Examples:
        >>> format_currency(1234.56)
        'S$1,234.56'
        >>> format_currency(1234.56, "USD")
        '$1,234.56'
        >>> format_currency(1000000, show_code=True)
        'S$1,000,000.00 SGD'
    """
    if value is None:
        return na_value
    
    try:
        # Determine precision based on currency
        if precision is None:
            precision = 0 if currency_code in ZERO_DECIMAL_CURRENCIES else DECIMAL_PLACES["currency"]
        
        decimal_value = to_decimal(value)
        
        # Format the number
        formatted_number = format_number(decimal_value, precision=precision)
        
        # Get symbol
        symbol = CURRENCY_SYMBOLS.get(currency_code, currency_code)
        
        # Build result
        if show_symbol:
            result = f"{symbol}{formatted_number}"
        else:
            result = formatted_number
        
        if show_code:
            result = f"{result} {currency_code}"
        
        return result
    except (ValueError, TypeError):
        return na_value


def format_sgd(
    value: int | float | Decimal | None,
    precision: int = 2,
    show_symbol: bool = True,
    na_value: str = "N/A",
) -> str:
    """
    Format a value as Singapore Dollars.
    
    Convenience function for SGD formatting.
    
    Args:
        value: Amount to format
        precision: Decimal places
        show_symbol: Whether to show S$ symbol
        na_value: String to return for None values
        
    Returns:
        Formatted SGD string
        
    Examples:
        >>> format_sgd(1234.56)
        'S$1,234.56'
    """
    return format_currency(value, "SGD", precision, show_symbol, na_value=na_value)


# ============================================================================
# PERCENTAGE FORMATTING
# ============================================================================

def format_percentage(
    value: int | float | Decimal | None,
    precision: int = 2,
    show_symbol: bool = True,
    multiply_by_100: bool = False,
    na_value: str = "N/A",
) -> str:
    """
    Format a value as a percentage.
    
    Args:
        value: Percentage value
        precision: Decimal places
        show_symbol: Whether to include % symbol
        multiply_by_100: If True, multiply value by 100 (for decimal ratios)
        na_value: String to return for None values
        
    Returns:
        Formatted percentage string
        
    Examples:
        >>> format_percentage(25.5)
        '25.50%'
        >>> format_percentage(0.255, multiply_by_100=True)
        '25.50%'
    """
    if value is None:
        return na_value
    
    try:
        decimal_value = to_decimal(value)
        
        if multiply_by_100:
            decimal_value = decimal_value * Decimal("100")
        
        formatted = format_number(decimal_value, precision=precision, use_thousands_sep=False)
        
        if show_symbol:
            return f"{formatted}%"
        return formatted
    except (ValueError, TypeError):
        return na_value


def format_change(
    value: int | float | Decimal | None,
    precision: int = 2,
    show_sign: bool = True,
    is_percentage: bool = True,
    na_value: str = "N/A",
) -> str:
    """
    Format a change value with +/- sign.
    
    Args:
        value: Change value
        precision: Decimal places
        show_sign: Whether to show + for positive values
        is_percentage: Whether to append % symbol
        na_value: String to return for None values
        
    Returns:
        Formatted change string
        
    Examples:
        >>> format_change(10.5)
        '+10.50%'
        >>> format_change(-5.2)
        '-5.20%'
    """
    if value is None:
        return na_value
    
    try:
        decimal_value = to_decimal(value)
        
        # Determine sign
        if decimal_value > 0 and show_sign:
            sign = "+"
        elif decimal_value < 0:
            sign = ""  # Negative sign included in number
        else:
            sign = ""
        
        formatted = format_number(decimal_value, precision=precision, use_thousands_sep=False)
        
        if is_percentage:
            return f"{sign}{formatted}%"
        return f"{sign}{formatted}"
    except (ValueError, TypeError):
        return na_value


# ============================================================================
# RATIO FORMATTING
# ============================================================================

def format_ratio(
    value: int | float | Decimal | None,
    precision: int = 2,
    suffix: str = "x",
    na_value: str = "N/A",
) -> str:
    """
    Format a ratio value.
    
    Args:
        value: Ratio value
        precision: Decimal places
        suffix: Suffix to append (e.g., "x" for "1.5x")
        na_value: String to return for None values
        
    Returns:
        Formatted ratio string
        
    Examples:
        >>> format_ratio(1.5)
        '1.50x'
        >>> format_ratio(2.345, precision=1)
        '2.3x'
    """
    if value is None:
        return na_value
    
    try:
        formatted = format_number(value, precision=precision, use_thousands_sep=False)
        return f"{formatted}{suffix}"
    except (ValueError, TypeError):
        return na_value


def format_days(
    value: int | float | Decimal | None,
    precision: int = 0,
    na_value: str = "N/A",
) -> str:
    """
    Format a value as days.
    
    Args:
        value: Number of days
        precision: Decimal places (usually 0)
        na_value: String to return for None values
        
    Returns:
        Formatted days string
        
    Examples:
        >>> format_days(45)
        '45 days'
    """
    if value is None:
        return na_value
    
    try:
        decimal_value = to_decimal(value)
        formatted = format_number(decimal_value, precision=precision, use_thousands_sep=False)
        
        # Handle singular/plural
        if abs(float(decimal_value)) == 1:
            return f"{formatted} day"
        return f"{formatted} days"
    except (ValueError, TypeError):
        return na_value


# ============================================================================
# VALUE WITH UNIT FORMATTING
# ============================================================================

def format_value_with_unit(
    value: int | float | Decimal | None,
    unit: str,
    precision: int = 2,
    currency_code: str = DEFAULT_CURRENCY,
    na_value: str = "N/A",
) -> str:
    """
    Format a value with its appropriate unit.
    
    Args:
        value: Value to format
        unit: Unit type ("percentage", "ratio", "currency", "days", "count")
        precision: Decimal places
        currency_code: Currency code for currency unit
        na_value: String to return for None values
        
    Returns:
        Formatted value with unit
        
    Examples:
        >>> format_value_with_unit(25.5, "percentage")
        '25.50%'
        >>> format_value_with_unit(1.5, "ratio")
        '1.50x'
    """
    if value is None:
        return na_value
    
    unit_lower = unit.lower()
    
    if unit_lower in ("percentage", "percent", "%"):
        return format_percentage(value, precision=precision)
    elif unit_lower in ("ratio", "x", "times"):
        return format_ratio(value, precision=precision)
    elif unit_lower in ("currency", "money", "amount"):
        return format_currency(value, currency_code=currency_code, precision=precision)
    elif unit_lower in ("days", "day"):
        return format_days(value, precision=precision)
    elif unit_lower in ("count", "number", "integer"):
        return format_number(value, precision=0)
    else:
        return format_number(value, precision=precision)


# ============================================================================
# INDICATOR FORMATTING
# ============================================================================

def format_trend_indicator(
    direction: str,
    include_text: bool = False,
) -> str:
    """
    Get a trend indicator symbol.
    
    Args:
        direction: Trend direction ("increasing", "decreasing", "stable", "volatile")
        include_text: Whether to include text after symbol
        
    Returns:
        Trend indicator string
        
    Examples:
        >>> format_trend_indicator("increasing")
        '↑'
        >>> format_trend_indicator("decreasing", include_text=True)
        '↓ Decreasing'
    """
    direction_lower = direction.lower()
    symbol = TREND_SYMBOLS.get(direction_lower, "")
    
    if include_text and symbol:
        return f"{symbol} {direction.title()}"
    return symbol


def format_status_indicator(
    status: str,
    include_text: bool = False,
) -> str:
    """
    Get a status indicator symbol.
    
    Args:
        status: Status ("good", "warning", "error", "info", "pass", "fail")
        include_text: Whether to include text after symbol
        
    Returns:
        Status indicator string
        
    Examples:
        >>> format_status_indicator("good")
        '✅'
        >>> format_status_indicator("warning", include_text=True)
        '⚠️ Warning'
    """
    status_lower = status.lower()
    symbol = STATUS_SYMBOLS.get(status_lower, "")
    
    if include_text and symbol:
        return f"{symbol} {status.title()}"
    return symbol


def format_plausibility_indicator(
    is_plausible: bool,
    include_text: bool = False,
) -> str:
    """
    Get a plausibility indicator.
    
    Args:
        is_plausible: Whether value is plausible
        include_text: Whether to include text
        
    Returns:
        Plausibility indicator string
    """
    if is_plausible:
        return format_status_indicator("pass", include_text)
    return format_status_indicator("warning", include_text)


# ============================================================================
# MARKDOWN TABLE FORMATTING
# ============================================================================

def format_markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    alignment: Sequence[str] | None = None,
) -> str:
    """
    Generate a Markdown table.
    
    Args:
        headers: Column headers
        rows: Table rows (each row is a sequence of values)
        alignment: Column alignments ("left", "center", "right") for each column
        
    Returns:
        Markdown table string
        
    Examples:
        >>> print(format_markdown_table(
        ...     ["Metric", "Value"],
        ...     [["Revenue", "$100M"], ["Profit", "$20M"]]
        ... ))
        | Metric | Value |
        |--------|-------|
        | Revenue | $100M |
        | Profit | $20M |
    """
    if not headers or not rows:
        return ""
    
    # Convert all values to strings
    str_headers = [str(h) for h in headers]
    str_rows = [[str(cell) for cell in row] for row in rows]
    
    # Calculate column widths
    num_cols = len(headers)
    col_widths = [len(h) for h in str_headers]
    
    for row in str_rows:
        for i, cell in enumerate(row[:num_cols]):
            col_widths[i] = max(col_widths[i], len(cell))
    
    # Pad cells to column width
    def pad_cell(text: str, width: int, align: str = "left") -> str:
        if align == "right":
            return text.rjust(width)
        elif align == "center":
            return text.center(width)
        return text.ljust(width)
    
    # Default alignment
    if alignment is None:
        alignment = ["left"] * num_cols
    else:
        alignment = list(alignment) + ["left"] * (num_cols - len(alignment))
    
    # Build header row
    header_cells = [pad_cell(h, col_widths[i], alignment[i]) for i, h in enumerate(str_headers)]
    header_line = "| " + " | ".join(header_cells) + " |"
    
    # Build separator row
    def separator_cell(width: int, align: str) -> str:
        if align == "right":
            return "-" * (width - 1) + ":"
        elif align == "center":
            return ":" + "-" * (width - 2) + ":"
        return "-" * width
    
    separator_cells = [separator_cell(col_widths[i], alignment[i]) for i in range(num_cols)]
    separator_line = "|" + "|".join(separator_cells) + "|"
    
    # Build data rows
    data_lines = []
    for row in str_rows:
        # Pad row to have correct number of columns
        padded_row = list(row) + [""] * (num_cols - len(row))
        row_cells = [pad_cell(padded_row[i], col_widths[i], alignment[i]) for i in range(num_cols)]
        data_lines.append("| " + " | ".join(row_cells) + " |")
    
    # Combine all lines
    return "\n".join([header_line, separator_line] + data_lines)


def format_key_value_list(
    items: dict[str, Any],
    prefix: str = "- ",
) -> str:
    """
    Format a dictionary as a key-value list.
    
    Args:
        items: Dictionary of key-value pairs
        prefix: Prefix for each line
        
    Returns:
        Formatted key-value list string
        
    Examples:
        >>> print(format_key_value_list({"Name": "Company A", "Revenue": "$100M"}))
        - **Name**: Company A
        - **Revenue**: $100M
    """
    lines = []
    for key, value in items.items():
        lines.append(f"{prefix}**{key}**: {value}")
    return "\n".join(lines)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_currency_symbol(currency_code: str) -> str:
    """
    Get the symbol for a currency code.
    
    Args:
        currency_code: ISO currency code
        
    Returns:
        Currency symbol or the code itself if not found
    """
    return CURRENCY_SYMBOLS.get(currency_code.upper(), currency_code)


def truncate_text(
    text: str,
    max_length: int = 100,
    suffix: str = "...",
) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to append if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix
