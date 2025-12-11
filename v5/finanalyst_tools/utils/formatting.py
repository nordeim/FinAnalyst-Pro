# finanalyst_tools/utils/formatting.py
"""
Formatting utilities for financial data display.

Provides consistent formatting for:
- Numbers with thousands separators
- Currency values with symbols
- Percentages and ratios
- Trend indicators
- Markdown tables

All functions handle None gracefully with configurable fallback values.

Version: 3.2.1 - Addresses Issues #10, #11, #12 + new text utilities
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Sequence

from finanalyst_tools.config import (
    DECIMAL_PLACES,
    CURRENCY_SYMBOLS,
    CURRENCY_SYMBOL_POSITION,
    DEFAULT_CURRENCY,
    TREND_SYMBOLS,
    STATUS_SYMBOLS,
    LARGE_NUMBER_SUFFIXES,
    METRIC_UNITS,
)
from finanalyst_tools.utils.math_ops import to_decimal


# ============================================================================
# TEXT UTILITIES (New in v3.2.1)
# ============================================================================

def escape_markdown(text: str) -> str:
    """
    Escape characters that have special meaning in Markdown.
    
    Handles pipe characters for tables (Issue #10 fix).
    
    Args:
        text: Text to escape
        
    Returns:
        Escaped text safe for Markdown
    """
    if not text:
        return text
    
    # Escape pipe characters (breaks tables)
    text = text.replace("|", "\\|")
    
    # Escape newlines (breaks table rows)
    text = text.replace("\n", " ").replace("\r", "")
    
    # Optionally escape other markdown characters
    # text = text.replace("*", "\\*")
    # text = text.replace("_", "\\_")
    # text = text.replace("`", "\\`")
    
    return text


def smart_truncate(
    text: str,
    max_length: int = 100,
    suffix: str = "...",
) -> str:
    """
    Truncate text at word boundaries for readability.
    
    Issue #59 fix: Avoids cutting words mid-way.
    
    Args:
        text: Text to truncate
        max_length: Maximum length (including suffix)
        suffix: String to append when truncated
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    # Account for suffix length
    target_length = max_length - len(suffix)
    if target_length <= 0:
        return suffix[:max_length]
    
    truncated = text[:target_length]
    
    # Find last space to avoid cutting words
    last_space = truncated.rfind(" ")
    
    # Only use word boundary if we don't lose too much text
    if last_space > target_length // 2:
        truncated = truncated[:last_space]
    
    return truncated.rstrip() + suffix


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
    precision: int | None = None,
    show_symbol: bool = True,
    symbol_position: str | None = None,
    fallback: str = "N/A",
) -> str:
    """
    Format a monetary value with currency symbol.
    
    Issue #12 fix: Supports symbol_position for prefix/suffix placement.
    
    Args:
        value: Amount to format
        currency_code: ISO currency code
        precision: Decimal places (None = use currency default)
        show_symbol: Whether to include currency symbol
        symbol_position: "prefix" or "suffix" (None = use currency default)
        fallback: String to return if value is None
        
    Returns:
        Formatted currency string (e.g., "S$1,234.56" or "1,234.56₫")
    """
    if value is None:
        return fallback
    
    # Determine precision from currency if not specified
    if precision is None:
        from finanalyst_tools.utils.currency import get_currency_decimals
        precision = get_currency_decimals(currency_code)
    
    dec_value = to_decimal(value, precision=precision)
    formatted = f"{abs(float(dec_value)):,.{precision}f}"
    
    # Handle negative values
    is_negative = dec_value < 0
    negative_prefix = "-" if is_negative else ""
    
    if show_symbol:
        symbol = CURRENCY_SYMBOLS.get(currency_code, currency_code)
        
        # Determine symbol position
        if symbol_position is None:
            symbol_position = CURRENCY_SYMBOL_POSITION.get(currency_code, "prefix")
        
        if symbol_position == "suffix":
            return f"{negative_prefix}{formatted}{symbol}"
        else:  # prefix
            return f"{negative_prefix}{symbol}{formatted}"
    
    return f"{negative_prefix}{formatted}"


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


def format_days(
    value: float | Decimal | int | None,
    fallback: str = "N/A",
) -> str:
    """
    Format a days value with proper pluralization.
    
    Issue #11 fix: Safe integer conversion with fallback.
    
    Args:
        value: Number of days
        fallback: String to return if value is None
        
    Returns:
        Formatted string (e.g., 
