# finanalyst_tools/utils/__init__.py
"""
Utility functions for FinAnalyst-Pro Agent Tools.

This module exports:
- Math operations (safe division, decimal handling, statistics)
- Formatting functions (numbers, currency, percentages)
- Currency utilities (SGD-specific, GST calculations)

Version: 3.2.1 - Updated exports for new functions
"""

from finanalyst_tools.utils.math_ops import (
    # Type conversion
    to_decimal,
    is_effectively_zero,
    # Safe arithmetic
    safe_divide,
    round_decimal,
    clamp_value,
    # Percentage & growth
    calculate_percentage,
    calculate_growth_rate,
    calculate_cagr,
    # Statistics
    calculate_average,
    calculate_weighted_average,
    calculate_variance,
    calculate_std_dev,
    calculate_min_max,
    calculate_median,
    calculate_percentile,
    calculate_coefficient_of_variation,
)

from finanalyst_tools.utils.formatting import (
    # Number formatting
    format_number,
    format_currency,
    format_percentage,
    format_ratio,
    format_change,
    format_large_number,
    # Indicators
    format_trend_indicator,
    format_status_indicator,
    # Tables
    format_markdown_table,
    # Smart formatting
    format_value_with_unit,
    # Text utilities
    smart_truncate,
    escape_markdown,
)

from finanalyst_tools.utils.currency import (
    # Currency info
    get_currency_symbol,
    get_currency_decimals,
    get_currency_name,
    get_symbol_position,
    is_supported_currency,
    # SGD formatting
    format_sgd,
    # GST calculations
    get_gst_rate,
    calculate_gst_exclusive,
    calculate_gst_inclusive,
    calculate_gst_amount,
    # SFRS qualification
    is_sfrs_small_entity,
    # Constants
    SGD_GST_RATE,
    CURRENCY_INFO,
)


__version__ = "3.2.1"


__all__ = [
    # Version
    "__version__",
    # Math operations - Type conversion
    "to_decimal",
    "is_effectively_zero",
    # Math operations - Safe arithmetic
    "safe_divide",
    "round_decimal",
    "clamp_value",
    # Math operations - Percentage & growth
    "calculate_percentage",
    "calculate_growth_rate",
    "calculate_cagr",
    # Math operations - Statistics
    "calculate_average",
    "calculate_weighted_average",
    "calculate_variance",
    "calculate_std_dev",
    "calculate_min_max",
    "calculate_median",
    "calculate_percentile",
    "calculate_coefficient_of_variation",
    # Formatting - Numbers
    "format_number",
    "format_currency",
    "format_percentage",
    "format_ratio",
    "format_change",
    "format_large_number",
    # Formatting - Indicators
    "format_trend_indicator",
    "format_status_indicator",
    # Formatting - Tables
    "format_markdown_table",
    # Formatting - Smart
    "format_value_with_unit",
    "smart_truncate",
    "escape_markdown",
    # Currency - Info
    "get_currency_symbol",
    "get_currency_decimals",
    "get_currency_name",
    "get_symbol_position",
    "is_supported_currency",
    # Currency - SGD
    "format_sgd",
    # Currency - GST
    "get_gst_rate",
    "calculate_gst_exclusive",
    "calculate_gst_inclusive",
    "calculate_gst_amount",
    # Currency - SFRS
    "is_sfrs_small_entity",
    # Currency - Constants
    "SGD_GST_RATE",
    "CURRENCY_INFO",
]
