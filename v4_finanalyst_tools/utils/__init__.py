# finanalyst_tools/utils/__init__.py
"""
Utility functions for FinAnalyst-Pro tools.

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
    compare_values,
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
    "compare_values",
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
