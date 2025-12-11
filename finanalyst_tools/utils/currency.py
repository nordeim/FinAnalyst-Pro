# finanalyst_tools/utils/currency.py
"""
Currency handling utilities with Singapore focus.

This module provides:
- Currency information (symbols, decimal places, names)
- Currency validation
- Singapore-specific utilities (GST calculations, SFRS thresholds)
- Placeholder for future FX integration

Singapore SMB Context:
- Default currency is SGD (Singapore Dollar)
- GST rate is 9% (as of 2024)
- SFRS for Small Entities has specific thresholds

Author: FinAnalyst-Pro Team
Version: 1.0.0
"""

from __future__ import annotations

from decimal import Decimal
from typing import Final

from finanalyst_tools.config import (
    DEFAULT_CURRENCY,
    SUPPORTED_CURRENCIES,
    CURRENCY_SYMBOLS,
    ZERO_DECIMAL_CURRENCIES,
    SingaporeConstants,
)
from finanalyst_tools.utils.math_ops import to_decimal, round_decimal


# ============================================================================
# CURRENCY INFORMATION
# ============================================================================

# Comprehensive currency information
CURRENCY_INFO: Final[dict[str, dict[str, str | int]]] = {
    "SGD": {"symbol": "S$", "name": "Singapore Dollar", "decimals": 2, "country": "Singapore"},
    "USD": {"symbol": "$", "name": "US Dollar", "decimals": 2, "country": "United States"},
    "EUR": {"symbol": "€", "name": "Euro", "decimals": 2, "country": "Eurozone"},
    "GBP": {"symbol": "£", "name": "British Pound", "decimals": 2, "country": "United Kingdom"},
    "JPY": {"symbol": "¥", "name": "Japanese Yen", "decimals": 0, "country": "Japan"},
    "CNY": {"symbol": "¥", "name": "Chinese Yuan", "decimals": 2, "country": "China"},
    "HKD": {"symbol": "HK$", "name": "Hong Kong Dollar", "decimals": 2, "country": "Hong Kong"},
    "AUD": {"symbol": "A$", "name": "Australian Dollar", "decimals": 2, "country": "Australia"},
    "MYR": {"symbol": "RM", "name": "Malaysian Ringgit", "decimals": 2, "country": "Malaysia"},
    "IDR": {"symbol": "Rp", "name": "Indonesian Rupiah", "decimals": 0, "country": "Indonesia"},
    "THB": {"symbol": "฿", "name": "Thai Baht", "decimals": 2, "country": "Thailand"},
    "INR": {"symbol": "₹", "name": "Indian Rupee", "decimals": 2, "country": "India"},
    "KRW": {"symbol": "₩", "name": "South Korean Won", "decimals": 0, "country": "South Korea"},
    "NZD": {"symbol": "NZ$", "name": "New Zealand Dollar", "decimals": 2, "country": "New Zealand"},
    "PHP": {"symbol": "₱", "name": "Philippine Peso", "decimals": 2, "country": "Philippines"},
    "VND": {"symbol": "₫", "name": "Vietnamese Dong", "decimals": 0, "country": "Vietnam"},
    "TWD": {"symbol": "NT$", "name": "Taiwan Dollar", "decimals": 2, "country": "Taiwan"},
    "CHF": {"symbol": "CHF", 
