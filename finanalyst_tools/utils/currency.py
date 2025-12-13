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
