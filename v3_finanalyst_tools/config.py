"""
Configuration and constants for FinAnalyst-Pro tools.

This module centralizes all configuration values, thresholds, and constants
used throughout the package. Modify these values to adjust behavior globally.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, ROUND_HALF_EVEN
from enum import Enum
from typing import Final


# ============================================================================
# PRECISION & ROUNDING CONFIGURATION
# ============================================================================

class RoundingMode(str, Enum):
    """Supported rounding modes for financial calculations."""
    STANDARD = "ROUND_HALF_UP"     # Default: 0.5 rounds up (most common)
    BANKERS = "ROUND_HALF_EVEN"    # Banker's rounding: 0.5 rounds to even
    
    def get_decimal_rounding(self):
        """Get the decimal module rounding constant."""
        if self == RoundingMode.STANDARD:
            return ROUND_HALF_UP
        return ROUND_HALF_EVEN


# Default decimal precision for different contexts
DECIMAL_PLACES: Final[dict[str, int]] = {
    "currency": 2,       # Monetary values: $1,234.56
    "percentage": 2,     # Percentages: 45.67%
    "ratio": 4,          # Ratios: 1.5234
    "shares": 0,         # Share counts: whole numbers
    "growth_rate": 4,    # Growth rates: 12.3456%
    "days": 0,           # Days (turnover): whole numbers
}

# Default rounding mode
DEFAULT_ROUNDING: Final[RoundingMode] = RoundingMode.STANDARD


# ============================================================================
# PLAUSIBILITY THRESHOLDS
# ============================================================================

class PlausibilityRanges:
    """
    Acceptable ranges for financial ratios and metrics.
    
    Values outside these ranges trigger warnings (not errors).
    Ranges are based on typical business metrics; specific industries
    may have different norms.
    
    Format: (minimum, maximum) as percentages for margins, absolute for ratios
    """
    
    # ─────────────────────────────────────────────────────────────────────
    # Profitability Metrics (percentages)
    # ─────────────────────────────────────────────────────────────────────
    GROSS_MARGIN: tuple[float, float] = (-50.0, 95.0)
    OPERATING_MARGIN: tuple[float, float] = (-100.0, 60.0)
    NET_MARGIN: tuple[float, float] = (-200.0, 50.0)
    EBITDA_MARGIN: tuple[float, float] = (-50.0, 70.0)
    ROA: tuple[float, float] = (-50.0, 40.0)
    ROE: tuple[float, float] = (-100.0, 60.0)
    ROCE: tuple[float, float] = (-50.0, 50.0)
    
    # ─────────────────────────────────────────────────────────────────────
    # Liquidity Ratios (absolute ratios)
    # ─────────────────────────────────────────────────────────────────────
    CURRENT_RATIO: tuple[float, float] = (0.1, 10.0)
    QUICK_RATIO: tuple[float, float] = (0.05, 8.0)
    CASH_RATIO: tuple[float, float] = (0.0, 5.0)
    
    # ─────────────────────────────────────────────────────────────────────
    # Solvency Ratios (absolute ratios)
    # ─────────────────────────────────────────────────────────────────────
    DEBT_TO_EQUITY: tuple[float, float] = (0.0, 10.0)
    DEBT_TO_ASSETS: tuple[float, float] = (0.0, 1.5)
    INTEREST_COVERAGE: tuple[float, float] = (-10.0, 100.0)
    EQUITY_RATIO: tuple[float, float] = (0.0, 1.0)
    
    # ─────────────────────────────────────────────────────────────────────
    # Efficiency Ratios (absolute ratios / days)
    # ─────────────────────────────────────────────────────────────────────
    ASSET_TURNOVER: tuple[float, float] = (0.1, 5.0)
    INVENTORY_TURNOVER: tuple[float, float] = (0.5, 50.0)
    RECEIVABLES_TURNOVER: tuple[float, float] = (1.0, 30.0)
    PAYABLES_TURNOVER: tuple[float, float] = (1.0, 30.0)
    DAYS_SALES_OUTSTANDING: tuple[float, float] = (5.0, 180.0)
    DAYS_INVENTORY_OUTSTANDING: tuple[float, float] = (5.0, 365.0)
    DAYS_PAYABLES_OUTSTANDING: tuple[float, float] = (5.0, 180.0)
    
    # ─────────────────────────────────────────────────────────────────────
    # Growth Metrics (percentages)
    # ─────────────────────────────────────────────────────────────────────
    REVENUE_GROWTH: tuple[float, float] = (-80.0, 500.0)
    NET_INCOME_GROWTH: tuple[float, float] = (-200.0, 1000.0)
    ASSET_GROWTH: tuple[float, float] = (-50.0, 200.0)
    
    @classmethod
    def get_range(cls, metric_name: str) -> tuple[float, float] | None:
        """
        Get plausibility range for a metric by name.
        
        Args:
            metric_name: Name of the metric (case-insensitive, underscores normalized)
            
        Returns:
            Tuple of (min, max) or None if metric not found
        """
        normalized = metric_name.upper().replace(" ", "_").replace("-", "_")
        return getattr(cls, normalized, None)
    
    @classmethod
    def is_plausible(cls, metric_name: str, value: float) -> bool:
        """
        Check if a metric value is within plausible range.
        
        Args:
            metric_name: Name of the metric
            value: The value to check
            
        Returns:
            True if within range or range not defined, False otherwise
        """
        range_tuple = cls.get_range(metric_name)
        if range_tuple is None:
            return True  # No range defined = assume plausible
        return range_tuple[0] <= value <= range_tuple[1]


# ============================================================================
# RECONCILIATION TOLERANCES
# ============================================================================

class ReconciliationTolerances:
    """
    Acceptable tolerance levels for cross-statement reconciliation.
    
    Expressed as percentage of the larger value being compared.
    Financial statements may have minor rounding differences.
    """
    
    # Strict: for values that should match exactly
    STRICT: Final[float] = 0.001  # 0.1% tolerance
    
    # Normal: for values that may have minor rounding differences
    NORMAL: Final[float] = 0.01  # 1% tolerance
    
    # Loose: for derived values that may have compounding differences
    LOOSE: Final[float] = 0.05  # 5% tolerance
    
    # Default tolerance for unspecified checks
    DEFAULT: Final[float] = NORMAL
    
    @classmethod
    def get_tolerance(cls, check_type: str) -> float:
        """Get tolerance for a specific check type."""
        tolerances = {
            "net_income": cls.STRICT,
            "cash_balance": cls.STRICT,
            "retained_earnings": cls.NORMAL,
            "total_assets": cls.STRICT,
            "working_capital": cls.NORMAL,
            "balance_sheet_equation": cls.STRICT,
        }
        return tolerances.get(check_type.lower(), cls.DEFAULT)


# ============================================================================
# CURRENCY CONFIGURATION
# ============================================================================

DEFAULT_CURRENCY: Final[str] = "SGD"

SUPPORTED_CURRENCIES: Final[set[str]] = {
    "SGD",  # Singapore Dollar (default)
    "USD",  # US Dollar
    "EUR",  # Euro
    "GBP",  # British Pound
    "JPY",  # Japanese Yen
    "CNY",  # Chinese Yuan
    "HKD",  # Hong Kong Dollar
    "AUD",  # Australian Dollar
    "MYR",  # Malaysian Ringgit
    "IDR",  # Indonesian Rupiah
    "THB",  # Thai Baht
    "INR",  # Indian Rupee
}

# Currency symbols for formatting
CURRENCY_SYMBOLS: Final[dict[str, str]] = {
    "SGD": "S$",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CNY": "¥",
    "HKD": "HK$",
    "AUD": "A$",
    "MYR": "RM",
    "IDR": "Rp",
    "THB": "฿",
    "INR": "₹",
}

# Currencies that don't use decimal places (or use different decimals)
ZERO_DECIMAL_CURRENCIES: Final[set[str]] = {"JPY", "IDR"}


# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

# Minimum periods required for trend analysis
MIN_PERIODS_FOR_TREND: Final[int] = 3

# Default forecast horizon (periods)
DEFAULT_FORECAST_PERIODS: Final[int] = 3

# Days in year for turnover calculations
DAYS_IN_YEAR: Final[int] = 365

# Days in quarter
DAYS_IN_QUARTER: Final[int] = 91


# ============================================================================
# SINGAPORE SMB CONTEXT
# ============================================================================

class SingaporeConstants:
    """Singapore-specific financial constants and thresholds."""
    
    # GST Rate (as of 2024)
    GST_RATE: Final[float] = 0.09  # 9%
    
    # SFRS for Small Entities thresholds (meet 2 of 3 to qualify)
    SFRS_SMALL_ENTITY_REVENUE: Final[int] = 10_000_000  # S$10M
    SFRS_SMALL_ENTITY_ASSETS: Final[int] = 10_000_000   # S$10M
    SFRS_SMALL_ENTITY_EMPLOYEES: Final[int] = 50
    
    # SME definition thresholds
    SME_ANNUAL_SALES: Final[int] = 100_000_000  # S$100M
    SME_EMPLOYEES: Final[int] = 200
    
    # Common financial year end months
    COMMON_FYE_MONTHS: Final[list[int]] = [12, 3, 6]  # Dec, Mar, Jun
    
    @classmethod
    def calculate_gst_exclusive(cls, gst_inclusive: float) -> float:
        """Convert GST-inclusive amount to GST-exclusive."""
        return gst_inclusive / (1 + cls.GST_RATE)
    
    @classmethod
    def calculate_gst_amount(cls, gst_exclusive: float) -> float:
        """Calculate GST on a GST-exclusive amount."""
        return gst_exclusive * cls.GST_RATE


# ============================================================================
# METRIC METADATA
# ============================================================================

METRIC_FORMULAS: Final[dict[str, str]] = {
    "gross_profit_margin": "(Revenue - COGS) / Revenue × 100",
    "operating_profit_margin": "(Revenue - COGS - OpEx) / Revenue × 100",
    "net_profit_margin": "Net Income / Revenue × 100",
    "ebitda_margin": "EBITDA / Revenue × 100",
    "roa": "Net Income / Average Total Assets × 100",
    "roe": "Net Income / Average Shareholders' Equity × 100",
    "roce": "EBIT / (Total Assets - Current Liabilities) × 100",
    "current_ratio": "Current Assets / Current Liabilities",
    "quick_ratio": "(Current Assets - Inventory) / Current Liabilities",
    "cash_ratio": "Cash and Equivalents / Current Liabilities",
    "working_capital": "Current Assets - Current Liabilities",
    "debt_to_equity": "Total Liabilities / Shareholders' Equity",
    "debt_to_assets": "Total Liabilities / Total Assets",
    "interest_coverage": "EBIT / Interest Expense",
    "asset_turnover": "Revenue / Average Total Assets",
    "inventory_turnover": "COGS / Average Inventory",
}

METRIC_UNITS: Final[dict[str, str]] = {
    "gross_profit_margin": "percentage",
    "operating_profit_margin": "percentage",
    "net_profit_margin": "percentage",
    "ebitda_margin": "percentage",
    "roa": "percentage",
    "roe": "percentage",
    "roce": "percentage",
    "current_ratio": "ratio",
    "quick_ratio": "ratio",
    "cash_ratio": "ratio",
    "working_capital": "currency",
    "debt_to_equity": "ratio",
    "debt_to_assets": "ratio",
    "interest_coverage": "ratio",
    "asset_turnover": "ratio",
    "inventory_turnover": "ratio",
}
