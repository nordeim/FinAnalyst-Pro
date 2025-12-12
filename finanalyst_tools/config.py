# File: finanalyst_tools/config.py
"""
Configuration and constants for FinAnalyst-Pro Agent Tools.

This module provides centralized configuration for:
- Decimal precision and rounding modes
- Plausibility ranges for financial metrics
- Reconciliation tolerances
- Currency settings (SGD default for Singapore SMB context)
- Analysis parameters
- Metric formulas and units metadata

All constants use Final for immutability and are fully typed.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, ROUND_HALF_EVEN
from enum import Enum
from typing import Final, Any


# ============================================================================
# PRECISION & ROUNDING CONFIGURATION
# ============================================================================

class RoundingMode(str, Enum):
    """
    Supported rounding modes for financial calculations.
    
    STANDARD: Round half up (0.5 → 1) - Most common in financial reporting
    BANKERS: Round half to even (banker's rounding) - Reduces cumulative bias
    """
    STANDARD = "ROUND_HALF_UP"
    BANKERS = "ROUND_HALF_EVEN"
    
    def get_decimal_rounding(self) -> str:
        """Get the decimal module rounding constant."""
        if self == RoundingMode.STANDARD:
            return ROUND_HALF_UP
        return ROUND_HALF_EVEN


# Default decimal precision for different contexts
DECIMAL_PLACES: Final[dict[str, int]] = {
    "currency": 2,       # Monetary values: $1,234.56
    "percentage": 2,     # Percentages: 12.34%
    "ratio": 4,          # Financial ratios: 1.5432
    "shares": 0,         # Share counts: whole numbers
    "growth_rate": 4,    # Growth rates: 0.1234 (12.34%)
    "turnover": 2,       # Turnover ratios: 4.56x
    "days": 0,           # Day counts: whole numbers
}

# Default rounding mode for all calculations
DEFAULT_ROUNDING: Final[RoundingMode] = RoundingMode.STANDARD


# ============================================================================
# PLAUSIBILITY THRESHOLDS
# ============================================================================

class PlausibilityRanges:
    """
    Acceptable ranges for financial ratios and metrics.
    
    Values outside these ranges trigger warnings (not errors) during analysis.
    Ranges are intentionally wide to accommodate various industries and situations
    while catching obvious data errors.
    
    All percentage values are expressed as actual percentages (e.g., 20.0 = 20%).
    All ratios are expressed as decimal values (e.g., 1.5 = 1.5x).
    """
    
    # -------------------------------------------------------------------------
    # PROFITABILITY METRICS (percentages)
    # -------------------------------------------------------------------------
    GROSS_MARGIN: Final[tuple[float, float]] = (-50.0, 95.0)
    OPERATING_MARGIN: Final[tuple[float, float]] = (-100.0, 60.0)
    NET_MARGIN: Final[tuple[float, float]] = (-200.0, 50.0)
    EBITDA_MARGIN: Final[tuple[float, float]] = (-50.0, 70.0)
    ROA: Final[tuple[float, float]] = (-50.0, 40.0)
    ROE: Final[tuple[float, float]] = (-100.0, 80.0)
    ROCE: Final[tuple[float, float]] = (-50.0, 60.0)
    
    # -------------------------------------------------------------------------
    # LIQUIDITY METRICS (ratios)
    # -------------------------------------------------------------------------
    CURRENT_RATIO: Final[tuple[float, float]] = (0.1, 10.0)
    QUICK_RATIO: Final[tuple[float, float]] = (0.05, 8.0)
    CASH_RATIO: Final[tuple[float, float]] = (0.0, 5.0)
    
    # -------------------------------------------------------------------------
    # SOLVENCY METRICS (ratios)
    # -------------------------------------------------------------------------
    DEBT_TO_EQUITY: Final[tuple[float, float]] = (0.0, 10.0)
    DEBT_TO_ASSETS: Final[tuple[float, float]] = (0.0, 1.5)
    INTEREST_COVERAGE: Final[tuple[float, float]] = (-10.0, 100.0)
    EQUITY_RATIO: Final[tuple[float, float]] = (-0.5, 1.0)
    
    # -------------------------------------------------------------------------
    # EFFICIENCY METRICS (ratios/turnover)
    # -------------------------------------------------------------------------
    ASSET_TURNOVER: Final[tuple[float, float]] = (0.1, 5.0)
    INVENTORY_TURNOVER: Final[tuple[float, float]] = (0.5, 50.0)
    RECEIVABLES_TURNOVER: Final[tuple[float, float]] = (1.0, 50.0)
    PAYABLES_TURNOVER: Final[tuple[float, float]] = (1.0, 30.0)
    FIXED_ASSET_TURNOVER: Final[tuple[float, float]] = (0.1, 20.0)
    DAYS_SALES_OUTSTANDING: Final[tuple[float, float]] = (5.0, 180.0)
    DAYS_INVENTORY_OUTSTANDING: Final[tuple[float, float]] = (5.0, 365.0)
    DAYS_PAYABLES_OUTSTANDING: Final[tuple[float, float]] = (5.0, 180.0)
    
    # -------------------------------------------------------------------------
    # GROWTH METRICS (percentages)
    # -------------------------------------------------------------------------
    REVENUE_GROWTH: Final[tuple[float, float]] = (-80.0, 500.0)
    NET_INCOME_GROWTH: Final[tuple[float, float]] = (-500.0, 1000.0)
    ASSET_GROWTH: Final[tuple[float, float]] = (-50.0, 200.0)
    
    @classmethod
    def get_range(cls, metric_name: str) -> tuple[float, float] | None:
        """
        Get plausibility range for a metric by name.
        
        Args:
            metric_name: Name of the metric (case-insensitive, underscores/spaces flexible)
            
        Returns:
            Tuple of (min, max) or None if metric not found
        """
        normalized = metric_name.upper().replace(" ", "_").replace("-", "_")
        # Handle common aliases
        aliases = {
            "GROSS_PROFIT_MARGIN": "GROSS_MARGIN",
            "OPERATING_PROFIT_MARGIN": "OPERATING_MARGIN",
            "NET_PROFIT_MARGIN": "NET_MARGIN",
            "RETURN_ON_ASSETS": "ROA",
            "RETURN_ON_EQUITY": "ROE",
            "RETURN_ON_CAPITAL_EMPLOYED": "ROCE",
        }
        normalized = aliases.get(normalized, normalized)
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
            return True
        return range_tuple[0] <= value <= range_tuple[1]
    
    @classmethod
    def get_assessment(cls, metric_name: str, value: float) -> str:
        """
        Get a human-readable assessment of a metric value.
        
        Args:
            metric_name: Name of the metric
            value: The value to assess
            
        Returns:
            Assessment string: "within_range", "below_range", "above_range", or "unknown"
        """
        range_tuple = cls.get_range(metric_name)
        if range_tuple is None:
            return "unknown"
        
        if value < range_tuple[0]:
            return "below_range"
        elif value > range_tuple[1]:
            return "above_range"
        return "within_range"


# ============================================================================
# RECONCILIATION TOLERANCES
# ============================================================================

class ReconciliationTolerances:
    """
    Acceptable tolerance levels for cross-statement reconciliation.
    
    Expressed as a proportion (0.01 = 1%) of the larger value being compared.
    Different tolerance levels for different reconciliation contexts.
    """
    
    STRICT: Final[float] = 0.001   # 0.1% - Values that should match exactly
    NORMAL: Final[float] = 0.01    # 1% - Minor rounding differences allowed
    LOOSE: Final[float] = 0.05     # 5% - Derived values with compounding differences
    DEFAULT: Final[float] = NORMAL
    
    # Specific tolerances for different check types
    CHECK_TOLERANCES: Final[dict[str, float]] = {
        "net_income": STRICT,
        "cash_balance": STRICT,
        "retained_earnings": NORMAL,
        "total_assets": STRICT,
        "working_capital": NORMAL,
        "balance_sheet_equation": STRICT,
    }
    
    @classmethod
    def get_tolerance(cls, check_type: str) -> float:
        """Get tolerance for a specific check type."""
        return cls.CHECK_TOLERANCES.get(check_type.lower(), cls.DEFAULT)
    
    @classmethod
    def is_within_tolerance(
        cls, 
        value_a: float, 
        value_b: float, 
        tolerance: float | None = None
    ) -> bool:
        """
        Check if two values are within tolerance of each other.
        
        Args:
            value_a: First value
            value_b: Second value
            tolerance: Tolerance level (proportion). Uses DEFAULT if not specified.
            
        Returns:
            True if values are within tolerance
        """
        if tolerance is None:
            tolerance = cls.DEFAULT
        
        if value_a == 0 and value_b == 0:
            return True
        
        base = max(abs(value_a), abs(value_b))
        if base == 0:
            return True
        
        difference = abs(value_a - value_b)
        return (difference / base) <= tolerance


# ============================================================================
# CURRENCY CONFIGURATION
# ============================================================================

DEFAULT_CURRENCY: Final[str] = "SGD"

SUPPORTED_CURRENCIES: Final[frozenset[str]] = frozenset({
    "SGD", "USD", "EUR", "GBP", "JPY", "CNY", "HKD", "AUD",
    "MYR", "IDR", "THB", "INR", "KRW", "NZD", "PHP", "VND",
})

CURRENCY_SYMBOLS: Final[dict[str, str]] = {
    "SGD": "S$", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥",
    "CNY": "¥", "HKD": "HK$", "AUD": "A$", "MYR": "RM", "IDR": "Rp",
    "THB": "฿", "INR": "₹", "KRW": "₩", "NZD": "NZ$", "PHP": "₱", "VND": "₫",
}

ZERO_DECIMAL_CURRENCIES: Final[frozenset[str]] = frozenset({"JPY", "IDR", "KRW", "VND"})


# ============================================================================
# SINGAPORE SMB CONTEXT
# ============================================================================

class SingaporeConstants:
    """Singapore-specific financial constants and thresholds."""
    
    GST_RATE: Final[float] = 0.09  # 9% as of 2024
    
    # SFRS for Small Entities thresholds
    SFRS_SMALL_ENTITY_REVENUE: Final[int] = 10_000_000    # S$10M
    SFRS_SMALL_ENTITY_ASSETS: Final[int] = 10_000_000     # S$10M
    SFRS_SMALL_ENTITY_EMPLOYEES: Final[int] = 50
    
    # SME definition thresholds
    SME_ANNUAL_SALES: Final[int] = 100_000_000  # S$100M
    SME_EMPLOYEES: Final[int] = 200
    
    # Common financial year end months
    COMMON_FYE_MONTHS: Final[list[int]] = [12, 3, 6]
    
    @classmethod
    def calculate_gst_exclusive(cls, gst_inclusive: float) -> float:
        """Convert GST-inclusive amount to GST-exclusive."""
        return gst_inclusive / (1 + cls.GST_RATE)
    
    @classmethod
    def calculate_gst_amount(cls, gst_exclusive: float) -> float:
        """Calculate GST on a GST-exclusive amount."""
        return gst_exclusive * cls.GST_RATE


# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

MIN_PERIODS_FOR_TREND: Final[int] = 3
DEFAULT_FORECAST_PERIODS: Final[int] = 3
MAX_ANALYSIS_PERIODS: Final[int] = 10
DAYS_IN_YEAR: Final[int] = 365
DAYS_IN_MONTH: Final[float] = 30.44
MONTHS_IN_YEAR: Final[int] = 12


# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================

MAX_MONETARY_VALUE: Final[float] = 1e15
ZERO_THRESHOLD: Final[float] = 1e-10


# ============================================================================
# METRIC FORMULAS (from V1 - valuable for documentation)
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
    "receivables_turnover": "Revenue / Average Accounts Receivable",
    "days_sales_outstanding": "365 / Receivables Turnover",
    "days_inventory_outstanding": "365 / Inventory Turnover",
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
    "receivables_turnover": "ratio",
    "days_sales_outstanding": "days",
    "days_inventory_outstanding": "days",
}


# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

TREND_SYMBOLS: Final[dict[str, str]] = {
    "increasing": "↑",
    "decreasing": "↓",
    "stable": "→",
    "volatile": "↕",
}

STATUS_SYMBOLS: Final[dict[str, str]] = {
    "good": "✅",
    "warning": "⚠️",
    "error": "❌",
    "info": "ℹ️",
    "unknown": "❓",
}

LARGE_NUMBER_SUFFIXES: Final[list[tuple[float, str]]] = [
    (1e12, "T"),
    (1e9, "B"),
    (1e6, "M"),
    (1e3, "K"),
]
