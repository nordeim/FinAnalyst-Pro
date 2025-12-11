# finanalyst_tools/config.py
"""
Configuration and constants for FinAnalyst-Pro Agent Tools.

This module provides centralized configuration for:
- Decimal precision and rounding modes
- Plausibility ranges for financial metrics
- Reconciliation tolerances
- Currency settings (SGD default for Singapore SMB context)
- Analysis parameters
- Metric formulas and units for documentation

All constants use Final for immutability and are fully typed.

Author: FinAnalyst-Pro Team
Version: 1.0.0
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
    "eps": 4,            # Earnings per share: 1.2345
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
    GROSS_PROFIT_MARGIN: Final[tuple[float, float]] = (-50.0, 95.0)  # Alias
    OPERATING_MARGIN: Final[tuple[float, float]] = (-100.0, 60.0)
    OPERATING_PROFIT_MARGIN: Final[tuple[float, float]] = (-100.0, 60.0)  # Alias
    NET_MARGIN: Final[tuple[float, float]] = (-200.0, 50.0)
    NET_PROFIT_MARGIN: Final[tuple[float, float]] = (-200.0, 50.0)  # Alias
    EBITDA_MARGIN: Final[tuple[float, float]] = (-50.0, 70.0)
    ROA: Final[tuple[float, float]] = (-50.0, 40.0)
    RETURN_ON_ASSETS: Final[tuple[float, float]] = (-50.0, 40.0)  # Alias
    ROE: Final[tuple[float, float]] = (-100.0, 80.0)
    RETURN_ON_EQUITY: Final[tuple[float, float]] = (-100.0, 80.0)  # Alias
    ROCE: Final[tuple[float, float]] = (-50.0, 60.0)
    RETURN_ON_CAPITAL_EMPLOYED: Final[tuple[float, float]] = (-50.0, 60.0)  # Alias
    
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
    
    @classmethod
    def get_warning_message(cls, metric_name: str, value: float) -> str | None:
        """
        Get a warning message if value is outside plausible range.
        
        Args:
            metric_name: Name of the metric
            value: The value to check
            
        Returns:
            Warning message string or None if value is plausible
        """
        range_tuple = cls.get_range(metric_name)
        if range_tuple is None:
            return None
        
        if value < range_tuple[0]:
            return (
                f"{metric_name} value of {value:.2f} is below the typical range "
                f"({range_tuple[0]:.2f} to {range_tuple[1]:.2f}). "
                "Please verify the input data."
            )
        elif value > range_tuple[1]:
            return (
                f"{metric_name} value of {value:.2f} is above the typical range "
                f"({range_tuple[0]:.2f} to {range_tuple[1]:.2f}). "
                "Please verify the input data."
            )
        return None


# ============================================================================
# RECONCILIATION TOLERANCES
# ============================================================================

class ReconciliationTolerances:
    """
    Acceptable tolerance levels for cross-statement reconciliation.
    
    Expressed as a proportion (0.01 = 1%) of the larger value being compared.
    Different tolerance levels for different reconciliation contexts.
    """
    
    STRICT: Final[float] = 0.001   # 0.1% - For values that must match exactly
    NORMAL: Final[float] = 0.01    # 1% - For minor rounding differences
    LOOSE: Final[float] = 0.05     # 5% - For derived values with compounding
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
    def get_tolerance(cls, level_or_check: str) -> float:
        """
        Get tolerance value by level name or check type.
        
        Args:
            level_or_check: "strict", "normal", "loose", or a check name
            
        Returns:
            Tolerance as a proportion
        """
        upper = level_or_check.upper()
        if upper == "STRICT":
            return cls.STRICT
        elif upper == "LOOSE":
            return cls.LOOSE
        elif upper == "NORMAL":
            return cls.NORMAL
        
        # Try check-specific tolerance
        return cls.CHECK_TOLERANCES.get(level_or_check.lower(), cls.DEFAULT)
    
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
        
        # Use the larger absolute value as the base
        base = max(abs(value_a), abs(value_b))
        if base == 0:
            return True
        
        difference = abs(value_a - value_b)
        return (difference / base) <= tolerance
    
    @classmethod
    def calculate_difference_percentage(cls, value_a: float, value_b: float) -> float:
        """
        Calculate the percentage difference between two values.
        
        Args:
            value_a: First value
            value_b: Second value
            
        Returns:
            Percentage difference (0.05 = 5%)
        """
        if value_a == 0 and value_b == 0:
            return 0.0
        
        base = max(abs(value_a), abs(value_b))
        if base == 0:
            return 0.0
        
        return abs(value_a - value_b) / base


# ============================================================================
# CURRENCY CONFIGURATION
# ============================================================================

DEFAULT_CURRENCY: Final[str] = "SGD"

SUPPORTED_CURRENCIES: Final[frozenset[str]] = frozenset({
    "SGD",  # Singapore Dollar (primary)
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
    "KRW",  # South Korean Won
    "NZD",  # New Zealand Dollar
    "PHP",  # Philippine Peso
    "VND",  # Vietnamese Dong
    "TWD",  # Taiwan Dollar
    "CHF",  # Swiss Franc
})

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
    "KRW": "₩",
    "NZD": "NZ$",
    "PHP": "₱",
    "VND": "₫",
    "TWD": "NT$",
    "CHF": "CHF",
}

# Currencies that don't use decimal places
ZERO_DECIMAL_CURRENCIES: Final[frozenset[str]] = frozenset({"JPY", "KRW", "VND", "IDR"})


# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

MIN_PERIODS_FOR_TREND: Final[int] = 3
DEFAULT_FORECAST_PERIODS: Final[int] = 3
MAX_ANALYSIS_PERIODS: Final[int] = 10
DAYS_IN_YEAR: Final[int] = 365
DAYS_IN_MONTH: Final[float] = 30.44
MONTHS_IN_YEAR: Final[int] = 12
DAYS_IN_QUARTER: Final[int] = 91


# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================

MAX_MONETARY_VALUE: Final[float] = 1e15  # 1 quadrillion
ZERO_THRESHOLD: Final[float] = 1e-10


# ============================================================================
# SINGAPORE SMB CONTEXT
# ============================================================================

class SingaporeConstants:
    """Singapore-specific financial constants and thresholds."""
    
    # GST Rate (as of 2024)
    GST_RATE: Final[float] = 0.09  # 9%
    
    # SFRS for Small Entities thresholds (meet 2 of 3 to qualify)
    SFRS_SMALL_ENTITY_REVENUE: Final[int] = 10_000_000   # S$10M
    SFRS_SMALL_ENTITY_ASSETS: Final[int] = 10_000_000    # S$10M
    SFRS_SMALL_ENTITY_EMPLOYEES: Final[int] = 50
    
    # SME definition thresholds
    SME_ANNUAL_SALES: Final[int] = 100_000_000  # S$100M
    SME_EMPLOYEES: Final[int] = 200
    
    # Common financial year end months
    COMMON_FYE_MONTHS: Final[list[int]] = [12, 3, 6]  # Dec, Mar, Jun
    
    # Corporate tax rate
    CORPORATE_TAX_RATE: Final[float] = 0.17  # 17%
    
    # Withholding tax rates
    WITHHOLDING_TAX_DIVIDEND: Final[float] = 0.0   # No WHT on dividends
    WITHHOLDING_TAX_INTEREST: Final[float] = 0.15  # 15% on interest
    
    @classmethod
    def calculate_gst_exclusive(cls, gst_inclusive: float) -> float:
        """Convert GST-inclusive amount to GST-exclusive."""
        return gst_inclusive / (1 + cls.GST_RATE)
    
    @classmethod
    def calculate_gst_inclusive(cls, gst_exclusive: float) -> float:
        """Convert GST-exclusive amount to GST-inclusive."""
        return gst_exclusive * (1 + cls.GST_RATE)
    
    @classmethod
    def calculate_gst_amount(cls, gst_exclusive: float) -> float:
        """Calculate GST on a GST-exclusive amount."""
        return gst_exclusive * cls.GST_RATE


# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

TREND_SYMBOLS: Final[dict[str, str]] = {
    "increasing": "↑",
    "decreasing": "↓",
    "stable": "→",
    "volatile": "↕",
    "improving": "↑",
    "declining": "↓",
}

STATUS_SYMBOLS: Final[dict[str, str]] = {
    "good": "✅",
    "warning": "⚠️",
    "error": "❌",
    "info": "ℹ️",
    "unknown": "❓",
    "pass": "✅",
    "fail": "❌",
}

LARGE_NUMBER_SUFFIXES: Final[list[tuple[float, str]]] = [
    (1e12, "T"),   # Trillion
    (1e9, "B"),    # Billion
    (1e6, "M"),    # Million
    (1e3, "K"),    # Thousand
]


# ============================================================================
# METRIC METADATA (From V1 - Valuable for documentation)
# ============================================================================

METRIC_FORMULAS: Final[dict[str, str]] = {
    # Profitability
    "gross_profit_margin": "(Revenue - COGS) / Revenue × 100",
    "operating_profit_margin": "(Revenue - COGS - OpEx) / Revenue × 100",
    "net_profit_margin": "Net Income / Revenue × 100",
    "ebitda_margin": "EBITDA / Revenue × 100",
    "return_on_assets": "Net Income / Average Total Assets × 100",
    "return_on_equity": "Net Income / Average Shareholders' Equity × 100",
    "return_on_capital_employed": "EBIT / (Total Assets - Current Liabilities) × 100",
    # Liquidity
    "current_ratio": "Current Assets / Current Liabilities",
    "quick_ratio": "(Current Assets - Inventory) / Current Liabilities",
    "cash_ratio": "Cash and Equivalents / Current Liabilities",
    "working_capital": "Current Assets - Current Liabilities",
    # Solvency
    "debt_to_equity": "Total Liabilities / Shareholders' Equity",
    "debt_to_assets": "Total Liabilities / Total Assets",
    "interest_coverage": "EBIT / Interest Expense",
    "equity_ratio": "Shareholders' Equity / Total Assets",
    # Efficiency
    "asset_turnover": "Revenue / Average Total Assets",
    "inventory_turnover": "COGS / Average Inventory",
    "receivables_turnover": "Revenue / Average Accounts Receivable",
    "payables_turnover": "COGS / Average Accounts Payable",
    "days_sales_outstanding": "365 / Receivables Turnover",
    "days_inventory_outstanding": "365 / Inventory Turnover",
    "days_payables_outstanding": "365 / Payables Turnover",
    "cash_conversion_cycle": "DIO + DSO - DPO",
    # Growth
    "revenue_growth": "(Current Revenue - Prior Revenue) / Prior Revenue × 100",
    "net_income_growth": "(Current NI - Prior NI) / Prior NI × 100",
}

METRIC_UNITS: Final[dict[str, str]] = {
    # Profitability
    "gross_profit_margin": "percentage",
    "operating_profit_margin": "percentage",
    "net_profit_margin": "percentage",
    "ebitda_margin": "percentage",
    "return_on_assets": "percentage",
    "return_on_equity": "percentage",
    "return_on_capital_employed": "percentage",
    # Liquidity
    "current_ratio": "ratio",
    "quick_ratio": "ratio",
    "cash_ratio": "ratio",
    "working_capital": "currency",
    # Solvency
    "debt_to_equity": "ratio",
    "debt_to_assets": "ratio",
    "interest_coverage": "ratio",
    "equity_ratio": "ratio",
    # Efficiency
    "asset_turnover": "ratio",
    "inventory_turnover": "ratio",
    "receivables_turnover": "ratio",
    "payables_turnover": "ratio",
    "days_sales_outstanding": "days",
    "days_inventory_outstanding": "days",
    "days_payables_outstanding": "days",
    "cash_conversion_cycle": "days",
    # Growth
    "revenue_growth": "percentage",
    "net_income_growth": "percentage",
}

METRIC_CATEGORIES: Final[dict[str, str]] = {
    "gross_profit_margin": "profitability",
    "operating_profit_margin": "profitability",
    "net_profit_margin": "profitability",
    "ebitda_margin": "profitability",
    "return_on_assets": "profitability",
    "return_on_equity": "profitability",
    "return_on_capital_employed": "profitability",
    "current_ratio": "liquidity",
    "quick_ratio": "liquidity",
    "cash_ratio": "liquidity",
    "working_capital": "liquidity",
    "debt_to_equity": "solvency",
    "debt_to_assets": "solvency",
    "interest_coverage": "solvency",
    "equity_ratio": "solvency",
    "asset_turnover": "efficiency",
    "inventory_turnover": "efficiency",
    "receivables_turnover": "efficiency",
    "payables_turnover": "efficiency",
    "days_sales_outstanding": "efficiency",
    "days_inventory_outstanding": "efficiency",
    "days_payables_outstanding": "efficiency",
    "cash_conversion_cycle": "efficiency",
    "revenue_growth": "growth",
    "net_income_growth": "growth",
}


# ============================================================================
# ANALYSIS TYPE REQUIREMENTS
# ============================================================================

ANALYSIS_REQUIRED_FIELDS: Final[dict[str, dict[str, list[str]]]] = {
    "profitability": {
        "income_statement": ["total_revenue", "cost_of_goods_sold", "net_income"],
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
            "total_assets", "total_liabilities", 
            "total_shareholders_equity", "long_term_debt"
        ],
        "income_statement": ["operating_income", "interest_expense"],
    },
    "efficiency": {
        "income_statement": ["total_revenue", "cost_of_goods_sold"],
        "balance_sheet": [
            "total_assets", "inventory", 
            "accounts_receivable", "accounts_payable"
        ],
    },
    "comprehensive": {
        "income_statement": [
            "total_revenue", "cost_of_goods_sold", 
            "operating_income", "net_income", "interest_expense"
        ],
        "balance_sheet": [
            "total_assets", "current_assets", "current_liabilities",
            "total_liabilities", "total_shareholders_equity",
            "cash_and_equivalents", "inventory", 
            "accounts_receivable", "accounts_payable"
        ],
        "cash_flow": ["net_cash_from_operating"],
    },
}
