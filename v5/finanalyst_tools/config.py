# finanalyst_tools/config.py
"""
Configuration and constants for FinAnalyst-Pro Agent Tools.

This module provides centralized configuration for:
- Decimal precision and rounding modes
- Plausibility ranges for financial metrics
- Reconciliation tolerances
- Currency settings (SGD default for Singapore SMB context)
- Analysis parameters
- Metric formulas and units metadata
- Confidence scoring weights
- Recommendation thresholds
- Report formatting constants

All constants use Final for immutability and are fully typed.

Version: 3.2.1 - Addresses Issues #1, #3, #4, #13, #44, #55, #57, #60
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
from enum import Enum
from typing import Final, Any, NamedTuple


# ============================================================================
# VERSION
# ============================================================================

__version__: Final[str] = "3.2.1"


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
        """
        Get the decimal module rounding constant.
        
        Returns:
            Rounding mode string constant from decimal module
        """
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
    "high_precision": 8, # Intermediate calculations
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
    
    Updated in v3.2.1: Widened ROE range to handle edge cases (Issue #4)
    """
    
    # -------------------------------------------------------------------------
    # PROFITABILITY METRICS (percentages)
    # -------------------------------------------------------------------------
    GROSS_MARGIN: Final[tuple[float, float]] = (-50.0, 95.0)
    OPERATING_MARGIN: Final[tuple[float, float]] = (-100.0, 60.0)
    NET_MARGIN: Final[tuple[float, float]] = (-200.0, 50.0)
    EBITDA_MARGIN: Final[tuple[float, float]] = (-50.0, 70.0)
    ROA: Final[tuple[float, float]] = (-50.0, 50.0)
    # Widened ROE range for high-growth companies with high leverage (Issue #4)
    ROE: Final[tuple[float, float]] = (-150.0, 150.0)
    ROCE: Final[tuple[float, float]] = (-50.0, 80.0)
    
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
    CASH_CONVERSION_CYCLE: Final[tuple[float, float]] = (-60.0, 300.0)
    
    # -------------------------------------------------------------------------
    # GROWTH METRICS (percentages)
    # -------------------------------------------------------------------------
    REVENUE_GROWTH: Final[tuple[float, float]] = (-80.0, 500.0)
    NET_INCOME_GROWTH: Final[tuple[float, float]] = (-500.0, 1000.0)
    ASSET_GROWTH: Final[tuple[float, float]] = (-50.0, 200.0)
    
    # -------------------------------------------------------------------------
    # ADDITIONAL METRICS
    # -------------------------------------------------------------------------
    FREE_CASH_FLOW_MARGIN: Final[tuple[float, float]] = (-100.0, 50.0)
    OPERATING_CASH_FLOW_RATIO: Final[tuple[float, float]] = (0.0, 5.0)
    
    # Alias mapping for flexible lookup
    _ALIASES: Final[dict[str, str]] = {
        "GROSS_PROFIT_MARGIN": "GROSS_MARGIN",
        "OPERATING_PROFIT_MARGIN": "OPERATING_MARGIN",
        "NET_PROFIT_MARGIN": "NET_MARGIN",
        "RETURN_ON_ASSETS": "ROA",
        "RETURN_ON_EQUITY": "ROE",
        "RETURN_ON_CAPITAL_EMPLOYED": "ROCE",
        "DSO": "DAYS_SALES_OUTSTANDING",
        "DIO": "DAYS_INVENTORY_OUTSTANDING",
        "DPO": "DAYS_PAYABLES_OUTSTANDING",
        "CCC": "CASH_CONVERSION_CYCLE",
    }
    
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
        
        # Check aliases first
        normalized = cls._ALIASES.get(normalized, normalized)
        
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
        "net_income_reconciliation": STRICT,
        "cash_balance": STRICT,
        "cash_balance_reconciliation": STRICT,
        "retained_earnings": NORMAL,
        "retained_earnings_rollforward": NORMAL,
        "total_assets": STRICT,
        "working_capital": NORMAL,
        "working_capital_consistency": NORMAL,
        "balance_sheet_equation": STRICT,
    }
    
    @classmethod
    def get_tolerance(cls, check_type: str) -> float:
        """
        Get tolerance for a specific check type.
        
        Args:
            check_type: Name of the reconciliation check (case-insensitive)
            
        Returns:
            Tolerance level as float (proportion)
        """
        # Normalize the check type name
        normalized = check_type.lower().replace(" ", "_").replace("-", "_")
        return cls.CHECK_TOLERANCES.get(normalized, cls.DEFAULT)
    
    @classmethod
    def is_within_tolerance(
        cls, 
        value_a: float, 
        value_b: float, 
        tolerance: float | None = None,
        check_type: str | None = None,
    ) -> bool:
        """
        Check if two values are within tolerance of each other.
        
        Args:
            value_a: First value
            value_b: Second value
            tolerance: Tolerance level (proportion). Uses DEFAULT if not specified.
            check_type: Optional check type to look up tolerance
            
        Returns:
            True if values are within tolerance
        """
        if tolerance is None:
            if check_type:
                tolerance = cls.get_tolerance(check_type)
            else:
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

# Currency symbol position (prefix or suffix)
CURRENCY_SYMBOL_POSITION: Final[dict[str, str]] = {
    "SGD": "prefix",
    "USD": "prefix",
    "EUR": "prefix",  # Can be suffix in some locales, but prefix is more common
    "GBP": "prefix",
    "JPY": "prefix",
    "CNY": "prefix",
    "HKD": "prefix",
    "AUD": "prefix",
    "MYR": "prefix",
    "IDR": "prefix",
    "THB": "prefix",
    "INR": "prefix",
    "KRW": "prefix",
    "NZD": "prefix",
    "PHP": "prefix",
    "VND": "suffix",
}


# ============================================================================
# SINGAPORE SMB CONTEXT
# ============================================================================

class GSTRate(NamedTuple):
    """Singapore GST rate for a specific period."""
    rate: Decimal
    effective_from: date
    effective_to: date | None  # None = current/ongoing


class SingaporeConstants:
    """
    Singapore-specific financial constants and thresholds.
    
    Updated in v3.2.1: Added historical GST rates (Issue #13)
    """
    
    # Historical GST rates (Issue #13 fix)
    GST_HISTORY: Final[list[GSTRate]] = [
        GSTRate(Decimal("0.03"), date(1994, 4, 1), date(2003, 12, 31)),
        GSTRate(Decimal("0.04"), date(2004, 1, 1), date(2004, 12, 31)),
        GSTRate(Decimal("0.05"), date(2005, 1, 1), date(2007, 6, 30)),
        GSTRate(Decimal("0.07"), date(2007, 7, 1), date(2022, 12, 31)),
        GSTRate(Decimal("0.08"), date(2023, 1, 1), date(2023, 12, 31)),
        GSTRate(Decimal("0.09"), date(2024, 1, 1), None),  # Current rate
    ]
    
    # Current GST rate (as Decimal for precision - Issue #1 fix)
    GST_RATE: Final[Decimal] = Decimal("0.09")  # 9% as of 2024
    
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
    def get_gst_rate(cls, as_of: date | None = None) -> Decimal:
        """
        Get the applicable GST rate for a given date.
        
        Args:
            as_of: Date to get rate for (defaults to today)
            
        Returns:
            GST rate as Decimal (e.g., Decimal("0.09") for 9%)
        """
        if as_of is None:
            as_of = date.today()
        
        for gst in reversed(cls.GST_HISTORY):
            if gst.effective_from <= as_of:
                if gst.effective_to is None or as_of <= gst.effective_to:
                    return gst.rate
        
        # Fallback to current rate
        return cls.GST_RATE
    
    @classmethod
    def calculate_gst_exclusive(
        cls, 
        gst_inclusive: Decimal | float,
        as_of: date | None = None,
    ) -> Decimal:
        """
        Convert GST-inclusive amount to GST-exclusive.
        
        Args:
            gst_inclusive: Amount including GST
            as_of: Date for applicable GST rate
            
        Returns:
            Amount excluding GST
        """
        rate = cls.get_gst_rate(as_of)
        amount = Decimal(str(gst_inclusive)) if not isinstance(gst_inclusive, Decimal) else gst_inclusive
        return amount / (1 + rate)
    
    @classmethod
    def calculate_gst_amount(
        cls, 
        gst_exclusive: Decimal | float,
        as_of: date | None = None,
    ) -> Decimal:
        """
        Calculate GST on a GST-exclusive amount.
        
        Args:
            gst_exclusive: Amount excluding GST
            as_of: Date for applicable GST rate
            
        Returns:
            GST amount
        """
        rate = cls.get_gst_rate(as_of)
        amount = Decimal(str(gst_exclusive)) if not isinstance(gst_exclusive, Decimal) else gst_exclusive
        return amount * rate


# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

MIN_PERIODS_FOR_TREND: Final[int] = 3
DEFAULT_FORECAST_PERIODS: Final[int] = 3
MAX_ANALYSIS_PERIODS: Final[int] = 10
DAYS_IN_YEAR: Final[int] = 365
DAYS_IN_MONTH: Final[Decimal] = Decimal("30.44")
MONTHS_IN_YEAR: Final[int] = 12


# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================

MAX_MONETARY_VALUE: Final[Decimal] = Decimal("1e15")
ZERO_THRESHOLD: Final[Decimal] = Decimal("1e-10")


# ============================================================================
# METRIC FORMULAS (Enhanced - Issue #3 fix)
# ============================================================================

METRIC_FORMULAS: Final[dict[str, str]] = {
    # Profitability
    "gross_profit_margin": "(Revenue - COGS) / Revenue × 100",
    "operating_profit_margin": "(Revenue - COGS - OpEx) / Revenue × 100",
    "net_profit_margin": "Net Income / Revenue × 100",
    "ebitda_margin": "EBITDA / Revenue × 100",
    "roa": "Net Income / Average Total Assets × 100",
    "return_on_assets": "Net Income / Average Total Assets × 100",
    "roe": "Net Income / Average Shareholders' Equity × 100",
    "return_on_equity": "Net Income / Average Shareholders' Equity × 100",
    "roce": "EBIT / (Total Assets - Current Liabilities) × 100",
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
    "debt_service_coverage": "Net Operating Income / Total Debt Service",
    
    # Efficiency
    "asset_turnover": "Revenue / Average Total Assets",
    "inventory_turnover": "COGS / Average Inventory",
    "receivables_turnover": "Revenue / Average Accounts Receivable",
    "payables_turnover": "COGS / Average Accounts Payable",
    "fixed_asset_turnover": "Revenue / Average Fixed Assets",
    "days_sales_outstanding": "365 / Receivables Turnover",
    "days_inventory_outstanding": "365 / Inventory Turnover",
    "days_payables_outstanding": "365 / Payables Turnover",
    
    # Cash Flow (Issue #3 additions)
    "cash_conversion_cycle": "DIO + DSO - DPO",
    "operating_cash_flow_ratio": "Operating Cash Flow / Current Liabilities",
    "free_cash_flow": "Operating Cash Flow - Capital Expenditures",
    "free_cash_flow_margin": "Free Cash Flow / Revenue × 100",
    
    # Valuation (Issue #3 additions)
    "earnings_per_share": "Net Income / Weighted Average Shares Outstanding",
    "price_to_earnings": "Stock Price / Earnings Per Share",
    "price_to_book": "Stock Price / Book Value Per Share",
    "book_value_per_share": "Shareholders' Equity / Shares Outstanding",
    
    # Growth
    "revenue_growth": "(Current Revenue - Prior Revenue) / Prior Revenue × 100",
    "net_income_growth": "(Current NI - Prior NI) / |Prior NI| × 100",
    "cagr": "((Ending Value / Beginning Value) ^ (1/n) - 1) × 100",
}

METRIC_UNITS: Final[dict[str, str]] = {
    # Profitability (percentages)
    "gross_profit_margin": "percentage",
    "operating_profit_margin": "percentage",
    "net_profit_margin": "percentage",
    "ebitda_margin": "percentage",
    "roa": "percentage",
    "return_on_assets": "percentage",
    "roe": "percentage",
    "return_on_equity": "percentage",
    "roce": "percentage",
    "return_on_capital_employed": "percentage",
    "free_cash_flow_margin": "percentage",
    
    # Liquidity (ratios)
    "current_ratio": "ratio",
    "quick_ratio": "ratio",
    "cash_ratio": "ratio",
    "working_capital": "currency",
    "operating_cash_flow_ratio": "ratio",
    
    # Solvency (ratios)
    "debt_to_equity": "ratio",
    "debt_to_assets": "ratio",
    "interest_coverage": "times",
    "equity_ratio": "ratio",
    "debt_service_coverage": "times",
    
    # Efficiency (turnover = times, days = days)
    "asset_turnover": "times",
    "inventory_turnover": "times",
    "receivables_turnover": "times",
    "payables_turnover": "times",
    "fixed_asset_turnover": "times",
    "days_sales_outstanding": "days",
    "days_inventory_outstanding": "days",
    "days_payables_outstanding": "days",
    "cash_conversion_cycle": "days",
    
    # Cash Flow
    "free_cash_flow": "currency",
    
    # Valuation
    "earnings_per_share": "currency",
    "price_to_earnings": "ratio",
    "price_to_book": "ratio",
    "book_value_per_share": "currency",
    
    # Growth (percentages)
    "revenue_growth": "percentage",
    "net_income_growth": "percentage",
    "cagr": "percentage",
}


# ============================================================================
# PROFITABILITY WARNING THRESHOLDS (Issue #44 fix)
# ============================================================================

class ProfitabilityWarningThresholds:
    """Centralized thresholds for profitability metric warnings."""
    
    GROSS_MARGIN_MIN_NEGATIVE: Final[Decimal] = Decimal("0")
    GROSS_MARGIN_MAX_TYPICAL: Final[Decimal] = Decimal("100")
    
    OPERATING_MARGIN_MIN_SEVERE: Final[Decimal] = Decimal("-50")
    OPERATING_MARGIN_MAX_EXCEPTIONAL: Final[Decimal] = Decimal("50")
    
    NET_MARGIN_MIN_SEVERE: Final[Decimal] = Decimal("-100")
    NET_MARGIN_MAX_EXCEPTIONAL: Final[Decimal] = Decimal("50")
    
    EBITDA_MARGIN_MIN_SEVERE: Final[Decimal] = Decimal("-30")
    EBITDA_MARGIN_MAX_EXCEPTIONAL: Final[Decimal] = Decimal("60")
    
    ROA_MIN_SEVERE: Final[Decimal] = Decimal("-30")
    ROA_MAX_EXCEPTIONAL: Final[Decimal] = Decimal("40")
    
    ROE_MIN_SEVERE: Final[Decimal] = Decimal("-50")
    ROE_MAX_HIGH_LEVERAGE: Final[Decimal] = Decimal("80")
    
    ROCE_MIN_SEVERE: Final[Decimal] = Decimal("-20")
    ROCE_MAX_EXCEPTIONAL: Final[Decimal] = Decimal("50")


# ============================================================================
# RECOMMENDATION THRESHOLDS (Issue #55 fix)
# ============================================================================

class RecommendationThresholds:
    """Centralized thresholds for generating recommendations."""
    
    # Net Profit Margin
    NPM_LOW: Final[Decimal] = Decimal("5")
    NPM_HIGH: Final[Decimal] = Decimal("20")
    
    # Current Ratio
    CURRENT_RATIO_LOW: Final[Decimal] = Decimal("1")
    CURRENT_RATIO_HIGH: Final[Decimal] = Decimal("3")
    
    # Quick Ratio
    QUICK_RATIO_LOW: Final[Decimal] = Decimal("0.8")
    QUICK_RATIO_HIGH: Final[Decimal] = Decimal("2")
    
    # Debt to Equity
    DEBT_TO_EQUITY_LOW: Final[Decimal] = Decimal("0.5")
    DEBT_TO_EQUITY_HIGH: Final[Decimal] = Decimal("2")
    
    # ROE
    ROE_LOW: Final[Decimal] = Decimal("10")
    ROE_HIGH: Final[Decimal] = Decimal("25")
    
    # Interest Coverage
    INTEREST_COVERAGE_LOW: Final[Decimal] = Decimal("2")
    INTEREST_COVERAGE_SAFE: Final[Decimal] = Decimal("5")


# ============================================================================
# CONFIDENCE SCORING WEIGHTS (Issue #57 fix)
# ============================================================================

class ScoringWeights:
    """Centralized weights for confidence scoring."""
    
    # Penalty per issue type
    VALIDATION_ERROR: Final[float] = 20.0
    VALIDATION_WARNING: Final[float] = 5.0
    IMPLAUSIBLE_METRIC: Final[float] = 10.0
    RECONCILIATION_FAILURE: Final[float] = 15.0
    COMPLETENESS_MAX_PENALTY: Final[float] = 30.0
    
    # Thresholds for confidence levels
    HIGH_THRESHOLD: Final[float] = 80.0
    MEDIUM_THRESHOLD: Final[float] = 50.0
    
    # Maximum score
    MAX_SCORE: Final[float] = 100.0


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


# ============================================================================
# REPORT FORMATTING (Issue #60 fix)
# ============================================================================

REPORT_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M"
REPORT_DATE_FORMAT_ISO: Final[str] = "%Y-%m-%dT%H:%M:%S"
REPORT_DATE_FORMAT_LONG: Final[str] = "%B %d, %Y at %H:%M"


# ============================================================================
# DEBUG CONFIGURATION
# ============================================================================

import os

DEBUG: Final[bool] = os.environ.get("FINANALYST_DEBUG", "false").lower() == "true"
