Implementation Plan: Batch 1 (Files 1-5)
Executive Summary
This batch covers the foundational infrastructure files that all other modules depend on. Fixing these first ensures downstream modules have stable, correct foundations.

Batch 1 File List
#	File	Issues Addressed	Priority Fixes
1	config.py	#1, #3, #4 + new constants	GST Decimal, missing metrics, thresholds
2	exceptions.py	#5 + enhancements	Error code acronyms, new exception types
3	utils/__init__.py	Export updates	New function exports
4	utils/math_ops.py	#6, #7, #8, #9 + new functions	CAGR precision, None handling, statistics
5	utils/formatting.py	#10, #11, #12 + enhancements	Pipe escaping, safe conversion, smart truncation
Detailed Implementation Plan
File 1: config.py
Changes:

Change	Type	Issue #	Description
GST_RATE type	Fix	#1	Change from float to Decimal
Add GST history	New	#13 (prep)	Historical GST rates for date-aware calculations
Add missing metrics	Fix	#3	Add CCC, FCF, EPS, etc. to METRIC_FORMULAS
Widen ROE range	Fix	#4	Expand to (-150%, 150%)
Add ScoringWeights	New	#57 (prep)	Centralize confidence scoring weights
Add RecommendationThresholds	New	#55 (prep)	Centralize recommendation thresholds
Add ProfitabilityWarningThresholds	New	#44 (prep)	Centralize margin warning thresholds
Add REPORT_DATE_FORMAT	New	#60 (prep)	Centralize date formatting
File 2: exceptions.py
Changes:

Change	Type	Issue #	Description
Fix error code generation	Fix	#5	Handle acronyms (XML, IO) properly
Add TimeoutError	New	—	For long-running analyses
Add CurrencyConversionError	New	—	For future FX operations
Add ConfigurationError	New	—	For invalid configuration
File 3: utils/__init__.py
Changes:

Change	Type	Description
Add version	New	__version__ = "3.2.1"
Add new math exports	Update	median, percentile, CV functions
Add new formatting exports	Update	smart_truncate, escape functions
Add new currency exports	Update	get_gst_rate, historical functions
File 4: utils/math_ops.py
Changes:

Change	Type	Issue #	Description
Fix CAGR precision	Fix	#6	Use high-precision Decimal arithmetic
Add return_none_on_zero	Fix	#7	Consistent None vs Zero behavior
Fix is_effectively_zero	Fix	#8	Native Decimal comparison
Fix std_dev precision	Fix	#9	Use Decimal.sqrt()
Add calculate_median	New	—	Statistical function
Add calculate_percentile	New	—	Statistical function
Add calculate_coefficient_of_variation	New	—	Volatility metric
File 5: utils/formatting.py
Changes:

Change	Type	Issue #	Description
Add pipe escaping	Fix	#10	Escape `
Fix days conversion	Fix	#11	Safe int conversion with fallback
Add symbol_position	Fix	#12	Support prefix/suffix currency symbols
Add smart_truncate	New	#59 (prep)	Smart text truncation at word boundaries
Add escape_markdown	New	—	General markdown escaping
Implementation: Batch 1 Complete Files
File 1 of 5: finanalyst_tools/config.py
Python

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
File 2 of 5: finanalyst_tools/exceptions.py
Python

# finanalyst_tools/exceptions.py
"""
Custom exception hierarchy for FinAnalyst-Pro Agent Tools.

Provides specific exception types for different error categories:
- Calculation errors (arithmetic, division by zero, invalid inputs)
- Validation errors (schema, reconciliation, plausibility)
- Data errors (parsing, missing data)
- Tool errors (not found, execution failure)
- System errors (timeout, configuration)

All exceptions support:
- Serialization to dict/JSON for structured error handling
- Auto-generated error codes for programmatic handling
- Actionable suggestions for LLM consumption

Version: 3.2.1 - Addresses Issue #5 (error code acronym handling)
"""

from __future__ import annotations

import json
import re
from typing import Any


# ============================================================================
# BASE EXCEPTION
# ============================================================================

class FinAnalystError(Exception):
    """
    Base exception for all FinAnalyst-Pro errors.
    
    Features:
    - Auto-generated error_code from class name (with acronym handling)
    - Optional details dictionary for context
    - Optional suggestion for resolution
    - JSON serialization support
    
    All custom exceptions should inherit from this class.
    """
    
    def __init__(
        self, 
        message: str, 
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
        suggestion: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional context
            error_code: Optional error code (auto-generated if not provided)
            suggestion: Optional actionable suggestion for resolution
            **kwargs: Additional key-value pairs to include in details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.details.update(kwargs)
        self.error_code = error_code or self._generate_error_code()
        self.suggestion = suggestion
    
    def _generate_error_code(self) -> str:
        """
        Generate error code from class name.
        
        Handles acronyms properly (Issue #5 fix):
        - XMLParsingError → XML_PARSING
        - IOError → IO
        - DivisionByZeroError → DIVISION_BY_ZERO
        
        Returns:
            SCREAMING_SNAKE_CASE error code
        """
        name = self.__class__.__name__
        
        # Remove 'Error' suffix
        if name.endswith("Error"):
            name = name[:-5]
        
        # Handle acronyms and camelCase conversion
        # First, insert underscore before sequences of uppercase followed by lowercase
        # e.g., "XMLParsing" → "XML_Parsing"
        result = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
        
        # Then, insert underscore before single uppercase letters following lowercase
        # e.g., "divisionBy" → "division_By"
        result = re.sub(r'([a-z])([A-Z])', r'\1_\2', result)
        
        return result.upper()
    
    @property
    def error_type(self) -> str:
        """Get the exception class name."""
        return self.__class__.__name__
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        result = {
            "error_type": self.error_type,
            "error_code": self.error_code,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        if self.suggestion:
            result["suggestion"] = self.suggestion
        return result
    
    def to_json(self) -> str:
        """Convert exception to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def __str__(self) -> str:
        """Format error message with details."""
        parts = [f"[{self.error_code}] {self.message}"]
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)
    
    def __repr__(self) -> str:
        if self.details:
            return f"{self.error_type}({self.message!r}, details={self.details!r})"
        return f"{self.error_type}({self.message!r})"


# ============================================================================
# CALCULATION ERRORS
# ============================================================================

class CalculationError(FinAnalystError):
    """
    Base exception for calculation-related errors.
    
    Raised when a financial calculation cannot be completed
    due to mathematical issues or invalid inputs.
    """
    
    def __init__(
        self,
        message: str,
        metric_name: str | None = None,
        formula: str | None = None,
        suggestion: str | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if metric_name:
            details["metric_name"] = metric_name
        if formula:
            details["formula"] = formula
        super().__init__(
            message, 
            details=details, 
            suggestion=suggestion or "Check input values and try again"
        )


class DivisionByZeroError(CalculationError):
    """
    Raised when a calculation would result in division by zero.
    
    Includes information about the numerator and denominator
    to aid in debugging data issues.
    """
    
    def __init__(
        self,
        numerator: Any,
        denominator_name: str = "denominator",
        metric_name: str | None = None,
        **kwargs: Any
    ) -> None:
        message = f"Cannot divide {numerator} by zero ({denominator_name} is zero)"
        super().__init__(
            message,
            metric_name=metric_name,
            suggestion=f"Ensure {denominator_name} is non-zero before calculation",
            numerator=numerator,
            denominator_name=denominator_name,
            **kwargs
        )


class InvalidInputError(CalculationError):
    """
    Raised when input values are invalid for calculation.
    
    Examples:
    - Negative values where positive required
    - Wrong data types
    - Values outside acceptable ranges
    """
    
    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        actual_value: Any = None,
        expected: str | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if field_name:
            details["field_name"] = field_name
        if actual_value is not None:
            details["actual_value"] = actual_value
        if expected:
            details["expected"] = expected
        
        suggestion = f"Provide a value that is: {expected}" if expected else None
        super().__init__(message, suggestion=suggestion, **details)


class NegativeEquityError(CalculationError):
    """
    Raised when ROE or similar calculations encounter negative equity.
    
    Negative equity makes certain financial ratios mathematically
    valid but financially meaningless or misleading.
    """
    
    def __init__(
        self,
        metric_name: str,
        equity_value: Any,
        net_income: Any | None = None,
        **kwargs: Any
    ) -> None:
        message = f"Cannot calculate meaningful {metric_name} with negative equity ({equity_value})"
        
        suggestion = "Review equity position. Negative equity indicates liabilities exceed assets."
        if net_income is not None and float(net_income) > 0:
            suggestion += " Positive earnings with negative equity produces misleading ratios."
        
        super().__init__(
            message,
            metric_name=metric_name,
            suggestion=suggestion,
            equity_value=equity_value,
            net_income=net_income,
            **kwargs
        )


# ============================================================================
# VALIDATION ERRORS
# ============================================================================

class ValidationError(FinAnalystError):
    """
    Base exception for validation-related errors.
    
    Raised when data fails validation checks.
    """
    
    def __init__(
        self,
        message: str,
        field: str | None = None,
        validation_type: str | None = None,
        suggestion: str | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if field:
            details["field"] = field
        if validation_type:
            details["validation_type"] = validation_type
        super().__init__(message, details=details, suggestion=suggestion)


class SchemaValidationError(ValidationError):
    """Raised when data doesn't conform to expected schema."""
    
    def __init__(
        self,
        message: str,
        schema_name: str | None = None,
        field_errors: dict[str, str] | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if schema_name:
            details["schema_name"] = schema_name
        if field_errors:
            details["field_errors"] = field_errors
        super().__init__(
            message,
            validation_type="schema",
            suggestion="Verify data structure matches expected schema",
            **details
        )


class DataCompletenessError(ValidationError):
    """Raised when required data is missing for an analysis."""
    
    def __init__(
        self,
        analysis_type: str,
        missing_fields: list[str],
        **kwargs: Any
    ) -> None:
        message = (
            f"Insufficient data for {analysis_type} analysis. "
            f"Missing: {', '.join(missing_fields)}"
        )
        super().__init__(
            message,
            validation_type="completeness",
            suggestion=f"Provide the following fields: {', '.join(missing_fields)}",
            analysis_type=analysis_type,
            missing_fields=missing_fields,
            **kwargs
        )


class ReconciliationError(ValidationError):
    """
    Raised when cross-statement reconciliation fails.
    
    Indicates that values that should match across statements
    are inconsistent beyond acceptable tolerance.
    """
    
    def __init__(
        self,
        message: str,
        check_name: str,
        expected_value: Any,
        actual_value: Any,
        difference: Any = None,
        tolerance: float | None = None,
        **kwargs: Any
    ) -> None:
        details = {
            "check_name": check_name,
            "expected_value": expected_value,
            "actual_value": actual_value,
        }
        if difference is not None:
            details["difference"] = difference
        if tolerance is not None:
            details["tolerance_used"] = tolerance
        details.update(kwargs)
        
        super().__init__(
            message,
            validation_type="reconciliation",
            suggestion="Verify data accuracy or confirm known discrepancy",
            **details
        )


class PlausibilityError(ValidationError):
    """
    Raised when a calculated metric is outside plausible range.
    
    Note: This is typically a warning, not an error, unless explicitly strict.
    """
    
    def __init__(
        self,
        metric_name: str,
        value: float,
        plausible_range: tuple[float, float],
        **kwargs: Any
    ) -> None:
        message = (
            f"{metric_name} value of {value:.2f} is outside the plausible range "
            f"({plausible_range[0]:.2f} to {plausible_range[1]:.2f})"
        )
        super().__init__(
            message,
            validation_type="plausibility",
            suggestion="Verify input data accuracy. This may indicate a data entry error.",
            metric_name=metric_name,
            value=value,
            min_plausible=plausible_range[0],
            max_plausible=plausible_range[1],
            **kwargs
        )


# ============================================================================
# DATA ERRORS
# ============================================================================

class DataError(FinAnalystError):
    """Base exception for data-related errors."""
    pass


class DataParsingError(DataError):
    """
    Raised when data cannot be parsed from input format.
    """
    
    def __init__(
        self,
        message: str,
        source: str | None = None,
        line_number: int | None = None,
        raw_data: str | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if source:
            details["source"] = source
        if line_number is not None:
            details["line_number"] = line_number
        if raw_data:
            # Truncate raw data for readability
            details["raw_data"] = raw_data[:500] if len(raw_data) > 500 else raw_data
        super().__init__(
            message, 
            details=details,
            suggestion="Check source format and encoding"
        )


class MissingDataError(DataError):
    """Raised when required data is missing."""
    
    def __init__(
        self,
        message: str,
        missing_fields: list[str] | None = None,
        required_for: str | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if missing_fields:
            details["missing_fields"] = missing_fields
        if required_for:
            details["required_for"] = required_for
        
        suggestion = None
        if missing_fields:
            suggestion = f"Provide the following fields: {', '.join(missing_fields)}"
        
        super().__init__(message, details=details, suggestion=suggestion)


# ============================================================================
# TOOL ERRORS
# ============================================================================

class ToolError(FinAnalystError):
    """Base exception for tool-related errors."""
    
    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if tool_name:
            details["tool_name"] = tool_name
        super().__init__(message, details=details)


class ToolNotFoundError(ToolError):
    """Raised when a requested tool does not exist."""
    
    def __init__(
        self,
        tool_name: str,
        available_tools: list[str] | None = None,
        **kwargs: Any
    ) -> None:
        suggestions = self._find_similar(tool_name, available_tools or [])
        message = f"Tool '{tool_name}' not found"
        if suggestions:
            message += f". Did you mean: {', '.join(suggestions)}?"
        
        super().__init__(
            message,
            tool_name=tool_name,
            suggestion=f"Use one of the available tools",
            available_tools=available_tools[:10] if available_tools else None,
            **kwargs
        )
    
    @staticmethod
    def _find_similar(name: str, available: list[str]) -> list[str]:
        """Find tools with similar names using basic substring matching."""
        if not available:
            return []
        
        name_lower = name.lower()
        
        # Find tools containing the search term or vice versa
        similar = [
            t for t in available 
            if name_lower in t.lower() or t.lower() in name_lower
        ]
        
        if similar:
            return similar[:3]
        
        # Fall back to tools starting with same letter
        same_start = [t for t in available if t.lower().startswith(name_lower[0])]
        return same_start[:3] if same_start else available[:3]


class ToolExecutionError(ToolError):
    """Raised when a tool fails during execution."""
    
    def __init__(
        self,
        tool_name: str,
        original_error: Exception,
        parameters: dict[str, Any] | None = None,
        traceback_str: str | None = None,
        **kwargs: Any
    ) -> None:
        message = f"Tool '{tool_name}' failed: {str(original_error)}"
        
        details = {
            "original_error_type": type(original_error).__name__,
            "original_error_message": str(original_error),
        }
        if parameters:
            details["parameters"] = parameters
        if traceback_str:
            details["traceback"] = traceback_str
        details.update(kwargs)
        
        super().__init__(
            message,
            tool_name=tool_name,
            suggestion="Check tool parameters and try again",
            **details
        )
        self.original_error = original_error


class ToolParameterError(ToolError):
    """Raised when tool parameters are invalid."""
    
    def __init__(
        self,
        tool_name: str,
        parameter_name: str,
        message: str,
        expected_type: str | None = None,
        actual_value: Any = None,
        allowed_values: list[str] | None = None,
        **kwargs: Any
    ) -> None:
        full_message = f"Parameter '{parameter_name}' for tool '{tool_name}': {message}"
        
        details = {
            "parameter_name": parameter_name,
        }
        if expected_type:
            details["expected_type"] = expected_type
        if actual_value is not None:
            details["actual_value"] = str(actual_value)[:100]
        if allowed_values:
            details["allowed_values"] = allowed_values
        details.update(kwargs)
        
        suggestion = f"Provide a valid value for '{parameter_name}'"
        if allowed_values:
            suggestion += f". Allowed values: {', '.join(allowed_values)}"
        
        super().__init__(
            full_message,
            tool_name=tool_name,
            suggestion=suggestion,
            **details
        )


# ============================================================================
# SYSTEM ERRORS (New in v3.2.1)
# ============================================================================

class SystemError(FinAnalystError):
    """Base exception for system-level errors."""
    pass


class TimeoutError(SystemError):
    """Raised when an operation exceeds time limit."""
    
    def __init__(
        self,
        operation: str,
        timeout_seconds: float,
        **kwargs: Any
    ) -> None:
        message = f"Operation '{operation}' timed out after {timeout_seconds:.1f} seconds"
        super().__init__(
            message,
            suggestion="Try with simpler inputs or increase timeout limit",
            operation=operation,
            timeout_seconds=timeout_seconds,
            **kwargs
        )


class ConfigurationError(SystemError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        current_value: Any = None,
        expected: str | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if config_key:
            details["config_key"] = config_key
        if current_value is not None:
            details["current_value"] = str(current_value)
        if expected:
            details["expected"] = expected
        
        super().__init__(
            message,
            details=details,
            suggestion="Review configuration settings"
        )


class CurrencyConversionError(FinAnalystError):
    """Raised when currency conversion fails."""
    
    def __init__(
        self,
        source_currency: str,
        target_currency: str,
        reason: str = "conversion rate unavailable",
        **kwargs: Any
    ) -> None:
        message = f"Cannot convert {source_currency} to {target_currency}: {reason}"
        super().__init__(
            message,
            suggestion="Verify currency codes or provide exchange rate",
            source_currency=source_currency,
            target_currency=target_currency,
            **kwargs
        )
File 3 of 5: finanalyst_tools/utils/__init__.py
Python

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
File 4 of 5: finanalyst_tools/utils/math_ops.py
Python

# finanalyst_tools/utils/math_ops.py
"""
Safe mathematical operations for financial calculations.

This module provides Decimal-based arithmetic functions that:
- Ensure precision (no floating-point errors)
- Handle edge cases gracefully (division by zero, None values)
- Support configurable rounding
- Provide statistical functions

All monetary and ratio calculations should use these functions
to ensure consistency and accuracy.

Version: 3.2.1 - Addresses Issues #6, #7, #8, #9 + new statistical functions
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, getcontext, ROUND_HALF_UP
from typing import Sequence, Any, overload
import math

from finanalyst_tools.config import (
    DECIMAL_PLACES,
    DEFAULT_ROUNDING,
    ZERO_THRESHOLD,
    RoundingMode,
)
from finanalyst_tools.exceptions import (
    DivisionByZeroError,
    InvalidInputError,
)


# Type alias for numeric types
Numeric = int | float | Decimal


# Set high precision for intermediate calculations
getcontext().prec = 50


# ============================================================================
# TYPE CONVERSION
# ============================================================================

def to_decimal(
    value: Numeric | str | None,
    default: Decimal | None = None,
    precision: int | None = None,
) -> Decimal:
    """
    Safely convert a value to Decimal.
    
    Handles various input types and returns a default value
    for None or unconvertible inputs.
    
    Args:
        value: Value to convert (int, float, str, Decimal, or None)
        default: Default value if conversion fails (default: Decimal("0"))
        precision: Optional precision to round to
        
    Returns:
        Decimal representation of the value
        
    Examples:
        >>> to_decimal(100)
        Decimal('100')
        >>> to_decimal("123.45")
        Decimal('123.45')
        >>> to_decimal(None, default=Decimal("0"))
        Decimal('0')
    """
    if default is None:
        default = Decimal("0")
    
    if value is None:
        return default
    
    try:
        if isinstance(value, Decimal):
            result = value
        elif isinstance(value, float):
            # Use string conversion to avoid float precision issues
            # Handle special float values
            if math.isnan(value) or math.isinf(value):
                return default
            result = Decimal(str(value))
        else:
            result = Decimal(str(value))
        
        if precision is not None:
            result = round_decimal(result, precision)
        
        return result
    except (InvalidOperation, ValueError, TypeError):
        return default


def is_effectively_zero(
    value: Numeric | None, 
    threshold: Numeric | None = None,
) -> bool:
    """
    Check if a value is effectively zero (within threshold).
    
    Uses Decimal arithmetic to avoid float precision issues (Issue #8 fix).
    
    Args:
        value: Value to check
        threshold: Threshold for zero comparison (default: ZERO_THRESHOLD)
        
    Returns:
        True if value is None or within threshold of zero
    """
    if value is None:
        return True
    
    # Use Decimal for comparison to avoid float precision issues
    if threshold is None:
        threshold_dec = ZERO_THRESHOLD
    else:
        threshold_dec = to_decimal(threshold)
    
    value_dec = to_decimal(value)
    
    return abs(value_dec) < threshold_dec


# ============================================================================
# SAFE ARITHMETIC
# ============================================================================

def safe_divide(
    numerator: Numeric | None,
    denominator: Numeric | None,
    default: Decimal | None = None,
    precision: int = DECIMAL_PLACES["ratio"],
    raise_on_zero: bool = False,
    return_none_on_zero: bool = False,
) -> Decimal | None:
    """
    Safely divide two numbers with zero handling.
    
    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return if division is impossible (default: Decimal("0"))
        precision: Decimal places to round result to
        raise_on_zero: If True, raise DivisionByZeroError instead of returning default
        return_none_on_zero: If True, return None instead of default on zero (Issue #7 fix)
        
    Returns:
        Result of division, or default/None if denominator is zero/None
        
    Raises:
        DivisionByZeroError: If raise_on_zero is True and denominator is zero
        
    Examples:
        >>> safe_divide(100, 4)
        Decimal('25.0000')
        >>> safe_divide(100, 0)
        Decimal('0')
        >>> safe_divide(100, 0, return_none_on_zero=True)
        None
        >>> safe_divide(100, 0, raise_on_zero=True)
        DivisionByZeroError: Cannot divide 100 by zero
    """
    if default is None:
        default = Decimal("0")
    
    if numerator is None:
        return None if return_none_on_zero else default
    
    num = to_decimal(numerator)
    denom = to_decimal(denominator)
    
    if is_effectively_zero(denom):
        if raise_on_zero:
            raise DivisionByZeroError(
                numerator=float(num),
                denominator_name="denominator",
            )
        return None if return_none_on_zero else default
    
    result = num / denom
    return round_decimal(result, precision)


def round_decimal(
    value: Numeric | None,
    precision: int = 2,
    rounding: RoundingMode = DEFAULT_ROUNDING,
) -> Decimal:
    """
    Round a Decimal value to specified precision.
    
    Args:
        value: Value to round
        precision: Number of decimal places
        rounding: Rounding mode to use
        
    Returns:
        Rounded Decimal value
    """
    if value is None:
        return Decimal("0")
    
    dec_value = to_decimal(value)
    quantize_str = "0." + "0" * precision if precision > 0 else "1"
    return dec_value.quantize(Decimal(quantize_str), rounding=rounding.get_decimal_rounding())


def clamp_value(
    value: Numeric,
    min_value: Numeric | None = None,
    max_value: Numeric | None = None,
) -> Decimal:
    """
    Clamp a value within a range.
    
    Args:
        value: Value to clamp
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        
    Returns:
        Clamped value as Decimal
    """
    result = to_decimal(value)
    
    if min_value is not None:
        min_dec = to_decimal(min_value)
        if result < min_dec:
            result = min_dec
    
    if max_value is not None:
        max_dec = to_decimal(max_value)
        if result > max_dec:
            result = max_dec
    
    return result


# ============================================================================
# PERCENTAGE & GROWTH CALCULATIONS
# ============================================================================

def calculate_percentage(
    part: Numeric | None,
    whole: Numeric | None,
    precision: int = DECIMAL_PLACES["percentage"],
    return_none_on_zero: bool = False,
) -> Decimal | None:
    """
    Calculate percentage: (part / whole) × 100.
    
    Args:
        part: The numerator
        whole: The denominator
        precision: Decimal places for result
        return_none_on_zero: If True, return None when whole is zero (Issue #7 fix)
        
    Returns:
        Percentage value (e.g., 25.00 for 25%), or None if return_none_on_zero
    """
    if part is None or whole is None:
        return None if return_none_on_zero else Decimal("0")
    
    ratio = safe_divide(
        part, 
        whole, 
        precision=precision + 2,
        return_none_on_zero=return_none_on_zero,
    )
    
    if ratio is None:
        return None
    
    return round_decimal(ratio * 100, precision)


def calculate_growth_rate(
    current: Numeric | None,
    previous: Numeric | None,
    precision: int = DECIMAL_PLACES["percentage"],
) -> Decimal | None:
    """
    Calculate period-over-period growth rate.
    
    Formula: ((current - previous) / |previous|) × 100
    
    Args:
        current: Current period value
        previous: Previous period value
        precision: Decimal places for result
        
    Returns:
        Growth rate as percentage, or None if calculation impossible
    """
    if current is None or previous is None:
        return None
    
    curr = to_decimal(current)
    prev = to_decimal(previous)
    
    if is_effectively_zero(prev):
        return None
    
    change = curr - prev
    rate = (change / abs(prev)) * 100
    return round_decimal(rate, precision)


def calculate_cagr(
    beginning_value: Numeric | None,
    ending_value: Numeric | None,
    periods: int,
    precision: int = DECIMAL_PLACES["percentage"],
) -> Decimal | None:
    """
    Calculate Compound Annual Growth Rate (CAGR).
    
    Formula: ((ending / beginning) ^ (1/periods) - 1) × 100
    
    Uses high-precision Decimal arithmetic (Issue #6 fix).
    
    Args:
        beginning_value: Value at start
        ending_value: Value at end
        periods: Number of periods (years)
        precision: Decimal places for result
        
    Returns:
        CAGR as percentage, or None if calculation impossible
    """
    if beginning_value is None or ending_value is None or periods <= 0:
        return None
    
    begin = to_decimal(beginning_value)
    end = to_decimal(ending_value)
    
    # Both values must be positive for CAGR
    if begin <= 0 or end <= 0:
        return None
    
    try:
        # Use high-precision Decimal arithmetic
        # CAGR = (end/begin)^(1/n) - 1
        # Using logarithms: CAGR = exp(ln(end/begin) / n) - 1
        
        ratio = end / begin
        
        # Use Python's math.log for the logarithm, then convert back to Decimal
        # This is a pragmatic approach that maintains good precision
        # while avoiding the complexity of implementing ln() in pure Decimal
        
        # Ensure we're working with sufficient precision
        getcontext().prec = 50
        
        ln_ratio = Decimal(str(math.log(float(ratio))))
        exponent = ln_ratio / Decimal(str(periods))
        
        # Calculate exp(exponent) using Decimal
        # exp(x) ≈ 1 + x + x²/2! + x³/3! + ... (Taylor series)
        growth_factor = _decimal_exp(exponent)
        
        cagr = (growth_factor - Decimal("1")) * Decimal("100")
        return round_decimal(cagr, precision)
        
    except (ValueError, ZeroDivisionError, OverflowError, InvalidOperation):
        return None


def _decimal_exp(x: Decimal, iterations: int = 50) -> Decimal:
    """
    Calculate e^x using Taylor series with Decimal precision.
    
    Args:
        x: Exponent
        iterations: Number of Taylor series terms
        
    Returns:
        e^x as Decimal
    """
    getcontext().prec = 50
    
    result = Decimal("1")
    term = Decimal("1")
    
    for i in range(1, iterations):
        term *= x / Decimal(str(i))
        result += term
        
        # Early termination if term is negligible
        if abs(term) < Decimal("1e-40"):
            break
    
    return result


# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

def calculate_average(
    *values: Numeric | None,
    precision: int = DECIMAL_PLACES["ratio"],
) -> Decimal:
    """
    Calculate arithmetic mean of values.
    
    Args:
        *values: Values to average
        precision: Decimal places for result
        
    Returns:
        Arithmetic mean, or Decimal("0") if no valid values
    """
    valid_values = [to_decimal(v) for v in values if v is not None]
    
    if not valid_values:
        return Decimal("0")
    
    total = sum(valid_values)
    return round_decimal(total / len(valid_values), precision)


def calculate_weighted_average(
    values: Sequence[Numeric | None],
    weights: Sequence[Numeric | None],
    precision: int = DECIMAL_PLACES["ratio"],
) -> Decimal | None:
    """
    Calculate weighted average of values.
    
    Args:
        values: Values to average
        weights: Corresponding weights
        precision: Decimal places for result
        
    Returns:
        Weighted average, or None if calculation impossible
    """
    if len(values) != len(weights):
        return None
    
    pairs = [
        (to_decimal(v), to_decimal(w))
        for v, w in zip(values, weights)
        if v is not None and w is not None
    ]
    
    if not pairs:
        return None
    
    weighted_sum = sum(v * w for v, w in pairs)
    total_weight = sum(w for _, w in pairs)
    
    if is_effectively_zero(total_weight):
        return None
    
    return round_decimal(weighted_sum / total_weight, precision)


def calculate_variance(
    values: Sequence[Numeric | None],
    population: bool = True,
) -> Decimal | None:
    """
    Calculate variance of values.
    
    Args:
        values: Values to calculate variance for
        population: If True, use population variance (N); else sample variance (N-1)
        
    Returns:
        Variance, or None if insufficient values
    """
    valid_values = [to_decimal(v) for v in values if v is not None]
    n = len(valid_values)
    
    if n < 2:
        return None
    
    mean = sum(valid_values) / n
    squared_diffs = [(v - mean) ** 2 for v in valid_values]
    
    divisor = n if population else (n - 1)
    return round_decimal(sum(squared_diffs) / divisor, 6)


def calculate_std_dev(
    values: Sequence[Numeric | None],
    population: bool = True,
) -> Decimal | None:
    """
    Calculate standard deviation of values.
    
    Uses Decimal.sqrt() for precision (Issue #9 fix).
    
    Args:
        values: Values to calculate std dev for
        population: If True, use population std dev; else sample std dev
        
    Returns:
        Standard deviation, or None if insufficient values
    """
    variance = calculate_variance(values, population)
    if variance is None:
        return None
    
    # Use Decimal.sqrt() for precision (Issue #9 fix)
    std_dev = variance.sqrt()
    return round_decimal(std_dev, 4)


def calculate_min_max(
    values: Sequence[Numeric | None],
) -> tuple[Decimal, Decimal] | None:
    """
    Calculate minimum and maximum of values.
    
    Args:
        values: Values to find min/max for
        
    Returns:
        Tuple of (min, max), or None if no valid values
    """
    valid_values = [to_decimal(v) for v in values if v is not None]
    
    if not valid_values:
        return None
    
    return (min(valid_values), max(valid_values))


def calculate_median(
    values: Sequence[Numeric | None],
    precision: int = DECIMAL_PLACES["ratio"],
) -> Decimal | None:
    """
    Calculate median of values.
    
    Args:
        values: Values to calculate median for
        precision: Decimal places for result
        
    Returns:
        Median value, or None if no valid values
    """
    valid_values = sorted([to_decimal(v) for v in values if v is not None])
    
    if not valid_values:
        return None
    
    n = len(valid_values)
    mid = n // 2
    
    if n % 2 == 0:
        # Even number of values - average the two middle values
        median = (valid_values[mid - 1] + valid_values[mid]) / 2
    else:
        # Odd number of values - take the middle value
        median = valid_values[mid]
    
    return round_decimal(median, precision)


def calculate_percentile(
    values: Sequence[Numeric | None],
    percentile: float,
    precision: int = DECIMAL_PLACES["ratio"],
) -> Decimal | None:
    """
    Calculate nth percentile of values using linear interpolation.
    
    Args:
        values: Values to calculate percentile for
        percentile: Percentile to calculate (0-100)
        precision: Decimal places for result
        
    Returns:
        Percentile value, or None if no valid values
    """
    if not 0 <= percentile <= 100:
        return None
    
    valid_values = sorted([to_decimal(v) for v in values if v is not None])
    
    if not valid_values:
        return None
    
    n = len(valid_values)
    
    if n == 1:
        return round_decimal(valid_values[0], precision)
    
    # Calculate the index
    index = (percentile / 100) * (n - 1)
    lower_idx = int(index)
    upper_idx = lower_idx + 1
    
    if upper_idx >= n:
        return round_decimal(valid_values[-1], precision)
    
    # Linear interpolation
    fraction = Decimal(str(index - lower_idx))
    lower_val = valid_values[lower_idx]
    upper_val = valid_values[upper_idx]
    
    result = lower_val + fraction * (upper_val - lower_val)
    return round_decimal(result, precision)


def calculate_coefficient_of_variation(
    values: Sequence[Numeric | None],
    as_percentage: bool = True,
) -> Decimal | None:
    """
    Calculate coefficient of variation (CV = std_dev / mean).
    
    CV is a standardized measure of dispersion, useful for comparing
    volatility across different metrics.
    
    Args:
        values: Values to calculate CV for
        as_percentage: If True, return as percentage (CV * 100)
        
    Returns:
        Coefficient of variation, or None if calculation impossible
    """
    valid_values = [to_decimal(v) for v in values if v is not None]
    
    if len(valid_values) < 2:
        return None
    
    std_dev = calculate_std_dev(valid_values, population=True)
    if std_dev is None:
        return None
    
    mean = calculate_average(*valid_values)
    
    if is_effectively_zero(mean):
        return None
    
    cv = std_dev / abs(mean)
    
    if as_percentage:
        cv = cv * 100
    
    return round_decimal(cv, 4)
File 5 of 5: finanalyst_tools/utils/formatting.py
Python

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
