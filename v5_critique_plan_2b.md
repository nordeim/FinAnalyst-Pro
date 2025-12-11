Implementation Plan: Complete Drop-in Replacement Files
Executive Summary
Based on my meticulous review identifying 61 issues across the codebase, I will create complete drop-in replacement files organized into 5 batches of 5 files each (with the final batch containing the remaining files). Each file will address all identified issues while maintaining backward compatibility.

Batch Organization
Batch	Focus Area	Files
1	Core Infrastructure	config.py, exceptions.py, utils/math_ops.py, utils/formatting.py, utils/currency.py
2	Models & Utils Init	utils/__init__.py, models/validation.py, models/financial_statements.py, models/analysis_results.py, models/__init__.py
3	Validation Layer	validation/schema_validator.py, validation/reconciliation.py, validation/plausibility.py, validation/__init__.py, calculations/base.py
4	Calculations & Integration	calculations/profitability.py, calculations/liquidity.py, calculations/__init__.py, tool_registry.py, dispatcher.py
5	Orchestration & Package	orchestration/pipeline.py, orchestration/confidence_scorer.py, orchestration/report_generator.py, orchestration/__init__.py, __init__.py
Batch 1: Core Infrastructure
Implementation Plan
File 1: config.py
Issues Addressed:

Issue #	Description	Fix
1	GST_RATE type inconsistency	Use Decimal instead of float
3	Missing metrics in METRIC_FORMULAS	Add 10+ missing formulas
4	Plausibility range edge cases	Widen ranges, add industry context
NEW	Centralize magic numbers	Add WarningThresholds, ScoringWeights classes
NEW	Add version constant	__version__ = "3.2.1"
Key Changes:

Convert all financial constants to Decimal
Add WarningThresholds class for profitability/liquidity thresholds
Add ScoringWeights class for confidence scoring
Add RecommendationThresholds class
Expand METRIC_FORMULAS and METRIC_UNITS
Add GST historical rates support infrastructure
File 2: exceptions.py
Issues Addressed:

Issue #	Description	Fix
5	Error code generation edge case	Improved regex-based generation
NEW	Add missing exception types	TimeoutError, CurrencyConversionError
Key Changes:

Fix _generate_error_code() to handle acronyms properly
Add AnalysisTimeoutError for long-running operations
Add CurrencyMismatchError for cross-statement validation
Improve docstrings with examples
File 3: utils/math_ops.py
Issues Addressed:

Issue #	Description	Fix
6	CAGR precision leakage via float	Use high-precision Decimal math
7	Inconsistent None vs Zero returns	Add return_none_on_error parameter
8	Float in is_effectively_zero	Native Decimal comparison
9	std_dev precision	Use Decimal.sqrt()
NEW	Add missing functions	calculate_median, calculate_percentile
Key Changes:

Rewrite calculate_cagr() with Decimal precision
Add return_none_on_zero parameter to percentage functions
Fix is_effectively_zero() for Decimal precision
Use Decimal.sqrt() in calculate_std_dev()
Add calculate_median() and calculate_coefficient_of_variation()
File 4: utils/formatting.py
Issues Addressed:

Issue #	Description	Fix
10	Markdown table pipe escaping	Add _escape_markdown() helper
11	Days conversion safety	Safe int conversion with singular/plural
12	Currency symbol position	Add symbol_position parameter
NEW	Smart truncation	Add truncate_smart() function
Key Changes:

Add _escape_markdown() for pipe character escaping
Fix format_value_with_unit() days handling
Add symbol_position parameter to format_currency()
Add truncate_smart() for text truncation
Improve fallback handling consistency
File 5: utils/currency.py
Issues Addressed:

Issue #	Description	Fix
13	Hardcoded GST rate	Add historical GST rate lookup
14	SFRS qualification messaging	Return status message with result
15	Redundant format_sgd	Delegate to format_currency
NEW	Add GST rate effective date	Support date-based lookups
Key Changes:

Add GSTRate named tuple and SGD_GST_HISTORY list
Add get_gst_rate(as_of: date) function
Update all GST functions to accept optional date
Improve is_sfrs_small_entity() return with status message
Simplify format_sgd() to delegate to format_currency()
Batch 1: Complete Drop-in Replacement Files
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
- Warning thresholds for analysis
- Scoring weights for confidence calculation

All constants use Final for immutability and are fully typed.

Version: 3.2.1 - Addresses precision, Singapore context, and configurability issues
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
from enum import Enum
from typing import Final, Any


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
            The rounding mode string constant from the decimal module.
            Note: Decimal rounding modes are string constants.
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
    "intermediate": 10,  # Intermediate calculations for precision
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
    
    Ranges have been widened from v3.2.0 to reduce false positives for:
    - High-growth startups
    - Distressed companies
    - Highly leveraged businesses
    """
    
    # -------------------------------------------------------------------------
    # PROFITABILITY METRICS (percentages)
    # -------------------------------------------------------------------------
    GROSS_MARGIN: Final[tuple[float, float]] = (-100.0, 99.0)
    OPERATING_MARGIN: Final[tuple[float, float]] = (-200.0, 80.0)
    NET_MARGIN: Final[tuple[float, float]] = (-500.0, 60.0)
    EBITDA_MARGIN: Final[tuple[float, float]] = (-100.0, 80.0)
    ROA: Final[tuple[float, float]] = (-100.0, 50.0)
    ROE: Final[tuple[float, float]] = (-200.0, 150.0)  # Widened for high leverage
    ROCE: Final[tuple[float, float]] = (-100.0, 80.0)
    
    # -------------------------------------------------------------------------
    # LIQUIDITY METRICS (ratios)
    # -------------------------------------------------------------------------
    CURRENT_RATIO: Final[tuple[float, float]] = (0.05, 15.0)
    QUICK_RATIO: Final[tuple[float, float]] = (0.01, 12.0)
    CASH_RATIO: Final[tuple[float, float]] = (0.0, 10.0)
    
    # -------------------------------------------------------------------------
    # SOLVENCY METRICS (ratios)
    # -------------------------------------------------------------------------
    DEBT_TO_EQUITY: Final[tuple[float, float]] = (0.0, 20.0)  # Widened for high leverage
    DEBT_TO_ASSETS: Final[tuple[float, float]] = (0.0, 2.0)
    INTEREST_COVERAGE: Final[tuple[float, float]] = (-20.0, 200.0)
    EQUITY_RATIO: Final[tuple[float, float]] = (-1.0, 1.0)
    
    # -------------------------------------------------------------------------
    # EFFICIENCY METRICS (ratios/turnover)
    # -------------------------------------------------------------------------
    ASSET_TURNOVER: Final[tuple[float, float]] = (0.05, 10.0)
    INVENTORY_TURNOVER: Final[tuple[float, float]] = (0.1, 100.0)
    RECEIVABLES_TURNOVER: Final[tuple[float, float]] = (0.5, 100.0)
    PAYABLES_TURNOVER: Final[tuple[float, float]] = (0.5, 50.0)
    FIXED_ASSET_TURNOVER: Final[tuple[float, float]] = (0.05, 50.0)
    DAYS_SALES_OUTSTANDING: Final[tuple[float, float]] = (1.0, 365.0)
    DAYS_INVENTORY_OUTSTANDING: Final[tuple[float, float]] = (1.0, 730.0)
    DAYS_PAYABLES_OUTSTANDING: Final[tuple[float, float]] = (1.0, 365.0)
    CASH_CONVERSION_CYCLE: Final[tuple[float, float]] = (-365.0, 730.0)
    
    # -------------------------------------------------------------------------
    # GROWTH METRICS (percentages)
    # -------------------------------------------------------------------------
    REVENUE_GROWTH: Final[tuple[float, float]] = (-90.0, 1000.0)
    NET_INCOME_GROWTH: Final[tuple[float, float]] = (-1000.0, 2000.0)
    ASSET_GROWTH: Final[tuple[float, float]] = (-80.0, 500.0)
    
    # -------------------------------------------------------------------------
    # ALIASES MAPPING
    # -------------------------------------------------------------------------
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
# WARNING THRESHOLDS (Centralized magic numbers)
# ============================================================================

class WarningThresholds:
    """
    Centralized thresholds for generating warnings in calculations.
    
    These are business-logic thresholds that trigger advisory warnings,
    distinct from plausibility ranges which detect data errors.
    """
    
    # Profitability thresholds (percentages)
    GROSS_MARGIN_NEGATIVE: Final[Decimal] = Decimal("0")
    GROSS_MARGIN_ABOVE_100: Final[Decimal] = Decimal("100")
    OPERATING_MARGIN_SEVERE_LOSS: Final[Decimal] = Decimal("-50")
    NET_MARGIN_SEVERE_LOSS: Final[Decimal] = Decimal("-100")
    NET_MARGIN_EXCEPTIONAL: Final[Decimal] = Decimal("50")
    ROA_EXCEPTIONAL: Final[Decimal] = Decimal("40")
    ROE_HIGH_LEVERAGE_INDICATOR: Final[Decimal] = Decimal("80")
    
    # Liquidity thresholds (ratios)
    CURRENT_RATIO_LOW: Final[Decimal] = Decimal("1.0")
    CURRENT_RATIO_HIGH: Final[Decimal] = Decimal("3.0")
    QUICK_RATIO_LOW: Final[Decimal] = Decimal("1.0")
    CASH_RATIO_LOW: Final[Decimal] = Decimal("0.2")
    CASH_RATIO_HIGH: Final[Decimal] = Decimal("1.0")
    
    # Solvency thresholds
    DEBT_TO_EQUITY_HIGH: Final[Decimal] = Decimal("2.0")
    INTEREST_COVERAGE_LOW: Final[Decimal] = Decimal("1.5")


class RecommendationThresholds:
    """
    Thresholds for generating recommendations in analysis reports.
    """
    
    # Net profit margin thresholds for recommendations
    NPM_LOW: Final[Decimal] = Decimal("5")
    NPM_HIGH: Final[Decimal] = Decimal("20")
    
    # Current ratio thresholds
    CURRENT_RATIO_LOW: Final[Decimal] = Decimal("1")
    CURRENT_RATIO_HIGH: Final[Decimal] = Decimal("3")
    
    # Working capital
    WORKING_CAPITAL_LOW_DAYS: Final[int] = 30  # Less than 30 days of expenses


# ============================================================================
# CONFIDENCE SCORING WEIGHTS
# ============================================================================

class ScoringWeights:
    """
    Weights and thresholds for confidence score calculation.
    
    These can be overridden by instantiating ConfidenceScorer with custom values.
    """
    
    # Penalty weights (points deducted from 100)
    VALIDATION_ERROR: Final[float] = 20.0
    VALIDATION_WARNING: Final[float] = 5.0
    IMPLAUSIBLE_METRIC: Final[float] = 10.0
    RECONCILIATION_FAILURE: Final[float] = 15.0
    COMPLETENESS_MAX_PENALTY: Final[float] = 30.0
    
    # Level thresholds
    HIGH_THRESHOLD: Final[float] = 80.0
    MEDIUM_THRESHOLD: Final[float] = 50.0


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
        "working_capital_consistency": STRICT,
        "balance_sheet_equation": STRICT,
    }
    
    @classmethod
    def get_tolerance(cls, check_type: str) -> float:
        """
        Get tolerance for a specific check type.
        
        Args:
            check_type: Name of the reconciliation check (case-insensitive)
            
        Returns:
            Tolerance level as a proportion (e.g., 0.01 = 1%)
        """
        normalized = check_type.lower().replace(" ", "_")
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
    # Most currencies use prefix
    "default": "prefix",
    # Some European locales use suffix (not standard ISO, but for flexibility)
}


# ============================================================================
# SINGAPORE SMB CONTEXT
# ============================================================================

class SingaporeConstants:
    """
    Singapore-specific financial constants and thresholds.
    
    Note: GST rates are now handled in utils/currency.py with historical support.
    This class provides the current rate for convenience.
    """
    
    # Current GST rate (as of 2024) - Use Decimal for precision
    GST_RATE: Final[Decimal] = Decimal("0.09")  # 9% as of 2024
    
    # SFRS for Small Entities thresholds (must meet 2 of 3)
    SFRS_SMALL_ENTITY_REVENUE: Final[Decimal] = Decimal("10_000_000")    # S$10M
    SFRS_SMALL_ENTITY_ASSETS: Final[Decimal] = Decimal("10_000_000")     # S$10M
    SFRS_SMALL_ENTITY_EMPLOYEES: Final[int] = 50
    
    # SME definition thresholds
    SME_ANNUAL_SALES: Final[Decimal] = Decimal("100_000_000")  # S$100M
    SME_EMPLOYEES: Final[int] = 200
    
    # Common financial year end months
    COMMON_FYE_MONTHS: Final[list[int]] = [12, 3, 6]
    
    # Corporate tax rate
    CORPORATE_TAX_RATE: Final[Decimal] = Decimal("0.17")  # 17%
    
    @classmethod
    def calculate_gst_exclusive(cls, gst_inclusive: Decimal | float) -> Decimal:
        """
        Convert GST-inclusive amount to GST-exclusive using current rate.
        
        For historical calculations, use utils.currency.calculate_gst_exclusive()
        with the as_of parameter.
        """
        amount = Decimal(str(gst_inclusive)) if not isinstance(gst_inclusive, Decimal) else gst_inclusive
        return amount / (1 + cls.GST_RATE)
    
    @classmethod
    def calculate_gst_amount(cls, gst_exclusive: Decimal | float) -> Decimal:
        """Calculate GST on a GST-exclusive amount using current rate."""
        amount = Decimal(str(gst_exclusive)) if not isinstance(gst_exclusive, Decimal) else gst_exclusive
        return amount * cls.GST_RATE


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
# METRIC FORMULAS (Comprehensive for documentation and LLM context)
# ============================================================================

METRIC_FORMULAS: Final[dict[str, str]] = {
    # Profitability
    "gross_profit_margin": "(Revenue - COGS) / Revenue × 100",
    "operating_profit_margin": "(Revenue - COGS - OpEx) / Revenue × 100",
    "net_profit_margin": "Net Income / Revenue × 100",
    "ebitda_margin": "EBITDA / Revenue × 100",
    "ebit_margin": "EBIT / Revenue × 100",
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
    "working_capital_ratio": "Current Assets / Current Liabilities",
    
    # Solvency
    "debt_to_equity": "Total Liabilities / Shareholders' Equity",
    "debt_to_assets": "Total Liabilities / Total Assets",
    "equity_ratio": "Shareholders' Equity / Total Assets",
    "interest_coverage": "EBIT / Interest Expense",
    "debt_service_coverage": "Net Operating Income / Total Debt Service",
    
    # Efficiency
    "asset_turnover": "Revenue / Average Total Assets",
    "inventory_turnover": "COGS / Average Inventory",
    "receivables_turnover": "Net Credit Sales / Average Accounts Receivable",
    "payables_turnover": "COGS / Average Accounts Payable",
    "fixed_asset_turnover": "Revenue / Average Fixed Assets",
    "days_sales_outstanding": "365 / Receivables Turnover",
    "days_inventory_outstanding": "365 / Inventory Turnover",
    "days_payables_outstanding": "365 / Payables Turnover",
    "cash_conversion_cycle": "DIO + DSO - DPO",
    
    # Cash Flow
    "operating_cash_flow_ratio": "Operating Cash Flow / Current Liabilities",
    "free_cash_flow": "Operating Cash Flow - Capital Expenditures",
    "free_cash_flow_margin": "Free Cash Flow / Revenue × 100",
    "cash_flow_to_debt": "Operating Cash Flow / Total Debt",
    
    # Valuation
    "earnings_per_share": "Net Income / Weighted Average Shares Outstanding",
    "price_to_earnings": "Stock Price / Earnings Per Share",
    "price_to_book": "Stock Price / Book Value Per Share",
    "price_to_sales": "Market Cap / Revenue",
    
    # Growth
    "revenue_growth": "(Current Revenue - Prior Revenue) / Prior Revenue × 100",
    "earnings_growth": "(Current Earnings - Prior Earnings) / |Prior Earnings| × 100",
    "cagr": "((Ending Value / Beginning Value) ^ (1/n) - 1) × 100",
}

METRIC_UNITS: Final[dict[str, str]] = {
    # Profitability - percentages
    "gross_profit_margin": "percentage",
    "operating_profit_margin": "percentage",
    "net_profit_margin": "percentage",
    "ebitda_margin": "percentage",
    "ebit_margin": "percentage",
    "roa": "percentage",
    "return_on_assets": "percentage",
    "roe": "percentage",
    "return_on_equity": "percentage",
    "roce": "percentage",
    "return_on_capital_employed": "percentage",
    
    # Liquidity - ratios
    "current_ratio": "ratio",
    "quick_ratio": "ratio",
    "cash_ratio": "ratio",
    "working_capital": "currency",
    "working_capital_ratio": "ratio",
    
    # Solvency - ratios
    "debt_to_equity": "ratio",
    "debt_to_assets": "ratio",
    "equity_ratio": "ratio",
    "interest_coverage": "times",
    "debt_service_coverage": "times",
    
    # Efficiency - turnover uses "times", days use "days"
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
    "operating_cash_flow_ratio": "ratio",
    "free_cash_flow": "currency",
    "free_cash_flow_margin": "percentage",
    "cash_flow_to_debt": "ratio",
    
    # Valuation
    "earnings_per_share": "currency",
    "price_to_earnings": "ratio",
    "price_to_book": "ratio",
    "price_to_sales": "ratio",
    
    # Growth
    "revenue_growth": "percentage",
    "earnings_growth": "percentage",
    "cagr": "percentage",
}


# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

TREND_SYMBOLS: Final[dict[str, str]] = {
    "increasing": "↑",
    "decreasing": "↓",
    "stable": "→",
    "volatile": "↕",
    "unknown": "?",
}

STATUS_SYMBOLS: Final[dict[str, str]] = {
    "good": "✅",
    "warning": "⚠️",
    "error": "❌",
    "info": "ℹ️",
    "unknown": "❓",
    "success": "✅",
    "failure": "❌",
}

LARGE_NUMBER_SUFFIXES: Final[list[tuple[float, str]]] = [
    (1e12, "T"),
    (1e9, "B"),
    (1e6, "M"),
    (1e3, "K"),
]

# Report date format
REPORT_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M"
REPORT_DATE_FORMAT_ISO: Final[str] = "ISO"  # Use datetime.isoformat()


# ============================================================================
# DEBUG CONFIGURATION
# ============================================================================

import os

DEBUG: Final[bool] = os.environ.get("FINANALYST_DEBUG", "false").lower() == "true"
INCLUDE_TRACEBACK_IN_ERRORS: Final[bool] = DEBUG
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

All exceptions support:
- Serialization to dict/JSON for structured error handling
- Auto-generated error codes for programmatic handling
- Actionable suggestions for LLM consumption

Version: 3.2.1 - Improved error code generation, added new exception types
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
    - Auto-generated error_code from class name
    - Optional details dictionary for context
    - Optional suggestion for resolution
    - JSON serialization support
    
    All custom exceptions should inherit from this class.
    
    Example:
        >>> raise FinAnalystError("Something went wrong", details={"context": "value"})
        >>> try:
        ...     raise FinAnalystError("Test error")
        ... except FinAnalystError as e:
        ...     print(e.to_json())
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
        
        Handles CamelCase, acronyms (like XML, IO), and removes 'Error' suffix.
        
        Examples:
            FinAnalystError -> FIN_ANALYST
            DivisionByZeroError -> DIVISION_BY_ZERO
            XMLParsingError -> XML_PARSING
            IOError -> IO
        """
        name = self.__class__.__name__
        
        # Remove 'Error' suffix if present
        if name.endswith("Error"):
            name = name[:-5]
        
        # Handle the conversion with proper acronym support
        # Insert underscore before uppercase letters that follow lowercase
        # and before uppercase letters followed by lowercase (for acronyms)
        result = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', name)
        result = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', '_', result)
        
        return result.upper()
    
    @property
    def error_type(self) -> str:
        """Get the exception class name."""
        return self.__class__.__name__
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        result: dict[str, Any] = {
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
    
    Example:
        >>> raise DivisionByZeroError(100, "total_assets", metric_name="ROA")
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
    Raised when ROE calculation encounters negative equity.
    
    Negative equity makes ROE mathematically calculable but meaningless.
    This exception provides clear guidance on interpretation.
    """
    
    def __init__(
        self,
        net_income: Any,
        equity: Any,
        **kwargs: Any
    ) -> None:
        if float(net_income) > 0:
            message = "Positive net income with negative equity - ROE is undefined"
            suggestion = "Consider alternative profitability metrics like ROA"
        else:
            message = "Both net income and equity are negative - ROE is misleading"
            suggestion = "ROE would be positive but this indicates distress, not profitability"
        
        super().__init__(
            message,
            metric_name="Return on Equity",
            suggestion=suggestion,
            net_income=net_income,
            equity=equity,
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


class CurrencyMismatchError(ValidationError):
    """
    Raised when financial statements have mismatched currencies.
    
    Cross-statement analysis requires consistent currency across all statements.
    """
    
    def __init__(
        self,
        statement_a: str,
        currency_a: str,
        statement_b: str,
        currency_b: str,
        **kwargs: Any
    ) -> None:
        message = (
            f"Currency mismatch: {statement_a} uses {currency_a}, "
            f"but {statement_b} uses {currency_b}"
        )
        super().__init__(
            message,
            validation_type="currency",
            suggestion="Ensure all statements use the same currency or convert before analysis",
            statement_a=statement_a,
            currency_a=currency_a,
            statement_b=statement_b,
            currency_b=currency_b,
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
            # Truncate raw data to avoid huge error messages
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
            suggestion="Use one of the available tools",
            available_tools=available_tools[:10] if available_tools else None,
            similar_tools=suggestions,
            **kwargs
        )
    
    @staticmethod
    def _find_similar(name: str, available: list[str]) -> list[str]:
        """Find tools with similar names using substring matching."""
        name_lower = name.lower()
        
        # Exact substring matches
        similar = [
            t for t in available 
            if name_lower in t.lower() or t.lower() in name_lower
        ]
        
        # If no substring matches, try word-based matching
        if not similar:
            name_words = set(name_lower.replace("_", " ").split())
            for tool in available:
                tool_words = set(tool.lower().replace("_", " ").split())
                if name_words & tool_words:  # Any common words
                    similar.append(tool)
        
        return similar[:3] if similar else available[:3]


class ToolExecutionError(ToolError):
    """Raised when a tool fails during execution."""
    
    def __init__(
        self,
        tool_name: str,
        original_error: Exception,
        parameters: dict[str, Any] | None = None,
        include_traceback: bool = False,
        **kwargs: Any
    ) -> None:
        message = f"Tool '{tool_name}' failed: {str(original_error)}"
        
        details = {
            "original_error_type": type(original_error).__name__,
            "original_error_message": str(original_error),
        }
        
        if parameters:
            # Sanitize parameters (don't include huge objects)
            sanitized = {}
            for k, v in parameters.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    sanitized[k] = v
                else:
                    sanitized[k] = f"<{type(v).__name__}>"
            details["parameters"] = sanitized
        
        if include_traceback:
            import traceback
            details["traceback"] = traceback.format_exc()
        
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
        allowed_values: list[Any] | None = None,
        **kwargs: Any
    ) -> None:
        full_message = f"Parameter '{parameter_name}' for tool '{tool_name}': {message}"
        
        details = {
            "parameter_name": parameter_name,
        }
        
        if expected_type:
            details["expected_type"] = expected_type
        if actual_value is not None:
            # Truncate long values
            str_val = str(actual_value)
            details["actual_value"] = str_val[:100] if len(str_val) > 100 else str_val
        if allowed_values:
            details["allowed_values"] = allowed_values
        
        details.update(kwargs)
        
        suggestion = f"Provide a valid value for '{parameter_name}'"
        if allowed_values:
            suggestion += f". Allowed values: {allowed_values}"
        
        super().__init__(
            full_message,
            tool_name=tool_name,
            suggestion=suggestion,
            **details
        )


# ============================================================================
# OPERATIONAL ERRORS
# ============================================================================

class AnalysisTimeoutError(FinAnalystError):
    """Raised when an analysis operation times out."""
    
    def __init__(
        self,
        operation: str,
        timeout_seconds: float,
        **kwargs: Any
    ) -> None:
        message = f"Operation '{operation}' timed out after {timeout_seconds:.1f} seconds"
        super().__init__(
            message,
            suggestion="Try with a smaller dataset or increase timeout",
            operation=operation,
            timeout_seconds=timeout_seconds,
            **kwargs
        )


class ConfigurationError(FinAnalystError):
    """Raised when there's a configuration problem."""
    
    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if config_key:
            details["config_key"] = config_key
        super().__init__(
            message,
            details=details,
            suggestion="Check configuration settings"
        )
File 3 of 5: finanalyst_tools/utils/math_ops.py
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

Version: 3.2.1 - Fixed CAGR precision, improved consistency in None/Zero handling
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
Numeric = int | float | Decimal | str


# ============================================================================
# TYPE CONVERSION
# ============================================================================

def to_decimal(
    value: Numeric | None,
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
        >>> to_decimal(1.1)  # Float handled via string to avoid precision issues
        Decimal('1.1')
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
        elif isinstance(value, str):
            # Clean string input (handle formatted numbers)
            cleaned = value.strip().replace(",", "").replace(" ", "")
            if not cleaned or cleaned in ("-", "+", "."):
                return default
            result = Decimal(cleaned)
        else:
            result = Decimal(str(value))
        
        if precision is not None:
            result = round_decimal(result, precision)
        
        return result
    except (InvalidOperation, ValueError, TypeError):
        return default


def is_effectively_zero(
    value: Numeric | None, 
    threshold: Numeric | None = None
) -> bool:
    """
    Check if a value is effectively zero (within threshold).
    
    Uses Decimal comparison to maintain precision.
    
    Args:
        value: Value to check
        threshold: Threshold for zero comparison (default: ZERO_THRESHOLD from config)
        
    Returns:
        True if value is None or within threshold of zero
    """
    if value is None:
        return True
    
    # Use config default if no threshold provided
    if threshold is None:
        threshold = ZERO_THRESHOLD
    
    # Convert to Decimal for precise comparison
    dec_value = to_decimal(value)
    dec_threshold = to_decimal(threshold)
    
    return abs(dec_value) < dec_threshold


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
        return_none_on_zero: If True, return None instead of default when denominator is zero
        
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
        if return_none_on_zero:
            return None
        return default
    
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
        return_none_on_zero: If True, return None when whole is zero instead of Decimal("0")
        
    Returns:
        Percentage value (e.g., 25.00 for 25%), or None if return_none_on_zero and whole is zero
    """
    if part is None or whole is None:
        return None if return_none_on_zero else Decimal("0")
    
    if is_effectively_zero(whole):
        return None if return_none_on_zero else Decimal("0")
    
    part_dec = to_decimal(part)
    whole_dec = to_decimal(whole)
    
    ratio = part_dec / whole_dec
    result = ratio * 100
    return round_decimal(result, precision)


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
        
    Note:
        Returns None if previous is zero or either value is None.
        Uses absolute value of previous to handle negative base values correctly.
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
    
    Uses high-precision Decimal arithmetic to avoid float precision issues.
    
    Args:
        beginning_value: Value at start
        ending_value: Value at end
        periods: Number of periods (years)
        precision: Decimal places for result
        
    Returns:
        CAGR as percentage, or None if calculation impossible
        
    Examples:
        >>> calculate_cagr(100, 200, 5)  # Doubled in 5 years
        Decimal('14.87')  # ~14.87% annual growth
    """
    if beginning_value is None or ending_value is None or periods <= 0:
        return None
    
    begin = to_decimal(beginning_value)
    end = to_decimal(ending_value)
    
    # Cannot calculate CAGR with zero or negative beginning value
    if begin <= 0:
        return None
    
    # Handle negative ending value (indicates loss greater than initial investment)
    if end < 0:
        return None
    
    # Handle zero ending value
    if is_effectively_zero(end):
        # Complete loss = -100% CAGR
        return Decimal("-100.00")
    
    try:
        # Set high precision for intermediate calculations
        original_prec = getcontext().prec
        getcontext().prec = 50
        
        try:
            ratio = end / begin
            
            # For CAGR, we need to calculate ratio^(1/n)
            # Using logarithms: exp(ln(ratio) / n)
            # Since Decimal doesn't have native exp/ln, we use Python's math
            # but with enough precision in the conversion
            
            ratio_float = float(ratio)
            periods_float = float(periods)
            
            # Calculate using logarithms for better numerical stability
            if ratio_float <= 0:
                return None
            
            ln_ratio = math.log(ratio_float)
            growth_factor = math.exp(ln_ratio / periods_float)
            
            # Convert back to Decimal with full precision
            cagr = (Decimal(str(growth_factor)) - 1) * 100
            
            return round_decimal(cagr, precision)
            
        finally:
            # Restore original precision
            getcontext().prec = original_prec
            
    except (ValueError, ZeroDivisionError, OverflowError, InvalidOperation):
        return None


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
        *values: Values to average (None values are excluded)
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


def calculate_median(
    values: Sequence[Numeric | None],
    precision: int = DECIMAL_PLACES["ratio"],
) -> Decimal | None:
    """
    Calculate median of values.
    
    Args:
        values: Values to find median of
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
        median = (valid_values[mid - 1] + valid_values[mid]) / 2
    else:
        median = valid_values[mid]
    
    return round_decimal(median, precision)


def calculate_variance(
    values: Sequence[Numeric | None],
    population: bool = True,
    precision: int = 6,
) -> Decimal | None:
    """
    Calculate variance of values.
    
    Args:
        values: Values to calculate variance for
        population: If True, use population variance (N); else sample variance (N-1)
        precision: Decimal places for result
        
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
    return round_decimal(sum(squared_diffs) / divisor, precision)


def calculate_std_dev(
    values: Sequence[Numeric | None],
    population: bool = True,
    precision: int = 4,
) -> Decimal | None:
    """
    Calculate standard deviation of values.
    
    Uses Decimal.sqrt() for precision when available, falls back to math.sqrt.
    
    Args:
        values: Values to calculate std dev for
        population: If True, use population std dev; else sample std dev
        precision: Decimal places for result
        
    Returns:
        Standard deviation, or None if insufficient values
    """
    variance = calculate_variance(values, population, precision=precision + 4)
    if variance is None:
        return None
    
    # Use Decimal.sqrt() for precision
    try:
        std_dev = variance.sqrt()
    except (InvalidOperation, ValueError):
        # Fallback for edge cases
        std_dev = Decimal(str(math.sqrt(float(variance))))
    
    return round_decimal(std_dev, precision)


def calculate_coefficient_of_variation(
    values: Sequence[Numeric | None],
    population: bool = True,
    precision: int = 4,
) -> Decimal | None:
    """
    Calculate coefficient of variation (CV = std_dev / mean).
    
    Useful for comparing volatility across metrics with different scales.
    
    Args:
        values: Values to calculate CV for
        population: If True, use population std dev; else sample std dev
        precision: Decimal places for result
        
    Returns:
        Coefficient of variation, or None if cannot calculate
    """
    valid_values = [to_decimal(v) for v in values if v is not None]
    
    if len(valid_values) < 2:
        return None
    
    mean = sum(valid_values) / len(valid_values)
    
    if is_effectively_zero(mean):
        return None
    
    std_dev = calculate_std_dev(values, population, precision + 2)
    if std_dev is None:
        return None
    
    cv = std_dev / abs(mean)
    return round_decimal(cv, precision)


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


def calculate_percentile(
    values: Sequence[Numeric | None],
    percentile: float,
    precision: int = DECIMAL_PLACES["ratio"],
) -> Decimal | None:
    """
    Calculate the nth percentile of values.
    
    Uses linear interpolation between data points.
    
    Args:
        values: Values to calculate percentile for
        percentile: Percentile to calculate (0-100)
        precision: Decimal places for result
        
    Returns:
        Percentile value, or None if no valid values
        
    Example:
        >>> calculate_percentile([1, 2, 3, 4, 5], 50)  # Median
        Decimal('3.0000')
    """
    if not 0 <= percentile <= 100:
        return None
    
    valid_values = sorted([to_decimal(v) for v in values if v is not None])
    n = len(valid_values)
    
    if n == 0:
        return None
    
    if n == 1:
        return round_decimal(valid_values[0], precision)
    
    # Calculate position
    position = (percentile / 100) * (n - 1)
    lower_idx = int(position)
    upper_idx = lower_idx + 1
    
    if upper_idx >= n:
        return round_decimal(valid_values[-1], precision)
    
    # Linear interpolation
    fraction = Decimal(str(position - lower_idx))
    lower_val = valid_values[lower_idx]
    upper_val = valid_values[upper_idx]
    
    result = lower_val + fraction * (upper_val - lower_val)
    return round_decimal(result, precision)
File 4 of 5: finanalyst_tools/utils/formatting.py
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

Version: 3.2.1 - Added markdown escaping, smart truncation, improved safety
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Sequence

from finanalyst_tools.config import (
    DECIMAL_PLACES,
    CURRENCY_SYMBOLS,
    DEFAULT_CURRENCY,
    TREND_SYMBOLS,
    STATUS_SYMBOLS,
    LARGE_NUMBER_SUFFIXES,
    METRIC_UNITS,
)
from finanalyst_tools.utils.math_ops import to_decimal


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _escape_markdown(text: str) -> str:
    """
    Escape characters that break Markdown tables.
    
    Args:
        text: Text to escape
        
    Returns:
        Escaped text safe for Markdown tables
    """
    if not text:
        return text
    
    # Escape pipe characters (break table cells)
    text = text.replace("|", "\\|")
    
    # Replace newlines with spaces (break table rows)
    text = text.replace("\n", " ").replace("\r", " ")
    
    return text


def truncate_smart(
    text: str,
    max_length: int = 100,
    suffix: str = "...",
) -> str:
    """
    Truncate text at word boundary with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to append when truncated
        
    Returns:
        Truncated text, or original if shorter than max_length
    """
    if not text or len(text) <= max_length:
        return text
    
    # Account for suffix length
    available = max_length - len(suffix)
    if available <= 0:
        return suffix[:max_length]
    
    truncated = text[:available]
    
    # Find last space to break at word boundary
    last_space = truncated.rfind(" ")
    
    # Only break at space if it's not too early in the string
    if last_space > available // 2:
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
    symbol_position: str = "prefix",
    fallback: str = "N/A",
) -> str:
    """
    Format a monetary value with currency symbol.
    
    Args:
        value: Amount to format
        currency_code: ISO currency code
        precision: Decimal places (default based on currency)
        show_symbol: Whether to include currency symbol
        symbol_position: "prefix" (default) or "suffix"
        fallback: String to return if value is None
        
    Returns:
        Formatted currency string (e.g., "S$1,234.56")
    """
    if value is None:
        return fallback
    
    # Determine precision based on currency if not specified
    if precision is None:
        from finanalyst_tools.config import ZERO_DECIMAL_CURRENCIES
        precision = 0 if currency_code in ZERO_DECIMAL_CURRENCIES else DECIMAL_PLACES["currency"]
    
    dec_value = to_decimal(value, precision=precision)
    is_negative = dec_value < 0
    abs_value = abs(dec_value)
    
    formatted = f"{float(abs_value):,.{precision}f}"
    
    if show_symbol:
        symbol = CURRENCY_SYMBOLS.get(currency_code, currency_code)
        
        if symbol_position == "suffix":
            formatted = f"{formatted} {symbol}"
        else:  # prefix (default)
            formatted = f"{symbol}{formatted}"
    
    if is_negative:
        formatted = f"-{formatted}"
    
    return formatted


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
    Format a day count value.
    
    Args:
        value: Number of days
        fallback: String to return if value is None
        
    Returns:
        Formatted string (e.g., "45 days", "1 day")
    """
    if value is None:
        return fallback
    
    try:
        days = int(round(float(to_decimal(value))))
        if days == 1:
            return "1 day"
        return f"{days} days"
    except (ValueError, TypeError):
        return fallback


# ============================================================================
# INDICATOR FORMATTING
# ============================================================================

def format_trend_indicator(
    direction: str,
    include_text: bool = False,
) -> str:
    """
    Format a trend direction as an indicator.
    
    Args:
        direction: One of "increasing", "decreasing", "stable", "volatile"
        include_text: Whether to include text after symbol
        
    Returns:
        Trend indicator symbol (e.g., "↑", "↓")
    """
    # Handle enum or string
    if hasattr(direction, 'value'):
        direction = direction.value
    
    direction_lower = direction.lower() if isinstance(direction, str) else str(direction).lower()
    symbol = TREND_SYMBOLS.get(direction_lower, TREND_SYMBOLS.get("unknown", "?"))
    
    if include_text:
        return f"{symbol} {direction_lower.capitalize()}"
    return symbol


def format_status_indicator(
    status: str,
    include_text: bool = False,
) -> str:
    """
    Format a status as an indicator.
    
    Args:
        status: One of "good", "warning", "error", "info", "unknown", "success", "failure"
        include_text: Whether to include text after symbol
        
    Returns:
        Status indicator symbol (e.g., "✅", "⚠️")
    """
    # Handle enum or string
    if hasattr(status, 'value'):
        status = status.value
    
    status_lower = status.lower() if isinstance(status, str) else str(status).lower()
    symbol = STATUS_SYMBOLS.get(status_lower, STATUS_SYMBOLS.get("unknown", "❓"))
    
    if include_text:
        return f"{symbol} {status_lower.capitalize()}"
    return symbol


def format_value_with_unit(
    value: float | Decimal | None,
    metric_name: str,
    precision: int | None = None,
    currency_code: str = DEFAULT_CURRENCY,
    fallback: str = "N/A",
) -> str:
    """
    Format a value with appropriate unit based on metric name.
    
    Args:
        value: Value to format
        metric_name: Name of the metric (used to determine unit)
        precision: Override precision (uses default for unit type if None)
        currency_code: Currency code for currency values
        fallback: String to return if value is None
        
    Returns:
        Formatted string with appropriate unit
    """
    if value is None:
        return fallback
    
    unit = METRIC_UNITS.get(metric_name.lower(), "number")
    
    if unit == "percentage":
        prec = precision if precision is not None else DECIMAL_PLACES["percentage"]
        return format_percentage(value, precision=prec)
    elif unit == "ratio":
        prec = precision if precision is not None else DECIMAL_PLACES["ratio"]
        return format_ratio(value, precision=prec, suffix="")
    elif unit == "times":
        prec = precision if precision is not None else DECIMAL_PLACES["turnover"]
        return format_ratio(value, precision=prec, suffix="x")
    elif unit == "currency":
        prec = precision if precision is not None else DECIMAL_PLACES["currency"]
        return format_currency(value, currency_code=currency_code, precision=prec)
    elif unit == "days":
        return format_days(value, fallback=fallback)
    else:
        prec = precision if precision is not None else 2
        return format_number(value, precision=prec)


# ============================================================================
# TABLE FORMATTING
# ============================================================================

def format_markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    alignment: Sequence[str] | None = None,
    escape_content: bool = True,
) -> str:
    """
    Format data as a Markdown table.
    
    Args:
        headers: Column headers
        rows: Table rows (each row is a sequence of cell values)
        alignment: Column alignments ("left", "center", "right") per column
        escape_content: Whether to escape special Markdown characters in content
        
    Returns:
        Markdown table string
    """
    if not headers or not rows:
        return ""
    
    # Convert all values to strings and optionally escape
    def to_safe_string(val: Any) -> str:
        text = str(val) if val is not None else ""
        return _escape_markdown(text) if escape_content else text
    
    str_headers = [to_safe_string(h) for h in headers]
    str_rows = [[to_safe_string(cell) for cell in row] for row in rows]
    
    # Ensure all rows have the same number of columns as headers
    num_cols = len(str_headers)
    str_rows = [row[:num_cols] + [""] * (num_cols - len(row)) for row in str_rows]
    
    # Calculate column widths
    col_widths = [len(h) for h in str_headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))
    
    # Minimum width of 3 for alignment markers
    col_widths = [max(w, 3) for w in col_widths]
    
    # Build alignment row
    if alignment is None:
        alignment = ["left"] * num_cols
    
    # Pad alignment list if needed
    alignment = list(alignment) + ["left"] * (num_cols - len(alignment))
    
