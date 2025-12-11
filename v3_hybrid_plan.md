Phase 1: Hybrid Implementation Plan — Complete Execution
Executive Summary
I will now implement 25 production-ready files following the hybrid approach. Each file will be complete, thoroughly documented, and ready for drop-in use.

Implementation Execution Plan
File Inventory (25 files)
#	File	Dependencies	Status
1	config.py	None	🔄
2	exceptions.py	None	🔄
3	utils/math_ops.py	config, exceptions	🔄
4	utils/formatting.py	config	🔄
5	utils/currency.py	config	🔄
6	utils/__init__.py	utils/*	🔄
7	models/validation.py	None	🔄
8	models/financial_statements.py	config	🔄
9	models/analysis_results.py	models/validation, config	🔄
10	models/__init__.py	models/*	🔄
11	validation/schema_validator.py	models, utils	🔄
12	validation/reconciliation.py	models, utils, config	🔄
13	validation/plausibility.py	models, config	🔄
14	validation/__init__.py	validation/*	🔄
15	calculations/base.py	models, utils, config	🔄
16	calculations/profitability.py	base, models, utils	🔄
17	calculations/liquidity.py	base, models, utils	🔄
18	calculations/__init__.py	calculations/*	🔄
19	tool_registry.py	calculations, validation	🔄
20	dispatcher.py	tool_registry, exceptions	🔄
21	orchestration/pipeline.py	validation, calculations, models	🔄
22	orchestration/confidence_scorer.py	models	🔄
23	orchestration/report_generator.py	models, utils	🔄
24	orchestration/__init__.py	orchestration/*	🔄
25	__init__.py	All modules	🔄
Complete File Implementations
File 1 of 25: finanalyst_tools/config.py
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
File 2 of 25: finanalyst_tools/exceptions.py
Python

# finanalyst_tools/exceptions.py
"""
Custom exception hierarchy for FinAnalyst-Pro Agent Tools.

This module provides a structured exception hierarchy that enables:
- Precise error categorization for programmatic handling
- Detailed error context for debugging
- Actionable error messages and suggestions for LLM consumption
- Consistent error handling across the package
- JSON serialization for structured error responses

Exception Hierarchy:
    FinAnalystError (base)
    ├── CalculationError
    │   ├── DivisionByZeroError
    │   └── InvalidInputError
    ├── ValidationError
    │   ├── SchemaError
    │   ├── ReconciliationError
    │   └── PlausibilityError
    ├── DataError
    │   ├── DataParsingError
    │   ├── MissingDataError
    │   └── DataCompletenessError
    └── ToolError
        ├── ToolNotFoundError
        ├── ToolExecutionError
        └── ToolParameterError

Author: FinAnalyst-Pro Team
Version: 1.0.0
"""

from __future__ import annotations

import json
from typing import Any


# ============================================================================
# BASE EXCEPTION
# ============================================================================

class FinAnalystError(Exception):
    """
    Base exception for all FinAnalyst-Pro errors.
    
    Provides common functionality:
    - Message storage with optional details dictionary
    - Auto-generated error codes for programmatic handling
    - Actionable suggestions for resolution
    - JSON serialization support for LLM responses
    
    All custom exceptions should inherit from this class.
    
    Attributes:
        message: Human-readable error description
        details: Additional context as key-value pairs
        error_code: Machine-readable error identifier (auto-generated if not provided)
        suggestion: Actionable suggestion for resolution
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
            error_code: Machine-readable error code (auto-generated if None)
            suggestion: Actionable suggestion for resolving the error
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
        Generate default error code from class name.
        
        Converts CamelCase to SCREAMING_SNAKE_CASE.
        Example: DivisionByZeroError -> DIVISION_BY_ZERO
        """
        name = self.__class__.__name__
        result = []
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                result.append("_")
            result.append(char.upper())
        return "".join(result).replace("_ERROR", "")
    
    @property
    def error_type(self) -> str:
        """Get the exception class name."""
        return self.__class__.__name__
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert exception to dictionary for JSON serialization.
        
        Returns:
            Dictionary with error_type, error_code, message, details, and suggestion
        """
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
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert exception to JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON representation of the error
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def __str__(self) -> str:
        """Format error message with optional suggestion."""
        parts = [self.message]
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)
    
    def __repr__(self) -> str:
        if self.details:
            return f"{self.error_type}({self.message!r}, error_code={self.error_code!r}, details={self.details!r})"
        return f"{self.error_type}({self.message!r}, error_code={self.error_code!r})"


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
        """
        Initialize calculation error.
        
        Args:
            message: Error description
            metric_name: Name of the metric being calculated
            formula: The formula that failed
            suggestion: How to fix the issue
            **kwargs: Additional context
        """
        details = kwargs
        if metric_name:
            details["metric_name"] = metric_name
        if formula:
            details["formula"] = formula
        
        super().__init__(
            message,
            details=details,
            error_code="CALCULATION_ERROR",
            suggestion=suggestion or "Check input values and try again",
        )
        self.metric_name = metric_name
        self.formula = formula


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
        """
        Initialize division by zero error.
        
        Args:
            numerator: The dividend value
            denominator_name: Name of the divisor field
            metric_name: Name of the metric being calculated
            **kwargs: Additional context
        """
        message = f"Cannot divide {numerator} by zero ({denominator_name} is zero)"
        
        super().__init__(
            message,
            metric_name=metric_name,
            suggestion=f"Ensure {denominator_name} is non-zero before calculation",
            numerator=numerator,
            denominator_name=denominator_name,
            **kwargs
        )
        self.error_code = "DIVISION_BY_ZERO"
        self.numerator = numerator
        self.denominator_name = denominator_name


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
        received_value: Any = None,
        expected: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize invalid input error.
        
        Args:
            message: Error description
            field_name: Name of the invalid field
            received_value: The value that was received
            expected: Description of what was expected
            **kwargs: Additional context
        """
        suggestion = f"Provide a value that is: {expected}" if expected else None
        
        super().__init__(
            message,
            suggestion=suggestion,
            field_name=field_name,
            received_value=received_value,
            expected=expected,
            **kwargs
        )
        self.error_code = "INVALID_INPUT"
        self.field_name = field_name
        self.received_value = received_value
        self.expected = expected


# ============================================================================
# VALIDATION ERRORS
# ============================================================================

class ValidationError(FinAnalystError):
    """
    Base exception for validation-related errors.
    
    Raised when data fails validation checks (schema, reconciliation, etc.).
    """
    
    def __init__(
        self,
        message: str,
        field: str | None = None,
        validation_type: str | None = None,
        suggestion: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize validation error.
        
        Args:
            message: Error description
            field: The field that failed validation
            validation_type: Type of validation that failed
            suggestion: How to fix the issue
            **kwargs: Additional context
        """
        details = kwargs
        if field:
            details["field"] = field
        if validation_type:
            details["validation_type"] = validation_type
        
        super().__init__(
            message,
            details=details,
            error_code="VALIDATION_ERROR",
            suggestion=suggestion,
        )
        self.field = field
        self.validation_type = validation_type


class SchemaError(ValidationError):
    """
    Raised when data doesn't conform to expected schema.
    
    Typically occurs during Pydantic model validation or when
    required fields are missing or have wrong types.
    """
    
    def __init__(
        self,
        message: str,
        field: str | None = None,
        expected_type: str | None = None,
        received_type: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize schema error.
        
        Args:
            message: Error description
            field: The field with schema error
            expected_type: Expected data type
            received_type: Actual data type received
            **kwargs: Additional context
        """
        suggestion = None
        if expected_type and received_type:
            suggestion = f"Convert the value to {expected_type}"
        
        super().__init__(
            message,
            field=field,
            validation_type="schema",
            suggestion=suggestion,
            expected_type=expected_type,
            received_type=received_type,
            **kwargs
        )
        self.error_code = "SCHEMA_ERROR"
        self.expected_type = expected_type
        self.received_type = received_type


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
        value_a: Any,
        source_a: str,
        value_b: Any,
        source_b: str,
        difference: Any = None,
        tolerance: float | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize reconciliation error.
        
        Args:
            message: Error description
            check_name: Name of the reconciliation check
            value_a: First value
            source_a: Source of first value (e.g., "Income Statement")
            value_b: Second value
            source_b: Source of second value (e.g., "Cash Flow Statement")
            difference: Calculated difference
            tolerance: Tolerance threshold that was exceeded
            **kwargs: Additional context
        """
        super().__init__(
            message,
            validation_type="reconciliation",
            suggestion="Verify data accuracy in both statements or confirm this is a known discrepancy",
            check_name=check_name,
            value_a=value_a,
            source_a=source_a,
            value_b=value_b,
            source_b=source_b,
            difference=difference,
            tolerance=tolerance,
            **kwargs
        )
        self.error_code = "RECONCILIATION_ERROR"
        self.check_name = check_name
        self.value_a = value_a
        self.source_a = source_a
        self.value_b = value_b
        self.source_b = source_b
        self.difference = difference
        self.tolerance = tolerance


class PlausibilityError(ValidationError):
    """
    Raised when a calculated metric is outside plausible range.
    
    Note: This is typically a warning, not a blocking error,
    unless explicitly configured to be strict.
    """
    
    def __init__(
        self,
        message: str,
        metric_name: str,
        value: Any,
        expected_range: tuple[float, float],
        **kwargs: Any
    ) -> None:
        """
        Initialize plausibility error.
        
        Args:
            message: Error description
            metric_name: Name of the metric
            value: The implausible value
            expected_range: Tuple of (min, max) expected values
            **kwargs: Additional context
        """
        super().__init__(
            message,
            validation_type="plausibility",
            suggestion="Verify input data accuracy. This may indicate a data entry error.",
            metric_name=metric_name,
            value=value,
            expected_range=expected_range,
            **kwargs
        )
        self.error_code = "PLAUSIBILITY_ERROR"
        self.metric_name = metric_name
        self.value = value
        self.expected_range = expected_range


# ============================================================================
# DATA ERRORS
# ============================================================================

class DataError(FinAnalystError):
    """
    Base exception for data-related errors.
    
    Raised when there are issues with the input data itself.
    """
    
    def __init__(
        self,
        message: str,
        suggestion: str | None = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            message,
            details=kwargs,
            error_code="DATA_ERROR",
            suggestion=suggestion,
        )


class DataParsingError(DataError):
    """
    Raised when data cannot be parsed from input format.
    
    Examples:
    - Invalid JSON/CSV structure
    - Corrupted file data
    - Encoding issues
    """
    
    def __init__(
        self,
        message: str,
        source: str | None = None,
        line_number: int | None = None,
        raw_data: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize parsing error.
        
        Args:
            message: Error description
            source: Source of the data (filename, URL, etc.)
            line_number: Line number where error occurred
            raw_data: Snippet of raw data that failed to parse
            **kwargs: Additional context
        """
        details = kwargs
        if source:
            details["source"] = source
        if line_number is not None:
            details["line_number"] = line_number
        if raw_data:
            # Truncate if too long
            details["raw_data"] = raw_data[:200] + "..." if len(raw_data) > 200 else raw_data
        
        super().__init__(
            message,
            suggestion="Check source format and encoding, then try again",
            **details
        )
        self.error_code = "DATA_PARSING_ERROR"
        self.source = source
        self.line_number = line_number


class MissingDataError(DataError):
    """
    Raised when required data fields are missing.
    
    Includes information about what data is needed and
    which analysis requires it.
    """
    
    def __init__(
        self,
        message: str,
        missing_fields: list[str] | None = None,
        required_for: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize missing data error.
        
        Args:
            message: Error description
            missing_fields: List of missing field names
            required_for: What analysis/calculation requires the data
            **kwargs: Additional context
        """
        suggestion = None
        if missing_fields:
            suggestion = f"Provide the following fields: {', '.join(missing_fields)}"
        
        super().__init__(
            message,
            suggestion=suggestion,
            missing_fields=missing_fields,
            required_for=required_for,
            **kwargs
        )
        self.error_code = "MISSING_DATA"
        self.missing_fields = missing_fields or []
        self.required_for = required_for


class DataCompletenessError(DataError):
    """
    Raised when data is insufficient for the requested analysis type.
    
    More specific than MissingDataError - indicates that while some
    data may be present, it's not sufficient for the analysis.
    """
    
    def __init__(
        self,
        analysis_type: str,
        missing_fields: list[str],
        available_fields: list[str] | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize data completeness error.
        
        Args:
            analysis_type: Type of analysis that cannot be performed
            missing_fields: List of missing field names
            available_fields: List of fields that are available
            **kwargs: Additional context
        """
        message = (
            f"Insufficient data for {analysis_type} analysis. "
            f"Missing: {', '.join(missing_fields)}"
        )
        
        super().__init__(
            message,
            suggestion=f"Provide the missing fields to enable {analysis_type} analysis",
            analysis_type=analysis_type,
            missing_fields=missing_fields,
            available_fields=available_fields,
            **kwargs
        )
        self.error_code = "DATA_INCOMPLETE"
        self.analysis_type = analysis_type
        self.missing_fields = missing_fields
        self.available_fields = available_fields or []


# ============================================================================
# TOOL ERRORS
# ============================================================================

class ToolError(FinAnalystError):
    """
    Base exception for tool-related errors.
    
    Raised when issues occur during tool discovery or execution.
    """
    
    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        suggestion: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize tool error.
        
        Args:
            message: Error description
            tool_name: Name of the tool
            suggestion: How to fix the issue
            **kwargs: Additional context
        """
        details = kwargs
        if tool_name:
            details["tool_name"] = tool_name
        
        super().__init__(
            message,
            details=details,
            error_code="TOOL_ERROR",
            suggestion=suggestion,
        )
        self.tool_name = tool_name


class ToolNotFoundError(ToolError):
    """
    Raised when a requested tool does not exist.
    
    Includes suggestions for similar tool names if available.
    """
    
    def __init__(
        self,
        tool_name: str,
        available_tools: list[str] | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize tool not found error.
        
        Args:
            tool_name: Name of the tool that wasn't found
            available_tools: List of all available tool names
            **kwargs: Additional context
        """
        # Find similar tool names
        suggestions = []
        if available_tools:
            tool_lower = tool_name.lower()
            for t in available_tools:
                t_lower = t.lower()
                # Check for substring match or similar words
                if tool_lower in t_lower or t_lower in tool_lower:
                    suggestions.append(t)
                elif any(word in t_lower for word in tool_lower.split("_")):
                    suggestions.append(t)
            suggestions = suggestions[:5]  # Limit to 5 suggestions
        
        message = f"Tool '{tool_name}' not found"
        suggestion = None
        if suggestions:
            suggestion = f"Did you mean: {', '.join(suggestions)}?"
        elif available_tools:
            suggestion = f"Use list_tools() to see available tools"
        
        super().__init__(
            message,
            tool_name=tool_name,
            suggestion=suggestion,
            available_tools=available_tools[:20] if available_tools else None,
            suggestions=suggestions,
            **kwargs
        )
        self.error_code = "TOOL_NOT_FOUND"
        self.available_tools = available_tools or []
        self.suggestions = suggestions


class ToolExecutionError(ToolError):
    """
    Raised when a tool fails during execution.
    
    Wraps the original exception with tool context for debugging.
    """
    
    def __init__(
        self,
        tool_name: str,
        original_error: Exception,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize tool execution error.
        
        Args:
            tool_name: Name of the tool that failed
            original_error: The original exception that was raised
            parameters: Parameters that were passed to the tool
            **kwargs: Additional context
        """
        message = f"Tool '{tool_name}' execution failed: {str(original_error)}"
        
        super().__init__(
            message,
            tool_name=tool_name,
            suggestion="Check the parameters and input data, then try again",
            original_error_type=type(original_error).__name__,
            original_error_message=str(original_error),
            parameters=parameters,
            **kwargs
        )
        self.error_code = "TOOL_EXECUTION_ERROR"
        self.original_error = original_error
        self.parameters = parameters


class ToolParameterError(ToolError):
    """
    Raised when tool parameters are invalid.
    
    Provides details about which parameters are wrong and why.
    """
    
    def __init__(
        self,
        tool_name: str,
        parameter_name: str,
        message: str,
        received_value: Any = None,
        expected_type: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize tool parameter error.
        
        Args:
            tool_name: Name of the tool
            parameter_name: Name of the invalid parameter
            message: Description of what's wrong
            received_value: The invalid value received
            expected_type: What type was expected
            **kwargs: Additional context
        """
        full_message = f"Invalid parameter '{parameter_name}' for tool '{tool_name}': {message}"
        
        suggestion = None
        if expected_type:
            suggestion = f"Provide a valid {expected_type} value for '{parameter_name}'"
        
        super().__init__(
            full_message,
            tool_name=tool_name,
            suggestion=suggestion,
            parameter_name=parameter_name,
            received_value=str(received_value)[:100] if received_value is not None else None,
            expected_type=expected_type,
            **kwargs
        )
        self.error_code = "TOOL_PARAMETER_ERROR"
        self.parameter_name = parameter_name
        self.received_value = received_value
        self.expected_type = expected_type
File 3 of 25: finanalyst_tools/utils/math_ops.py
Python

# finanalyst_tools/utils/math_ops.py
"""
Safe mathematical operations for financial calculations.

This module provides Decimal-based arithmetic functions that:
- Ensure precision (no floating-point errors)
- Handle edge cases gracefully (division by zero, None values)
- Support configurable rounding
- Provide statistical functions for trend analysis

All monetary and ratio calculations should use these functions
to ensure consistency and accuracy throughout the package.

Author: FinAnalyst-Pro Team
Version: 1.0.0
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
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
    CalculationError,
)


# Type alias for numeric values
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
    
    Handles various input types (int, float, str, Decimal) and returns
    a default value for None or unconvertible inputs.
    
    Args:
        value: Value to convert (int, float, str, Decimal, or None)
        default: Default value if conversion fails (default: Decimal("0"))
        precision: Optional decimal places for rounding
        
    Returns:
        Decimal representation of the value
        
    Raises:
        InvalidInputError: If value cannot be converted and no default provided
        
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
    
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, (int, float)):
        try:
            # Use str() for floats to avoid floating-point precision issues
            result = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return default
    elif isinstance(value, str):
        try:
            # Remove common formatting characters
            cleaned = value.strip().replace(",", "").replace("$", "").replace("S$", "")
            # Handle percentage strings
            if cleaned.endswith("%"):
                cleaned = cleaned[:-1]
                result = Decimal(cleaned)
            else:
                result = Decimal(cleaned)
        except (InvalidOperation, ValueError):
            return default
    else:
        return default
    
    if precision is not None:
        result = round_decimal(result, precision)
    
    return result


def is_effectively_zero(value: Numeric | None, threshold: float = ZERO_THRESHOLD) -> bool:
    """
    Check if a value is effectively zero (within threshold).
    
    Args:
        value: Value to check
        threshold: Threshold for considering value as zero
        
    Returns:
        True if value is None or within threshold of zero
    """
    if value is None:
        return True
    
    try:
        decimal_value = to_decimal(value)
        return abs(float(decimal_value)) < threshold
    except (ValueError, InvalidOperation):
        return True


# ============================================================================
# SAFE ARITHMETIC OPERATIONS
# ============================================================================

def safe_divide(
    numerator: Numeric | None,
    denominator: Numeric | None,
    default: Decimal | None = None,
    precision: int | None = None,
    raise_on_zero: bool = False,
    denominator_name: str = "denominator",
) -> Decimal:
    """
    Safely divide two numbers, handling zero denominators.
    
    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return if division impossible (default: Decimal("0"))
        precision: Decimal places for result (default: uses ratio precision)
        raise_on_zero: If True, raise exception instead of returning default
        denominator_name: Name of denominator for error messages
        
    Returns:
        Result of division or default value
        
    Raises:
        DivisionByZeroError: If raise_on_zero=True and denominator is zero
        
    Examples:
        >>> safe_divide(100, 4)
        Decimal('25.0000')
        >>> safe_divide(100, 0)
        Decimal('0')
        >>> safe_divide(100, 0, raise_on_zero=True)
        DivisionByZeroError: Cannot divide 100 by zero
    """
    if default is None:
        default = Decimal("0")
    if precision is None:
        precision = DECIMAL_PLACES["ratio"]
    
    num = to_decimal(numerator)
    denom = to_decimal(denominator)
    
    if is_effectively_zero(denom):
        if raise_on_zero:
            raise DivisionByZeroError(
                numerator=num,
                denominator_name=denominator_name,
            )
        return default
    
    result = num / denom
    return round_decimal(result, precision)


def round_decimal(
    value: Numeric | None,
    precision: int = 2,
    rounding: RoundingMode = DEFAULT_ROUNDING,
) -> Decimal:
    """
    Round a Decimal to specified precision.
    
    Args:
        value: Value to round
        precision: Number of decimal places
        rounding: Rounding mode to use
        
    Returns:
        Rounded Decimal value
        
    Examples:
        >>> round_decimal(Decimal("1.2345"), 2)
        Decimal('1.23')
        >>> round_decimal(Decimal("1.235"), 2, RoundingMode.BANKERS)
        Decimal('1.24')
    """
    if value is None:
        return Decimal("0")
    
    decimal_value = to_decimal(value)
    quantize_str = "1." + "0" * precision if precision > 0 else "1"
    
    return decimal_value.quantize(
        Decimal(quantize_str),
        rounding=rounding.get_decimal_rounding()
    )


# ============================================================================
# PERCENTAGE & GROWTH CALCULATIONS
# ============================================================================

def calculate_percentage(
    part: Numeric | None,
    whole: Numeric | None,
    precision: int = 2,
    multiply_by_100: bool = True,
) -> Decimal:
    """
    Calculate percentage: (part / whole) × 100.
    
    Args:
        part: The numerator (part of the whole)
        whole: The denominator (the whole)
        precision: Decimal places for result
        multiply_by_100: If True, return as percentage (e.g., 25.00 for 25%)
        
    Returns:
        Percentage value
        
    Examples:
        >>> calculate_percentage(25, 100)
        Decimal('25.00')
        >>> calculate_percentage(1, 4, multiply_by_100=False)
        Decimal('0.25')
    """
    result = safe_divide(part, whole, precision=precision + 2)
    
    if multiply_by_100:
        result = result * Decimal("100")
    
    return round_decimal(result, precision)


def calculate_growth_rate(
    current: Numeric | None,
    previous: Numeric | None,
    precision: int = 2,
) -> Decimal | None:
    """
    Calculate period-over-period growth rate: ((current - previous) / previous) × 100.
    
    Args:
        current: Current period value
        previous: Previous period value
        precision: Decimal places for result
        
    Returns:
        Growth rate as percentage, or None if cannot be calculated
        
    Examples:
        >>> calculate_growth_rate(110, 100)
        Decimal('10.00')
        >>> calculate_growth_rate(80, 100)
        Decimal('-20.00')
    """
    curr = to_decimal(current)
    prev = to_decimal(previous)
    
    if is_effectively_zero(prev):
        return None  # Cannot calculate growth from zero
    
    growth = ((curr - prev) / prev) * Decimal("100")
    return round_decimal(growth, precision)


def calculate_cagr(
    beginning_value: Numeric | None,
    ending_value: Numeric | None,
    periods: int,
    precision: int = 2,
) -> Decimal | None:
    """
    Calculate Compound Annual Growth Rate (CAGR).
    
    Formula: ((Ending / Beginning) ^ (1/periods) - 1) × 100
    
    Args:
        beginning_value: Starting value
        ending_value: Ending value
        periods: Number of periods
        precision: Decimal places for result
        
    Returns:
        CAGR as percentage, or None if cannot be calculated
        
    Examples:
        >>> calculate_cagr(100, 200, 5)  # Doubled in 5 years
        Decimal('14.87')
    """
    if periods <= 0:
        return None
    
    begin = to_decimal(beginning_value)
    end = to_decimal(ending_value)
    
    if is_effectively_zero(begin) or begin < 0 or end < 0:
        return None
    
    try:
        # Use float for power calculation, then convert back
        ratio = float(end / begin)
        if ratio <= 0:
            return None
        
        cagr = (math.pow(ratio, 1 / periods) - 1) * 100
        return round_decimal(Decimal(str(cagr)), precision)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

def calculate_average(
    *values: Numeric | None,
    precision: int = 2,
    exclude_none: bool = True,
) -> Decimal:
    """
    Calculate arithmetic mean of values.
    
    Args:
        *values: Values to average
        precision: Decimal places for result
        exclude_none: If True, skip None values; if False, treat as 0
        
    Returns:
        Average value
        
    Examples:
        >>> calculate_average(10, 20, 30)
        Decimal('20.00')
        >>> calculate_average(10, None, 30, exclude_none=True)
        Decimal('20.00')
    """
    if not values:
        return Decimal("0")
    
    if exclude_none:
        valid_values = [to_decimal(v) for v in values if v is not None]
    else:
        valid_values = [to_decimal(v) for v in values]
    
    if not valid_values:
        return Decimal("0")
    
    total = sum(valid_values, Decimal("0"))
    count = Decimal(str(len(valid_values)))
    
    return round_decimal(total / count, precision)


def calculate_weighted_average(
    values: Sequence[Numeric | None],
    weights: Sequence[Numeric | None],
    precision: int = 2,
) -> Decimal:
    """
    Calculate weighted average.
    
    Args:
        values: Sequence of values
        weights: Sequence of weights (must match length of values)
        precision: Decimal places for result
        
    Returns:
        Weighted average
        
    Raises:
        InvalidInputError: If values and weights have different lengths
        
    Examples:
        >>> calculate_weighted_average([10, 20, 30], [1, 2, 1])
        Decimal('20.00')  # (10*1 + 20*2 + 30*1) / (1+2+1)
    """
    if len(values) != len(weights):
        raise InvalidInputError(
            "Values and weights must have the same length",
            field_name="weights",
            expected=f"sequence of length {len(values)}",
        )
    
    if not values:
        return Decimal("0")
    
    weighted_sum = Decimal("0")
    weight_sum = Decimal("0")
    
    for value, weight in zip(values, weights):
        if value is not None and weight is not None:
            v = to_decimal(value)
            w = to_decimal(weight)
            weighted_sum += v * w
            weight_sum += w
    
    if is_effectively_zero(weight_sum):
        return Decimal("0")
    
    return round_decimal(weighted_sum / weight_sum, precision)


def calculate_variance(
    values: Sequence[Numeric | None],
    population: bool = True,
    precision: int = 4,
) -> Decimal:
    """
    Calculate variance of values.
    
    Args:
        values: Sequence of values
        population: If True, population variance; if False, sample variance
        precision: Decimal places for result
        
    Returns:
        Variance value
        
    Examples:
        >>> calculate_variance([2, 4, 4, 4, 5, 5, 7, 9])
        Decimal('4.0000')  # Population variance
    """
    valid_values = [to_decimal(v) for v in values if v is not None]
    n = len(valid_values)
    
    if n < 2:
        return Decimal("0")
    
    mean = calculate_average(*valid_values, precision=precision + 2)
    
    squared_diffs = [(v - mean) ** 2 for v in valid_values]
    sum_squared = sum(squared_diffs, Decimal("0"))
    
    divisor = n if population else (n - 1)
    variance = sum_squared / Decimal(str(divisor))
    
    return round_decimal(variance, precision)


def calculate_std_dev(
    values: Sequence[Numeric | None],
    population: bool = True,
    precision: int = 4,
) -> Decimal:
    """
    Calculate standard deviation of values.
    
    Args:
        values: Sequence of values
        population: If True, population std dev; if False, sample std dev
        precision: Decimal places for result
        
    Returns:
        Standard deviation
        
    Examples:
        >>> calculate_std_dev([2, 4, 4, 4, 5, 5, 7, 9])
        Decimal('2.0000')
    """
    variance = calculate_variance(values, population=population, precision=precision + 2)
    
    if variance == 0:
        return Decimal("0")
    
    # Use float for square root, then convert back
    std_dev = Decimal(str(math.sqrt(float(variance))))
    return round_decimal(std_dev, precision)


def calculate_min_max(
    values: Sequence[Numeric | None],
) -> tuple[Decimal, Decimal]:
    """
    Calculate minimum and maximum of values.
    
    Args:
        values: Sequence of values
        
    Returns:
        Tuple of (minimum, maximum)
        
    Examples:
        >>> calculate_min_max([5, 2, 8, 1, 9])
        (Decimal('1'), Decimal('9'))
    """
    valid_values = [to_decimal(v) for v in values if v is not None]
    
    if not valid_values:
        return (Decimal("0"), Decimal("0"))
    
    return (min(valid_values), max(valid_values))


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def compare_values(
    value_a: Numeric | None,
    value_b: Numeric | None,
    tolerance: float = ZERO_THRESHOLD,
) -> int:
    """
    Compare two values with tolerance.
    
    Args:
        value_a: First value
        value_b: Second value
        tolerance: Tolerance for equality comparison
        
    Returns:
        -1 if a < b, 0 if a ≈ b (within tolerance), 1 if a > b
        
    Examples:
        >>> compare_values(100, 100.0001)
        0  # Within tolerance
        >>> compare_values(100, 200)
        -1
    """
    a = to_decimal(value_a)
    b = to_decimal(value_b)
    
    diff = float(a - b)
    
    if abs(diff) <= tolerance:
        return 0
    elif diff < 0:
        return -1
    else:
        return 1


def is_within_range(
    value: Numeric | None,
    min_value: Numeric | None,
    max_value: Numeric | None,
) -> bool:
    """
    Check if value is within range [min_value, max_value].
    
    Args:
        value: Value to check
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        True if value is within range (inclusive)
        
    Examples:
        >>> is_within_range(50, 0, 100)
        True
        >>> is_within_range(150, 0, 100)
        False
    """
    if value is None:
        return False
    
    v = to_decimal(value)
    
    if min_value is not None:
        if v < to_decimal(min_value):
            return False
    
    if max_value is not None:
        if v > to_decimal(max_value):
            return False
    
    return True


# ============================================================================
# FINANCIAL-SPECIFIC CALCULATIONS
# ============================================================================

def calculate_average_balance(
    beginning: Numeric | None,
    ending: Numeric | None,
    precision: int = 2,
) -> Decimal:
    """
    Calculate average balance from beginning and ending values.
    
    Common in financial calculations like ROA, ROE where average
    balance sheet values are used.
    
    Args:
        beginning: Beginning period balance
        ending: Ending period balance
        precision: Decimal places for result
        
    Returns:
        Average of beginning and ending
        
    Examples:
        >>> calculate_average_balance(1000, 1200)
        Decimal('1100.00')
    """
    return calculate_average(beginning, ending, precision=precision)


def annualize_rate(
    periodic_rate: Numeric | None,
    periods_per_year: int,
    precision: int = 2,
) -> Decimal:
    """
    Annualize a periodic rate.
    
    Args:
        periodic_rate: Rate for one period (as percentage)
        periods_per_year: Number of periods in a year
        precision: Decimal places for result
        
    Returns:
        Annualized rate as percentage
        
    Examples:
        >>> annualize_rate(2.5, 4)  # 2.5% quarterly
        Decimal('10.00')  # Simple annualization
    """
    if periods_per_year <= 0:
        return Decimal("0")
    
    rate = to_decimal(periodic_rate)
    annual = rate * Decimal(str(periods_per_year))
    
    return round_decimal(annual, precision)
I'll continue with the remaining files. Given the extensive nature of this implementation (25 files), I'll provide them in logical batches.

File 4 of 25: finanalyst_tools/utils/formatting.py
Python

# finanalyst_tools/utils/formatting.py
"""
Number, currency, and output formatting utilities.

This module provides consistent formatting functions for:
- Currency values with proper symbols and locale conventions
- Percentages with configurable precision
- Ratios and financial metrics
- Trend indicators and status symbols
- Markdown tables for reports

All functions handle None/invalid inputs gracefully,
returning "N/A" or appropriate placeholder text.

Author: FinAnalyst-Pro Team
Version: 1.0.0
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Sequence

from finanalyst_tools.config import (
    DECIMAL_PLACES,
    CURRENCY_SYMBOLS,
    ZERO_DECIMAL_CURRENCIES,
    LARGE_NUMBER_SUFFIXES,
    TREND_SYMBOLS,
    STATUS_SYMBOLS,
    DEFAULT_CURRENCY,
)
from finanalyst_tools.utils.math_ops import to_decimal, is_effectively_zero


# ============================================================================
# NUMBER FORMATTING
# ============================================================================

def format_number(
    value: int | float | Decimal | None,
    precision: int = 2,
    use_thousands_sep: bool = True,
    na_value: str = "N/A",
) -> str:
    """
    Format a number with optional thousands separator.
    
    Args:
        value: Number to format
        precision: Decimal places
        use_thousands_sep: Whether to include thousand separators
        na_value: String to return for None values
        
    Returns:
        Formatted number string
        
    Examples:
        >>> format_number(1234567.89)
        '1,234,567.89'
        >>> format_number(None)
        'N/A'
    """
    if value is None:
        return na_value
    
    try:
        decimal_value = to_decimal(value)
        
        if precision == 0:
            formatted = f"{int(decimal_value):,}" if use_thousands_sep else str(int(decimal_value))
        else:
            if use_thousands_sep:
                formatted = f"{float(decimal_value):,.{precision}f}"
            else:
                formatted = f"{float(decimal_value):.{precision}f}"
        
        return formatted
    except (ValueError, TypeError, InvalidOperation):
        return na_value


def format_large_number(
    value: int | float | Decimal | None,
    precision: int = 1,
    na_value: str = "N/A",
) -> str:
    """
    Format large numbers with K/M/B/T suffixes.
    
    Args:
        value: Number to format
        precision: Decimal places after suffix
        na_value: String to return for None values
        
    Returns:
        Formatted string with suffix
        
    Examples:
        >>> format_large_number(1500000)
        '1.5M'
        >>> format_large_number(2500000000)
        '2.5B'
    """
    if value is None:
        return na_value
    
    try:
        num = float(to_decimal(value))
        
        # Handle negative numbers
        sign = "-" if num < 0 else ""
        num = abs(num)
        
        # Find appropriate suffix
        for threshold, suffix in LARGE_NUMBER_SUFFIXES:
            if num >= threshold:
                formatted = f"{sign}{num / threshold:.{precision}f}{suffix}"
                return formatted
        
        # No suffix needed for small numbers
        return format_number(value, precision=precision)
    except (ValueError, TypeError):
        return na_value


# ============================================================================
# CURRENCY FORMATTING
# ============================================================================

def format_currency(
    value: int | float | Decimal | None,
    currency_code: str = DEFAULT_CURRENCY,
    precision: int | None = None,
    show_symbol: bool = True,
    show_code: bool = False,
    na_value: str = "N/A",
) -> str:
    """
    Format a monetary value with currency symbol.
    
    Args:
        value: Amount to format
        currency_code: ISO currency code (default: SGD)
        precision: Decimal places (auto-detected based on currency if None)
        show_symbol: Whether to show currency symbol
        show_code: Whether to show currency code (e.g., "SGD")
        na_value: String to return for None values
        
    Returns:
        Formatted currency string
        
    Examples:
        >>> format_currency(1234.56)
        'S$1,234.56'
        >>> format_currency(1234.56, "USD")
        '$1,234.56'
        >>> format_currency(1000000, show_code=True)
        'S$1,000,000.00 SGD'
    """
    if value is None:
        return na_value
    
    try:
        # Determine precision based on currency
        if precision is None:
            precision = 0 if currency_code in ZERO_DECIMAL_CURRENCIES else DECIMAL_PLACES["currency"]
        
        decimal_value = to_decimal(value)
        
        # Format the number
        formatted_number = format_number(decimal_value, precision=precision)
        
        # Get symbol
        symbol = CURRENCY_SYMBOLS.get(currency_code, currency_code)
        
        # Build result
        if show_symbol:
            result = f"{symbol}{formatted_number}"
        else:
            result = formatted_number
        
        if show_code:
            result = f"{result} {currency_code}"
        
        return result
    except (ValueError, TypeError):
        return na_value


def format_sgd(
    value: int | float | Decimal | None,
    precision: int = 2,
    show_symbol: bool = True,
    na_value: str = "N/A",
) -> str:
    """
    Format a value as Singapore Dollars.
    
    Convenience function for SGD formatting.
    
    Args:
        value: Amount to format
        precision: Decimal places
        show_symbol: Whether to show S$ symbol
        na_value: String to return for None values
        
    Returns:
        Formatted SGD string
        
    Examples:
        >>> format_sgd(1234.56)
        'S$1,234.56'
    """
    return format_currency(value, "SGD", precision, show_symbol, na_value=na_value)


# ============================================================================
# PERCENTAGE FORMATTING
# ============================================================================

def format_percentage(
    value: int | float | Decimal | None,
    precision: int = 2,
    show_symbol: bool = True,
    multiply_by_100: bool = False,
    na_value: str = "N/A",
) -> str:
    """
    Format a value as a percentage.
    
    Args:
        value: Percentage value
        precision: Decimal places
        show_symbol: Whether to include % symbol
        multiply_by_100: If True, multiply value by 100 (for decimal ratios)
        na_value: String to return for None values
        
    Returns:
        Formatted percentage string
        
    Examples:
        >>> format_percentage(25.5)
        '25.50%'
        >>> format_percentage(0.255, multiply_by_100=True)
        '25.50%'
    """
    if value is None:
        return na_value
    
    try:
        decimal_value = to_decimal(value)
        
        if multiply_by_100:
            decimal_value = decimal_value * Decimal("100")
        
        formatted = format_number(decimal_value, precision=precision, use_thousands_sep=False)
        
        if show_symbol:
            return f"{formatted}%"
        return formatted
    except (ValueError, TypeError):
        return na_value


def format_change(
    value: int | float | Decimal | None,
    precision: int = 2,
    show_sign: bool = True,
    is_percentage: bool = True,
    na_value: str = "N/A",
) -> str:
    """
    Format a change value with +/- sign.
    
    Args:
        value: Change value
        precision: Decimal places
        show_sign: Whether to show + for positive values
        is_percentage: Whether to append % symbol
        na_value: String to return for None values
        
    Returns:
        Formatted change string
        
    Examples:
        >>> format_change(10.5)
        '+10.50%'
        >>> format_change(-5.2)
        '-5.20%'
    """
    if value is None:
        return na_value
    
    try:
        decimal_value = to_decimal(value)
        
        # Determine sign
        if decimal_value > 0 and show_sign:
            sign = "+"
        elif decimal_value < 0:
            sign = ""  # Negative sign included in number
        else:
            sign = ""
        
        formatted = format_number(decimal_value, precision=precision, use_thousands_sep=False)
        
        if is_percentage:
            return f"{sign}{formatted}%"
        return f"{sign}{formatted}"
    except (ValueError, TypeError):
        return na_value


# ============================================================================
# RATIO FORMATTING
# ============================================================================

def format_ratio(
    value: int | float | Decimal | None,
    precision: int = 2,
    suffix: str = "x",
    na_value: str = "N/A",
) -> str:
    """
    Format a ratio value.
    
    Args:
        value: Ratio value
        precision: Decimal places
        suffix: Suffix to append (e.g., "x" for "1.5x")
        na_value: String to return for None values
        
    Returns:
        Formatted ratio string
        
    Examples:
        >>> format_ratio(1.5)
        '1.50x'
        >>> format_ratio(2.345, precision=1)
        '2.3x'
    """
    if value is None:
        return na_value
    
    try:
        formatted = format_number(value, precision=precision, use_thousands_sep=False)
        return f"{formatted}{suffix}"
    except (ValueError, TypeError):
        return na_value


def format_days(
    value: int | float | Decimal | None,
    precision: int = 0,
    na_value: str = "N/A",
) -> str:
    """
    Format a value as days.
    
    Args:
        value: Number of days
        precision: Decimal places (usually 0)
        na_value: String to return for None values
        
    Returns:
        Formatted days string
        
    Examples:
        >>> format_days(45)
        '45 days'
    """
    if value is None:
        return na_value
    
    try:
        decimal_value = to_decimal(value)
        formatted = format_number(decimal_value, precision=precision, use_thousands_sep=False)
        
        # Handle singular/plural
        if abs(float(decimal_value)) == 1:
            return f"{formatted} day"
        return f"{formatted} days"
    except (ValueError, TypeError):
        return na_value


# ============================================================================
# VALUE WITH UNIT FORMATTING
# ============================================================================

def format_value_with_unit(
    value: int | float | Decimal | None,
    unit: str,
    precision: int = 2,
    currency_code: str = DEFAULT_CURRENCY,
    na_value: str = "N/A",
) -> str:
    """
    Format a value with its appropriate unit.
    
    Args:
        value: Value to format
        unit: Unit type ("percentage", "ratio", "currency", "days", "count")
        precision: Decimal places
        currency_code: Currency code for currency unit
        na_value: String to return for None values
        
    Returns:
        Formatted value with unit
        
    Examples:
        >>> format_value_with_unit(25.5, "percentage")
        '25.50%'
        >>> format_value_with_unit(1.5, "ratio")
        '1.50x'
    """
    if value is None:
        return na_value
    
    unit_lower = unit.lower()
    
    if unit_lower in ("percentage", "percent", "%"):
        return format_percentage(value, precision=precision)
    elif unit_lower in ("ratio", "x", "times"):
        return format_ratio(value, precision=precision)
    elif unit_lower in ("currency", "money", "amount"):
        return format_currency(value, currency_code=currency_code, precision=precision)
    elif unit_lower in ("days", "day"):
        return format_days(value, precision=precision)
    elif unit_lower in ("count", "number", "integer"):
        return format_number(value, precision=0)
    else:
        return format_number(value, precision=precision)


# ============================================================================
# INDICATOR FORMATTING
# ============================================================================

def format_trend_indicator(
    direction: str,
    include_text: bool = False,
) -> str:
    """
    Get a trend indicator symbol.
    
    Args:
        direction: Trend direction ("increasing", "decreasing", "stable", "volatile")
        include_text: Whether to include text after symbol
        
    Returns:
        Trend indicator string
        
    Examples:
        >>> format_trend_indicator("increasing")
        '↑'
        >>> format_trend_indicator("decreasing", include_text=True)
        '↓ Decreasing'
    """
    direction_lower = direction.lower()
    symbol = TREND_SYMBOLS.get(direction_lower, "")
    
    if include_text and symbol:
        return f"{symbol} {direction.title()}"
    return symbol


def format_status_indicator(
    status: str,
    include_text: bool = False,
) -> str:
    """
    Get a status indicator symbol.
    
    Args:
        status: Status ("good", "warning", "error", "info", "pass", "fail")
        include_text: Whether to include text after symbol
        
    Returns:
        Status indicator string
        
    Examples:
        >>> format_status_indicator("good")
        '✅'
        >>> format_status_indicator("warning", include_text=True)
        '⚠️ Warning'
    """
    status_lower = status.lower()
    symbol = STATUS_SYMBOLS.get(status_lower, "")
    
    if include_text and symbol:
        return f"{symbol} {status.title()}"
    return symbol


def format_plausibility_indicator(
    is_plausible: bool,
    include_text: bool = False,
) -> str:
    """
    Get a plausibility indicator.
    
    Args:
        is_plausible: Whether value is plausible
        include_text: Whether to include text
        
    Returns:
        Plausibility indicator string
    """
    if is_plausible:
        return format_status_indicator("pass", include_text)
    return format_status_indicator("warning", include_text)


# ============================================================================
# MARKDOWN TABLE FORMATTING
# ============================================================================

def format_markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    alignment: Sequence[str] | None = None,
) -> str:
    """
    Generate a Markdown table.
    
    Args:
        headers: Column headers
        rows: Table rows (each row is a sequence of values)
        alignment: Column alignments ("left", "center", "right") for each column
        
    Returns:
        Markdown table string
        
    Examples:
        >>> print(format_markdown_table(
        ...     ["Metric", "Value"],
        ...     [["Revenue", "$100M"], ["Profit", "$20M"]]
        ... ))
        | Metric | Value |
        |--------|-------|
        | Revenue | $100M |
        | Profit | $20M |
    """
    if not headers or not rows:
        return ""
    
    # Convert all values to strings
    str_headers = [str(h) for h in headers]
    str_rows = [[str(cell) for cell in row] for row in rows]
    
    # Calculate column widths
    num_cols = len(headers)
    col_widths = [len(h) for h in str_headers]
    
    for row in str_rows:
        for i, cell in enumerate(row[:num_cols]):
            col_widths[i] = max(col_widths[i], len(cell))
    
    # Pad cells to column width
    def pad_cell(text: str, width: int, align: str = "left") -> str:
        if align == "right":
            return text.rjust(width)
        elif align == "center":
            return text.center(width)
        return text.ljust(width)
    
    # Default alignment
    if alignment is None:
        alignment = ["left"] * num_cols
    else:
        alignment = list(alignment) + ["left"] * (num_cols - len(alignment))
    
    # Build header row
    header_cells = [pad_cell(h, col_widths[i], alignment[i]) for i, h in enumerate(str_headers)]
    header_line = "| " + " | ".join(header_cells) + " |"
    
    # Build separator row
    def separator_cell(width: int, align: str) -> str:
        if align == "right":
            return "-" * (width - 1) + ":"
        elif align == "center":
            return ":" + "-" * (width - 2) + ":"
        return "-" * width
    
    separator_cells = [separator_cell(col_widths[i], alignment[i]) for i in range(num_cols)]
    separator_line = "|" + "|".join(separator_cells) + "|"
    
    # Build data rows
    data_lines = []
    for row in str_rows:
        # Pad row to have correct number of columns
        padded_row = list(row) + [""] * (num_cols - len(row))
        row_cells = [pad_cell(padded_row[i], col_widths[i], alignment[i]) for i in range(num_cols)]
        data_lines.append("| " + " | ".join(row_cells) + " |")
    
    # Combine all lines
    return "\n".join([header_line, separator_line] + data_lines)


def format_key_value_list(
    items: dict[str, Any],
    prefix: str = "- ",
) -> str:
    """
    Format a dictionary as a key-value list.
    
    Args:
        items: Dictionary of key-value pairs
        prefix: Prefix for each line
        
    Returns:
        Formatted key-value list string
        
    Examples:
        >>> print(format_key_value_list({"Name": "Company A", "Revenue": "$100M"}))
        - **Name**: Company A
        - **Revenue**: $100M
    """
    lines = []
    for key, value in items.items():
        lines.append(f"{prefix}**{key}**: {value}")
    return "\n".join(lines)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_currency_symbol(currency_code: str) -> str:
    """
    Get the symbol for a currency code.
    
    Args:
        currency_code: ISO currency code
        
    Returns:
        Currency symbol or the code itself if not found
    """
    return CURRENCY_SYMBOLS.get(currency_code.upper(), currency_code)


def truncate_text(
    text: str,
    max_length: int = 100,
    suffix: str = "...",
) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to append if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix
File 5 of 25: finanalyst_tools/utils/currency.py
Python

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
