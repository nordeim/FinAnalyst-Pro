# Hybrid Implementation Plan — Complete Execution

## Executive Summary

I will now implement the complete hybrid approach with **24 production-ready files**. This implementation combines the best of V1 and V2 while adding the critical orchestration layer for system prompt compliance.

---

## Implementation Plan

### File Inventory (24 Files)

| Order | File | Source | Status |
|-------|------|--------|--------|
| 1 | `config.py` | V2 + V1 additions | 🔨 |
| 2 | `exceptions.py` | V1 enhanced | 🔨 |
| 3 | `utils/__init__.py` | V2 | 🔨 |
| 4 | `utils/math_ops.py` | V2 enhanced | 🔨 |
| 5 | `utils/formatting.py` | V2 enhanced | 🔨 |
| 6 | `utils/currency.py` | V2 | 🔨 |
| 7 | `models/__init__.py` | Hybrid | 🔨 |
| 8 | `models/validation.py` | V1 enhanced | 🔨 |
| 9 | `models/financial_statements.py` | V1 + V2 | 🔨 |
| 10 | `models/analysis_results.py` | V2 + V1 | 🔨 |
| 11 | `validation/__init__.py` | Hybrid | 🔨 |
| 12 | `validation/schema_validator.py` | V2 | 🔨 |
| 13 | `validation/reconciliation.py` | V1 (critical) | 🔨 |
| 14 | `validation/plausibility.py` | V1 (critical) | 🔨 |
| 15 | `calculations/__init__.py` | Hybrid | 🔨 |
| 16 | `calculations/base.py` | V1 | 🔨 |
| 17 | `calculations/profitability.py` | Hybrid | 🔨 |
| 18 | `calculations/liquidity.py` | Hybrid | 🔨 |
| 19 | `tool_registry.py` | V2 + V1 | 🔨 |
| 20 | `dispatcher.py` | Complete | 🔨 |
| 21 | `orchestration/__init__.py` | NEW | 🔨 |
| 22 | `orchestration/pipeline.py` | NEW | 🔨 |
| 23 | `orchestration/confidence_scorer.py` | NEW | 🔨 |
| 24 | `orchestration/report_generator.py` | NEW | 🔨 |
| 25 | `__init__.py` | Hybrid | 🔨 |

---

## Complete Implementation

### File 1 of 25: `finanalyst_tools/config.py`

```python
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
```

---

### File 2 of 25: `finanalyst_tools/exceptions.py`

```python
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
    
    Features:
    - Auto-generated error_code from class name
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
        """Generate error code from class name (CamelCase → SCREAMING_SNAKE)."""
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
        """Find tools with similar names."""
        name_lower = name.lower()
        similar = [
            t for t in available 
            if name_lower in t.lower() or t.lower() in name_lower
        ]
        return similar[:3] if similar else available[:3]


class ToolExecutionError(ToolError):
    """Raised when a tool fails during execution."""
    
    def __init__(
        self,
        tool_name: str,
        original_error: Exception,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> None:
        message = f"Tool '{tool_name}' failed: {str(original_error)}"
        super().__init__(
            message,
            tool_name=tool_name,
            suggestion="Check tool parameters and try again",
            original_error_type=type(original_error).__name__,
            original_error_message=str(original_error),
            parameters=parameters,
            **kwargs
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
        **kwargs: Any
    ) -> None:
        full_message = f"Parameter '{parameter_name}' for tool '{tool_name}': {message}"
        super().__init__(
            full_message,
            tool_name=tool_name,
            suggestion=f"Provide a valid value for '{parameter_name}'",
            parameter_name=parameter_name,
            expected_type=expected_type,
            actual_value=str(actual_value)[:100] if actual_value is not None else None,
            **kwargs
        )
```

---

### File 3 of 25: `finanalyst_tools/utils/__init__.py`

```python
# finanalyst_tools/utils/__init__.py
"""
Utility functions for FinAnalyst-Pro Agent Tools.

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
    clamp_value,
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
    "clamp_value",
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
```

---

### File 4 of 25: `finanalyst_tools/utils/math_ops.py`

```python
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
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Sequence, Any
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
            result = Decimal(str(value))
        else:
            result = Decimal(str(value))
        
        if precision is not None:
            result = round_decimal(result, precision)
        
        return result
    except (InvalidOperation, ValueError, TypeError):
        return default


def is_effectively_zero(value: Numeric | None, threshold: float = ZERO_THRESHOLD) -> bool:
    """
    Check if a value is effectively zero (within threshold).
    
    Args:
        value: Value to check
        threshold: Threshold for zero comparison
        
    Returns:
        True if value is None or within threshold of zero
    """
    if value is None:
        return True
    return abs(float(value)) < threshold


# ============================================================================
# SAFE ARITHMETIC
# ============================================================================

def safe_divide(
    numerator: Numeric | None,
    denominator: Numeric | None,
    default: Decimal | None = None,
    precision: int = DECIMAL_PLACES["ratio"],
    raise_on_zero: bool = False,
) -> Decimal:
    """
    Safely divide two numbers with zero handling.
    
    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return if division is impossible (default: Decimal("0"))
        precision: Decimal places to round result to
        raise_on_zero: If True, raise DivisionByZeroError instead of returning default
        
    Returns:
        Result of division, or default if denominator is zero/None
        
    Raises:
        DivisionByZeroError: If raise_on_zero is True and denominator is zero
        
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
    
    if numerator is None:
        return default
    
    num = to_decimal(numerator)
    denom = to_decimal(denominator)
    
    if is_effectively_zero(denom):
        if raise_on_zero:
            raise DivisionByZeroError(
                numerator=float(num),
                denominator_name="denominator",
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
) -> Decimal:
    """
    Calculate percentage: (part / whole) × 100.
    
    Args:
        part: The numerator
        whole: The denominator
        precision: Decimal places for result
        
    Returns:
        Percentage value (e.g., 25.00 for 25%)
    """
    if part is None or whole is None:
        return Decimal("0")
    
    ratio = safe_divide(part, whole, precision=precision + 2)
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
    
    if is_effectively_zero(begin) or begin < 0 or end < 0:
        return None
    
    try:
        ratio = float(end / begin)
        cagr = (ratio ** (1 / periods) - 1) * 100
        return round_decimal(Decimal(str(cagr)), precision)
    except (ValueError, ZeroDivisionError, OverflowError):
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
    
    Args:
        values: Values to calculate std dev for
        population: If True, use population std dev; else sample std dev
        
    Returns:
        Standard deviation, or None if insufficient values
    """
    variance = calculate_variance(values, population)
    if variance is None:
        return None
    
    std_dev = Decimal(str(math.sqrt(float(variance))))
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
```

---

### File 5 of 25: `finanalyst_tools/utils/formatting.py`

```python
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
    precision: int = DECIMAL_PLACES["currency"],
    show_symbol: bool = True,
    fallback: str = "N/A",
) -> str:
    """
    Format a monetary value with currency symbol.
    
    Args:
        value: Amount to format
        currency_code: ISO currency code
        precision: Decimal places
        show_symbol: Whether to include currency symbol
        fallback: String to return if value is None
        
    Returns:
        Formatted currency string (e.g., "S$1,234.56")
    """
    if value is None:
        return fallback
    
    dec_value = to_decimal(value, precision=precision)
    formatted = f"{float(dec_value):,.{precision}f}"
    
    if show_symbol:
        symbol = CURRENCY_SYMBOLS.get(currency_code, currency_code)
        # Handle negative values
        if dec_value < 0:
            return f"-{symbol}{formatted.lstrip('-')}"
        return f"{symbol}{formatted}"
    
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
    symbol = TREND_SYMBOLS.get(direction.lower(), "?")
    
    if include_text:
        return f"{symbol} {direction.capitalize()}"
    return symbol


def format_status_indicator(
    status: str,
    include_text: bool = False,
) -> str:
    """
    Format a status as an indicator.
    
    Args:
        status: One of "good", "warning", "error", "info", "unknown"
        include_text: Whether to include text after symbol
        
    Returns:
        Status indicator symbol (e.g., "✅", "⚠️")
    """
    symbol = STATUS_SYMBOLS.get(status.lower(), STATUS_SYMBOLS["unknown"])
    
    if include_text:
        return f"{symbol} {status.capitalize()}"
    return symbol


def format_value_with_unit(
    value: float | Decimal | None,
    metric_name: str,
    precision: int | None = None,
    currency_code: str = DEFAULT_CURRENCY,
) -> str:
    """
    Format a value with appropriate unit based on metric name.
    
    Args:
        value: Value to format
        metric_name: Name of the metric (used to determine unit)
        precision: Override precision (uses default for unit type if None)
        currency_code: Currency code for currency values
        
    Returns:
        Formatted string with appropriate unit
    """
    if value is None:
        return "N/A"
    
    unit = METRIC_UNITS.get(metric_name.lower(), "number")
    
    if unit == "percentage":
        prec = precision if precision is not None else DECIMAL_PLACES["percentage"]
        return format_percentage(value, precision=prec)
    elif unit == "ratio":
        prec = precision if precision is not None else DECIMAL_PLACES["ratio"]
        return format_ratio(value, precision=prec)
    elif unit == "currency":
        prec = precision if precision is not None else DECIMAL_PLACES["currency"]
        return format_currency(value, currency_code=currency_code, precision=prec)
    elif unit == "days":
        prec = precision if precision is not None else 0
        return f"{int(value)} days"
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
) -> str:
    """
    Format data as a Markdown table.
    
    Args:
        headers: Column headers
        rows: Table rows (each row is a sequence of cell values)
        alignment: Column alignments ("left", "center", "right") per column
        
    Returns:
        Markdown table string
    """
    if not headers or not rows:
        return ""
    
    # Convert all values to strings
    str_headers = [str(h) for h in headers]
    str_rows = [[str(cell) for cell in row] for row in rows]
    
    # Calculate column widths
    col_widths = [len(h) for h in str_headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))
    
    # Build alignment row
    if alignment is None:
        alignment = ["left"] * len(headers)
    
    align_row = []
    for i, align in enumerate(alignment):
        width = col_widths[i] if i < len(col_widths) else 3
        if align == "center":
            align_row.append(":" + "-" * (width - 2) + ":")
        elif align == "right":
            align_row.append("-" * (width - 1) + ":")
        else:  # left
            align_row.append(":" + "-" * (width - 1))
    
    # Build table
    lines = []
    
    # Header row
    header_cells = [h.ljust(col_widths[i]) for i, h in enumerate(str_headers)]
    lines.append("| " + " | ".join(header_cells) + " |")
    
    # Alignment row
    lines.append("| " + " | ".join(align_row) + " |")
    
    # Data rows
    for row in str_rows:
        cells = []
        for i, cell in enumerate(row):
            width = col_widths[i] if i < len(col_widths) else len(cell)
            cells.append(cell.ljust(width))
        lines.append("| " + " | ".join(cells) + " |")
    
    return "\n".join(lines)
```

---

### File 6 of 25: `finanalyst_tools/utils/currency.py`

```python
# finanalyst_tools/utils/currency.py
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
    SUPPORTED_CURRENCIES,
    CURRENCY_SYMBOLS,
    ZERO_DECIMAL_CURRENCIES,
    DECIMAL_PLACES,
)
from finanalyst_tools.utils.math_ops import to_decimal, round_decimal


# ============================================================================
# SINGAPORE CONSTANTS
# ============================================================================

SGD_GST_RATE: Final[Decimal] = Decimal("0.09")  # 9% as of 2024

# SFRS for Small Entities thresholds (qualify if meet 2 of 3)
SFRS_THRESHOLDS: Final[dict[str, int]] = {
    "revenue": 10_000_000,      # S$10M
    "total_assets": 10_000_000, # S$10M
    "employees": 50,
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
```

---

# File: finanalyst_tools/validation/utils.py
```py
# finanalyst_tools/validation/utils.py
"""
Utility functions for validation system integration.

Provides:
- Exception to ValidationResult conversion
- ValidationResult to reasoning block formatting
- Cross-system type mapping utilities
- Enhanced error handling and recovery
"""

from __future__ import annotations

from typing import Any, cast
from decimal import Decimal

from finanalyst_tools.models.validation import (
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
)
from finanalyst_tools.exceptions import (
    FinAnalystError,
    CalculationError,
    ValidationError,
    DataError,
    ToolError,
    DivisionByZeroError,
    InvalidInputError,
    SchemaValidationError,
    DataCompletenessError,
    ReconciliationError,
    PlausibilityError,
    DataParsingError,
    MissingDataError,
    ToolNotFoundError,
    ToolExecutionError,
    ToolParameterError,
)


def exception_to_validation_result(
    exc: Exception,
    field: str = "unknown",
    context: str = "general operation"
) -> ValidationResult:
    """
    Convert any exception to a ValidationResult object.
    
    This is the central integration point between the exception hierarchy
    and the validation system.
    
    Args:
        exc: Exception to convert
        field: Field name for validation issue
        context: Context description for error message
        
    Returns:
        ValidationResult with the error
    """
    result = ValidationResult()
    
    # Handle different exception types with appropriate severity mapping
    severity_map = {
        DivisionByZeroError: ValidationSeverity.ERROR,
        InvalidInputError: ValidationSeverity.ERROR,
        SchemaValidationError: ValidationSeverity.ERROR,
        DataCompletenessError: ValidationSeverity.ERROR,
        ReconciliationError: ValidationSeverity.ERROR,
        ToolNotFoundError: ValidationSeverity.ERROR,
        ToolExecutionError: ValidationSeverity.ERROR,
        ToolParameterError: ValidationSeverity.ERROR,
        PlausibilityError: ValidationSeverity.WARNING,
        DataParsingError: ValidationSeverity.ERROR,
        MissingDataError: ValidationSeverity.WARNING,
        ValueError: ValidationSeverity.ERROR,
        TypeError: ValidationSeverity.ERROR,
    }
    
    severity = severity_map.get(type(exc), ValidationSeverity.ERROR)
    
    # Get exception details
    details = {}
    if hasattr(exc, 'details'):
        details = getattr(exc, 'details', {})
    elif hasattr(exc, '__dict__'):
        details = exc.__dict__
    
    # Create validation issue
    issue = ValidationIssue(
        field=field,
        message=str(exc),
        severity=severity,
        actual_value=str(details.get('actual_value', 'unknown')),
        expected=str(details.get('expected', 'valid value')),
        suggestion=get_exception_suggestion(exc, context)
    )
    
    result.add_issue(issue)
    result.context["error_type"] = type(exc).__name__
    result.context["context"] = context
    
    return result


def get_exception_suggestion(exc: Exception, context: str = "general operation") -> str:
    """
    Get a helpful suggestion for resolving an exception.
    
    Args:
        exc: Exception to get suggestion for
        context: Context of the error
        
    Returns:
        Suggestion string
    """
    if isinstance(exc, DivisionByZeroError):
        return "Check denominator values are non-zero before calculation"
    elif isinstance(exc, InvalidInputError):
        if hasattr(exc, 'expected'):
            return f"Provide a value that is: {exc.expected}"
        return "Verify input data format and values"
    elif isinstance(exc, SchemaValidationError):
        return "Check that your data matches the expected schema structure"
    elif isinstance(exc, DataCompletenessError):
        if hasattr(exc, 'missing_fields'):
            return f"Provide the missing fields: {', '.join(exc.missing_fields)}"
        return "Ensure all required data fields are provided"
    elif isinstance(exc, ReconciliationError):
        return "Verify data accuracy across financial statements"
    elif isinstance(exc, PlausibilityError):
        return "Review input data for accuracy - values may be outside normal ranges"
    elif isinstance(exc, ToolNotFoundError):
        if hasattr(exc, 'available_tools'):
            return f"Use one of the available tools: {', '.join(exc.available_tools[:3])}"
        return "Check that the tool name is correct and available"
    elif isinstance(exc, ToolExecutionError):
        return "Check tool parameters and try again"
    elif isinstance(exc, ToolParameterError):
        if hasattr(exc, 'expected_type'):
            return f"Provide a valid {exc.expected_type} value for this parameter"
        return "Check parameter requirements and provide valid values"
    elif isinstance(exc, DataParsingError):
        return "Verify source data format and encoding"
    elif isinstance(exc, MissingDataError):
        if hasattr(exc, 'missing_fields'):
            return f"Provide the missing fields: {', '.join(exc.missing_fields)}"
        return "Ensure all required data is available"
    
    return f"Review the error and try again. Contact support if the issue persists."


def result_to_reasoning_block(result: ValidationResult) -> str:
    """
    Convert ValidationResult to formatted reasoning block.
    
    Args:
        result: Validation result to format
        
    Returns:
        Formatted markdown block
    """
    lines = [
        f"### Validation Result for {result.context.get('context', 'analysis')}",
        "",
        "**Summary**:",
        f"- Status: {'✅ Valid' if result.is_valid else '❌ Invalid'}",
        f"- Errors: {result.error_count}",
        f"- Warnings: {result.warning_count}",
        f"- Info: {result.info_count}",
        "",
    ]
    
    if not result.is_valid:
        lines.append("**Errors**:")
        for issue in result.issues:
            error_icon = "❌ " if issue.severity == ValidationSeverity.ERROR else "⚠️ "
            lines.append(f"  - {error_icon}{issue.field}: {issue.message}")
            if issue.actual_value is not None:
                lines.append(f"    Actual: {issue.actual_value}, Expected: {issue.expected or 'valid value'}")
            if issue.suggestion:
                lines.append(f"    Suggestion: {issue.suggestion}")
        lines.append("")
    
    if result.warning_count > 0:
        lines.append("**Warnings**:")
        for issue in result.warnings:
            lines.append(f"  - ⚠️ {issue.field}: {issue.message}")
            if issue.suggestion:
                lines.append(f"    Suggestion: {issue.suggestion}")
        lines.append("")
    
    if result.info_count > 0:
        lines.append("**Information**:")
        for issue in result.info:
            lines.append(f"  - ℹ️ {issue.field}: {issue.message}")
        lines.append("")
    
    lines.append("**Recommendation**:")
    if result.can_proceed:
        lines.append("✅ Analysis can proceed with the provided data.")
        if result.warning_count > 0:
            lines.append("⚠️ However, please review the warnings for potential data quality issues.")
    else:
        lines.append("❌ Analysis cannot proceed due to validation errors.")
        lines.append("Please correct the errors before continuing.")
    
    return "\n".join(lines)


def merge_validation_results(results: list[ValidationResult]) -> ValidationResult:
    """
    Merge multiple ValidationResult objects into one.
    
    Args:
        results: List of validation results to merge
        
    Returns:
        Merged ValidationResult
    """
    merged = ValidationResult()
    
    for result in results:
        merged = merged.merge(result)
    
    return merged


def is_validation_successful(result: ValidationResult) -> bool:
    """
    Check if validation was successful (no errors).
    
    Args:
        result: ValidationResult to check
        
    Returns:
        True if no errors, False otherwise
    """
    return result.can_proceed


def get_validation_summary(result: ValidationResult) -> dict[str, Any]:
    """
    Get a summary of validation results.
    
    Args:
        result: ValidationResult to summarize
        
    Returns:
        Summary dictionary
    """
    return {
        "is_valid": result.is_valid,
        "can_proceed": result.can_proceed,
        "error_count": result.error_count,
        "warning_count": result.warning_count,
        "info_count": result.info_count,
        "total_issues": result.total_issue_count,
    }
```

