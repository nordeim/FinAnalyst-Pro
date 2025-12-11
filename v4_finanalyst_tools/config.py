# finanalyst_tools/config.py
"""
Configuration and constants for FinAnalyst-Pro tools.

This module provides centralized configuration for:
- Decimal precision and rounding modes
- Plausibility ranges for financial metrics
- Reconciliation tolerances
- Currency settings (SGD default)
- Analysis parameters

All constants use Final for immutability and are fully typed.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, ROUND_HALF_EVEN
from enum import Enum
from typing import Final


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
    
    # Gross Margin: (Revenue - COGS) / Revenue
    # Range: Can be negative (selling below cost) to very high (software/services)
    GROSS_MARGIN: Final[tuple[float, float]] = (-50.0, 95.0)
    
    # Operating Margin: Operating Income / Revenue
    # Range: Negative (losses) to moderate (even best companies rarely exceed 50%)
    OPERATING_MARGIN: Final[tuple[float, float]] = (-100.0, 60.0)
    
    # Net Margin: Net Income / Revenue
    # Range: Deep losses possible; >50% is extremely rare and suspicious
    NET_MARGIN: Final[tuple[float, float]] = (-200.0, 50.0)
    
    # EBITDA Margin: EBITDA / Revenue
    EBITDA_MARGIN: Final[tuple[float, float]] = (-50.0, 70.0)
    
    # Return on Assets: Net Income / Average Total Assets
    # Range: Negative possible; >40% is exceptional
    ROA: Final[tuple[float, float]] = (-50.0, 40.0)
    
    # Return on Equity: Net Income / Average Shareholders' Equity
    # Range: Can be extreme with low equity; >60% is very high
    ROE: Final[tuple[float, float]] = (-100.0, 80.0)
    
    # Return on Capital Employed
    ROCE: Final[tuple[float, float]] = (-50.0, 60.0)
    
    # -------------------------------------------------------------------------
    # LIQUIDITY METRICS (ratios)
    # -------------------------------------------------------------------------
    
    # Current Ratio: Current Assets / Current Liabilities
    # Range: Below 1.0 indicates liquidity issues; very high may indicate inefficiency
    CURRENT_RATIO: Final[tuple[float, float]] = (0.1, 10.0)
    
    # Quick Ratio: (Current Assets - Inventory) / Current Liabilities
    QUICK_RATIO: Final[tuple[float, float]] = (0.05, 8.0)
    
    # Cash Ratio: Cash / Current Liabilities
    CASH_RATIO: Final[tuple[float, float]] = (0.0, 5.0)
    
    # -------------------------------------------------------------------------
    # SOLVENCY METRICS (ratios)
    # -------------------------------------------------------------------------
    
    # Debt to Equity: Total Liabilities / Shareholders' Equity
    # Range: 0 (no debt) to very high (highly leveraged)
    DEBT_TO_EQUITY: Final[tuple[float, float]] = (0.0, 10.0)
    
    # Debt to Assets: Total Liabilities / Total Assets
    # Range: 0 to slightly above 1.0 (insolvent but possible)
    DEBT_TO_ASSETS: Final[tuple[float, float]] = (0.0, 1.5)
    
    # Interest Coverage: EBIT / Interest Expense
    # Range: Negative (not covering) to very high (minimal debt)
    INTEREST_COVERAGE: Final[tuple[float, float]] = (-10.0, 100.0)
    
    # Equity Ratio: Shareholders' Equity / Total Assets
    EQUITY_RATIO: Final[tuple[float, float]] = (-0.5, 1.0)
    
    # -------------------------------------------------------------------------
    # EFFICIENCY METRICS (ratios/turnover)
    # -------------------------------------------------------------------------
    
    # Asset Turnover: Revenue / Average Total Assets
    ASSET_TURNOVER: Final[tuple[float, float]] = (0.1, 5.0)
    
    # Inventory Turnover: COGS / Average Inventory
    INVENTORY_TURNOVER: Final[tuple[float, float]] = (0.5, 50.0)
    
    # Receivables Turnover: Revenue / Average Accounts Receivable
    RECEIVABLES_TURNOVER: Final[tuple[float, float]] = (1.0, 50.0)
    
    # Payables Turnover: COGS / Average Accounts Payable
    PAYABLES_TURNOVER: Final[tuple[float, float]] = (1.0, 30.0)
    
    # Fixed Asset Turnover: Revenue / Average Fixed Assets
    FIXED_ASSET_TURNOVER: Final[tuple[float, float]] = (0.1, 20.0)
    
    # -------------------------------------------------------------------------
    # GROWTH METRICS (percentages)
    # -------------------------------------------------------------------------
    
    # Revenue Growth: (Current - Prior) / Prior
    REVENUE_GROWTH: Final[tuple[float, float]] = (-80.0, 500.0)
    
    # Net Income Growth
    NET_INCOME_GROWTH: Final[tuple[float, float]] = (-500.0, 1000.0)
    
    # Asset Growth
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
        # Normalize the metric name
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


# ============================================================================
# RECONCILIATION TOLERANCES
# ============================================================================

class ReconciliationTolerances:
    """
    Acceptable tolerance levels for cross-statement reconciliation.
    
    Expressed as a proportion (0.01 = 1%) of the larger value being compared.
    Different tolerance levels for different reconciliation contexts.
    """
    
    # Strict: For values that should match exactly (e.g., net income across statements)
    STRICT: Final[float] = 0.001  # 0.1%
    
    # Normal: For values that may have minor rounding differences
    NORMAL: Final[float] = 0.01  # 1%
    
    # Loose: For derived values that may have compounding differences
    LOOSE: Final[float] = 0.05  # 5%
    
    # Default tolerance when not specified
    DEFAULT: Final[float] = NORMAL
    
    @classmethod
    def get_tolerance(cls, level: str) -> float:
        """
        Get tolerance value by level name.
        
        Args:
            level: One of "strict", "normal", "loose"
            
        Returns:
            Tolerance as a proportion
        """
        level_upper = level.upper()
        if level_upper == "STRICT":
            return cls.STRICT
        elif level_upper == "LOOSE":
            return cls.LOOSE
        return cls.NORMAL
    
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


# ============================================================================
# CURRENCY CONFIGURATION
# ============================================================================

# Default currency for Singapore SMB context
DEFAULT_CURRENCY: Final[str] = "SGD"

# Supported currencies for the system
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
})


# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

# Minimum number of periods required for meaningful trend analysis
MIN_PERIODS_FOR_TREND: Final[int] = 3

# Default number of periods for forecasting
DEFAULT_FORECAST_PERIODS: Final[int] = 3

# Maximum number of periods to include in analysis
MAX_ANALYSIS_PERIODS: Final[int] = 10

# Days in year for turnover/day calculations
DAYS_IN_YEAR: Final[int] = 365

# Days in month (average) for monthly calculations
DAYS_IN_MONTH: Final[float] = 30.44

# Months in year
MONTHS_IN_YEAR: Final[int] = 12


# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================

# Maximum absolute value for any monetary amount (sanity check)
MAX_MONETARY_VALUE: Final[float] = 1e15  # 1 quadrillion

# Minimum value that's considered effectively zero
ZERO_THRESHOLD: Final[float] = 1e-10


# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

# Symbols for trend indication
TREND_SYMBOLS: Final[dict[str, str]] = {
    "increasing": "↑",
    "decreasing": "↓",
    "stable": "→",
    "volatile": "↕",
}

# Symbols for status indication
STATUS_SYMBOLS: Final[dict[str, str]] = {
    "good": "✅",
    "warning": "⚠️",
    "error": "❌",
    "info": "ℹ️",
    "unknown": "❓",
}

# Large number suffixes
LARGE_NUMBER_SUFFIXES: Final[list[tuple[float, str]]] = [
    (1e12, "T"),   # Trillion
    (1e9, "B"),    # Billion
    (1e6, "M"),    # Million
    (1e3, "K"),    # Thousand
]
