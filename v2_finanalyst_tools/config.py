# finanalyst_tools/config.py
"""
Configuration and constants for FinAnalyst-Pro tools.
"""

from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Final

# ============================================================================
# PRECISION & ROUNDING CONFIGURATION
# ============================================================================

class RoundingMode(Enum):
    """Supported rounding modes for financial calculations."""
    STANDARD = ROUND_HALF_UP  # Default: 0.5 rounds up
    BANKERS = "ROUND_HALF_EVEN"  # Banker's rounding

# Default decimal precision for different contexts
DECIMAL_PLACES: Final[dict[str, int]] = {
    "currency": 2,
    "percentage": 2,
    "ratio": 4,
    "shares": 0,
    "growth_rate": 4,
}

# Default rounding mode
DEFAULT_ROUNDING: Final[RoundingMode] = RoundingMode.STANDARD

# ============================================================================
# PLAUSIBILITY THRESHOLDS
# ============================================================================

class PlausibilityRanges:
    """
    Acceptable ranges for financial ratios.
    Values outside these ranges trigger warnings (not errors).
    """
    
    # Profitability (percentages)
    GROSS_MARGIN: tuple[float, float] = (-50.0, 95.0)
    OPERATING_MARGIN: tuple[float, float] = (-100.0, 60.0)
    NET_MARGIN: tuple[float, float] = (-200.0, 50.0)
    ROA: tuple[float, float] = (-50.0, 40.0)
    ROE: tuple[float, float] = (-100.0, 60.0)
    
    # Liquidity (ratios)
    CURRENT_RATIO: tuple[float, float] = (0.1, 10.0)
    QUICK_RATIO: tuple[float, float] = (0.05, 8.0)
    CASH_RATIO: tuple[float, float] = (0.0, 5.0)
    
    # Solvency (ratios)
    DEBT_TO_EQUITY: tuple[float, float] = (0.0, 10.0)
    DEBT_TO_ASSETS: tuple[float, float] = (0.0, 1.5)
    INTEREST_COVERAGE: tuple[float, float] = (-10.0, 100.0)
    
    # Efficiency (ratios)
    ASSET_TURNOVER: tuple[float, float] = (0.1, 5.0)
    INVENTORY_TURNOVER: tuple[float, float] = (0.5, 50.0)
    RECEIVABLES_TURNOVER: tuple[float, float] = (1.0, 30.0)
    
    # Growth (percentages)
    REVENUE_GROWTH: tuple[float, float] = (-80.0, 500.0)
    
    @classmethod
    def get_range(cls, metric_name: str) -> tuple[float, float] | None:
        """Get plausibility range for a metric by name."""
        return getattr(cls, metric_name.upper(), None)

# ============================================================================
# RECONCILIATION TOLERANCES
# ============================================================================

class ReconciliationTolerances:
    """
    Acceptable tolerance levels for cross-statement reconciliation.
    Expressed as percentage of the larger value being compared.
    """
    
    # Strict: for values that should match exactly
    STRICT: Final[float] = 0.001  # 0.1%
    
    # Normal: for values that may have minor rounding differences
    NORMAL: Final[float] = 0.01  # 1%
    
    # Loose: for derived values that may have compounding differences
    LOOSE: Final[float] = 0.05  # 5%
    
    # Default tolerance
    DEFAULT: Final[float] = NORMAL

# ============================================================================
# CURRENCY CONFIGURATION
# ============================================================================

DEFAULT_CURRENCY: Final[str] = "SGD"

SUPPORTED_CURRENCIES: Final[set[str]] = {
    "SGD", "USD", "EUR", "GBP", "JPY", "CNY", "HKD", "AUD", "MYR", "IDR", "THB"
}

# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

# Minimum periods required for trend analysis
MIN_PERIODS_FOR_TREND: Final[int] = 3

# Default forecast horizon (periods)
DEFAULT_FORECAST_PERIODS: Final[int] = 3

# Days in year for turnover calculations
DAYS_IN_YEAR: Final[int] = 365
