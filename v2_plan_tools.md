Deep Analysis: FinAnalyst-Pro AI Agent Tool Suite
Executive Summary
I have conducted a systematic analysis of tool requirements for the FinAnalyst-Pro AI agent, applying first-principles thinking to identify where LLM limitations create risks and where deterministic Python tools provide essential augmentation. This analysis produces a comprehensive 12-category tool taxonomy with 47 individual tools, complete with architectural design, implementation specifications, and production-ready Python code.

Core Insight: LLMs excel at reasoning, interpretation, and natural language but are fundamentally unreliable for:

Precise arithmetic (especially multi-step calculations)
Deterministic data validation
Complex financial formulas with edge cases
Consistent formatting and rounding
Statistical computations
Solution: Offload all calculation, validation, and data transformation to deterministic Python tools while leveraging the LLM for orchestration, interpretation, and recommendation synthesis.

Phase 1: Requirements Analysis
1.1 LLM Limitation Mapping → Tool Requirements
text

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        LLM LIMITATIONS → TOOL REQUIREMENTS                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  LLM WEAKNESS                          TOOL SOLUTION                                │
│  ─────────────────────────────────────────────────────────────────────────────────  │
│                                                                                     │
│  ❌ Arithmetic errors (15-25%          ✅ Deterministic calculation engine          │
│     error rate on multi-step)              with decimal precision                   │
│                                                                                     │
│  ❌ Inconsistent rounding              ✅ Standardized rounding/formatting          │
│     (different results each run)           functions with configurable rules        │
│                                                                                     │
│  ❌ Cannot validate data               ✅ Schema validation with                    │
│     structure systematically               JSON Schema / Pydantic models            │
│                                                                                     │
│  ❌ May miss cross-reference           ✅ Automated reconciliation                  │
│     inconsistencies                        checks with tolerance thresholds         │
│                                                                                     │
│  ❌ Statistical computation            ✅ NumPy/SciPy-based statistical             │
│     unreliable                             analysis tools                           │
│                                                                                     │
│  ❌ Cannot generate actual             ✅ Matplotlib/Plotly visualization          │
│     images/charts                          generators returning base64/URLs         │
│                                                                                     │
│  ❌ No persistent state                ✅ Session state management                  │
│     across conversation                    with analysis history                    │
│                                                                                     │
│  ❌ Cannot access real-time            ✅ Market data connectors                    │
│     market data                            (with user permission)                   │
│                                                                                     │
│  ❌ Complex financial models           ✅ Specialized calculators                   │
│     (DCF, WACC, scenarios)                 (DCF, sensitivity, Monte Carlo)         │
│                                                                                     │
│  ❌ Document parsing                   ✅ PDF/Excel extraction tools                │
│     from uploaded files                    with structured output                   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
1.2 Financial Analysis Workflow → Tool Touchpoints
text

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    FINANCIAL ANALYSIS WORKFLOW WITH TOOL INTEGRATION                │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   USER INPUT                                                                        │
│       │                                                                             │
│       ▼                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────────────┐  │
│   │ PHASE 1: DATA INGESTION                                                     │  │
│   │ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │  │
│   │ │ 🔧 PDF Parser   │  │ 🔧 Excel Parser │  │ 🔧 JSON Parser  │              │  │
│   │ └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │  │
│   │          └────────────────────┼────────────────────┘                        │  │
│   │                               ▼                                             │  │
│   │                    ┌─────────────────────┐                                  │  │
│   │                    │ 🔧 Data Normalizer  │                                  │  │
│   │                    └──────────┬──────────┘                                  │  │
│   └───────────────────────────────┼─────────────────────────────────────────────┘  │
│                                   ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────────┐  │
│   │ PHASE 2: VALIDATION                                                         │  │
│   │ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │  │
│   │ │ 🔧 Schema       │  │ 🔧 Reconciler   │  │ 🔧 Plausibility │              │  │
│   │ │    Validator    │  │                 │  │    Checker      │              │  │
│   │ └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │  │
│   │          └────────────────────┼────────────────────┘                        │  │
│   │                               ▼                                             │  │
│   │                    ┌─────────────────────┐                                  │  │
│   │                    │ 🔧 Validation Report│                                  │  │
│   │                    └──────────┬──────────┘                                  │  │
│   └───────────────────────────────┼─────────────────────────────────────────────┘  │
│                                   ▼                                                 │
│                          [LLM DECISION POINT]                                       │
│                          Proceed / Request Fix                                      │
│                                   │                                                 │
│                                   ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────────┐  │
│   │ PHASE 3: CALCULATION                                                        │  │
│   │ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │  │
│   │ │ 🔧 Profitability│  │ 🔧 Liquidity    │  │ 🔧 Solvency     │              │  │
│   │ │    Calculator   │  │    Calculator   │  │    Calculator   │              │  │
│   │ └─────────────────┘  └─────────────────┘  └─────────────────┘              │  │
│   │ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │  │
│   │ │ 🔧 Efficiency   │  │ 🔧 Growth Rate  │  │ 🔧 Statistical  │              │  │
│   │ │    Calculator   │  │    Calculator   │  │    Analyzer     │              │  │
│   │ └─────────────────┘  └─────────────────┘  └─────────────────┘              │  │
│   └───────────────────────────────┼─────────────────────────────────────────────┘  │
│                                   ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────────┐  │
│   │ PHASE 4: ADVANCED ANALYSIS                                                  │  │
│   │ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │  │
│   │ │ 🔧 Trend        │  │ 🔧 DuPont       │  │ 🔧 Altman Z     │              │  │
│   │ │    Analyzer     │  │    Decomposition│  │    Score        │              │  │
│   │ └─────────────────┘  └─────────────────┘  └─────────────────┘              │  │
│   │ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │  │
│   │ │ 🔧 Common-Size  │  │ 🔧 Variance     │  │ 🔧 Forecasting  │              │  │
│   │ │    Generator    │  │    Analyzer     │  │    Engine       │              │  │
│   │ └─────────────────┘  └─────────────────┘  └─────────────────┘              │  │
│   └───────────────────────────────┼─────────────────────────────────────────────┘  │
│                                   ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────────┐  │
│   │ PHASE 5: VISUALIZATION & REPORTING                                          │  │
│   │ ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │  │
│   │ │ 🔧 Chart        │  │ 🔧 Report       │  │ 🔧 Export       │              │  │
│   │ │    Generator    │  │    Formatter    │  │    Manager      │              │  │
│   │ └─────────────────┘  └─────────────────┘  └─────────────────┘              │  │
│   └───────────────────────────────┼─────────────────────────────────────────────┘  │
│                                   ▼                                                 │
│                          [LLM INTERPRETATION]                                       │
│                          Synthesize & Recommend                                     │
│                                   │                                                 │
│                                   ▼                                                 │
│                            FINAL RESPONSE                                           │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
Phase 2: Tool Taxonomy & Architecture
2.1 Complete Tool Taxonomy
text

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         FINANALYST-PRO TOOL TAXONOMY                                │
│                              47 Tools in 12 Categories                              │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  CATEGORY 1: DATA INGESTION (4 tools)                                              │
│  ├── parse_pdf_financial_statement                                                 │
│  ├── parse_excel_financial_statement                                               │
│  ├── parse_csv_financial_data                                                      │
│  └── normalize_financial_data                                                      │
│                                                                                     │
│  CATEGORY 2: DATA VALIDATION (5 tools)                                             │
│  ├── validate_schema                                                               │
│  ├── check_data_completeness                                                       │
│  ├── reconcile_statements                                                          │
│  ├── check_plausibility                                                            │
│  └── generate_validation_report                                                    │
│                                                                                     │
│  CATEGORY 3: PROFITABILITY CALCULATIONS (7 tools)                                  │
│  ├── calculate_gross_profit_margin                                                 │
│  ├── calculate_operating_profit_margin                                             │
│  ├── calculate_net_profit_margin                                                   │
│  ├── calculate_ebitda_margin                                                       │
│  ├── calculate_roa                                                                 │
│  ├── calculate_roe                                                                 │
│  └── calculate_roce                                                                │
│                                                                                     │
│  CATEGORY 4: LIQUIDITY CALCULATIONS (4 tools)                                      │
│  ├── calculate_current_ratio                                                       │
│  ├── calculate_quick_ratio                                                         │
│  ├── calculate_cash_ratio                                                          │
│  └── calculate_working_capital                                                     │
│                                                                                     │
│  CATEGORY 5: SOLVENCY CALCULATIONS (5 tools)                                       │
│  ├── calculate_debt_to_equity                                                      │
│  ├── calculate_debt_to_assets                                                      │
│  ├── calculate_interest_coverage                                                   │
│  ├── calculate_equity_ratio                                                        │
│  └── calculate_debt_service_coverage                                               │
│                                                                                     │
│  CATEGORY 6: EFFICIENCY CALCULATIONS (6 tools)                                     │
│  ├── calculate_asset_turnover                                                      │
│  ├── calculate_inventory_turnover                                                  │
│  ├── calculate_receivables_turnover                                                │
│  ├── calculate_payables_turnover                                                   │
│  ├── calculate_cash_conversion_cycle                                               │
│  └── calculate_fixed_asset_turnover                                                │
│                                                                                     │
│  CATEGORY 7: GROWTH & TREND ANALYSIS (5 tools)                                     │
│  ├── calculate_yoy_growth                                                          │
│  ├── calculate_cagr                                                                │
│  ├── calculate_moving_average                                                      │
│  ├── detect_trend                                                                  │
│  └── calculate_volatility                                                          │
│                                                                                     │
│  CATEGORY 8: ADVANCED ANALYSIS (5 tools)                                           │
│  ├── perform_dupont_analysis                                                       │
│  ├── calculate_altman_z_score                                                      │
│  ├── generate_common_size_statement                                                │
│  ├── perform_variance_analysis                                                     │
│  └── calculate_break_even_point                                                    │
│                                                                                     │
│  CATEGORY 9: FORECASTING (3 tools)                                                 │
│  ├── forecast_linear_projection                                                    │
│  ├── forecast_growth_based                                                         │
│  └── generate_scenario_analysis                                                    │
│                                                                                     │
│  CATEGORY 10: VISUALIZATION (4 tools)                                              │
│  ├── generate_trend_chart                                                          │
│  ├── generate_ratio_dashboard                                                      │
│  ├── generate_waterfall_chart                                                      │
│  └── generate_comparison_chart                                                     │
│                                                                                     │
│  CATEGORY 11: REPORTING (3 tools)                                                  │
│  ├── format_financial_table                                                        │
│  ├── generate_executive_summary                                                    │
│  └── export_report                                                                 │
│                                                                                     │
│  CATEGORY 12: UTILITIES (6 tools)                                                  │
│  ├── convert_currency                                                              │
│  ├── format_number                                                                 │
│  ├── calculate_percentage                                                          │
│  ├── safe_divide                                                                   │
│  ├── get_period_dates                                                              │
│  └── aggregate_periods                                                             │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
2.2 System Architecture
text

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         FINANALYST-PRO ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│                              ┌─────────────────────┐                                │
│                              │    LLM ORCHESTRATOR │                                │
│                              │   (FinAnalyst-Pro)  │                                │
│                              └──────────┬──────────┘                                │
│                                         │                                           │
│                                         │ Tool Calls (JSON)                         │
│                                         ▼                                           │
│                              ┌─────────────────────┐                                │
│                              │   TOOL DISPATCHER   │                                │
│                              │  (Function Router)  │                                │
│                              └──────────┬──────────┘                                │
│                                         │                                           │
│          ┌──────────────────────────────┼──────────────────────────────┐            │
│          │                              │                              │            │
│          ▼                              ▼                              ▼            │
│  ┌───────────────┐            ┌───────────────┐            ┌───────────────┐       │
│  │   INGESTION   │            │  CALCULATION  │            │    OUTPUT     │       │
│  │    MODULE     │            │    ENGINE     │            │    MODULE     │       │
│  ├───────────────┤            ├───────────────┤            ├───────────────┤       │
│  │ • PDF Parser  │            │ • Ratio Calcs │            │ • Charts      │       │
│  │ • Excel Parser│            │ • Statistics  │            │ • Reports     │       │
│  │ • Normalizer  │            │ • Forecasting │            │ • Exports     │       │
│  └───────┬───────┘            └───────┬───────┘            └───────┬───────┘       │
│          │                            │                            │               │
│          └──────────────────────────────────────────────────────────┘               │
│                                       │                                             │
│                                       ▼                                             │
│                            ┌─────────────────────┐                                  │
│                            │   DATA LAYER        │                                  │
│                            ├─────────────────────┤                                  │
│                            │ • Pydantic Models   │                                  │
│                            │ • Validation Rules  │                                  │
│                            │ • Session State     │                                  │
│                            └─────────────────────┘                                  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
Phase 3: Detailed Tool Specifications & Implementation
3.1 Project Structure
text

finanalyst_tools/
├── __init__.py
├── config.py                    # Configuration and constants
├── models/
│   ├── __init__.py
│   ├── financial_statements.py  # Pydantic models for financial data
│   ├── analysis_results.py      # Models for calculation outputs
│   └── validation.py            # Validation result models
├── ingestion/
│   ├── __init__.py
│   ├── pdf_parser.py
│   ├── excel_parser.py
│   └── normalizer.py
├── validation/
│   ├── __init__.py
│   ├── schema_validator.py
│   ├── reconciliation.py
│   └── plausibility.py
├── calculations/
│   ├── __init__.py
│   ├── profitability.py
│   ├── liquidity.py
│   ├── solvency.py
│   ├── efficiency.py
│   ├── growth.py
│   └── advanced.py
├── forecasting/
│   ├── __init__.py
│   ├── projections.py
│   └── scenarios.py
├── visualization/
│   ├── __init__.py
│   ├── charts.py
│   └── dashboards.py
├── reporting/
│   ├── __init__.py
│   ├── formatters.py
│   └── exporters.py
├── utils/
│   ├── __init__.py
│   ├── math_ops.py
│   ├── formatting.py
│   └── currency.py
├── tool_registry.py             # Central tool registration
└── dispatcher.py                # Tool call routing
3.2 Core Infrastructure Implementation
3.2.1 Configuration & Constants
Python

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
3.2.2 Core Data Models
Python

# finanalyst_tools/models/financial_statements.py
"""
Pydantic models for financial statement data structures.
Provides validation, serialization, and type safety.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator, model_validator


class StatementType(str, Enum):
    """Types of financial statements."""
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"


class PeriodType(str, Enum):
    """Financial reporting period types."""
    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    TTM = "trailing_twelve_months"


# Type aliases for clarity
MonetaryValue = Annotated[Decimal, Field(description="Monetary value in base currency")]
Percentage = Annotated[Decimal, Field(ge=-1000, le=1000, description="Percentage value")]
Ratio = Annotated[Decimal, Field(description="Financial ratio")]


class FinancialPeriod(BaseModel):
    """Represents a financial reporting period."""
    
    year: int = Field(..., ge=1900, le=2100, description="Fiscal year")
    period_type: PeriodType = Field(default=PeriodType.ANNUAL)
    quarter: int | None = Field(default=None, ge=1, le=4, description="Quarter number if quarterly")
    start_date: date | None = Field(default=None, description="Period start date")
    end_date: date | None = Field(default=None, description="Period end date")
    
    @model_validator(mode="after")
    def validate_quarter(self) -> "FinancialPeriod":
        if self.period_type == PeriodType.QUARTERLY and self.quarter is None:
            raise ValueError("Quarter must be specified for quarterly periods")
        return self
    
    def __str__(self) -> str:
        if self.period_type == PeriodType.QUARTERLY:
            return f"Q{self.quarter} {self.year}"
        return str(self.year)


class IncomeStatementData(BaseModel):
    """Income Statement / Profit & Loss data model."""
    
    period: FinancialPeriod
    currency: str = Field(default="SGD", min_length=3, max_length=3)
    
    # Revenue
    total_revenue: MonetaryValue = Field(..., alias="total_net_sales")
    cost_of_goods_sold: MonetaryValue = Field(..., alias="cost_of_sales")
    
    # Operating expenses (can be broken down or aggregated)
    operating_expenses: MonetaryValue | None = None
    marketing_expenses: MonetaryValue | None = None
    research_development: MonetaryValue | None = None
    general_administrative: MonetaryValue | None = None
    depreciation_amortization: MonetaryValue | None = None
    
    # Other income/expenses
    interest_income: MonetaryValue = Field(default=Decimal("0"))
    interest_expense: MonetaryValue = Field(default=Decimal("0"))
    other_income: MonetaryValue = Field(default=Decimal("0"))
    other_expenses: MonetaryValue = Field(default=Decimal("0"))
    
    # Taxes and bottom line
    income_tax_expense: MonetaryValue = Field(default=Decimal("0"), alias="taxes")
    net_income: MonetaryValue | None = None
    
    # Per share data
    earnings_per_share: Decimal | None = None
    diluted_eps: Decimal | None = None
    shares_outstanding: int | None = None
    
    model_config = {"populate_by_name": True}
    
    @property
    def gross_profit(self) -> Decimal:
        """Calculate gross profit."""
        return self.total_revenue - self.cost_of_goods_sold
    
    @property
    def total_operating_expenses(self) -> Decimal:
        """Calculate total operating expenses from components or use provided value."""
        if self.operating_expenses is not None:
            base = self.operating_expenses
        else:
            base = Decimal("0")
        
        # Add any separately listed expenses
        if self.marketing_expenses:
            base += self.marketing_expenses
        if self.research_development:
            base += self.research_development
        if self.general_administrative:
            base += self.general_administrative
            
        return base
    
    @property
    def operating_income(self) -> Decimal:
        """Calculate operating income (EBIT approximation)."""
        return self.gross_profit - self.total_operating_expenses
    
    @property
    def calculated_net_income(self) -> Decimal:
        """Calculate net income if not provided."""
        if self.net_income is not None:
            return self.net_income
        
        ebt = (
            self.operating_income 
            + self.interest_income 
            - self.interest_expense
            + self.other_income
            - self.other_expenses
        )
        return ebt - self.income_tax_expense


class BalanceSheetData(BaseModel):
    """Balance Sheet data model."""
    
    period: FinancialPeriod
    currency: str = Field(default="SGD", min_length=3, max_length=3)
    
    # Current Assets
    cash_and_equivalents: MonetaryValue = Field(..., alias="cash")
    accounts_receivable: MonetaryValue = Field(default=Decimal("0"))
    inventory: MonetaryValue = Field(default=Decimal("0"), alias="inventories")
    prepaid_expenses: MonetaryValue = Field(default=Decimal("0"))
    other_current_assets: MonetaryValue = Field(default=Decimal("0"))
    total_current_assets: MonetaryValue | None = Field(default=None, alias="current_assets")
    
    # Non-Current Assets
    property_plant_equipment: MonetaryValue = Field(default=Decimal("0"))
    intangible_assets: MonetaryValue = Field(default=Decimal("0"))
    long_term_investments: MonetaryValue = Field(default=Decimal("0"))
    other_non_current_assets: MonetaryValue = Field(default=Decimal("0"))
    total_non_current_assets: MonetaryValue | None = None
    
    # Total Assets
    total_assets: MonetaryValue | None = None
    
    # Current Liabilities
    accounts_payable: MonetaryValue = Field(default=Decimal("0"))
    short_term_debt: MonetaryValue = Field(default=Decimal("0"))
    accrued_liabilities: MonetaryValue = Field(default=Decimal("0"))
    deferred_revenue: MonetaryValue = Field(default=Decimal("0"))
    other_current_liabilities: MonetaryValue = Field(default=Decimal("0"))
    total_current_liabilities: MonetaryValue | None = Field(default=None, alias="current_liabilities")
    
    # Non-Current Liabilities
    long_term_debt: MonetaryValue = Field(default=Decimal("0"))
    deferred_tax_liabilities: MonetaryValue = Field(default=Decimal("0"))
    other_non_current_liabilities: MonetaryValue = Field(default=Decimal("0"))
    total_non_current_liabilities: MonetaryValue | None = None
    
    # Total Liabilities
    total_liabilities: MonetaryValue | None = None
    
    # Shareholders' Equity
    common_stock: MonetaryValue = Field(default=Decimal("0"))
    retained_earnings: MonetaryValue = Field(default=Decimal("0"))
    additional_paid_in_capital: MonetaryValue = Field(default=Decimal("0"))
    treasury_stock: MonetaryValue = Field(default=Decimal("0"))
    total_shareholders_equity: MonetaryValue | None = Field(default=None, alias="shareholder_equity")
    
    model_config = {"populate_by_name": True}
    
    @property
    def calculated_current_assets(self) -> Decimal:
        """Calculate total current assets if not provided."""
        if self.total_current_assets is not None:
            return self.total_current_assets
        return (
            self.cash_and_equivalents
            + self.accounts_receivable
            + self.inventory
            + self.prepaid_expenses
            + self.other_current_assets
        )
    
    @property
    def calculated_total_assets(self) -> Decimal:
        """Calculate total assets if not provided."""
        if self.total_assets is not None:
            return self.total_assets
        
        non_current = self.total_non_current_assets or (
            self.property_plant_equipment
            + self.intangible_assets
            + self.long_term_investments
            + self.other_non_current_assets
        )
        return self.calculated_current_assets + non_current
    
    @property
    def calculated_current_liabilities(self) -> Decimal:
        """Calculate total current liabilities if not provided."""
        if self.total_current_liabilities is not None:
            return self.total_current_liabilities
        return (
            self.accounts_payable
            + self.short_term_debt
            + self.accrued_liabilities
            + self.deferred_revenue
            + self.other_current_liabilities
        )
    
    @property
    def calculated_total_liabilities(self) -> Decimal:
        """Calculate total liabilities if not provided."""
        if self.total_liabilities is not None:
            return self.total_liabilities
        
        non_current = self.total_non_current_liabilities or (
            self.long_term_debt
            + self.deferred_tax_liabilities
            + self.other_non_current_liabilities
        )
        return self.calculated_current_liabilities + non_current
    
    @property
    def calculated_equity(self) -> Decimal:
        """Calculate shareholders' equity if not provided."""
        if self.total_shareholders_equity is not None:
            return self.total_shareholders_equity
        return (
            self.common_stock
            + self.retained_earnings
            + self.additional_paid_in_capital
            - self.treasury_stock
        )


class CashFlowStatementData(BaseModel):
    """Cash Flow Statement data model."""
    
    period: FinancialPeriod
    currency: str = Field(default="SGD", min_length=3, max_length=3)
    
    # Operating Activities
    net_income: MonetaryValue = Field(...)
    depreciation_amortization: MonetaryValue = Field(default=Decimal("0"))
    stock_based_compensation: MonetaryValue = Field(default=Decimal("0"))
    changes_in_working_capital: MonetaryValue = Field(default=Decimal("0"))
    other_operating_activities: MonetaryValue = Field(default=Decimal("0"))
    net_cash_from_operating: MonetaryValue | None = None
    
    # Investing Activities
    capital_expenditures: MonetaryValue = Field(default=Decimal("0"), alias="purchase_of_equipment")
    acquisitions: MonetaryValue = Field(default=Decimal("0"))
    investment_purchases: MonetaryValue = Field(default=Decimal("0"))
    investment_sales: MonetaryValue = Field(default=Decimal("0"))
    other_investing_activities: MonetaryValue = Field(default=Decimal("0"))
    net_cash_from_investing: MonetaryValue | None = None
    
    # Financing Activities
    debt_issued: MonetaryValue = Field(default=Decimal("0"))
    debt_repaid: MonetaryValue = Field(default=Decimal("0"))
    dividends_paid: MonetaryValue = Field(default=Decimal("0"))
    stock_issued: MonetaryValue = Field(default=Decimal("0"))
    stock_repurchased: MonetaryValue = Field(default=Decimal("0"))
    other_financing_activities: MonetaryValue = Field(default=Decimal("0"))
    net_cash_from_financing: MonetaryValue | None = None
    
    # Net change
    net_change_in_cash: MonetaryValue | None = Field(default=None, alias="net_cash")
    beginning_cash: MonetaryValue | None = None
    ending_cash: MonetaryValue | None = None
    
    model_config = {"populate_by_name": True}
    
    @property
    def calculated_operating_cash_flow(self) -> Decimal:
        """Calculate operating cash flow if not provided."""
        if self.net_cash_from_operating is not None:
            return self.net_cash_from_operating
        return (
            self.net_income
            + self.depreciation_amortization
            + self.stock_based_compensation
            + self.changes_in_working_capital
            + self.other_operating_activities
        )


class FinancialStatementSet(BaseModel):
    """Complete set of financial statements for a period."""
    
    company_name: str | None = None
    period: FinancialPeriod
    income_statement: IncomeStatementData
    balance_sheet: BalanceSheetData
    cash_flow_statement: CashFlowStatementData | None = None
    
    @model_validator(mode="after")
    def validate_periods_match(self) -> "FinancialStatementSet":
        """Ensure all statements are for the same period."""
        periods = [self.income_statement.period, self.balance_sheet.period]
        if self.cash_flow_statement:
            periods.append(self.cash_flow_statement.period)
        
        if not all(p.year == self.period.year for p in periods):
            raise ValueError("All statements must be for the same period")
        return self


class MultiPeriodFinancialData(BaseModel):
    """Financial data across multiple periods for trend analysis."""
    
    company_name: str | None = None
    currency: str = Field(default="SGD")
    periods: list[FinancialStatementSet] = Field(..., min_length=1)
    
    @field_validator("periods")
    @classmethod
    def sort_periods(cls, v: list[FinancialStatementSet]) -> list[FinancialStatementSet]:
        """Sort periods chronologically."""
        return sorted(v, key=lambda x: x.period.year)
    
    @property
    def years(self) -> list[int]:
        """Get list of years in the dataset."""
        return [p.period.year for p in self.periods]
    
    def get_period(self, year: int) -> FinancialStatementSet | None:
        """Get financial statements for a specific year."""
        for p in self.periods:
            if p.period.year == year:
                return p
        return None
3.2.3 Utility Functions (Math Operations)
Python

# finanalyst_tools/utils/math_ops.py
"""
Safe mathematical operations for financial calculations.
Uses Decimal for precision and handles edge cases.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import TypeVar

from ..config import DECIMAL_PLACES, DEFAULT_ROUNDING


Numeric = TypeVar("Numeric", int, float, Decimal)


class CalculationError(Exception):
    """Raised when a calculation cannot be performed."""
    pass


class DivisionByZeroError(CalculationError):
    """Raised when division by zero is attempted."""
    pass


class InvalidInputError(CalculationError):
    """Raised when input values are invalid."""
    pass


def to_decimal(value: Numeric | str | None, default: Decimal = Decimal("0")) -> Decimal:
    """
    Safely convert a value to Decimal.
    
    Args:
        value: The value to convert
        default: Default value if conversion fails
        
    Returns:
        Decimal representation of the value
    """
    if value is None:
        return default
    
    if isinstance(value, Decimal):
        return value
    
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def safe_divide(
    numerator: Numeric,
    denominator: Numeric,
    default: Decimal | None = None,
    precision: int = DECIMAL_PLACES["ratio"],
    raise_on_zero: bool = False
) -> Decimal | None:
    """
    Safely divide two numbers, handling division by zero.
    
    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return if division by zero (None means return None)
        precision: Decimal places for rounding
        raise_on_zero: If True, raise exception instead of returning default
        
    Returns:
        Result of division or default value
        
    Raises:
        DivisionByZeroError: If raise_on_zero is True and denominator is zero
    """
    num = to_decimal(numerator)
    denom = to_decimal(denominator)
    
    if denom == 0:
        if raise_on_zero:
            raise DivisionByZeroError(
                f"Cannot divide {numerator} by zero"
            )
        return default
    
    result = num / denom
    return round_decimal(result, precision)


def round_decimal(
    value: Decimal | float,
    precision: int = 2,
    rounding: str = ROUND_HALF_UP
) -> Decimal:
    """
    Round a decimal value to specified precision.
    
    Args:
        value: The value to round
        precision: Number of decimal places
        rounding: Rounding mode
        
    Returns:
        Rounded Decimal value
    """
    dec_value = to_decimal(value)
    quantize_str = "0." + "0" * precision if precision > 0 else "0"
    return dec_value.quantize(Decimal(quantize_str), rounding=rounding)


def calculate_percentage(
    part: Numeric,
    whole: Numeric,
    precision: int = DECIMAL_PLACES["percentage"]
) -> Decimal | None:
    """
    Calculate percentage (part/whole * 100).
    
    Args:
        part: The numerator
        whole: The denominator (total)
        precision: Decimal places for result
        
    Returns:
        Percentage value or None if whole is zero
    """
    result = safe_divide(part, whole, precision=precision + 2)
    if result is None:
        return None
    return round_decimal(result * 100, precision)


def calculate_growth_rate(
    current: Numeric,
    previous: Numeric,
    precision: int = DECIMAL_PLACES["growth_rate"]
) -> Decimal | None:
    """
    Calculate growth rate ((current - previous) / previous * 100).
    
    Args:
        current: Current period value
        previous: Previous period value
        precision: Decimal places for result
        
    Returns:
        Growth rate as percentage or None if previous is zero
    """
    curr = to_decimal(current)
    prev = to_decimal(previous)
    
    if prev == 0:
        return None
    
    growth = ((curr - prev) / prev) * 100
    return round_decimal(growth, precision)


def calculate_average(*values: Numeric, precision: int = 2) -> Decimal:
    """
    Calculate arithmetic mean of values.
    
    Args:
        *values: Values to average
        precision: Decimal places for result
        
    Returns:
        Average value
        
    Raises:
        InvalidInputError: If no values provided
    """
    if not values:
        raise InvalidInputError("Cannot calculate average of empty sequence")
    
    decimals = [to_decimal(v) for v in values]
    total = sum(decimals)
    return round_decimal(total / len(decimals), precision)


def calculate_cagr(
    beginning_value: Numeric,
    ending_value: Numeric,
    periods: int,
    precision: int = DECIMAL_PLACES["growth_rate"]
) -> Decimal | None:
    """
    Calculate Compound Annual Growth Rate.
    
    CAGR = (Ending Value / Beginning Value)^(1/n) - 1
    
    Args:
        beginning_value: Starting value
        ending_value: Ending value
        periods: Number of periods
        precision: Decimal places for result
        
    Returns:
        CAGR as percentage or None if calculation not possible
    """
    if periods <= 0:
        return None
    
    begin = to_decimal(beginning_value)
    end = to_decimal(ending_value)
    
    if begin <= 0 or end <= 0:
        return None
    
    # Calculate (end/begin)^(1/periods) - 1
    ratio = float(end / begin)
    cagr = (ratio ** (1 / periods)) - 1
    
    return round_decimal(Decimal(str(cagr * 100)), precision)


def calculate_weighted_average(
    values: list[Numeric],
    weights: list[Numeric],
    precision: int = 2
) -> Decimal:
    """
    Calculate weighted average.
    
    Args:
        values: List of values
        weights: List of weights (must match length of values)
        precision: Decimal places for result
        
    Returns:
        Weighted average
        
    Raises:
        InvalidInputError: If lists are empty or different lengths
    """
    if not values or not weights:
        raise InvalidInputError("Values and weights cannot be empty")
    
    if len(values) != len(weights):
        raise InvalidInputError("Values and weights must have same length")
    
    dec_values = [to_decimal(v) for v in values]
    dec_weights = [to_decimal(w) for w in weights]
    
    total_weight = sum(dec_weights)
    if total_weight == 0:
        raise InvalidInputError("Total weight cannot be zero")
    
    weighted_sum = sum(v * w for v, w in zip(dec_values, dec_weights))
    return round_decimal(weighted_sum / total_weight, precision)
3.3 Validation Tools Implementation
Python

# finanalyst_tools/validation/schema_validator.py
"""
Schema validation for financial data.
Ensures data completeness and structural correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import ValidationError

from ..models.financial_statements import (
    IncomeStatementData,
    BalanceSheetData,
    CashFlowStatementData,
    FinancialStatementSet,
)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Must be fixed before proceeding
    WARNING = "warning"  # Can proceed but results may be affected
    INFO = "info"        # Informational note


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    field: str
    message: str
    severity: ValidationSeverity
    actual_value: Any = None
    expected: str | None = None


@dataclass
class ValidationResult:
    """Complete validation result."""
    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    info: list[ValidationIssue] = field(default_factory=list)
    
    @property
    def error_count(self) -> int:
        return len(self.issues)
    
    @property
    def warning_count(self) -> int:
        return len(self.warnings)
    
    @property
    def can_proceed(self) -> bool:
        """Check if analysis can proceed (no errors)."""
        return self.error_count == 0
    
    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        if issue.severity == ValidationSeverity.ERROR:
            self.issues.append(issue)
            self.is_valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warnings.append(issue)
        else:
            self.info.append(issue)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "can_proceed": self.can_proceed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "errors": [
                {"field": i.field, "message": i.message, "value": str(i.actual_value)}
                for i in self.issues
            ],
            "warnings": [
                {"field": i.field, "message": i.message, "value": str(i.actual_value)}
                for i in self.warnings
            ],
            "info": [
                {"field": i.field, "message": i.message}
                for i in self.info
            ]
        }


def validate_income_statement_schema(data: dict[str, Any]) -> ValidationResult:
    """
    Validate income statement data against schema.
    
    Args:
        data: Raw income statement data dictionary
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult(is_valid=True)
    
    # Required fields for basic analysis
    required_fields = {
        "total_revenue": ["total_net_sales", "revenue", "sales"],
        "cost_of_goods_sold": ["cost_of_sales", "cogs", "cost_of_revenue"],
    }
    
    # Check required fields
    for canonical_name, aliases in required_fields.items():
        found = False
        for alias in [canonical_name] + aliases:
            if alias in data and data[alias] is not None:
                found = True
                break
        
        if not found:
            result.add_issue(ValidationIssue(
                field=canonical_name,
                message=f"Required field '{canonical_name}' is missing",
                severity=ValidationSeverity.ERROR,
                expected=f"One of: {[canonical_name] + aliases}"
            ))
    
    # Try to parse with Pydantic for detailed validation
    try:
        IncomeStatementData.model_validate(data)
    except ValidationError as e:
        for error in e.errors():
            field_name = ".".join(str(loc) for loc in error["loc"])
            result.add_issue(ValidationIssue(
                field=field_name,
                message=error["msg"],
                severity=ValidationSeverity.ERROR,
                actual_value=error.get("input")
            ))
    
    return result


def validate_balance_sheet_schema(data: dict[str, Any]) -> ValidationResult:
    """
    Validate balance sheet data against schema.
    
    Args:
        data: Raw balance sheet data dictionary
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult(is_valid=True)
    
    # Required fields for basic analysis
    required_fields = ["cash", "total_current_assets", "total_current_liabilities"]
    
    for field_name in required_fields:
        if field_name not in data or data[field_name] is None:
            # Check for common aliases
            aliases = {
                "cash": ["cash_and_equivalents", "cash_and_cash_equivalents"],
                "total_current_assets": ["current_assets"],
                "total_current_liabilities": ["current_liabilities"],
            }
            
            alias_found = False
            for alias in aliases.get(field_name, []):
                if alias in data and data[alias] is not None:
                    alias_found = True
                    break
            
            if not alias_found:
                result.add_issue(ValidationIssue(
                    field=field_name,
                    message=f"Required field '{field_name}' is missing",
                    severity=ValidationSeverity.ERROR
                ))
    
    # Check for total assets (critical for many ratios)
    if "total_assets" not in data or data.get("total_assets") is None:
        # Check if we can derive it
        current = data.get("total_current_assets") or data.get("current_assets")
        non_current = data.get("total_non_current_assets")
        
        if current is None:
            result.add_issue(ValidationIssue(
                field="total_assets",
                message="Total assets not provided and cannot be derived",
                severity=ValidationSeverity.WARNING,
                expected="Provide total_assets or component assets"
            ))
    
    return result


def validate_cash_flow_schema(data: dict[str, Any]) -> ValidationResult:
    """
    Validate cash flow statement data against schema.
    
    Args:
        data: Raw cash flow statement data dictionary
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult(is_valid=True)
    
    # Net income is required
    if "net_income" not in data or data["net_income"] is None:
        result.add_issue(ValidationIssue(
            field="net_income",
            message="Net income is required for cash flow analysis",
            severity=ValidationSeverity.ERROR
        ))
    
    # Check for major cash flow categories
    cf_categories = [
        ("net_cash_from_operating", "operating cash flow"),
        ("net_cash_from_investing", "investing cash flow"),
        ("net_cash_from_financing", "financing cash flow"),
    ]
    
    for field_name, description in cf_categories:
        if field_name not in data or data.get(field_name) is None:
            result.add_issue(ValidationIssue(
                field=field_name,
                message=f"{description.title()} not provided; will attempt to derive from components",
                severity=ValidationSeverity.INFO
            ))
    
    return result


def validate_financial_data_completeness(
    income_statement: dict[str, Any] | None,
    balance_sheet: dict[str, Any] | None,
    cash_flow: dict[str, Any] | None,
    analysis_type: str = "comprehensive"
) -> ValidationResult:
    """
    Validate completeness of financial data for requested analysis type.
    
    Args:
        income_statement: Income statement data
        balance_sheet: Balance sheet data
        cash_flow: Cash flow statement data
        analysis_type: Type of analysis requested
        
    Returns:
        ValidationResult indicating data readiness
    """
    result = ValidationResult(is_valid=True)
    
    # Define required statements per analysis type
    requirements = {
        "profitability": {"income_statement": True, "balance_sheet": False, "cash_flow": False},
        "liquidity": {"income_statement": False, "balance_sheet": True, "cash_flow": False},
        "solvency": {"income_statement": False, "balance_sheet": True, "cash_flow": False},
        "efficiency": {"income_statement": True, "balance_sheet": True, "cash_flow": False},
        "cash_flow": {"income_statement": False, "balance_sheet": False, "cash_flow": True},
        "comprehensive": {"income_statement": True, "balance_sheet": True, "cash_flow": True},
    }
    
    reqs = requirements.get(analysis_type, requirements["comprehensive"])
    
    if reqs["income_statement"] and not income_statement:
        result.add_issue(ValidationIssue(
            field="income_statement",
            message=f"Income statement required for {analysis_type} analysis",
            severity=ValidationSeverity.ERROR
        ))
    
    if reqs["balance_sheet"] and not balance_sheet:
        result.add_issue(ValidationIssue(
            field="balance_sheet",
            message=f"Balance sheet required for {analysis_type} analysis",
            severity=ValidationSeverity.ERROR
        ))
    
    if reqs["cash_flow"] and not cash_flow:
        result.add_issue(ValidationIssue(
            field="cash_flow_statement",
            message=f"Cash flow statement required for {analysis_type} analysis",
            severity=ValidationSeverity.WARNING  # Often can proceed without
        ))
    
    # Validate individual statements if provided
    if income_statement:
        is_result = validate_income_statement_schema(income_statement)
        for issue in is_result.issues + is_result.warnings + is_result.info:
            issue.field = f"income_statement.{issue.field}"
            result.add_issue(issue)
    
    if balance_sheet:
        bs_result = validate_balance_sheet_schema(balance_sheet)
        for issue in bs_result.issues + bs_result.warnings + bs_result.info:
            issue.field = f"balance_sheet.{issue.field}"
            result.add_issue(issue)
    
    if cash_flow:
        cf_result = validate_cash_flow_schema(cash_flow)
        for issue in cf_result.issues + cf_result.warnings + cf_result.info:
            issue.field = f"cash_flow.{issue.field}"
            result.add_issue(issue)
    
    return result
3.4 Calculation Tools Implementation
Python

# finanalyst_tools/calculations/profitability.py
"""
Profitability ratio calculations.
All functions return structured results with audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from ..config import PlausibilityRanges, DECIMAL_PLACES
from ..models.financial_statements import IncomeStatementData, BalanceSheetData
from ..utils.math_ops import (
    safe_divide,
    calculate_percentage,
    round_decimal,
    to_decimal,
)


@dataclass
class CalculationResult:
    """Structured result from a financial calculation."""
    
    metric_name: str
    value: Decimal | None
    unit: str  # "percentage", "ratio", "currency", "days"
    formula: str
    inputs: dict[str, Any]
    calculation_steps: list[str]
    is_plausible: bool
    plausibility_range: tuple[float, float] | None
    warnings: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metric_name": self.metric_name,
            "value": float(self.value) if self.value is not None else None,
            "unit": self.unit,
            "formula": self.formula,
            "inputs": {k: float(v) if isinstance(v, Decimal) else v for k, v in self.inputs.items()},
            "calculation_steps": self.calculation_steps,
            "is_plausible": self.is_plausible,
            "plausibility_range": self.plausibility_range,
            "warnings": self.warnings,
        }


def calculate_gross_profit_margin(
    revenue: Decimal | float,
    cost_of_goods_sold: Decimal | float
) -> CalculationResult:
    """
    Calculate Gross Profit Margin.
    
    Formula: (Revenue - COGS) / Revenue × 100
    
    Args:
        revenue: Total revenue/sales
        cost_of_goods_sold: Cost of goods sold
        
    Returns:
        CalculationResult with gross profit margin percentage
    """
    rev = to_decimal(revenue)
    cogs = to_decimal(cost_of_goods_sold)
    
    steps = [
        f"Step 1: Identify values → Revenue = {rev:,.2f}, COGS = {cogs:,.2f}",
        f"Step 2: Calculate Gross Profit = Revenue - COGS",
        f"Step 3: Gross Profit = {rev:,.2f} - {cogs:,.2f} = {rev - cogs:,.2f}",
    ]
    
    gross_profit = rev - cogs
    margin = calculate_percentage(gross_profit, rev)
    
    warnings = []
    is_plausible = True
    plausibility_range = PlausibilityRanges.GROSS_MARGIN
    
    if margin is not None:
        steps.append(f"Step 4: Gross Margin = ({gross_profit:,.2f} / {rev:,.2f}) × 100 = {margin:.2f}%")
        
        if not (plausibility_range[0] <= float(margin) <= plausibility_range[1]):
            is_plausible = False
            warnings.append(
                f"Gross margin {margin:.2f}% is outside typical range "
                f"({plausibility_range[0]}% to {plausibility_range[1]}%)"
            )
    else:
        steps.append("Step 4: Cannot calculate - revenue is zero")
        warnings.append("Revenue is zero; cannot calculate margin")
    
    return CalculationResult(
        metric_name="Gross Profit Margin",
        value=margin,
        unit="percentage",
        formula="(Revenue - COGS) / Revenue × 100",
        inputs={"revenue": rev, "cost_of_goods_sold": cogs, "gross_profit": gross_profit},
        calculation_steps=steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=warnings,
    )


def calculate_operating_profit_margin(
    revenue: Decimal | float,
    cost_of_goods_sold: Decimal | float,
    operating_expenses: Decimal | float,
    marketing_expenses: Decimal | float = Decimal("0"),
    include_marketing_in_opex: bool = False
) -> CalculationResult:
    """
    Calculate Operating Profit Margin (EBIT Margin).
    
    Formula: Operating Profit / Revenue × 100
    Where: Operating Profit = Revenue - COGS - Operating Expenses [- Marketing if separate]
    
    Args:
        revenue: Total revenue/sales
        cost_of_goods_sold: Cost of goods sold
        operating_expenses: Operating expenses
        marketing_expenses: Marketing expenses (if tracked separately)
        include_marketing_in_opex: If True, marketing is already in operating_expenses
        
    Returns:
        CalculationResult with operating profit margin percentage
    """
    rev = to_decimal(revenue)
    cogs = to_decimal(cost_of_goods_sold)
    opex = to_decimal(operating_expenses)
    marketing = to_decimal(marketing_expenses)
    
    gross_profit = rev - cogs
    
    if include_marketing_in_opex:
        total_opex = opex
        formula_note = "(Marketing included in OpEx)"
    else:
        total_opex = opex + marketing
        formula_note = "(Marketing added separately)"
    
    operating_profit = gross_profit - total_opex
    
    steps = [
        f"Step 1: Revenue = {rev:,.2f}, COGS = {cogs:,.2f}",
        f"Step 2: Gross Profit = {rev:,.2f} - {cogs:,.2f} = {gross_profit:,.2f}",
        f"Step 3: Total Operating Expenses = {total_opex:,.2f} {formula_note}",
        f"Step 4: Operating Profit = {gross_profit:,.2f} - {total_opex:,.2f} = {operating_profit:,.2f}",
    ]
    
    margin = calculate_percentage(operating_profit, rev)
    
    warnings = []
    is_plausible = True
    plausibility_range = PlausibilityRanges.OPERATING_MARGIN
    
    if margin is not None:
        steps.append(f"Step 5: Operating Margin = ({operating_profit:,.2f} / {rev:,.2f}) × 100 = {margin:.2f}%")
        
        if not (plausibility_range[0] <= float(margin) <= plausibility_range[1]):
            is_plausible = False
            warnings.append(
                f"Operating margin {margin:.2f}% is outside typical range "
                f"({plausibility_range[0]}% to {plausibility_range[1]}%)"
            )
    else:
        steps.append("Step 5: Cannot calculate - revenue is zero")
        warnings.append("Revenue is zero; cannot calculate margin")
    
    return CalculationResult(
        metric_name="Operating Profit Margin",
        value=margin,
        unit="percentage",
        formula="(Revenue - COGS - OpEx) / Revenue × 100",
        inputs={
            "revenue": rev,
            "cost_of_goods_sold": cogs,
            "operating_expenses": opex,
            "marketing_expenses": marketing,
            "gross_profit": gross_profit,
            "operating_profit": operating_profit,
        },
        calculation_steps=steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=warnings,
    )


def calculate_net_profit_margin(
    revenue: Decimal | float,
    net_income: Decimal | float
) -> CalculationResult:
    """
    Calculate Net Profit Margin.
    
    Formula: Net Income / Revenue × 100
    
    Args:
        revenue: Total revenue/sales
        net_income: Net income (after taxes)
        
    Returns:
        CalculationResult with net profit margin percentage
    """
    rev = to_decimal(revenue)
    net = to_decimal(net_income)
    
    steps = [
        f"Step 1: Identify values → Revenue = {rev:,.2f}, Net Income = {net:,.2f}",
    ]
    
    margin = calculate_percentage(net, rev)
    
    warnings = []
    is_plausible = True
    plausibility_range = PlausibilityRanges.NET_MARGIN
    
    if margin is not None:
        steps.append(f"Step 2: Net Margin = ({net:,.2f} / {rev:,.2f}) × 100 = {margin:.2f}%")
        
        if not (plausibility_range[0] <= float(margin) <= plausibility_range[1]):
            is_plausible = False
            warnings.append(
                f"Net margin {margin:.2f}% is outside typical range "
                f"({plausibility_range[0]}% to {plausibility_range[1]}%)"
            )
        
        # Special check for impossibly high margins
        if float(margin) >= 100:
            warnings.append(
                f"CRITICAL: Net margin ≥ 100% indicates data error. "
                f"Net income ({net:,.2f}) cannot exceed revenue ({rev:,.2f})"
            )
            is_plausible = False
    else:
        steps.append("Step 2: Cannot calculate - revenue is zero")
        warnings.append("Revenue is zero; cannot calculate margin")
    
    return CalculationResult(
        metric_name="Net Profit Margin",
        value=margin,
        unit="percentage",
        formula="Net Income / Revenue × 100",
        inputs={"revenue": rev, "net_income": net},
        calculation_steps=steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=warnings,
    )


def calculate_return_on_assets(
    net_income: Decimal | float,
    total_assets_beginning: Decimal | float,
    total_assets_ending: Decimal | float
) -> CalculationResult:
    """
    Calculate Return on Assets (ROA).
    
    Formula: Net Income / Average Total Assets × 100
    
    Args:
        net_income: Net income for the period
        total_assets_beginning: Total assets at start of period
        total_assets_ending: Total assets at end of period
        
    Returns:
        CalculationResult with ROA percentage
    """
    net = to_decimal(net_income)
    assets_begin = to_decimal(total_assets_beginning)
    assets_end = to_decimal(total_assets_ending)
    
    avg_assets = (assets_begin + assets_end) / 2
    
    steps = [
        f"Step 1: Net Income = {net:,.2f}",
        f"Step 2: Beginning Assets = {assets_begin:,.2f}, Ending Assets = {assets_end:,.2f}",
        f"Step 3: Average Assets = ({assets_begin:,.2f} + {assets_end:,.2f}) / 2 = {avg_assets:,.2f}",
    ]
    
    roa = calculate_percentage(net, avg_assets)
    
    warnings = []
    is_plausible = True
    plausibility_range = PlausibilityRanges.ROA
    
    if roa is not None:
        steps.append(f"Step 4: ROA = ({net:,.2f} / {avg_assets:,.2f}) × 100 = {roa:.2f}%")
        
        if not (plausibility_range[0] <= float(roa) <= plausibility_range[1]):
            is_plausible = False
            warnings.append(
                f"ROA {roa:.2f}% is outside typical range "
                f"({plausibility_range[0]}% to {plausibility_range[1]}%)"
            )
    else:
        steps.append("Step 4: Cannot calculate - average assets is zero")
        warnings.append("Average assets is zero; cannot calculate ROA")
    
    return CalculationResult(
        metric_name="Return on Assets (ROA)",
        value=roa,
        unit="percentage",
        formula="Net Income / Average Total Assets × 100",
        inputs={
            "net_income": net,
            "total_assets_beginning": assets_begin,
            "total_assets_ending": assets_end,
            "average_total_assets": avg_assets,
        },
        calculation_steps=steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=warnings,
    )


def calculate_return_on_equity(
    net_income: Decimal | float,
    shareholders_equity_beginning: Decimal | float,
    shareholders_equity_ending: Decimal | float
) -> CalculationResult:
    """
    Calculate Return on Equity (ROE).
    
    Formula: Net Income / Average Shareholders' Equity × 100
    
    Args:
        net_income: Net income for the period
        shareholders_equity_beginning: Equity at start of period
        shareholders_equity_ending: Equity at end of period
        
    Returns:
        CalculationResult with ROE percentage
    """
    net = to_decimal(net_income)
    equity_begin = to_decimal(shareholders_equity_beginning)
    equity_end = to_decimal(shareholders_equity_ending)
    
    avg_equity = (equity_begin + equity_end) / 2
    
    steps = [
        f"Step 1: Net Income = {net:,.2f}",
        f"Step 2: Beginning Equity = {equity_begin:,.2f}, Ending Equity = {equity_end:,.2f}",
        f"Step 3: Average Equity = ({equity_begin:,.2f} + {equity_end:,.2f}) / 2 = {avg_equity:,.2f}",
    ]
    
    roe = calculate_percentage(net, avg_equity)
    
    warnings = []
    is_plausible = True
    plausibility_range = PlausibilityRanges.ROE
    
    if roe is not None:
        steps.append(f"Step 4: ROE = ({net:,.2f} / {avg_equity:,.2f}) × 100 = {roe:.2f}%")
        
        if not (plausibility_range[0] <= float(roe) <= plausibility_range[1]):
            is_plausible = False
            warnings.append(
                f"ROE {roe:.2f}% is outside typical range "
                f"({plausibility_range[0]}% to {plausibility_range[1]}%)"
            )
    else:
        steps.append("Step 4: Cannot calculate - average equity is zero")
        warnings.append("Average shareholders' equity is zero; cannot calculate ROE")
    
    return CalculationResult(
        metric_name="Return on Equity (ROE)",
        value=roe,
        unit="percentage",
        formula="Net Income / Average Shareholders' Equity × 100",
        inputs={
            "net_income": net,
            "shareholders_equity_beginning": equity_begin,
            "shareholders_equity_ending": equity_end,
            "average_shareholders_equity": avg_equity,
        },
        calculation_steps=steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=warnings,
    )
3.5 Liquidity Tools Implementation
Python

# finanalyst_tools/calculations/liquidity.py
"""
Liquidity ratio calculations.
"""

from __future__ import annotations

from decimal import Decimal

from ..config import PlausibilityRanges
from ..utils.math_ops import safe_divide, to_decimal
from .profitability import CalculationResult


def calculate_current_ratio(
    current_assets: Decimal | float,
    current_liabilities: Decimal | float
) -> CalculationResult:
    """
    Calculate Current Ratio.
    
    Formula: Current Assets / Current Liabilities
    
    Interpretation:
    - > 2.0: Strong liquidity, possibly inefficient asset use
    - 1.5 - 2.0: Healthy liquidity
    - 1.0 - 1.5: Adequate but monitor closely
    - < 1.0: Potential liquidity problems
    
    Args:
        current_assets: Total current assets
        current_liabilities: Total current liabilities
        
    Returns:
        CalculationResult with current ratio
    """
    ca = to_decimal(current_assets)
    cl = to_decimal(current_liabilities)
    
    steps = [
        f"Step 1: Current Assets = {ca:,.2f}",
        f"Step 2: Current Liabilities = {cl:,.2f}",
    ]
    
    ratio = safe_divide(ca, cl, precision=2)
    
    warnings = []
    is_plausible = True
    plausibility_range = PlausibilityRanges.CURRENT_RATIO
    
    if ratio is not None:
        steps.append(f"Step 3: Current Ratio = {ca:,.2f} / {cl:,.2f} = {ratio:.2f}")
        
        # Interpretation
        if ratio < 1:
            warnings.append(
                f"Current ratio below 1.0 indicates current liabilities exceed current assets. "
                f"Potential short-term liquidity risk."
            )
        
        if not (plausibility_range[0] <= float(ratio) <= plausibility_range[1]):
            is_plausible = False
            warnings.append(
                f"Current ratio {ratio:.2f} is outside typical range "
                f"({plausibility_range[0]} to {plausibility_range[1]})"
            )
    else:
        steps.append("Step 3: Cannot calculate - current liabilities is zero")
        warnings.append("Current liabilities is zero; cannot calculate ratio")
    
    return CalculationResult(
        metric_name="Current Ratio",
        value=ratio,
        unit="ratio",
        formula="Current Assets / Current Liabilities",
        inputs={"current_assets": ca, "current_liabilities": cl},
        calculation_steps=steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=warnings,
    )


def calculate_quick_ratio(
    current_assets: Decimal | float,
    inventory: Decimal | float,
    current_liabilities: Decimal | float
) -> CalculationResult:
    """
    Calculate Quick Ratio (Acid Test).
    
    Formula: (Current Assets - Inventory) / Current Liabilities
    
    More conservative than current ratio as inventory may not be easily liquidated.
    
    Args:
        current_assets: Total current assets
        inventory: Total inventory
        current_liabilities: Total current liabilities
        
    Returns:
        CalculationResult with quick ratio
    """
    ca = to_decimal(current_assets)
    inv = to_decimal(inventory)
    cl = to_decimal(current_liabilities)
    
    quick_assets = ca - inv
    
    steps = [
        f"Step 1: Current Assets = {ca:,.2f}, Inventory = {inv:,.2f}",
        f"Step 2: Quick Assets = {ca:,.2f} - {inv:,.2f} = {quick_assets:,.2f}",
        f"Step 3: Current Liabilities = {cl:,.2f}",
    ]
    
    ratio = safe_divide(quick_assets, cl, precision=2)
    
    warnings = []
    is_plausible = True
    plausibility_range = PlausibilityRanges.QUICK_RATIO
    
    if ratio is not None:
        steps.append(f"Step 4: Quick Ratio = {quick_assets:,.2f} / {cl:,.2f} = {ratio:.2f}")
        
        if ratio < 1:
            warnings.append(
                f"Quick ratio below 1.0 indicates the company may struggle to meet "
                f"short-term obligations without selling inventory."
            )
        
        if not (plausibility_range[0] <= float(ratio) <= plausibility_range[1]):
            is_plausible = False
            warnings.append(
                f"Quick ratio {ratio:.2f} is outside typical range "
                f"({plausibility_range[0]} to {plausibility_range[1]})"
            )
    else:
        steps.append("Step 4: Cannot calculate - current liabilities is zero")
        warnings.append("Current liabilities is zero; cannot calculate ratio")
    
    return CalculationResult(
        metric_name="Quick Ratio (Acid Test)",
        value=ratio,
        unit="ratio",
        formula="(Current Assets - Inventory) / Current Liabilities",
        inputs={
            "current_assets": ca,
            "inventory": inv,
            "quick_assets": quick_assets,
            "current_liabilities": cl,
        },
        calculation_steps=steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=warnings,
    )


def calculate_cash_ratio(
    cash_and_equivalents: Decimal | float,
    current_liabilities: Decimal | float
) -> CalculationResult:
    """
    Calculate Cash Ratio.
    
    Formula: Cash and Cash Equivalents / Current Liabilities
    
    Most conservative liquidity measure - only considers cash.
    
    Args:
        cash_and_equivalents: Cash and cash equivalents
        current_liabilities: Total current liabilities
        
    Returns:
        CalculationResult with cash ratio
    """
    cash = to_decimal(cash_and_equivalents)
    cl = to_decimal(current_liabilities)
    
    steps = [
        f"Step 1: Cash and Equivalents = {cash:,.2f}",
        f"Step 2: Current Liabilities = {cl:,.2f}",
    ]
    
    ratio = safe_divide(cash, cl, precision=2)
    
    warnings = []
    is_plausible = True
    plausibility_range = PlausibilityRanges.CASH_RATIO
    
    if ratio is not None:
        steps.append(f"Step 3: Cash Ratio = {cash:,.2f} / {cl:,.2f} = {ratio:.2f}")
        
        if not (plausibility_range[0] <= float(ratio) <= plausibility_range[1]):
            is_plausible = False
            warnings.append(
                f"Cash ratio {ratio:.2f} is outside typical range "
                f"({plausibility_range[0]} to {plausibility_range[1]})"
            )
    else:
        steps.append("Step 3: Cannot calculate - current liabilities is zero")
        warnings.append("Current liabilities is zero; cannot calculate ratio")
    
    return CalculationResult(
        metric_name="Cash Ratio",
        value=ratio,
        unit="ratio",
        formula="Cash and Equivalents / Current Liabilities",
        inputs={"cash_and_equivalents": cash, "current_liabilities": cl},
        calculation_steps=steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=warnings,
    )


def calculate_working_capital(
    current_assets: Decimal | float,
    current_liabilities: Decimal | float
) -> CalculationResult:
    """
    Calculate Working Capital.
    
    Formula: Current Assets - Current Liabilities
    
    Args:
        current_assets: Total current assets
        current_liabilities: Total current liabilities
        
    Returns:
        CalculationResult with working capital amount
    """
    ca = to_decimal(current_assets)
    cl = to_decimal(current_liabilities)
    
    working_capital = ca - cl
    
    steps = [
        f"Step 1: Current Assets = {ca:,.2f}",
        f"Step 2: Current Liabilities = {cl:,.2f}",
        f"Step 3: Working Capital = {ca:,.2f} - {cl:,.2f} = {working_capital:,.2f}",
    ]
    
    warnings = []
    if working_capital < 0:
        warnings.append(
            f"Negative working capital ({working_capital:,.2f}) indicates "
            f"current liabilities exceed current assets. Review short-term financing needs."
        )
    
    return CalculationResult(
        metric_name="Working Capital",
        value=working_capital,
        unit="currency",
        formula="Current Assets - Current Liabilities",
        inputs={"current_assets": ca, "current_liabilities": cl},
        calculation_steps=steps,
        is_plausible=True,  # Any value is technically valid
        plausibility_range=None,
        warnings=warnings,
    )
3.6 Tool Registry & Dispatcher
Python

# finanalyst_tools/tool_registry.py
"""
Central registry for all FinAnalyst tools.
Provides tool metadata for LLM function calling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from enum import Enum


class ToolCategory(str, Enum):
    """Categories of tools."""
    INGESTION = "ingestion"
    VALIDATION = "validation"
    PROFITABILITY = "profitability"
    LIQUIDITY = "liquidity"
    SOLVENCY = "solvency"
    EFFICIENCY = "efficiency"
    GROWTH = "growth"
    ADVANCED = "advanced"
    FORECASTING = "forecasting"
    VISUALIZATION = "visualization"
    REPORTING = "reporting"
    UTILITY = "utility"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "number", "string", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


@dataclass
class ToolDefinition:
    """Complete definition of a tool for LLM consumption."""
    name: str
    description: str
    category: ToolCategory
    parameters: list[ToolParameter]
    returns: str
    example: str | None = None
    function: Callable | None = None
    
    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling schema."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }


class ToolRegistry:
    """Registry of all available tools."""
    
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._register_all_tools()
    
    def _register_all_tools(self) -> None:
        """Register all available tools."""
        
        # Import tool functions
        from .calculations.profitability import (
            calculate_gross_profit_margin,
            calculate_operating_profit_margin,
            calculate_net_profit_margin,
            calculate_return_on_assets,
            calculate_return_on_equity,
        )
        from .calculations.liquidity import (
            calculate_current_ratio,
            calculate_quick_ratio,
            calculate_cash_ratio,
            calculate_working_capital,
        )
        from .validation.schema_validator import (
            validate_financial_data_completeness,
        )
        
        # ============================================================
        # VALIDATION TOOLS
        # ============================================================
        
        self.register(ToolDefinition(
            name="validate_financial_data",
            description="Validate completeness and structure of financial data before analysis. "
                       "MUST be called before any calculation tools.",
            category=ToolCategory.VALIDATION,
            parameters=[
                ToolParameter(
                    name="income_statement",
                    type="object",
                    description="Income statement data with revenue, COGS, expenses, net income",
                    required=False,
                ),
                ToolParameter(
                    name="balance_sheet",
                    type="object",
                    description="Balance sheet data with assets, liabilities, equity",
                    required=False,
                ),
                ToolParameter(
                    name="cash_flow",
                    type="object",
                    description="Cash flow statement data",
                    required=False,
                ),
                ToolParameter(
                    name="analysis_type",
                    type="string",
                    description="Type of analysis to validate for",
                    required=False,
                    default="comprehensive",
                    enum=["profitability", "liquidity", "solvency", "efficiency", 
                          "cash_flow", "comprehensive"],
                ),
            ],
            returns="ValidationResult with is_valid, errors, warnings",
            function=validate_financial_data_completeness,
        ))
        
        # ============================================================
        # PROFITABILITY TOOLS
        # ============================================================
        
        self.register(ToolDefinition(
            name="calculate_gross_profit_margin",
            description="Calculate Gross Profit Margin percentage. "
                       "Formula: (Revenue - COGS) / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter(
                    name="revenue",
                    type="number",
                    description="Total revenue or net sales",
                ),
                ToolParameter(
                    name="cost_of_goods_sold",
                    type="number",
                    description="Cost of goods sold (COGS)",
                ),
            ],
            returns="CalculationResult with value (percentage), formula, steps, and plausibility check",
            example="calculate_gross_profit_margin(revenue=35000, cost_of_goods_sold=7000) → 80.00%",
            function=calculate_gross_profit_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_operating_profit_margin",
            description="Calculate Operating Profit Margin (EBIT margin) percentage. "
                       "Formula: (Revenue - COGS - OpEx) / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter(
                    name="revenue",
                    type="number",
                    description="Total revenue or net sales",
                ),
                ToolParameter(
                    name="cost_of_goods_sold",
                    type="number",
                    description="Cost of goods sold",
                ),
                ToolParameter(
                    name="operating_expenses",
                    type="number",
                    description="Total operating expenses",
                ),
                ToolParameter(
                    name="marketing_expenses",
                    type="number",
                    description="Marketing expenses if tracked separately",
                    required=False,
                    default=0,
                ),
                ToolParameter(
                    name="include_marketing_in_opex",
                    type="boolean",
                    description="Set True if marketing is already included in operating_expenses",
                    required=False,
                    default=False,
                ),
            ],
            returns="CalculationResult with operating margin percentage",
            function=calculate_operating_profit_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_net_profit_margin",
            description="Calculate Net Profit Margin percentage. "
                       "Formula: Net Income / Revenue × 100. "
                       "WARNING: Will flag if margin >= 100% (data error indicator).",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter(
                    name="revenue",
                    type="number",
                    description="Total revenue or net sales",
                ),
                ToolParameter(
                    name="net_income",
                    type="number",
                    description="Net income after taxes",
                ),
            ],
            returns="CalculationResult with net margin percentage and plausibility warnings",
            function=calculate_net_profit_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_return_on_assets",
            description="Calculate Return on Assets (ROA) percentage. "
                       "Formula: Net Income / Average Total Assets × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter(
                    name="net_income",
                    type="number",
                    description="Net income for the period",
                ),
                ToolParameter(
                    name="total_assets_beginning",
                    type="number",
                    description="Total assets at beginning of period",
                ),
                ToolParameter(
                    name="total_assets_ending",
                    type="number",
                    description="Total assets at end of period",
                ),
            ],
            returns="CalculationResult with ROA percentage",
            function=calculate_return_on_assets,
        ))
        
        self.register(ToolDefinition(
            name="calculate_return_on_equity",
            description="Calculate Return on Equity (ROE) percentage. "
                       "Formula: Net Income / Average Shareholders' Equity × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter(
                    name="net_income",
                    type="number",
                    description="Net income for the period",
                ),
                ToolParameter(
                    name="shareholders_equity_beginning",
                    type="number",
                    description="Shareholders' equity at beginning of period",
                ),
                ToolParameter(
                    name="shareholders_equity_ending",
                    type="number",
                    description="Shareholders' equity at end of period",
                ),
            ],
            returns="CalculationResult with ROE percentage",
            function=calculate_return_on_equity,
        ))
        
        # ============================================================
        # LIQUIDITY TOOLS
        # ============================================================
        
        self.register(ToolDefinition(
            name="calculate_current_ratio",
            description="Calculate Current Ratio for liquidity assessment. "
                       "Formula: Current Assets / Current Liabilities. "
                       "Healthy range: 1.5 - 2.0",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter(
                    name="current_assets",
                    type="number",
                    description="Total current assets",
                ),
                ToolParameter(
                    name="current_liabilities",
                    type="number",
                    description="Total current liabilities",
                ),
            ],
            returns="CalculationResult with current ratio and interpretation",
            function=calculate_current_ratio,
        ))
        
        self.register(ToolDefinition(
            name="calculate_quick_ratio",
            description="Calculate Quick Ratio (Acid Test) for conservative liquidity assessment. "
                       "Formula: (Current Assets - Inventory) / Current Liabilities. "
                       "Excludes inventory as it may not be quickly liquidated.",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter(
                    name="current_assets",
                    type="number",
                    description="Total current assets",
                ),
                ToolParameter(
                    name="inventory",
                    type="number",
                    description="Total inventory",
                ),
                ToolParameter(
                    name="current_liabilities",
                    type="number",
                    description="Total current liabilities",
                ),
            ],
            returns="CalculationResult with quick ratio",
            function=calculate_quick_ratio,
        ))
        
        self.register(ToolDefinition(
            name="calculate_cash_ratio",
            description="Calculate Cash Ratio - most conservative liquidity measure. "
                       "Formula: Cash and Equivalents / Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter(
                    name="cash_and_equivalents",
                    type="number",
                    description="Cash and cash equivalents",
                ),
                ToolParameter(
                    name="current_liabilities",
                    type="number",
                    description="Total current liabilities",
                ),
            ],
            returns="CalculationResult with cash ratio",
            function=calculate_cash_ratio,
        ))
        
        self.register(ToolDefinition(
            name="calculate_working_capital",
            description="Calculate Working Capital (absolute amount, not ratio). "
                       "Formula: Current Assets - Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter(
                    name="current_assets",
                    type="number",
                    description="Total current assets",
                ),
                ToolParameter(
                    name="current_liabilities",
                    type="number",
                    description="Total current liabilities",
                ),
            ],
            returns="CalculationResult with working capital amount",
            function=calculate_working_capital,
        ))
    
    def register(self, tool: ToolDefinition) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self, category: ToolCategory | None = None) -> list[ToolDefinition]:
        """List all tools, optionally filtered by category."""
        if category is None:
            return list(self._tools.values())
        return [t for t in self._tools.values() if t.category == category]
    
    def get_openai_tools(self, categories: list[ToolCategory] | None = None) -> list[dict]:
        """Get all tools in OpenAI function calling format."""
        tools = self.list_tools()
        if categories:
            tools = [t for t in tools if t.category in categories]
        return [t.to_openai_schema() for t in tools]
    
    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all tools for prompt inclusion."""
        lines = ["## Available Tools\n"]
        
        by_category: dict[ToolCategory, list[ToolDefinition]] = {}
        for tool in self._tools.values():
            by_category.setdefault(tool.category, []).append(tool)
        
        for category in ToolCategory:
            if category not in by_category:
                continue
            
            lines.append(f"\n### {category.value.title()}\n")
            for tool in by_category[category]:
                params = ", ".join(p.name for p in tool.parameters if p.required)
                lines.append(f"- **{tool.name}**({params}): {tool.description}")
        
        return "\n".join(lines)


# Global registry instance
TOOL_REGISTRY = ToolRegistry()
3.7 Tool Dispatcher
```Python
# finanalyst_tools/dispatcher.py
"""
Tool dispatcher for executing tool calls from LLM.
Handles parameter validation, execution, and error handling.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from typing import Any

from .tool_registry import TOOL_REGISTRY, ToolDefinition


@dataclass
class ToolCallResult:
    """Result of a tool execution."""
    tool_name: str
    success: bool
    result: Any
    error: str | None = None
    execution_time_ms: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result if self.success else None,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string for LLM consumption."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class ToolDispatcher:
    """
    Dispatches tool calls from LLM to actual Python functions.
    Handles validation, execution, and error recovery.
    """
    
    def __init__(self):
        self.registry = TOOL_REGISTRY
    
    def execute(self, tool_name: str, parameters: dict[str, Any]) -> ToolCallResult:
        """
        Execute a tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Dictionary of parameter values
            
        Returns:
            ToolCallResult with execution outcome
        """
        import time
        start_time = time.perf_counter()
        
        # Get tool definition
        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Unknown tool: {tool_name}. Use list_available_tools to see valid options.",
            )
        
        # Validate 
```
