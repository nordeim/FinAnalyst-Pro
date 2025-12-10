FinAnalyst-Pro Tool Architecture: Comprehensive Design & Implementation
Executive Summary
I have conducted a systematic analysis of the capabilities required to maximize FinAnalyst-Pro's effectiveness. The core insight is that LLMs excel at reasoning, interpretation, and natural language but struggle with precise arithmetic, file handling, data persistence, and visualization. A well-designed tool suite addresses these limitations while amplifying the LLM's strengths.

Design Philosophy:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TOOL DESIGN PRINCIPLES                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. PRECISION OFFLOADING  → Calculations in code, interpretation in LLM    │
│  2. SINGLE RESPONSIBILITY → Each tool does one thing excellently           │
│  3. COMPOSABILITY         → Tools chain together for complex workflows     │
│  4. GRACEFUL DEGRADATION  → Clear errors, never silent failures            │
│  5. AUDIT TRAIL           → Every operation logged and traceable           │
│  6. TYPE SAFETY           → Pydantic schemas for all inputs/outputs        │
└─────────────────────────────────────────────────────────────────────────────┘
```

Tool Categories Identified:

Category	Purpose	Tools Count
Data Ingestion	Parse financial data from various formats	5
Data Validation	Ensure data quality and consistency	4
Calculation Engine	Precise financial computations	6
Analysis Tools	Advanced analytical capabilities	5
Visualization	Generate charts and visual reports	4
Reporting	Produce formatted output documents	3
Memory & State	Persist context across sessions	3
External Integration	Connect to accounting systems	3
Total		33 Tools
Phase 1: Architecture Overview
1.1 System Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              FinAnalyst-Pro AI Agent                                │
│                                   (LLM Core)                                        │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │  • Natural Language Understanding    • Reasoning & Interpretation            │  │
│  │  • Recommendation Generation         • Report Narrative Writing              │  │
│  │  • User Intent Classification        • Assumption Documentation              │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                            │
│                                        ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                           TOOL ORCHESTRATION LAYER                            │  │
│  │         (Function Calling Interface / Tool Use Protocol)                      │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                            │
└────────────────────────────────────────┼────────────────────────────────────────────┘
                                         │
         ┌───────────────┬───────────────┼───────────────┬───────────────┐
         ▼               ▼               ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│    DATA     │  │ VALIDATION  │  │ CALCULATION │  │  ANALYSIS   │  │   OUTPUT    │
│  INGESTION  │  │   ENGINE    │  │   ENGINE    │  │    TOOLS    │  │  GENERATION │
├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤
│ • PDF Parse │  │ • Schema    │  │ • Ratios    │  │ • Trends    │  │ • Charts    │
│ • Excel     │  │ • Reconcile │  │ • Margins   │  │ • Forecasts │  │ • PDF Docs  │
│ • CSV       │  │ • Anomaly   │  │ • Growth    │  │ • Scenarios │  │ • Excel     │
│ • JSON      │  │ • Complete  │  │ • Stats     │  │ • Benchmark │  │ • Markdown  │
│ • OCR       │  │             │  │             │  │             │  │             │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │                │                │
       └────────────────┴────────────────┼────────────────┴────────────────┘
                                         │
                                         ▼
                        ┌─────────────────────────────────┐
                        │      SHARED INFRASTRUCTURE      │
                        ├─────────────────────────────────┤
                        │  • Memory Store (SQLite/Redis)  │
                        │  • Audit Logger                 │
                        │  • Error Handler                │
                        │  • Schema Registry              │
                        └─────────────────────────────────┘
```

1.2 Core Infrastructure Components
Before individual tools, we need foundational infrastructure:

```python
# ============================================================================
# FILE: finanalyst_pro/core/infrastructure.py
# PURPOSE: Shared infrastructure for all tools
# ============================================================================

from __future__ import annotations

import logging
import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, Generic
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

# ============================================================================
# CONFIGURATION
# ============================================================================

class ToolConfig(BaseModel):
    """Global configuration for FinAnalyst-Pro tools."""
    
    model_config = ConfigDict(frozen=True)
    
    # Decimal precision for financial calculations
    decimal_places: int = 2
    percentage_decimal_places: int = 2
    
    # Currency settings
    default_currency: str = "SGD"
    currency_symbol: str = "$"
    
    # Audit settings
    enable_audit_log: bool = True
    audit_log_path: Path = Path("./logs/audit.jsonl")
    
    # Memory settings
    memory_backend: str = "sqlite"  # "sqlite" | "redis" | "memory"
    memory_db_path: Path = Path("./data/memory.db")
    
    # Plausibility ranges for validation
    plausibility_ranges: dict[str, tuple[float, float]] = Field(default_factory=lambda: {
        "gross_margin": (-50.0, 95.0),
        "operating_margin": (-100.0, 80.0),
        "net_margin": (-200.0, 50.0),
        "current_ratio": (0.1, 10.0),
        "quick_ratio": (0.05, 8.0),
        "debt_to_equity": (0.0, 20.0),
        "asset_turnover": (0.01, 10.0),
        "inventory_turnover": (0.1, 100.0),
        "roe": (-100.0, 100.0),
        "roa": (-50.0, 50.0),
    })


# Global config instance
CONFIG = ToolConfig()


# ============================================================================
# ENUMS
# ============================================================================

class ConfidenceLevel(str, Enum):
    """Confidence level for analysis results."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class AnalysisType(str, Enum):
    """Types of financial analysis supported."""
    INCOME_STATEMENT = "income_statement"
    CASH_FLOW = "cash_flow"
    BALANCE_SHEET = "balance_sheet"
    PROFITABILITY = "profitability"
    LIQUIDITY = "liquidity"
    SOLVENCY = "solvency"
    EFFICIENCY = "efficiency"
    COMPREHENSIVE = "comprehensive"
    TREND = "trend"
    FORECAST = "forecast"


class ValidationStatus(str, Enum):
    """Status of data validation checks."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================================
# BASE SCHEMAS
# ============================================================================

class ToolResult(BaseModel, Generic[TypeVar("T")]):
    """Standardized result wrapper for all tool outputs."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    success: bool
    tool_name: str
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: Any | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float | None = None
    
    def to_llm_context(self) -> str:
        """Format result for LLM consumption."""
        if self.success:
            return json.dumps({
                "status": "success",
                "tool": self.tool_name,
                "data": self.data,
                "warnings": self.warnings if self.warnings else None,
            }, indent=2, default=str)
        else:
            return json.dumps({
                "status": "error",
                "tool": self.tool_name,
                "error": self.error,
            }, indent=2)


class AuditLogEntry(BaseModel):
    """Audit log entry for compliance and debugging."""
    
    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tool_name: str
    operation: str
    input_hash: str  # SHA-256 of inputs for traceability without storing PII
    success: bool
    execution_time_ms: float
    user_session_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None


# ============================================================================
# DECIMAL UTILITIES
# ============================================================================

class FinancialDecimal:
    """Precise decimal operations for financial calculations."""
    
    @staticmethod
    def from_value(value: int | float | str | Decimal) -> Decimal:
        """Convert any numeric value to Decimal."""
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))
    
    @staticmethod
    def round_currency(value: Decimal, places: int = 2) -> Decimal:
        """Round to currency precision."""
        quantize_str = "0." + "0" * places
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
    
    @staticmethod
    def round_percentage(value: Decimal, places: int = 2) -> Decimal:
        """Round percentage to specified precision."""
        quantize_str = "0." + "0" * places
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
    
    @staticmethod
    def safe_divide(
        numerator: Decimal, 
        denominator: Decimal, 
        default: Decimal | None = None
    ) -> Decimal | None:
        """Safely divide, returning default if denominator is zero."""
        if denominator == 0:
            return default
        return numerator / denominator
    
    @staticmethod
    def percentage(value: Decimal, total: Decimal) -> Decimal | None:
        """Calculate percentage of total."""
        result = FinancialDecimal.safe_divide(value * 100, total)
        if result is not None:
            return FinancialDecimal.round_percentage(result)
        return None


# ============================================================================
# AUDIT LOGGER
# ============================================================================

class AuditLogger:
    """Centralized audit logging for compliance."""
    
    def __init__(self, config: ToolConfig = CONFIG):
        self.config = config
        self.logger = logging.getLogger("finanalyst.audit")
        
        if config.enable_audit_log:
            config.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, entry: AuditLogEntry) -> None:
        """Write audit log entry."""
        if not self.config.enable_audit_log:
            return
        
        with open(self.config.audit_log_path, "a") as f:
            f.write(entry.model_dump_json() + "\n")
    
    @staticmethod
    def hash_input(data: Any) -> str:
        """Create SHA-256 hash of input data for audit trail."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]


# Global audit logger instance
AUDIT = AuditLogger()
```

Phase 2: Data Ingestion Tools
2.1 Tool Catalog — Data Ingestion

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION TOOLS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TOOL: parse_excel_financial_statements                             │   │
│  │  PURPOSE: Extract structured data from Excel financial reports      │   │
│  │  INPUT: Excel file path, sheet mapping configuration                │   │
│  │  OUTPUT: Normalized FinancialStatements object                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TOOL: parse_csv_financial_data                                     │   │
│  │  PURPOSE: Parse CSV files containing financial data                 │   │
│  │  INPUT: CSV file path, column mapping                               │   │
│  │  OUTPUT: Normalized financial data records                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TOOL: parse_pdf_financial_statements                               │   │
│  │  PURPOSE: OCR and extract tables from PDF financial reports         │   │
│  │  INPUT: PDF file path, extraction hints                             │   │
│  │  OUTPUT: Extracted financial data with confidence scores            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TOOL: parse_json_financial_data                                    │   │
│  │  PURPOSE: Parse JSON financial data (API responses, exports)        │   │
│  │  INPUT: JSON string or file path, schema mapping                    │   │
│  │  OUTPUT: Validated FinancialStatements object                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TOOL: normalize_financial_data                                     │   │
│  │  PURPOSE: Standardize disparate formats to canonical schema         │   │
│  │  INPUT: Raw financial data from any source                          │   │
│  │  OUTPUT: Canonicalized FinancialStatements with metadata            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

2.2 Financial Data Schemas

```python
# ============================================================================
# FILE: finanalyst_pro/schemas/financial_statements.py
# PURPOSE: Canonical data models for financial statements
# ============================================================================

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Period(BaseModel):
    """Financial reporting period."""
    
    start_date: date
    end_date: date
    fiscal_year: int
    quarter: Literal[1, 2, 3, 4] | None = None
    period_type: Literal["annual", "quarterly", "monthly", "ytd"] = "annual"
    
    @property
    def label(self) -> str:
        """Human-readable period label."""
        if self.period_type == "annual":
            return f"FY{self.fiscal_year}"
        elif self.quarter:
            return f"Q{self.quarter} FY{self.fiscal_year}"
        return f"{self.start_date} to {self.end_date}"


class IncomeStatementLine(BaseModel):
    """Single line item from income statement."""
    
    period: Period
    
    # Revenue
    total_revenue: Decimal | None = Field(None, alias="total_net_sales")
    cost_of_goods_sold: Decimal | None = Field(None, alias="cost_of_sales")
    
    # Operating expenses (detailed)
    selling_general_admin: Decimal | None = None
    marketing_expense: Decimal | None = None
    research_development: Decimal | None = None
    depreciation_amortization: Decimal | None = None
    other_operating_expense: Decimal | None = None
    total_operating_expenses: Decimal | None = None
    
    # Other income/expense
    interest_income: Decimal | None = None
    interest_expense: Decimal | None = None
    other_income: Decimal | None = None
    other_expense: Decimal | None = None
    
    # Tax and profit
    income_tax_expense: Decimal | None = None
    net_income: Decimal | None = None
    
    # Per share data
    earnings_per_share_basic: Decimal | None = None
    earnings_per_share_diluted: Decimal | None = None
    shares_outstanding: Decimal | None = None
    
    @field_validator("*", mode="before")
    @classmethod
    def convert_to_decimal(cls, v):
        """Convert numeric strings to Decimal."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        if isinstance(v, str):
            # Remove currency symbols and commas
            cleaned = v.replace("$", "").replace(",", "").replace(" ", "")
            if cleaned.startswith("(") and cleaned.endswith(")"):
                cleaned = "-" + cleaned[1:-1]
            try:
                return Decimal(cleaned)
            except Exception:
                return None
        return v


class BalanceSheetLine(BaseModel):
    """Single line item from balance sheet."""
    
    period: Period
    
    # Current Assets
    cash_and_equivalents: Decimal | None = None
    short_term_investments: Decimal | None = None
    accounts_receivable: Decimal | None = None
    inventory: Decimal | None = None
    prepaid_expenses: Decimal | None = None
    other_current_assets: Decimal | None = None
    total_current_assets: Decimal | None = None
    
    # Non-current Assets
    property_plant_equipment: Decimal | None = None
    accumulated_depreciation: Decimal | None = None
    net_ppe: Decimal | None = None
    intangible_assets: Decimal | None = None
    goodwill: Decimal | None = None
    long_term_investments: Decimal | None = None
    other_non_current_assets: Decimal | None = None
    total_non_current_assets: Decimal | None = None
    
    # Total Assets
    total_assets: Decimal | None = None
    
    # Current Liabilities
    accounts_payable: Decimal | None = None
    short_term_debt: Decimal | None = None
    current_portion_long_term_debt: Decimal | None = None
    accrued_expenses: Decimal | None = None
    deferred_revenue: Decimal | None = None
    other_current_liabilities: Decimal | None = None
    total_current_liabilities: Decimal | None = None
    
    # Non-current Liabilities
    long_term_debt: Decimal | None = None
    deferred_tax_liabilities: Decimal | None = None
    other_non_current_liabilities: Decimal | None = None
    total_non_current_liabilities: Decimal | None = None
    
    # Total Liabilities
    total_liabilities: Decimal | None = None
    
    # Shareholders' Equity
    common_stock: Decimal | None = None
    retained_earnings: Decimal | None = None
    additional_paid_in_capital: Decimal | None = None
    treasury_stock: Decimal | None = None
    accumulated_other_comprehensive_income: Decimal | None = None
    total_shareholders_equity: Decimal | None = None
    
    @field_validator("*", mode="before")
    @classmethod
    def convert_to_decimal(cls, v):
        """Convert numeric strings to Decimal."""
        if v is None or isinstance(v, Period):
            return v
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        if isinstance(v, str):
            cleaned = v.replace("$", "").replace(",", "").replace(" ", "")
            if cleaned.startswith("(") and cleaned.endswith(")"):
                cleaned = "-" + cleaned[1:-1]
            try:
                return Decimal(cleaned)
            except Exception:
                return None
        return v


class CashFlowLine(BaseModel):
    """Single line item from cash flow statement."""
    
    period: Period
    
    # Operating Activities
    net_income: Decimal | None = None
    depreciation_amortization: Decimal | None = None
    stock_based_compensation: Decimal | None = None
    deferred_taxes: Decimal | None = None
    changes_in_working_capital: Decimal | None = None
    change_in_receivables: Decimal | None = None
    change_in_inventory: Decimal | None = None
    change_in_payables: Decimal | None = None
    other_operating_activities: Decimal | None = None
    net_cash_from_operating: Decimal | None = None
    
    # Investing Activities
    capital_expenditures: Decimal | None = None
    acquisitions: Decimal | None = None
    purchases_of_investments: Decimal | None = None
    sales_of_investments: Decimal | None = None
    other_investing_activities: Decimal | None = None
    net_cash_from_investing: Decimal | None = None
    
    # Financing Activities
    debt_issued: Decimal | None = None
    debt_repaid: Decimal | None = None
    equity_issued: Decimal | None = None
    dividends_paid: Decimal | None = None
    share_repurchases: Decimal | None = None
    other_financing_activities: Decimal | None = None
    net_cash_from_financing: Decimal | None = None
    
    # Net Change
    net_change_in_cash: Decimal | None = None
    beginning_cash: Decimal | None = None
    ending_cash: Decimal | None = None


class FinancialStatements(BaseModel):
    """Complete set of financial statements for analysis."""
    
    company_name: str | None = None
    currency: str = "SGD"
    reporting_standard: str | None = None  # e.g., "SFRS", "IFRS", "GAAP"
    
    income_statements: list[IncomeStatementLine] = Field(default_factory=list)
    balance_sheets: list[BalanceSheetLine] = Field(default_factory=list)
    cash_flow_statements: list[CashFlowLine] = Field(default_factory=list)
    
    # Metadata
    source_file: str | None = None
    extraction_timestamp: str | None = None
    extraction_confidence: float | None = None
    notes: list[str] = Field(default_factory=list)
    
    def get_periods(self) -> list[Period]:
        """Get all unique periods across statements."""
        periods = set()
        for stmt in self.income_statements:
            periods.add((stmt.period.fiscal_year, stmt.period.quarter))
        for stmt in self.balance_sheets:
            periods.add((stmt.period.fiscal_year, stmt.period.quarter))
        for stmt in self.cash_flow_statements:
            periods.add((stmt.period.fiscal_year, stmt.period.quarter))
        return sorted(periods, key=lambda x: (x[0], x[1] or 0))
    
    def get_income_statement(self, fiscal_year: int) -> IncomeStatementLine | None:
        """Get income statement for a specific year."""
        for stmt in self.income_statements:
            if stmt.period.fiscal_year == fiscal_year:
                return stmt
        return None
    
    def get_balance_sheet(self, fiscal_year: int) -> BalanceSheetLine | None:
        """Get balance sheet for a specific year."""
        for stmt in self.balance_sheets:
            if stmt.period.fiscal_year == fiscal_year:
                return stmt
        return None
    
    def get_cash_flow(self, fiscal_year: int) -> CashFlowLine | None:
        """Get cash flow statement for a specific year."""
        for stmt in self.cash_flow_statements:
            if stmt.period.fiscal_year == fiscal_year:
                return stmt
        return None
```

2.3 Excel Parser Tool Implementation

```python
# ============================================================================
# FILE: finanalyst_pro/tools/ingestion/excel_parser.py
# PURPOSE: Parse financial statements from Excel files
# ============================================================================

from __future__ import annotations

import time
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
import openpyxl
from openpyxl.worksheet.worksheet import Worksheet

from finanalyst_pro.core.infrastructure import (
    ToolResult, AuditLogEntry, AUDIT, ConfidenceLevel
)
from finanalyst_pro.schemas.financial_statements import (
    FinancialStatements, IncomeStatementLine, BalanceSheetLine, 
    CashFlowLine, Period
)


# ============================================================================
# INPUT/OUTPUT SCHEMAS
# ============================================================================

class SheetMapping(BaseModel):
    """Configuration for mapping Excel sheets to statement types."""
    
    income_statement_sheet: str | None = None
    balance_sheet_sheet: str | None = None
    cash_flow_sheet: str | None = None
    
    # Row/column hints for data extraction
    header_row: int = 1
    data_start_row: int = 2
    label_column: int = 1  # Column A = 1
    
    # Year columns (e.g., {2020: 2, 2021: 3, 2022: 4})
    year_columns: dict[int, int] = Field(default_factory=dict)


class ExcelParseInput(BaseModel):
    """Input schema for Excel parsing tool."""
    
    file_path: str = Field(
        description="Path to the Excel file containing financial statements"
    )
    sheet_mapping: SheetMapping | None = Field(
        default=None,
        description="Optional mapping configuration for sheet names and columns"
    )
    auto_detect: bool = Field(
        default=True,
        description="Attempt to auto-detect sheet structure if mapping not provided"
    )
    company_name: str | None = Field(
        default=None,
        description="Company name to include in parsed output"
    )
    currency: str = Field(
        default="SGD",
        description="Currency code for monetary values"
    )


class ExcelParseOutput(BaseModel):
    """Output schema for Excel parsing tool."""
    
    financial_statements: FinancialStatements
    parsing_confidence: ConfidenceLevel
    sheets_found: list[str]
    sheets_parsed: list[str]
    warnings: list[str] = Field(default_factory=list)
    field_coverage: dict[str, float] = Field(
        default_factory=dict,
        description="Percentage of expected fields found per statement type"
    )


# ============================================================================
# TOOL IMPLEMENTATION
# ============================================================================

class ExcelFinancialParser:
    """
    Tool for parsing financial statements from Excel files.
    
    Handles common formats including:
    - Multi-sheet workbooks with separate IS/BS/CF tabs
    - Single-sheet consolidated statements
    - Various row/column orientations
    """
    
    TOOL_NAME = "parse_excel_financial_statements"
    
    # Common sheet name patterns for auto-detection
    INCOME_STATEMENT_PATTERNS = [
        "income statement", "income", "p&l", "profit and loss", 
        "profit & loss", "operations", "statement of operations"
    ]
    BALANCE_SHEET_PATTERNS = [
        "balance sheet", "balance", "bs", "financial position",
        "statement of financial position"
    ]
    CASH_FLOW_PATTERNS = [
        "cash flow", "cash flows", "cf", "statement of cash flows"
    ]
    
    # Common row labels for auto-detection
    REVENUE_LABELS = ["revenue", "total revenue", "net sales", "total net sales", "sales"]
    COGS_LABELS = ["cost of sales", "cost of goods sold", "cogs", "cost of revenue"]
    NET_INCOME_LABELS = ["net income", "net profit", "net earnings", "profit for the year"]
    
    def execute(self, input_data: ExcelParseInput) -> ToolResult[ExcelParseOutput]:
        """
        Parse financial statements from an Excel file.
        
        Args:
            input_data: Parsing configuration and file path
            
        Returns:
            ToolResult containing parsed financial statements or error
        """
        start_time = time.time()
        warnings: list[str] = []
        
        try:
            # Validate file exists
            file_path = Path(input_data.file_path)
            if not file_path.exists():
                return self._error_result(f"File not found: {file_path}", start_time)
            
            if not file_path.suffix.lower() in [".xlsx", ".xls", ".xlsm"]:
                return self._error_result(
                    f"Unsupported file format: {file_path.suffix}", start_time
                )
            
            # Load workbook
            workbook = openpyxl.load_workbook(file_path, data_only=True)
            sheets_found = workbook.sheetnames
            
            # Determine sheet mapping
            if input_data.sheet_mapping:
                mapping = input_data.sheet_mapping
            elif input_data.auto_detect:
                mapping, detect_warnings = self._auto_detect_mapping(workbook)
                warnings.extend(detect_warnings)
            else:
                return self._error_result(
                    "No sheet mapping provided and auto_detect is disabled", 
                    start_time
                )
            
            # Parse each statement type
            income_statements = []
            balance_sheets = []
            cash_flows = []
            sheets_parsed = []
            field_coverage = {}
            
            # Parse Income Statement
            if mapping.income_statement_sheet:
                if mapping.income_statement_sheet in sheets_found:
                    sheet = workbook[mapping.income_statement_sheet]
                    is_data, is_warnings, is_coverage = self._parse_income_statement(
                        sheet, mapping
                    )
                    income_statements.extend(is_data)
                    warnings.extend(is_warnings)
                    sheets_parsed.append(mapping.income_statement_sheet)
                    field_coverage["income_statement"] = is_coverage
                else:
                    warnings.append(
                        f"Income statement sheet '{mapping.income_statement_sheet}' not found"
                    )
            
            # Parse Balance Sheet
            if mapping.balance_sheet_sheet:
                if mapping.balance_sheet_sheet in sheets_found:
                    sheet = workbook[mapping.balance_sheet_sheet]
                    bs_data, bs_warnings, bs_coverage = self._parse_balance_sheet(
                        sheet, mapping
                    )
                    balance_sheets.extend(bs_data)
                    warnings.extend(bs_warnings)
                    sheets_parsed.append(mapping.balance_sheet_sheet)
                    field_coverage["balance_sheet"] = bs_coverage
                else:
                    warnings.append(
                        f"Balance sheet '{mapping.balance_sheet_sheet}' not found"
                    )
            
            # Parse Cash Flow
            if mapping.cash_flow_sheet:
                if mapping.cash_flow_sheet in sheets_found:
                    sheet = workbook[mapping.cash_flow_sheet]
                    cf_data, cf_warnings, cf_coverage = self._parse_cash_flow(
                        sheet, mapping
                    )
                    cash_flows.extend(cf_data)
                    warnings.extend(cf_warnings)
                    sheets_parsed.append(mapping.cash_flow_sheet)
                    field_coverage["cash_flow"] = cf_coverage
                else:
                    warnings.append(
                        f"Cash flow sheet '{mapping.cash_flow_sheet}' not found"
                    )
            
            # Build financial statements object
            financial_statements = FinancialStatements(
                company_name=input_data.company_name,
                currency=input_data.currency,
                income_statements=income_statements,
                balance_sheets=balance_sheets,
                cash_flow_statements=cash_flows,
                source_file=str(file_path),
                extraction_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            
            # Determine parsing confidence
            confidence = self._assess_confidence(
                income_statements, balance_sheets, cash_flows, field_coverage
            )
            
            output = ExcelParseOutput(
                financial_statements=financial_statements,
                parsing_confidence=confidence,
                sheets_found=sheets_found,
                sheets_parsed=sheets_parsed,
                warnings=warnings,
                field_coverage=field_coverage,
            )
            
            # Log audit entry
            self._log_audit(input_data, True, start_time)
            
            execution_time = (time.time() - start_time) * 1000
            
            return ToolResult(
                success=True,
                tool_name=self.TOOL_NAME,
                data=output.model_dump(),
                warnings=warnings,
                execution_time_ms=execution_time,
                metadata={
                    "file": str(file_path),
                    "sheets_parsed": len(sheets_parsed),
                    "periods_found": len(financial_statements.get_periods()),
                }
            )
            
        except Exception as e:
            self._log_audit(input_data, False, start_time, str(e))
            return self._error_result(str(e), start_time)
    
    def _auto_detect_mapping(
        self, 
        workbook: openpyxl.Workbook
    ) -> tuple[SheetMapping, list[str]]:
        """Auto-detect sheet mapping from workbook structure."""
        warnings = []
        mapping = SheetMapping()
        
        for sheet_name in workbook.sheetnames:
            name_lower = sheet_name.lower().strip()
            
            if any(p in name_lower for p in self.INCOME_STATEMENT_PATTERNS):
                mapping.income_statement_sheet = sheet_name
            elif any(p in name_lower for p in self.BALANCE_SHEET_PATTERNS):
                mapping.balance_sheet_sheet = sheet_name
            elif any(p in name_lower for p in self.CASH_FLOW_PATTERNS):
                mapping.cash_flow_sheet = sheet_name
        
        # Try to detect year columns from first detected sheet
        detected_sheet = (
            mapping.income_statement_sheet or 
            mapping.balance_sheet_sheet or 
            mapping.cash_flow_sheet
        )
        
        if detected_sheet:
            sheet = workbook[detected_sheet]
            year_columns = self._detect_year_columns(sheet)
            mapping.year_columns = year_columns
        
        if not any([
            mapping.income_statement_sheet,
            mapping.balance_sheet_sheet,
            mapping.cash_flow_sheet
        ]):
            warnings.append(
                "Could not auto-detect any financial statement sheets. "
                "Please provide explicit sheet_mapping."
            )
        
        return mapping, warnings
    
    def _detect_year_columns(self, sheet: Worksheet) -> dict[int, int]:
        """Detect which columns contain which fiscal years."""
        year_columns = {}
        
        # Check first few rows for year headers
        for row_idx in range(1, 5):
            for col_idx in range(1, sheet.max_column + 1):
                cell_value = sheet.cell(row=row_idx, column=col_idx).value
                if cell_value:
                    # Try to extract year from cell
                    year = self._extract_year(cell_value)
                    if year and 1990 <= year <= 2100:
                        year_columns[year] = col_idx
        
        return year_columns
    
    def _extract_year(self, value: Any) -> int | None:
        """Extract a year from a cell value."""
        if isinstance(value, int) and 1990 <= value <= 2100:
            return value
        if isinstance(value, str):
            # Try patterns like "2022", "FY2022", "FY 2022"
            import re
            match = re.search(r'\b(19|20)\d{2}\b', value)
            if match:
                return int(match.group())
        return None
    
    def _parse_income_statement(
        self,
        sheet: Worksheet,
        mapping: SheetMapping
    ) -> tuple[list[IncomeStatementLine], list[str], float]:
        """Parse income statement data from a sheet."""
        statements = []
        warnings = []
        fields_found = 0
        fields_expected = 10  # Key fields we expect
        
        # Build label-to-row mapping
        label_rows = {}
        for row_idx in range(mapping.data_start_row, sheet.max_row + 1):
            label = sheet.cell(row=row_idx, column=mapping.label_column).value
            if label:
                label_rows[str(label).lower().strip()] = row_idx
        
        # Parse each year column
        for year, col_idx in mapping.year_columns.items():
            period = Period(
                start_date=date(year, 1, 1),
                end_date=date(year, 12, 31),
                fiscal_year=year,
                period_type="annual"
            )
            
            # Extract values by matching labels
            stmt_data = {"period": period}
            
            # Revenue
            revenue_row = self._find_row(label_rows, self.REVENUE_LABELS)
            
```
