"""
FinAnalyst-Pro Agent Tool Suite
================================

A comprehensive toolkit for financial statement analysis, designed to augment
LLM-based financial analyst agents with precise, deterministic calculations.

This package provides:
- Pydantic models for structured financial data
- Validation tools (schema, reconciliation, plausibility)
- Financial ratio calculators (profitability, liquidity, solvency, efficiency)
- Tool registry and dispatcher for LLM function calling integration

Example Usage:
    from finanalyst_tools import DISPATCHER, TOOL_REGISTRY
    
    # Get available tools for LLM
    tools = TOOL_REGISTRY.get_openai_tools()
    
    # Execute a tool call
    result = DISPATCHER.execute(
        "calculate_gross_profit_margin",
        {"revenue": 100000, "cost_of_goods_sold": 60000}
    )

Author: FinAnalyst-Pro Team
License: MIT
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "FinAnalyst-Pro Team"
__description__ = "Financial analysis tools for AI agents"

# Core configuration
from finanalyst_tools.config import (
    DEFAULT_CURRENCY,
    DECIMAL_PLACES,
    PlausibilityRanges,
    ReconciliationTolerances,
    RoundingMode,
)

# Custom exceptions
from finanalyst_tools.exceptions import (
    FinAnalystError,
    CalculationError,
    ValidationError,
    ReconciliationError,
    DataParsingError,
    ToolExecutionError,
)

# Data models
from finanalyst_tools.models import (
    # Financial statements
    FinancialPeriod,
    PeriodType,
    StatementType,
    IncomeStatementData,
    BalanceSheetData,
    CashFlowStatementData,
    FinancialStatementSet,
    MultiPeriodFinancialData,
    # Results
    CalculationResult,
    MetricCategory,
    TrendDirection,
    # Validation
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
)

# Utility functions
from finanalyst_tools.utils import (
    safe_divide,
    to_decimal,
    round_decimal,
    calculate_percentage,
    calculate_growth_rate,
    format_currency,
    format_percentage,
)

# Validation functions
from finanalyst_tools.validation import (
    validate_financial_data_completeness,
    validate_income_statement_schema,
    validate_balance_sheet_schema,
    run_all_reconciliations,
    check_plausibility,
)

# Calculation functions
from finanalyst_tools.calculations import (
    # Profitability
    calculate_gross_profit_margin,
    calculate_operating_profit_margin,
    calculate_net_profit_margin,
    calculate_ebitda_margin,
    calculate_return_on_assets,
    calculate_return_on_equity,
    # Liquidity
    calculate_current_ratio,
    calculate_quick_ratio,
    calculate_cash_ratio,
    calculate_working_capital,
)

# Tool infrastructure
from finanalyst_tools.tool_registry import TOOL_REGISTRY, ToolRegistry, ToolDefinition
from finanalyst_tools.dispatcher import DISPATCHER, ToolDispatcher, ToolCallResult

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Config
    "DEFAULT_CURRENCY",
    "DECIMAL_PLACES",
    "PlausibilityRanges",
    "ReconciliationTolerances",
    "RoundingMode",
    # Exceptions
    "FinAnalystError",
    "CalculationError",
    "ValidationError",
    "ReconciliationError",
    "DataParsingError",
    "ToolExecutionError",
    # Models - Statements
    "FinancialPeriod",
    "PeriodType",
    "StatementType",
    "IncomeStatementData",
    "BalanceSheetData",
    "CashFlowStatementData",
    "FinancialStatementSet",
    "MultiPeriodFinancialData",
    # Models - Results
    "CalculationResult",
    "MetricCategory",
    "TrendDirection",
    # Models - Validation
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    # Utils
    "safe_divide",
    "to_decimal",
    "round_decimal",
    "calculate_percentage",
    "calculate_growth_rate",
    "format_currency",
    "format_percentage",
    # Validation
    "validate_financial_data_completeness",
    "validate_income_statement_schema",
    "validate_balance_sheet_schema",
    "run_all_reconciliations",
    "check_plausibility",
    # Calculations - Profitability
    "calculate_gross_profit_margin",
    "calculate_operating_profit_margin",
    "calculate_net_profit_margin",
    "calculate_ebitda_margin",
    "calculate_return_on_assets",
    "calculate_return_on_equity",
    # Calculations - Liquidity
    "calculate_current_ratio",
    "calculate_quick_ratio",
    "calculate_cash_ratio",
    "calculate_working_capital",
    # Tool infrastructure
    "TOOL_REGISTRY",
    "ToolRegistry",
    "ToolDefinition",
    "DISPATCHER",
    "ToolDispatcher",
    "ToolCallResult",
]
