Phase 1: Complete Core Infrastructure — Detailed Implementation Plan
Executive Summary
This document provides a comprehensive implementation plan for Phase 1 of the FinAnalyst-Pro Agent Tool Suite. Phase 1 establishes the foundational infrastructure upon which all subsequent phases will build. I will first present the complete file inventory with detailed specifications, then validate the plan, and finally provide complete, production-ready implementations.

1. Phase 1 Scope Definition
1.1 Objectives
Objective	Description	Success Criteria
Foundation	Establish package structure and configuration	All imports work, config accessible
Type Safety	Pydantic models for all financial data	Validation errors on bad data
Precision	Decimal-based math utilities	Zero floating-point errors
Validation	Schema + reconciliation + plausibility	All system prompt Phase 1 checks
Core Metrics	Profitability + Liquidity calculators	11 metrics with audit trails
Integration	Tool registry + dispatcher	OpenAI-compatible function calling
1.2 File Inventory
text

finanalyst_tools/                          # 20 files total
│
├── __init__.py                            # [F01] Package initialization
├── config.py                              # [F02] Configuration & constants
├── exceptions.py                          # [F03] Custom exception hierarchy
│
├── models/
│   ├── __init__.py                        # [F04] Models package exports
│   ├── financial_statements.py            # [F05] Core financial data models
│   ├── analysis_results.py                # [F06] Calculation result models
│   └── validation.py                      # [F07] Validation result models
│
├── utils/
│   ├── __init__.py                        # [F08] Utils package exports
│   ├── math_ops.py                        # [F09] Safe mathematical operations
│   └── formatting.py                      # [F10] Number/currency formatting
│
├── validation/
│   ├── __init__.py                        # [F11] Validation package exports
│   ├── schema_validator.py                # [F12] Schema validation engine
│   ├── reconciliation.py                  # [F13] Cross-statement reconciliation
│   └── plausibility.py                    # [F14] Plausibility range checks
│
├── calculations/
│   ├── __init__.py                        # [F15] Calculations package exports
│   ├── base.py                            # [F16] Base calculation infrastructure
│   ├── profitability.py                   # [F17] Profitability ratio calculators
│   └── liquidity.py                       # [F18] Liquidity ratio calculators
│
├── tool_registry.py                       # [F19] Central tool registration
└── dispatcher.py                          # [F20] Tool call routing & execution
2. Detailed File Specifications
[F01] finanalyst_tools/__init__.py
Attribute	Value
Purpose	Package initialization with version, metadata, and public API exports
Dependencies	All submodules
Consumers	External code importing the package
Features Checklist:

 Package version constant (__version__)
 Package metadata (__author__, __description__)
 Public API exports (key classes and functions)
 Lazy loading for heavy modules (optional)
 __all__ definition for controlled exports
[F02] finanalyst_tools/config.py
Attribute	Value
Purpose	Centralized configuration constants and settings
Dependencies	decimal, enum (stdlib only)
Consumers	All calculation and validation modules
Features Checklist:

 RoundingMode enum (STANDARD, BANKERS)
 DECIMAL_PLACES dict for different contexts (currency, percentage, ratio)
 PlausibilityRanges class with all metric ranges from system prompt
 ReconciliationTolerances class (STRICT, NORMAL, LOOSE)
 Currency configuration (DEFAULT_CURRENCY, SUPPORTED_CURRENCIES)
 Analysis configuration (MIN_PERIODS_FOR_TREND, DAYS_IN_YEAR)
 Singapore-specific constants (GST_RATE, SFRS thresholds)
[F03] finanalyst_tools/exceptions.py
Attribute	Value
Purpose	Custom exception hierarchy for precise error handling
Dependencies	None (stdlib only)
Consumers	All modules for error raising/handling
Features Checklist:

 FinAnalystError (base exception)
 CalculationError (math operation failures)
 DivisionByZeroError (safe_divide failures)
 InvalidInputError (bad input data)
 ValidationError (schema/plausibility failures)
 ReconciliationError (cross-statement mismatches)
 DataParsingError (ingestion failures)
 ToolExecutionError (dispatcher failures)
 Each exception includes: message, details dict, error_code
[F04] finanalyst_tools/models/__init__.py
Attribute	Value
Purpose	Export all model classes for convenient import
Dependencies	Sibling model modules
Consumers	Any code needing data models
Features Checklist:

 Export all financial statement models
 Export all result models
 Export all validation models
 __all__ definition
[F05] finanalyst_tools/models/financial_statements.py
Attribute	Value
Purpose	Pydantic models for structured financial statement data
Dependencies	pydantic, decimal, datetime, enum
Consumers	Validation, calculation, and reporting modules
Features Checklist:

 StatementType enum (INCOME_STATEMENT, BALANCE_SHEET, CASH_FLOW)
 PeriodType enum (ANNUAL, QUARTERLY, MONTHLY, TTM)
 FinancialPeriod model (year, period_type, quarter, dates)
 IncomeStatementData model with:
 All revenue/sales fields
 COGS and gross profit calculation
 Operating expenses (with breakdown options)
 Interest, taxes, net income
 Per-share data (EPS, diluted EPS)
 Calculated properties (gross_profit, operating_income)
 Field aliases for common naming variations
 BalanceSheetData model with:
 Current assets breakdown
 Non-current assets breakdown
 Current liabilities breakdown
 Non-current liabilities breakdown
 Shareholders' equity components
 Calculated properties for totals
 CashFlowStatementData model with:
 Operating activities section
 Investing activities section
 Financing activities section
 Net change calculation
 FinancialStatementSet (combines all three for a period)
 MultiPeriodFinancialData (list of periods for trend analysis)
 Model validators for cross-field consistency
[F06] finanalyst_tools/models/analysis_results.py
Attribute	Value
Purpose	Structured models for calculation outputs with audit trails
Dependencies	pydantic, decimal, enum
Consumers	Calculation modules, report generator
Features Checklist:

 MetricCategory enum (PROFITABILITY, LIQUIDITY, SOLVENCY, EFFICIENCY, GROWTH)
 TrendDirection enum (IMPROVING, STABLE, DECLINING, VOLATILE)
 CalculationResult dataclass/model with:
 metric_name, value, unit
 formula (human-readable)
 inputs dict (all values used)
 calculation_steps (list of step strings for audit)
 is_plausible flag
 plausibility_range tuple
 warnings list
 to_dict() method for JSON serialization
 to_reasoning_block() for system prompt format
 MetricResult (extends CalculationResult with category, period)
 TrendAnalysisResult (metric over time with direction)
 ComprehensiveAnalysisResult (all metrics grouped by category)
[F07] finanalyst_tools/models/validation.py
Attribute	Value
Purpose	Models for validation results and issues
Dependencies	pydantic, enum
Consumers	Validation modules, dispatcher, report generator
Features Checklist:

 ValidationSeverity enum (ERROR, WARNING, INFO)
 ValidationIssue dataclass with:
 field name
 message
 severity
 actual_value
 expected (description)
 suggestion (how to fix)
 ValidationResult dataclass with:
 is_valid flag
 issues list (errors)
 warnings list
 info list
 Properties: error_count, warning_count, can_proceed
 add_issue() method
 merge() method (combine results)
 to_dict() for JSON
 to_table() for markdown table output
 ReconciliationResult (extends ValidationResult with matched values)
 PlausibilityResult (extends with metric-specific context)
[F08] finanalyst_tools/utils/__init__.py
Attribute	Value
Purpose	Export utility functions for convenient import
Dependencies	Sibling util modules
Consumers	All modules needing utilities
Features Checklist:

 Export all math operations
 Export all formatting functions
 __all__ definition
[F09] finanalyst_tools/utils/math_ops.py
Attribute	Value
Purpose	Safe, precise mathematical operations using Decimal
Dependencies	decimal, typing, ..config, ..exceptions
Consumers	All calculation modules
Features Checklist:

 Numeric type alias (int | float | Decimal)
 to_decimal() - safe conversion with default handling
 safe_divide() - division with zero handling, configurable behavior
 round_decimal() - precision rounding with mode selection
 calculate_percentage() - (part/whole) × 100
 calculate_growth_rate() - ((current-previous)/previous) × 100
 calculate_average() - arithmetic mean with precision
 calculate_cagr() - compound annual growth rate
 calculate_weighted_average() - weighted mean
 is_within_range() - check if value in (min, max) range
 All functions handle None gracefully
 All functions raise appropriate custom exceptions
[F10] finanalyst_tools/utils/formatting.py
Attribute	Value
Purpose	Number, currency, and percentage formatting utilities
Dependencies	decimal, ..config
Consumers	Report generator, display functions
Features Checklist:

 format_currency() - locale-aware currency formatting
 format_percentage() - percentage with configurable decimals
 format_ratio() - ratio formatting (e.g., "2.5x")
 format_number() - general number with thousands separators
 format_compact() - compact notation (1.2M, 500K)
 parse_currency_string() - reverse of format_currency
 Singapore-specific formatting (SGD symbol placement)
[F11] finanalyst_tools/validation/__init__.py
Attribute	Value
Purpose	Export validation functions and classes
Dependencies	Sibling validation modules
Consumers	Dispatcher, pipeline orchestrator
Features Checklist:

 Export all validation functions
 Export result types
 validate_all() convenience function
 __all__ definition
[F12] finanalyst_tools/validation/schema_validator.py
Attribute	Value
Purpose	Validate financial data structure and completeness
Dependencies	pydantic, ..models, ..config
Consumers	Dispatcher (Phase 1 mandatory step)
Features Checklist:

 validate_income_statement_schema() - check IS structure
 validate_balance_sheet_schema() - check BS structure
 validate_cash_flow_schema() - check CF structure
 validate_financial_data_completeness() - check all required data exists for analysis type
 check_required_fields() - helper for field presence
 check_field_types() - ensure numeric fields are numeric
 normalize_field_names() - handle aliases (total_revenue vs revenue vs sales)
 Map analysis types to required fields
 Return ValidationResult with detailed issues
[F13] finanalyst_tools/validation/reconciliation.py
Attribute	Value
Purpose	Cross-statement consistency verification
Dependencies	..models, ..utils.math_ops, ..config
Consumers	Dispatcher (Phase 1 mandatory step)
Features Checklist:

 reconcile_net_income() - IS net income vs CF net income
 reconcile_cash_balance() - BS cash vs CF ending cash
 reconcile_retained_earnings() - RE rollforward consistency
 reconcile_total_assets() - assets = liabilities + equity
 reconcile_working_capital() - derived vs stated
 check_with_tolerance() - helper for approximate matching
 ReconciliationCheck dataclass (check_name, expected, actual, tolerance, passed)
 run_all_reconciliations() - execute all applicable checks
 Return ReconciliationResult with detailed findings
[F14] finanalyst_tools/validation/plausibility.py
Attribute	Value
Purpose	Check calculated metrics against plausible ranges
Dependencies	..models, ..config.PlausibilityRanges
Consumers	Calculation modules (post-calculation check)
Features Checklist:

 check_plausibility() - single metric against range
 check_all_plausibility() - batch check multiple metrics
 get_industry_ranges() - industry-specific ranges (future extension point)
 PlausibilityCheck dataclass (metric, value, range, is_plausible, severity)
 Handle edge cases (negative values where unexpected)
 Provide contextual warnings (not just "out of range")
 Return PlausibilityResult with flags and suggestions
[F15] finanalyst_tools/calculations/__init__.py
Attribute	Value
Purpose	Export all calculation functions
Dependencies	Sibling calculation modules
Consumers	Dispatcher, analysis pipelines
Features Checklist:

 Export all profitability calculators
 Export all liquidity calculators
 Export base classes
 calculate_all_ratios() convenience function
 __all__ definition
[F16] finanalyst_tools/calculations/base.py
Attribute	Value
Purpose	Base classes and utilities for all calculators
Dependencies	..models, ..utils, ..config
Consumers	All calculation modules
Features Checklist:

 BaseCalculator abstract class with:
 Standard result formatting
 Step logging mechanism
 Plausibility check integration
 Warning accumulation
 CalculationStep dataclass (step_number, description, values, result)
 create_calculation_result() factory function
 format_step() helper for consistent step formatting
 add_plausibility_warning() helper
 Common validation (e.g., check all inputs provided)
[F17] finanalyst_tools/calculations/profitability.py
Attribute	Value
Purpose	Calculate profitability ratios with full audit trails
Dependencies	..models, ..utils, ..config, .base
Consumers	Dispatcher, analysis pipelines
Features Checklist:

 calculate_gross_profit_margin() - (Revenue - COGS) / Revenue × 100
 calculate_operating_profit_margin() - includes OpEx, Marketing handling
 calculate_net_profit_margin() - Net Income / Revenue × 100
 calculate_ebitda_margin() - EBITDA / Revenue × 100
 calculate_return_on_assets() - uses average total assets
 calculate_return_on_equity() - uses average shareholders' equity
 calculate_return_on_capital_employed() - EBIT / Capital Employed
 Each function returns CalculationResult with:
 Step-by-step calculation log
 All input values recorded
 Plausibility assessment
 Contextual warnings
 Helper: extract_profitability_inputs() from statement models
[F18] finanalyst_tools/calculations/liquidity.py
Attribute	Value
Purpose	Calculate liquidity ratios with full audit trails
Dependencies	..models, ..utils, ..config, .base
Consumers	Dispatcher, analysis pipelines
Features Checklist:

 calculate_current_ratio() - CA / CL
 calculate_quick_ratio() - (CA - Inventory) / CL
 calculate_cash_ratio() - Cash / CL
 calculate_working_capital() - CA - CL (absolute amount)
 Each function returns CalculationResult with:
 Step-by-step calculation log
 Interpretation guidance in warnings
 Plausibility assessment
 Helper: extract_liquidity_inputs() from balance sheet model
[F19] finanalyst_tools/tool_registry.py
Attribute	Value
Purpose	Central registry of all tools with metadata for LLM function calling
Dependencies	All calculation and validation modules, typing, dataclasses
Consumers	Dispatcher, LLM integration layer
Features Checklist:

 ToolCategory enum (all 12 categories)
 ToolParameter dataclass (name, type, description, required, default, enum)
 ToolDefinition dataclass:
 name, description, category
 parameters list
 returns description
 example usage
 function reference
 to_openai_schema() method
 to_anthropic_schema() method (future-proof)
 ToolRegistry class:
 register() method
 get() method
 list_tools() with category filter
 get_openai_tools() for function calling
 get_tool_descriptions() for prompt inclusion
 Register all Phase 1 tools (validation + profitability + liquidity)
 TOOL_REGISTRY singleton instance
[F20] finanalyst_tools/dispatcher.py
Attribute	Value
Purpose	Route tool calls from LLM to Python functions, handle execution
Dependencies	.tool_registry, .exceptions, typing, json, time
Consumers	LLM integration layer, agent framework
Features Checklist:

 ToolCallResult dataclass:
 tool_name, success, result
 error message (if failed)
 execution_time_ms
 to_dict() and to_json() methods
 ToolDispatcher class:
 execute() - main execution entry point
 validate_parameters() - check params against schema
 coerce_types() - convert strings to Decimal, etc.
 handle_error() - graceful error recovery
 log_execution() - optional execution logging
 Parameter validation against tool schema
 Type coercion (string → Decimal, string → int)
 Execution timing measurement
 Comprehensive error handling with actionable messages
 Unknown tool handling with suggestions
 DISPATCHER singleton instance
3. Dependency Graph
text

                                    ┌─────────────────┐
                                    │   __init__.py   │
                                    │      [F01]      │
                                    └────────┬────────┘
                                             │ imports
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
                    ▼                        ▼                        ▼
           ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
           │   config.py   │        │ exceptions.py │        │  dispatcher   │
           │     [F02]     │        │     [F03]     │        │    [F20]      │
           └───────┬───────┘        └───────┬───────┘        └───────┬───────┘
                   │                        │                        │
                   │                        │                        │
    ┌──────────────┼──────────────┐         │              ┌─────────┴─────────┐
    │              │              │         │              │                   │
    ▼              ▼              ▼         │              ▼                   │
┌───────┐    ┌──────────┐   ┌──────────┐   │      ┌──────────────┐            │
│models/│    │  utils/  │   │validation│   │      │tool_registry │            │
│ [F04- │    │ [F08-10] │   │ [F11-14] │   │      │    [F19]     │            │
│  F07] │    └────┬─────┘   └────┬─────┘   │      └──────┬───────┘            │
└───┬───┘         │              │         │             │                    │
    │             │              │         │             │                    │
    │             └──────┬───────┘         │             │                    │
    │                    │                 │             │                    │
    ▼                    ▼                 │             ▼                    │
┌─────────────────────────────────────┐    │    ┌────────────────┐            │
│           calculations/             │◄───┘    │ All validation │            │
│            [F15-18]                 │         │ & calculation  │◄───────────┘
└─────────────────────────────────────┘         │   functions    │
                                                └────────────────┘
4. Implementation Order
Based on dependencies, files must be implemented in this order:

Order	File(s)	Rationale
1	config.py [F02]	No dependencies, used by all
2	exceptions.py [F03]	No dependencies, used by all
3	models/validation.py [F07]	Needed by validation modules
4	models/financial_statements.py [F05]	Core data structures
5	models/analysis_results.py [F06]	Needed by calculations
6	models/__init__.py [F04]	Exports after models done
7	utils/math_ops.py [F09]	Needed by calculations
8	utils/formatting.py [F10]	Needed by results
9	utils/__init__.py [F08]	Exports after utils done
10	validation/schema_validator.py [F12]	Uses models
11	validation/reconciliation.py [F13]	Uses models, math_ops
12	validation/plausibility.py [F14]	Uses config, models
13	validation/__init__.py [F11]	Exports after validation done
14	calculations/base.py [F16]	Foundation for calculators
15	calculations/profitability.py [F17]	Uses base, utils
16	calculations/liquidity.py [F18]	Uses base, utils
17	calculations/__init__.py [F15]	Exports after calcs done
18	tool_registry.py [F19]	Registers all tools
19	dispatcher.py [F20]	Uses registry
20	__init__.py [F01]	Final package exports
5. Validation of Implementation Plan
5.1 Completeness Check
System Prompt Requirement	Covered By	Status
Required statements present check	schema_validator.py [F12]	✅
Required fields for analysis exist	schema_validator.py [F12]	✅
Cross-statement reconciliation	reconciliation.py [F13]	✅
Plausibility checks (margins, ratios)	plausibility.py [F14]	✅
Step-by-step calculation reasoning	analysis_results.py [F06]	✅
Gross Margin formula	profitability.py [F17]	✅
Operating Margin formula	profitability.py [F17]	✅
Net Margin formula	profitability.py [F17]	✅
ROA formula	profitability.py [F17]	✅
ROE formula	profitability.py [F17]	✅
Current Ratio formula	liquidity.py [F18]	✅
Quick Ratio formula	liquidity.py [F18]	✅
SGD default currency	config.py [F02]	✅
5.2 Architecture Validation
Principle	Implementation	Status
Decimal precision for finance	math_ops.py uses Decimal throughout	✅
Structured results with audit trail	CalculationResult has calculation_steps	✅
OpenAI function calling compatible	ToolDefinition.to_openai_schema()	✅
Graceful error handling	Custom exceptions + dispatcher handling	✅
Validation-first enforcement	Dispatcher can enforce validation	✅
5.3 Risk Assessment
Risk	Mitigation	Status
Pydantic v1 vs v2 compatibility	Use v2 syntax exclusively	✅
Circular imports	Careful dependency ordering	✅
Type hints for Python 3.11+	Use `	` union syntax, modern annotations
Missing edge cases	Comprehensive None/zero handling	✅
6. Implementation Plan: Approved ✓
The plan has been validated. Now proceeding to implementation.

7. Complete File Implementations
[F01] finanalyst_tools/__init__.py
Python

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
[F02] finanalyst_tools/config.py
Python

"""
Configuration and constants for FinAnalyst-Pro tools.

This module centralizes all configuration values, thresholds, and constants
used throughout the package. Modify these values to adjust behavior globally.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, ROUND_HALF_EVEN
from enum import Enum
from typing import Final


# ============================================================================
# PRECISION & ROUNDING CONFIGURATION
# ============================================================================

class RoundingMode(str, Enum):
    """Supported rounding modes for financial calculations."""
    STANDARD = "ROUND_HALF_UP"     # Default: 0.5 rounds up (most common)
    BANKERS = "ROUND_HALF_EVEN"    # Banker's rounding: 0.5 rounds to even
    
    def get_decimal_rounding(self):
        """Get the decimal module rounding constant."""
        if self == RoundingMode.STANDARD:
            return ROUND_HALF_UP
        return ROUND_HALF_EVEN


# Default decimal precision for different contexts
DECIMAL_PLACES: Final[dict[str, int]] = {
    "currency": 2,       # Monetary values: $1,234.56
    "percentage": 2,     # Percentages: 45.67%
    "ratio": 4,          # Ratios: 1.5234
    "shares": 0,         # Share counts: whole numbers
    "growth_rate": 4,    # Growth rates: 12.3456%
    "days": 0,           # Days (turnover): whole numbers
}

# Default rounding mode
DEFAULT_ROUNDING: Final[RoundingMode] = RoundingMode.STANDARD


# ============================================================================
# PLAUSIBILITY THRESHOLDS
# ============================================================================

class PlausibilityRanges:
    """
    Acceptable ranges for financial ratios and metrics.
    
    Values outside these ranges trigger warnings (not errors).
    Ranges are based on typical business metrics; specific industries
    may have different norms.
    
    Format: (minimum, maximum) as percentages for margins, absolute for ratios
    """
    
    # ─────────────────────────────────────────────────────────────────────
    # Profitability Metrics (percentages)
    # ─────────────────────────────────────────────────────────────────────
    GROSS_MARGIN: tuple[float, float] = (-50.0, 95.0)
    OPERATING_MARGIN: tuple[float, float] = (-100.0, 60.0)
    NET_MARGIN: tuple[float, float] = (-200.0, 50.0)
    EBITDA_MARGIN: tuple[float, float] = (-50.0, 70.0)
    ROA: tuple[float, float] = (-50.0, 40.0)
    ROE: tuple[float, float] = (-100.0, 60.0)
    ROCE: tuple[float, float] = (-50.0, 50.0)
    
    # ─────────────────────────────────────────────────────────────────────
    # Liquidity Ratios (absolute ratios)
    # ─────────────────────────────────────────────────────────────────────
    CURRENT_RATIO: tuple[float, float] = (0.1, 10.0)
    QUICK_RATIO: tuple[float, float] = (0.05, 8.0)
    CASH_RATIO: tuple[float, float] = (0.0, 5.0)
    
    # ─────────────────────────────────────────────────────────────────────
    # Solvency Ratios (absolute ratios)
    # ─────────────────────────────────────────────────────────────────────
    DEBT_TO_EQUITY: tuple[float, float] = (0.0, 10.0)
    DEBT_TO_ASSETS: tuple[float, float] = (0.0, 1.5)
    INTEREST_COVERAGE: tuple[float, float] = (-10.0, 100.0)
    EQUITY_RATIO: tuple[float, float] = (0.0, 1.0)
    
    # ─────────────────────────────────────────────────────────────────────
    # Efficiency Ratios (absolute ratios / days)
    # ─────────────────────────────────────────────────────────────────────
    ASSET_TURNOVER: tuple[float, float] = (0.1, 5.0)
    INVENTORY_TURNOVER: tuple[float, float] = (0.5, 50.0)
    RECEIVABLES_TURNOVER: tuple[float, float] = (1.0, 30.0)
    PAYABLES_TURNOVER: tuple[float, float] = (1.0, 30.0)
    DAYS_SALES_OUTSTANDING: tuple[float, float] = (5.0, 180.0)
    DAYS_INVENTORY_OUTSTANDING: tuple[float, float] = (5.0, 365.0)
    DAYS_PAYABLES_OUTSTANDING: tuple[float, float] = (5.0, 180.0)
    
    # ─────────────────────────────────────────────────────────────────────
    # Growth Metrics (percentages)
    # ─────────────────────────────────────────────────────────────────────
    REVENUE_GROWTH: tuple[float, float] = (-80.0, 500.0)
    NET_INCOME_GROWTH: tuple[float, float] = (-200.0, 1000.0)
    ASSET_GROWTH: tuple[float, float] = (-50.0, 200.0)
    
    @classmethod
    def get_range(cls, metric_name: str) -> tuple[float, float] | None:
        """
        Get plausibility range for a metric by name.
        
        Args:
            metric_name: Name of the metric (case-insensitive, underscores normalized)
            
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


# ============================================================================
# RECONCILIATION TOLERANCES
# ============================================================================

class ReconciliationTolerances:
    """
    Acceptable tolerance levels for cross-statement reconciliation.
    
    Expressed as percentage of the larger value being compared.
    Financial statements may have minor rounding differences.
    """
    
    # Strict: for values that should match exactly
    STRICT: Final[float] = 0.001  # 0.1% tolerance
    
    # Normal: for values that may have minor rounding differences
    NORMAL: Final[float] = 0.01  # 1% tolerance
    
    # Loose: for derived values that may have compounding differences
    LOOSE: Final[float] = 0.05  # 5% tolerance
    
    # Default tolerance for unspecified checks
    DEFAULT: Final[float] = NORMAL
    
    @classmethod
    def get_tolerance(cls, check_type: str) -> float:
        """Get tolerance for a specific check type."""
        tolerances = {
            "net_income": cls.STRICT,
            "cash_balance": cls.STRICT,
            "retained_earnings": cls.NORMAL,
            "total_assets": cls.STRICT,
            "working_capital": cls.NORMAL,
            "balance_sheet_equation": cls.STRICT,
        }
        return tolerances.get(check_type.lower(), cls.DEFAULT)


# ============================================================================
# CURRENCY CONFIGURATION
# ============================================================================

DEFAULT_CURRENCY: Final[str] = "SGD"

SUPPORTED_CURRENCIES: Final[set[str]] = {
    "SGD",  # Singapore Dollar (default)
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
}

# Currency symbols for formatting
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
}

# Currencies that don't use decimal places (or use different decimals)
ZERO_DECIMAL_CURRENCIES: Final[set[str]] = {"JPY", "IDR"}


# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

# Minimum periods required for trend analysis
MIN_PERIODS_FOR_TREND: Final[int] = 3

# Default forecast horizon (periods)
DEFAULT_FORECAST_PERIODS: Final[int] = 3

# Days in year for turnover calculations
DAYS_IN_YEAR: Final[int] = 365

# Days in quarter
DAYS_IN_QUARTER: Final[int] = 91


# ============================================================================
# SINGAPORE SMB CONTEXT
# ============================================================================

class SingaporeConstants:
    """Singapore-specific financial constants and thresholds."""
    
    # GST Rate (as of 2024)
    GST_RATE: Final[float] = 0.09  # 9%
    
    # SFRS for Small Entities thresholds (meet 2 of 3 to qualify)
    SFRS_SMALL_ENTITY_REVENUE: Final[int] = 10_000_000  # S$10M
    SFRS_SMALL_ENTITY_ASSETS: Final[int] = 10_000_000   # S$10M
    SFRS_SMALL_ENTITY_EMPLOYEES: Final[int] = 50
    
    # SME definition thresholds
    SME_ANNUAL_SALES: Final[int] = 100_000_000  # S$100M
    SME_EMPLOYEES: Final[int] = 200
    
    # Common financial year end months
    COMMON_FYE_MONTHS: Final[list[int]] = [12, 3, 6]  # Dec, Mar, Jun
    
    @classmethod
    def calculate_gst_exclusive(cls, gst_inclusive: float) -> float:
        """Convert GST-inclusive amount to GST-exclusive."""
        return gst_inclusive / (1 + cls.GST_RATE)
    
    @classmethod
    def calculate_gst_amount(cls, gst_exclusive: float) -> float:
        """Calculate GST on a GST-exclusive amount."""
        return gst_exclusive * cls.GST_RATE


# ============================================================================
# METRIC METADATA
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
}
[F03] finanalyst_tools/exceptions.py
Python

"""
Custom exception hierarchy for FinAnalyst-Pro tools.

This module defines a structured exception hierarchy that enables:
- Precise error categorization
- Detailed error context
- Actionable error messages for LLM consumption
- Consistent error handling across the package
"""

from __future__ import annotations

from typing import Any


class FinAnalystError(Exception):
    """
    Base exception for all FinAnalyst-Pro errors.
    
    All custom exceptions inherit from this class, enabling
    catch-all handling when needed.
    
    Attributes:
        message: Human-readable error description
        details: Additional context as key-value pairs
        error_code: Machine-readable error identifier
        suggestion: Actionable suggestion for resolution
    """
    
    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
        suggestion: str | None = None,
    ):
        self.message = message
        self.details = details or {}
        self.error_code = error_code or self._default_error_code()
        self.suggestion = suggestion
        super().__init__(self.message)
    
    def _default_error_code(self) -> str:
        """Generate default error code from class name."""
        # Convert CamelCase to SCREAMING_SNAKE_CASE
        name = self.__class__.__name__
        result = []
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                result.append("_")
            result.append(char.upper())
        return "".join(result).replace("_ERROR", "")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "suggestion": self.suggestion,
        }
    
    def __str__(self) -> str:
        """Format error message with details."""
        parts = [self.message]
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"Details: {details_str}")
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)


# ============================================================================
# CALCULATION ERRORS
# ============================================================================

class CalculationError(FinAnalystError):
    """
    Raised when a financial calculation cannot be completed.
    
    This covers general calculation failures that aren't more specifically
    categorized (e.g., not division by zero, not invalid input).
    """
    
    def __init__(
        self,
        message: str,
        metric_name: str | None = None,
        details: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ):
        details = details or {}
        if metric_name:
            details["metric_name"] = metric_name
        super().__init__(
            message=message,
            details=details,
            error_code="CALC_FAILED",
            suggestion=suggestion or "Check input values and try again",
        )
        self.metric_name = metric_name


class DivisionByZeroError(CalculationError):
    """
    Raised when division by zero is attempted in a calculation.
    
    This is a specific case of CalculationError that provides
    context about which values caused the issue.
    """
    
    def __init__(
        self,
        numerator: Any,
        denominator_name: str = "denominator",
        metric_name: str | None = None,
    ):
        message = f"Cannot divide {numerator} by zero ({denominator_name} is zero)"
        super().__init__(
            message=message,
            metric_name=metric_name,
            details={
                "numerator": numerator,
                "denominator_name": denominator_name,
            },
            suggestion=f"Ensure {denominator_name} is non-zero before calculation",
        )
        self.error_code = "DIVISION_BY_ZERO"


class InvalidInputError(CalculationError):
    """
    Raised when input values are invalid for the requested calculation.
    
    Examples:
    - Negative values where only positive are valid
    - Non-numeric values where numbers are required
    - Missing required inputs
    """
    
    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        actual_value: Any = None,
        expected: str | None = None,
    ):
        details = {}
        if field_name:
            details["field_name"] = field_name
        if actual_value is not None:
            details["actual_value"] = actual_value
        if expected:
            details["expected"] = expected
        
        suggestion = None
        if expected:
            suggestion = f"Provide a value that is: {expected}"
        
        super().__init__(
            message=message,
            details=details,
            suggestion=suggestion,
        )
        self.error_code = "INVALID_INPUT"
        self.field_name = field_name
        self.actual_value = actual_value
        self.expected = expected


# ============================================================================
# VALIDATION ERRORS
# ============================================================================

class ValidationError(FinAnalystError):
    """
    Raised when data validation fails.
    
    This covers schema validation, completeness checks, and
    data quality issues.
    """
    
    def __init__(
        self,
        message: str,
        field_errors: dict[str, str] | None = None,
        missing_fields: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ):
        details = details or {}
        if field_errors:
            details["field_errors"] = field_errors
        if missing_fields:
            details["missing_fields"] = missing_fields
        
        suggestion = None
        if missing_fields:
            suggestion = f"Provide the following fields: {', '.join(missing_fields)}"
        
        super().__init__(
            message=message,
            details=details,
            error_code="VALIDATION_FAILED",
            suggestion=suggestion,
        )
        self.field_errors = field_errors or {}
        self.missing_fields = missing_fields or []


class SchemaValidationError(ValidationError):
    """Raised when data doesn't conform to expected schema."""
    
    def __init__(
        self,
        message: str,
        schema_name: str | None = None,
        field_errors: dict[str, str] | None = None,
    ):
        super().__init__(
            message=message,
            field_errors=field_errors,
            details={"schema_name": schema_name} if schema_name else None,
        )
        self.error_code = "SCHEMA_INVALID"
        self.schema_name = schema_name


class DataCompletenessError(ValidationError):
    """Raised when required data is missing for an analysis."""
    
    def __init__(
        self,
        analysis_type: str,
        missing_fields: list[str],
    ):
        message = (
            f"Insufficient data for {analysis_type} analysis. "
            f"Missing: {', '.join(missing_fields)}"
        )
        super().__init__(
            message=message,
            missing_fields=missing_fields,
            details={"analysis_type": analysis_type},
        )
        self.error_code = "DATA_INCOMPLETE"
        self.analysis_type = analysis_type


# ============================================================================
# RECONCILIATION ERRORS
# ============================================================================

class ReconciliationError(FinAnalystError):
    """
    Raised when cross-statement reconciliation fails.
    
    This indicates an inconsistency between related values
    across different financial statements.
    """
    
    def __init__(
        self,
        message: str,
        check_name: str,
        expected_value: Any,
        actual_value: Any,
        difference: Any = None,
        tolerance: float | None = None,
    ):
        details = {
            "check_name": check_name,
            "expected_value": expected_value,
            "actual_value": actual_value,
        }
        if difference is not None:
            details["difference"] = difference
        if tolerance is not None:
            details["tolerance_used"] = tolerance
        
        super().__init__(
            message=message,
            details=details,
            error_code="RECONCILIATION_FAILED",
            suggestion="Verify data accuracy or confirm known discrepancy",
        )
        self.check_name = check_name
        self.expected_value = expected_value
        self.actual_value = actual_value


# ============================================================================
# DATA PARSING ERRORS
# ============================================================================

class DataParsingError(FinAnalystError):
    """
    Raised when financial data cannot be parsed.
    
    This covers issues with raw data extraction from
    PDFs, Excel files, or other sources.
    """
    
    def __init__(
        self,
        message: str,
        source_type: str | None = None,
        source_location: str | None = None,
        parse_error: str | None = None,
    ):
        details = {}
        if source_type:
            details["source_type"] = source_type
        if source_location:
            details["source_location"] = source_location
        if parse_error:
            details["parse_error"] = parse_error
        
        super().__init__(
            message=message,
            details=details,
            error_code="PARSE_FAILED",
            suggestion="Check source format and try again",
        )
        self.source_type = source_type


# ============================================================================
# TOOL EXECUTION ERRORS
# ============================================================================

class ToolExecutionError(FinAnalystError):
    """
    Raised when a tool execution fails.
    
    This wraps errors that occur during tool dispatch and execution,
    providing context about which tool failed and why.
    """
    
    def __init__(
        self,
        message: str,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        details = {"tool_name": tool_name}
        if parameters:
            details["parameters"] = parameters
        if original_error:
            details["original_error"] = str(original_error)
            details["original_error_type"] = type(original_error).__name__
        
        super().__init__(
            message=message,
            details=details,
            error_code="TOOL_EXECUTION_FAILED",
            suggestion="Check tool parameters and try again",
        )
        self.tool_name = tool_name
        self.parameters = parameters
        self.original_error = original_error


class UnknownToolError(ToolExecutionError):
    """Raised when an unknown tool is requested."""
    
    def __init__(
        self,
        tool_name: str,
        available_tools: list[str] | None = None,
    ):
        message = f"Unknown tool: '{tool_name}'"
        super().__init__(
            message=message,
            tool_name=tool_name,
        )
        self.error_code = "UNKNOWN_TOOL"
        if available_tools:
            self.details["available_tools"] = available_tools[:10]  # Limit list size
            self.suggestion = f"Use one of the available tools. Similar: {self._find_similar(tool_name, available_tools)}"
    
    def _find_similar(self, name: str, available: list[str]) -> list[str]:
        """Find tools with similar names."""
        name_lower = name.lower()
        similar = [t for t in available if name_lower in t.lower() or t.lower() in name_lower]
        return similar[:3] if similar else available[:3]


class ToolParameterError(ToolExecutionError):
    """Raised when tool parameters are invalid."""
    
    def __init__(
        self,
        tool_name: str,
        parameter_name: str,
        message: str,
        expected_type: str | None = None,
        actual_value: Any = None,
    ):
        details = {
            "parameter_name": parameter_name,
        }
        if expected_type:
            details["expected_type"] = expected_type
        if actual_value is not None:
            details["actual_value"] = str(actual_value)[:100]  # Truncate long values
        
        super().__init__(
            message=f"Parameter '{parameter_name}': {message}",
            tool_name=tool_name,
            parameters=details,
        )
        self.error_code = "INVALID_PARAMETER"
        self.parameter_name = parameter_name


# ============================================================================
# PLAUSIBILITY ERRORS
# ============================================================================

class PlausibilityError(FinAnalystError):
    """
    Raised when a calculated value fails plausibility checks.
    
    Note: This is typically a warning, not a blocking error.
    Only raised when explicitly requested to enforce plausibility.
    """
    
    def __init__(
        self,
        metric_name: str,
        value: float,
        plausible_range: tuple[float, float],
    ):
        message = (
            f"{metric_name} value of {value:.2f} is outside the plausible range "
            f"({plausible_range[0]:.2f} to {plausible_range[1]:.2f})"
        )
        super().__init__(
            message=message,
            details={
                "metric_name": metric_name,
                "value": value,
                "min_plausible": plausible_range[0],
                "max_plausible": plausible_range[1],
            },
            error_code="IMPLAUSIBLE_VALUE",
            suggestion="Verify input data accuracy. This may indicate a data entry error.",
        )
        self.metric_name = metric_name
        self.value = value
        self.plausible_range = plausible_range
[F04] finanalyst_tools/models/__init__.py
Python

"""
Data models for FinAnalyst-Pro tools.

This package provides Pydantic models for:
- Financial statement data (Income Statement, Balance Sheet, Cash Flow)
- Calculation results with audit trails
- Validation results and issues

All models support:
- Type validation
- JSON serialization
- Field aliases for common naming variations
"""

from __future__ import annotations

# Financial statement models
from finanalyst_tools.models.financial_statements import (
    StatementType,
    PeriodType,
    FinancialPeriod,
    IncomeStatementData,
    BalanceSheetData,
    CashFlowStatementData,
    FinancialStatementSet,
    MultiPeriodFinancialData,
)

# Analysis result models
from finanalyst_tools.models.analysis_results import (
    MetricCategory,
    TrendDirection,
    CalculationResult,
    MetricResult,
    TrendAnalysisResult,
    ComprehensiveAnalysisResult,
)

# Validation models
from finanalyst_tools.models.validation import (
    ValidationSeverity,
    ValidationIssue,
    ValidationResult,
    ReconciliationCheck,
    ReconciliationResult,
    PlausibilityCheck,
    PlausibilityResult,
)

__all__ = [
    # Statement types
    "StatementType",
    "PeriodType",
    "FinancialPeriod",
    # Financial statements
    "IncomeStatementData",
    "BalanceSheetData",
    "CashFlowStatementData",
    "FinancialStatementSet",
    "MultiPeriodFinancialData",
    # Result types
    "MetricCategory",
    "TrendDirection",
    "CalculationResult",
    "MetricResult",
    "TrendAnalysisResult",
    "ComprehensiveAnalysisResult",
    # Validation types
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationResult",
    "ReconciliationCheck",
    "ReconciliationResult",
    "PlausibilityCheck",
    "PlausibilityResult",
]
[F05] finanalyst_tools/models/financial_statements.py
Python

"""
Pydantic models for financial statement data structures.

These models provide:
- Strong type validation for financial data
- Field aliases for common naming variations
- Calculated properties for derived values
- JSON serialization for LLM communication
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


# Type aliases for documentation
MonetaryValue = Annotated[
    Decimal, 
    Field(description="Monetary value in base currency")
]


class FinancialPeriod(BaseModel):
    """
    Represents a financial reporting period.
    
    Examples:
        - Annual: FinancialPeriod(year=2023, period_type=PeriodType.ANNUAL)
        - Quarterly: FinancialPeriod(year=2023, period_type=PeriodType.QUARTERLY, quarter=2)
    """
    
    year: int = Field(..., ge=1900, le=2100, description="Fiscal year")
    period_type: PeriodType = Field(default=PeriodType.ANNUAL)
    quarter: int | None = Field(
        default=None, 
        ge=1, 
        le=4, 
        description="Quarter number (1-4) if quarterly"
    )
    month: int | None = Field(
        default=None,
        ge=1,
        le=12,
        description="Month number (1-12) if monthly"
    )
    start_date: date | None = Field(default=None, description="Period start date")
    end_date: date | None = Field(default=None, description="Period end date")
    
    model_config = {"extra": "forbid"}
    
    @model_validator(mode="after")
    def validate_period_details(self) -> "FinancialPeriod":
        """Validate period-specific fields."""
        if self.period_type == PeriodType.QUARTERLY and self.quarter is None:
            raise ValueError("Quarter must be specified for quarterly periods")
        if self.period_type == PeriodType.MONTHLY and self.month is None:
            raise ValueError("Month must be specified for monthly periods")
        return self
    
    def __str__(self) -> str:
        """Human-readable period representation."""
        if self.period_type == PeriodType.QUARTERLY:
            return f"Q{self.quarter} {self.year}"
        if self.period_type == PeriodType.MONTHLY:
            return f"{self.year}-{self.month:02d}"
        if self.period_type == PeriodType.TTM:
            return f"TTM {self.year}"
        return str(self.year)
    
    def __lt__(self, other: "FinancialPeriod") -> bool:
        """Enable sorting by period."""
        if self.year != other.year:
            return self.year < other.year
        # Same year - compare by quarter/month
        self_sub = self.quarter or self.month or 0
        other_sub = other.quarter or other.month or 0
        return self_sub < other_sub


class IncomeStatementData(BaseModel):
    """
    Income Statement / Profit & Loss data model.
    
    Supports multiple naming conventions through field aliases.
    Provides calculated properties for derived values.
    """
    
    period: FinancialPeriod
    currency: str = Field(default="SGD", min_length=3, max_length=3)
    
    # ─────────────────────────────────────────────────────────────────────
    # Revenue
    # ─────────────────────────────────────────────────────────────────────
    total_revenue: MonetaryValue = Field(
        ...,
        description="Total revenue / net sales"
    )
    cost_of_goods_sold: MonetaryValue = Field(
        ...,
        description="Cost of goods sold / cost of sales"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Operating Expenses (flexible structure)
    # ─────────────────────────────────────────────────────────────────────
    operating_expenses: MonetaryValue | None = Field(
        default=None,
        description="Total operating expenses (if provided as aggregate)"
    )
    selling_general_admin: MonetaryValue | None = Field(
        default=None,
        description="Selling, General & Administrative expenses"
    )
    marketing_expenses: MonetaryValue | None = Field(
        default=None,
        description="Marketing and advertising expenses"
    )
    research_development: MonetaryValue | None = Field(
        default=None,
        description="Research & Development expenses"
    )
    depreciation_amortization: MonetaryValue | None = Field(
        default=None,
        description="Depreciation and amortization"
    )
    other_operating_expenses: MonetaryValue | None = Field(
        default=None,
        description="Other operating expenses"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Non-Operating Items
    # ─────────────────────────────────────────────────────────────────────
    interest_income: MonetaryValue = Field(
        default=Decimal("0"),
        description="Interest and investment income"
    )
    interest_expense: MonetaryValue = Field(
        default=Decimal("0"),
        description="Interest expense on debt"
    )
    other_income: MonetaryValue = Field(
        default=Decimal("0"),
        description="Other non-operating income"
    )
    other_expenses: MonetaryValue = Field(
        default=Decimal("0"),
        description="Other non-operating expenses"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Taxes and Bottom Line
    # ─────────────────────────────────────────────────────────────────────
    income_tax_expense: MonetaryValue = Field(
        default=Decimal("0"),
        description="Income tax expense"
    )
    net_income: MonetaryValue | None = Field(
        default=None,
        description="Net income / net profit"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Per Share Data (optional)
    # ─────────────────────────────────────────────────────────────────────
    earnings_per_share: Decimal | None = Field(
        default=None,
        description="Basic earnings per share"
    )
    diluted_eps: Decimal | None = Field(
        default=None,
        description="Diluted earnings per share"
    )
    shares_outstanding: int | None = Field(
        default=None,
        description="Weighted average shares outstanding"
    )
    
    model_config = {
        "populate_by_name": True,
        "extra": "allow",  # Allow additional fields
    }
    
    @field_validator("total_revenue", "cost_of_goods_sold", mode="before")
    @classmethod
    def convert_to_decimal(cls, v: Any) -> Decimal:
        """Convert numeric inputs to Decimal."""
        if v is None:
            raise ValueError("This field is required")
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))
    
    @property
    def gross_profit(self) -> Decimal:
        """Calculate gross profit."""
        return self.total_revenue - self.cost_of_goods_sold
    
    @property
    def total_operating_expenses(self) -> Decimal:
        """
        Calculate total operating expenses.
        
        Uses provided aggregate if available, otherwise sums components.
        """
        if self.operating_expenses is not None:
            return self.operating_expenses
        
        total = Decimal("0")
        expense_fields = [
            self.selling_general_admin,
            self.marketing_expenses,
            self.research_development,
            self.depreciation_amortization,
            self.other_operating_expenses,
        ]
        for expense in expense_fields:
            if expense is not None:
                total += expense
        return total
    
    @property
    def operating_income(self) -> Decimal:
        """Calculate operating income (EBIT approximation)."""
        return self.gross_profit - self.total_operating_expenses
    
    @property
    def ebitda(self) -> Decimal:
        """Calculate EBITDA (operating income + D&A)."""
        da = self.depreciation_amortization or Decimal("0")
        return self.operating_income + da
    
    @property
    def earnings_before_tax(self) -> Decimal:
        """Calculate earnings before tax."""
        return (
            self.operating_income
            + self.interest_income
            - self.interest_expense
            + self.other_income
            - self.other_expenses
        )
    
    @property
    def calculated_net_income(self) -> Decimal:
        """
        Calculate net income from components.
        
        Returns provided net_income if available, otherwise calculates.
        """
        if self.net_income is not None:
            return self.net_income
        return self.earnings_before_tax - self.income_tax_expense
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with calculated fields included."""
        data = self.model_dump()
        data["gross_profit"] = float(self.gross_profit)
        data["operating_income"] = float(self.operating_income)
        data["ebitda"] = float(self.ebitda)
        data["calculated_net_income"] = float(self.calculated_net_income)
        return data


class BalanceSheetData(BaseModel):
    """
    Balance Sheet data model.
    
    Organized into Current Assets, Non-Current Assets, Current Liabilities,
    Non-Current Liabilities, and Shareholders' Equity sections.
    """
    
    period: FinancialPeriod
    currency: str = Field(default="SGD", min_length=3, max_length=3)
    
    # ─────────────────────────────────────────────────────────────────────
    # Current Assets
    # ─────────────────────────────────────────────────────────────────────
    cash_and_equivalents: MonetaryValue = Field(
        ...,
        description="Cash and cash equivalents"
    )
    short_term_investments: MonetaryValue = Field(
        default=Decimal("0"),
        description="Short-term investments"
    )
    accounts_receivable: MonetaryValue = Field(
        default=Decimal("0"),
        description="Accounts receivable, net"
    )
    inventory: MonetaryValue = Field(
        default=Decimal("0"),
        description="Inventories"
    )
    prepaid_expenses: MonetaryValue = Field(
        default=Decimal("0"),
        description="Prepaid expenses and other current assets"
    )
    other_current_assets: MonetaryValue = Field(
        default=Decimal("0"),
        description="Other current assets"
    )
    total_current_assets: MonetaryValue | None = Field(
        default=None,
        description="Total current assets (if provided)"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Non-Current Assets
    # ─────────────────────────────────────────────────────────────────────
    property_plant_equipment: MonetaryValue = Field(
        default=Decimal("0"),
        description="Property, plant and equipment, net"
    )
    intangible_assets: MonetaryValue = Field(
        default=Decimal("0"),
        description="Intangible assets (including goodwill)"
    )
    goodwill: MonetaryValue = Field(
        default=Decimal("0"),
        description="Goodwill (if separate from intangibles)"
    )
    long_term_investments: MonetaryValue = Field(
        default=Decimal("0"),
        description="Long-term investments"
    )
    deferred_tax_assets: MonetaryValue = Field(
        default=Decimal("0"),
        description="Deferred tax assets"
    )
    other_non_current_assets: MonetaryValue = Field(
        default=Decimal("0"),
        description="Other non-current assets"
    )
    total_non_current_assets: MonetaryValue | None = Field(
        default=None,
        description="Total non-current assets"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Total Assets
    # ─────────────────────────────────────────────────────────────────────
    total_assets: MonetaryValue | None = Field(
        default=None,
        description="Total assets"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Current Liabilities
    # ─────────────────────────────────────────────────────────────────────
    accounts_payable: MonetaryValue = Field(
        default=Decimal("0"),
        description="Accounts payable"
    )
    short_term_debt: MonetaryValue = Field(
        default=Decimal("0"),
        description="Short-term debt / current portion of long-term debt"
    )
    accrued_liabilities: MonetaryValue = Field(
        default=Decimal("0"),
        description="Accrued liabilities"
    )
    deferred_revenue: MonetaryValue = Field(
        default=Decimal("0"),
        description="Deferred revenue / unearned revenue"
    )
    income_taxes_payable: MonetaryValue = Field(
        default=Decimal("0"),
        description="Income taxes payable"
    )
    other_current_liabilities: MonetaryValue = Field(
        default=Decimal("0"),
        description="Other current liabilities"
    )
    total_current_liabilities: MonetaryValue | None = Field(
        default=None,
        description="Total current liabilities"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Non-Current Liabilities
    # ─────────────────────────────────────────────────────────────────────
    long_term_debt: MonetaryValue = Field(
        default=Decimal("0"),
        description="Long-term debt"
    )
    deferred_tax_liabilities: MonetaryValue = Field(
        default=Decimal("0"),
        description="Deferred tax liabilities"
    )
    pension_liabilities: MonetaryValue = Field(
        default=Decimal("0"),
        description="Pension and post-retirement obligations"
    )
    other_non_current_liabilities: MonetaryValue = Field(
        default=Decimal("0"),
        description="Other non-current liabilities"
    )
    total_non_current_liabilities: MonetaryValue | None = Field(
        default=None,
        description="Total non-current liabilities"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Total Liabilities
    # ─────────────────────────────────────────────────────────────────────
    total_liabilities: MonetaryValue | None = Field(
        default=None,
        description="Total liabilities"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Shareholders' Equity
    # ─────────────────────────────────────────────────────────────────────
    common_stock: MonetaryValue = Field(
        default=Decimal("0"),
        description="Common stock / share capital"
    )
    preferred_stock: MonetaryValue = Field(
        default=Decimal("0"),
        description="Preferred stock"
    )
    additional_paid_in_capital: MonetaryValue = Field(
        default=Decimal("0"),
        description="Additional paid-in capital"
    )
    retained_earnings: MonetaryValue = Field(
        default=Decimal("0"),
        description="Retained earnings"
    )
    treasury_stock: MonetaryValue = Field(
        default=Decimal("0"),
        description="Treasury stock (contra-equity)"
    )
    accumulated_other_comprehensive_income: MonetaryValue = Field(
        default=Decimal("0"),
        description="Accumulated other comprehensive income/loss"
    )
    total_shareholders_equity: MonetaryValue | None = Field(
        default=None,
        description="Total shareholders' equity"
    )
    
    # Non-controlling interest (for consolidated statements)
    non_controlling_interest: MonetaryValue = Field(
        default=Decimal("0"),
        description="Non-controlling / minority interest"
    )
    
    model_config = {
        "populate_by_name": True,
        "extra": "allow",
    }
    
    @field_validator("cash_and_equivalents", mode="before")
    @classmethod
    def convert_to_decimal(cls, v: Any) -> Decimal:
        """Convert numeric inputs to Decimal."""
        if v is None:
            raise ValueError("Cash and equivalents is required")
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))
    
    @property
    def calculated_current_assets(self) -> Decimal:
        """Calculate total current assets from components."""
        if self.total_current_assets is not None:
            return self.total_current_assets
        return (
            self.cash_and_equivalents
            + self.short_term_investments
            + self.accounts_receivable
            + self.inventory
            + self.prepaid_expenses
            + self.other_current_assets
        )
    
    @property
    def calculated_non_current_assets(self) -> Decimal:
        """Calculate total non-current assets from components."""
        if self.total_non_current_assets is not None:
            return self.total_non_current_assets
        return (
            self.property_plant_equipment
            + self.intangible_assets
            + self.goodwill
            + self.long_term_investments
            + self.deferred_tax_assets
            + self.other_non_current_assets
        )
    
    @property
    def calculated_total_assets(self) -> Decimal:
        """Calculate total assets from components."""
        if self.total_assets is not None:
            return self.total_assets
        return self.calculated_current_assets + self.calculated_non_current_assets
    
    @property
    def calculated_current_liabilities(self) -> Decimal:
        """Calculate total current liabilities from components."""
        if self.total_current_liabilities is not None:
            return self.total_current_liabilities
        return (
            self.accounts_payable
            + self.short_term_debt
            + self.accrued_liabilities
            + self.deferred_revenue
            + self.income_taxes_payable
            + self.other_current_liabilities
        )
    
    @property
    def calculated_non_current_liabilities(self) -> Decimal:
        """Calculate total non-current liabilities from components."""
        if self.total_non_current_liabilities is not None:
            return self.total_non_current_liabilities
        return (
            self.long_term_debt
            + self.deferred_tax_liabilities
            + self.pension_liabilities
            + self.other_non_current_liabilities
        )
    
    @property
    def calculated_total_liabilities(self) -> Decimal:
        """Calculate total liabilities from components."""
        if self.total_liabilities is not None:
            return self.total_liabilities
        return self.calculated_current_liabilities + self.calculated_non_current_liabilities
    
    @property
    def calculated_shareholders_equity(self) -> Decimal:
        """Calculate shareholders' equity from components."""
        if self.total_shareholders_equity is not None:
            return self.total_shareholders_equity
        return (
            self.common_stock
            + self.preferred_stock
            + self.additional_paid_in_capital
            + self.retained_earnings
            - self.treasury_stock
            + self.accumulated_other_comprehensive_income
        )
    
    @property
    def calculated_total_equity(self) -> Decimal:
        """Calculate total equity including non-controlling interest."""
        return self.calculated_shareholders_equity + self.non_controlling_interest
    
    @property
    def working_capital(self) -> Decimal:
        """Calculate working capital."""
        return self.calculated_current_assets - self.calculated_current_liabilities
    
    @property
    def total_debt(self) -> Decimal:
        """Calculate total debt (short-term + long-term)."""
        return self.short_term_debt + self.long_term_debt
    
    def check_balance_sheet_equation(self, tolerance: Decimal = Decimal("0.01")) -> bool:
        """
        Verify Assets = Liabilities + Equity.
        
        Args:
            tolerance: Acceptable difference (default 0.01 = 1 cent)
            
        Returns:
            True if equation balances within tolerance
        """
        assets = self.calculated_total_assets
        liab_equity = self.calculated_total_liabilities + self.calculated_total_equity
        difference = abs(assets - liab_equity)
        return difference <= tolerance


class CashFlowStatementData(BaseModel):
    """
    Cash Flow Statement data model.
    
    Organized into Operating, Investing, and Financing activities.
    """
    
    period: FinancialPeriod
    currency: str = Field(default="SGD", min_length=3, max_length=3)
    
    # ─────────────────────────────────────────────────────────────────────
    # Operating Activities
    # ─────────────────────────────────────────────────────────────────────
    net_income: MonetaryValue = Field(
        ...,
        description="Net income (starting point for indirect method)"
    )
    depreciation_amortization: MonetaryValue = Field(
        default=Decimal("0"),
        description="Depreciation and amortization"
    )
    stock_based_compensation: MonetaryValue = Field(
        default=Decimal("0"),
        description="Stock-based compensation expense"
    )
    deferred_taxes: MonetaryValue = Field(
        default=Decimal("0"),
        description="Deferred income taxes"
    )
    
    # Working capital changes
    change_in_receivables: MonetaryValue = Field(
        default=Decimal("0"),
        description="Change in accounts receivable"
    )
    change_in_inventory: MonetaryValue = Field(
        default=Decimal("0"),
        description="Change in inventories"
    )
    change_in_payables: MonetaryValue = Field(
        default=Decimal("0"),
        description="Change in accounts payable"
    )
    change_in_other_working_capital: MonetaryValue = Field(
        default=Decimal("0"),
        description="Changes in other working capital items"
    )
    other_operating_activities: MonetaryValue = Field(
        default=Decimal("0"),
        description="Other operating activities"
    )
    net_cash_from_operating: MonetaryValue | None = Field(
        default=None,
        description="Net cash from operating activities"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Investing Activities
    # ─────────────────────────────────────────────────────────────────────
    capital_expenditures: MonetaryValue = Field(
        default=Decimal("0"),
        description="Capital expenditures (purchases of PP&E)"
    )
    acquisitions: MonetaryValue = Field(
        default=Decimal("0"),
        description="Acquisitions, net of cash acquired"
    )
    investment_purchases: MonetaryValue = Field(
        default=Decimal("0"),
        description="Purchases of investments"
    )
    investment_sales: MonetaryValue = Field(
        default=Decimal("0"),
        description="Sales/maturities of investments"
    )
    other_investing_activities: MonetaryValue = Field(
        default=Decimal("0"),
        description="Other investing activities"
    )
    net_cash_from_investing: MonetaryValue | None = Field(
        default=None,
        description="Net cash from investing activities"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Financing Activities
    # ─────────────────────────────────────────────────────────────────────
    debt_issued: MonetaryValue = Field(
        default=Decimal("0"),
        description="Proceeds from debt issuance"
    

