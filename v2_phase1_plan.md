# Phase 1: Complete Core Infrastructure — Detailed Implementation Plan

## Executive Summary

I have conducted a deep, systematic analysis of Phase 1 requirements. This plan covers **18 files** organized by dependency order, with detailed feature descriptions and validation checklists for each. After plan validation, I will deliver complete, production-ready, drop-in replacement code for every file.

---

## 1. Phase 1 Scope Definition

### 1.1 Objectives
| Objective | Description | Success Criteria |
|-----------|-------------|------------------|
| **Foundation** | Establish package structure and core utilities | All imports work, no circular dependencies |
| **Type Safety** | Implement Pydantic models for all data structures | All models validate correctly |
| **Precision** | Decimal-based math for financial calculations | Zero floating-point errors |
| **Calculations** | Profitability + Liquidity metrics operational | All 9 metrics calculate correctly |
| **Integration** | Tool registry + dispatcher fully functional | LLM can call any registered tool |

### 1.2 File Inventory

```
finanalyst_tools/                          # 18 files total
│
├── __init__.py                            # [NEW] Package exports
├── config.py                              # [ENHANCE] Add missing constants
├── exceptions.py                          # [NEW] Custom exception hierarchy
│
├── models/
│   ├── __init__.py                        # [NEW] Model exports
│   ├── financial_statements.py            # [REFACTOR] Minor improvements
│   ├── analysis_results.py                # [NEW] Extract from profitability.py
│   └── validation.py                      # [NEW] Extract from schema_validator.py
│
├── utils/
│   ├── __init__.py                        # [NEW] Utility exports
│   ├── math_ops.py                        # [ENHANCE] Add missing functions
│   ├── formatting.py                      # [NEW] Number/currency formatting
│   └── currency.py                        # [NEW] Currency handling + SGD
│
├── validation/
│   ├── __init__.py                        # [NEW] Validation exports
│   └── schema_validator.py                # [REFACTOR] Use models/validation.py
│
├── calculations/
│   ├── __init__.py                        # [NEW] Calculator exports
│   ├── profitability.py                   # [REFACTOR] Use models/analysis_results.py
│   └── liquidity.py                       # [REFACTOR] Use models/analysis_results.py
│
├── tool_registry.py                       # [ENHANCE] Complete all Phase 1 tools
└── dispatcher.py                          # [COMPLETE] Was truncated
```

---

## 2. Dependency Graph

```
                                    ┌─────────────────┐
                                    │    config.py    │
                                    │   (constants)   │
                                    └────────┬────────┘
                                             │
                         ┌───────────────────┼───────────────────┐
                         │                   │                   │
                         ▼                   ▼                   ▼
                ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
                │  exceptions.py  │ │ utils/math_ops  │ │ utils/currency  │
                │                 │ │                 │ │                 │
                └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
                         │                   │                   │
                         │                   ▼                   │
                         │          ┌─────────────────┐          │
                         │          │utils/formatting │          │
                         │          └────────┬────────┘          │
                         │                   │                   │
                         └───────────────────┼───────────────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │     models/     │
                                    │ (all 3 files)   │
                                    └────────┬────────┘
                                             │
                         ┌───────────────────┴───────────────────┐
                         │                                       │
                         ▼                                       ▼
                ┌─────────────────┐                     ┌─────────────────┐
                │   validation/   │                     │  calculations/  │
                │schema_validator │                     │ profitability   │
                └────────┬────────┘                     │   liquidity     │
                         │                              └────────┬────────┘
                         │                                       │
                         └───────────────────┬───────────────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │ tool_registry   │
                                    └────────┬────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │   dispatcher    │
                                    └────────┬────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │   __init__.py   │
                                    │  (main export)  │
                                    └─────────────────┘
```

---

## 3. Detailed File Specifications

### 3.1 `finanalyst_tools/config.py`

**Purpose**: Central configuration constants, thresholds, and settings.

**Features**:
| Feature | Description |
|---------|-------------|
| Decimal precision settings | Configurable precision for currency, percentages, ratios |
| Rounding modes | ROUND_HALF_UP (standard), ROUND_HALF_EVEN (banker's) |
| Plausibility ranges | Min/max thresholds for all financial metrics |
| Reconciliation tolerances | Acceptable variance for cross-statement checks |
| Currency defaults | SGD as default, list of supported currencies |
| Analysis settings | Min periods for trends, forecast horizons |

**Checklist**:
- [ ] `RoundingMode` enum with STANDARD and BANKERS modes
- [ ] `DECIMAL_PLACES` dict with precision for: currency, percentage, ratio, shares, growth_rate
- [ ] `PlausibilityRanges` class with ranges for all 15+ metrics
- [ ] `ReconciliationTolerances` class with STRICT, NORMAL, LOOSE thresholds
- [ ] `DEFAULT_CURRENCY` = "SGD"
- [ ] `SUPPORTED_CURRENCIES` set with 10+ currencies
- [ ] `MIN_PERIODS_FOR_TREND` = 3
- [ ] `DEFAULT_FORECAST_PERIODS` = 3
- [ ] `DAYS_IN_YEAR` = 365
- [ ] Type hints on all constants using `Final`
- [ ] Docstrings explaining each configuration section

---

### 3.2 `finanalyst_tools/exceptions.py`

**Purpose**: Custom exception hierarchy for precise error handling.

**Features**:
| Feature | Description |
|---------|-------------|
| Base exception | `FinAnalystError` as parent for all custom exceptions |
| Calculation errors | `CalculationError`, `DivisionByZeroError`, `InvalidInputError` |
| Validation errors | `ValidationError`, `SchemaError`, `ReconciliationError` |
| Data errors | `DataParsingError`, `MissingDataError`, `PlausibilityError` |
| Tool errors | `ToolNotFoundError`, `ToolExecutionError` |

**Checklist**:
- [ ] `FinAnalystError(Exception)` - base class with message and optional details dict
- [ ] `CalculationError(FinAnalystError)` - calculation failures
- [ ] `DivisionByZeroError(CalculationError)` - division by zero
- [ ] `InvalidInputError(CalculationError)` - invalid calculation inputs
- [ ] `ValidationError(FinAnalystError)` - validation failures
- [ ] `SchemaError(ValidationError)` - schema validation failures
- [ ] `ReconciliationError(ValidationError)` - reconciliation failures
- [ ] `DataError(FinAnalystError)` - data-related errors
- [ ] `MissingDataError(DataError)` - required data missing
- [ ] `PlausibilityError(DataError)` - implausible values detected
- [ ] `ToolError(FinAnalystError)` - tool execution errors
- [ ] `ToolNotFoundError(ToolError)` - unknown tool requested
- [ ] `ToolExecutionError(ToolError)` - tool execution failed
- [ ] All exceptions have `__init__` accepting message and optional kwargs
- [ ] All exceptions have `to_dict()` method for JSON serialization

---

### 3.3 `finanalyst_tools/models/__init__.py`

**Purpose**: Export all model classes for easy importing.

**Checklist**:
- [ ] Import and export all classes from `financial_statements.py`
- [ ] Import and export all classes from `analysis_results.py`
- [ ] Import and export all classes from `validation.py`
- [ ] `__all__` list defined with all public classes

---

### 3.4 `finanalyst_tools/models/financial_statements.py`

**Purpose**: Pydantic models for financial statement data structures.

**Features**:
| Feature | Description |
|---------|-------------|
| Statement enums | `StatementType`, `PeriodType` |
| Period model | `FinancialPeriod` with year, quarter, dates |
| Income Statement | `IncomeStatementData` with all line items |
| Balance Sheet | `BalanceSheetData` with assets, liabilities, equity |
| Cash Flow | `CashFlowStatementData` with all sections |
| Statement Set | `FinancialStatementSet` combining all three |
| Multi-period | `MultiPeriodFinancialData` for trend analysis |

**Checklist**:
- [ ] `StatementType` enum: INCOME_STATEMENT, BALANCE_SHEET, CASH_FLOW
- [ ] `PeriodType` enum: ANNUAL, QUARTERLY, MONTHLY, TTM
- [ ] `FinancialPeriod` model with validation
- [ ] `IncomeStatementData` with:
  - [ ] Revenue, COGS fields with aliases
  - [ ] Operating expense breakdown (OpEx, Marketing, R&D, G&A, D&A)
  - [ ] Interest income/expense, other income/expense
  - [ ] Tax expense, net income
  - [ ] EPS fields (basic, diluted)
  - [ ] Computed properties: `gross_profit`, `operating_income`, `calculated_net_income`
- [ ] `BalanceSheetData` with:
  - [ ] Current assets breakdown (cash, AR, inventory, prepaid, other)
  - [ ] Non-current assets (PPE, intangibles, investments, other)
  - [ ] Current liabilities breakdown (AP, short-term debt, accrued, deferred rev)
  - [ ] Non-current liabilities (long-term debt, deferred tax, other)
  - [ ] Equity breakdown (common stock, retained earnings, APIC, treasury)
  - [ ] Computed properties for all subtotals
- [ ] `CashFlowStatementData` with:
  - [ ] Operating activities section
  - [ ] Investing activities section
  - [ ] Financing activities section
  - [ ] Net change and beginning/ending cash
  - [ ] Computed property: `calculated_operating_cash_flow`
- [ ] `FinancialStatementSet` with period matching validation
- [ ] `MultiPeriodFinancialData` with chronological sorting
- [ ] All models have `model_config = {"populate_by_name": True}`
- [ ] Type aliases: `MonetaryValue`, `Percentage`, `Ratio`

---

### 3.5 `finanalyst_tools/models/analysis_results.py`

**Purpose**: Structured result models for calculations and analysis.

**Features**:
| Feature | Description |
|---------|-------------|
| Calculation result | Complete audit trail for single metric |
| Metric collection | Group of related metrics |
| Trend result | Multi-period trend analysis output |
| Confidence level | HIGH/MEDIUM/LOW classification |

**Checklist**:
- [ ] `MetricUnit` enum: PERCENTAGE, RATIO, CURRENCY, DAYS, COUNT
- [ ] `TrendDirection` enum: INCREASING, DECREASING, STABLE, VOLATILE
- [ ] `ConfidenceLevel` enum: HIGH, MEDIUM, LOW
- [ ] `CalculationResult` dataclass with:
  - [ ] `metric_name: str`
  - [ ] `value: Decimal | None`
  - [ ] `unit: MetricUnit`
  - [ ] `formula: str`
  - [ ] `inputs: dict[str, Any]`
  - [ ] `calculation_steps: list[str]`
  - [ ] `is_plausible: bool`
  - [ ] `plausibility_range: tuple[float, float] | None`
  - [ ] `warnings: list[str]`
  - [ ] `to_dict()` method
  - [ ] `to_json()` method
- [ ] `MetricCollection` dataclass with:
  - [ ] `category: str`
  - [ ] `period: FinancialPeriod`
  - [ ] `metrics: list[CalculationResult]`
  - [ ] `summary: dict[str, Decimal]` (quick access to values)
  - [ ] `to_dict()` method
- [ ] `TrendResult` dataclass with:
  - [ ] `metric_name: str`
  - [ ] `periods: list[str]`
  - [ ] `values: list[Decimal]`
  - [ ] `direction: TrendDirection`
  - [ ] `growth_rate: Decimal | None` (CAGR)
  - [ ] `volatility: Decimal | None`
  - [ ] `interpretation: str`
- [ ] `ConfidenceAssessment` dataclass with:
  - [ ] `level: ConfidenceLevel`
  - [ ] `justification: str`
  - [ ] `factors: dict[str, str]`

---

### 3.6 `finanalyst_tools/models/validation.py`

**Purpose**: Models for validation results and issues.

**Features**:
| Feature | Description |
|---------|-------------|
| Severity levels | ERROR, WARNING, INFO classification |
| Validation issue | Single issue with field, message, severity |
| Validation result | Complete validation outcome |
| Reconciliation result | Cross-statement check outcome |

**Checklist**:
- [ ] `ValidationSeverity` enum: ERROR, WARNING, INFO
- [ ] `ValidationIssue` dataclass with:
  - [ ] `field: str`
  - [ ] `message: str`
  - [ ] `severity: ValidationSeverity`
  - [ ] `actual_value: Any = None`
  - [ ] `expected: str | None = None`
  - [ ] `to_dict()` method
- [ ] `ValidationResult` dataclass with:
  - [ ] `is_valid: bool`
  - [ ] `issues: list[ValidationIssue]` (errors)
  - [ ] `warnings: list[ValidationIssue]`
  - [ ] `info: list[ValidationIssue]`
  - [ ] `@property error_count`
  - [ ] `@property warning_count`
  - [ ] `@property can_proceed` (no errors)
  - [ ] `add_issue()` method
  - [ ] `merge()` method (combine two results)
  - [ ] `to_dict()` method
- [ ] `ReconciliationCheck` dataclass with:
  - [ ] `check_name: str`
  - [ ] `statement_a: str`
  - [ ] `value_a: Decimal`
  - [ ] `statement_b: str`
  - [ ] `value_b: Decimal`
  - [ ] `difference: Decimal`
  - [ ] `tolerance: float`
  - [ ] `passed: bool`
  - [ ] `message: str`
- [ ] `ReconciliationResult` dataclass with:
  - [ ] `checks: list[ReconciliationCheck]`
  - [ ] `all_passed: bool`
  - [ ] `to_dict()` method

---

### 3.7 `finanalyst_tools/utils/__init__.py`

**Purpose**: Export all utility functions.

**Checklist**:
- [ ] Import and export all from `math_ops.py`
- [ ] Import and export all from `formatting.py`
- [ ] Import and export all from `currency.py`
- [ ] `__all__` list defined

---

### 3.8 `finanalyst_tools/utils/math_ops.py`

**Purpose**: Safe mathematical operations with Decimal precision.

**Features**:
| Feature | Description |
|---------|-------------|
| Type conversion | Safe conversion to Decimal |
| Safe division | Division with zero handling |
| Rounding | Configurable precision rounding |
| Percentages | Percentage calculation |
| Growth rates | Period-over-period and CAGR |
| Averages | Simple and weighted averages |
| Statistics | Variance, standard deviation |

**Checklist**:
- [ ] `to_decimal(value, default)` - safe conversion
- [ ] `safe_divide(numerator, denominator, default, precision, raise_on_zero)` - safe division
- [ ] `round_decimal(value, precision, rounding)` - configurable rounding
- [ ] `calculate_percentage(part, whole, precision)` - percentage calculation
- [ ] `calculate_growth_rate(current, previous, precision)` - YoY growth
- [ ] `calculate_cagr(beginning, ending, periods, precision)` - compound growth
- [ ] `calculate_average(*values, precision)` - arithmetic mean
- [ ] `calculate_weighted_average(values, weights, precision)` - weighted mean
- [ ] `calculate_variance(values)` - population variance
- [ ] `calculate_std_dev(values)` - standard deviation
- [ ] `calculate_min_max(values)` - returns (min, max) tuple
- [ ] All functions have comprehensive docstrings
- [ ] All functions handle edge cases (empty lists, None values)
- [ ] All functions use Decimal internally

---

### 3.9 `finanalyst_tools/utils/formatting.py`

**Purpose**: Consistent number and output formatting.

**Features**:
| Feature | Description |
|---------|-------------|
| Number formatting | Thousands separators, decimal places |
| Percentage formatting | With % symbol, configurable precision |
| Ratio formatting | Appropriate decimal places |
| Currency formatting | With symbol and code |
| Table formatting | Markdown table generation |

**Checklist**:
- [ ] `format_number(value, precision, use_thousands_sep)` - general formatting
- [ ] `format_currency(value, currency_code, precision, show_symbol)` - currency display
- [ ] `format_percentage(value, precision, show_symbol)` - percentage display
- [ ] `format_ratio(value, precision)` - ratio display
- [ ] `format_change(value, precision, show_sign)` - change with +/- sign
- [ ] `format_large_number(value, precision)` - K/M/B suffixes
- [ ] `get_currency_symbol(currency_code)` - symbol lookup
- [ ] `format_trend_indicator(direction)` - ↑ ↓ → symbols
- [ ] `format_status_indicator(is_good, threshold)` - ✅ ⚠️ ❌
- [ ] `format_markdown_table(headers, rows)` - table generation
- [ ] All functions handle None gracefully (return "N/A" or similar)

---

### 3.10 `finanalyst_tools/utils/currency.py`

**Purpose**: Currency handling with Singapore focus.

**Features**:
| Feature | Description |
|---------|-------------|
| Currency info | Symbols, decimal places per currency |
| Currency validation | Check if supported |
| Singapore specifics | GST calculations, SFRS thresholds |
| Conversion stubs | Placeholder for future FX integration |

**Checklist**:
- [ ] `CURRENCY_INFO` dict with symbol, decimal_places, name for each currency
- [ ] `get_currency_symbol(code)` - get symbol (e.g., "SGD" → "S$")
- [ ] `get_currency_decimals(code)` - get standard decimal places
- [ ] `is_supported_currency(code)` - validation check
- [ ] `SGD_GST_RATE` constant = Decimal("0.09")
- [ ] `calculate_gst_exclusive(gst_inclusive)` - remove GST
- [ ] `calculate_gst_inclusive(gst_exclusive)` - add GST
- [ ] `calculate_gst_amount(base_amount)` - GST portion
- [ ] `SFRS_SMALL_ENTITY_THRESHOLDS` dict (revenue, assets, employees)
- [ ] `is_sfrs_small_entity(revenue, assets, employees)` - qualification check
- [ ] `format_sgd(amount, show_symbol)` - SGD-specific formatting
- [ ] Docstrings with Singapore regulatory references

---

### 3.11 `finanalyst_tools/validation/__init__.py`

**Purpose**: Export validation functions and classes.

**Checklist**:
- [ ] Import and export from `schema_validator.py`
- [ ] `__all__` list defined

---

### 3.12 `finanalyst_tools/validation/schema_validator.py`

**Purpose**: Schema validation for financial data.

**Features**:
| Feature | Description |
|---------|-------------|
| Statement validation | Validate each statement type |
| Completeness check | Verify required fields for analysis type |
| Pydantic integration | Use models for detailed validation |
| Alias handling | Support multiple field name conventions |

**Checklist**:
- [ ] Import `ValidationResult`, `ValidationIssue`, `ValidationSeverity` from models
- [ ] `REQUIRED_FIELDS` dict mapping analysis types to required fields
- [ ] `FIELD_ALIASES` dict mapping canonical names to accepted aliases
- [ ] `validate_income_statement_schema(data)` → `ValidationResult`
- [ ] `validate_balance_sheet_schema(data)` → `ValidationResult`
- [ ] `validate_cash_flow_schema(data)` → `ValidationResult`
- [ ] `validate_financial_data_completeness(is_data, bs_data, cf_data, analysis_type)` → `ValidationResult`
- [ ] `find_field_value(data, canonical_name, aliases)` - helper to find field by any alias
- [ ] `check_required_fields(data, required, aliases)` - batch check
- [ ] Integration with Pydantic `ValidationError` for detailed issues
- [ ] All functions return structured `ValidationResult` (not raise exceptions)

---

### 3.13 `finanalyst_tools/calculations/__init__.py`

**Purpose**: Export all calculator functions.

**Checklist**:
- [ ] Import and export all from `profitability.py`
- [ ] Import and export all from `liquidity.py`
- [ ] `__all__` list with all public functions

---

### 3.14 `finanalyst_tools/calculations/profitability.py`

**Purpose**: Profitability ratio calculations.

**Features**:
| Feature | Description |
|---------|-------------|
| Gross margin | Revenue - COGS analysis |
| Operating margin | EBIT margin calculation |
| Net margin | Bottom-line profitability |
| ROA | Return on assets |
| ROE | Return on equity |
| All with audit trail | Step-by-step calculation recording |

**Checklist**:
- [ ] Import `CalculationResult`, `MetricUnit` from models
- [ ] Import `safe_divide`, `calculate_percentage`, `to_decimal` from utils
- [ ] Import `PlausibilityRanges` from config
- [ ] `calculate_gross_profit_margin(revenue, cogs)` → `CalculationResult`
  - [ ] Step-by-step calculation in `calculation_steps`
  - [ ] Plausibility check against range
  - [ ] Warnings for out-of-range values
- [ ] `calculate_operating_profit_margin(revenue, cogs, opex, marketing, include_marketing_in_opex)` → `CalculationResult`
  - [ ] Handle marketing separately or included
  - [ ] Full calculation breakdown
- [ ] `calculate_net_profit_margin(revenue, net_income)` → `CalculationResult`
  - [ ] Flag if margin >= 100% (data error)
- [ ] `calculate_return_on_assets(net_income, assets_begin, assets_end)` → `CalculationResult`
  - [ ] Calculate average assets
  - [ ] Show all intermediate values
- [ ] `calculate_return_on_equity(net_income, equity_begin, equity_end)` → `CalculationResult`
  - [ ] Calculate average equity
  - [ ] Warn on negative equity
- [ ] `calculate_all_profitability_metrics(income_statement, balance_sheet, prior_balance_sheet)` → `MetricCollection`
  - [ ] Convenience function to calculate all 5 metrics
- [ ] All functions handle zero denominators gracefully
- [ ] All functions return None value (not error) when calculation impossible

---

### 3.15 `finanalyst_tools/calculations/liquidity.py`

**Purpose**: Liquidity ratio calculations.

**Features**:
| Feature | Description |
|---------|-------------|
| Current ratio | Basic liquidity measure |
| Quick ratio | Acid test (excludes inventory) |
| Cash ratio | Most conservative measure |
| Working capital | Absolute dollar amount |
| All with interpretation | Contextual warnings |

**Checklist**:
- [ ] Import `CalculationResult`, `MetricUnit` from models
- [ ] Import `safe_divide`, `to_decimal` from utils
- [ ] Import `PlausibilityRanges` from config
- [ ] `calculate_current_ratio(current_assets, current_liabilities)` → `CalculationResult`
  - [ ] Interpretation warnings (< 1.0 = risk, > 3.0 = possibly inefficient)
  - [ ] Plausibility check
- [ ] `calculate_quick_ratio(current_assets, inventory, current_liabilities)` → `CalculationResult`
  - [ ] Calculate quick assets first
  - [ ] Compare to current ratio in warnings
- [ ] `calculate_cash_ratio(cash, current_liabilities)` → `CalculationResult`
  - [ ] Most conservative interpretation
- [ ] `calculate_working_capital(current_assets, current_liabilities)` → `CalculationResult`
  - [ ] Currency unit (not ratio)
  - [ ] Negative working capital warning
- [ ] `calculate_all_liquidity_metrics(balance_sheet)` → `MetricCollection`
  - [ ] Convenience function for all 4 metrics
- [ ] All functions produce detailed `calculation_steps`

---

### 3.16 `finanalyst_tools/tool_registry.py`

**Purpose**: Central registry of all tools with metadata for LLM consumption.

**Features**:
| Feature | Description |
|---------|-------------|
| Tool categories | Enum for organizing tools |
| Tool parameters | Typed parameter definitions |
| Tool definitions | Complete metadata for each tool |
| OpenAI schema | Function calling format export |
| Discovery | List tools by category |

**Checklist**:
- [ ] `ToolCategory` enum with all categories (for Phase 1: VALIDATION, PROFITABILITY, LIQUIDITY)
- [ ] `ToolParameter` dataclass with:
  - [ ] `name`, `type`, `description`, `required`, `default`, `enum`
- [ ] `ToolDefinition` dataclass with:
  - [ ] `name`, `description`, `category`, `parameters`, `returns`, `example`, `function`
  - [ ] `to_openai_schema()` method
  - [ ] `to_anthropic_schema()` method (for Claude)
- [ ] `ToolRegistry` class with:
  - [ ] `_tools: dict[str, ToolDefinition]`
  - [ ] `register(tool)` method
  - [ ] `get(name)` method
  - [ ] `list_tools(category=None)` method
  - [ ] `get_openai_tools(categories=None)` method
  - [ ] `get_tool_descriptions()` method (for prompt inclusion)
  - [ ] `_register_all_tools()` private method
- [ ] Register all Phase 1 tools:
  - [ ] `validate_financial_data`
  - [ ] `calculate_gross_profit_margin`
  - [ ] `calculate_operating_profit_margin`
  - [ ] `calculate_net_profit_margin`
  - [ ] `calculate_return_on_assets`
  - [ ] `calculate_return_on_equity`
  - [ ] `calculate_current_ratio`
  - [ ] `calculate_quick_ratio`
  - [ ] `calculate_cash_ratio`
  - [ ] `calculate_working_capital`
- [ ] `TOOL_REGISTRY` global singleton instance
- [ ] Each tool has accurate parameter types and descriptions

---

### 3.17 `finanalyst_tools/dispatcher.py`

**Purpose**: Execute tool calls from LLM with validation and error handling.

**Features**:
| Feature | Description |
|---------|-------------|
| Parameter validation | Validate and coerce parameters |
| Execution timing | Track execution duration |
| Error handling | Graceful error recovery |
| Result formatting | Structured output for LLM |

**Checklist**:
- [ ] `ToolCallResult` dataclass with:
  - [ ] `tool_name: str`
  - [ ] `success: bool`
  - [ ] `result: Any`
  - [ ] `error: str | None`
  - [ ] `execution_time_ms: float | None`
  - [ ] `to_dict()` method
  - [ ] `to_json()` method
- [ ] `ToolDispatcher` class with:
  - [ ] `__init__(self)` - initialize with registry
  - [ ] `execute(tool_name, parameters)` → `ToolCallResult`
    - [ ] Validate tool exists
    - [ ] Validate required parameters present
    - [ ] Coerce parameter types (str→Decimal for numbers)
    - [ ] Execute function with timing
    - [ ] Catch and format exceptions
    - [ ] Return structured result
  - [ ] `validate_parameters(tool, parameters)` → `tuple[bool, str | None]`
  - [ ] `coerce_parameters(tool, parameters)` → `dict`
  - [ ] `format_error(exception)` → `str`
- [ ] `execute_tool(tool_name, parameters)` module-level convenience function
- [ ] Comprehensive error messages for debugging
- [ ] Timing uses `time.perf_counter()` for precision

---

### 3.18 `finanalyst_tools/__init__.py`

**Purpose**: Main package initialization and public API exports.

**Features**:
| Feature | Description |
|---------|-------------|
| Version | Package version string |
| Public API | Export commonly used classes/functions |
| Convenience | Easy import patterns |

**Checklist**:
- [ ] `__version__ = "0.1.0"`
- [ ] Import and export key classes:
  - [ ] `CalculationResult`, `MetricCollection`, `ConfidenceLevel`
  - [ ] `ValidationResult`, `ValidationIssue`
  - [ ] `FinancialStatementSet`, `MultiPeriodFinancialData`
  - [ ] `IncomeStatementData`, `BalanceSheetData`, `CashFlowStatementData`
- [ ] Import and export key functions:
  - [ ] All profitability calculators
  - [ ] All liquidity calculators
  - [ ] `validate_financial_data_completeness`
- [ ] Import and export tool integration:
  - [ ] `TOOL_REGISTRY`
  - [ ] `ToolDispatcher`, `execute_tool`
- [ ] `__all__` list with all public names
- [ ] Module docstring with package overview

---

## 4. Implementation Order

Based on the dependency graph, files will be implemented in this order:

| Order | File | Depends On |
|-------|------|------------|
| 1 | `config.py` | (none) |
| 2 | `exceptions.py` | (none) |
| 3 | `utils/__init__.py` | (none) |
| 4 | `utils/math_ops.py` | config |
| 5 | `utils/formatting.py` | config |
| 6 | `utils/currency.py` | config |
| 7 | `models/__init__.py` | (none) |
| 8 | `models/validation.py` | (none) |
| 9 | `models/financial_statements.py` | config |
| 10 | `models/analysis_results.py` | models/validation, models/financial_statements |
| 11 | `validation/__init__.py` | (none) |
| 12 | `validation/schema_validator.py` | models |
| 13 | `calculations/__init__.py` | (none) |
| 14 | `calculations/profitability.py` | models, utils, config |
| 15 | `calculations/liquidity.py` | models, utils, config |
| 16 | `tool_registry.py` | calculations, validation |
| 17 | `dispatcher.py` | tool_registry, exceptions |
| 18 | `__init__.py` | (all) |

---

## 5. Validation Checkpoint

### 5.1 Plan Validation Matrix

| Criteria | Status | Notes |
|----------|--------|-------|
| **Completeness** | ✅ | All 18 files specified with checklists |
| **Dependencies** | ✅ | Dependency graph verified, no cycles |
| **System Prompt Alignment** | ✅ | Supports Phase 1 VALIDATE, Phase 2-4 CALCULATE |
| **Type Safety** | ✅ | Pydantic models + type hints throughout |
| **Precision** | ✅ | Decimal-based math operations |
| **Audit Trail** | ✅ | CalculationResult with step-by-step |
| **LLM Integration** | ✅ | OpenAI function calling schema ready |
| **Error Handling** | ✅ | Custom exception hierarchy |
| **Singapore Context** | ✅ | GST, SFRS in currency.py |

### 5.2 Risk Assessment

| Risk | Mitigation |
|------|------------|
| Circular imports | Careful dependency ordering, TYPE_CHECKING for type hints |
| Pydantic v1/v2 compatibility | Using v2 syntax exclusively |
| Decimal serialization | Custom JSON encoder in to_dict/to_json methods |
| Tool registration timing | Lazy loading in registry |

### 5.3 Success Criteria

After Phase 1 implementation, the following must work:

```python
# Test 1: Basic import
from finanalyst_tools import (
    calculate_gross_profit_margin,
    calculate_current_ratio,
    ValidationResult,
    TOOL_REGISTRY,
    execute_tool,
)

# Test 2: Direct calculation
result = calculate_gross_profit_margin(revenue=100000, cost_of_goods_sold=40000)
assert result.value == Decimal("60.00")
assert result.is_plausible == True
assert len(result.calculation_steps) >= 3

# Test 3: Tool execution via dispatcher
result = execute_tool("calculate_current_ratio", {
    "current_assets": 150000,
    "current_liabilities": 100000
})
assert result.success == True
assert result.result.value == Decimal("1.50")

# Test 4: Tool discovery
tools = TOOL_REGISTRY.get_openai_tools()
assert len(tools) >= 10
```

---

## 6. Confirmation Request

**The Phase 1 Implementation Plan is complete and validated.**

Before I proceed with implementation, please confirm:

1. ✅ **Plan Approval**: Does this detailed plan meet your expectations?
2. ✅ **File Structure**: Is the proposed file organization acceptable?
3. ✅ **Feature Scope**: Should any features be added or removed from Phase 1?
4. ✅ **Proceed to Implementation**: Shall I begin creating all 18 files?

---

Phase 1 Implementation: Complete Core Infrastructure
Execution Status
I will now implement all 18 files in dependency order. Each file is complete, production-ready, and follows the detailed checklist from the plan.

File 1 of 18: finanalyst_tools/config.py
Python

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
File 2 of 18: finanalyst_tools/exceptions.py
Python

# finanalyst_tools/exceptions.py
"""
Custom exception hierarchy for FinAnalyst-Pro tools.

Provides specific exception types for different error categories:
- Calculation errors (arithmetic, division by zero, invalid inputs)
- Validation errors (schema, reconciliation, plausibility)
- Data errors (parsing, missing data)
- Tool errors (not found, execution failure)

All exceptions support serialization to dict/JSON for structured error handling.
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
    - Message storage
    - Optional details dictionary for context
    - JSON serialization support
    
    All custom exceptions should inherit from this class.
    """
    
    def __init__(
        self, 
        message: str, 
        details: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional context
            **kwargs: Additional key-value pairs to include in details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.details.update(kwargs)
    
    @property
    def error_type(self) -> str:
        """Get the exception class name."""
        return self.__class__.__name__
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert exception to dictionary for JSON serialization.
        
        Returns:
            Dictionary with error_type, message, and details
        """
        return {
            "error_type": self.error_type,
            "message": self.message,
            "details": self.details,
        }
    
    def to_json(self) -> str:
        """
        Convert exception to JSON string.
        
        Returns:
            JSON representation of the error
        """
        return json.dumps(self.to_dict(), indent=2, default=str)
    
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
        **kwargs: Any
    ) -> None:
        """
        Initialize calculation error.
        
        Args:
            message: Error description
            metric_name: Name of the metric being calculated
            formula: The formula that failed
            **kwargs: Additional context
        """
        details = kwargs
        if metric_name:
            details["metric_name"] = metric_name
        if formula:
            details["formula"] = formula
        super().__init__(message, details=details)


class DivisionByZeroError(CalculationError):
    """
    Raised when a calculation would result in division by zero.
    
    Includes information about the numerator and denominator
    to aid in debugging data issues.
    """
    
    def __init__(
        self,
        numerator: Any,
        denominator: Any,
        metric_name: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize division by zero error.
        
        Args:
            numerator: The dividend value
            denominator: The divisor (zero)
            metric_name: Name of the metric being calculated
            **kwargs: Additional context
        """
        message = f"Cannot divide {numerator} by zero"
        super().__init__(
            message,
            metric_name=metric_name,
            numerator=numerator,
            denominator=denominator,
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
        parameter_name: str | None = None,
        received_value: Any = None,
        expected: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize invalid input error.
        
        Args:
            message: Error description
            parameter_name: Name of the invalid parameter
            received_value: The value that was received
            expected: Description of what was expected
            **kwargs: Additional context
        """
        details = kwargs
        if parameter_name:
            details["parameter_name"] = parameter_name
        if received_value is not None:
            details["received_value"] = received_value
        if expected:
            details["expected"] = expected
        super().__init__(message, **details)


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
        **kwargs: Any
    ) -> None:
        """
        Initialize validation error.
        
        Args:
            message: Error description
            field: The field that failed validation
            validation_type: Type of validation that failed
            **kwargs: Additional context
        """
        details = kwargs
        if field:
            details["field"] = field
        if validation_type:
            details["validation_type"] = validation_type
        super().__init__(message, details=details)


class SchemaError(ValidationError):
    """
    Raised when data doesn't conform to expected schema.
    
    Typically occurs during Pydantic model validation.
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
        super().__init__(
            message,
            field=field,
            validation_type="schema",
            expected_type=expected_type,
            received_type=received_type,
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
            check_name=check_name,
            value_a=value_a,
            source_a=source_a,
            value_b=value_b,
            source_b=source_b,
            difference=difference,
            tolerance=tolerance,
            **kwargs
        )


class PlausibilityError(ValidationError):
    """
    Raised when a calculated metric is outside plausible range.
    
    Note: This is typically a warning, not an error, unless explicitly strict.
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
            metric_name=metric_name,
            value=value,
            expected_range=expected_range,
            **kwargs
        )


# ============================================================================
# DATA ERRORS
# ============================================================================

class DataError(FinAnalystError):
    """
    Base exception for data-related errors.
    
    Raised when there are issues with the input data itself.
    """
    pass


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
            details["raw_data"] = raw_data[:500] if len(raw_data) > 500 else raw_data
        super().__init__(message, details=details)


class MissingDataError(DataError):
    """
    Raised when required data is missing.
    
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
        details = kwargs
        if missing_fields:
            details["missing_fields"] = missing_fields
        if required_for:
            details["required_for"] = required_for
        super().__init__(message, details=details)


# ============================================================================
# TOOL ERRORS
# ============================================================================

class ToolError(FinAnalystError):
    """
    Base exception for tool-related errors.
    
    Raised when issues occur during tool execution.
    """
    
    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize tool error.
        
        Args:
            message: Error description
            tool_name: Name of the tool
            **kwargs: Additional context
        """
        details = kwargs
        if tool_name:
            details["tool_name"] = tool_name
        super().__init__(message, details=details)


class ToolNotFoundError(ToolError):
    """
    Raised when a requested tool does not exist.
    
    Includes suggestions for similar tool names if available.
    """
    
    def __init__(
        self,
        tool_name: str,
        available_tools: list[str] | None = None,
        suggestions: list[str] | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize tool not found error.
        
        Args:
            tool_name: Name of the tool that wasn't found
            available_tools: List of all available tool names
            suggestions: Similar tool names as suggestions
            **kwargs: Additional context
        """
        message = f"Tool '{tool_name}' not found"
        if suggestions:
            message += f". Did you mean: {', '.join(suggestions)}?"
        
        super().__init__(
            message,
            tool_name=tool_name,
            available_tools=available_tools,
            suggestions=suggestions,
            **kwargs
        )


class ToolExecutionError(ToolError):
    """
    Raised when a tool fails during execution.
    
    Wraps the original exception with tool context.
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
        message = f"Tool '{tool_name}' failed: {str(original_error)}"
        
        super().__init__(
            message,
            tool_name=tool_name,
            original_error_type=type(original_error).__name__,
            original_error_message=str(original_error),
            parameters=parameters,
            **kwargs
        )
        self.original_error = original_error


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
        
        super().__init__(
            full_message,
            tool_name=tool_name,
            parameter_name=parameter_name,
            received_value=received_value,
            expected_type=expected_type,
            **kwargs
        )
File 3 of 18: finanalyst_tools/utils/__init__.py
Python

# finanalyst_tools/utils/__init__.py
"""
Utility functions for FinAnalyst-Pro tools.

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
    compare_values,
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
    "compare_values",
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
File 4 of 18: finanalyst_tools/utils/math_ops.py
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
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Sequence, TypeVar

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


# Type variable for numeric types
Numeric = TypeVar("Numeric", int, float, Decimal)


# ============================================================================
# TYPE CONVERSION
# ============================================================================

def to_decimal(
    value: Numeric | str | None,
    default: Decimal | None = None
) -> Decimal:
    """
    Safely convert a value to Decimal.
    
    Handles various input types and returns a default value
    for None or unconvertible inputs.
    
    Args:
        value: 
