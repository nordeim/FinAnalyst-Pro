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

**Awaiting your confirmation to proceed with full implementation of all Phase 1 files.**
