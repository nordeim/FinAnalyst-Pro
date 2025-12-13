# FinAnalyst-Pro — Agent Toolset Handbook (Single Source of Truth)

This handbook complements `AGENT_SYSTEM_PROMPT.md`.

- **`AGENT_SYSTEM_PROMPT.md`** defines *who the agent is*, the mandatory pipeline, and the output/report contract.
- **This handbook** defines *what tools exist*, *exactly how to call them*, and *what they return*, based on the code in `finanalyst_tools/`.

---

## Scope and authority

This document is intended to be **authoritative** for the tool surface implemented in:

- `finanalyst_tools/tool_registry.py`
- `finanalyst_tools/dispatcher.py`

Tool names, parameters, enums, defaults, and return behavior are taken directly from `ToolRegistry` registrations and execution logic.

---

## Tool invocation contract

### Tool boundary type coercion (important)

Tool inputs are validated and coerced at execution time.

- **`number`**
  - Accepts: `Decimal`, `int`, `float`, `str`.
  - `float` is converted via `Decimal(str(value))`.
  - **Non-finite values** (`NaN`, `Infinity`) are rejected.

- **`object` / `array`**
  - Accepts: native `dict`/`list`, or a **JSON string** that parses to `dict`/`list`.
  - Nested `float` values inside objects/arrays are normalized to `Decimal`.

- **`boolean`**
  - Accepts booleans or strings like `"true"`, `"1"`, `"yes"`.

- **`string`**
  - Coerced via `str(value)`.

If coercion/validation fails, the registry raises a `ToolParameterError` (or returns a formatted validation block when executing via `ToolDefinition.execute`).

### Currency formatting

The toolkit supports dynamic currency symbol formatting. When a `CalculationResult` has `unit=CURRENCY`, the `formatted_value` property uses the result's `currency` field:

| Currency Code | Symbol | Example |
|---------------|--------|----------|
| USD | $ | $1,000.00 |
| SGD | S$ | S$1,000.00 |
| EUR | € | €1,000.00 |
| GBP | £ | £1,000.00 |
| JPY / CNY | ¥ | ¥1,000.00 |

Unknown currencies fall back to displaying the currency code as prefix.

---

## Two execution paths (choose based on integration)

### A) `ToolRegistry.execute_tool(...)` (agent-friendly output)

`ToolRegistry.execute_tool(tool_name, **kwargs) -> str`

- **Returns a string**:
  - `CalculationResult` → converted to a Markdown-style reasoning block (`CalculationResult.to_reasoning_block()`).
  - `ValidationResult` → converted to a validation summary block (via `result_to_reasoning_block` from `validation.utils`).
  - `dict` → pretty JSON string.
  - `str` → returned as-is.

This is the simplest path when you want the tool to return a ready-to-insert block into the final report.

### B) `ToolDispatcher.execute(tool_name, parameters)` (structured runtime result)

`ToolDispatcher.execute(...) -> ToolCallResult`

- Returns a `ToolCallResult` object with:
  - `success` boolean
  - `result` (raw object/string)
  - error fields when failed
  - execution time

This path is useful when you want structured programmatic branching on success/failure.

---

## LLM tool menu vs full registry

`ToolRegistry.get_openai_tools()` and `ToolRegistry.get_anthropic_tools()` return **only tools with `expose_to_llm=True`**.

- **LLM-exposed tools (current)**:
  - `analyze_financials`
  - `validate_financial_data`

All other tools are registered but `expose_to_llm=False` (internal-only by design).

---

## Canonical workflow mapping (matches `AGENT_SYSTEM_PROMPT.md`)

- **Phase 1: VALIDATE (Mandatory)**
  - Use `validate_financial_data` if you need a validation-only pass before analysis.
  - `analyze_financials` also performs validation internally as part of its pipeline.

- **Phases 2–5: ANALYZE → CALCULATE → INTERPRET → VERIFY**
  - Use `analyze_financials` to run the full 5-phase pipeline and generate the final report.

---

# Tool reference

## 1) Agent-facing orchestration tools (LLM-exposed)

### `analyze_financials`

- **Category**: `analysis`
- **Description**: Run the full 5-phase analysis pipeline and return a formatted report
- **Returns**: `Formatted report string`

**Parameters**

- `statement_set` (`object`, required)
  - FinancialStatementSet payload (`income_statement`, `balance_sheet`, optional `cash_flow_statement`).
- `prior_statement_set` (`object`, optional)
  - Prior-period FinancialStatementSet (optional).
- `analysis_type` (`string`, optional, default: `"comprehensive"`)
  - Allowed values:
    - `profitability`
    - `liquidity`
    - `solvency`
    - `efficiency`
    - `comprehensive`
- `include_trends` (`boolean`, optional, default: `false`)
  - Whether to include trend analysis.
- `currency` (`string`, optional, default: `"SGD"`)
  - Reporting currency.
- `report_format` (`string`, optional, default: `"markdown"`)
  - Allowed values:
    - `markdown`
    - `json`
- `include_audit_trail` (`boolean`, optional, default: `true`)
  - Whether to include the calculation audit trail.

**Behavior notes (from implementation)**

- `statement_set` and `prior_statement_set` are parsed using `FinancialStatementSet.model_validate(...)` when provided as dictionaries.
- Internally constructs an `AnalysisRequest` and runs `AnalysisPipeline().execute(request)`.
- If `report_format == "json"`, returns `result.to_json()`.
- Otherwise returns Markdown via `generate_financial_report(..., format=ReportFormat.MARKDOWN, include_audit_trail=...)`.

> **Important (v1.0.0)**: Balance sheet validation now requires these fields:
> - `cash_and_equivalents`
> - `total_assets`
> - `total_liabilities`
> - `total_shareholders_equity`
>
> Balance sheets missing any of these will fail validation.

**Example (ToolDispatcher style)**

```json
{
  "tool_name": "analyze_financials",
  "parameters": {
    "statement_set": {
      "income_statement": {
        "period": {"year": 2023, "period_type": "annual"},
        "currency": "SGD",
        "total_revenue": 1000000,
        "cost_of_goods_sold": 600000,
        "net_income": 150000
      },
      "balance_sheet": {
        "period": {"year": 2023, "period_type": "annual"},
        "currency": "SGD",
        "cash_and_equivalents": 100000,
        "total_assets": 800000,
        "total_liabilities": 450000,
        "total_shareholders_equity": 350000,
        "current_assets": 500000,
        "current_liabilities": 300000,
        "inventory": 50000
      }
    },
    "analysis_type": "comprehensive",
    "include_trends": false,
    "currency": "SGD",
    "report_format": "markdown",
    "include_audit_trail": true
  }
}
```

---

### `validate_financial_data`

- **Category**: `validation`
- **Description**: Validate that financial data is complete and properly formatted for analysis
- **Returns**: `ValidationResult with any issues found`

**Parameters**

- `income_statement` (`object`, optional)
  - Income statement data.
- `balance_sheet` (`object`, optional)
  - Balance sheet data.
- `cash_flow` (`object`, optional)
  - Cash flow statement data.
- `analysis_type` (`string`, required)
  - Allowed values:
    - `profitability`
    - `liquidity`
    - `solvency`
    - `efficiency`
    - `comprehensive`

**Behavior notes (from implementation)**

- Calls `validate_financial_data_completeness(...)`.
- Intended for a Phase-1 validation pass.
- When executed via `ToolRegistry.execute_tool(...)`, the `ValidationResult` is converted into a formatted validation block.

**Example (ToolDispatcher style)**

```json
{
  "tool_name": "validate_financial_data",
  "parameters": {
    "income_statement": {"total_revenue": 1000000, "cost_of_goods_sold": 600000},
    "balance_sheet": {"cash_and_equivalents": 100000},
    "analysis_type": "profitability"
  }
}
```

---

## 2) Internal calculation tools (registered, not LLM-exposed)

These tools are marked `expose_to_llm=False`. They are primarily intended to be used internally by orchestration layers.

> If invoked via `ToolRegistry.execute_tool(...)`, they return a **reasoning block string** (because they return `CalculationResult`).

### Profitability (`profitability`)

#### `calculate_gross_profit_margin`
- **Description**: Calculate Gross Profit Margin: (Revenue - COGS) / Revenue × 100
- **Parameters**:
  - `revenue` (`number`, required)
  - `cost_of_goods_sold` (`number`, required)
- **Returns**: CalculationResult with gross profit margin percentage and calculation steps

#### `calculate_operating_profit_margin`
- **Description**: Calculate Operating Profit Margin: (Revenue - COGS - OpEx) / Revenue × 100
- **Parameters**:
  - `revenue` (`number`, required)
  - `cost_of_goods_sold` (`number`, required)
  - `operating_expenses` (`number`, required)
  - `marketing_expenses` (`number`, optional)
- **Returns**: CalculationResult with operating profit margin percentage and calculation steps

#### `calculate_net_profit_margin`
- **Description**: Calculate Net Profit Margin: Net Income / Revenue × 100
- **Parameters**:
  - `revenue` (`number`, required)
  - `net_income` (`number`, required)
- **Returns**: CalculationResult with net profit margin percentage and calculation steps

#### `calculate_ebitda_margin`
- **Description**: Calculate EBITDA Margin: EBITDA / Revenue × 100
- **Parameters**:
  - `revenue` (`number`, required)
  - `ebitda` (`number`, required)
- **Returns**: CalculationResult with EBITDA margin percentage and calculation steps

#### `calculate_return_on_assets`
- **Description**: Calculate ROA: Net Income / Average Total Assets × 100
- **Parameters**:
  - `net_income` (`number`, required)
  - `total_assets_beginning` (`number`, required)
  - `total_assets_ending` (`number`, required)
- **Returns**: CalculationResult with return on assets percentage and calculation steps

#### `calculate_return_on_equity`
- **Description**: Calculate ROE: Net Income / Average Shareholders' Equity × 100
- **Parameters**:
  - `net_income` (`number`, required)
  - `shareholders_equity_beginning` (`number`, required)
  - `shareholders_equity_ending` (`number`, required)
- **Returns**: CalculationResult with return on equity percentage and calculation steps

#### `calculate_return_on_capital_employed`
- **Description**: Calculate ROCE: EBIT / Capital Employed × 100
- **Parameters**:
  - `ebit` (`number`, required)
  - `total_assets` (`number`, required)
  - `current_liabilities` (`number`, required)
- **Returns**: CalculationResult with ROCE percentage and calculation steps

---

### Liquidity (`liquidity`)

#### `calculate_current_ratio`
- **Description**: Calculate Current Ratio: Current Assets / Current Liabilities
- **Parameters**:
  - `current_assets` (`number`, required)
  - `current_liabilities` (`number`, required)
- **Returns**: CalculationResult with current ratio and calculation steps

#### `calculate_quick_ratio`
- **Description**: Calculate Quick Ratio (Acid Test): (Current Assets - Inventory) / Current Liabilities
- **Parameters**:
  - `current_assets` (`number`, required)
  - `inventory` (`number`, required)
  - `current_liabilities` (`number`, required)
- **Returns**: CalculationResult with quick ratio and calculation steps

#### `calculate_cash_ratio`
- **Description**: Calculate Cash Ratio: Cash and Equivalents / Current Liabilities
- **Parameters**:
  - `cash_and_equivalents` (`number`, required)
  - `current_liabilities` (`number`, required)
- **Returns**: CalculationResult with cash ratio and calculation steps

#### `calculate_working_capital`
- **Description**: Calculate Working Capital: Current Assets - Current Liabilities
- **Parameters**:
  - `current_assets` (`number`, required)
  - `current_liabilities` (`number`, required)
- **Returns**: CalculationResult with working capital amount and calculation steps

---

## Practical guidance for the agent

### Prefer `analyze_financials` for end-user answers

If the user asks for a report/analysis, `analyze_financials` is the single call that:
- validates
- calculates
- interprets
- verifies
- produces a formatted report

### Use `validate_financial_data` when you need a validation-only step

Use this tool when:
- you need to report missing fields before running a full analysis
- you want to enforce the Phase-1 contract explicitly before running `analyze_financials`

### Parameter shape pitfalls to avoid

- `validate_financial_data` expects **separate statement objects** (`income_statement`, `balance_sheet`, `cash_flow`).
- `analyze_financials` expects a **single `statement_set` object** containing (at minimum) `income_statement` and `balance_sheet`.

---

## Appendix: provider tool schemas

To generate provider-specific function/tool schemas (LLM-visible tools only):

```python
from finanalyst_tools.tool_registry import ToolRegistry

registry = ToolRegistry()
openai_tools = registry.get_openai_tools()
anthropic_tools = registry.get_anthropic_tools()
```

---

## Change control

If you add or modify tools:

- Update registrations in `finanalyst_tools/tool_registry.py`.
- Decide model visibility via `expose_to_llm`.
- Keep this handbook aligned with the registry definitions (this file is intended to remain the single agent-facing reference).
