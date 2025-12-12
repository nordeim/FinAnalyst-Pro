# finanalyst_tools/tool_registry.py
```5
# finanalyst_tools/tool_registry.py
"""
Central registry of all tools with metadata for LLM function calling.

Provides:
- Tool definitions with parameters and descriptions
- OpenAI/Anthropic schema generation
- Tool discovery and listing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import json


class ToolCategory(str, Enum):
    """Categories for organizing tools."""
    VALIDATION = "validation"
    PROFITABILITY = "profitability"
    LIQUIDITY = "liquidity"
    SOLVENCY = "solvency"
    EFFICIENCY = "efficiency"
    GROWTH = "growth"
    TREND = "trend"
    RECONCILIATION = "reconciliation"
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
    
    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function parameter schema."""
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        return schema
    
    def to_anthropic_schema(self) -> dict[str, Any]:
        """Convert to Anthropic tool parameter schema."""
        return self.to_openai_schema()  # Same format


@dataclass
class ToolDefinition:
    """Complete definition of a tool."""
    name: str
    description: str
    category: ToolCategory
    parameters: list[ToolParameter] = field(default_factory=list)
    returns: str = "CalculationResult object with value and audit trail"
    example: str | None = None
    function: Callable[..., Any] | None = None
    
    @property
    def required_parameters(self) -> list[str]:
        """Get list of required parameter names."""
        return [p.name for p in self.parameters if p.required]
    
    @property
    def optional_parameters(self) -> list[str]:
        """Get list of optional parameter names."""
        return [p.name for p in self.parameters if not p.required]
    
    def to_openai_schema(self) -> dict[str, Any]:
        """
        Convert to OpenAI function calling schema.
        
        Returns:
            Dictionary matching OpenAI's function schema format
        """
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_openai_schema()
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
                },
            },
        }
    
    def to_anthropic_schema(self) -> dict[str, Any]:
        """
        Convert to Anthropic tool use schema.
        
        Returns:
            Dictionary matching Anthropic's tool schema format
        """
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_anthropic_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                }
                for p in self.parameters
            ],
            "returns": self.returns,
            "example": self.example,
        }


class ToolRegistry:
    """
    Central registry for all available tools.
    
    Provides:
    - Tool registration
    - Tool lookup by name
    - Tool listing by category
    - Schema generation for LLM integration
    """
    
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._register_all_tools()
    
    def register(self, tool: ToolDefinition) -> None:
        """
        Register a tool.
        
        Args:
            tool: Tool definition to register
        """
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> ToolDefinition | None:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            ToolDefinition or None if not found
        """
        return self._tools.get(name)
    
    def list_tools(
        self,
        category: ToolCategory | None = None,
    ) -> list[ToolDefinition]:
        """
        List all registered tools.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tool definitions
        """
        tools = list(self._tools.values())
        
        if category:
            tools = [t for t in tools if t.category == category]
        
        return sorted(tools, key=lambda t: (t.category.value, t.name))
    
    def get_tool_names(
        self,
        category: ToolCategory | None = None,
    ) -> list[str]:
        """Get list of tool names."""
        return [t.name for t in self.list_tools(category)]
    
    def get_openai_tools(
        self,
        categories: list[ToolCategory] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get tools in OpenAI function calling format.
        
        Args:
            categories: Optional list of categories to include
            
        Returns:
            List of tool schemas for OpenAI API
        """
        tools = self.list_tools()
        
        if categories:
            tools = [t for t in tools if t.category in categories]
        
        return [t.to_openai_schema() for t in tools]
    
    def get_anthropic_tools(
        self,
        categories: list[ToolCategory] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get tools in Anthropic tool use format.
        
        Args:
            categories: Optional list of categories to include
            
        Returns:
            List of tool schemas for Anthropic API
        """
        tools = self.list_tools()
        
        if categories:
            tools = [t for t in tools if t.category in categories]
        
        return [t.to_anthropic_schema() for t in tools]
    
    def get_tool_descriptions(self) -> str:
        """
        Get human-readable descriptions of all tools.
        
        Returns:
            Formatted string with tool descriptions
        """
        lines = ["# Available Tools\n"]
        
        current_category = None
        for tool in self.list_tools():
            if tool.category != current_category:
                current_category = tool.category
                lines.append(f"\n## {current_category.value.title()}\n")
            
            lines.append(f"### {tool.name}")
            lines.append(f"{tool.description}\n")
            
            if tool.parameters:
                lines.append("**Parameters:**")
                for param in tool.parameters:
                    req = "(required)" if param.required else "(optional)"
                    lines.append(f"- `{param.name}` ({param.type}) {req}: {param.description}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _register_all_tools(self) -> None:
        """Register all Phase 1 tools."""
        # Import here to avoid circular imports
        from finanalyst_tools.calculations.profitability import (
            calculate_gross_profit_margin,
            calculate_operating_profit_margin,
            calculate_net_profit_margin,
            calculate_ebitda_margin,
            calculate_return_on_assets,
            calculate_return_on_equity,
            calculate_return_on_capital_employed,
        )
        from finanalyst_tools.calculations.liquidity import (
            calculate_current_ratio,
            calculate_quick_ratio,
            calculate_cash_ratio,
            calculate_working_capital,
        )
        from finanalyst_tools.validation.schema_validator import (
            validate_financial_data_completeness,
        )
        
        # ─────────────────────────────────────────────────────────────────
        # VALIDATION TOOLS
        # ─────────────────────────────────────────────────────────────────
        
        self.register(ToolDefinition(
            name="validate_financial_data",
            description="Validate that financial data is complete and properly formatted for analysis",
            category=ToolCategory.VALIDATION,
            parameters=[
                ToolParameter("income_statement", "object", "Income statement data"),
                ToolParameter("balance_sheet", "object", "Balance sheet data"),
                ToolParameter("cash_flow", "object", "Cash flow statement data", required=False),
                ToolParameter("analysis_type", "string", "Type of analysis to validate for",
                            enum=["profitability", "liquidity", "solvency", "efficiency", "comprehensive"]),
            ],
            returns="ValidationResult with any issues found",
            function=validate_financial_data_completeness,
        ))
        
        # ─────────────────────────────────────────────────────────────────
        # PROFITABILITY TOOLS
        # ─────────────────────────────────────────────────────────────────
        
        self.register(ToolDefinition(
            name="calculate_gross_profit_margin",
            description="Calculate Gross Profit Margin: (Revenue - COGS) / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("revenue", "number", "Total revenue / net sales"),
                ToolParameter("cost_of_goods_sold", "number", "Cost of goods sold / cost of sales"),
            ],
            returns="CalculationResult with gross profit margin percentage",
            example='{"revenue": 1000000, "cost_of_goods_sold": 600000}',
            function=calculate_gross_profit_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_operating_profit_margin",
            description="Calculate Operating Profit Margin: (Revenue - COGS - OpEx) / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("revenue", "number", "Total revenue"),
                ToolParameter("cost_of_goods_sold", "number", "Cost of goods sold"),
                ToolParameter("operating_expenses", "number", "Operating expenses"),
                ToolParameter("marketing_expenses", "number", "Marketing expenses if tracked separately", required=False),
            ],
            returns="CalculationResult with operating profit margin percentage",
            function=calculate_operating_profit_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_net_profit_margin",
            description="Calculate Net Profit Margin: Net Income / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("revenue", "number", "Total revenue"),
                ToolParameter("net_income", "number", "Net income (profit after tax)"),
            ],
            returns="CalculationResult with net profit margin percentage",
            function=calculate_net_profit_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_ebitda_margin",
            description="Calculate EBITDA Margin: EBITDA / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("revenue", "number", "Total revenue"),
                ToolParameter("ebitda", "number", "Earnings Before Interest, Taxes, Depreciation, and Amortization"),
            ],
            returns="CalculationResult with EBITDA margin percentage",
            function=calculate_ebitda_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_return_on_assets",
            description="Calculate ROA: Net Income / Average Total Assets × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("net_income", "number", "Net income for the period"),
                ToolParameter("total_assets_beginning", "number", "Total assets at period start"),
                ToolParameter("total_assets_ending", "number", "Total assets at period end"),
            ],
            returns="CalculationResult with ROA percentage",
            function=calculate_return_on_assets,
        ))
        
        self.register(ToolDefinition(
            name="calculate_return_on_equity",
            description="Calculate ROE: Net Income / Average Shareholders' Equity × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("net_income", "number", "Net income for the period"),
                ToolParameter("shareholders_equity_beginning", "number", "Equity at period start"),
                ToolParameter("shareholders_equity_ending", "number", "Equity at period end"),
            ],
            returns="CalculationResult with ROE percentage",
            function=calculate_return_on_equity,
        ))
        
        self.register(ToolDefinition(
            name="calculate_return_on_capital_employed",
            description="Calculate ROCE: EBIT / Capital Employed × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("ebit", "number", "Earnings Before Interest and Taxes"),
                ToolParameter("total_assets", "number", "Total assets"),
                ToolParameter("current_liabilities", "number", "Current liabilities"),
            ],
            returns="CalculationResult with ROCE percentage",
            function=calculate_return_on_capital_employed,
        ))
        
        # ─────────────────────────────────────────────────────────────────
        # LIQUIDITY TOOLS
        # ─────────────────────────────────────────────────────────────────
        
        self.register(ToolDefinition(
            name="calculate_current_ratio",
            description="Calculate Current Ratio: Current Assets / Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter("current_assets", "number", "Total current assets"),
                ToolParameter("current_liabilities", "number", "Total current liabilities"),
            ],
            returns="CalculationResult with current ratio",
            example='{"current_assets": 500000, "current_liabilities": 300000}',
            function=calculate_current_ratio,
        ))
        
        self.register(ToolDefinition(
            name="calculate_quick_ratio",
            description="Calculate Quick Ratio (Acid Test): (Current Assets - Inventory) / Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter("current_assets", "number", "Total current assets"),
                ToolParameter("inventory", "number", "Inventory value"),
                ToolParameter("current_liabilities", "number", "Total current liabilities"),
            ],
            returns="CalculationResult with quick ratio",
            function=calculate_quick_ratio,
        ))
        
        self.register(ToolDefinition(
            name="calculate_cash_ratio",
            description="Calculate Cash Ratio: Cash and Equivalents / Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter("cash_and_equivalents", "number", "Cash and cash equivalents"),
                ToolParameter("current_liabilities", "number", "Total current liabilities"),
            ],
            returns="CalculationResult with cash ratio",
            function=calculate_cash_ratio,
        ))
        
        self.register(ToolDefinition(
            name="calculate_working_capital",
            description="Calculate Working Capital: Current Assets - Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter("current_assets", "number", "Total current assets"),
                ToolParameter("current_liabilities", "number", "Total current 

```

# finanalyst_tools/calculations/liquidity.py
```5
# finanalyst_tools/calculations/liquidity.py
"""
Liquidity ratio calculations.

Provides calculations for:
- Current Ratio
- Quick Ratio (Acid Test)
- Cash Ratio
- Working Capital
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from finanalyst_tools.config import METRIC_FORMULAS, PlausibilityRanges
from finanalyst_tools.models.analysis_results import (
    CalculationResult,
    MetricUnit,
    MetricCategory,
    MetricCollection,
)
from finanalyst_tools.models.financial_statements import (
    BalanceSheetData,
    FinancialStatementSet,
)
from finanalyst_tools.calculations.base import (
    BaseCalculator,
    create_calculation_result,
    extract_value,
)
from finanalyst_tools.utils.math_ops import (
    to_decimal,
    safe_divide,
    round_decimal,
    is_effectively_zero,
)


# ============================================================================
# STANDALONE CALCULATION FUNCTIONS
# ============================================================================

def calculate_current_ratio(
    current_assets: Decimal | float | int,
    current_liabilities: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Current Ratio.
    
    Formula: Current Assets / Current Liabilities
    
    Interpretation:
    - < 1.0: May have difficulty meeting short-term obligations
    - 1.0 - 2.0: Generally healthy
    - > 3.0: May have excess assets not being utilized efficiently
    
    Args:
        current_assets: Total current assets
        current_liabilities: Total current liabilities
        
    Returns:
        CalculationResult with current ratio
    """
    steps = []
    warnings = []
    
    ca = to_decimal(current_assets)
    cl = to_decimal(current_liabilities)
    
    steps.append(f"Current Assets = {ca:,.2f}")
    steps.append(f"Current Liabilities = {cl:,.2f}")
    
    if is_effectively_zero(cl):
        warnings.append("Current liabilities is zero - cannot calculate ratio")
        # If no liabilities, could interpret as "infinite" liquidity
        if ca > 0:
            warnings.append("No current liabilities with positive current assets indicates strong liquidity")
        return create_calculation_result(
            metric_name="Current Ratio",
            value=None,
            formula=METRIC_FORMULAS.get("current_ratio", "Current Assets / Current Liabilities"),
            inputs={"current_assets": float(ca), "current_liabilities": float(cl)},
            steps=steps,
            category=MetricCategory.LIQUIDITY,
            warnings=warnings,
        )
    
    ratio = safe_divide(ca, cl, precision=4)
    steps.append(f"Current Ratio = Current Assets / Current Liabilities = {ca:,.2f} / {cl:,.2f} = {ratio:.4f}")
    
    # Interpretation warnings
    if ratio < Decimal("1.0"):
        warnings.append("Current ratio < 1.0 indicates potential liquidity risk")
        warnings.append("Company may have difficulty meeting short-term obligations")
    elif ratio < Decimal("1.2"):
        warnings.append("Current ratio is low - monitor liquidity position closely")
    elif ratio > Decimal("3.0"):
        warnings.append("High current ratio may indicate inefficient use of assets")
    else:
        steps.append("Current ratio is within healthy range (1.2 - 3.0)")
    
    return create_calculation_result(
        metric_name="Current Ratio",
        value=ratio,
        formula=METRIC_FORMULAS.get("current_ratio", "Current Assets / Current Liabilities"),
        inputs={"current_assets": float(ca), "current_liabilities": float(cl)},
        steps=steps,
        category=MetricCategory.LIQUIDITY,
        warnings=warnings,
    )


def calculate_quick_ratio(
    current_assets: Decimal | float | int,
    inventory: Decimal | float | int,
    current_liabilities: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Quick Ratio (Acid Test Ratio).
    
    Formula: (Current Assets - Inventory) / Current Liabilities
    
    More conservative than current ratio as inventory may not be
    quickly convertible to cash.
    
    Args:
        current_assets: Total current assets
        inventory: Inventory value
        current_liabilities: Total current liabilities
        
    Returns:
        CalculationResult with quick ratio
    """
    steps = []
    warnings = []
    
    ca = to_decimal(current_assets)
    inv = to_decimal(inventory)
    cl = to_decimal(current_liabilities)
    
    steps.append(f"Current Assets = {ca:,.2f}")
    steps.append(f"Inventory = {inv:,.2f}")
    steps.append(f"Current Liabilities = {cl:,.2f}")
    
    # Calculate quick assets
    quick_assets = ca - inv
    steps.append(f"Quick Assets = Current Assets - Inventory = {ca:,.2f} - {inv:,.2f} = {quick_assets:,.2f}")
    
    if is_effectively_zero(cl):
        warnings.append("Current liabilities is zero - cannot calculate ratio")
        return create_calculation_result(
            metric_name="Quick Ratio",
            value=None,
            formula=METRIC_FORMULAS.get("quick_ratio", "(Current Assets - Inventory) / Current Liabilities"),
            inputs={
                "current_assets": float(ca),
                "inventory": float(inv),
                "current_liabilities": float(cl),
            },
            steps=steps,
            category=MetricCategory.LIQUIDITY,
            warnings=warnings,
        )
    
    ratio = safe_divide(quick_assets, cl, precision=4)
    steps.append(f"Quick Ratio = Quick Assets / Current Liabilities = {quick_assets:,.2f} / {cl:,.2f} = {ratio:.4f}")
    
    # Interpretation
    if ratio < Decimal("1.0"):
        warnings.append("Quick ratio < 1.0 indicates reliance on inventory to meet obligations")
    elif ratio < Decimal("0.5"):
        warnings.append("Very low quick ratio - significant liquidity concern")
    
    # Compare to current ratio
    current_ratio = safe_divide(ca, cl, precision=4)
    if current_ratio > 0:
        inv_impact = current_ratio - ratio
        if inv_impact > Decimal("0.5"):
            steps.append(f"Note: Inventory accounts for {inv_impact:.2f} of the current ratio difference")
    
    return create_calculation_result(
        metric_name="Quick Ratio",
        value=ratio,
        formula=METRIC_FORMULAS.get("quick_ratio", "(Current Assets - Inventory) / Current Liabilities"),
        inputs={
            "current_assets": float(ca),
            "inventory": float(inv),
            "current_liabilities": float(cl),
            "quick_assets": float(quick_assets),
        },
        steps=steps,
        category=MetricCategory.LIQUIDITY,
        warnings=warnings,
    )


def calculate_cash_ratio(
    cash_and_equivalents: Decimal | float | int,
    current_liabilities: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Cash Ratio.
    
    Formula: Cash and Equivalents / Current Liabilities
    
    Most conservative liquidity measure - only considers cash
    and cash equivalents.
    
    Args:
        cash_and_equivalents: Cash and cash equivalents
        current_liabilities: Total current liabilities
        
    Returns:
        CalculationResult with cash ratio
    """
    steps = []
    warnings = []
    
    cash = to_decimal(cash_and_equivalents)
    cl = to_decimal(current_liabilities)
    
    steps.append(f"Cash and Equivalents = {cash:,.2f}")
    steps.append(f"Current Liabilities = {cl:,.2f}")
    
    if is_effectively_zero(cl):
        warnings.append("Current liabilities is zero - cannot calculate ratio")
        return create_calculation_result(
            metric_name="Cash Ratio",
            value=None,
            formula=METRIC_FORMULAS.get("cash_ratio", "Cash and Equivalents / Current Liabilities"),
            inputs={"cash_and_equivalents": float(cash), "current_liabilities": float(cl)},
            steps=steps,
            category=MetricCategory.LIQUIDITY,
            warnings=warnings,
        )
    
    ratio = safe_divide(cash, cl, precision=4)
    steps.append(f"Cash Ratio = Cash / Current Liabilities = {cash:,.2f} / {cl:,.2f} = {ratio:.4f}")
    
    # Interpretation
    if ratio < Decimal("0.2"):
        warnings.append("Low cash ratio - limited ability to pay obligations from cash alone")
    elif ratio > Decimal("1.0"):
        warnings.append("High cash ratio may indicate excess cash not being deployed")
        steps.append("Consider whether excess cash could be invested for better returns")
    
    return create_calculation_result(
        metric_name="Cash Ratio",
        value=ratio,
        formula=METRIC_FORMULAS.get("cash_ratio", "Cash and Equivalents / Current Liabilities"),
        inputs={"cash_and_equivalents": float(cash), "current_liabilities": float(cl)},
        steps=steps,
        category=MetricCategory.LIQUIDITY,
        warnings=warnings,
    )


def calculate_working_capital(
    current_assets: Decimal | float | int,
    current_liabilities: Decimal | float | int,
    currency: str = "SGD",
) -> CalculationResult:
    """
    Calculate Working Capital.
    
    Formula: Current Assets - Current Liabilities
    
    This is an absolute amount, not a ratio.
    
    Args:
        current_assets: Total current assets
        current_liabilities: Total current liabilities
        currency: Currency code for the result
        
    Returns:
        CalculationResult with working capital amount
    """
    steps = []
    warnings = []
    
    ca = to_decimal(current_assets)
    cl = to_decimal(current_liabilities)
    
    steps.append(f"Current Assets = {ca:,.2f}")
    steps.append(f"Current Liabilities = {cl:,.2f}")
    
    working_capital = ca - cl
    steps.append(f"Working Capital = Current Assets - Current Liabilities = {ca:,.2f} - {cl:,.2f} = {working_capital:,.2f}")
    
    # Interpretation
    if working_capital < 0:
        warnings.append("Negative working capital indicates current liabilities exceed current assets")
        warnings.append("This may indicate liquidity problems or aggressive working capital management")
    elif working_capital == 0:
        warnings.append("Zero working capital - current assets exactly match current liabilities")
    
    result = create_calculation_result(
        metric_name="Working Capital",
        value=round_decimal(working_capital, 2),
        formula=METRIC_FORMULAS.get("working_capital", "Current Assets - Current Liabilities"),
        inputs={"current_assets": float(ca), "current_liabilities": float(cl)},
        steps=steps,
        category=MetricCategory.LIQUIDITY,
        warnings=warnings,
    )
    
    # Override unit to currency
    result.unit = MetricUnit.CURRENCY
    
    return result


def calculate_all_liquidity_metrics(
    balance_sheet: BalanceSheetData,
) -> MetricCollection:
    """
    Calculate all liquidity metrics.
    
    Args:
        balance_sheet: Balance sheet data
        
    Returns:
        MetricCollection with all liquidity metrics
    """
    collection = MetricCollection(
        category=MetricCategory.LIQUIDITY,
        period=balance_sheet.period,
    )
    
    # Current Ratio
    current = calculate_current_ratio(
        current_assets=balance_sheet.calculated_current_assets,
        current_liabilities=balance_sheet.calculated_current_liabilities,
    )
    collection.add_metric(current)
    
    # Quick Ratio
    quick = calculate_quick_ratio(
        current_assets=balance_sheet.calculated_current_assets,
        inventory=balance_sheet.inventory,
        current_liabilities=balance_sheet.calculated_current_liabilities,
    )
    collection.add_metric(quick)
    
    # Cash Ratio
    cash = calculate_cash_ratio(
        cash_and_equivalents=balance_sheet.cash_and_equivalents,
        current_liabilities=balance_sheet.calculated_current_liabilities,
    )
    collection.add_metric(cash)
    
    # Working Capital
    wc = calculate_working_capital(
        current_assets=balance_sheet.calculated_current_assets,
        current_liabilities=balance_sheet.calculated_current_liabilities,
        currency=balance_sheet.currency,
    )
    collection.add_metric(wc)
    
    return collection


class LiquidityCalculator(BaseCalculator):
    """Class-based calculator for liquidity metrics."""
    
    def __init__(self):
        super().__init__(MetricCategory.LIQUIDITY)
    
    def calculate_all(
        self,
        statement_set: FinancialStatementSet,
        prior_statement_set: FinancialStatementSet | None = None,
    ) -> MetricCollection:
        """Calculate all liquidity metrics."""
        return calculate_all_liquidity_metrics(statement_set.balance_sheet)

```

# finanalyst_tools/calculations/profitability.py
```5
# finanalyst_tools/calculations/profitability.py
"""
Profitability ratio calculations.

Provides calculations for:
- Gross Profit Margin
- Operating Profit Margin
- Net Profit Margin
- EBITDA Margin
- Return on Assets (ROA)
- Return on Equity (ROE)
- Return on Capital Employed (ROCE)
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from finanalyst_tools.config import METRIC_FORMULAS, PlausibilityRanges
from finanalyst_tools.models.analysis_results import (
    CalculationResult,
    MetricUnit,
    MetricCategory,
    MetricCollection,
)
from finanalyst_tools.models.financial_statements import (
    IncomeStatementData,
    BalanceSheetData,
    FinancialStatementSet,
)
from finanalyst_tools.calculations.base import (
    BaseCalculator,
    create_calculation_result,
    extract_value,
)
from finanalyst_tools.utils.math_ops import (
    to_decimal,
    safe_divide,
    round_decimal,
    calculate_percentage,
    calculate_average,
    is_effectively_zero,
)


# ============================================================================
# STANDALONE CALCULATION FUNCTIONS
# ============================================================================

def calculate_gross_profit_margin(
    revenue: Decimal | float | int,
    cost_of_goods_sold: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Gross Profit Margin.
    
    Formula: (Revenue - COGS) / Revenue × 100
    
    Args:
        revenue: Total revenue
        cost_of_goods_sold: Cost of goods sold
        
    Returns:
        CalculationResult with gross profit margin percentage
    """
    steps = []
    warnings = []
    
    # Convert inputs
    rev = to_decimal(revenue)
    cogs = to_decimal(cost_of_goods_sold)
    
    steps.append(f"Revenue = {rev:,.2f}")
    steps.append(f"COGS = {cogs:,.2f}")
    
    # Calculate gross profit
    gross_profit = rev - cogs
    steps.append(f"Gross Profit = Revenue - COGS = {rev:,.2f} - {cogs:,.2f} = {gross_profit:,.2f}")
    
    # Check for zero revenue
    if is_effectively_zero(rev):
        warnings.append("Revenue is zero, cannot calculate margin")
        return create_calculation_result(
            metric_name="Gross Profit Margin",
            value=None,
            formula=METRIC_FORMULAS.get("gross_profit_margin", "(Revenue - COGS) / Revenue × 100"),
            inputs={"revenue": float(rev), "cost_of_goods_sold": float(cogs)},
            steps=steps,
            category=MetricCategory.PROFITABILITY,
            warnings=warnings,
        )
    
    # Calculate margin
    margin = calculate_percentage(gross_profit, rev)
    steps.append(f"Gross Profit Margin = (Gross Profit / Revenue) × 100 = ({gross_profit:,.2f} / {rev:,.2f}) × 100 = {margin:.2f}%")
    
    # Add contextual warnings
    if margin < Decimal("0"):
        warnings.append("Negative gross margin indicates selling below cost")
    elif margin > Decimal("90"):
        warnings.append("Very high gross margin - verify COGS is complete")
    
    return create_calculation_result(
        metric_name="Gross Profit Margin",
        value=margin,
        formula=METRIC_FORMULAS.get("gross_profit_margin", "(Revenue - COGS) / Revenue × 100"),
        inputs={"revenue": float(rev), "cost_of_goods_sold": float(cogs), "gross_profit": float(gross_profit)},
        steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
    )


def calculate_operating_profit_margin(
    revenue: Decimal | float | int,
    cost_of_goods_sold: Decimal | float | int,
    operating_expenses: Decimal | float | int,
    marketing_expenses: Decimal | float | int | None = None,
    include_marketing_in_opex: bool = True,
) -> CalculationResult:
    """
    Calculate Operating Profit Margin.
    
    Formula: (Revenue - COGS - OpEx) / Revenue × 100
    
    Args:
        revenue: Total revenue
        cost_of_goods_sold: Cost of goods sold
        operating_expenses: Operating expenses
        marketing_expenses: Marketing expenses (optional, if tracked separately)
        include_marketing_in_opex: Whether marketing is already in OpEx
        
    Returns:
        CalculationResult with operating profit margin percentage
    """
    steps = []
    warnings = []
    
    # Convert inputs
    rev = to_decimal(revenue)
    cogs = to_decimal(cost_of_goods_sold)
    opex = to_decimal(operating_expenses)
    marketing = to_decimal(marketing_expenses) if marketing_expenses else Decimal("0")
    
    steps.append(f"Revenue = {rev:,.2f}")
    steps.append(f"COGS = {cogs:,.2f}")
    steps.append(f"Operating Expenses = {opex:,.2f}")
    
    # Handle marketing separately if needed
    total_opex = opex
    if marketing_expenses and not include_marketing_in_opex:
        total_opex = opex + marketing
        steps.append(f"Marketing Expenses = {marketing:,.2f} (added separately)")
        steps.append(f"Total Operating Expenses = {total_opex:,.2f}")
    
    # Calculate operating profit
    gross_profit = rev - cogs
    operating_profit = gross_profit - total_opex
    steps.append(f"Gross Profit = {gross_profit:,.2f}")
    steps.append(f"Operating Profit = Gross Profit - OpEx = {gross_profit:,.2f} - {total_opex:,.2f} = {operating_profit:,.2f}")
    
    # Check for zero revenue
    if is_effectively_zero(rev):
        warnings.append("Revenue is zero, cannot calculate margin")
        return create_calculation_result(
            metric_name="Operating Profit Margin",
            value=None,
            formula=METRIC_FORMULAS.get("operating_profit_margin", "(Revenue - COGS - OpEx) / Revenue × 100"),
            inputs={
                "revenue": float(rev),
                "cost_of_goods_sold": float(cogs),
                "operating_expenses": float(total_opex),
            },
            steps=steps,
            category=MetricCategory.PROFITABILITY,
            warnings=warnings,
        )
    
    # Calculate margin
    margin = calculate_percentage(operating_profit, rev)
    steps.append(f"Operating Profit Margin = (Operating Profit / Revenue) × 100 = {margin:.2f}%")
    
    # Contextual warnings
    if margin < Decimal("-50"):
        warnings.append("Severely negative operating margin - company may be in distress")
    
    return create_calculation_result(
        metric_name="Operating Profit Margin",
        value=margin,
        formula=METRIC_FORMULAS.get("operating_profit_margin", "(Revenue - COGS - OpEx) / Revenue × 100"),
        inputs={
            "revenue": float(rev),
            "cost_of_goods_sold": float(cogs),
            "operating_expenses": float(total_opex),
            "operating_profit": float(operating_profit),
        },
        steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
    )


def calculate_net_profit_margin(
    revenue: Decimal | float | int,
    net_income: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Net Profit Margin.
    
    Formula: Net Income / Revenue × 100
    
    Args:
        revenue: Total revenue
        net_income: Net income (after tax)
        
    Returns:
        CalculationResult with net profit margin percentage
    """
    steps = []
    warnings = []
    
    rev = to_decimal(revenue)
    ni = to_decimal(net_income)
    
    steps.append(f"Revenue = {rev:,.2f}")
    steps.append(f"Net Income = {ni:,.2f}")
    
    if is_effectively_zero(rev):
        warnings.append("Revenue is zero, cannot calculate margin")
        return create_calculation_result(
            metric_name="Net Profit Margin",
            value=None,
            formula=METRIC_FORMULAS.get("net_profit_margin", "Net Income / Revenue × 100"),
            inputs={"revenue": float(rev), "net_income": float(ni)},
            steps=steps,
            category=MetricCategory.PROFITABILITY,
            warnings=warnings,
        )
    
    margin = calculate_percentage(ni, rev)
    steps.append(f"Net Profit Margin = (Net Income / Revenue) × 100 = ({ni:,.2f} / {rev:,.2f}) × 100 = {margin:.2f}%")
    
    # Warnings
    if margin > Decimal("50"):
        warnings.append("Net margin >50% is unusual - verify data accuracy")
    if margin < Decimal("-100"):
        warnings.append("Net losses exceed revenue - severe financial distress")
    
    return create_calculation_result(
        metric_name="Net Profit Margin",
        value=margin,
        formula=METRIC_FORMULAS.get("net_profit_margin", "Net Income / Revenue × 100"),
        inputs={"revenue": float(rev), "net_income": float(ni)},
        steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
    )


def calculate_ebitda_margin(
    revenue: Decimal | float | int,
    ebitda: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate EBITDA Margin.
    
    Formula: EBITDA / Revenue × 100
    
    Args:
        revenue: Total revenue
        ebitda: Earnings Before Interest, Taxes, Depreciation, and Amortization
        
    Returns:
        CalculationResult with EBITDA margin percentage
    """
    steps = []
    warnings = []
    
    rev = to_decimal(revenue)
    ebitda_val = to_decimal(ebitda)
    
    steps.append(f"Revenue = {rev:,.2f}")
    steps.append(f"EBITDA = {ebitda_val:,.2f}")
    
    if is_effectively_zero(rev):
        warnings.append("Revenue is zero, cannot calculate margin")
        return create_calculation_result(
            metric_name="EBITDA Margin",
            value=None,
            formula=METRIC_FORMULAS.get("ebitda_margin", "EBITDA / Revenue × 100"),
            inputs={"revenue": float(rev), "ebitda": float(ebitda_val)},
            steps=steps,
            category=MetricCategory.PROFITABILITY,
            warnings=warnings,
        )
    
    margin = calculate_percentage(ebitda_val, rev)
    steps.append(f"EBITDA Margin = (EBITDA / Revenue) × 100 = ({ebitda_val:,.2f} / {rev:,.2f}) × 100 = {margin:.2f}%")
    
    return create_calculation_result(
        metric_name="EBITDA Margin",
        value=margin,
        formula=METRIC_FORMULAS.get("ebitda_margin", "EBITDA / Revenue × 100"),
        inputs={"revenue": float(rev), "ebitda": float(ebitda_val)},
        steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
    )


def calculate_return_on_assets(
    net_income: Decimal | float | int,
    total_assets_beginning: Decimal | float | int,
    total_assets_ending: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Return on Assets (ROA).
    
    Formula: Net Income / Average Total Assets × 100
    
    Args:
        net_income: Net income for the period
        total_assets_beginning: Total assets at beginning of period
        total_assets_ending: Total assets at end of period
        
    Returns:
        CalculationResult with ROA percentage
    """
    steps = []
    warnings = []
    
    ni = to_decimal(net_income)
    assets_begin = to_decimal(total_assets_beginning)
    assets_end = to_decimal(total_assets_ending)
    
    steps.append(f"Net Income = {ni:,.2f}")
    steps.append(f"Total Assets (Beginning) = {assets_begin:,.2f}")
    steps.append(f"Total Assets (Ending) = {assets_end:,.2f}")
    
    # Calculate average assets
    avg_assets = calculate_average(assets_begin, assets_end)
    steps.append(f"Average Total Assets = ({assets_begin:,.2f} + {assets_end:,.2f}) / 2 = {avg_assets:,.2f}")
    
    if is_effectively_zero(avg_assets):
        warnings.append("Average assets is zero, cannot calculate ROA")
        return create_calculation_result(
            metric_name="Return on Assets",
            value=None,
            formula=METRIC_FORMULAS.get("roa", "Net Income / Average Total Assets × 100"),
            inputs={
                "net_income": float(ni),
                "total_assets_beginning": float(assets_begin),
                "total_assets_ending": float(assets_end),
            },
            steps=steps,
            category=MetricCategory.PROFITABILITY,
            warnings=warnings,
        )
    
    roa = calculate_percentage(ni, avg_assets)
    steps.append(f"ROA = (Net Income / Average Assets) × 100 = ({ni:,.2f} / {avg_assets:,.2f}) × 100 = {roa:.2f}%")
    
    # Warnings
    if avg_assets != assets_end:
        steps.append("Note: Using average assets for more accurate return calculation")
    
    return create_calculation_result(
        metric_name="Return on Assets",
        value=roa,
        formula=METRIC_FORMULAS.get("roa", "Net Income / Average Total Assets × 100"),
        inputs={
            "net_income": float(ni),
            "total_assets_beginning": float(assets_begin),
            "total_assets_ending": float(assets_end),
            "average_total_assets": float(avg_assets),
        },
        steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
    )


def calculate_return_on_equity(
    net_income: Decimal | float | int,
    shareholders_equity_beginning: Decimal | float | int,
    shareholders_equity_ending: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Return on Equity (ROE).
    
    Formula: Net Income / Average Shareholders' Equity × 100
    
    Args:
        net_income: Net income for the period
        shareholders_equity_beginning: Equity at beginning of period
        shareholders_equity_ending: Equity at end of period
        
    Returns:
        CalculationResult with ROE percentage
    """
    steps = []
    warnings = []
    
    ni = to_decimal(net_income)
    equity_begin = to_decimal(shareholders_equity_beginning)
    equity_end = to_decimal(shareholders_equity_ending)
    
    steps.append(f"Net Income = {ni:,.2f}")
    steps.append(f"Shareholders' Equity (Beginning) = {equity_begin:,.2f}")
    steps.append(f"Shareholders' Equity (Ending) = {equity_end:,.2f}")
    
    # Calculate average equity
    avg_equity = calculate_average(equity_begin, equity_end)
    steps.append(f"Average Shareholders' Equity = ({equity_begin:,.2f} + {equity_end:,.2f}) / 2 = {avg_equity:,.2f}")
    
    # Check for negative equity
    if equity_end < 0:
        warnings.append("Negative shareholders' equity indicates accumulated losses exceed capital")
    
    if is_effectively_zero(avg_equity):
        warnings.append("Average equity is zero or near-zero, ROE calculation may be unreliable")
        return create_calculation_result(
            metric_name="Return on Equity",
            value=None,
            formula=METRIC_FORMULAS.get("roe", "Net Income / Average Shareholders' Equity × 100"),
            inputs={
                "net_income": float(ni),
                "shareholders_equity_beginning": float(equity_begin),
                "shareholders_equity_ending": float(equity_end),
            },
            steps=steps,
            category=MetricCategory.PROFITABILITY,
            warnings=warnings,
        )
    
    roe = calculate_percentage(ni, avg_equity)
    steps.append(f"ROE = (Net Income / Average Equity) × 100 = ({ni:,.2f} / {avg_equity:,.2f}) × 100 = {roe:.2f}%")
    
    return create_calculation_result(
        metric_name="Return on Equity",
        value=roe,
        formula=METRIC_FORMULAS.get("roe", "Net Income / Average Shareholders' Equity × 100"),
        inputs={
            "net_income": float(ni),
            "shareholders_equity_beginning": float(equity_begin),
            "shareholders_equity_ending": float(equity_end),
            "average_shareholders_equity": float(avg_equity),
        },
        steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
    )


def calculate_return_on_capital_employed(
    ebit: Decimal | float | int,
    total_assets: Decimal | float | int,
    current_liabilities: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Return on Capital Employed (ROCE).
    
    Formula: EBIT / (Total Assets - Current Liabilities) × 100
    
    Args:
        ebit: Earnings Before Interest and Taxes
        total_assets: Total assets
        current_liabilities: Current liabilities
        
    Returns:
        CalculationResult with ROCE percentage
    """
    steps = []
    warnings = []
    
    ebit_val = to_decimal(ebit)
    assets = to_decimal(total_assets)
    cl = to_decimal(current_liabilities)
    
    steps.append(f"EBIT = {ebit_val:,.2f}")
    steps.append(f"Total Assets = {assets:,.2f}")
    steps.append(f"Current Liabilities = {cl:,.2f}")
    
    # Calculate capital employed
    capital_employed = assets - cl
    steps.append(f"Capital Employed = Total Assets - Current Liabilities = {assets:,.2f} - {cl:,.2f} = {capital_employed:,.2f}")
    
    if is_effectively_zero(capital_employed):
        warnings.append("Capital employed is zero, cannot calculate ROCE")
        return create_calculation_result(
            metric_name="Return on Capital Employed",
            value=None,
            formula=METRIC_FORMULAS.get("roce", "EBIT / (Total Assets - Current Liabilities) × 100"),
            inputs={
                "ebit": float(ebit_val),
                "total_assets": float(assets),
                "current_liabilities": float(cl),
            },
            steps=steps,
            category=MetricCategory.PROFITABILITY,
            warnings=warnings,
        )
    
    roce = calculate_percentage(ebit_val, capital_employed)
    steps.append(f"ROCE = (EBIT / Capital Employed) × 100 = ({ebit_val:,.2f} / {capital_employed:,.2f}) × 100 = {roce:.2f}%")
    
    return create_calculation_result(
        metric_name="Return on Capital Employed",
        value=roce,
        formula=METRIC_FORMULAS.get("roce", "EBIT / (Total Assets - Current Liabilities) × 100"),
        inputs={
            "ebit": float(ebit_val),
            "total_assets": float(assets),
            "current_liabilities": float(cl),
            "capital_employed": float(capital_employed),
        },
        steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
    )


def calculate_all_profitability_metrics(
    income_statement: IncomeStatementData,
    balance_sheet: BalanceSheetData,
    prior_balance_sheet: BalanceSheetData | None = None,
) -> MetricCollection:
    """
    Calculate all profitability metrics.
    
    Args:
        income_statement: Current period income statement
        balance_sheet: Current period balance sheet
        prior_balance_sheet: Prior period balance sheet (for averages)
        
    Returns:
        MetricCollection with all profitability metrics
    """
    collection = MetricCollection(
        category=MetricCategory.PROFITABILITY,
        period=income_statement.period,
    )
    
    # Use current balance sheet for both if no prior available
    if prior_balance_sheet is None:
        prior_balance_sheet = balance_sheet
    
    # Gross Profit Margin
    gpm = calculate_gross_profit_margin(
        revenue=income_statement.total_revenue,
        cost_of_goods_sold=income_statement.cost_of_goods_sold,
    )
    collection.add_metric(gpm)
    
    # Operating Profit Margin
    opm = calculate_operating_profit_margin(
        revenue=income_statement.total_revenue,
        cost_of_goods_sold=income_statement.cost_of_goods_sold,
        operating_expenses=income_statement.total_operating_expenses,
    )
    collection.add_metric(opm)
    
    # Net Profit Margin
    npm = calculate_net_profit_margin(
        revenue=income_statement.total_revenue,
        net_income=income_statement.calculated_net_income,
    )
    collection.add_metric(npm)
    
    # EBITDA Margin
    ebitda_m = calculate_ebitda_margin(
        revenue=income_statement.total_revenue,
        ebitda=income_statement.ebitda,
    )
    collection.add_metric(ebitda_m)
    
    # ROA
    roa = calculate_return_on_assets(
        net_income=income_statement.calculated_net_income,
        total_assets_beginning=prior_balance_sheet.calculated_total_assets,
        total_assets_ending=balance_sheet.calculated_total_assets,
    )
    collection.add_metric(roa)
    
    # ROE
    roe = calculate_return_on_equity(
        net_income=income_statement.calculated_net_income,
        shareholders_equity_beginning=prior_balance_sheet.calculated_shareholders_equity,
        shareholders_equity_ending=balance_sheet.calculated_shareholders_equity,
    )
    collection.add_metric(roe)
    
    # ROCE
    roce = calculate_return_on_capital_employed(
        ebit=income_statement.ebit,
        total_assets=balance_sheet.calculated_total_assets,
        current_liabilities=balance_sheet.calculated_current_liabilities,
    )
    collection.add_metric(roce)
    
    return collection


class ProfitabilityCalculator(BaseCalculator):
    """Class-based calculator for profitability metrics."""
    
    def __init__(self):
        super().__init__(MetricCategory.PROFITABILITY)
    
    def calculate_all(
        self,
        statement_set: FinancialStatementSet,
        prior_statement_set: FinancialStatementSet | None = None,
    ) -> MetricCollection:
        """Calculate all profitability metrics."""
        prior_bs = prior_statement_set.balance_sheet if prior_statement_set else None
        
        return calculate_all_profitability_metrics(
            income_statement=statement_set.income_statement,
            balance_sheet=statement_set.balance_sheet,
            prior_balance_sheet=prior_bs,
        )

```

# finanalyst_tools/calculations/base.py
```5
# finanalyst_tools/calculations/base.py
"""
Base classes and utilities for financial calculations.

Provides:
- BaseCalculator abstract class for consistent calculation patterns
- Factory function for creating CalculationResult objects
- Helper functions for extracting values from financial statements
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Callable

from finanalyst_tools.config import (
    DECIMAL_PLACES,
    PlausibilityRanges,
    METRIC_FORMULAS,
    METRIC_UNITS,
)
from finanalyst_tools.models.analysis_results import (
    CalculationResult,
    MetricUnit,
    MetricCategory,
    MetricCollection,
)
from finanalyst_tools.models.financial_statements import (
    IncomeStatementData,
    BalanceSheetData,
    CashFlowStatementData,
    FinancialStatementSet,
)
from finanalyst_tools.utils.math_ops import (
    to_decimal,
    safe_divide,
    round_decimal,
    calculate_percentage,
    calculate_average,
)


def get_metric_unit(metric_name: str) -> MetricUnit:
    """
    Get the appropriate unit for a metric.
    
    Args:
        metric_name: Name of the metric
        
    Returns:
        MetricUnit enum value
    """
    unit_str = METRIC_UNITS.get(metric_name.lower(), "ratio")
    unit_map = {
        "percentage": MetricUnit.PERCENTAGE,
        "ratio": MetricUnit.RATIO,
        "currency": MetricUnit.CURRENCY,
        "days": MetricUnit.DAYS,
        "count": MetricUnit.COUNT,
        "times": MetricUnit.TIMES,
    }
    return unit_map.get(unit_str, MetricUnit.RATIO)


def get_metric_formula(metric_name: str) -> str:
    """
    Get the formula for a metric.
    
    Args:
        metric_name: Name of the metric
        
    Returns:
        Formula string
    """
    return METRIC_FORMULAS.get(metric_name.lower(), "N/A")


def create_calculation_result(
    metric_name: str,
    value: Decimal | None,
    formula: str,
    inputs: dict[str, Any],
    steps: list[str],
    category: MetricCategory | None = None,
    warnings: list[str] | None = None,
) -> CalculationResult:
    """
    Factory function to create a CalculationResult with plausibility check.
    
    Args:
        metric_name: Name of the metric
        value: Calculated value (or None if calculation failed)
        formula: Formula used for calculation
        inputs: Dictionary of input values used
        steps: List of calculation steps for audit trail
        category: Metric category
        warnings: List of warning messages
        
    Returns:
        Complete CalculationResult with plausibility assessment
    """
    # Get plausibility range
    plausibility_range = PlausibilityRanges.get_range(metric_name)
    
    # Check plausibility
    is_plausible = True
    if value is not None and plausibility_range is not None:
        float_value = float(value)
        is_plausible = plausibility_range[0] <= float_value <= plausibility_range[1]
    
    # Get unit
    unit = get_metric_unit(metric_name)
    
    # Build result
    result = CalculationResult(
        metric_name=metric_name,
        value=value,
        unit=unit,
        formula=formula,
        inputs=inputs,
        calculation_steps=steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=warnings or [],
        category=category,
    )
    
    # Add plausibility warning if needed
    if not is_plausible and plausibility_range:
        result.add_warning(
            f"Value {float(value):.2f} is outside typical range "
            f"({plausibility_range[0]:.1f} to {plausibility_range[1]:.1f})"
        )
    
    return result


def extract_value(
    data: dict[str, Any] | IncomeStatementData | BalanceSheetData | CashFlowStatementData,
    field_name: str,
    default: Decimal | None = None,
) -> Decimal:
    """
    Extract a value from financial data, handling both dict and model inputs.
    
    Args:
        data: Financial data (dict or Pydantic model)
        field_name: Name of the field to extract
        default: Default value if field not found
        
    Returns:
        Decimal value
    """
    if default is None:
        default = Decimal("0")
    
    # Convert model to dict if needed
    if hasattr(data, "model_dump"):
        data_dict = data.model_dump(by_alias=False)
    elif hasattr(data, "__dict__"):
        data_dict = data.__dict__
    else:
        data_dict = data
    
    # Try to get the value
    value = data_dict.get(field_name)
    
    if value is None:
        return default
    
    return to_decimal(value, default=default)


class BaseCalculator(ABC):
    """
    Abstract base class for financial calculators.
    
    Provides common functionality:
    - Consistent result creation
    - Step logging
    - Plausibility checking
    - Warning accumulation
    """
    
    def __init__(self, category: MetricCategory):
        """
        Initialize the calculator.
        
        Args:
            category: The category of metrics this calculator produces
        """
        self.category = category
        self._current_steps: list[str] = []
        self._current_warnings: list[str] = []
        self._current_inputs: dict[str, Any] = {}
    
    def _reset(self) -> None:
        """Reset calculation state for a new calculation."""
        self._current_steps = []
        self._current_warnings = []
        self._current_inputs = {}
    
    def _add_step(self, step: str) -> None:
        """Add a calculation step."""
        step_num = len(self._current_steps) + 1
        self._current_steps.append(f"Step {step_num}: {step}")
    
    def _add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self._current_warnings.append(warning)
    
    def _record_input(self, name: str, value: Any) -> None:
        """Record an input value."""
        if isinstance(value, Decimal):
            self._current_inputs[name] = float(value)
        else:
            self._current_inputs[name] = value
    
    def _create_result(
        self,
        metric_name: str,
        value: Decimal | None,
        formula: str,
    ) -> CalculationResult:
        """Create a calculation result with current state."""
        return create_calculation_result(
            metric_name=metric_name,
            value=value,
            formula=formula,
            inputs=self._current_inputs.copy(),
            steps=self._current_steps.copy(),
            category=self.category,
            warnings=self._current_warnings.copy(),
        )
    
    @abstractmethod
    def calculate_all(
        self,
        statement_set: FinancialStatementSet,
        prior_statement_set: FinancialStatementSet | None = None,
    ) -> MetricCollection:
        """
        Calculate all metrics for this category.
        
        Args:
            statement_set: Current period financial statements
            prior_statement_set: Prior period statements (for averages)
            
        Returns:
            MetricCollection with all calculated metrics
        """
        pass

```

# finanalyst_tools/calculations/__init__.py
```5
# finanalyst_tools/calculations/__init__.py
"""
Financial calculation functions for FinAnalyst-Pro Agent Tools.

This package provides calculation functions for:
- Profitability ratios (margins, ROA, ROE, ROCE)
- Liquidity ratios (current, quick, cash)
- Solvency ratios (debt ratios, interest coverage) [Phase 2]
- Efficiency ratios (turnover, days outstanding) [Phase 2]

All calculations:
- Use Decimal precision for accuracy
- Return CalculationResult with full audit trail
- Include plausibility checking
- Handle edge cases gracefully
"""

from finanalyst_tools.calculations.base import (
    BaseCalculator,
    create_calculation_result,
    extract_value,
)

from finanalyst_tools.calculations.profitability import (
    calculate_gross_profit_margin,
    calculate_operating_profit_margin,
    calculate_net_profit_margin,
    calculate_ebitda_margin,
    calculate_return_on_assets,
    calculate_return_on_equity,
    calculate_return_on_capital_employed,
    calculate_all_profitability_metrics,
    ProfitabilityCalculator,
)

from finanalyst_tools.calculations.liquidity import (
    calculate_current_ratio,
    calculate_quick_ratio,
    calculate_cash_ratio,
    calculate_working_capital,
    calculate_all_liquidity_metrics,
    LiquidityCalculator,
)


__all__ = [
    # Base
    "BaseCalculator",
    "create_calculation_result",
    "extract_value",
    # Profitability
    "calculate_gross_profit_margin",
    "calculate_operating_profit_margin",
    "calculate_net_profit_margin",
    "calculate_ebitda_margin",
    "calculate_return_on_assets",
    "calculate_return_on_equity",
    "calculate_return_on_capital_employed",
    "calculate_all_profitability_metrics",
    "ProfitabilityCalculator",
    # Liquidity
    "calculate_current_ratio",
    "calculate_quick_ratio",
    "calculate_cash_ratio",
    "calculate_working_capital",
    "calculate_all_liquidity_metrics",
    "LiquidityCalculator",
]

```

