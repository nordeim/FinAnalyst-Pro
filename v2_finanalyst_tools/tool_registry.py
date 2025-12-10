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
