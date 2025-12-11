# finanalyst_tools/tool_registry.py
"""
Central registry of all tools with metadata for LLM function calling.

Provides:
- Tool definitions with parameters and descriptions
- OpenAI function calling schema generation
- Anthropic tool schema generation
- Tool discovery and lookup
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
    VALUATION = "valuation"
    TREND = "trend"
    FORECAST = "forecast"
    RECONCILIATION = "reconciliation"
    REPORTING = "reporting"
    UTILITY = "utility"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "number", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None
    
    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolDefinition:
    """Complete definition of a tool for LLM function calling."""
    name: str
    description: str
    category: ToolCategory
    parameters: list[ToolParameter]
    returns: str
    function: Callable[..., Any]
    example: str | None = None
    
    def to_openai_schema(self) -> dict[str, Any]:
        """
        Convert to OpenAI function calling schema.
        
        Returns:
            Dictionary in OpenAI function format
        """
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
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
        Convert to Anthropic tool schema.
        
        Returns:
            Dictionary in Anthropic tool format
        """
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
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
        """Convert to dictionary for documentation."""
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
    
    Provides methods for:
    - Registering tools
    - Looking up tools by name
    - Listing tools by category
    - Generating LLM-compatible schemas
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
            Tool definition or None if not found
        """
        return self._tools.get(name)
    
    def list_tools(
        self,
        category: ToolCategory | None = None,
    ) -> list[ToolDefinition]:
        """
        List all tools, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of matching tool definitions
        """
        if category is None:
            return list(self._tools.values())
        return [t for t in self._tools.values() if t.category == category]
    
    def get_tool_names(
        self,
        category: ToolCategory | None = None,
    ) -> list[str]:
        """Get list of tool names."""
        tools = self.list_tools(category)
        return [t.name for t in tools]
    
    def get_openai_tools(
        self,
        categories: list[ToolCategory] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get OpenAI function calling schemas for tools.
        
        Args:
            categories: Optional list of categories to include
            
        Returns:
            List of OpenAI function schemas
        """
        tools = self._tools.values()
        if categories:
            tools = [t for t in tools if t.category in categories]
        return [t.to_openai_schema() for t in tools]
    
    def get_anthropic_tools(
        self,
        categories: list[ToolCategory] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get Anthropic tool schemas.
        
        Args:
            categories: Optional list of categories to include
            
        Returns:
            List of Anthropic tool schemas
        """
        tools = self._tools.values()
        if categories:
            tools = [t for t in tools if t.category in categories]
        return [t.to_anthropic_schema() for t in tools]
    
    def get_tool_descriptions(self) -> str:
        """
        Get formatted descriptions of all tools for prompt inclusion.
        
        Returns:
            Markdown-formatted tool descriptions
        """
        lines = ["# Available Financial Analysis Tools\n"]
        
        # Group by category
        categories: dict[ToolCategory, list[ToolDefinition]] = {}
        for tool in self._tools.values():
            if tool.category not in categories:
                categories[tool.category] = []
            categories[tool.category].append(tool)
        
        for category in sorted(categories.keys(), key=lambda c: c.value):
            lines.append(f"\n## {category.value.title()}\n")
            for tool in sorted(categories[category], key=lambda t: t.name):
                lines.append(f"### `{tool.name}`")
                lines.append(f"{tool.description}\n")
                if tool.parameters:
                    lines.append("**Parameters:**")
                    for param in tool.parameters:
                        req = "(required)" if param.required else "(optional)"
                        lines.append(f"- `{param.name}` ({param.type}) {req}: {param.description}")
                lines.append(f"\n**Returns:** {tool.returns}\n")
        
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
        
        # Validation Tools
        self.register(ToolDefinition(
            name="validate_financial_data",
            description="Validate that financial data is complete and properly structured for the requested analysis type",
            category=ToolCategory.VALIDATION,
            parameters=[
                ToolParameter("income_statement", "object", "Income statement data", required=False),
                ToolParameter("balance_sheet", "object", "Balance sheet data", required=False),
                ToolParameter("cash_flow", "object", "Cash flow statement data", required=False),
                ToolParameter("analysis_type", "string", "Type of analysis to validate for", 
                            required=True, enum=["profitability", "liquidity", "solvency", "efficiency", "comprehensive"]),
            ],
            returns="ValidationResult with any issues found",
            function=validate_financial_data_completeness,
            example='validate_financial_data(income_statement={...}, balance_sheet={...}, analysis_type="profitability")',
        ))
        
        # Profitability Tools
        self.register(ToolDefinition(
            name="calculate_gross_profit_margin",
            description="Calculate Gross Profit Margin: (Revenue - COGS) / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("revenue", "number", "Total revenue or net sales"),
                ToolParameter("cost_of_goods_sold", "number", "Cost of goods sold"),
            ],
            returns="CalculationResult with margin percentage and audit trail",
            function=calculate_gross_profit_margin,
            example="calculate_gross_profit_margin(revenue=1000000, cost_of_goods_sold=600000)",
        ))
        
        self.register(ToolDefinition(
            name="calculate_operating_profit_margin",
            description="Calculate Operating Profit Margin: (Revenue - COGS - OpEx) / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("revenue", "number", "Total revenue"),
                ToolParameter("cost_of_goods_sold", "number", "Cost of goods sold"),
                ToolParameter("operating_expenses", "number", "Total operating expenses", required=False),
                ToolParameter("selling_general_admin", "number", "SG&A expenses", required=False),
                ToolParameter("marketing_expenses", "number", "Marketing expenses", required=False),
                ToolParameter("research_development", "number", "R&D expenses", required=False),
                ToolParameter("depreciation_amortization", "number", "D&A", required=False),
            ],
            returns="CalculationResult with margin percentage and audit trail",
            function=calculate_operating_profit_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_net_profit_margin",
            description="Calculate Net Profit Margin: Net Income / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("revenue", "number", "Total revenue"),
                ToolParameter("net_income", "number", "Net income / net profit"),
            ],
            returns="CalculationResult with margin percentage and audit trail",
            function=calculate_net_profit_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_ebitda_margin",
            description="Calculate EBITDA Margin: EBITDA / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("revenue", "number", "Total revenue"),
                ToolParameter("ebitda", "number", "EBITDA"),
            ],
            returns="CalculationResult with margin percentage and audit trail",
            function=calculate_ebitda_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_return_on_assets",
            description="Calculate ROA: Net Income / Average Total Assets × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("net_income", "number", "Net income for the period"),
                ToolParameter("total_assets_begin", "number", "Total assets at period start"),
                ToolParameter("total_assets_end", "number", "Total assets at period end", required=False),
            ],
            returns="CalculationResult with ROA percentage and audit trail",
            function=calculate_return_on_assets,
        ))
        
        self.register(ToolDefinition(
            name="calculate_return_on_equity",
            description="Calculate ROE: Net Income / Average Shareholders' Equity × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("net_income", "number", "Net income for the period"),
                ToolParameter("shareholders_equity_begin", "number", "Equity at period start"),
                ToolParameter("shareholders_equity_end", "number", "Equity at period end", required=False),
            ],
            returns="CalculationResult with ROE percentage and audit trail",
            function=calculate_return_on_equity,
        ))
        
        self.register(ToolDefinition(
            name="calculate_return_on_capital_employed",
            description="Calculate ROCE: EBIT / (Total Assets - Current Liabilities) × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("ebit", "number", "Earnings Before Interest and Taxes"),
                ToolParameter("total_assets", "number", "Total assets"),
                ToolParameter("current_liabilities", "number", "Current liabilities"),
            ],
            returns="CalculationResult with ROCE percentage and audit trail",
            function=calculate_return_on_capital_employed,
        ))
        
        # Liquidity Tools
        self.register(ToolDefinition(
            name="calculate_current_ratio",
            description="Calculate Current Ratio: Current Assets / Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter("current_assets", "number", "Total current assets"),
                ToolParameter("current_liabilities", "number", "Total current liabilities"),
            ],
            returns="CalculationResult with ratio value and interpretation",
            function=calculate_current_ratio,
        ))
        
        self.register(ToolDefinition(
            name="calculate_quick_ratio",
            description="Calculate Quick Ratio: (Current Assets - Inventory) / Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter("current_assets", "number", "Total current assets"),
                ToolParameter("inventory", "number", "Inventory value"),
                ToolParameter("current_liabilities", "number", "Total current liabilities"),
            ],
            returns="CalculationResult with ratio value and interpretation",
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
            returns="CalculationResult with ratio value and interpretation",
            function=calculate_cash_ratio,
        ))
        
        self.register(ToolDefinition(
            name="calculate_working_capital",
            description="Calculate Working Capital: Current Assets - Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter("current_assets", "number", "Total current assets"),
                ToolParameter("current_liabilities", "number", "Total current liabilities"),
                ToolParameter("currency", "string", "Currency code", required=False, default="SGD"),
            ],
            returns="CalculationResult with currency value",
            function=calculate_working_capital,
        ))


# Global singleton instance
TOOL_REGISTRY = ToolRegistry()
