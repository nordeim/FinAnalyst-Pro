# finanalyst_tools/tool_registry.py
"""
Central registry of all tools for LLM function calling.

Provides:
- Tool definitions with complete metadata
- OpenAI and Anthropic schema generation
- Tool discovery and filtering
- Parameter validation schemas
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
    CASH_FLOW = "cash_flow"
    RECONCILIATION = "reconciliation"
    FORMATTING = "formatting"
    UTILITY = "utility"
    ANALYSIS = "analysis"


@dataclass
class ToolParameter:
    """
    Definition of a tool parameter.
    
    Attributes:
        name: Parameter name
        type: Parameter type (string, number, integer, boolean, object, array)
        description: Human-readable description
        required: Whether the parameter is required
        default: Default value if not provided
        enum: List of allowed values (for string enums)
    """
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None
    
    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI parameter schema."""
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema
    
    def to_anthropic_schema(self) -> dict[str, Any]:
        """Convert to Anthropic parameter schema."""
        # Anthropic uses similar format to OpenAI
        return self.to_openai_schema()


@dataclass
class ToolDefinition:
    """
    Complete definition of a tool.
    
    Attributes:
        name: Tool name (function name for calling)
        description: Human-readable description
        category: Tool category
        parameters: List of parameters
        returns: Description of return value
        example: Example usage
        function: Reference to the actual function
    """
    name: str
    description: str
    category: ToolCategory
    parameters: list[ToolParameter]
    returns: str
    example: str = ""
    function: Callable | None = None
    
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
            Dictionary in OpenAI function format
        """
        properties = {}
        for param in self.parameters:
            properties[param.name] = param.to_openai_schema()
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": self.required_parameters,
                },
            },
        }
    
    def to_anthropic_schema(self) -> dict[str, Any]:
        """
        Convert to Anthropic tool use schema.
        
        Returns:
            Dictionary in Anthropic tool format
        """
        properties = {}
        for param in self.parameters:
            properties[param.name] = param.to_anthropic_schema()
        
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": self.required_parameters,
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
    - Tool registration and lookup
    - Filtering by category
    - Schema generation for LLM integration
    """
    
    def __init__(self):
        """Initialize the registry."""
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
        List all tools, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tool definitions
        """
        if category is None:
            return list(self._tools.values())
        return [t for t in self._tools.values() if t.category == category]
    
    def list_tool_names(
        self,
        category: ToolCategory | None = None,
    ) -> list[str]:
        """
        List tool names, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tool names
        """
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
            List of tool schemas
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
        Get tools in Anthropic tool use format.
        
        Args:
            categories: Optional list of categories to include
            
        Returns:
            List of tool schemas
        """
        tools = self._tools.values()
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
        
        # Group by category
        by_category: dict[ToolCategory, list[ToolDefinition]] = {}
        for tool in self._tools.values():
            if tool.category not in by_category:
                by_category[tool.category] = []
            by_category[tool.category].append(tool)
        
        for category in ToolCategory:
            if category in by_category:
                lines.append(f"\n## {category.value.title()}\n")
                for tool in by_category[category]:
                    lines.append(f"### {tool.name}")
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
        
        # Import calculation functions
        from finanalyst_tools.calculations.profitability import (
            calculate_gross_profit_margin,
            calculate_operating_profit_margin,
            calculate_net_profit_margin,
            calculate_ebitda_margin,
            calculate_return_on_assets,
            calculate_return_on_equity,
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
        
        # ================================================================
        # VALIDATION TOOLS
        # ================================================================
        
        self.register(ToolDefinition(
            name="validate_financial_data",
            description="Validate that financial data is complete and correct for a specific analysis type",
            category=ToolCategory.VALIDATION,
            parameters=[
                ToolParameter(
                    name="income_statement",
                    type="object",
                    description="Income statement data with revenue, COGS, expenses, and net income",
                    required=False,
                ),
                ToolParameter(
                    name="balance_sheet",
                    type="object",
                    description="Balance sheet data with assets, liabilities, and equity",
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
                    required=True,
                    enum=["profitability", "liquidity", "solvency", "efficiency", "comprehensive"],
                ),
            ],
            returns="ValidationResult with any issues found",
            example="validate_financial_data(income_statement={...}, balance_sheet={...}, analysis_type='profitability')",
            function=validate_financial_data_completeness,
        ))
        
        # ================================================================
        # PROFITABILITY TOOLS
        # ================================================================
        
        self.register(ToolDefinition(
            name="calculate_gross_profit_margin",
            description="Calculate gross profit margin: (Revenue - COGS) / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter(
                    name="revenue",
                    type="number",
                    description="Total revenue / net sales",
                    required=True,
                ),
                ToolParameter(
                    name="cost_of_goods_sold",
                    type="number",
                    description="Cost of goods sold / cost of sales",
                    required=True,
                ),
            ],
            returns="CalculationResult with gross profit margin percentage and calculation steps",
            example="calculate_gross_profit_margin(revenue=1000000, cost_of_goods_sold=600000)",
            function=calculate_gross_profit_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_operating_profit_margin",
            description="Calculate operating profit margin: (Revenue - COGS - OpEx) / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter(
                    name="revenue",
                    type="number",
                    description="Total revenue",
                    required=True,
                ),
                ToolParameter(
                    name="cost_of_goods_sold",
                    type="number",
                    description="Cost of goods sold",
                    required=True,
                ),
                ToolParameter(
                    name="operating_expenses",
                    type="number",
                    description="Total operating expenses",
                    required=True,
                ),
                ToolParameter(
                    name="marketing_expenses",
                    type="number",
                    description="Marketing expenses (if separate from OpEx)",
                    required=False,
                ),
            ],
            returns="CalculationResult with operating margin percentage and calculation steps",
            example="calculate_operating_profit_margin(revenue=1000000, cost_of_goods_sold=600000, operating_expenses=200000)",
            function=calculate_operating_profit_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_net_profit_margin",
            description="Calculate net profit margin: Net Income / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter(
                    name="revenue",
                    type="number",
                    description="Total revenue",
                    required=True,
                ),
                ToolParameter(
                    name="net_income",
                    type="number",
                    description="Net income (bottom line profit)",
                    required=True,
                ),
            ],
            returns="CalculationResult with net margin percentage and calculation steps",
            example="calculate_net_profit_margin(revenue=1000000, net_income=100000)",
            function=calculate_net_profit_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_ebitda_margin",
            description="Calculate EBITDA margin: EBITDA / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter(
                    name="revenue",
                    type="number",
                    description="Total revenue",
                    required=True,
                ),
                ToolParameter(
                    name="ebitda",
                    type="number",
                    description="EBITDA (Earnings Before Interest, Taxes, Depreciation, Amortization)",
                    required=True,
                ),
            ],
            returns="CalculationResult with EBITDA margin percentage",
            example="calculate_ebitda_margin(revenue=1000000, ebitda=250000)",
            function=calculate_ebitda_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_return_on_assets",
            description="Calculate ROA: Net Income / Average Total Assets × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter(
                    name="net_income",
                    type="number",
                    description="Net income for the period",
                    required=True,
                ),
                ToolParameter(
                    name="total_assets_beginning",
                    type="number",
                    description="Total assets at beginning of period",
                    required=True,
                ),
                ToolParameter(
                    name="total_assets_ending",
                    type="number",
                    description="Total assets at end of period",
                    required=True,
                ),
            ],
            returns="CalculationResult with ROA percentage",
            example="calculate_return_on_assets(net_income=100000, total_assets_beginning=500000, total_assets_ending=550000)",
            function=calculate_return_on_assets,
        ))
        
        self.register(ToolDefinition(
            name="calculate_return_on_equity",
            description="Calculate ROE: Net Income / Average Shareholders' Equity × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter(
                    name="net_income",
                    type="number",
                    description="Net income for the period",
                    required=True,
                ),
                ToolParameter(
                    name="equity_beginning",
                    type="number",
                    description="Shareholders' equity at beginning of period",
                    required=True,
                ),
                ToolParameter(
                    name="equity_ending",
                    type="number",
                    description="Shareholders' equity at end of period",
                    required=True,
                ),
            ],
            returns="CalculationResult with ROE percentage",
            example="calculate_return_on_equity(net_income=100000, equity_beginning=300000, equity_ending=350000)",
            function=calculate_return_on_equity,
        ))
        
        # ================================================================
        # LIQUIDITY TOOLS
        # ================================================================
        
        self.register(ToolDefinition(
            name="calculate_current_ratio",
            description="Calculate current ratio: Current Assets / Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter(
                    name="current_assets",
                    type="number",
                    description="Total current assets",
                    required=True,
                ),
                ToolParameter(
                    name="current_liabilities",
                    type="number",
                    description="Total current liabilities",
                    required=True,
                ),
            ],
            returns="CalculationResult with current ratio and interpretation",
            example="calculate_current_ratio(current_assets=500000, current_liabilities=300000)",
            function=calculate_current_ratio,
        ))
        
        self.register(ToolDefinition(
            name="calculate_quick_ratio",
            description="Calculate quick ratio (acid test): (Current Assets - Inventory) / Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter(
                    name="current_assets",
                    type="number",
                    description="Total current assets",
                    required=True,
                ),
                ToolParameter(
                    name="inventory",
                    type="number",
                    description="Inventory value",
                    required=True,
                ),
                ToolParameter(
                    name="current_liabilities",
                    type="number",
                    description="Total current liabilities",
                    required=True,
                ),
            ],
            returns="CalculationResult with quick ratio",
            example="calculate_quick_ratio(current_assets=500000, inventory=150000, current_liabilities=300000)",
            function=calculate_quick_ratio,
        ))
        
        self.register(ToolDefinition(
            name="calculate_cash_ratio",
            description="Calculate cash ratio: Cash and Equivalents / Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter(
                    name="cash_and_equivalents",
                    type="number",
                    description="Cash and cash equivalents",
                    required=True,
                ),
                ToolParameter(
                    name="current_liabilities",
                    type="number",
                    description="Total current liabilities",
                    required=True,
                ),
            ],
            returns="CalculationResult with cash ratio",
            example="calculate_cash_ratio(cash_and_equivalents=100000, current_liabilities=300000)",
            function=calculate_cash_ratio,
        ))
        
        self.register(ToolDefinition(
            name="calculate_working_capital",
            description="Calculate working capital: Current Assets - Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter(
                    name="current_assets",
                    type="number",
                    description="Total current assets",
                    required=True,
                ),
                ToolParameter(
                    name="current_liabilities",
                    type="number",
                    description="Total current liabilities",
                    required=True,
                ),
            ],
            returns="CalculationResult with working capital amount",
            example="calculate_working_capital(current_assets=500000, current_liabilities=300000)",
            function=calculate_working_capital,
        ))


# Global singleton instance
TOOL_REGISTRY = ToolRegistry()
