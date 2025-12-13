# finanalyst_tools/tool_registry.py
"""
Central registry of all tools with metadata for LLM function calling.

Provides:
- Tool definitions with parameters and descriptions
- OpenAI/Anthropic schema generation
- Tool discovery and listing
- Comprehensive parameter validation and type coercion
- Seamless integration with validation and calculation systems
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, cast
import json
from decimal import Decimal, InvalidOperation
import math

from finanalyst_tools.models.analysis_results import CalculationResult
from finanalyst_tools.models.validation import ValidationResult, ValidationIssue, ValidationSeverity
from finanalyst_tools.exceptions import ToolExecutionError, ToolParameterError
from finanalyst_tools.config import METRIC_FORMULAS
from finanalyst_tools.validation.utils import exception_to_validation_result, result_to_reasoning_block


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid numeric constant: {value}")


def _normalize_nested_numbers(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("Non-finite Decimal is not allowed")
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite float is not allowed")
        return Decimal(str(value))

    if isinstance(value, dict):
        return {k: _normalize_nested_numbers(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_normalize_nested_numbers(v) for v in value]

    if isinstance(value, tuple):
        return tuple(_normalize_nested_numbers(v) for v in value)

    return value


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
        if self.default is not None:
            schema["default"] = self.default
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
    expose_to_llm: bool = True
    
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
                    "default": p.default,
                    "enum": p.enum,
                }
                for p in self.parameters
            ],
            "returns": self.returns,
            "example": self.example,
        }

    def _validate_and_coerce_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}

        for param in self.parameters:
            if param.required and param.name not in parameters:
                raise ToolParameterError(
                    tool_name=self.name,
                    parameter_name=param.name,
                    message="Required parameter is missing",
                    expected_type=param.type,
                )

        for param in self.parameters:
            if param.name in parameters:
                result[param.name] = self._coerce_parameter(param, parameters[param.name])
            elif param.default is not None:
                result[param.name] = param.default

        return result

    def _coerce_parameter(self, param: ToolParameter, value: Any) -> Any:
        if value is None:
            if param.required:
                raise ToolParameterError(
                    tool_name=self.name,
                    parameter_name=param.name,
                    message="Value cannot be None",
                    expected_type=param.type,
                )
            return param.default

        try:
            if param.type == "number":
                if isinstance(value, Decimal):
                    coerced = value
                elif isinstance(value, float):
                    if not math.isfinite(value):
                        raise ValueError("Non-finite float is not allowed")
                    coerced = Decimal(str(value))
                elif isinstance(value, int):
                    coerced = Decimal(value)
                elif isinstance(value, str):
                    coerced = Decimal(value)
                else:
                    raise ValueError(f"Cannot convert {type(value).__name__} to number")

                if not coerced.is_finite():
                    raise ValueError("Non-finite Decimal is not allowed")

            elif param.type == "integer":
                coerced = int(value)

            elif param.type == "boolean":
                if isinstance(value, bool):
                    coerced = value
                elif isinstance(value, str):
                    coerced = value.lower() in ("true", "1", "yes")
                else:
                    coerced = bool(value)

            elif param.type == "string":
                coerced = str(value)

            elif param.type == "object":
                if isinstance(value, dict):
                    coerced = value
                elif isinstance(value, str):
                    coerced = json.loads(value, parse_float=Decimal, parse_constant=_reject_json_constant)
                else:
                    raise ValueError("Expected object/dictionary")

                if not isinstance(coerced, dict):
                    raise ValueError("Expected object/dictionary")
                coerced = _normalize_nested_numbers(coerced)

            elif param.type == "array":
                if isinstance(value, list):
                    coerced = value
                elif isinstance(value, str):
                    coerced = json.loads(value, parse_float=Decimal, parse_constant=_reject_json_constant)
                else:
                    raise ValueError("Expected array/list")

                if not isinstance(coerced, list):
                    raise ValueError("Expected array/list")
                coerced = _normalize_nested_numbers(coerced)

            else:
                coerced = value

            if param.enum is not None and coerced not in param.enum:
                raise ValueError(f"Value must be one of: {', '.join(param.enum)}")

            return coerced

        except (ValueError, InvalidOperation, json.JSONDecodeError) as e:
            raise ToolParameterError(
                tool_name=self.name,
                parameter_name=param.name,
                message=f"Cannot convert to {param.type}: {str(e)}",
                expected_type=param.type,
                actual_value=value,
            )

    def execute(self, **kwargs: Any) -> str:
        """
        Execute the tool function with proper error handling and return formatting.
        
        This is the key integration point that ensures all tools return properly
        formatted reasoning blocks for LLM consumption.
        
        Args:
            **kwargs: Parameters to pass to the tool function
            
        Returns:
            Formatted reasoning block string
        """
        try:
            if self.function is None:
                raise ToolExecutionError(
                    tool_name=self.name,
                    original_error=ValueError("Tool function not defined"),
                    parameters=kwargs
                )

            validated_kwargs = self._validate_and_coerce_parameters(kwargs)
            
            # Execute the function
            result = self.function(**validated_kwargs)
            
            # Handle different return types
            if isinstance(result, CalculationResult):
                # This is the expected return type for calculation tools
                return result.to_reasoning_block()
            elif isinstance(result, ValidationResult):
                # Handle validation results
                return result_to_reasoning_block(result)
            elif isinstance(result, dict):
                # Handle dictionary results (convert to JSON)
                return json.dumps(result, indent=2)
            elif isinstance(result, str):
                # Return strings directly
                return result
            else:
                # Convert other types to string representation
                return str(result)
                
        except Exception as e:
            # Convert any exception to a proper validation result
            validation_result = exception_to_validation_result(
                e,
                field=self.name,
                context=f"tool execution: {self.name}"
            )
            return result_to_reasoning_block(validation_result)


class ToolRegistry:
    """
    Central registry for all available tools.
    
    Provides:
    - Tool registration
    - Tool lookup by name
    - Tool listing by category
    - Schema generation for LLM integration
    - Comprehensive parameter validation
    - Execution with proper error handling and formatting
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
    
    def list_tool_names(
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
        tools = [t for t in self.list_tools() if t.expose_to_llm]
        
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
        tools = [t for t in self.list_tools() if t.expose_to_llm]
        
        if categories:
            tools = [t for t in tools if t.category in categories]
        
        return [t.to_anthropic_schema() for t in tools]
    
    def execute_tool(self, tool_name: str, **kwargs: Any) -> str:
        """
        Execute a tool by name with the given parameters.
        
        This is the main entry point for tool execution that ensures proper
        formatting and error handling.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Parameters to pass to the tool
            
        Returns:
            Formatted reasoning block string
            
        Raises:
            ToolNotFoundError: If tool doesn't exist
            ToolParameterError: If parameters are invalid
        """
        tool = self.get(tool_name)
        if tool is None:
            from finanalyst_tools.exceptions import ToolNotFoundError
            raise ToolNotFoundError(
                tool_name=tool_name,
                available_tools=self.list_tool_names()
            )
        
        return tool.execute(**kwargs)
    
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
                    if param.enum:
                        lines.append(f"  - Allowed values: {', '.join(param.enum)}")
                    if param.default is not None:
                        lines.append(f"  - Default: {param.default}")
                lines.append("")
            
            lines.append(f"**Returns:** {tool.returns}")
            if tool.example:
                lines.append(f"**Example:** `{tool.example}`")
            lines.append("")
        
        return "\n".join(lines)
    
    def _register_all_tools(self) -> None:
        """Register all Phase 1 tools with proper integration."""
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

        def analyze_financials(
            statement_set: dict[str, Any] | "FinancialStatementSet",
            prior_statement_set: dict[str, Any] | "FinancialStatementSet" | None = None,
            analysis_type: str = "comprehensive",
            include_trends: bool = False,
            currency: str = "SGD",
            report_format: str = "markdown",
            include_audit_trail: bool = True,
        ) -> str:
            from finanalyst_tools.models.financial_statements import FinancialStatementSet
            from finanalyst_tools.orchestration.pipeline import AnalysisPipeline, AnalysisRequest
            from finanalyst_tools.orchestration.report_generator import (
                generate_financial_report,
                ReportFormat,
            )

            if isinstance(statement_set, FinancialStatementSet):
                parsed_statement_set = statement_set
            else:
                parsed_statement_set = FinancialStatementSet.model_validate(statement_set)

            parsed_prior_statement_set: FinancialStatementSet | None
            if prior_statement_set is None:
                parsed_prior_statement_set = None
            elif isinstance(prior_statement_set, FinancialStatementSet):
                parsed_prior_statement_set = prior_statement_set
            else:
                parsed_prior_statement_set = FinancialStatementSet.model_validate(prior_statement_set)

            request = AnalysisRequest(
                statement_set=parsed_statement_set,
                prior_statement_set=parsed_prior_statement_set,
                analysis_type=analysis_type,
                include_trends=include_trends,
                currency=currency,
            )

            result = AnalysisPipeline().execute(request)

            fmt = report_format.lower().strip()
            if fmt == "json":
                return result.to_json()

            return generate_financial_report(
                result,
                format=ReportFormat.MARKDOWN,
                include_audit_trail=include_audit_trail,
            )

        # ─────────────────────────────────────────────────────────────────
        # VALIDATION TOOLS
        # ─────────────────────────────────────────────────────────────────
        
        self.register(ToolDefinition(
            name="analyze_financials",
            description="Run the full 5-phase analysis pipeline and return a formatted report",
            category=ToolCategory.ANALYSIS,
            parameters=[
                ToolParameter("statement_set", "object", "FinancialStatementSet payload (income_statement, balance_sheet, optional cash_flow_statement)"),
                ToolParameter("prior_statement_set", "object", "Prior-period FinancialStatementSet (optional)", required=False),
                ToolParameter(
                    "analysis_type",
                    "string",
                    "Type of analysis to run",
                    enum=["profitability", "liquidity", "solvency", "efficiency", "comprehensive"],
                    required=False,
                    default="comprehensive",
                ),
                ToolParameter("include_trends", "boolean", "Whether to include trend analysis", required=False, default=False),
                ToolParameter("currency", "string", "Reporting currency", required=False, default="SGD"),
                ToolParameter("report_format", "string", "Report format: markdown or json", required=False, enum=["markdown", "json"], default="markdown"),
                ToolParameter("include_audit_trail", "boolean", "Whether to include the calculation audit trail", required=False, default=True),
            ],
            returns="Formatted report string",
            function=analyze_financials,
            example='{"statement_set": {"income_statement": {"period": {"year": 2023}, "total_revenue": 1000000, "cost_of_goods_sold": 600000}, "balance_sheet": {"period": {"year": 2023}, "cash_and_equivalents": 100000}}, "analysis_type": "profitability"}',
            expose_to_llm=True,
        ))
        
        self.register(ToolDefinition(
            name="validate_financial_data",
            description="Validate that financial data is complete and properly formatted for analysis",
            category=ToolCategory.VALIDATION,
            parameters=[
                ToolParameter("income_statement", "object", "Income statement data", required=False),
                ToolParameter("balance_sheet", "object", "Balance sheet data", required=False),
                ToolParameter("cash_flow", "object", "Cash flow statement data", required=False),
                ToolParameter(
                    "analysis_type",
                    "string",
                    "Type of analysis to validate for",
                    enum=["profitability", "liquidity", "solvency", "efficiency", "comprehensive"],
                    required=True,
                ),
            ],
            returns="ValidationResult with any issues found",
            function=validate_financial_data_completeness,
            example='validate_financial_data(income_statement={"total_revenue": 1000000}, analysis_type="profitability")',
            expose_to_llm=True,
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
            returns="CalculationResult with gross profit margin percentage and calculation steps",
            example='{"revenue": 1000000, "cost_of_goods_sold": 600000}',
            function=calculate_gross_profit_margin,
            expose_to_llm=False,
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
            returns="CalculationResult with operating profit margin percentage and calculation steps",
            function=calculate_operating_profit_margin,
            example='{"revenue": 1000000, "cost_of_goods_sold": 600000, "operating_expenses": 200000}',
            expose_to_llm=False,
        ))
        
        self.register(ToolDefinition(
            name="calculate_net_profit_margin",
            description="Calculate Net Profit Margin: Net Income / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("revenue", "number", "Total revenue"),
                ToolParameter("net_income", "number", "Net income (profit after tax)"),
            ],
            returns="CalculationResult with net profit margin percentage and calculation steps",
            function=calculate_net_profit_margin,
            example='{"revenue": 1000000, "net_income": 100000}',
            expose_to_llm=False,
        ))
        
        self.register(ToolDefinition(
            name="calculate_ebitda_margin",
            description="Calculate EBITDA Margin: EBITDA / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("revenue", "number", "Total revenue"),
                ToolParameter("ebitda", "number", "Earnings Before Interest, Taxes, Depreciation, and Amortization"),
            ],
            returns="CalculationResult with EBITDA margin percentage and calculation steps",
            function=calculate_ebitda_margin,
            example='{"revenue": 1000000, "ebitda": 250000}',
            expose_to_llm=False,
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
            returns="CalculationResult with return on assets percentage and calculation steps",
            function=calculate_return_on_assets,
            example='{"net_income": 100000, "total_assets_beginning": 500000, "total_assets_ending": 550000}',
            expose_to_llm=False,
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
            returns="CalculationResult with return on equity percentage and calculation steps",
            function=calculate_return_on_equity,
            example='{"net_income": 100000, "shareholders_equity_beginning": 300000, "shareholders_equity_ending": 350000}',
            expose_to_llm=False,
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
            returns="CalculationResult with ROCE percentage and calculation steps",
            function=calculate_return_on_capital_employed,
            example='{"ebit": 200000, "total_assets": 800000, "current_liabilities": 250000}',
            expose_to_llm=False,
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
            returns="CalculationResult with current ratio and calculation steps",
            example='{"current_assets": 500000, "current_liabilities": 300000}',
            function=calculate_current_ratio,
            expose_to_llm=False,
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
            returns="CalculationResult with quick ratio and calculation steps",
            function=calculate_quick_ratio,
            example='{"current_assets": 500000, "inventory": 150000, "current_liabilities": 300000}',
            expose_to_llm=False,
        ))
        
        self.register(ToolDefinition(
            name="calculate_cash_ratio",
            description="Calculate Cash Ratio: Cash and Equivalents / Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter("cash_and_equivalents", "number", "Cash and cash equivalents"),
                ToolParameter("current_liabilities", "number", "Total current liabilities"),
            ],
            returns="CalculationResult with cash ratio and calculation steps",
            function=calculate_cash_ratio,
            example='{"cash_and_equivalents": 100000, "current_liabilities": 300000}',
            expose_to_llm=False,
        ))
        
        self.register(ToolDefinition(
            name="calculate_working_capital",
            description="Calculate Working Capital: Current Assets - Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter("current_assets", "number", "Total current assets"),
                ToolParameter("current_liabilities", "number", "Total current liabilities"),
            ],
            returns="CalculationResult with working capital amount and calculation steps",
            function=calculate_working_capital,
            example='{"current_assets": 500000, "current_liabilities": 300000, "currency": "SGD"}',
            expose_to_llm=False,
        ))


# Global singleton instance
TOOL_REGISTRY = ToolRegistry()
