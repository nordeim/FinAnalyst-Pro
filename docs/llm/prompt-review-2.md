# finanalyst_tools/tool_registry.py
```py
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
from finanalyst_tools.validation.utils import exception_to_validation_result


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
                return _validation_result_to_reasoning_block(result)
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
            return _validation_result_to_reasoning_block(validation_result)


def _validation_result_to_reasoning_block(result: ValidationResult) -> str:
    """
    Convert a ValidationResult to a formatted reasoning block.
    
    Args:
        result: Validation result to format
        
    Returns:
        Formatted markdown block
    """
    lines = [
        f"### Validation Result for {result.context.get('analysis_type', 'analysis')}",
        "",
        "**Summary**:",
        f"- Status: {'✅ Valid' if result.is_valid else '❌ Invalid'}",
        f"- Errors: {result.error_count}",
        f"- Warnings: {result.warning_count}",
        f"- Info: {result.info_count}",
        "",
    ]
    
    if not result.is_valid:
        lines.append("**Errors**:")
        for issue in result.issues:
            error_icon = "❌ " if issue.severity == ValidationSeverity.ERROR else "⚠️ "
            lines.append(f"  - {error_icon}{issue.field}: {issue.message}")
            if issue.actual_value is not None:
                lines.append(f"    Actual: {issue.actual_value}, Expected: {issue.expected or 'valid value'}")
            if issue.suggestion:
                lines.append(f"    Suggestion: {issue.suggestion}")
        lines.append("")
    
    if result.warning_count > 0:
        lines.append("**Warnings**:")
        for issue in result.warnings:
            lines.append(f"  - ⚠️ {issue.field}: {issue.message}")
            if issue.suggestion:
                lines.append(f"    Suggestion: {issue.suggestion}")
        lines.append("")
    
    if result.info_count > 0:
        lines.append("**Information**:")
        for issue in result.info:
            lines.append(f"  - ℹ️ {issue.field}: {issue.message}")
        lines.append("")
    
    lines.append("**Recommendation**:")
    if result.can_proceed:
        lines.append("✅ Analysis can proceed with the provided data.")
        if result.warning_count > 0:
            lines.append("⚠️ However, please review the warnings for potential data quality issues.")
    else:
        lines.append("❌ Analysis cannot proceed due to validation errors.")
        lines.append("Please correct the errors before continuing.")
    
    return "\n".join(lines)


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

```

# finanalyst_tools/utils/__init__.py
```py
# File: finanalyst_tools/utils/__init__.py
"""
Utility functions for FinAnalyst-Pro Agent Tools.

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
    clamp_value,
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
    "clamp_value",
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

```

# finanalyst_tools/utils/math_ops.py
```py
# File: finanalyst_tools/utils/math_ops.py
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
from typing import Sequence, Any
import math

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


# Type alias for numeric types
Numeric = int | float | Decimal


# ============================================================================
# TYPE CONVERSION
# ============================================================================

def to_decimal(
    value: Numeric | str | None,
    default: Decimal | None = None,
    precision: int | None = None,
) -> Decimal:
    """
    Safely convert a value to Decimal.
    
    Handles various input types and returns a default value
    for None or unconvertible inputs.
    
    Args:
        value: Value to convert (int, float, str, Decimal, or None)
        default: Default value if conversion fails (default: Decimal("0"))
        precision: Optional precision to round to
        
    Returns:
        Decimal representation of the value
        
    Examples:
        >>> to_decimal(100)
        Decimal('100')
        >>> to_decimal("123.45")
        Decimal('123.45')
        >>> to_decimal(None, default=Decimal("0"))
        Decimal('0')
    """
    if default is None:
        default = Decimal("0")
    
    if value is None:
        return default
    
    try:
        if isinstance(value, Decimal):
            result = value
        elif isinstance(value, float):
            # Use string conversion to avoid float precision issues
            result = Decimal(str(value))
        else:
            result = Decimal(str(value))
        
        if precision is not None:
            result = round_decimal(result, precision)
        
        return result
    except (InvalidOperation, ValueError, TypeError):
        return default


def is_effectively_zero(value: Numeric | None, threshold: float = ZERO_THRESHOLD) -> bool:
    """
    Check if a value is effectively zero (within threshold).
    
    Args:
        value: Value to check
        threshold: Threshold for zero comparison
        
    Returns:
        True if value is None or within threshold of zero
    """
    if value is None:
        return True
    return abs(float(value)) < threshold


# ============================================================================
# SAFE ARITHMETIC
# ============================================================================

def safe_divide(
    numerator: Numeric | None,
    denominator: Numeric | None,
    default: Decimal | None = None,
    precision: int = DECIMAL_PLACES["ratio"],
    raise_on_zero: bool = False,
) -> Decimal:
    """
    Safely divide two numbers with zero handling.
    
    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return if division is impossible (default: Decimal("0"))
        precision: Decimal places to round result to
        raise_on_zero: If True, raise DivisionByZeroError instead of returning default
        
    Returns:
        Result of division, or default if denominator is zero/None
        
    Raises:
        DivisionByZeroError: If raise_on_zero is True and denominator is zero
        
    Examples:
        >>> safe_divide(100, 4)
        Decimal('25.0000')
        >>> safe_divide(100, 0)
        Decimal('0')
        >>> safe_divide(100, 0, raise_on_zero=True)
        DivisionByZeroError: Cannot divide 100 by zero
    """
    if default is None:
        default = Decimal("0")
    
    if numerator is None:
        return default
    
    num = to_decimal(numerator)
    denom = to_decimal(denominator)
    
    if is_effectively_zero(denom):
        if raise_on_zero:
            raise DivisionByZeroError(
                numerator=float(num),
                denominator_name="denominator",
            )
        return default
    
    result = num / denom
    return round_decimal(result, precision)


def round_decimal(
    value: Numeric | None,
    precision: int = 2,
    rounding: RoundingMode = DEFAULT_ROUNDING,
) -> Decimal:
    """
    Round a Decimal value to specified precision.
    
    Args:
        value: Value to round
        precision: Number of decimal places
        rounding: Rounding mode to use
        
    Returns:
        Rounded Decimal value
    """
    if value is None:
        return Decimal("0")
    
    dec_value = to_decimal(value)
    quantize_str = "0." + "0" * precision if precision > 0 else "1"
    return dec_value.quantize(Decimal(quantize_str), rounding=rounding.get_decimal_rounding())


def clamp_value(
    value: Numeric,
    min_value: Numeric | None = None,
    max_value: Numeric | None = None,
) -> Decimal:
    """
    Clamp a value within a range.
    
    Args:
        value: Value to clamp
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        
    Returns:
        Clamped value as Decimal
    """
    result = to_decimal(value)
    
    if min_value is not None:
        min_dec = to_decimal(min_value)
        if result < min_dec:
            result = min_dec
    
    if max_value is not None:
        max_dec = to_decimal(max_value)
        if result > max_dec:
            result = max_dec
    
    return result


# ============================================================================
# PERCENTAGE & GROWTH CALCULATIONS
# ============================================================================

def calculate_percentage(
    part: Numeric | None,
    whole: Numeric | None,
    precision: int = DECIMAL_PLACES["percentage"],
) -> Decimal:
    """
    Calculate percentage: (part / whole) × 100.
    
    Args:
        part: The numerator
        whole: The denominator
        precision: Decimal places for result
        
    Returns:
        Percentage value (e.g., 25.00 for 25%)
    """
    if part is None or whole is None:
        return Decimal("0")
    
    ratio = safe_divide(part, whole, precision=precision + 2)
    return round_decimal(ratio * 100, precision)


def calculate_growth_rate(
    current: Numeric | None,
    previous: Numeric | None,
    precision: int = DECIMAL_PLACES["percentage"],
) -> Decimal | None:
    """
    Calculate period-over-period growth rate.
    
    Formula: ((current - previous) / |previous|) × 100
    
    Args:
        current: Current period value
        previous: Previous period value
        precision: Decimal places for result
        
    Returns:
        Growth rate as percentage, or None if calculation impossible
    """
    if current is None or previous is None:
        return None
    
    curr = to_decimal(current)
    prev = to_decimal(previous)
    
    if is_effectively_zero(prev):
        return None
    
    change = curr - prev
    rate = (change / abs(prev)) * 100
    return round_decimal(rate, precision)


def calculate_cagr(
    beginning_value: Numeric | None,
    ending_value: Numeric | None,
    periods: int,
    precision: int = DECIMAL_PLACES["percentage"],
) -> Decimal | None:
    """
    Calculate Compound Annual Growth Rate (CAGR).
    
    Formula: ((ending / beginning) ^ (1/periods) - 1) × 100
    
    Args:
        beginning_value: Value at start
        ending_value: Value at end
        periods: Number of periods (years)
        precision: Decimal places for result
        
    Returns:
        CAGR as percentage, or None if calculation impossible
    """
    if beginning_value is None or ending_value is None or periods <= 0:
        return None
    
    begin = to_decimal(beginning_value)
    end = to_decimal(ending_value)
    
    if is_effectively_zero(begin) or begin < 0 or end < 0:
        return None
    
    try:
        ratio = float(end / begin)
        cagr = (ratio ** (1 / periods) - 1) * 100
        return round_decimal(Decimal(str(cagr)), precision)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

def calculate_average(
    *values: Numeric | None,
    precision: int = DECIMAL_PLACES["ratio"],
) -> Decimal:
    """
    Calculate arithmetic mean of values.
    
    Args:
        *values: Values to average
        precision: Decimal places for result
        
    Returns:
        Arithmetic mean, or Decimal("0") if no valid values
    """
    valid_values = [to_decimal(v) for v in values if v is not None]
    
    if not valid_values:
        return Decimal("0")
    
    total = sum(valid_values)
    return round_decimal(total / len(valid_values), precision)


def calculate_weighted_average(
    values: Sequence[Numeric | None],
    weights: Sequence[Numeric | None],
    precision: int = DECIMAL_PLACES["ratio"],
) -> Decimal | None:
    """
    Calculate weighted average of values.
    
    Args:
        values: Values to average
        weights: Corresponding weights
        precision: Decimal places for result
        
    Returns:
        Weighted average, or None if calculation impossible
    """
    if len(values) != len(weights):
        return None
    
    pairs = [
        (to_decimal(v), to_decimal(w))
        for v, w in zip(values, weights)
        if v is not None and w is not None
    ]
    
    if not pairs:
        return None
    
    weighted_sum = sum(v * w for v, w in pairs)
    total_weight = sum(w for _, w in pairs)
    
    if is_effectively_zero(total_weight):
        return None
    
    return round_decimal(weighted_sum / total_weight, precision)


def calculate_variance(
    values: Sequence[Numeric | None],
    population: bool = True,
) -> Decimal | None:
    """
    Calculate variance of values.
    
    Args:
        values: Values to calculate variance for
        population: If True, use population variance (N); else sample variance (N-1)
        
    Returns:
        Variance, or None if insufficient values
    """
    valid_values = [to_decimal(v) for v in values if v is not None]
    n = len(valid_values)
    
    if n < 2:
        return None
    
    mean = sum(valid_values) / n
    squared_diffs = [(v - mean) ** 2 for v in valid_values]
    
    divisor = n if population else (n - 1)
    return round_decimal(sum(squared_diffs) / divisor, 6)


def calculate_std_dev(
    values: Sequence[Numeric | None],
    population: bool = True,
) -> Decimal | None:
    """
    Calculate standard deviation of values.
    
    Args:
        values: Values to calculate std dev for
        population: If True, use population std dev; else sample std dev
        
    Returns:
        Standard deviation, or None if insufficient values
    """
    variance = calculate_variance(values, population)
    if variance is None:
        return None
    
    std_dev = Decimal(str(math.sqrt(float(variance))))
    return round_decimal(std_dev, 4)


def calculate_min_max(
    values: Sequence[Numeric | None],
) -> tuple[Decimal, Decimal] | None:
    """
    Calculate minimum and maximum of values.
    
    Args:
        values: Values to find min/max for
        
    Returns:
        Tuple of (min, max), or None if no valid values
    """
    valid_values = [to_decimal(v) for v in values if v is not None]
    
    if not valid_values:
        return None
    
    return (min(valid_values), max(valid_values))

```

# finanalyst_tools/utils/formatting.py
```py
# File: finanalyst_tools/utils/formatting.py
"""
Formatting utilities for financial data display.

Provides consistent formatting for:
- Numbers with thousands separators
- Currency values with symbols
- Percentages and ratios
- Trend indicators
- Markdown tables

All functions handle None gracefully with configurable fallback values.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Sequence

from finanalyst_tools.config import (
    DECIMAL_PLACES,
    CURRENCY_SYMBOLS,
    DEFAULT_CURRENCY,
    TREND_SYMBOLS,
    STATUS_SYMBOLS,
    LARGE_NUMBER_SUFFIXES,
    METRIC_UNITS,
)
from finanalyst_tools.utils.math_ops import to_decimal


# ============================================================================
# NUMBER FORMATTING
# ============================================================================

def format_number(
    value: float | Decimal | int | None,
    precision: int = 2,
    use_thousands_sep: bool = True,
    fallback: str = "N/A",
) -> str:
    """
    Format a number with optional thousands separator.
    
    Args:
        value: Number to format
        precision: Decimal places
        use_thousands_sep: Whether to include thousand separators
        fallback: String to return if value is None
        
    Returns:
        Formatted string
    """
    if value is None:
        return fallback
    
    dec_value = to_decimal(value, precision=precision)
    
    if use_thousands_sep:
        return f"{float(dec_value):,.{precision}f}"
    return f"{float(dec_value):.{precision}f}"


def format_currency(
    value: float | Decimal | int | None,
    currency_code: str = DEFAULT_CURRENCY,
    precision: int = DECIMAL_PLACES["currency"],
    show_symbol: bool = True,
    fallback: str = "N/A",
) -> str:
    """
    Format a monetary value with currency symbol.
    
    Args:
        value: Amount to format
        currency_code: ISO currency code
        precision: Decimal places
        show_symbol: Whether to include currency symbol
        fallback: String to return if value is None
        
    Returns:
        Formatted currency string (e.g., "S$1,234.56")
    """
    if value is None:
        return fallback
    
    dec_value = to_decimal(value, precision=precision)
    formatted = f"{float(dec_value):,.{precision}f}"
    
    if show_symbol:
        symbol = CURRENCY_SYMBOLS.get(currency_code, currency_code)
        # Handle negative values
        if dec_value < 0:
            return f"-{symbol}{formatted.lstrip('-')}"
        return f"{symbol}{formatted}"
    
    return formatted


def format_percentage(
    value: float | Decimal | None,
    precision: int = DECIMAL_PLACES["percentage"],
    show_symbol: bool = True,
    show_sign: bool = False,
    fallback: str = "N/A",
) -> str:
    """
    Format a percentage value.
    
    Args:
        value: Percentage value (e.g., 25.5 for 25.5%)
        precision: Decimal places
        show_symbol: Whether to include % symbol
        show_sign: Whether to show + for positive values
        fallback: String to return if value is None
        
    Returns:
        Formatted percentage string
    """
    if value is None:
        return fallback
    
    dec_value = to_decimal(value, precision=precision)
    
    sign = ""
    if show_sign and dec_value > 0:
        sign = "+"
    
    formatted = f"{float(dec_value):.{precision}f}"
    
    if show_symbol:
        return f"{sign}{formatted}%"
    return f"{sign}{formatted}"


def format_ratio(
    value: float | Decimal | None,
    precision: int = 2,
    suffix: str = "x",
    fallback: str = "N/A",
) -> str:
    """
    Format a ratio value.
    
    Args:
        value: Ratio value (e.g., 1.5 for 1.5x)
        precision: Decimal places
        suffix: Suffix to append (default: "x")
        fallback: String to return if value is None
        
    Returns:
        Formatted ratio string (e.g., "1.50x")
    """
    if value is None:
        return fallback
    
    dec_value = to_decimal(value, precision=precision)
    return f"{float(dec_value):.{precision}f}{suffix}"


def format_change(
    value: float | Decimal | None,
    precision: int = 2,
    show_sign: bool = True,
    fallback: str = "N/A",
) -> str:
    """
    Format a change value with +/- sign.
    
    Args:
        value: Change value
        precision: Decimal places
        show_sign: Whether to show + for positive values
        fallback: String to return if value is None
        
    Returns:
        Formatted change string
    """
    if value is None:
        return fallback
    
    dec_value = to_decimal(value, precision=precision)
    
    if show_sign and dec_value > 0:
        return f"+{float(dec_value):.{precision}f}"
    return f"{float(dec_value):.{precision}f}"


def format_large_number(
    value: float | Decimal | int | None,
    precision: int = 1,
    fallback: str = "N/A",
) -> str:
    """
    Format large numbers with K/M/B/T suffixes.
    
    Args:
        value: Number to format
        precision: Decimal places
        fallback: String to return if value is None
        
    Returns:
        Formatted string (e.g., "1.5M", "500K")
    """
    if value is None:
        return fallback
    
    num = float(to_decimal(value))
    
    if num == 0:
        return "0"
    
    sign = "-" if num < 0 else ""
    num = abs(num)
    
    for threshold, suffix in LARGE_NUMBER_SUFFIXES:
        if num >= threshold:
            formatted = num / threshold
            return f"{sign}{formatted:.{precision}f}{suffix}"
    
    return f"{sign}{num:.{precision}f}"


# ============================================================================
# INDICATOR FORMATTING
# ============================================================================

def format_trend_indicator(
    direction: str,
    include_text: bool = False,
) -> str:
    """
    Format a trend direction as an indicator.
    
    Args:
        direction: One of "increasing", "decreasing", "stable", "volatile"
        include_text: Whether to include text after symbol
        
    Returns:
        Trend indicator symbol (e.g., "↑", "↓")
    """
    symbol = TREND_SYMBOLS.get(direction.lower(), "?")
    
    if include_text:
        return f"{symbol} {direction.capitalize()}"
    return symbol


def format_status_indicator(
    status: str,
    include_text: bool = False,
) -> str:
    """
    Format a status as an indicator.
    
    Args:
        status: One of "good", "warning", "error", "info", "unknown"
        include_text: Whether to include text after symbol
        
    Returns:
        Status indicator symbol (e.g., "✅", "⚠️")
    """
    symbol = STATUS_SYMBOLS.get(status.lower(), STATUS_SYMBOLS["unknown"])
    
    if include_text:
        return f"{symbol} {status.capitalize()}"
    return symbol


def format_value_with_unit(
    value: float | Decimal | None,
    metric_name: str,
    precision: int | None = None,
    currency_code: str = DEFAULT_CURRENCY,
) -> str:
    """
    Format a value with appropriate unit based on metric name.
    
    Args:
        value: Value to format
        metric_name: Name of the metric (used to determine unit)
        precision: Override precision (uses default for unit type if None)
        currency_code: Currency code for currency values
        
    Returns:
        Formatted string with appropriate unit
    """
    if value is None:
        return "N/A"
    
    unit = METRIC_UNITS.get(metric_name.lower(), "number")
    
    if unit == "percentage":
        prec = precision if precision is not None else DECIMAL_PLACES["percentage"]
        return format_percentage(value, precision=prec)
    elif unit == "ratio":
        prec = precision if precision is not None else DECIMAL_PLACES["ratio"]
        return format_ratio(value, precision=prec)
    elif unit == "currency":
        prec = precision if precision is not None else DECIMAL_PLACES["currency"]
        return format_currency(value, currency_code=currency_code, precision=prec)
    elif unit == "days":
        prec = precision if precision is not None else 0
        return f"{int(value)} days"
    else:
        prec = precision if precision is not None else 2
        return format_number(value, precision=prec)


# ============================================================================
# TABLE FORMATTING
# ============================================================================

def format_markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    alignment: Sequence[str] | None = None,
) -> str:
    """
    Format data as a Markdown table.
    
    Args:
        headers: Column headers
        rows: Table rows (each row is a sequence of cell values)
        alignment: Column alignments ("left", "center", "right") per column
        
    Returns:
        Markdown table string
    """
    if not headers or not rows:
        return ""
    
    # Convert all values to strings
    str_headers = [str(h) for h in headers]
    str_rows = [[str(cell) for cell in row] for row in rows]
    
    # Calculate column widths
    col_widths = [len(h) for h in str_headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))
    
    # Build alignment row
    if alignment is None:
        alignment = ["left"] * len(headers)
    
    align_row = []
    for i, align in enumerate(alignment):
        width = col_widths[i] if i < len(col_widths) else 3
        if align == "center":
            align_row.append(":" + "-" * (width - 2) + ":")
        elif align == "right":
            align_row.append("-" * (width - 1) + ":")
        else:  # left
            align_row.append(":" + "-" * (width - 1))
    
    # Build table
    lines = []
    
    # Header row
    header_cells = [h.ljust(col_widths[i]) for i, h in enumerate(str_headers)]
    lines.append("| " + " | ".join(header_cells) + " |")
    
    # Alignment row
    lines.append("| " + " | ".join(align_row) + " |")
    
    # Data rows
    for row in str_rows:
        cells = []
        for i, cell in enumerate(row):
            width = col_widths[i] if i < len(col_widths) else len(cell)
            cells.append(cell.ljust(width))
        lines.append("| " + " | ".join(cells) + " |")
    
    return "\n".join(lines)

```

# finanalyst_tools/utils/serialization.py
```py
from __future__ import annotations

from dataclasses import is_dataclass
from decimal import Decimal
from enum import Enum
from typing import Any


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]

    if is_dataclass(value):
        return to_jsonable(value.__dict__)

    if hasattr(value, "model_dump"):
        try:
            return to_jsonable(value.model_dump(by_alias=False))
        except TypeError:
            return to_jsonable(value.model_dump())

    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        return to_jsonable(value.to_dict())

    return str(value)

```

# finanalyst_tools/utils/currency.py
```py
# File: finanalyst_tools/utils/currency.py
"""
Currency handling utilities with Singapore focus.

Provides:
- Currency information (symbols, decimal places, names)
- SGD-specific formatting
- GST calculations (Singapore 9% rate)
- SFRS for Small Entities qualification checks
"""

from __future__ import annotations

from decimal import Decimal
from typing import Final

from finanalyst_tools.config import (
    SingaporeConstants,
    SUPPORTED_CURRENCIES,
    CURRENCY_SYMBOLS,
    ZERO_DECIMAL_CURRENCIES,
    DECIMAL_PLACES,
)
from finanalyst_tools.utils.math_ops import to_decimal, round_decimal


# ============================================================================
# SINGAPORE CONSTANTS
# ============================================================================

SGD_GST_RATE: Final[Decimal] = SingaporeConstants.GST_RATE

# SFRS for Small Entities thresholds (qualify if meet 2 of 3)
SFRS_THRESHOLDS: Final[dict[str, int]] = {
    "revenue": SingaporeConstants.SFRS_SMALL_ENTITY_REVENUE,
    "total_assets": SingaporeConstants.SFRS_SMALL_ENTITY_ASSETS,
    "employees": SingaporeConstants.SFRS_SMALL_ENTITY_EMPLOYEES,
}


# ============================================================================
# CURRENCY INFORMATION
# ============================================================================

CURRENCY_INFO: Final[dict[str, dict[str, str | int]]] = {
    "SGD": {"symbol": "S$", "decimals": 2, "name": "Singapore Dollar"},
    "USD": {"symbol": "$", "decimals": 2, "name": "US Dollar"},
    "EUR": {"symbol": "€", "decimals": 2, "name": "Euro"},
    "GBP": {"symbol": "£", "decimals": 2, "name": "British Pound"},
    "JPY": {"symbol": "¥", "decimals": 0, "name": "Japanese Yen"},
    "CNY": {"symbol": "¥", "decimals": 2, "name": "Chinese Yuan"},
    "HKD": {"symbol": "HK$", "decimals": 2, "name": "Hong Kong Dollar"},
    "AUD": {"symbol": "A$", "decimals": 2, "name": "Australian Dollar"},
    "MYR": {"symbol": "RM", "decimals": 2, "name": "Malaysian Ringgit"},
    "IDR": {"symbol": "Rp", "decimals": 0, "name": "Indonesian Rupiah"},
    "THB": {"symbol": "฿", "decimals": 2, "name": "Thai Baht"},
    "INR": {"symbol": "₹", "decimals": 2, "name": "Indian Rupee"},
    "KRW": {"symbol": "₩", "decimals": 0, "name": "South Korean Won"},
    "NZD": {"symbol": "NZ$", "decimals": 2, "name": "New Zealand Dollar"},
    "PHP": {"symbol": "₱", "decimals": 2, "name": "Philippine Peso"},
    "VND": {"symbol": "₫", "decimals": 0, "name": "Vietnamese Dong"},
}


# ============================================================================
# CURRENCY UTILITIES
# ============================================================================

def get_currency_symbol(currency_code: str) -> str:
    """
    Get the symbol for a currency code.
    
    Args:
        currency_code: ISO currency code (e.g., "SGD")
        
    Returns:
        Currency symbol (e.g., "S$")
    """
    info = CURRENCY_INFO.get(currency_code.upper())
    if info:
        return str(info["symbol"])
    return currency_code


def get_currency_decimals(currency_code: str) -> int:
    """
    Get the standard decimal places for a currency.
    
    Args:
        currency_code: ISO currency code
        
    Returns:
        Number of decimal places (0 for JPY, KRW, etc.)
    """
    info = CURRENCY_INFO.get(currency_code.upper())
    if info:
        return int(info["decimals"])
    return 2  # Default


def get_currency_name(currency_code: str) -> str:
    """
    Get the full name of a currency.
    
    Args:
        currency_code: ISO currency code
        
    Returns:
        Currency name (e.g., "Singapore Dollar")
    """
    info = CURRENCY_INFO.get(currency_code.upper())
    if info:
        return str(info["name"])
    return currency_code


def is_supported_currency(currency_code: str) -> bool:
    """
    Check if a currency is supported.
    
    Args:
        currency_code: ISO currency code
        
    Returns:
        True if supported
    """
    return currency_code.upper() in SUPPORTED_CURRENCIES


def format_sgd(
    amount: float | Decimal | int | None,
    show_symbol: bool = True,
    precision: int = 2,
) -> str:
    """
    Format an amount in Singapore Dollars.
    
    Args:
        amount: Amount to format
        show_symbol: Whether to include S$ symbol
        precision: Decimal places
        
    Returns:
        Formatted SGD string (e.g., "S$1,234.56")
    """
    if amount is None:
        return "N/A"
    
    dec_amount = to_decimal(amount, precision=precision)
    formatted = f"{float(dec_amount):,.{precision}f}"
    
    if show_symbol:
        if dec_amount < 0:
            return f"-S${formatted.lstrip('-')}"
        return f"S${formatted}"
    return formatted


# ============================================================================
# GST CALCULATIONS
# ============================================================================

def calculate_gst_exclusive(gst_inclusive: float | Decimal) -> Decimal:
    """
    Convert GST-inclusive amount to GST-exclusive.
    
    Args:
        gst_inclusive: Amount including GST
        
    Returns:
        Amount excluding GST
        
    Example:
        >>> calculate_gst_exclusive(109)
        Decimal('100.00')
    """
    amount = to_decimal(gst_inclusive)
    result = amount / (1 + SGD_GST_RATE)
    return round_decimal(result, DECIMAL_PLACES["currency"])


def calculate_gst_inclusive(gst_exclusive: float | Decimal) -> Decimal:
    """
    Convert GST-exclusive amount to GST-inclusive.
    
    Args:
        gst_exclusive: Amount excluding GST
        
    Returns:
        Amount including GST
        
    Example:
        >>> calculate_gst_inclusive(100)
        Decimal('109.00')
    """
    amount = to_decimal(gst_exclusive)
    result = amount * (1 + SGD_GST_RATE)
    return round_decimal(result, DECIMAL_PLACES["currency"])


def calculate_gst_amount(base_amount: float | Decimal) -> Decimal:
    """
    Calculate the GST portion on a GST-exclusive amount.
    
    Args:
        base_amount: GST-exclusive amount
        
    Returns:
        GST amount
        
    Example:
        >>> calculate_gst_amount(100)
        Decimal('9.00')
    """
    amount = to_decimal(base_amount)
    result = amount * SGD_GST_RATE
    return round_decimal(result, DECIMAL_PLACES["currency"])


# ============================================================================
# SFRS QUALIFICATION
# ============================================================================

def is_sfrs_small_entity(
    annual_revenue: float | Decimal | None = None,
    total_assets: float | Decimal | None = None,
    num_employees: int | None = None,
) -> tuple[bool, list[str]]:
    """
    Check if a company qualifies as a Small Entity under SFRS.
    
    A company qualifies if it meets at least 2 of the 3 criteria:
    - Annual revenue ≤ S$10M
    - Total assets ≤ S$10M
    - Employees ≤ 50
    
    Args:
        annual_revenue: Annual revenue in SGD
        total_assets: Total assets in SGD
        num_employees: Number of employees
        
    Returns:
        Tuple of (qualifies, list of met criteria)
    """
    criteria_met = []
    
    if annual_revenue is not None:
        if to_decimal(annual_revenue) <= SFRS_THRESHOLDS["revenue"]:
            criteria_met.append("revenue")
    
    if total_assets is not None:
        if to_decimal(total_assets) <= SFRS_THRESHOLDS["total_assets"]:
            criteria_met.append("total_assets")
    
    if num_employees is not None:
        if num_employees <= SFRS_THRESHOLDS["employees"]:
            criteria_met.append("employees")
    
    qualifies = len(criteria_met) >= 2
    return (qualifies, criteria_met)

```

# finanalyst_tools/validation/reconciliation.py
```py
# File: finanalyst_tools/validation/reconciliation.py
"""
Cross-statement reconciliation validation.

Verifies consistency between values that should match across
different financial statements:
- Net income (IS vs CF)
- Cash balance (BS vs CF)
- Retained earnings rollforward
- Balance sheet equation
- Working capital consistency
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from finanalyst_tools.config import ReconciliationTolerances
from finanalyst_tools.models.validation import (
    ReconciliationCheck,
    ReconciliationResult,
)
from finanalyst_tools.models.financial_statements import (
    IncomeStatementData,
    BalanceSheetData,
    CashFlowStatementData,
    FinancialStatementSet,
)
from finanalyst_tools.utils.math_ops import to_decimal, is_effectively_zero


def _create_check(
    check_name: str,
    statement_a: str,
    value_a: Decimal,
    statement_b: str,
    value_b: Decimal,
    tolerance_level: str = "normal",
) -> ReconciliationCheck:
    """
    Create a reconciliation check result.
    
    Args:
        check_name: Name of the check
        statement_a: Source of first value
        value_a: First value
        statement_b: Source of second value
        value_b: Second value
        tolerance_level: Tolerance level ("strict", "normal", "loose")
        
    Returns:
        ReconciliationCheck with pass/fail result
    """
    tolerance = ReconciliationTolerances.get_tolerance(tolerance_level)
    difference = abs(value_a - value_b)
    
    # Calculate if within tolerance
    passed = ReconciliationTolerances.is_within_tolerance(
        float(value_a),
        float(value_b),
        tolerance,
    )
    
    if passed:
        message = f"Values match within {tolerance:.1%} tolerance"
    else:
        pct_diff = (difference / max(abs(value_a), abs(value_b), Decimal("1"))) * 100
        message = f"Values differ by {difference:,.2f} ({pct_diff:.1f}%), exceeds {tolerance:.1%} tolerance"
    
    return ReconciliationCheck(
        check_name=check_name,
        statement_a=statement_a,
        value_a=value_a,
        statement_b=statement_b,
        value_b=value_b,
        difference=difference,
        tolerance=tolerance,
        passed=passed,
        message=message,
    )


def reconcile_net_income(
    income_statement: IncomeStatementData,
    cash_flow_statement: CashFlowStatementData,
) -> ReconciliationCheck:
    """
    Verify net income matches between Income Statement and Cash Flow Statement.
    
    Args:
        income_statement: Income statement data
        cash_flow_statement: Cash flow statement data
        
    Returns:
        ReconciliationCheck result
    """
    is_net_income = income_statement.calculated_net_income
    cf_net_income = cash_flow_statement.net_income
    
    return _create_check(
        check_name="Net Income Reconciliation",
        statement_a="Income Statement",
        value_a=is_net_income,
        statement_b="Cash Flow Statement",
        value_b=cf_net_income,
        tolerance_level="strict",
    )


def reconcile_cash_balance(
    balance_sheet: BalanceSheetData,
    cash_flow_statement: CashFlowStatementData,
) -> ReconciliationCheck | None:
    """
    Verify ending cash balance matches between Balance Sheet and Cash Flow Statement.
    
    Args:
        balance_sheet: Balance sheet data
        cash_flow_statement: Cash flow statement data
        
    Returns:
        ReconciliationCheck result or None if ending_cash not provided
    """
    if cash_flow_statement.ending_cash is None:
        return None
    
    bs_cash = balance_sheet.cash_and_equivalents
    cf_ending_cash = cash_flow_statement.ending_cash
    
    return _create_check(
        check_name="Cash Balance Reconciliation",
        statement_a="Balance Sheet",
        value_a=bs_cash,
        statement_b="Cash Flow (Ending)",
        value_b=cf_ending_cash,
        tolerance_level="strict",
    )


def reconcile_retained_earnings(
    current_balance_sheet: BalanceSheetData,
    prior_balance_sheet: BalanceSheetData | None,
    income_statement: IncomeStatementData,
    dividends_paid: Decimal | None = None,
) -> ReconciliationCheck | None:
    """
    Verify retained earnings rollforward.
    
    Formula: Prior RE + Net Income - Dividends = Current RE
    
    Args:
        current_balance_sheet: Current period balance sheet
        prior_balance_sheet: Prior period balance sheet
        income_statement: Current period income statement
        dividends_paid: Dividends paid during period (optional)
        
    Returns:
        ReconciliationCheck result or None if prior BS not provided
    """
    if prior_balance_sheet is None:
        return None
    
    prior_re = prior_balance_sheet.retained_earnings
    net_income = income_statement.calculated_net_income
    dividends = dividends_paid or Decimal("0")
    
    expected_re = prior_re + net_income - dividends
    actual_re = current_balance_sheet.retained_earnings
    
    return _create_check(
        check_name="Retained Earnings Rollforward",
        statement_a="Calculated (Prior RE + NI - Div)",
        value_a=expected_re,
        statement_b="Balance Sheet",
        value_b=actual_re,
        tolerance_level="normal",
    )


def reconcile_balance_sheet_equation(
    balance_sheet: BalanceSheetData,
) -> ReconciliationCheck:
    """
    Verify the fundamental accounting equation: Assets = Liabilities + Equity.
    
    Args:
        balance_sheet: Balance sheet data
        
    Returns:
        ReconciliationCheck result
    """
    total_assets = balance_sheet.calculated_total_assets
    total_liab_equity = (
        balance_sheet.calculated_total_liabilities +
        balance_sheet.calculated_total_equity
    )
    
    return _create_check(
        check_name="Balance Sheet Equation",
        statement_a="Total Assets",
        value_a=total_assets,
        statement_b="Liabilities + Equity",
        value_b=total_liab_equity,
        tolerance_level="strict",
    )


def reconcile_working_capital(
    balance_sheet: BalanceSheetData,
) -> ReconciliationCheck:
    """
    Verify working capital calculation consistency.
    
    Args:
        balance_sheet: Balance sheet data
        
    Returns:
        ReconciliationCheck result
    """
    current_assets = balance_sheet.calculated_current_assets
    current_liabilities = balance_sheet.calculated_current_liabilities
    calculated_wc = current_assets - current_liabilities
    
    # Compare with the property calculation
    property_wc = balance_sheet.working_capital
    
    return _create_check(
        check_name="Working Capital Consistency",
        statement_a="CA - CL Calculation",
        value_a=calculated_wc,
        statement_b="Working Capital Property",
        value_b=property_wc,
        tolerance_level="strict",
    )


def run_all_reconciliations(
    statement_set: FinancialStatementSet,
    prior_balance_sheet: BalanceSheetData | None = None,
    dividends_paid: Decimal | None = None,
) -> ReconciliationResult:
    """
    Run all applicable reconciliation checks.
    
    Args:
        statement_set: Complete set of financial statements
        prior_balance_sheet: Prior period balance sheet (optional)
        dividends_paid: Dividends paid during period (optional)
        
    Returns:
        ReconciliationResult with all check results
    """
    result = ReconciliationResult()
    
    # Balance sheet equation (always run)
    bs_equation = reconcile_balance_sheet_equation(statement_set.balance_sheet)
    result.add_check(bs_equation)
    
    # Working capital consistency (always run)
    wc_check = reconcile_working_capital(statement_set.balance_sheet)
    result.add_check(wc_check)
    
    # Net income reconciliation (if cash flow available)
    if statement_set.cash_flow_statement:
        ni_check = reconcile_net_income(
            statement_set.income_statement,
            statement_set.cash_flow_statement,
        )
        result.add_check(ni_check)
        
        # Cash balance reconciliation
        cash_check = reconcile_cash_balance(
            statement_set.balance_sheet,
            statement_set.cash_flow_statement,
        )
        if cash_check:
            result.add_check(cash_check)
    
    # Retained earnings rollforward (if prior BS available)
    if prior_balance_sheet:
        re_check = reconcile_retained_earnings(
            statement_set.balance_sheet,
            prior_balance_sheet,
            statement_set.income_statement,
            dividends_paid,
        )
        if re_check:
            result.add_check(re_check)
    
    return result

```

# finanalyst_tools/validation/__init__.py
```py
# File: finanalyst_tools/validation/__init__.py
"""
Validation functions for FinAnalyst-Pro Agent Tools.

This package provides:
- Schema validation for financial statements
- Cross-statement reconciliation
- Plausibility checks for calculated metrics
"""

from finanalyst_tools.validation.schema_validator import (
    validate_income_statement_schema,
    validate_balance_sheet_schema,
    validate_cash_flow_schema,
    validate_financial_data_completeness,
    validate_statement_set,
    REQUIRED_FIELDS,
    FIELD_ALIASES,
)

from finanalyst_tools.validation.reconciliation import (
    reconcile_net_income,
    reconcile_cash_balance,
    reconcile_retained_earnings,
    reconcile_balance_sheet_equation,
    reconcile_working_capital,
    run_all_reconciliations,
)

from finanalyst_tools.validation.plausibility import (
    check_plausibility,
    check_all_plausibility,
    PlausibilityChecker,
)


__all__ = [
    # Schema validation
    "validate_income_statement_schema",
    "validate_balance_sheet_schema",
    "validate_cash_flow_schema",
    "validate_financial_data_completeness",
    "validate_statement_set",
    "REQUIRED_FIELDS",
    "FIELD_ALIASES",
    # Reconciliation
    "reconcile_net_income",
    "reconcile_cash_balance",
    "reconcile_retained_earnings",
    "reconcile_balance_sheet_equation",
    "reconcile_working_capital",
    "run_all_reconciliations",
    # Plausibility
    "check_plausibility",
    "check_all_plausibility",
    "PlausibilityChecker",
]

```

