# finanalyst_tools/dispatcher.py
"""
Tool dispatcher for executing tool calls from LLM.

Provides:
- Parameter validation and type coercion
- Execution timing
- Error handling and formatting
- Structured result formatting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any
import json
import math
import time

from finanalyst_tools.tool_registry import (
    TOOL_REGISTRY,
    ToolDefinition,
    ToolParameter,
    _normalize_nested_numbers,
    _reject_json_constant,
)
from finanalyst_tools.exceptions import (
    ToolNotFoundError,
    ToolExecutionError,
    ToolParameterError,
    FinAnalystError,
)


@dataclass
class ToolCallResult:
    """
    Result of a tool execution.
    
    Attributes:
        tool_name: Name of the executed tool
        success: Whether execution succeeded
        result: The result (if successful)
        error: Error message (if failed)
        error_details: Additional error context
        execution_time_ms: Execution time in milliseconds
    """
    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    error_details: dict[str, Any] | None = None
    execution_time_ms: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data: dict[str, Any] = {
            "tool_name": self.tool_name,
            "success": self.success,
        }
        
        if self.success:
            if hasattr(self.result, "to_dict"):
                data["result"] = self.result.to_dict()
            else:
                data["result"] = self.result
        else:
            data["error"] = self.error
            if self.error_details:
                data["error_details"] = self.error_details
        
        if self.execution_time_ms is not None:
            data["execution_time_ms"] = round(self.execution_time_ms, 2)
        
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class ToolDispatcher:
    """
    Dispatcher for executing tool calls.
    
    Handles:
    - Tool lookup
    - Parameter validation
    - Type coercion (string → Decimal for numbers)
    - Execution with timing
    - Error handling
    """
    
    def __init__(self):
        """Initialize the dispatcher."""
        self.registry = TOOL_REGISTRY
    
    def execute(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ToolCallResult:
        """
        Execute a tool call.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Dictionary of parameters
            
        Returns:
            ToolCallResult with execution outcome
        """
        start_time = time.perf_counter()
        
        try:
            # Get tool definition
            tool = self.registry.get(tool_name)
            if tool is None:
                available = self.registry.list_tool_names()
                raise ToolNotFoundError(tool_name, available)
            
            # Validate parameters
            validated_params = self._validate_and_coerce_parameters(tool, parameters)
            
            # Execute the tool
            if tool.function is None:
                raise ToolExecutionError(
                    tool_name=tool_name,
                    original_error=ValueError("Tool function not registered"),
                )
            
            result = tool.function(**validated_params)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return ToolCallResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=execution_time,
            )
            
        except FinAnalystError as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                error_details=e.to_dict(),
                execution_time_ms=execution_time,
            )
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=f"Unexpected error: {str(e)}",
                error_details={
                    "error_type": type(e).__name__,
                    "message": str(e),
                },
                execution_time_ms=execution_time,
            )
    
    def _validate_and_coerce_parameters(
        self,
        tool: ToolDefinition,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate and coerce parameters for a tool.
        
        Args:
            tool: Tool definition
            parameters: Raw parameters
            
        Returns:
            Validated and coerced parameters
            
        Raises:
            ToolParameterError: If validation fails
        """
        result = {}
        
        # Check required parameters
        for param in tool.parameters:
            if param.required and param.name not in parameters:
                raise ToolParameterError(
                    tool_name=tool.name,
                    parameter_name=param.name,
                    message="Required parameter is missing",
                    expected_type=param.type,
                )
        
        # Validate and coerce each provided parameter
        for param in tool.parameters:
            if param.name in parameters:
                value = parameters[param.name]
                coerced = self._coerce_parameter(tool.name, param, value)
                if param.enum is not None and coerced not in param.enum:
                    raise ToolParameterError(
                        tool_name=tool.name,
                        parameter_name=param.name,
                        message=f"Value must be one of: {', '.join(param.enum)}",
                        expected_type=param.type,
                        actual_value=value,
                    )
                result[param.name] = coerced
            elif param.default is not None:
                result[param.name] = param.default
        
        return result
    
    def _coerce_parameter(
        self,
        tool_name: str,
        param: ToolParameter,
        value: Any,
    ) -> Any:
        """
        Coerce a parameter value to the expected type.
        
        Args:
            tool_name: Tool name (for error messages)
            param: Parameter definition
            value: Raw value
            
        Returns:
            Coerced value
            
        Raises:
            ToolParameterError: If coercion fails
        """
        if value is None:
            if param.required:
                raise ToolParameterError(
                    tool_name=tool_name,
                    parameter_name=param.name,
                    message="Value cannot be None",
                    expected_type=param.type,
                )
            return param.default
        
        try:
            if param.type == "number":
                # Convert to Decimal for financial precision
                if isinstance(value, Decimal):
                    if not value.is_finite():
                        raise ValueError("Non-finite Decimal is not allowed")
                    return value
                if isinstance(value, (int, float)):
                    if isinstance(value, float) and not math.isfinite(value):
                        raise ValueError("Non-finite float is not allowed")
                    return Decimal(str(value))
                if isinstance(value, str):
                    dec = Decimal(value)
                    if not dec.is_finite():
                        raise ValueError("Non-finite Decimal is not allowed")
                    return dec
                raise ValueError(f"Cannot convert {type(value).__name__} to number")
                
            elif param.type == "integer":
                return int(value)
                
            elif param.type == "boolean":
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes")
                return bool(value)
                
            elif param.type == "string":
                return str(value)
                
            elif param.type == "object":
                if isinstance(value, dict):
                    coerced = value
                if isinstance(value, str):
                    coerced = json.loads(value, parse_float=Decimal, parse_constant=_reject_json_constant)
                raise ValueError("Expected object/dictionary")

                if not isinstance(coerced, dict):
                    raise ValueError("Expected object/dictionary")
                return _normalize_nested_numbers(coerced)
                
            elif param.type == "array":
                if isinstance(value, list):
                    coerced = value
                if isinstance(value, str):
                    coerced = json.loads(value, parse_float=Decimal, parse_constant=_reject_json_constant)
                raise ValueError("Expected array/list")

                if not isinstance(coerced, list):
                    raise ValueError("Expected array/list")
                return _normalize_nested_numbers(coerced)
                
            else:
                # Unknown type - pass through
                return value
                
        except (ValueError, InvalidOperation, json.JSONDecodeError) as e:
            raise ToolParameterError(
                tool_name=tool_name,
                parameter_name=param.name,
                message=f"Cannot convert to {param.type}: {str(e)}",
                expected_type=param.type,
                actual_value=value,
            )
    
    def list_tools(self) -> list[str]:
        """List all available tool names."""
        return self.registry.list_tool_names()
    
    def get_tool_info(self, tool_name: str) -> dict[str, Any] | None:
        """Get information about a specific tool."""
        tool = self.registry.get(tool_name)
        if tool:
            return tool.to_dict()
        return None


# Global singleton instance
DISPATCHER = ToolDispatcher()


def execute_tool(tool_name: str, parameters: dict[str, Any]) -> ToolCallResult:
    """
    Convenience function to execute a tool.
    
    Args:
        tool_name: Name of the tool
        parameters: Tool parameters
        
    Returns:
        ToolCallResult
    """
    return DISPATCHER.execute(tool_name, parameters)
