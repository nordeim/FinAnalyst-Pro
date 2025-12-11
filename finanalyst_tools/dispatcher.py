# finanalyst_tools/dispatcher.py
"""
Tool call dispatcher for executing tools from LLM requests.

Provides:
- Parameter validation and type coercion
- Execution timing
- Error handling and formatting
- Structured result output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any
import json
import time

from finanalyst_tools.tool_registry import TOOL_REGISTRY, ToolDefinition
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
        success: Whether execution was successful
        result: The result data (if successful)
        error: Error message (if failed)
        error_details: Additional error context
        execution_time_ms: Execution time in milliseconds
    """
    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    error_details: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data: dict[str, Any] = {
            "tool_name": self.tool_name,
            "success": self.success,
        }
        
        if self.success:
            # Serialize result
            if hasattr(self.result, "to_dict"):
                data["result"] = self.result.to_dict()
            elif isinstance(self.result, Decimal):
                data["result"] = float(self.result)
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
    
    def __str__(self) -> str:
        if self.success:
            return f"✅ {self.tool_name}: Success ({self.execution_time_ms:.1f}ms)"
        return f"❌ {self.tool_name}: {self.error}"


class ToolDispatcher:
    """
    Dispatcher for executing tool calls.
    
    Handles:
    - Tool lookup
    - Parameter validation and coercion
    - Execution with timing
    - Error handling
    """
    
    def __init__(self):
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
            parameters: Parameters to pass to the tool
            
        Returns:
            ToolCallResult with execution outcome
        """
        start_time = time.perf_counter()
        
        try:
            # Get tool definition
            tool = self.registry.get(tool_name)
            if tool is None:
                available = self.registry.get_tool_names()
                raise ToolNotFoundError(tool_name, available_tools=available)
            
            # Validate and coerce parameters
            validated_params = self._validate_and_coerce_parameters(tool, parameters)
            
            # Execute the tool function
            result = tool.function(**validated_params)
            
            # Calculate execution time
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return ToolCallResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=execution_time,
            )
            
        except ToolNotFoundError as e:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                error_details=e.to_dict(),
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )
            
        except ToolParameterError as e:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                error_details=e.to_dict(),
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )
            
        except FinAnalystError as e:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                error_details=e.to_dict(),
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )
            
        except Exception as e:
            # Wrap unexpected errors
            wrapped = ToolExecutionError(tool_name, e, parameters)
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=str(wrapped),
                error_details=wrapped.to_dict(),
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )
    
    def _validate_and_coerce_parameters(
        self,
        tool: ToolDefinition,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate parameters against tool definition and coerce types.
        
        Args:
            tool: Tool definition
            parameters: Raw parameters from caller
            
        Returns:
            Validated and coerced parameters
            
        Raises:
            ToolParameterError: If validation fails
        """
        result = {}
        
        for param in tool.parameters:
            value = parameters.get(param.name)
            
            # Check required parameters
            if value is None:
                if param.required:
                    raise ToolParameterError(
                        tool_name=tool.name,
                        parameter_name=param.name,
                        message="Required parameter is missing",
                        expected_type=param.type,
                    )
                elif param.default is not None:
                    value = param.default
                else:
                    continue  # Optional with no default, skip
            
            # Coerce types
            try:
                coerced = self._coerce_value(value, param.type, param.name, tool.name)
                result[param.name] = coerced
            except (ValueError, TypeError, InvalidOperation) as e:
                raise ToolParameterError(
                    tool_name=tool.name,
                    parameter_name=param.name,
                    message=f"Cannot convert value to {param.type}: {e}",
                    expected_type=param.type,
                    actual_value=value,
                )
        
        return result
    
    def _coerce_value(
        self,
        value: Any,
        expected_type: str,
        param_name: str,
        tool_name: str,
    ) -> Any:
        """
        Coerce a value to the expected type.
        
        Args:
            value: Value to coerce
            expected_type: Expected type string
            param_name: Parameter name (for error messages)
            tool_name: Tool name (for error messages)
            
        Returns:
            Coerced value
        """
        if expected_type == "number":
            if isinstance(value, (int, float, Decimal)):
                return Decimal(str(value))
            if isinstance(value, str):
                # Remove currency symbols and commas
                cleaned = value.replace("$", "").replace(",", "").replace("S$", "").strip()
                return Decimal(cleaned)
            raise ValueError(f"Cannot convert {type(value).__name__} to number")
        
        elif expected_type == "integer":
            if isinstance(value, int):
                return value
            if isinstance(value, (float, Decimal)):
                return int(value)
            if isinstance(value, str):
                return int(float(value))
            raise ValueError(f"Cannot convert {type(value).__name__} to integer")
        
        elif expected_type == "string":
            return str(value)
        
        elif expected_type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "yes", "1")
            return bool(value)
        
        elif expected_type == "object":
            if isinstance(value, dict):
                return value
            raise ValueError(f"Expected object, got {type(value).__name__}")
        
        elif expected_type == "array":
            if isinstance(value, list):
                return value
            raise ValueError(f"Expected array, got {type(value).__name__}")
        
        # Unknown type, return as-is
        return value
    
    def get_available_tools(self) -> list[str]:
        """Get list of available tool names."""
        return self.registry.get_tool_names()
    
    def get_tool_info(self, tool_name: str) -> dict[str, Any] | None:
        """Get information about a specific tool."""
        tool = self.registry.get(tool_name)
        if tool:
            return tool.to_dict()
        return None


def execute_tool(
    tool_name: str,
    parameters: dict[str, Any],
) -> ToolCallResult:
    """
    Convenience function to execute a tool.
    
    Args:
        tool_name: Name of the tool to execute
        parameters: Parameters to pass to the tool
        
    Returns:
        ToolCallResult with execution outcome
    """
    dispatcher = ToolDispatcher()
    return dispatcher.execute(tool_name, parameters)


# Global dispatcher instance
DISPATCHER = ToolDispatcher()
