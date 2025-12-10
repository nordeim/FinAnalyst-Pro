# finanalyst_tools/dispatcher.py
"""
Tool dispatcher for executing tool calls from LLM.
Handles parameter validation, execution, and error handling.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from typing import Any

from .tool_registry import TOOL_REGISTRY, ToolDefinition


@dataclass
class ToolCallResult:
    """Result of a tool execution."""
    tool_name: str
    success: bool
    result: Any
    error: str | None = None
    execution_time_ms: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result if self.success else None,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string for LLM consumption."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class ToolDispatcher:
    """
    Dispatches tool calls from LLM to actual Python functions.
    Handles validation, execution, and error recovery.
    """
    
    def __init__(self):
        self.registry = TOOL_REGISTRY
    
    def execute(self, tool_name: str, parameters: dict[str, Any]) -> ToolCallResult:
        """
        Execute a tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Dictionary of parameter values
            
        Returns:
            ToolCallResult with execution outcome
        """
        import time
        start_time = time.perf_counter()
        
        # Get tool definition
        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Unknown tool: {tool_name}. Use list_available_tools to see valid options.",
            )
        
        # Validate 
