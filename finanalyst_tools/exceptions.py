# File: finanalyst_tools/exceptions.py
"""
Custom exception hierarchy for FinAnalyst-Pro Agent Tools.

Provides specific exception types for different error categories:
- Calculation errors (arithmetic, division by zero, invalid inputs)
- Validation errors (schema, reconciliation, plausibility)
- Data errors (parsing, missing data)
- Tool errors (not found, execution failure)

All exceptions support:
- Serialization to dict/JSON for structured error handling
- Auto-generated error codes for programmatic handling
- Actionable suggestions for LLM consumption
"""

from __future__ import annotations

import json
from typing import Any


# ============================================================================
# BASE EXCEPTION
# ============================================================================

class FinAnalystError(Exception):
    """
    Base exception for all FinAnalyst-Pro errors.
    
    Features:
    - Auto-generated error_code from class name
    - Optional details dictionary for context
    - Optional suggestion for resolution
    - JSON serialization support
    
    All custom exceptions should inherit from this class.
    """
    
    def __init__(
        self, 
        message: str, 
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
        suggestion: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional context
            error_code: Optional error code (auto-generated if not provided)
            suggestion: Optional actionable suggestion for resolution
            **kwargs: Additional key-value pairs to include in details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.details.update(kwargs)
        self.error_code = error_code or self._generate_error_code()
        self.suggestion = suggestion
    
    def _generate_error_code(self) -> str:
        """Generate error code from class name (CamelCase → SCREAMING_SNAKE)."""
        name = self.__class__.__name__
        result = []
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                result.append("_")
            result.append(char.upper())
        return "".join(result).replace("_ERROR", "")
    
    @property
    def error_type(self) -> str:
        """Get the exception class name."""
        return self.__class__.__name__
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        result = {
            "error_type": self.error_type,
            "error_code": self.error_code,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        if self.suggestion:
            result["suggestion"] = self.suggestion
        return result
    
    def to_json(self) -> str:
        """Convert exception to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def __str__(self) -> str:
        """Format error message with details."""
        parts = [f"[{self.error_code}] {self.message}"]
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)
    
    def __repr__(self) -> str:
        if self.details:
            return f"{self.error_type}({self.message!r}, details={self.details!r})"
        return f"{self.error_type}({self.message!r})"


# ============================================================================
# CALCULATION ERRORS
# ============================================================================

class CalculationError(FinAnalystError):
    """
    Base exception for calculation-related errors.
    
    Raised when a financial calculation cannot be completed
    due to mathematical issues or invalid inputs.
    """
    
    def __init__(
        self,
        message: str,
        metric_name: str | None = None,
        formula: str | None = None,
        suggestion: str | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if metric_name:
            details["metric_name"] = metric_name
        if formula:
            details["formula"] = formula
        super().__init__(
            message, 
            details=details, 
            suggestion=suggestion or "Check input values and try again"
        )


class DivisionByZeroError(CalculationError):
    """
    Raised when a calculation would result in division by zero.
    
    Includes information about the numerator and denominator
    to aid in debugging data issues.
    """
    
    def __init__(
        self,
        numerator: Any,
        denominator_name: str = "denominator",
        metric_name: str | None = None,
        **kwargs: Any
    ) -> None:
        message = f"Cannot divide {numerator} by zero ({denominator_name} is zero)"
        super().__init__(
            message,
            metric_name=metric_name,
            suggestion=f"Ensure {denominator_name} is non-zero before calculation",
            numerator=numerator,
            denominator_name=denominator_name,
            **kwargs
        )


class InvalidInputError(CalculationError):
    """
    Raised when input values are invalid for calculation.
    
    Examples:
    - Negative values where positive required
    - Wrong data types
    - Values outside acceptable ranges
    """
    
    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        actual_value: Any = None,
        expected: str | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if field_name:
            details["field_name"] = field_name
        if actual_value is not None:
            details["actual_value"] = actual_value
        if expected:
            details["expected"] = expected
        
        suggestion = f"Provide a value that is: {expected}" if expected else None
        super().__init__(message, suggestion=suggestion, **details)


# ============================================================================
# VALIDATION ERRORS
# ============================================================================

class ValidationError(FinAnalystError):
    """
    Base exception for validation-related errors.
    
    Raised when data fails validation checks.
    """
    
    def __init__(
        self,
        message: str,
        field: str | None = None,
        validation_type: str | None = None,
        suggestion: str | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if field:
            details["field"] = field
        if validation_type:
            details["validation_type"] = validation_type
        super().__init__(message, details=details, suggestion=suggestion)


class SchemaValidationError(ValidationError):
    """Raised when data doesn't conform to expected schema."""
    
    def __init__(
        self,
        message: str,
        schema_name: str | None = None,
        field_errors: dict[str, str] | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if schema_name:
            details["schema_name"] = schema_name
        if field_errors:
            details["field_errors"] = field_errors
        super().__init__(
            message,
            validation_type="schema",
            suggestion="Verify data structure matches expected schema",
            **details
        )


class DataCompletenessError(ValidationError):
    """Raised when required data is missing for an analysis."""
    
    def __init__(
        self,
        analysis_type: str,
        missing_fields: list[str],
        **kwargs: Any
    ) -> None:
        message = (
            f"Insufficient data for {analysis_type} analysis. "
            f"Missing: {', '.join(missing_fields)}"
        )
        super().__init__(
            message,
            validation_type="completeness",
            suggestion=f"Provide the following fields: {', '.join(missing_fields)}",
            analysis_type=analysis_type,
            missing_fields=missing_fields,
            **kwargs
        )


class ReconciliationError(ValidationError):
    """
    Raised when cross-statement reconciliation fails.
    
    Indicates that values that should match across statements
    are inconsistent beyond acceptable tolerance.
    """
    
    def __init__(
        self,
        message: str,
        check_name: str,
        expected_value: Any,
        actual_value: Any,
        difference: Any = None,
        tolerance: float | None = None,
        **kwargs: Any
    ) -> None:
        details = {
            "check_name": check_name,
            "expected_value": expected_value,
            "actual_value": actual_value,
        }
        if difference is not None:
            details["difference"] = difference
        if tolerance is not None:
            details["tolerance_used"] = tolerance
        details.update(kwargs)
        
        super().__init__(
            message,
            validation_type="reconciliation",
            suggestion="Verify data accuracy or confirm known discrepancy",
            **details
        )


class PlausibilityError(ValidationError):
    """
    Raised when a calculated metric is outside plausible range.
    
    Note: This is typically a warning, not an error, unless explicitly strict.
    """
    
    def __init__(
        self,
        metric_name: str,
        value: float,
        plausible_range: tuple[float, float],
        **kwargs: Any
    ) -> None:
        message = (
            f"{metric_name} value of {value:.2f} is outside the plausible range "
            f"({plausible_range[0]:.2f} to {plausible_range[1]:.2f})"
        )
        super().__init__(
            message,
            validation_type="plausibility",
            suggestion="Verify input data accuracy. This may indicate a data entry error.",
            metric_name=metric_name,
            value=value,
            min_plausible=plausible_range[0],
            max_plausible=plausible_range[1],
            **kwargs
        )


# ============================================================================
# DATA ERRORS
# ============================================================================

class DataError(FinAnalystError):
    """Base exception for data-related errors."""
    pass


class DataParsingError(DataError):
    """
    Raised when data cannot be parsed from input format.
    """
    
    def __init__(
        self,
        message: str,
        source: str | None = None,
        line_number: int | None = None,
        raw_data: str | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if source:
            details["source"] = source
        if line_number is not None:
            details["line_number"] = line_number
        if raw_data:
            details["raw_data"] = raw_data[:500] if len(raw_data) > 500 else raw_data
        super().__init__(
            message, 
            details=details,
            suggestion="Check source format and encoding"
        )


class MissingDataError(DataError):
    """Raised when required data is missing."""
    
    def __init__(
        self,
        message: str,
        missing_fields: list[str] | None = None,
        required_for: str | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if missing_fields:
            details["missing_fields"] = missing_fields
        if required_for:
            details["required_for"] = required_for
        
        suggestion = None
        if missing_fields:
            suggestion = f"Provide the following fields: {', '.join(missing_fields)}"
        
        super().__init__(message, details=details, suggestion=suggestion)


# ============================================================================
# TOOL ERRORS
# ============================================================================

class ToolError(FinAnalystError):
    """Base exception for tool-related errors."""
    
    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if tool_name:
            details["tool_name"] = tool_name
        super().__init__(message, details=details)


class ToolNotFoundError(ToolError):
    """Raised when a requested tool does not exist."""
    
    def __init__(
        self,
        tool_name: str,
        available_tools: list[str] | None = None,
        **kwargs: Any
    ) -> None:
        suggestions = self._find_similar(tool_name, available_tools or [])
        message = f"Tool '{tool_name}' not found"
        if suggestions:
            message += f". Did you mean: {', '.join(suggestions)}?"
        
        super().__init__(
            message,
            tool_name=tool_name,
            suggestion=f"Use one of the available tools",
            available_tools=available_tools[:10] if available_tools else None,
            **kwargs
        )
    
    @staticmethod
    def _find_similar(name: str, available: list[str]) -> list[str]:
        """Find tools with similar names."""
        name_lower = name.lower()
        similar = [
            t for t in available 
            if name_lower in t.lower() or t.lower() in name_lower
        ]
        return similar[:3] if similar else available[:3]


class ToolExecutionError(ToolError):
    """Raised when a tool fails during execution."""
    
    def __init__(
        self,
        tool_name: str,
        original_error: Exception,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> None:
        message = f"Tool '{tool_name}' failed: {str(original_error)}"
        super().__init__(
            message,
            tool_name=tool_name,
            suggestion="Check tool parameters and try again",
            original_error_type=type(original_error).__name__,
            original_error_message=str(original_error),
            parameters=parameters,
            **kwargs
        )
        self.original_error = original_error


class ToolParameterError(ToolError):
    """Raised when tool parameters are invalid."""
    
    def __init__(
        self,
        tool_name: str,
        parameter_name: str,
        message: str,
        expected_type: str | None = None,
        actual_value: Any = None,
        **kwargs: Any
    ) -> None:
        full_message = f"Parameter '{parameter_name}' for tool '{tool_name}': {message}"
        super().__init__(
            full_message,
            tool_name=tool_name,
            suggestion=f"Provide a valid value for '{parameter_name}'",
            parameter_name=parameter_name,
            expected_type=expected_type,
            actual_value=str(actual_value)[:100] if actual_value is not None else None,
            **kwargs
        )
