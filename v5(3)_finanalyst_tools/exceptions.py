# finanalyst_tools/exceptions.py
"""
Custom exception hierarchy for FinAnalyst-Pro Agent Tools.

This module provides a structured exception hierarchy that enables:
- Precise error categorization for programmatic handling
- Detailed error context for debugging
- Actionable error messages and suggestions for LLM consumption
- Consistent error handling across the package
- JSON serialization for structured error responses

Exception Hierarchy:
    FinAnalystError (base)
    ├── CalculationError
    │   ├── DivisionByZeroError
    │   └── InvalidInputError
    ├── ValidationError
    │   ├── SchemaError
    │   ├── ReconciliationError
    │   └── PlausibilityError
    ├── DataError
    │   ├── DataParsingError
    │   ├── MissingDataError
    │   └── DataCompletenessError
    └── ToolError
        ├── ToolNotFoundError
        ├── ToolExecutionError
        └── ToolParameterError

Author: FinAnalyst-Pro Team
Version: 1.0.0
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
    
    Provides common functionality:
    - Message storage with optional details dictionary
    - Auto-generated error codes for programmatic handling
    - Actionable suggestions for resolution
    - JSON serialization support for LLM responses
    
    All custom exceptions should inherit from this class.
    
    Attributes:
        message: Human-readable error description
        details: Additional context as key-value pairs
        error_code: Machine-readable error identifier (auto-generated if not provided)
        suggestion: Actionable suggestion for resolution
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
            error_code: Machine-readable error code (auto-generated if None)
            suggestion: Actionable suggestion for resolving the error
            **kwargs: Additional key-value pairs to include in details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.details.update(kwargs)
        self.error_code = error_code or self._generate_error_code()
        self.suggestion = suggestion
    
    def _generate_error_code(self) -> str:
        """
        Generate default error code from class name.
        
        Converts CamelCase to SCREAMING_SNAKE_CASE.
        Example: DivisionByZeroError -> DIVISION_BY_ZERO
        """
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
        """
        Convert exception to dictionary for JSON serialization.
        
        Returns:
            Dictionary with error_type, error_code, message, details, and suggestion
        """
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
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert exception to JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            JSON representation of the error
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def __str__(self) -> str:
        """Format error message with optional suggestion."""
        parts = [self.message]
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)
    
    def __repr__(self) -> str:
        if self.details:
            return f"{self.error_type}({self.message!r}, error_code={self.error_code!r}, details={self.details!r})"
        return f"{self.error_type}({self.message!r}, error_code={self.error_code!r})"


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
        """
        Initialize calculation error.
        
        Args:
            message: Error description
            metric_name: Name of the metric being calculated
            formula: The formula that failed
            suggestion: How to fix the issue
            **kwargs: Additional context
        """
        details = kwargs
        if metric_name:
            details["metric_name"] = metric_name
        if formula:
            details["formula"] = formula
        
        super().__init__(
            message,
            details=details,
            error_code="CALCULATION_ERROR",
            suggestion=suggestion or "Check input values and try again",
        )
        self.metric_name = metric_name
        self.formula = formula


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
        """
        Initialize division by zero error.
        
        Args:
            numerator: The dividend value
            denominator_name: Name of the divisor field
            metric_name: Name of the metric being calculated
            **kwargs: Additional context
        """
        message = f"Cannot divide {numerator} by zero ({denominator_name} is zero)"
        
        super().__init__(
            message,
            metric_name=metric_name,
            suggestion=f"Ensure {denominator_name} is non-zero before calculation",
            numerator=numerator,
            denominator_name=denominator_name,
            **kwargs
        )
        self.error_code = "DIVISION_BY_ZERO"
        self.numerator = numerator
        self.denominator_name = denominator_name


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
        received_value: Any = None,
        expected: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize invalid input error.
        
        Args:
            message: Error description
            field_name: Name of the invalid field
            received_value: The value that was received
            expected: Description of what was expected
            **kwargs: Additional context
        """
        suggestion = f"Provide a value that is: {expected}" if expected else None
        
        super().__init__(
            message,
            suggestion=suggestion,
            field_name=field_name,
            received_value=received_value,
            expected=expected,
            **kwargs
        )
        self.error_code = "INVALID_INPUT"
        self.field_name = field_name
        self.received_value = received_value
        self.expected = expected


# ============================================================================
# VALIDATION ERRORS
# ============================================================================

class ValidationError(FinAnalystError):
    """
    Base exception for validation-related errors.
    
    Raised when data fails validation checks (schema, reconciliation, etc.).
    """
    
    def __init__(
        self,
        message: str,
        field: str | None = None,
        validation_type: str | None = None,
        suggestion: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize validation error.
        
        Args:
            message: Error description
            field: The field that failed validation
            validation_type: Type of validation that failed
            suggestion: How to fix the issue
            **kwargs: Additional context
        """
        details = kwargs
        if field:
            details["field"] = field
        if validation_type:
            details["validation_type"] = validation_type
        
        super().__init__(
            message,
            details=details,
            error_code="VALIDATION_ERROR",
            suggestion=suggestion,
        )
        self.field = field
        self.validation_type = validation_type


class SchemaError(ValidationError):
    """
    Raised when data doesn't conform to expected schema.
    
    Typically occurs during Pydantic model validation or when
    required fields are missing or have wrong types.
    """
    
    def __init__(
        self,
        message: str,
        field: str | None = None,
        expected_type: str | None = None,
        received_type: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize schema error.
        
        Args:
            message: Error description
            field: The field with schema error
            expected_type: Expected data type
            received_type: Actual data type received
            **kwargs: Additional context
        """
        suggestion = None
        if expected_type and received_type:
            suggestion = f"Convert the value to {expected_type}"
        
        super().__init__(
            message,
            field=field,
            validation_type="schema",
            suggestion=suggestion,
            expected_type=expected_type,
            received_type=received_type,
            **kwargs
        )
        self.error_code = "SCHEMA_ERROR"
        self.expected_type = expected_type
        self.received_type = received_type


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
        value_a: Any,
        source_a: str,
        value_b: Any,
        source_b: str,
        difference: Any = None,
        tolerance: float | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize reconciliation error.
        
        Args:
            message: Error description
            check_name: Name of the reconciliation check
            value_a: First value
            source_a: Source of first value (e.g., "Income Statement")
            value_b: Second value
            source_b: Source of second value (e.g., "Cash Flow Statement")
            difference: Calculated difference
            tolerance: Tolerance threshold that was exceeded
            **kwargs: Additional context
        """
        super().__init__(
            message,
            validation_type="reconciliation",
            suggestion="Verify data accuracy in both statements or confirm this is a known discrepancy",
            check_name=check_name,
            value_a=value_a,
            source_a=source_a,
            value_b=value_b,
            source_b=source_b,
            difference=difference,
            tolerance=tolerance,
            **kwargs
        )
        self.error_code = "RECONCILIATION_ERROR"
        self.check_name = check_name
        self.value_a = value_a
        self.source_a = source_a
        self.value_b = value_b
        self.source_b = source_b
        self.difference = difference
        self.tolerance = tolerance


class PlausibilityError(ValidationError):
    """
    Raised when a calculated metric is outside plausible range.
    
    Note: This is typically a warning, not a blocking error,
    unless explicitly configured to be strict.
    """
    
    def __init__(
        self,
        message: str,
        metric_name: str,
        value: Any,
        expected_range: tuple[float, float],
        **kwargs: Any
    ) -> None:
        """
        Initialize plausibility error.
        
        Args:
            message: Error description
            metric_name: Name of the metric
            value: The implausible value
            expected_range: Tuple of (min, max) expected values
            **kwargs: Additional context
        """
        super().__init__(
            message,
            validation_type="plausibility",
            suggestion="Verify input data accuracy. This may indicate a data entry error.",
            metric_name=metric_name,
            value=value,
            expected_range=expected_range,
            **kwargs
        )
        self.error_code = "PLAUSIBILITY_ERROR"
        self.metric_name = metric_name
        self.value = value
        self.expected_range = expected_range


# ============================================================================
# DATA ERRORS
# ============================================================================

class DataError(FinAnalystError):
    """
    Base exception for data-related errors.
    
    Raised when there are issues with the input data itself.
    """
    
    def __init__(
        self,
        message: str,
        suggestion: str | None = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            message,
            details=kwargs,
            error_code="DATA_ERROR",
            suggestion=suggestion,
        )


class DataParsingError(DataError):
    """
    Raised when data cannot be parsed from input format.
    
    Examples:
    - Invalid JSON/CSV structure
    - Corrupted file data
    - Encoding issues
    """
    
    def __init__(
        self,
        message: str,
        source: str | None = None,
        line_number: int | None = None,
        raw_data: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize parsing error.
        
        Args:
            message: Error description
            source: Source of the data (filename, URL, etc.)
            line_number: Line number where error occurred
            raw_data: Snippet of raw data that failed to parse
            **kwargs: Additional context
        """
        details = kwargs
        if source:
            details["source"] = source
        if line_number is not None:
            details["line_number"] = line_number
        if raw_data:
            # Truncate if too long
            details["raw_data"] = raw_data[:200] + "..." if len(raw_data) > 200 else raw_data
        
        super().__init__(
            message,
            suggestion="Check source format and encoding, then try again",
            **details
        )
        self.error_code = "DATA_PARSING_ERROR"
        self.source = source
        self.line_number = line_number


class MissingDataError(DataError):
    """
    Raised when required data fields are missing.
    
    Includes information about what data is needed and
    which analysis requires it.
    """
    
    def __init__(
        self,
        message: str,
        missing_fields: list[str] | None = None,
        required_for: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize missing data error.
        
        Args:
            message: Error description
            missing_fields: List of missing field names
            required_for: What analysis/calculation requires the data
            **kwargs: Additional context
        """
        suggestion = None
        if missing_fields:
            suggestion = f"Provide the following fields: {', '.join(missing_fields)}"
        
        super().__init__(
            message,
            suggestion=suggestion,
            missing_fields=missing_fields,
            required_for=required_for,
            **kwargs
        )
        self.error_code = "MISSING_DATA"
        self.missing_fields = missing_fields or []
        self.required_for = required_for


class DataCompletenessError(DataError):
    """
    Raised when data is insufficient for the requested analysis type.
    
    More specific than MissingDataError - indicates that while some
    data may be present, it's not sufficient for the analysis.
    """
    
    def __init__(
        self,
        analysis_type: str,
        missing_fields: list[str],
        available_fields: list[str] | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize data completeness error.
        
        Args:
            analysis_type: Type of analysis that cannot be performed
            missing_fields: List of missing field names
            available_fields: List of fields that are available
            **kwargs: Additional context
        """
        message = (
            f"Insufficient data for {analysis_type} analysis. "
            f"Missing: {', '.join(missing_fields)}"
        )
        
        super().__init__(
            message,
            suggestion=f"Provide the missing fields to enable {analysis_type} analysis",
            analysis_type=analysis_type,
            missing_fields=missing_fields,
            available_fields=available_fields,
            **kwargs
        )
        self.error_code = "DATA_INCOMPLETE"
        self.analysis_type = analysis_type
        self.missing_fields = missing_fields
        self.available_fields = available_fields or []


# ============================================================================
# TOOL ERRORS
# ============================================================================

class ToolError(FinAnalystError):
    """
    Base exception for tool-related errors.
    
    Raised when issues occur during tool discovery or execution.
    """
    
    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        suggestion: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize tool error.
        
        Args:
            message: Error description
            tool_name: Name of the tool
            suggestion: How to fix the issue
            **kwargs: Additional context
        """
        details = kwargs
        if tool_name:
            details["tool_name"] = tool_name
        
        super().__init__(
            message,
            details=details,
            error_code="TOOL_ERROR",
            suggestion=suggestion,
        )
        self.tool_name = tool_name


class ToolNotFoundError(ToolError):
    """
    Raised when a requested tool does not exist.
    
    Includes suggestions for similar tool names if available.
    """
    
    def __init__(
        self,
        tool_name: str,
        available_tools: list[str] | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize tool not found error.
        
        Args:
            tool_name: Name of the tool that wasn't found
            available_tools: List of all available tool names
            **kwargs: Additional context
        """
        # Find similar tool names
        suggestions = []
        if available_tools:
            tool_lower = tool_name.lower()
            for t in available_tools:
                t_lower = t.lower()
                # Check for substring match or similar words
                if tool_lower in t_lower or t_lower in tool_lower:
                    suggestions.append(t)
                elif any(word in t_lower for word in tool_lower.split("_")):
                    suggestions.append(t)
            suggestions = suggestions[:5]  # Limit to 5 suggestions
        
        message = f"Tool '{tool_name}' not found"
        suggestion = None
        if suggestions:
            suggestion = f"Did you mean: {', '.join(suggestions)}?"
        elif available_tools:
            suggestion = f"Use list_tools() to see available tools"
        
        super().__init__(
            message,
            tool_name=tool_name,
            suggestion=suggestion,
            available_tools=available_tools[:20] if available_tools else None,
            suggestions=suggestions,
            **kwargs
        )
        self.error_code = "TOOL_NOT_FOUND"
        self.available_tools = available_tools or []
        self.suggestions = suggestions


class ToolExecutionError(ToolError):
    """
    Raised when a tool fails during execution.
    
    Wraps the original exception with tool context for debugging.
    """
    
    def __init__(
        self,
        tool_name: str,
        original_error: Exception,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize tool execution error.
        
        Args:
            tool_name: Name of the tool that failed
            original_error: The original exception that was raised
            parameters: Parameters that were passed to the tool
            **kwargs: Additional context
        """
        message = f"Tool '{tool_name}' execution failed: {str(original_error)}"
        
        super().__init__(
            message,
            tool_name=tool_name,
            suggestion="Check the parameters and input data, then try again",
            original_error_type=type(original_error).__name__,
            original_error_message=str(original_error),
            parameters=parameters,
            **kwargs
        )
        self.error_code = "TOOL_EXECUTION_ERROR"
        self.original_error = original_error
        self.parameters = parameters


class ToolParameterError(ToolError):
    """
    Raised when tool parameters are invalid.
    
    Provides details about which parameters are wrong and why.
    """
    
    def __init__(
        self,
        tool_name: str,
        parameter_name: str,
        message: str,
        received_value: Any = None,
        expected_type: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize tool parameter error.
        
        Args:
            tool_name: Name of the tool
            parameter_name: Name of the invalid parameter
            message: Description of what's wrong
            received_value: The invalid value received
            expected_type: What type was expected
            **kwargs: Additional context
        """
        full_message = f"Invalid parameter '{parameter_name}' for tool '{tool_name}': {message}"
        
        suggestion = None
        if expected_type:
            suggestion = f"Provide a valid {expected_type} value for '{parameter_name}'"
        
        super().__init__(
            full_message,
            tool_name=tool_name,
            suggestion=suggestion,
            parameter_name=parameter_name,
            received_value=str(received_value)[:100] if received_value is not None else None,
            expected_type=expected_type,
            **kwargs
        )
        self.error_code = "TOOL_PARAMETER_ERROR"
        self.parameter_name = parameter_name
        self.received_value = received_value
        self.expected_type = expected_type
