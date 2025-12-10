# finanalyst_tools/exceptions.py
"""
Custom exception hierarchy for FinAnalyst-Pro tools.

Provides specific exception types for different error categories:
- Calculation errors (arithmetic, division by zero, invalid inputs)
- Validation errors (schema, reconciliation, plausibility)
- Data errors (parsing, missing data)
- Tool errors (not found, execution failure)

All exceptions support serialization to dict/JSON for structured error handling.
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
    - Message storage
    - Optional details dictionary for context
    - JSON serialization support
    
    All custom exceptions should inherit from this class.
    """
    
    def __init__(
        self, 
        message: str, 
        details: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional context
            **kwargs: Additional key-value pairs to include in details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.details.update(kwargs)
    
    @property
    def error_type(self) -> str:
        """Get the exception class name."""
        return self.__class__.__name__
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert exception to dictionary for JSON serialization.
        
        Returns:
            Dictionary with error_type, message, and details
        """
        return {
            "error_type": self.error_type,
            "message": self.message,
            "details": self.details,
        }
    
    def to_json(self) -> str:
        """
        Convert exception to JSON string.
        
        Returns:
            JSON representation of the error
        """
        return json.dumps(self.to_dict(), indent=2, default=str)
    
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
        **kwargs: Any
    ) -> None:
        """
        Initialize calculation error.
        
        Args:
            message: Error description
            metric_name: Name of the metric being calculated
            formula: The formula that failed
            **kwargs: Additional context
        """
        details = kwargs
        if metric_name:
            details["metric_name"] = metric_name
        if formula:
            details["formula"] = formula
        super().__init__(message, details=details)


class DivisionByZeroError(CalculationError):
    """
    Raised when a calculation would result in division by zero.
    
    Includes information about the numerator and denominator
    to aid in debugging data issues.
    """
    
    def __init__(
        self,
        numerator: Any,
        denominator: Any,
        metric_name: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize division by zero error.
        
        Args:
            numerator: The dividend value
            denominator: The divisor (zero)
            metric_name: Name of the metric being calculated
            **kwargs: Additional context
        """
        message = f"Cannot divide {numerator} by zero"
        super().__init__(
            message,
            metric_name=metric_name,
            numerator=numerator,
            denominator=denominator,
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
        parameter_name: str | None = None,
        received_value: Any = None,
        expected: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize invalid input error.
        
        Args:
            message: Error description
            parameter_name: Name of the invalid parameter
            received_value: The value that was received
            expected: Description of what was expected
            **kwargs: Additional context
        """
        details = kwargs
        if parameter_name:
            details["parameter_name"] = parameter_name
        if received_value is not None:
            details["received_value"] = received_value
        if expected:
            details["expected"] = expected
        super().__init__(message, **details)


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
        **kwargs: Any
    ) -> None:
        """
        Initialize validation error.
        
        Args:
            message: Error description
            field: The field that failed validation
            validation_type: Type of validation that failed
            **kwargs: Additional context
        """
        details = kwargs
        if field:
            details["field"] = field
        if validation_type:
            details["validation_type"] = validation_type
        super().__init__(message, details=details)


class SchemaError(ValidationError):
    """
    Raised when data doesn't conform to expected schema.
    
    Typically occurs during Pydantic model validation.
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
        super().__init__(
            message,
            field=field,
            validation_type="schema",
            expected_type=expected_type,
            received_type=received_type,
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
            check_name=check_name,
            value_a=value_a,
            source_a=source_a,
            value_b=value_b,
            source_b=source_b,
            difference=difference,
            tolerance=tolerance,
            **kwargs
        )


class PlausibilityError(ValidationError):
    """
    Raised when a calculated metric is outside plausible range.
    
    Note: This is typically a warning, not an error, unless explicitly strict.
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
            metric_name=metric_name,
            value=value,
            expected_range=expected_range,
            **kwargs
        )


# ============================================================================
# DATA ERRORS
# ============================================================================

class DataError(FinAnalystError):
    """
    Base exception for data-related errors.
    
    Raised when there are issues with the input data itself.
    """
    pass


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
            details["raw_data"] = raw_data[:500] if len(raw_data) > 500 else raw_data
        super().__init__(message, details=details)


class MissingDataError(DataError):
    """
    Raised when required data is missing.
    
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
        details = kwargs
        if missing_fields:
            details["missing_fields"] = missing_fields
        if required_for:
            details["required_for"] = required_for
        super().__init__(message, details=details)


# ============================================================================
# TOOL ERRORS
# ============================================================================

class ToolError(FinAnalystError):
    """
    Base exception for tool-related errors.
    
    Raised when issues occur during tool execution.
    """
    
    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize tool error.
        
        Args:
            message: Error description
            tool_name: Name of the tool
            **kwargs: Additional context
        """
        details = kwargs
        if tool_name:
            details["tool_name"] = tool_name
        super().__init__(message, details=details)


class ToolNotFoundError(ToolError):
    """
    Raised when a requested tool does not exist.
    
    Includes suggestions for similar tool names if available.
    """
    
    def __init__(
        self,
        tool_name: str,
        available_tools: list[str] | None = None,
        suggestions: list[str] | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize tool not found error.
        
        Args:
            tool_name: Name of the tool that wasn't found
            available_tools: List of all available tool names
            suggestions: Similar tool names as suggestions
            **kwargs: Additional context
        """
        message = f"Tool '{tool_name}' not found"
        if suggestions:
            message += f". Did you mean: {', '.join(suggestions)}?"
        
        super().__init__(
            message,
            tool_name=tool_name,
            available_tools=available_tools,
            suggestions=suggestions,
            **kwargs
        )


class ToolExecutionError(ToolError):
    """
    Raised when a tool fails during execution.
    
    Wraps the original exception with tool context.
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
        message = f"Tool '{tool_name}' failed: {str(original_error)}"
        
        super().__init__(
            message,
            tool_name=tool_name,
            original_error_type=type(original_error).__name__,
            original_error_message=str(original_error),
            parameters=parameters,
            **kwargs
        )
        self.original_error = original_error


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
        
        super().__init__(
            full_message,
            tool_name=tool_name,
            parameter_name=parameter_name,
            received_value=received_value,
            expected_type=expected_type,
            **kwargs
        )
