"""
Custom exception hierarchy for FinAnalyst-Pro tools.

This module defines a structured exception hierarchy that enables:
- Precise error categorization
- Detailed error context
- Actionable error messages for LLM consumption
- Consistent error handling across the package
"""

from __future__ import annotations

from typing import Any


class FinAnalystError(Exception):
    """
    Base exception for all FinAnalyst-Pro errors.
    
    All custom exceptions inherit from this class, enabling
    catch-all handling when needed.
    
    Attributes:
        message: Human-readable error description
        details: Additional context as key-value pairs
        error_code: Machine-readable error identifier
        suggestion: Actionable suggestion for resolution
    """
    
    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
        suggestion: str | None = None,
    ):
        self.message = message
        self.details = details or {}
        self.error_code = error_code or self._default_error_code()
        self.suggestion = suggestion
        super().__init__(self.message)
    
    def _default_error_code(self) -> str:
        """Generate default error code from class name."""
        # Convert CamelCase to SCREAMING_SNAKE_CASE
        name = self.__class__.__name__
        result = []
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                result.append("_")
            result.append(char.upper())
        return "".join(result).replace("_ERROR", "")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "suggestion": self.suggestion,
        }
    
    def __str__(self) -> str:
        """Format error message with details."""
        parts = [self.message]
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"Details: {details_str}")
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)


# ============================================================================
# CALCULATION ERRORS
# ============================================================================

class CalculationError(FinAnalystError):
    """
    Raised when a financial calculation cannot be completed.
    
    This covers general calculation failures that aren't more specifically
    categorized (e.g., not division by zero, not invalid input).
    """
    
    def __init__(
        self,
        message: str,
        metric_name: str | None = None,
        details: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ):
        details = details or {}
        if metric_name:
            details["metric_name"] = metric_name
        super().__init__(
            message=message,
            details=details,
            error_code="CALC_FAILED",
            suggestion=suggestion or "Check input values and try again",
        )
        self.metric_name = metric_name


class DivisionByZeroError(CalculationError):
    """
    Raised when division by zero is attempted in a calculation.
    
    This is a specific case of CalculationError that provides
    context about which values caused the issue.
    """
    
    def __init__(
        self,
        numerator: Any,
        denominator_name: str = "denominator",
        metric_name: str | None = None,
    ):
        message = f"Cannot divide {numerator} by zero ({denominator_name} is zero)"
        super().__init__(
            message=message,
            metric_name=metric_name,
            details={
                "numerator": numerator,
                "denominator_name": denominator_name,
            },
            suggestion=f"Ensure {denominator_name} is non-zero before calculation",
        )
        self.error_code = "DIVISION_BY_ZERO"


class InvalidInputError(CalculationError):
    """
    Raised when input values are invalid for the requested calculation.
    
    Examples:
    - Negative values where only positive are valid
    - Non-numeric values where numbers are required
    - Missing required inputs
    """
    
    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        actual_value: Any = None,
        expected: str | None = None,
    ):
        details = {}
        if field_name:
            details["field_name"] = field_name
        if actual_value is not None:
            details["actual_value"] = actual_value
        if expected:
            details["expected"] = expected
        
        suggestion = None
        if expected:
            suggestion = f"Provide a value that is: {expected}"
        
        super().__init__(
            message=message,
            details=details,
            suggestion=suggestion,
        )
        self.error_code = "INVALID_INPUT"
        self.field_name = field_name
        self.actual_value = actual_value
        self.expected = expected


# ============================================================================
# VALIDATION ERRORS
# ============================================================================

class ValidationError(FinAnalystError):
    """
    Raised when data validation fails.
    
    This covers schema validation, completeness checks, and
    data quality issues.
    """
    
    def __init__(
        self,
        message: str,
        field_errors: dict[str, str] | None = None,
        missing_fields: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ):
        details = details or {}
        if field_errors:
            details["field_errors"] = field_errors
        if missing_fields:
            details["missing_fields"] = missing_fields
        
        suggestion = None
        if missing_fields:
            suggestion = f"Provide the following fields: {', '.join(missing_fields)}"
        
        super().__init__(
            message=message,
            details=details,
            error_code="VALIDATION_FAILED",
            suggestion=suggestion,
        )
        self.field_errors = field_errors or {}
        self.missing_fields = missing_fields or []


class SchemaValidationError(ValidationError):
    """Raised when data doesn't conform to expected schema."""
    
    def __init__(
        self,
        message: str,
        schema_name: str | None = None,
        field_errors: dict[str, str] | None = None,
    ):
        super().__init__(
            message=message,
            field_errors=field_errors,
            details={"schema_name": schema_name} if schema_name else None,
        )
        self.error_code = "SCHEMA_INVALID"
        self.schema_name = schema_name


class DataCompletenessError(ValidationError):
    """Raised when required data is missing for an analysis."""
    
    def __init__(
        self,
        analysis_type: str,
        missing_fields: list[str],
    ):
        message = (
            f"Insufficient data for {analysis_type} analysis. "
            f"Missing: {', '.join(missing_fields)}"
        )
        super().__init__(
            message=message,
            missing_fields=missing_fields,
            details={"analysis_type": analysis_type},
        )
        self.error_code = "DATA_INCOMPLETE"
        self.analysis_type = analysis_type


# ============================================================================
# RECONCILIATION ERRORS
# ============================================================================

class ReconciliationError(FinAnalystError):
    """
    Raised when cross-statement reconciliation fails.
    
    This indicates an inconsistency between related values
    across different financial statements.
    """
    
    def __init__(
        self,
        message: str,
        check_name: str,
        expected_value: Any,
        actual_value: Any,
        difference: Any = None,
        tolerance: float | None = None,
    ):
        details = {
            "check_name": check_name,
            "expected_value": expected_value,
            "actual_value": actual_value,
        }
        if difference is not None:
            details["difference"] = difference
        if tolerance is not None:
            details["tolerance_used"] = tolerance
        
        super().__init__(
            message=message,
            details=details,
            error_code="RECONCILIATION_FAILED",
            suggestion="Verify data accuracy or confirm known discrepancy",
        )
        self.check_name = check_name
        self.expected_value = expected_value
        self.actual_value = actual_value


# ============================================================================
# DATA PARSING ERRORS
# ============================================================================

class DataParsingError(FinAnalystError):
    """
    Raised when financial data cannot be parsed.
    
    This covers issues with raw data extraction from
    PDFs, Excel files, or other sources.
    """
    
    def __init__(
        self,
        message: str,
        source_type: str | None = None,
        source_location: str | None = None,
        parse_error: str | None = None,
    ):
        details = {}
        if source_type:
            details["source_type"] = source_type
        if source_location:
            details["source_location"] = source_location
        if parse_error:
            details["parse_error"] = parse_error
        
        super().__init__(
            message=message,
            details=details,
            error_code="PARSE_FAILED",
            suggestion="Check source format and try again",
        )
        self.source_type = source_type


# ============================================================================
# TOOL EXECUTION ERRORS
# ============================================================================

class ToolExecutionError(FinAnalystError):
    """
    Raised when a tool execution fails.
    
    This wraps errors that occur during tool dispatch and execution,
    providing context about which tool failed and why.
    """
    
    def __init__(
        self,
        message: str,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        details = {"tool_name": tool_name}
        if parameters:
            details["parameters"] = parameters
        if original_error:
            details["original_error"] = str(original_error)
            details["original_error_type"] = type(original_error).__name__
        
        super().__init__(
            message=message,
            details=details,
            error_code="TOOL_EXECUTION_FAILED",
            suggestion="Check tool parameters and try again",
        )
        self.tool_name = tool_name
        self.parameters = parameters
        self.original_error = original_error


class UnknownToolError(ToolExecutionError):
    """Raised when an unknown tool is requested."""
    
    def __init__(
        self,
        tool_name: str,
        available_tools: list[str] | None = None,
    ):
        message = f"Unknown tool: '{tool_name}'"
        super().__init__(
            message=message,
            tool_name=tool_name,
        )
        self.error_code = "UNKNOWN_TOOL"
        if available_tools:
            self.details["available_tools"] = available_tools[:10]  # Limit list size
            self.suggestion = f"Use one of the available tools. Similar: {self._find_similar(tool_name, available_tools)}"
    
    def _find_similar(self, name: str, available: list[str]) -> list[str]:
        """Find tools with similar names."""
        name_lower = name.lower()
        similar = [t for t in available if name_lower in t.lower() or t.lower() in name_lower]
        return similar[:3] if similar else available[:3]


class ToolParameterError(ToolExecutionError):
    """Raised when tool parameters are invalid."""
    
    def __init__(
        self,
        tool_name: str,
        parameter_name: str,
        message: str,
        expected_type: str | None = None,
        actual_value: Any = None,
    ):
        details = {
            "parameter_name": parameter_name,
        }
        if expected_type:
            details["expected_type"] = expected_type
        if actual_value is not None:
            details["actual_value"] = str(actual_value)[:100]  # Truncate long values
        
        super().__init__(
            message=f"Parameter '{parameter_name}': {message}",
            tool_name=tool_name,
            parameters=details,
        )
        self.error_code = "INVALID_PARAMETER"
        self.parameter_name = parameter_name


# ============================================================================
# PLAUSIBILITY ERRORS
# ============================================================================

class PlausibilityError(FinAnalystError):
    """
    Raised when a calculated value fails plausibility checks.
    
    Note: This is typically a warning, not a blocking error.
    Only raised when explicitly requested to enforce plausibility.
    """
    
    def __init__(
        self,
        metric_name: str,
        value: float,
        plausible_range: tuple[float, float],
    ):
        message = (
            f"{metric_name} value of {value:.2f} is outside the plausible range "
            f"({plausible_range[0]:.2f} to {plausible_range[1]:.2f})"
        )
        super().__init__(
            message=message,
            details={
                "metric_name": metric_name,
                "value": value,
                "min_plausible": plausible_range[0],
                "max_plausible": plausible_range[1],
            },
            error_code="IMPLAUSIBLE_VALUE",
            suggestion="Verify input data accuracy. This may indicate a data entry error.",
        )
        self.metric_name = metric_name
        self.value = value
        self.plausible_range = plausible_range
