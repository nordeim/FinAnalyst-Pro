# finanalyst_tools/calculations/base.py
"""
Base classes and utilities for financial calculations.

Provides:
- BaseCalculator abstract class for consistent calculation patterns
- Factory function for creating CalculationResult objects with full compatibility
- Helper functions for extracting values from financial statements
- Advanced decimal handling and validation utilities
- Complete integration with ValidationResult system
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal, getcontext
from typing import Any, Callable, Optional, cast

from finanalyst_tools.config import (
    DECIMAL_PLACES,
    PlausibilityRanges,
    METRIC_FORMULAS,
    METRIC_UNITS,
)
from finanalyst_tools.models.analysis_results import (
    CalculationResult,
    MetricUnit,
    MetricCategory,
    MetricCollection,
)
from finanalyst_tools.models.financial_statements import (
    IncomeStatementData,
    BalanceSheetData,
    CashFlowStatementData,
    FinancialStatementSet,
)
from finanalyst_tools.utils.math_ops import (
    to_decimal,
    safe_divide,
    round_decimal,
    calculate_percentage,
    calculate_average,
    is_effectively_zero,
)
from finanalyst_tools.models.validation import (
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
)
from finanalyst_tools.exceptions import (
    CalculationError,
    DivisionByZeroError,
    InvalidInputError,
)


# Set high precision for financial calculations
getcontext().prec = 28


def get_metric_unit(metric_name: str) -> MetricUnit:
    """
    Get the appropriate unit for a metric.
    
    Args:
        metric_name: Name of the metric
        
    Returns:
        MetricUnit enum value
    """
    unit_str = METRIC_UNITS.get(metric_name.lower(), "ratio")
    unit_map = {
        "percentage": MetricUnit.PERCENTAGE,
        "ratio": MetricUnit.RATIO,
        "currency": MetricUnit.CURRENCY,
        "days": MetricUnit.DAYS,
        "count": MetricUnit.COUNT,
        "times": MetricUnit.TIMES,
    }
    return unit_map.get(unit_str, MetricUnit.RATIO)


def get_metric_formula(metric_name: str) -> str:
    """
    Get the formula for a metric.
    
    Args:
        metric_name: Name of the metric
        
    Returns:
        Formula string
    """
    return METRIC_FORMULAS.get(metric_name.lower(), "N/A")


def _format_reasoning_block(result: CalculationResult) -> str:
    """
    Format a CalculationResult as a reasoning block for LLM output.
    
    This implements the required format from the system prompt.
    
    Args:
        result: Calculation result to format
        
    Returns:
        Formatted markdown block
    """
    lines = [
        f"### {result.metric_name}",
        f"**Value**: {result.formatted_value}",
        f"**Formula**: {result.formula}",
        "",
        "**Calculation Steps**:",
    ]
    
    for step in result.calculation_steps:
        lines.append(f"  - {step}")
    
    lines.append("")
    lines.append("**Inputs Used**:")
    for key, val in result.inputs.items():
        if isinstance(val, Decimal):
            lines.append(f"  - {key}: {float(val):,.2f}")
        else:
            lines.append(f"  - {key}: {val}")
    
    if result.warnings:
        lines.append("")
        lines.append("**Warnings**:")
        for warning in result.warnings:
            lines.append(f"  - ⚠️ {warning}")
    
    plausibility_status = "✅ Within range" if result.is_plausible else "⚠️ Outside expected range"
    if result.plausibility_range:
        lines.append(f"\n**Plausibility**: {plausibility_status} ({result.plausibility_range[0]:.1f} to {result.plausibility_range[1]:.1f})")
    
    return "\n".join(lines)


def create_calculation_result(
    metric_name: str,
    value: Decimal | None,
    formula: str,
    inputs: dict[str, Any],
    steps: list[str],
    category: MetricCategory | None = None,
    warnings: list[str] | None = None,
    unit: MetricUnit | None = None,
    currency: str = "USD",
) -> CalculationResult:
    """
    Factory function to create a CalculationResult with full compatibility.
    
    This function now creates CalculationResult objects that are fully compatible
    with the system prompt requirements, including proper formatting methods.
    
    Args:
        metric_name: Name of the metric
        value: Calculated value (or None if calculation failed)
        formula: Formula used for calculation
        inputs: Dictionary of input values used
        steps: List of calculation steps for audit trail
        category: Metric category
        warnings: List of warning messages
        unit: Override metric unit if needed
        currency: Currency code for CURRENCY unit type (default: USD)
        
    Returns:
        Complete CalculationResult with plausibility assessment
    """
    # Get plausibility range
    plausibility_range = PlausibilityRanges.get_range(metric_name)
    
    # Check plausibility
    is_plausible = True
    warning_list = warnings.copy() if warnings else []
    
    if value is not None and plausibility_range is not None:
        float_value = float(value)
        is_plausible = plausibility_range[0] <= float_value <= plausibility_range[1]
        
        if not is_plausible:
            warning_list.append(
                f"Value {float_value:.2f} is outside typical range "
                f"({plausibility_range[0]:.1f} to {plausibility_range[1]:.1f})"
            )
    
    # Determine unit if not provided
    if unit is None:
        unit = get_metric_unit(metric_name)
    
    # Preserve inputs as-is (Decimal-safe). JSON serialization is handled at output boundaries.
    serializable_inputs = dict(inputs)
    
    # Build result with all required properties
    result = CalculationResult(
        metric_name=metric_name,
        value=value,
        unit=unit,
        formula=formula,
        inputs=serializable_inputs,
        calculation_steps=steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=warning_list,
        category=category,
        currency=currency,
    )
    
    return result


def convert_exception_to_validation_result(
    exc: Exception,
    field: str = "calculation",
    context: str = "financial calculation"
) -> ValidationResult:
    """
    Convert an exception to a ValidationResult object.
    
    This bridges the exception hierarchy with the validation system.
    
    Args:
        exc: Exception to convert
        field: Field name for validation issue
        context: Context description for error message
        
    Returns:
        ValidationResult with the error
    """
    result = ValidationResult()
    
    # Handle different exception types
    if isinstance(exc, DivisionByZeroError):
        severity = ValidationSeverity.ERROR
        message = f"Division by zero in {context}: {exc.message}"
        suggestion = exc.suggestion or "Check denominator values are non-zero"
    elif isinstance(exc, InvalidInputError):
        severity = ValidationSeverity.ERROR
        message = f"Invalid input in {context}: {exc.message}"
        suggestion = exc.suggestion or "Verify input data format and values"
    elif isinstance(exc, CalculationError):
        severity = ValidationSeverity.ERROR
        message = f"Calculation error in {context}: {exc.message}"
        suggestion = exc.suggestion or "Review calculation inputs and formula"
    else:
        severity = ValidationSeverity.ERROR
        message = f"Unexpected error in {context}: {str(exc)}"
        suggestion = "Contact support for assistance"
    
    # Get details from exception if available
    details = {}
    if hasattr(exc, 'details'):
        details = getattr(exc, 'details', {})
    elif hasattr(exc, '__dict__'):
        details = exc.__dict__
    
    # Create validation issue
    issue = ValidationIssue(
        field=field,
        message=message,
        severity=severity,
        actual_value=str(details.get('actual_value', 'unknown')),
        expected=str(details.get('expected', 'valid numeric value')),
        suggestion=suggestion
    )
    
    result.add_issue(issue)
    return result


def extract_value(
    data: dict[str, Any] | IncomeStatementData | BalanceSheetData | CashFlowStatementData,
    field_name: str,
    default: Decimal | None = None,
) -> Decimal:
    """
    Extract a value from financial data, handling both dict and model inputs.
    
    Args:
        data: Financial data (dict or Pydantic model)
        field_name: Name of the field to extract
        default: Default value if field not found
        
    Returns:
        Decimal value
    """
    if default is None:
        default = Decimal("0")
    
    # Handle different data types
    if hasattr(data, "model_dump"):
        # Pydantic model
        data_dict = data.model_dump(by_alias=False)
    elif hasattr(data, "__dict__"):
        # Regular object
        data_dict = data.__dict__
    else:
        # Dictionary or other type
        data_dict = data
    
    # Try to get the value
    value = data_dict.get(field_name)
    
    if value is None:
        return default
    
    return to_decimal(value, default=default)


def extract_decimal_value(
    value: Any,
    field_name: str = "value",
    default: Decimal | None = None,
) -> Decimal:
    if value is None:
        return default if default is not None else Decimal("0")

    if isinstance(value, Decimal):
        return value

    if isinstance(value, bool):
        raise InvalidInputError(
            f"Invalid boolean for {field_name}",
            field_name=field_name,
            actual_value=value,
            expected="numeric",
        )

    if isinstance(value, int):
        return Decimal(value)

    if isinstance(value, float):
        return Decimal(str(value))

    if isinstance(value, str):
        return Decimal(value)

    raise InvalidInputError(
        f"Cannot convert {type(value).__name__} to Decimal for {field_name}",
        field_name=field_name,
        actual_value=value,
        expected="numeric",
    )


def validate_calculation_inputs(inputs: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate calculation inputs for common issues.
    
    Args:
        inputs: Dictionary of input values
        
    Returns:
        Tuple of (is_valid, list of warnings)
    """
    warnings = []
    is_valid = True
    
    # Check for negative values where they shouldn't be
    for key, value in inputs.items():
        if isinstance(value, (int, float, Decimal)):
            # Check for negative revenue, assets, etc.
            negative_indicators = [
                "revenue", "sales", "assets", "equity", "cash",
                "income", "profit", "margin", "ratio"
            ]
            if any(indicator in key.lower() for indicator in negative_indicators):
                if value < 0:
                    warnings.append(f"Negative value for {key}: {value}")
                    # Don't mark as invalid, but warn
    
    return is_valid, warnings


class BaseCalculator(ABC):
    """
    Abstract base class for financial calculators.
    
    Provides common functionality:
    - Consistent result creation
    - Step logging
    - Plausibility checking
    - Warning accumulation
    - Input validation
    - Exception handling integration
    """
    
    def __init__(self, category: MetricCategory):
        """
        Initialize the calculator.
        
        Args:
            category: The category of metrics this calculator produces
        """
        self.category = category
        self._current_steps: list[str] = []
        self._current_warnings: list[str] = []
        self._current_inputs: dict[str, Any] = {}
        self._step_counter: int = 1
    
    def _reset(self) -> None:
        """Reset calculation state for a new calculation."""
        self._current_steps = []
        self._current_warnings = []
        self._current_inputs = {}
        self._step_counter = 1
    
    def _add_step(self, step_description: str, value: Optional[Any] = None) -> None:
        """
        Add a calculation step with automatic numbering.
        
        Args:
            step_description: Description of the step
            value: Optional value to display
        """
        step_text = f"Step {self._step_counter}: {step_description}"
        if value is not None:
            if isinstance(value, Decimal):
                step_text += f" = {value:,.4f}"
            else:
                step_text += f" = {value}"
        self._current_steps.append(step_text)
        self._step_counter += 1
    
    def _add_warning(self, warning: str) -> None:
        """
        Add a warning message.
        
        Args:
            warning: Warning message to add
        """
        self._current_warnings.append(warning)
    
    def _record_input(self, name: str, value: Any) -> None:
        """
        Record an input value for audit trail.
        
        Args:
            name: Input name
            value: Input value
        """
        self._current_inputs[name] = value
    
    def _validate_inputs(self, inputs: dict[str, Any]) -> None:
        """
        Validate inputs and add warnings if needed.
        
        Args:
            inputs: Input values to validate
        """
        _, warnings = validate_calculation_inputs(inputs)
        for warning in warnings:
            self._add_warning(warning)
    
    def _safe_divide(
        self,
        numerator: Decimal,
        denominator: Decimal,
        description: str,
        precision: int = None,
    ) -> Decimal | None:
        """
        Perform safe division with automatic step logging and error handling.
        
        Args:
            numerator: Numerator value
            denominator: Denominator value
            description: Description of the division
            precision: Decimal places for result (uses config if None)
            
        Returns:
            Result or None if division by zero
        """
        if precision is None:
            precision = DECIMAL_PLACES["ratio"]
        
        if is_effectively_zero(denominator):
            self._add_step(f"{description}: Cannot calculate (denominator is zero)")
            self._add_warning(f"Division by zero in {description}")
            return None
        
        result = safe_divide(numerator, denominator, precision)
        self._add_step(f"{description}", result)
        return result
    
    def _handle_calculation_error(
        self,
        error: Exception,
        metric_name: str,
        formula: str,
        inputs: dict[str, Any],
    ) -> CalculationResult:
        """
        Handle calculation errors gracefully and return a proper result.
        
        Args:
            error: Exception that occurred
            metric_name: Name of the metric
            formula: Formula used
            inputs: Input values
            
        Returns:
            CalculationResult with error information
        """
        # Convert exception to validation result
        validation_result = convert_exception_to_validation_result(
            error,
            field=metric_name,
            context=f"{metric_name} calculation"
        )
        
        # Get unit for the metric
        unit = get_metric_unit(metric_name)
        
        # Create result with error information
        return create_calculation_result(
            metric_name=metric_name,
            value=None,
            formula=formula,
            inputs=inputs,
            steps=self._current_steps,
            category=self.category,
            warnings=[issue.message for issue in validation_result.all_issues] + self._current_warnings,
            unit=unit,
        )
    
    def _create_result(
        self,
        metric_name: str,
        value: Decimal | None,
        formula: str,
        unit: MetricUnit | None = None,
    ) -> CalculationResult:
        """
        Create a calculation result with current state.
        
        Args:
            metric_name: Name of the metric
            value: Calculated value
            formula: Formula used
            unit: Override metric unit
            
        Returns:
            Complete CalculationResult
        """
        return create_calculation_result(
            metric_name=metric_name,
            value=value,
            formula=formula,
            inputs=self._current_inputs.copy(),
            steps=self._current_steps.copy(),
            category=self.category,
            warnings=self._current_warnings.copy(),
            unit=unit,
        )
    
    @abstractmethod
    def calculate_all(
        self,
        statement_set: FinancialStatementSet,
        prior_statement_set: FinancialStatementSet | None = None,
    ) -> MetricCollection:
        """
        Calculate all metrics for this category.
        
        Args:
            statement_set: Current period financial statements
            prior_statement_set: Prior period statements (for averages)
            
        Returns:
            MetricCollection with all calculated metrics
        """
        pass
