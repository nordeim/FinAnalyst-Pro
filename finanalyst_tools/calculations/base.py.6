# finanalyst_tools/calculations/base.py
"""
Base calculation infrastructure for FinAnalyst-Pro Agent Tools.

Provides:
- BaseCalculator class with common functionality
- Factory functions for creating calculation results
- Utility functions for extracting and validating inputs
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any

from finanalyst_tools.config import (
    PlausibilityRanges,
    METRIC_FORMULAS,
    METRIC_UNITS,
    DECIMAL_PLACES,
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
    FinancialPeriod,
)
from finanalyst_tools.utils.math_ops import (
    to_decimal,
    safe_divide,
    round_decimal,
    calculate_percentage,
    calculate_average,
    is_effectively_zero,
)


def extract_decimal_value(
    value: Any,
    field_name: str = "value",
    default: Decimal | None = None,
) -> Decimal:
    """
    Extract a Decimal value from various input types.
    
    Args:
        value: Input value (int, float, Decimal, str, or None)
        field_name: Name of the field (for error messages)
        default: Default value if conversion fails
        
    Returns:
        Decimal value
    """
    if value is None:
        if default is not None:
            return default
        return Decimal("0")
    
    return to_decimal(value, default=default or Decimal("0"))


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
    Get the formula string for a metric.
    
    Args:
        metric_name: Name of the metric
        
    Returns:
        Formula string
    """
    return METRIC_FORMULAS.get(metric_name.lower(), "Custom calculation")


def create_calculation_result(
    metric_name: str,
    value: Decimal | None,
    inputs: dict[str, Any],
    calculation_steps: list[str],
    category: MetricCategory | None = None,
    warnings: list[str] | None = None,
    unit: MetricUnit | None = None,
    formula: str | None = None,
) -> CalculationResult:
    """
    Factory function to create a CalculationResult with plausibility check.
    
    Args:
        metric_name: Name of the metric
        value: Calculated value
        inputs: Dictionary of input values used
        calculation_steps: List of calculation step descriptions
        category: Metric category
        warnings: List of warning messages
        unit: Override unit type
        formula: Override formula string
        
    Returns:
        Complete CalculationResult with plausibility assessment
    """
    # Get unit and formula if not provided
    if unit is None:
        unit = get_metric_unit(metric_name)
    if formula is None:
        formula = get_metric_formula(metric_name)
    
    # Get plausibility range
    plausibility_range = PlausibilityRanges.get_range(metric_name)
    
    # Check plausibility
    is_plausible = True
    result_warnings = list(warnings) if warnings else []
    
    if value is not None and plausibility_range is not None:
        float_value = float(value)
        min_val, max_val = plausibility_range
        
        if float_value < min_val:
            is_plausible = False
            result_warnings.append(
                f"Value {float_value:.2f} is below typical range ({min_val:.1f} to {max_val:.1f})"
            )
        elif float_value > max_val:
            is_plausible = False
            result_warnings.append(
                f"Value {float_value:.2f} is above typical range ({min_val:.1f} to {max_val:.1f})"
            )
    
    # Convert inputs to serializable format
    serializable_inputs = {}
    for key, val in inputs.items():
        if isinstance(val, Decimal):
            serializable_inputs[key] = float(val)
        else:
            serializable_inputs[key] = val
    
    return CalculationResult(
        metric_name=metric_name,
        value=value,
        unit=unit,
        formula=formula,
        inputs=serializable_inputs,
        calculation_steps=calculation_steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=result_warnings,
        category=category,
    )


class BaseCalculator(ABC):
    """
    Abstract base class for financial calculators.
    
    Provides common functionality for:
    - Input extraction and validation
    - Result creation with audit trail
    - Plausibility checking
    - Warning accumulation
    """
    
    def __init__(self):
        """Initialize the calculator."""
        self._warnings: list[str] = []
        self._steps: list[str] = []
    
    def reset(self) -> None:
        """Reset the calculator state for a new calculation."""
        self._warnings = []
        self._steps = []
    
    def add_step(self, description: str, value: Any = None) -> None:
        """
        Add a calculation step to the audit trail.
        
        Args:
            description: Description of the step
            value: Optional value to include
        """
        step_num = len(self._steps) + 1
        if value is not None:
            if isinstance(value, Decimal):
                self._steps.append(f"Step {step_num}: {description} = {float(value):,.4f}")
            else:
                self._steps.append(f"Step {step_num}: {description} = {value}")
        else:
            self._steps.append(f"Step {step_num}: {description}")
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self._warnings.append(warning)
    
    def get_steps(self) -> list[str]:
        """Get the calculation steps."""
        return self._steps.copy()
    
    def get_warnings(self) -> list[str]:
        """Get the warning messages."""
        return self._warnings.copy()
    
    def create_result(
        self,
        metric_name: str,
        value: Decimal | None,
        inputs: dict[str, Any],
        category: MetricCategory | None = None,
        unit: MetricUnit | None = None,
        formula: str | None = None,
    ) -> CalculationResult:
        """
        Create a calculation result with current steps and warnings.
        
        Args:
            metric_name: Name of the metric
            value: Calculated value
            inputs: Dictionary of input values
            category: Metric category
            unit: Override unit type
            formula: Override formula string
            
        Returns:
            Complete CalculationResult
        """
        return create_calculation_result(
            metric_name=metric_name,
            value=value,
            inputs=inputs,
            calculation_steps=self.get_steps(),
            category=category,
            warnings=self.get_warnings(),
            unit=unit,
            formula=formula,
        )
    
    @abstractmethod
    def calculate(self, *args, **kwargs) -> CalculationResult:
        """
        Perform the calculation.
        
        Must be implemented by subclasses.
        """
        pass
    
    def safe_divide_with_step(
        self,
        numerator: Decimal,
        denominator: Decimal,
        description: str,
        precision: int = DECIMAL_PLACES["ratio"],
    ) -> Decimal | None:
        """
        Perform safe division and add calculation step.
        
        Args:
            numerator: The dividend
            denominator: The divisor
            description: Description of what's being calculated
            precision: Decimal places for result
            
        Returns:
            Result or None if division by zero
        """
        if is_effectively_zero(denominator):
            self.add_step(f"{description}: Cannot calculate (denominator is zero)")
            self.add_warning(f"Division by zero in {description}")
            return None
        
        result = safe_divide(numerator, denominator, precision=precision)
        self.add_step(f"{description}", result)
        return result
    
    def calculate_average_with_step(
        self,
        value1: Decimal,
        value2: Decimal,
        description: str,
    ) -> Decimal:
        """
        Calculate average and add calculation step.
        
        Args:
            value1: First value
            value2: Second value
            description: Description of what's being averaged
            
        Returns:
            Average value
        """
        result = calculate_average(value1, value2)
        self.add_step(f"Calculate average {description}", result)
        return result
