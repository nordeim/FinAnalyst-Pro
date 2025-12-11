# finanalyst_tools/calculations/base.py
"""
Base classes and utilities for financial calculations.

Provides:
- BaseCalculator abstract class
- Result creation utilities
- Step formatting helpers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any

from finanalyst_tools.models.analysis_results import (
    CalculationResult,
    CalculationStep,
    MetricUnit,
    MetricCategory,
)
from finanalyst_tools.config import PlausibilityRanges, METRIC_FORMULAS, METRIC_UNITS
from finanalyst_tools.utils.math_ops import to_decimal, round_decimal


def format_step(
    step_number: int,
    description: str,
    formula: str | None = None,
    values: dict[str, Any] | None = None,
    result: Any = None,
) -> CalculationStep:
    """
    Create a formatted calculation step.
    
    Args:
        step_number: Step number in sequence
        description: What this step does
        formula: Formula being applied (optional)
        values: Input values for this step (optional)
        result: Result of this step (optional)
        
    Returns:
        CalculationStep instance
    """
    return CalculationStep(
        step_number=step_number,
        description=description,
        formula=formula,
        values=values or {},
        result=result,
    )


def create_calculation_result(
    metric_name: str,
    value: Decimal | None,
    unit: MetricUnit,
    formula: str,
    inputs: dict[str, Any],
    steps: list[CalculationStep],
    category: MetricCategory | None = None,
    interpretation: str = "",
) -> CalculationResult:
    """
    Create a CalculationResult with plausibility check.
    
    Args:
        metric_name: Name of the metric
        value: Calculated value
        unit: Unit of measurement
        formula: Formula used
        inputs: Input values
        steps: Calculation steps
        category: Metric category
        interpretation: Interpretation text
        
    Returns:
        CalculationResult with plausibility assessment
    """
    result = CalculationResult(
        metric_name=metric_name,
        value=value,
        unit=unit,
        formula=formula,
        inputs={k: str(v) for k, v in inputs.items()},
        calculation_steps=steps,
        category=category,
        interpretation=interpretation,
    )
    
    # Check plausibility
    if value is not None:
        range_tuple = PlausibilityRanges.get_range(metric_name)
        if range_tuple:
            result.plausibility_range = range_tuple
            float_value = float(value)
            result.is_plausible = range_tuple[0] <= float_value <= range_tuple[1]
            
            if not result.is_plausible:
                if float_value < range_tuple[0]:
                    result.add_warning(
                        f"Value {float_value:.2f} is below typical range "
                        f"(min: {range_tuple[0]:.2f})"
                    )
                else:
                    result.add_warning(
                        f"Value {float_value:.2f} is above typical range "
                        f"(max: {range_tuple[1]:.2f})"
                    )
    
    return result


def get_metric_formula(metric_name: str) -> str:
    """Get the formula for a metric."""
    normalized = metric_name.lower().replace(" ", "_")
    return METRIC_FORMULAS.get(normalized, "")


def get_metric_unit(metric_name: str) -> MetricUnit:
    """Get the unit for a metric."""
    normalized = metric_name.lower().replace(" ", "_")
    unit_str = METRIC_UNITS.get(normalized, "ratio")
    
    unit_map = {
        "percentage": MetricUnit.PERCENTAGE,
        "ratio": MetricUnit.RATIO,
        "currency": MetricUnit.CURRENCY,
        "days": MetricUnit.DAYS,
        "count": MetricUnit.COUNT,
    }
    return unit_map.get(unit_str, MetricUnit.RATIO)


class BaseCalculator(ABC):
    """
    Abstract base class for financial calculators.
    
    Provides common functionality for building calculation results
    with audit trails.
    """
    
    def __init__(self):
        self._steps: list[CalculationStep] = []
        self._warnings: list[str] = []
        self._inputs: dict[str, Any] = {}
    
    def _reset(self) -> None:
        """Reset state for new calculation."""
        self._steps = []
        self._warnings = []
        self._inputs = {}
    
    def _add_step(
        self,
        description: str,
        formula: str | None = None,
        values: dict[str, Any] | None = None,
        result: Any = None,
    ) -> None:
        """Add a calculation step."""
        step = format_step(
            step_number=len(self._steps) + 1,
            description=description,
            formula=formula,
            values=values,
            result=result,
        )
        self._steps.append(step)
    
    def _add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self._warnings.append(warning)
    
    def _record_input(self, name: str, value: Any) -> None:
        """Record an input value."""
        self._inputs[name] = value
    
    def _create_result(
        self,
        metric_name: str,
        value: Decimal | None,
        unit: MetricUnit,
        formula: str,
        category: MetricCategory,
        interpretation: str = "",
    ) -> CalculationResult:
        """Create the calculation result."""
        result = create_calculation_result(
            metric_name=metric_name,
            value=value,
            unit=unit,
            formula=formula,
            inputs=self._inputs.copy(),
            steps=self._steps.copy(),
            category=category,
            interpretation=interpretation,
        )
        
        # Add any accumulated warnings
        for warning in self._warnings:
            result.add_warning(warning)
        
        return result
    
    @abstractmethod
    def calculate(self, **kwargs: Any) -> CalculationResult:
        """Perform the calculation. Must be implemented by subclasses."""
        pass
