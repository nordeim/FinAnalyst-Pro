# finanalyst_tools/validation/plausibility.py
"""
Plausibility checks for calculated financial metrics.

Verifies that calculated values fall within reasonable ranges
based on typical business metrics.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from finanalyst_tools.models.validation import (
    PlausibilityCheck,
    PlausibilityResult,
    ValidationSeverity,
)
from finanalyst_tools.models.analysis_results import CalculationResult
from finanalyst_tools.config import PlausibilityRanges
from finanalyst_tools.utils.math_ops import to_decimal


def check_plausibility(
    metric_name: str,
    value: Decimal | float | None,
    custom_range: tuple[float, float] | None = None,
) -> PlausibilityCheck:
    """
    Check if a single metric value is within plausible range.
    
    Args:
        metric_name: Name of the metric
        value: The calculated value
        custom_range: Override the default plausibility range
        
    Returns:
        PlausibilityCheck result
    """
    if value is None:
        return PlausibilityCheck(
            metric_name=metric_name,
            value=Decimal("0"),
            min_plausible=0.0,
            max_plausible=0.0,
            is_plausible=True,  # None is not implausible, just missing
            severity=ValidationSeverity.INFO,
            message=f"{metric_name}: No value to check",
        )
    
    dec_value = to_decimal(value)
    float_value = float(dec_value)
    
    # Get range
    if custom_range:
        min_plausible, max_plausible = custom_range
    else:
        range_tuple = PlausibilityRanges.get_range(metric_name)
        if range_tuple is None:
            # No defined range - assume plausible
            return PlausibilityCheck(
                metric_name=metric_name,
                value=dec_value,
                min_plausible=float("-inf"),
                max_plausible=float("inf"),
                is_plausible=True,
                severity=ValidationSeverity.INFO,
                message=f"{metric_name}: No plausibility range defined",
            )
        min_plausible, max_plausible = range_tuple
    
    # Check if within range
    is_plausible = min_plausible <= float_value <= max_plausible
    
    # Determine severity
    if is_plausible:
        severity = ValidationSeverity.INFO
    else:
        # How far out of range?
        if float_value < min_plausible:
            deviation = min_plausible - float_value
        else:
            deviation = float_value - max_plausible
        
        range_size = max_plausible - min_plausible
        if range_size > 0:
            relative_deviation = deviation / range_size
        else:
            relative_deviation = deviation
        
        # Major deviation = error, minor = warning
        if relative_deviation > 0.5:
            severity = ValidationSeverity.ERROR
        else:
            severity = ValidationSeverity.WARNING
    
    return PlausibilityCheck(
        metric_name=metric_name,
        value=dec_value,
        min_plausible=min_plausible,
        max_plausible=max_plausible,
        is_plausible=is_plausible,
        severity=severity,
    )


def check_all_plausibility(
    metrics: dict[str, Decimal | float | None],
) -> PlausibilityResult:
    """
    Check plausibility for multiple metrics.
    
    Args:
        metrics: Dictionary of metric_name -> value
        
    Returns:
        PlausibilityResult with all checks
    """
    result = PlausibilityResult()
    
    for metric_name, value in metrics.items():
        check = check_plausibility(metric_name, value)
        result.add_check(check)
    
    return result


class PlausibilityChecker:
    """
    Configurable plausibility checker.
    
    Allows custom ranges and severity thresholds.
    """
    
    def __init__(
        self,
        custom_ranges: dict[str, tuple[float, float]] | None = None,
        strict_mode: bool = False,
    ):
        """
        Initialize the checker.
        
        Args:
            custom_ranges: Override default ranges for specific metrics
            strict_mode: If True, treat warnings as errors
        """
        self.custom_ranges = custom_ranges or {}
        self.strict_mode = strict_mode
    
    def check(
        self,
        metric_name: str,
        value: Decimal | float | None,
    ) -> PlausibilityCheck:
        """Check a single metric."""
        custom_range = self.custom_ranges.get(metric_name)
        check = check_plausibility(metric_name, value, custom_range)
        
        if self.strict_mode and check.severity == ValidationSeverity.WARNING:
            check.severity = ValidationSeverity.ERROR
        
        return check
    
    def check_calculation_result(
        self,
        result: CalculationResult,
    ) -> PlausibilityCheck:
        """
        Check plausibility of a CalculationResult.
        
        Updates the result's plausibility fields.
        """
        check = self.check(result.metric_name, result.value)
        
        # Update the calculation result
        result.is_plausible = check.is_plausible
        result.plausibility_range = (check.min_plausible, check.max_plausible)
        
        if not check.is_plausible:
            result.add_warning(check.message)
        
        return check
    
    def check_all(
        self,
        metrics: dict[str, Decimal | float | None],
    ) -> PlausibilityResult:
        """Check multiple metrics."""
        result = PlausibilityResult()
        
        for metric_name, value in metrics.items():
            check = self.check(metric_name, value)
            result.add_check(check)
        
        return result
    
    def check_calculation_results(
        self,
        results: list[CalculationResult],
    ) -> PlausibilityResult:
        """Check multiple CalculationResults."""
        plausibility_result = PlausibilityResult()
        
        for result in results:
            check = self.check_calculation_result(result)
            plausibility_result.add_check(check)
        
        return plausibility_result
