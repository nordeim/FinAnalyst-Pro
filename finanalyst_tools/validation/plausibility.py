# File: finanalyst_tools/validation/plausibility.py
"""
Plausibility checking for calculated financial metrics.

Verifies that calculated values fall within reasonable ranges
based on typical business metrics and industry norms.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from finanalyst_tools.config import PlausibilityRanges
from finanalyst_tools.models.validation import (
    ValidationSeverity,
    PlausibilityCheck,
    PlausibilityResult,
)
from finanalyst_tools.models.analysis_results import CalculationResult


def check_plausibility(
    metric_name: str,
    value: Decimal | float | None,
    custom_range: tuple[float, float] | None = None,
) -> PlausibilityCheck:
    """
    Check if a metric value is within plausible range.
    
    Args:
        metric_name: Name of the metric
        value: The calculated value
        custom_range: Optional custom range to use instead of default
        
    Returns:
        PlausibilityCheck result
    """
    if value is None:
        return PlausibilityCheck(
            metric_name=metric_name,
            value=Decimal("0"),
            plausible_range=(0, 0),
            is_plausible=True,
            assessment="not_calculated",
            severity=ValidationSeverity.INFO,
            message="Value not calculated",
        )
    
    dec_value = Decimal(str(value))
    float_value = float(dec_value)
    
    # Get range
    if custom_range:
        plausible_range = custom_range
    else:
        plausible_range = PlausibilityRanges.get_range(metric_name)
    
    if plausible_range is None:
        return PlausibilityCheck(
            metric_name=metric_name,
            value=dec_value,
            plausible_range=(float("-inf"), float("inf")),
            is_plausible=True,
            assessment="no_range_defined",
            severity=ValidationSeverity.INFO,
            message=f"No plausibility range defined for {metric_name}",
        )
    
    # Check against range
    min_val, max_val = plausible_range
    is_plausible = min_val <= float_value <= max_val
    
    if float_value < min_val:
        assessment = "below_range"
        message = f"{metric_name} of {float_value:.2f} is below typical range ({min_val:.1f} to {max_val:.1f})"
        severity = ValidationSeverity.WARNING
    elif float_value > max_val:
        assessment = "above_range"
        message = f"{metric_name} of {float_value:.2f} is above typical range ({min_val:.1f} to {max_val:.1f})"
        severity = ValidationSeverity.WARNING
    else:
        assessment = "within_range"
        message = f"{metric_name} of {float_value:.2f} is within typical range"
        severity = ValidationSeverity.INFO
    
    return PlausibilityCheck(
        metric_name=metric_name,
        value=dec_value,
        plausible_range=plausible_range,
        is_plausible=is_plausible,
        assessment=assessment,
        severity=severity,
        message=message,
    )


def check_all_plausibility(
    metrics: list[CalculationResult],
    mutate_metrics: bool = True,
) -> PlausibilityResult:
    """
    Check plausibility for a list of calculation results.
    
    Note: By default, this function MUTATES the input metrics by setting
    `is_plausible=False` and adding warnings for implausible values.
    Set `mutate_metrics=False` to disable this behavior.
    
    Args:
        metrics: List of calculation results to check
        mutate_metrics: If True (default), update metric.is_plausible and add warnings
        
    Returns:
        PlausibilityResult with all check results
    """
    result = PlausibilityResult()
    
    for metric in metrics:
        check = check_plausibility(
            metric_name=metric.metric_name,
            value=metric.value,
            custom_range=metric.plausibility_range,
        )
        result.add_check(check)
        
        # Update the metric's plausibility status (if mutation is enabled)
        if mutate_metrics and not check.is_plausible:
            metric.is_plausible = False
            metric.add_warning(check.message)
    
    return result


class PlausibilityChecker:
    """
    Class-based plausibility checker with customization options.
    """
    
    def __init__(
        self,
        custom_ranges: dict[str, tuple[float, float]] | None = None,
        strict_mode: bool = False,
    ):
        """
        Initialize the plausibility checker.
        
        Args:
            custom_ranges: Dictionary of custom ranges by metric name
            strict_mode: If True, implausible values raise errors instead of warnings
        """
        self.custom_ranges = custom_ranges or {}
        self.strict_mode = strict_mode
    
    def get_range(self, metric_name: str) -> tuple[float, float] | None:
        """Get the range for a metric, checking custom ranges first."""
        if metric_name in self.custom_ranges:
            return self.custom_ranges[metric_name]
        return PlausibilityRanges.get_range(metric_name)
    
    def check(
        self,
        metric_name: str,
        value: Decimal | float | None,
    ) -> PlausibilityCheck:
        """Check a single metric."""
        custom_range = self.custom_ranges.get(metric_name)
        result = check_plausibility(metric_name, value, custom_range)
        
        # Upgrade to error if strict mode
        if self.strict_mode and not result.is_plausible:
            result.severity = ValidationSeverity.ERROR
        
        return result
    
    def check_all(
        self,
        metrics: list[CalculationResult],
        mutate_metrics: bool = True,
    ) -> PlausibilityResult:
        """Check multiple metrics.
        
        Args:
            metrics: List of calculation results to check
            mutate_metrics: If True (default), update metric.is_plausible and add warnings
        """
        result = PlausibilityResult()
        
        for metric in metrics:
            check = self.check(metric.metric_name, metric.value)
            result.add_check(check)
            
            if mutate_metrics and not check.is_plausible:
                metric.is_plausible = False
                metric.add_warning(check.message)
        
        return result
    
    def add_custom_range(
        self,
        metric_name: str,
        min_value: float,
        max_value: float,
    ) -> None:
        """Add or update a custom range."""
        self.custom_ranges[metric_name] = (min_value, max_value)
