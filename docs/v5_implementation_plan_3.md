# finanalyst_tools/models/analysis_results.py
```py
# File: finanalyst_tools/models/analysis_results.py
"""
Structured result models for calculations and analysis.

Provides:
- CalculationResult: Complete audit trail for single metric
- MetricCollection: Group of related metrics
- TrendAnalysisResult: Multi-period trend analysis output
- ConfidenceAssessment: Confidence level with justification
- ComprehensiveAnalysisResult: Complete analysis output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any
import json

from finanalyst_tools.models.financial_statements import FinancialPeriod


class MetricUnit(str, Enum):
    """Units for financial metrics."""
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    CURRENCY = "currency"
    DAYS = "days"
    COUNT = "count"
    TIMES = "times"


class MetricCategory(str, Enum):
    """Categories for financial metrics."""
    PROFITABILITY = "profitability"
    LIQUIDITY = "liquidity"
    SOLVENCY = "solvency"
    EFFICIENCY = "efficiency"
    GROWTH = "growth"
    VALUATION = "valuation"


class TrendDirection(str, Enum):
    """Direction of a trend."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class ConfidenceLevel(str, Enum):
    """Confidence levels for analysis results."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class CalculationResult:
    """
    Complete result of a single metric calculation with audit trail.
    
    Provides:
    - The calculated value with unit
    - Formula used
    - All input values
    - Step-by-step calculation log
    - Plausibility assessment
    - Warnings for unusual values
    """
    metric_name: str
    value: Decimal | None
    unit: MetricUnit
    formula: str
    inputs: dict[str, Any] = field(default_factory=dict)
    calculation_steps: list[str] = field(default_factory=list)
    is_plausible: bool = True
    plausibility_range: tuple[float, float] | None = None
    warnings: list[str] = field(default_factory=list)
    category: MetricCategory | None = None
    
    def add_step(self, step: str) -> None:
        """Add a calculation step to the audit trail."""
        step_num = len(self.calculation_steps) + 1
        self.calculation_steps.append(f"Step {step_num}: {step}")
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    @property
    def is_calculable(self) -> bool:
        """Whether the calculation was successful."""
        return self.value is not None
    
    @property
    def formatted_value(self) -> str:
        """Get formatted value based on unit."""
        if self.value is None:
            return "N/A"
        
        if self.unit == MetricUnit.PERCENTAGE:
            return f"{float(self.value):.2f}%"
        elif self.unit == MetricUnit.RATIO:
            return f"{float(self.value):.4f}"
        elif self.unit == MetricUnit.CURRENCY:
            return f"${float(self.value):,.2f}"
        elif self.unit == MetricUnit.DAYS:
            return f"{int(self.value)} days"
        elif self.unit == MetricUnit.TIMES:
            return f"{float(self.value):.2f}x"
        return f"{float(self.value):.2f}"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metric_name": self.metric_name,
            "value": float(self.value) if self.value is not None else None,
            "formatted_value": self.formatted_value,
            "unit": self.unit.value,
            "formula": self.formula,
            "inputs": {k: float(v) if isinstance(v, Decimal) else v for k, v in self.inputs.items()},
            "calculation_steps": self.calculation_steps,
            "is_plausible": self.is_plausible,
            "plausibility_range": self.plausibility_range,
            "warnings": self.warnings,
            "category": self.category.value if self.category else None,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def to_reasoning_block(self) -> str:
        """
        Format as reasoning block for LLM output.
        
        Returns formatted block matching system prompt requirements.
        """
        lines = [
            f"### {self.metric_name}",
            f"**Value**: {self.formatted_value}",
            f"**Formula**: {self.formula}",
            "",
            "**Calculation Steps**:",
        ]
        
        for step in self.calculation_steps:
            lines.append(f"  - {step}")
        
        lines.append("")
        lines.append("**Inputs Used**:")
        for key, val in self.inputs.items():
            if isinstance(val, Decimal):
                lines.append(f"  - {key}: {float(val):,.2f}")
            else:
                lines.append(f"  - {key}: {val}")
        
        if self.warnings:
            lines.append("")
            lines.append("**Warnings**:")
            for warning in self.warnings:
                lines.append(f"  - ⚠️ {warning}")
        
        plausibility_status = "✅ Within range" if self.is_plausible else "⚠️ Outside expected range"
        if self.plausibility_range:
            lines.append(f"\n**Plausibility**: {plausibility_status} ({self.plausibility_range[0]} to {self.plausibility_range[1]})")
        
        return "\n".join(lines)


@dataclass
class MetricResult(CalculationResult):
    """Extended CalculationResult with period information."""
    period: FinancialPeriod | None = None


@dataclass
class MetricCollection:
    """
    Collection of related metrics.
    
    Groups metrics by category with summary statistics.
    """
    category: MetricCategory
    period: FinancialPeriod | str
    metrics: list[CalculationResult] = field(default_factory=list)
    
    def add_metric(self, metric: CalculationResult) -> None:
        """Add a metric to the collection."""
        self.metrics.append(metric)
    
    @property
    def metric_count(self) -> int:
        """Number of metrics in collection."""
        return len(self.metrics)
    
    @property
    def calculable_count(self) -> int:
        """Number of successfully calculated metrics."""
        return sum(1 for m in self.metrics if m.is_calculable)
    
    @property
    def plausible_count(self) -> int:
        """Number of plausible metrics."""
        return sum(1 for m in self.metrics if m.is_plausible)
    
    @property
    def summary(self) -> dict[str, Decimal | None]:
        """Quick access to metric values by name."""
        return {m.metric_name: m.value for m in self.metrics}
    
    @property
    def all_warnings(self) -> list[str]:
        """Aggregate all warnings from metrics."""
        warnings = []
        for metric in self.metrics:
            for warning in metric.warnings:
                warnings.append(f"{metric.metric_name}: {warning}")
        return warnings
    
    def get_metric(self, name: str) -> CalculationResult | None:
        """Get a specific metric by name."""
        for metric in self.metrics:
            if metric.metric_name.lower() == name.lower():
                return metric
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "category": self.category.value,
            "period": str(self.period),
            "metric_count": self.metric_count,
            "calculable_count": self.calculable_count,
            "plausible_count": self.plausible_count,
            "metrics": [m.to_dict() for m in self.metrics],
            "warnings": self.all_warnings,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def to_table(self) -> str:
        """Format as Markdown table."""
        lines = [
            f"## {self.category.value.title()} Metrics ({self.period})",
            "",
            "| Metric | Value | Plausible | Warnings |",
            "|--------|-------|-----------|----------|",
        ]
        
        for metric in self.metrics:
            plausible = "✅" if metric.is_plausible else "⚠️"
            warning_count = len(metric.warnings)
            warning_text = f"{warning_count} warning(s)" if warning_count > 0 else "None"
            lines.append(f"| {metric.metric_name} | {metric.formatted_value} | {plausible} | {warning_text} |")
        
        return "\n".join(lines)


@dataclass
class TrendAnalysisResult:
    """
    Result of multi-period trend analysis for a metric.
    """
    metric_name: str
    periods: list[str]
    values: list[Decimal | None]
    direction: TrendDirection
    growth_rate: Decimal | None = None  # CAGR or average growth
    volatility: Decimal | None = None   # Standard deviation
    interpretation: str = ""
    
    @property
    def period_count(self) -> int:
        """Number of periods analyzed."""
        return len(self.periods)
    
    @property
    def valid_value_count(self) -> int:
        """Number of periods with valid values."""
        return sum(1 for v in self.values if v is not None)
    
    @property
    def first_value(self) -> Decimal | None:
        """First valid value in series."""
        for v in self.values:
            if v is not None:
                return v
        return None
    
    @property
    def last_value(self) -> Decimal | None:
        """Last valid value in series."""
        for v in reversed(self.values):
            if v is not None:
                return v
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metric_name": self.metric_name,
            "periods": self.periods,
            "values": [float(v) if v is not None else None for v in self.values],
            "direction": self.direction.value,
            "growth_rate": float(self.growth_rate) if self.growth_rate else None,
            "volatility": float(self.volatility) if self.volatility else None,
            "interpretation": self.interpretation,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class ConfidenceAssessment:
    """
    Confidence level assessment for analysis results.
    
    Implements the mandatory confidence scoring from system prompt.
    """
    level: ConfidenceLevel
    justification: str
    factors: dict[str, str] = field(default_factory=dict)
    score: float = 100.0  # Internal score (0-100)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "level": self.level.value,
            "justification": self.justification,
            "factors": self.factors,
            "score": self.score,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_display(self) -> str:
        """Format for display in report."""
        return f"**{self.level.value}** — {self.justification}"


@dataclass
class ComprehensiveAnalysisResult:
    """
    Complete analysis result combining all components.
    
    This is the top-level result returned by the analysis pipeline.
    """
    analysis_type: str
    period: str
    currency: str
    metric_collections: list[MetricCollection] = field(default_factory=list)
    trend_analyses: list[TrendAnalysisResult] = field(default_factory=list)
    confidence: ConfidenceAssessment | None = None
    validation_summary: dict[str, Any] = field(default_factory=dict)
    reconciliation_summary: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    
    @property
    def total_metrics(self) -> int:
        """Total number of metrics calculated."""
        return sum(mc.metric_count for mc in self.metric_collections)
    
    @property
    def all_metrics(self) -> list[CalculationResult]:
        """Flatten all metrics from all collections."""
        metrics = []
        for collection in self.metric_collections:
            metrics.extend(collection.metrics)
        return metrics
    
    @property
    def all_warnings(self) -> list[str]:
        """Aggregate all warnings from all collections."""
        warnings = []
        for collection in self.metric_collections:
            warnings.extend(collection.all_warnings)
        return warnings
    
    def get_metric(self, name: str) -> CalculationResult | None:
        """Get a specific metric by name from any collection."""
        for collection in self.metric_collections:
            metric = collection.get_metric(name)
            if metric:
                return metric
        return None
    
    def add_collection(self, collection: MetricCollection) -> None:
        """Add a metric collection."""
        self.metric_collections.append(collection)
    
    def add_trend(self, trend: TrendAnalysisResult) -> None:
        """Add a trend analysis result."""
        self.trend_analyses.append(trend)
    
    def add_recommendation(self, recommendation: str) -> None:
        """Add a recommendation."""
        self.recommendations.append(recommendation)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "analysis_type": self.analysis_type,
            "period": self.period,
            "currency": self.currency,
            "total_metrics": self.total_metrics,
            "metric_collections": [mc.to_dict() for mc in self.metric_collections],
            "trend_analyses": [ta.to_dict() for ta in self.trend_analyses],
            "confidence": self.confidence.to_dict() if self.confidence else None,
            "validation_summary": self.validation_summary,
            "reconciliation_summary": self.reconciliation_summary,
            "recommendations": self.recommendations,
            "warnings": self.all_warnings,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

```

# finanalyst_tools/models/validation.py
```py
# File: finanalyst_tools/models/validation.py
"""
Models for validation results and issues.

Provides structured representations for:
- Validation issues with severity levels
- Validation results with aggregated issues
- Reconciliation checks and results
- Plausibility checks and results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any
import json


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    ERROR = "error"       # Blocking - cannot proceed
    WARNING = "warning"   # Non-blocking - proceed with caution
    INFO = "info"         # Informational only


@dataclass
class ValidationIssue:
    """
    Represents a single validation issue.
    
    Attributes:
        field: Name of the field with the issue
        message: Human-readable description of the issue
        severity: Issue severity level
        actual_value: The value that was found (optional)
        expected: Description of what was expected (optional)
        suggestion: Actionable suggestion for resolution (optional)
    """
    field: str
    message: str
    severity: ValidationSeverity
    actual_value: Any = None
    expected: str | None = None
    suggestion: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "field": self.field,
            "message": self.message,
            "severity": self.severity.value,
        }
        if self.actual_value is not None:
            result["actual_value"] = self.actual_value
        if self.expected:
            result["expected"] = self.expected
        if self.suggestion:
            result["suggestion"] = self.suggestion
        return result
    
    def __str__(self) -> str:
        prefix = {
            ValidationSeverity.ERROR: "❌",
            ValidationSeverity.WARNING: "⚠️",
            ValidationSeverity.INFO: "ℹ️",
        }.get(self.severity, "")
        return f"{prefix} [{self.field}] {self.message}"


@dataclass
class ValidationResult:
    """
    Aggregated result of validation checks.
    
    Provides methods to:
    - Add issues
    - Merge with other results
    - Check if processing can proceed
    - Format as table or dict
    """
    is_valid: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    info: list[ValidationIssue] = field(default_factory=list)
    
    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return len(self.issues)
    
    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return len(self.warnings)
    
    @property
    def info_count(self) -> int:
        """Count of info-level issues."""
        return len(self.info)
    
    @property
    def total_issue_count(self) -> int:
        """Total count of all issues."""
        return self.error_count + self.warning_count + self.info_count
    
    @property
    def can_proceed(self) -> bool:
        """Whether processing can proceed (no errors)."""
        return self.error_count == 0
    
    def add_issue(self, issue: ValidationIssue) -> None:
        """
        Add an issue to the appropriate list based on severity.
        
        Args:
            issue: The validation issue to add
        """
        if issue.severity == ValidationSeverity.ERROR:
            self.issues.append(issue)
            self.is_valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warnings.append(issue)
        else:
            self.info.append(issue)
    
    def add_error(
        self,
        field: str,
        message: str,
        actual_value: Any = None,
        expected: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Convenience method to add an error."""
        self.add_issue(ValidationIssue(
            field=field,
            message=message,
            severity=ValidationSeverity.ERROR,
            actual_value=actual_value,
            expected=expected,
            suggestion=suggestion,
        ))
    
    def add_warning(
        self,
        field: str,
        message: str,
        actual_value: Any = None,
        expected: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Convenience method to add a warning."""
        self.add_issue(ValidationIssue(
            field=field,
            message=message,
            severity=ValidationSeverity.WARNING,
            actual_value=actual_value,
            expected=expected,
            suggestion=suggestion,
        ))
    
    def add_info(
        self,
        field: str,
        message: str,
        actual_value: Any = None,
    ) -> None:
        """Convenience method to add an info message."""
        self.add_issue(ValidationIssue(
            field=field,
            message=message,
            severity=ValidationSeverity.INFO,
            actual_value=actual_value,
        ))
    
    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """
        Merge another ValidationResult into this one.
        
        Args:
            other: Another ValidationResult to merge
            
        Returns:
            Self for chaining
        """
        self.issues.extend(other.issues)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        if not other.is_valid:
            self.is_valid = False
        return self
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "can_proceed": self.can_proceed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "errors": [issue.to_dict() for issue in self.issues],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "info": [issue.to_dict() for issue in self.info],
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def to_table(self) -> str:
        """
        Format as Markdown table.
        
        Returns:
            Markdown table string
        """
        if self.total_issue_count == 0:
            return "✅ No validation issues found."
        
        lines = ["| Severity | Field | Message |", "|----------|-------|---------|"]
        
        all_issues = (
            [(i, "🔴 Error") for i in self.issues] +
            [(i, "🟡 Warning") for i in self.warnings] +
            [(i, "🔵 Info") for i in self.info]
        )
        
        for issue, severity_label in all_issues:
            lines.append(f"| {severity_label} | {issue.field} | {issue.message} |")
        
        return "\n".join(lines)
    
    def to_summary(self) -> str:
        """Generate a brief summary string."""
        if self.is_valid and self.warning_count == 0:
            return "✅ Validation passed with no issues"
        
        parts = []
        if self.error_count > 0:
            parts.append(f"{self.error_count} error(s)")
        if self.warning_count > 0:
            parts.append(f"{self.warning_count} warning(s)")
        if self.info_count > 0:
            parts.append(f"{self.info_count} info")
        
        status = "❌ Validation failed" if not self.is_valid else "⚠️ Validation passed with warnings"
        return f"{status}: {', '.join(parts)}"


@dataclass
class ReconciliationCheck:
    """
    Result of a single cross-statement reconciliation check.
    
    Attributes:
        check_name: Name of the reconciliation check
        statement_a: Source of first value
        value_a: First value
        statement_b: Source of second value
        value_b: Second value
        difference: Absolute difference between values
        tolerance: Tolerance used for comparison
        passed: Whether the check passed
        message: Human-readable result message
    """
    check_name: str
    statement_a: str
    value_a: Decimal
    statement_b: str
    value_b: Decimal
    difference: Decimal
    tolerance: float
    passed: bool
    message: str
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "check_name": self.check_name,
            "statement_a": self.statement_a,
            "value_a": float(self.value_a),
            "statement_b": self.statement_b,
            "value_b": float(self.value_b),
            "difference": float(self.difference),
            "tolerance": self.tolerance,
            "passed": self.passed,
            "message": self.message,
        }
    
    def __str__(self) -> str:
        status = "✅" if self.passed else "❌"
        return f"{status} {self.check_name}: {self.message}"


@dataclass
class ReconciliationResult:
    """
    Aggregated result of all reconciliation checks.
    """
    checks: list[ReconciliationCheck] = field(default_factory=list)
    
    @property
    def all_passed(self) -> bool:
        """Whether all checks passed."""
        return all(check.passed for check in self.checks)
    
    @property
    def passed_count(self) -> int:
        """Count of passed checks."""
        return sum(1 for check in self.checks if check.passed)
    
    @property
    def failed_count(self) -> int:
        """Count of failed checks."""
        return sum(1 for check in self.checks if not check.passed)
    
    @property
    def failed_checks(self) -> list[ReconciliationCheck]:
        """List of failed checks."""
        return [check for check in self.checks if not check.passed]
    
    def add_check(self, check: ReconciliationCheck) -> None:
        """Add a reconciliation check result."""
        self.checks.append(check)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "all_passed": self.all_passed,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "checks": [check.to_dict() for check in self.checks],
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def to_table(self) -> str:
        """Format as Markdown table."""
        if not self.checks:
            return "No reconciliation checks performed."
        
        lines = [
            "| Check | Status | Statement A | Value A | Statement B | Value B | Difference |",
            "|-------|--------|-------------|---------|-------------|---------|------------|"
        ]
        
        for check in self.checks:
            status = "✅" if check.passed else "❌"
            lines.append(
                f"| {check.check_name} | {status} | {check.statement_a} | "
                f"{check.value_a:,.2f} | {check.statement_b} | "
                f"{check.value_b:,.2f} | {check.difference:,.2f} |"
            )
        
        return "\n".join(lines)
    
    def to_validation_result(self) -> ValidationResult:
        """Convert to ValidationResult for unified handling."""
        result = ValidationResult()
        
        for check in self.checks:
            if not check.passed:
                result.add_error(
                    field=check.check_name,
                    message=check.message,
                    actual_value=f"{check.statement_a}={check.value_a}, {check.statement_b}={check.value_b}",
                    expected=f"Difference within {check.tolerance:.1%}",
                    suggestion="Verify data accuracy across statements",
                )
        
        return result


@dataclass
class PlausibilityCheck:
    """
    Result of a single plausibility check.
    
    Attributes:
        metric_name: Name of the metric checked
        value: The calculated value
        plausible_range: Expected (min, max) range
        is_plausible: Whether value is within range
        assessment: "within_range", "below_range", or "above_range"
        severity: Severity if implausible (usually WARNING)
        message: Human-readable result message
    """
    metric_name: str
    value: Decimal
    plausible_range: tuple[float, float]
    is_plausible: bool
    assessment: str
    severity: ValidationSeverity
    message: str
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metric_name": self.metric_name,
            "value": float(self.value),
            "plausible_range": self.plausible_range,
            "is_plausible": self.is_plausible,
            "assessment": self.assessment,
            "severity": self.severity.value,
            "message": self.message,
        }
    
    def __str__(self) -> str:
        status = "✅" if self.is_plausible else "⚠️"
        return f"{status} {self.metric_name}: {self.message}"


@dataclass
class PlausibilityResult:
    """
    Aggregated result of all plausibility checks.
    """
    checks: list[PlausibilityCheck] = field(default_factory=list)
    
    @property
    def all_plausible(self) -> bool:
        """Whether all checks passed."""
        return all(check.is_plausible for check in self.checks)
    
    @property
    def plausible_count(self) -> int:
        """Count of plausible values."""
        return sum(1 for check in self.checks if check.is_plausible)
    
    @property
    def implausible_count(self) -> int:
        """Count of implausible values."""
        return sum(1 for check in self.checks if not check.is_plausible)
    
    @property
    def implausible_checks(self) -> list[PlausibilityCheck]:
        """List of implausible checks."""
        return [check for check in self.checks if not check.is_plausible]
    
    def add_check(self, check: PlausibilityCheck) -> None:
        """Add a plausibility check result."""
        self.checks.append(check)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "all_plausible": self.all_plausible,
            "plausible_count": self.plausible_count,
            "implausible_count": self.implausible_count,
            "checks": [check.to_dict() for check in self.checks],
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def to_validation_result(self) -> ValidationResult:
        """Convert to ValidationResult for unified handling."""
        result = ValidationResult()
        
        for check in self.checks:
            if not check.is_plausible:
                result.add_warning(
                    field=check.metric_name,
                    message=check.message,
                    actual_value=float(check.value),
                    expected=f"Between {check.plausible_range[0]} and {check.plausible_range[1]}",
                    suggestion="Verify input data accuracy",
                )
        
        return result

```

# finanalyst_tools/models/financial_statements.py
```py
# File: finanalyst_tools/models/financial_statements.py
"""
Pydantic models for financial statement data structures.

These models provide:
- Strong type validation for financial data
- Field aliases for common naming variations
- Calculated properties for derived values
- JSON serialization for LLM communication
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator, model_validator


class StatementType(str, Enum):
    """Types of financial statements."""
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"


class PeriodType(str, Enum):
    """Financial reporting period types."""
    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    TTM = "trailing_twelve_months"


# Type aliases for documentation
MonetaryValue = Annotated[Decimal, Field(description="Monetary value in base currency")]
Percentage = Annotated[Decimal, Field(description="Percentage value")]
Ratio = Annotated[Decimal, Field(description="Ratio value")]


class FinancialPeriod(BaseModel):
    """
    Represents a financial reporting period.
    
    Examples:
        - Annual: FinancialPeriod(year=2023, period_type=PeriodType.ANNUAL)
        - Quarterly: FinancialPeriod(year=2023, period_type=PeriodType.QUARTERLY, quarter=2)
    """
    
    year: int = Field(..., ge=1900, le=2100, description="Fiscal year")
    period_type: PeriodType = Field(default=PeriodType.ANNUAL)
    quarter: int | None = Field(default=None, ge=1, le=4, description="Quarter number (1-4)")
    month: int | None = Field(default=None, ge=1, le=12, description="Month number (1-12)")
    start_date: date | None = Field(default=None, description="Period start date")
    end_date: date | None = Field(default=None, description="Period end date")
    
    model_config = {"extra": "forbid"}
    
    @model_validator(mode="after")
    def validate_period_details(self) -> "FinancialPeriod":
        """Validate period-specific fields."""
        if self.period_type == PeriodType.QUARTERLY and self.quarter is None:
            raise ValueError("Quarter must be specified for quarterly periods")
        if self.period_type == PeriodType.MONTHLY and self.month is None:
            raise ValueError("Month must be specified for monthly periods")
        return self
    
    def __str__(self) -> str:
        """Human-readable period representation."""
        if self.period_type == PeriodType.QUARTERLY:
            return f"Q{self.quarter} {self.year}"
        if self.period_type == PeriodType.MONTHLY:
            return f"{self.year}-{self.month:02d}"
        if self.period_type == PeriodType.TTM:
            return f"TTM {self.year}"
        return str(self.year)
    
    def __lt__(self, other: "FinancialPeriod") -> bool:
        """Enable sorting by period."""
        if self.year != other.year:
            return self.year < other.year
        self_sub = self.quarter or self.month or 0
        other_sub = other.quarter or other.month or 0
        return self_sub < other_sub
    
    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, FinancialPeriod):
            return False
        return (
            self.year == other.year and
            self.period_type == other.period_type and
            self.quarter == other.quarter and
            self.month == other.month
        )
    
    def __hash__(self) -> int:
        """Make hashable for use in sets/dicts."""
        return hash((self.year, self.period_type, self.quarter, self.month))


class IncomeStatementData(BaseModel):
    """
    Income Statement / Profit & Loss data model.
    
    Supports multiple naming conventions through field aliases.
    Provides calculated properties for derived values.
    """
    
    period: FinancialPeriod
    currency: str = Field(default="SGD", min_length=3, max_length=3)
    
    # ─────────────────────────────────────────────────────────────────────
    # Revenue
    # ─────────────────────────────────────────────────────────────────────
    total_revenue: MonetaryValue = Field(
        ...,
        alias="revenue",
        description="Total revenue / net sales"
    )
    cost_of_goods_sold: MonetaryValue = Field(
        ...,
        alias="cogs",
        description="Cost of goods sold / cost of sales"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Operating Expenses
    # ─────────────────────────────────────────────────────────────────────
    operating_expenses: MonetaryValue | None = Field(
        default=None,
        alias="opex",
        description="Total operating expenses"
    )
    selling_general_admin: MonetaryValue | None = Field(
        default=None,
        alias="sga",
        description="Selling, General & Administrative expenses"
    )
    marketing_expenses: MonetaryValue | None = Field(
        default=None,
        description="Marketing and advertising expenses"
    )
    research_development: MonetaryValue | None = Field(
        default=None,
        alias="r_and_d",
        description="Research & Development expenses"
    )
    depreciation_amortization: MonetaryValue | None = Field(
        default=None,
        alias="d_and_a",
        description="Depreciation and amortization"
    )
    other_operating_expenses: MonetaryValue | None = Field(
        default=None,
        description="Other operating expenses"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Non-Operating Items
    # ─────────────────────────────────────────────────────────────────────
    interest_income: MonetaryValue = Field(default=Decimal("0"))
    interest_expense: MonetaryValue = Field(default=Decimal("0"))
    other_income: MonetaryValue = Field(default=Decimal("0"))
    other_expenses: MonetaryValue = Field(default=Decimal("0"))
    
    # ─────────────────────────────────────────────────────────────────────
    # Taxes and Bottom Line
    # ─────────────────────────────────────────────────────────────────────
    income_tax_expense: MonetaryValue = Field(default=Decimal("0"))
    net_income: MonetaryValue | None = Field(default=None)
    
    # ─────────────────────────────────────────────────────────────────────
    # Per Share Data
    # ─────────────────────────────────────────────────────────────────────
    earnings_per_share: Decimal | None = Field(default=None, alias="eps")
    diluted_eps: Decimal | None = Field(default=None)
    shares_outstanding: int | None = Field(default=None)
    
    model_config = {
        "populate_by_name": True,
        "extra": "allow",
    }
    
    @field_validator("total_revenue", "cost_of_goods_sold", mode="before")
    @classmethod
    def convert_to_decimal(cls, v: Any) -> Decimal:
        """Convert numeric inputs to Decimal."""
        if v is None:
            raise ValueError("This field is required")
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))
    
    @property
    def gross_profit(self) -> Decimal:
        """Calculate gross profit."""
        return self.total_revenue - self.cost_of_goods_sold
    
    @property
    def total_operating_expenses(self) -> Decimal:
        """Calculate total operating expenses from components or aggregate."""
        if self.operating_expenses is not None:
            return self.operating_expenses
        
        total = Decimal("0")
        expense_fields = [
            self.selling_general_admin,
            self.marketing_expenses,
            self.research_development,
            self.depreciation_amortization,
            self.other_operating_expenses,
        ]
        for expense in expense_fields:
            if expense is not None:
                total += expense
        return total
    
    @property
    def operating_income(self) -> Decimal:
        """Calculate operating income (EBIT approximation)."""
        return self.gross_profit - self.total_operating_expenses
    
    @property
    def ebitda(self) -> Decimal:
        """Calculate EBITDA."""
        da = self.depreciation_amortization or Decimal("0")
        return self.operating_income + da
    
    @property
    def ebit(self) -> Decimal:
        """Calculate EBIT (same as operating_income)."""
        return self.operating_income
    
    @property
    def earnings_before_tax(self) -> Decimal:
        """Calculate earnings before tax."""
        return (
            self.operating_income
            + self.interest_income
            - self.interest_expense
            + self.other_income
            - self.other_expenses
        )
    
    @property
    def calculated_net_income(self) -> Decimal:
        """Calculate net income from components if not provided."""
        if self.net_income is not None:
            return self.net_income
        return self.earnings_before_tax - self.income_tax_expense
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with calculated fields included."""
        data = self.model_dump(by_alias=False)
        data["gross_profit"] = float(self.gross_profit)
        data["operating_income"] = float(self.operating_income)
        data["ebitda"] = float(self.ebitda)
        data["ebit"] = float(self.ebit)
        data["earnings_before_tax"] = float(self.earnings_before_tax)
        data["calculated_net_income"] = float(self.calculated_net_income)
        return data


class BalanceSheetData(BaseModel):
    """
    Balance Sheet data model.
    
    Organized into Current Assets, Non-Current Assets, Current Liabilities,
    Non-Current Liabilities, and Shareholders' Equity sections.
    """
    
    period: FinancialPeriod
    currency: str = Field(default="SGD", min_length=3, max_length=3)
    
    # ─────────────────────────────────────────────────────────────────────
    # Current Assets
    # ─────────────────────────────────────────────────────────────────────
    cash_and_equivalents: MonetaryValue = Field(
        ...,
        alias="cash",
        description="Cash and cash equivalents"
    )
    short_term_investments: MonetaryValue = Field(default=Decimal("0"))
    accounts_receivable: MonetaryValue = Field(default=Decimal("0"), alias="ar")
    inventory: MonetaryValue = Field(default=Decimal("0"))
    prepaid_expenses: MonetaryValue = Field(default=Decimal("0"))
    other_current_assets: MonetaryValue = Field(default=Decimal("0"))
    total_current_assets: MonetaryValue | None = Field(default=None)
    
    # ─────────────────────────────────────────────────────────────────────
    # Non-Current Assets
    # ─────────────────────────────────────────────────────────────────────
    property_plant_equipment: MonetaryValue = Field(default=Decimal("0"), alias="ppe")
    intangible_assets: MonetaryValue = Field(default=Decimal("0"))
    goodwill: MonetaryValue = Field(default=Decimal("0"))
    long_term_investments: MonetaryValue = Field(default=Decimal("0"))
    deferred_tax_assets: MonetaryValue = Field(default=Decimal("0"))
    other_non_current_assets: MonetaryValue = Field(default=Decimal("0"))
    total_non_current_assets: MonetaryValue | None = Field(default=None)
    
    total_assets: MonetaryValue | None = Field(default=None)
    
    # ─────────────────────────────────────────────────────────────────────
    # Current Liabilities
    # ─────────────────────────────────────────────────────────────────────
    accounts_payable: MonetaryValue = Field(default=Decimal("0"), alias="ap")
    short_term_debt: MonetaryValue = Field(default=Decimal("0"))
    accrued_liabilities: MonetaryValue = Field(default=Decimal("0"))
    deferred_revenue: MonetaryValue = Field(default=Decimal("0"))
    income_taxes_payable: MonetaryValue = Field(default=Decimal("0"))
    other_current_liabilities: MonetaryValue = Field(default=Decimal("0"))
    total_current_liabilities: MonetaryValue | None = Field(default=None)
    
    # ─────────────────────────────────────────────────────────────────────
    # Non-Current Liabilities
    # ─────────────────────────────────────────────────────────────────────
    long_term_debt: MonetaryValue = Field(default=Decimal("0"))
    deferred_tax_liabilities: MonetaryValue = Field(default=Decimal("0"))
    pension_liabilities: MonetaryValue = Field(default=Decimal("0"))
    other_non_current_liabilities: MonetaryValue = Field(default=Decimal("0"))
    total_non_current_liabilities: MonetaryValue | None = Field(default=None)
    
    total_liabilities: MonetaryValue | None = Field(default=None)
    
    # ─────────────────────────────────────────────────────────────────────
    # Shareholders' Equity
    # ─────────────────────────────────────────────────────────────────────
    common_stock: MonetaryValue = Field(default=Decimal("0"))
    preferred_stock: MonetaryValue = Field(default=Decimal("0"))
    additional_paid_in_capital: MonetaryValue = Field(default=Decimal("0"), alias="apic")
    retained_earnings: MonetaryValue = Field(default=Decimal("0"))
    treasury_stock: MonetaryValue = Field(default=Decimal("0"))
    accumulated_other_comprehensive_income: MonetaryValue = Field(default=Decimal("0"), alias="aoci")
    total_shareholders_equity: MonetaryValue | None = Field(default=None)
    non_controlling_interest: MonetaryValue = Field(default=Decimal("0"))
    
    model_config = {
        "populate_by_name": True,
        "extra": "allow",
    }
    
    @field_validator("cash_and_equivalents", mode="before")
    @classmethod
    def convert_to_decimal(cls, v: Any) -> Decimal:
        """Convert numeric inputs to Decimal."""
        if v is None:
            raise ValueError("Cash and equivalents is required")
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))
    
    @property
    def calculated_current_assets(self) -> Decimal:
        """Calculate total current assets from components."""
        if self.total_current_assets is not None:
            return self.total_current_assets
        return (
            self.cash_and_equivalents
            + self.short_term_investments
            + self.accounts_receivable
            + self.inventory
            + self.prepaid_expenses
            + self.other_current_assets
        )
    
    @property
    def calculated_non_current_assets(self) -> Decimal:
        """Calculate total non-current assets from components."""
        if self.total_non_current_assets is not None:
            return self.total_non_current_assets
        return (
            self.property_plant_equipment
            + self.intangible_assets
            + self.goodwill
            + self.long_term_investments
            + self.deferred_tax_assets
            + self.other_non_current_assets
        )
    
    @property
    def calculated_total_assets(self) -> Decimal:
        """Calculate total assets from components."""
        if self.total_assets is not None:
            return self.total_assets
        return self.calculated_current_assets + self.calculated_non_current_assets
    
    @property
    def calculated_current_liabilities(self) -> Decimal:
        """Calculate total current liabilities from components."""
        if self.total_current_liabilities is not None:
            return self.total_current_liabilities
        return (
            self.accounts_payable
            + self.short_term_debt
            + self.accrued_liabilities
            + self.deferred_revenue
            + self.income_taxes_payable
            + self.other_current_liabilities
        )
    
    @property
    def calculated_non_current_liabilities(self) -> Decimal:
        """Calculate total non-current liabilities from components."""
        if self.total_non_current_liabilities is not None:
            return self.total_non_current_liabilities
        return (
            self.long_term_debt
            + self.deferred_tax_liabilities
            + self.pension_liabilities
            + self.other_non_current_liabilities
        )
    
    @property
    def calculated_total_liabilities(self) -> Decimal:
        """Calculate total liabilities from components."""
        if self.total_liabilities is not None:
            return self.total_liabilities
        return self.calculated_current_liabilities + self.calculated_non_current_liabilities
    
    @property
    def calculated_shareholders_equity(self) -> Decimal:
        """Calculate shareholders' equity from components."""
        if self.total_shareholders_equity is not None:
            return self.total_shareholders_equity
        return (
            self.common_stock
            + self.preferred_stock
            + self.additional_paid_in_capital
            + self.retained_earnings
            - self.treasury_stock
            + self.accumulated_other_comprehensive_income
        )
    
    @property
    def calculated_total_equity(self) -> Decimal:
        """Calculate total equity including non-controlling interest."""
        return self.calculated_shareholders_equity + self.non_controlling_interest
    
    @property
    def working_capital(self) -> Decimal:
        """Calculate working capital."""
        return self.calculated_current_assets - self.calculated_current_liabilities
    
    @property
    def total_debt(self) -> Decimal:
        """Calculate total debt (short-term + long-term)."""
        return self.short_term_debt + self.long_term_debt
    
    def check_balance_sheet_equation(self, tolerance: Decimal = Decimal("0.01")) -> bool:
        """Verify Assets = Liabilities + Equity."""
        assets = self.calculated_total_assets
        liab_equity = self.calculated_total_liabilities + self.calculated_total_equity
        difference = abs(assets - liab_equity)
        return difference <= tolerance
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with calculated fields included."""
        data = self.model_dump(by_alias=False)
        data["calculated_current_assets"] = float(self.calculated_current_assets)
        data["calculated_non_current_assets"] = float(self.calculated_non_current_assets)
        data["calculated_total_assets"] = float(self.calculated_total_assets)
        data["calculated_current_liabilities"] = float(self.calculated_current_liabilities)
        data["calculated_non_current_liabilities"] = float(self.calculated_non_current_liabilities)
        data["calculated_total_liabilities"] = float(self.calculated_total_liabilities)
        data["calculated_shareholders_equity"] = float(self.calculated_shareholders_equity)
        data["calculated_total_equity"] = float(self.calculated_total_equity)
        data["working_capital"] = float(self.working_capital)
        data["total_debt"] = float(self.total_debt)
        return data


class CashFlowStatementData(BaseModel):
    """
    Cash Flow Statement data model.
    
    Organized into Operating, Investing, and Financing activities.
    """
    
    period: FinancialPeriod
    currency: str = Field(default="SGD", min_length=3, max_length=3)
    
    # ─────────────────────────────────────────────────────────────────────
    # Operating Activities
    # ─────────────────────────────────────────────────────────────────────
    net_income: MonetaryValue = Field(..., description="Net income (starting point)")
    depreciation_amortization: MonetaryValue = Field(default=Decimal("0"))
    stock_based_compensation: MonetaryValue = Field(default=Decimal("0"))
    deferred_taxes: MonetaryValue = Field(default=Decimal("0"))
    
    # Working capital changes
    change_in_receivables: MonetaryValue = Field(default=Decimal("0"))
    change_in_inventory: MonetaryValue = Field(default=Decimal("0"))
    change_in_payables: MonetaryValue = Field(default=Decimal("0"))
    change_in_other_working_capital: MonetaryValue = Field(default=Decimal("0"))
    other_operating_activities: MonetaryValue = Field(default=Decimal("0"))
    net_cash_from_operating: MonetaryValue | None = Field(default=None, alias="cfo")
    
    # ─────────────────────────────────────────────────────────────────────
    # Investing Activities
    # ─────────────────────────────────────────────────────────────────────
    capital_expenditures: MonetaryValue = Field(default=Decimal("0"), alias="capex")
    acquisitions: MonetaryValue = Field(default=Decimal("0"))
    investment_purchases: MonetaryValue = Field(default=Decimal("0"))
    investment_sales: MonetaryValue = Field(default=Decimal("0"))
    other_investing_activities: MonetaryValue = Field(default=Decimal("0"))
    net_cash_from_investing: MonetaryValue | None = Field(default=None, alias="cfi")
    
    # ─────────────────────────────────────────────────────────────────────
    # Financing Activities
    # ─────────────────────────────────────────────────────────────────────
    debt_issued: MonetaryValue = Field(default=Decimal("0"))
    debt_repaid: MonetaryValue = Field(default=Decimal("0"))
    shares_issued: MonetaryValue = Field(default=Decimal("0"))
    shares_repurchased: MonetaryValue = Field(default=Decimal("0"))
    dividends_paid: MonetaryValue = Field(default=Decimal("0"))
    other_financing_activities: MonetaryValue = Field(default=Decimal("0"))
    net_cash_from_financing: MonetaryValue | None = Field(default=None, alias="cff")
    
    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────
    beginning_cash: MonetaryValue | None = Field(default=None)
    ending_cash: MonetaryValue | None = Field(default=None)
    net_change_in_cash: MonetaryValue | None = Field(default=None)
    
    model_config = {
        "populate_by_name": True,
        "extra": "allow",
    }
    
    @field_validator("net_income", mode="before")
    @classmethod
    def convert_to_decimal(cls, v: Any) -> Decimal:
        """Convert numeric inputs to Decimal."""
        if v is None:
            raise ValueError("Net income is required")
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))
    
    @property
    def calculated_operating_cash_flow(self) -> Decimal:
        """Calculate operating cash flow from components."""
        if self.net_cash_from_operating is not None:
            return self.net_cash_from_operating
        return (
            self.net_income
            + self.depreciation_amortization
            + self.stock_based_compensation
            + self.deferred_taxes
            - self.change_in_receivables
            - self.change_in_inventory
            + self.change_in_payables
            + self.change_in_other_working_capital
            + self.other_operating_activities
        )
    
    @property
    def calculated_investing_cash_flow(self) -> Decimal:
        """Calculate investing cash flow from components."""
        if self.net_cash_from_investing is not None:
            return self.net_cash_from_investing
        return (
            -abs(self.capital_expenditures)  # CapEx is usually an outflow
            - abs(self.acquisitions)
            - self.investment_purchases
            + self.investment_sales
            + self.other_investing_activities
        )
    
    @property
    def calculated_financing_cash_flow(self) -> Decimal:
        """Calculate financing cash flow from components."""
        if self.net_cash_from_financing is not None:
            return self.net_cash_from_financing
        return (
            self.debt_issued
            - self.debt_repaid
            + self.shares_issued
            - self.shares_repurchased
            - abs(self.dividends_paid)
            + self.other_financing_activities
        )
    
    @property
    def calculated_net_change(self) -> Decimal:
        """Calculate net change in cash."""
        if self.net_change_in_cash is not None:
            return self.net_change_in_cash
        return (
            self.calculated_operating_cash_flow
            + self.calculated_investing_cash_flow
            + self.calculated_financing_cash_flow
        )
    
    @property
    def free_cash_flow(self) -> Decimal:
        """Calculate free cash flow (CFO - CapEx)."""
        return self.calculated_operating_cash_flow - abs(self.capital_expenditures)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with calculated fields included."""
        data = self.model_dump(by_alias=False)
        data["calculated_operating_cash_flow"] = float(self.calculated_operating_cash_flow)
        data["calculated_investing_cash_flow"] = float(self.calculated_investing_cash_flow)
        data["calculated_financing_cash_flow"] = float(self.calculated_financing_cash_flow)
        data["calculated_net_change"] = float(self.calculated_net_change)
        data["free_cash_flow"] = float(self.free_cash_flow)
        return data


class FinancialStatementSet(BaseModel):
    """
    A complete set of financial statements for a single period.
    
    Combines Income Statement, Balance Sheet, and Cash Flow Statement.
    """
    
    income_statement: IncomeStatementData
    balance_sheet: BalanceSheetData
    cash_flow_statement: CashFlowStatementData | None = None
    
    @model_validator(mode="after")
    def validate_period_consistency(self) -> "FinancialStatementSet":
        """Ensure all statements are for the same period."""
        is_period = self.income_statement.period
        bs_period = self.balance_sheet.period
        
        if is_period != bs_period:
            raise ValueError(
                f"Period mismatch: Income Statement is for {is_period}, "
                f"Balance Sheet is for {bs_period}"
            )
        
        if self.cash_flow_statement:
            cf_period = self.cash_flow_statement.period
            if is_period != cf_period:
                raise ValueError(
                    f"Period mismatch: Income Statement is for {is_period}, "
                    f"Cash Flow is for {cf_period}"
                )
        
        return self
    
    @property
    def period(self) -> FinancialPeriod:
        """Get the period for this statement set."""
        return self.income_statement.period
    
    @property
    def currency(self) -> str:
        """Get the currency for this statement set."""
        return self.income_statement.currency
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "period": str(self.period),
            "currency": self.currency,
            "income_statement": self.income_statement.to_dict(),
            "balance_sheet": self.balance_sheet.to_dict(),
        }
        if self.cash_flow_statement:
            result["cash_flow_statement"] = self.cash_flow_statement.to_dict()
        return result


class MultiPeriodFinancialData(BaseModel):
    """
    Financial data spanning multiple periods for trend analysis.
    """
    
    periods: list[FinancialStatementSet] = Field(
        ...,
        min_length=1,
        description="List of financial statement sets, one per period"
    )
    
    @model_validator(mode="after")
    def sort_by_period(self) -> "MultiPeriodFinancialData":
        """Sort periods chronologically."""
        self.periods.sort(key=lambda x: x.period)
        return self
    
    @property
    def period_count(self) -> int:
        """Number of periods in the data."""
        return len(self.periods)
    
    @property
    def earliest_period(self) -> FinancialPeriod:
        """Get the earliest period."""
        return self.periods[0].period
    
    @property
    def latest_period(self) -> FinancialPeriod:
        """Get the latest period."""
        return self.periods[-1].period
    
    @property
    def currency(self) -> str:
        """Get the currency (from first period)."""
        return self.periods[0].currency
    
    def get_period(self, period: FinancialPeriod) -> FinancialStatementSet | None:
        """Get statement set for a specific period."""
        for stmt_set in self.periods:
            if stmt_set.period == period:
                return stmt_set
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period_count": self.period_count,
            "earliest_period": str(self.earliest_period),
            "latest_period": str(self.latest_period),
            "currency": self.currency,
            "periods": [p.to_dict() for p in self.periods],
        }

```

# finanalyst_tools/validation/__init__.py
```py
# File: finanalyst_tools/validation/__init__.py
"""
Validation functions for FinAnalyst-Pro Agent Tools.

This package provides:
- Schema validation for financial statements
- Cross-statement reconciliation
- Plausibility checks for calculated metrics
"""

from finanalyst_tools.validation.schema_validator import (
    validate_income_statement_schema,
    validate_balance_sheet_schema,
    validate_cash_flow_schema,
    validate_financial_data_completeness,
    validate_statement_set,
    REQUIRED_FIELDS,
    FIELD_ALIASES,
)

from finanalyst_tools.validation.reconciliation import (
    reconcile_net_income,
    reconcile_cash_balance,
    reconcile_retained_earnings,
    reconcile_balance_sheet_equation,
    reconcile_working_capital,
    run_all_reconciliations,
)

from finanalyst_tools.validation.plausibility import (
    check_plausibility,
    check_all_plausibility,
    PlausibilityChecker,
)


__all__ = [
    # Schema validation
    "validate_income_statement_schema",
    "validate_balance_sheet_schema",
    "validate_cash_flow_schema",
    "validate_financial_data_completeness",
    "validate_statement_set",
    "REQUIRED_FIELDS",
    "FIELD_ALIASES",
    # Reconciliation
    "reconcile_net_income",
    "reconcile_cash_balance",
    "reconcile_retained_earnings",
    "reconcile_balance_sheet_equation",
    "reconcile_working_capital",
    "run_all_reconciliations",
    # Plausibility
    "check_plausibility",
    "check_all_plausibility",
    "PlausibilityChecker",
]

```

# finanalyst_tools/validation/plausibility.py
```py
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
) -> PlausibilityResult:
    """
    Check plausibility for a list of calculation results.
    
    Args:
        metrics: List of calculation results to check
        
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
        
        # Update the metric's plausibility status
        if not check.is_plausible:
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
    ) -> PlausibilityResult:
        """Check multiple metrics."""
        result = PlausibilityResult()
        
        for metric in metrics:
            check = self.check(metric.metric_name, metric.value)
            result.add_check(check)
            
            if not check.is_plausible:
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

```

# finanalyst_tools/validation/reconciliation.py
```py
# File: finanalyst_tools/validation/reconciliation.py
"""
Cross-statement reconciliation validation.

Verifies consistency between values that should match across
different financial statements:
- Net income (IS vs CF)
- Cash balance (BS vs CF)
- Retained earnings rollforward
- Balance sheet equation
- Working capital consistency
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from finanalyst_tools.config import ReconciliationTolerances
from finanalyst_tools.models.validation import (
    ReconciliationCheck,
    ReconciliationResult,
)
from finanalyst_tools.models.financial_statements import (
    IncomeStatementData,
    BalanceSheetData,
    CashFlowStatementData,
    FinancialStatementSet,
)
from finanalyst_tools.utils.math_ops import to_decimal, is_effectively_zero


def _create_check(
    check_name: str,
    statement_a: str,
    value_a: Decimal,
    statement_b: str,
    value_b: Decimal,
    tolerance_level: str = "normal",
) -> ReconciliationCheck:
    """
    Create a reconciliation check result.
    
    Args:
        check_name: Name of the check
        statement_a: Source of first value
        value_a: First value
        statement_b: Source of second value
        value_b: Second value
        tolerance_level: Tolerance level ("strict", "normal", "loose")
        
    Returns:
        ReconciliationCheck with pass/fail result
    """
    tolerance = ReconciliationTolerances.get_tolerance(tolerance_level)
    difference = abs(value_a - value_b)
    
    # Calculate if within tolerance
    passed = ReconciliationTolerances.is_within_tolerance(
        float(value_a),
        float(value_b),
        tolerance,
    )
    
    if passed:
        message = f"Values match within {tolerance:.1%} tolerance"
    else:
        pct_diff = (difference / max(abs(value_a), abs(value_b), Decimal("1"))) * 100
        message = f"Values differ by {difference:,.2f} ({pct_diff:.1f}%), exceeds {tolerance:.1%} tolerance"
    
    return ReconciliationCheck(
        check_name=check_name,
        statement_a=statement_a,
        value_a=value_a,
        statement_b=statement_b,
        value_b=value_b,
        difference=difference,
        tolerance=tolerance,
        passed=passed,
        message=message,
    )


def reconcile_net_income(
    income_statement: IncomeStatementData,
    cash_flow_statement: CashFlowStatementData,
) -> ReconciliationCheck:
    """
    Verify net income matches between Income Statement and Cash Flow Statement.
    
    Args:
        income_statement: Income statement data
        cash_flow_statement: Cash flow statement data
        
    Returns:
        ReconciliationCheck result
    """
    is_net_income = income_statement.calculated_net_income
    cf_net_income = cash_flow_statement.net_income
    
    return _create_check(
        check_name="Net Income Reconciliation",
        statement_a="Income Statement",
        value_a=is_net_income,
        statement_b="Cash Flow Statement",
        value_b=cf_net_income,
        tolerance_level="strict",
    )


def reconcile_cash_balance(
    balance_sheet: BalanceSheetData,
    cash_flow_statement: CashFlowStatementData,
) -> ReconciliationCheck | None:
    """
    Verify ending cash balance matches between Balance Sheet and Cash Flow Statement.
    
    Args:
        balance_sheet: Balance sheet data
        cash_flow_statement: Cash flow statement data
        
    Returns:
        ReconciliationCheck result or None if ending_cash not provided
    """
    if cash_flow_statement.ending_cash is None:
        return None
    
    bs_cash = balance_sheet.cash_and_equivalents
    cf_ending_cash = cash_flow_statement.ending_cash
    
    return _create_check(
        check_name="Cash Balance Reconciliation",
        statement_a="Balance Sheet",
        value_a=bs_cash,
        statement_b="Cash Flow (Ending)",
        value_b=cf_ending_cash,
        tolerance_level="strict",
    )


def reconcile_retained_earnings(
    current_balance_sheet: BalanceSheetData,
    prior_balance_sheet: BalanceSheetData | None,
    income_statement: IncomeStatementData,
    dividends_paid: Decimal | None = None,
) -> ReconciliationCheck | None:
    """
    Verify retained earnings rollforward.
    
    Formula: Prior RE + Net Income - Dividends = Current RE
    
    Args:
        current_balance_sheet: Current period balance sheet
        prior_balance_sheet: Prior period balance sheet
        income_statement: Current period income statement
        dividends_paid: Dividends paid during period (optional)
        
    Returns:
        ReconciliationCheck result or None if prior BS not provided
    """
    if prior_balance_sheet is None:
        return None
    
    prior_re = prior_balance_sheet.retained_earnings
    net_income = income_statement.calculated_net_income
    dividends = dividends_paid or Decimal("0")
    
    expected_re = prior_re + net_income - dividends
    actual_re = current_balance_sheet.retained_earnings
    
    return _create_check(
        check_name="Retained Earnings Rollforward",
        statement_a="Calculated (Prior RE + NI - Div)",
        value_a=expected_re,
        statement_b="Balance Sheet",
        value_b=actual_re,
        tolerance_level="normal",
    )


def reconcile_balance_sheet_equation(
    balance_sheet: BalanceSheetData,
) -> ReconciliationCheck:
    """
    Verify the fundamental accounting equation: Assets = Liabilities + Equity.
    
    Args:
        balance_sheet: Balance sheet data
        
    Returns:
        ReconciliationCheck result
    """
    total_assets = balance_sheet.calculated_total_assets
    total_liab_equity = (
        balance_sheet.calculated_total_liabilities +
        balance_sheet.calculated_total_equity
    )
    
    return _create_check(
        check_name="Balance Sheet Equation",
        statement_a="Total Assets",
        value_a=total_assets,
        statement_b="Liabilities + Equity",
        value_b=total_liab_equity,
        tolerance_level="strict",
    )


def reconcile_working_capital(
    balance_sheet: BalanceSheetData,
) -> ReconciliationCheck:
    """
    Verify working capital calculation consistency.
    
    Args:
        balance_sheet: Balance sheet data
        
    Returns:
        ReconciliationCheck result
    """
    current_assets = balance_sheet.calculated_current_assets
    current_liabilities = balance_sheet.calculated_current_liabilities
    calculated_wc = current_assets - current_liabilities
    
    # Compare with the property calculation
    property_wc = balance_sheet.working_capital
    
    return _create_check(
        check_name="Working Capital Consistency",
        statement_a="CA - CL Calculation",
        value_a=calculated_wc,
        statement_b="Working Capital Property",
        value_b=property_wc,
        tolerance_level="strict",
    )


def run_all_reconciliations(
    statement_set: FinancialStatementSet,
    prior_balance_sheet: BalanceSheetData | None = None,
    dividends_paid: Decimal | None = None,
) -> ReconciliationResult:
    """
    Run all applicable reconciliation checks.
    
    Args:
        statement_set: Complete set of financial statements
        prior_balance_sheet: Prior period balance sheet (optional)
        dividends_paid: Dividends paid during period (optional)
        
    Returns:
        ReconciliationResult with all check results
    """
    result = ReconciliationResult()
    
    # Balance sheet equation (always run)
    bs_equation = reconcile_balance_sheet_equation(statement_set.balance_sheet)
    result.add_check(bs_equation)
    
    # Working capital consistency (always run)
    wc_check = reconcile_working_capital(statement_set.balance_sheet)
    result.add_check(wc_check)
    
    # Net income reconciliation (if cash flow available)
    if statement_set.cash_flow_statement:
        ni_check = reconcile_net_income(
            statement_set.income_statement,
            statement_set.cash_flow_statement,
        )
        result.add_check(ni_check)
        
        # Cash balance reconciliation
        cash_check = reconcile_cash_balance(
            statement_set.balance_sheet,
            statement_set.cash_flow_statement,
        )
        if cash_check:
            result.add_check(cash_check)
    
    # Retained earnings rollforward (if prior BS available)
    if prior_balance_sheet:
        re_check = reconcile_retained_earnings(
            statement_set.balance_sheet,
            prior_balance_sheet,
            statement_set.income_statement,
            dividends_paid,
        )
        if re_check:
            result.add_check(re_check)
    
    return result

```

# finanalyst_tools/validation/schema_validator.py
```py
# File: finanalyst_tools/validation/schema_validator.py
"""
Schema validation for financial statement data.

Provides validation functions to verify:
- Required fields are present
- Field types are correct
- Data is complete for requested analysis type
"""

from __future__ import annotations

from typing import Any
from decimal import Decimal

from finanalyst_tools.models.validation import (
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
)
from finanalyst_tools.models.financial_statements import (
    IncomeStatementData,
    BalanceSheetData,
    CashFlowStatementData,
    FinancialStatementSet,
)


# ============================================================================
# FIELD DEFINITIONS
# ============================================================================

REQUIRED_FIELDS: dict[str, dict[str, list[str]]] = {
    "profitability": {
        "income_statement": ["total_revenue", "cost_of_goods_sold"],
        "balance_sheet": ["total_assets", "total_shareholders_equity"],
    },
    "liquidity": {
        "balance_sheet": [
            "current_assets", "current_liabilities",
            "cash_and_equivalents", "inventory"
        ],
    },
    "solvency": {
        "balance_sheet": [
            "total_liabilities", "total_shareholders_equity",
            "total_assets", "long_term_debt"
        ],
        "income_statement": ["interest_expense"],
    },
    "efficiency": {
        "income_statement": ["total_revenue", "cost_of_goods_sold"],
        "balance_sheet": ["inventory", "accounts_receivable", "accounts_payable", "total_assets"],
    },
    "comprehensive": {
        "income_statement": [
            "total_revenue", "cost_of_goods_sold", "net_income"
        ],
        "balance_sheet": [
            "total_assets", "total_liabilities", "total_shareholders_equity",
            "current_assets", "current_liabilities", "cash_and_equivalents"
        ],
    },
}

FIELD_ALIASES: dict[str, list[str]] = {
    "total_revenue": ["revenue", "net_revenue", "net_sales", "sales", "total_sales"],
    "cost_of_goods_sold": ["cogs", "cost_of_sales", "cost_of_revenue"],
    "net_income": ["net_profit", "net_earnings", "profit_after_tax"],
    "total_assets": ["assets"],
    "total_liabilities": ["liabilities"],
    "total_shareholders_equity": ["shareholders_equity", "equity", "total_equity", "stockholders_equity"],
    "current_assets": ["total_current_assets"],
    "current_liabilities": ["total_current_liabilities"],
    "cash_and_equivalents": ["cash", "cash_and_cash_equivalents"],
    "accounts_receivable": ["ar", "trade_receivables", "receivables"],
    "accounts_payable": ["ap", "trade_payables", "payables"],
    "inventory": ["inventories", "stock"],
    "long_term_debt": ["lt_debt", "non_current_debt"],
    "interest_expense": ["interest_cost", "finance_cost"],
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_field_value(
    data: dict[str, Any],
    canonical_name: str,
) -> tuple[Any, str | None]:
    """
    Find a field value by canonical name or any of its aliases.
    
    Args:
        data: Dictionary of data fields
        canonical_name: The canonical field name to look for
        
    Returns:
        Tuple of (value, found_field_name) or (None, None) if not found
    """
    # Check canonical name first
    if canonical_name in data and data[canonical_name] is not None:
        return data[canonical_name], canonical_name
    
    # Check aliases
    aliases = FIELD_ALIASES.get(canonical_name, [])
    for alias in aliases:
        if alias in data and data[alias] is not None:
            return data[alias], alias
    
    # Check case-insensitive
    data_lower = {k.lower(): (v, k) for k, v in data.items()}
    if canonical_name.lower() in data_lower:
        value, original_key = data_lower[canonical_name.lower()]
        if value is not None:
            return value, original_key
    
    for alias in aliases:
        if alias.lower() in data_lower:
            value, original_key = data_lower[alias.lower()]
            if value is not None:
                return value, original_key
    
    return None, None


def is_numeric(value: Any) -> bool:
    """Check if a value is numeric."""
    if value is None:
        return False
    if isinstance(value, (int, float, Decimal)):
        return True
    if isinstance(value, str):
        try:
            Decimal(value)
            return True
        except:
            return False
    return False


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_income_statement_schema(
    data: dict[str, Any] | IncomeStatementData,
) -> ValidationResult:
    """
    Validate income statement data structure.
    
    Args:
        data: Income statement data (dict or model)
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult()
    
    # Convert model to dict if needed
    if isinstance(data, IncomeStatementData):
        data_dict = data.model_dump(by_alias=True)
    else:
        data_dict = data
    
    # Required fields for basic income statement
    required = ["total_revenue", "cost_of_goods_sold"]
    
    for field in required:
        value, found_name = find_field_value(data_dict, field)
        if value is None:
            result.add_error(
                field=field,
                message=f"Required field '{field}' is missing",
                expected="Numeric value",
                suggestion=f"Provide {field} value. Accepted aliases: {FIELD_ALIASES.get(field, [])}",
            )
        elif not is_numeric(value):
            result.add_error(
                field=found_name or field,
                message=f"Field '{found_name or field}' must be numeric",
                actual_value=value,
                expected="Numeric value (int, float, or Decimal)",
            )
    
    # Validate revenue > 0 (usually)
    revenue, _ = find_field_value(data_dict, "total_revenue")
    if revenue is not None and is_numeric(revenue):
        if Decimal(str(revenue)) < 0:
            result.add_warning(
                field="total_revenue",
                message="Revenue is negative, which is unusual",
                actual_value=float(revenue),
                expected="Typically positive value",
            )
    
    return result


def validate_balance_sheet_schema(
    data: dict[str, Any] | BalanceSheetData,
) -> ValidationResult:
    """
    Validate balance sheet data structure.
    
    Args:
        data: Balance sheet data (dict or model)
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult()
    
    # Convert model to dict if needed
    if isinstance(data, BalanceSheetData):
        data_dict = data.model_dump(by_alias=True)
    else:
        data_dict = data
    
    # Required fields for basic balance sheet
    required = ["cash_and_equivalents"]
    
    for field in required:
        value, found_name = find_field_value(data_dict, field)
        if value is None:
            result.add_error(
                field=field,
                message=f"Required field '{field}' is missing",
                expected="Numeric value",
                suggestion=f"Provide {field} value. Accepted aliases: {FIELD_ALIASES.get(field, [])}",
            )
        elif not is_numeric(value):
            result.add_error(
                field=found_name or field,
                message=f"Field '{found_name or field}' must be numeric",
                actual_value=value,
                expected="Numeric value",
            )
    
    # Validate cash >= 0
    cash, _ = find_field_value(data_dict, "cash_and_equivalents")
    if cash is not None and is_numeric(cash):
        if Decimal(str(cash)) < 0:
            result.add_error(
                field="cash_and_equivalents",
                message="Cash cannot be negative",
                actual_value=float(cash),
                expected="Non-negative value",
            )
    
    # Check balance sheet equation if all components present
    assets, _ = find_field_value(data_dict, "total_assets")
    liabilities, _ = find_field_value(data_dict, "total_liabilities")
    equity, _ = find_field_value(data_dict, "total_shareholders_equity")
    
    if all(v is not None and is_numeric(v) for v in [assets, liabilities, equity]):
        assets_dec = Decimal(str(assets))
        liab_equity = Decimal(str(liabilities)) + Decimal(str(equity))
        diff = abs(assets_dec - liab_equity)
        
        # Allow 1% tolerance
        if assets_dec != 0:
            tolerance = abs(assets_dec) * Decimal("0.01")
            if diff > tolerance:
                result.add_warning(
                    field="balance_sheet_equation",
                    message="Assets ≠ Liabilities + Equity",
                    actual_value=f"Assets={assets}, L+E={liab_equity}, Diff={diff}",
                    expected="Assets = Liabilities + Equity (within 1%)",
                )
    
    return result


def validate_cash_flow_schema(
    data: dict[str, Any] | CashFlowStatementData,
) -> ValidationResult:
    """
    Validate cash flow statement data structure.
    
    Args:
        data: Cash flow statement data (dict or model)
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult()
    
    # Convert model to dict if needed
    if isinstance(data, CashFlowStatementData):
        data_dict = data.model_dump(by_alias=True)
    else:
        data_dict = data
    
    # Required field
    if "net_income" not in data_dict or data_dict["net_income"] is None:
        result.add_error(
            field="net_income",
            message="Required field 'net_income' is missing from cash flow statement",
            expected="Numeric value matching income statement",
        )
    
    return result


def validate_financial_data_completeness(
    income_statement: dict[str, Any] | IncomeStatementData | None,
    balance_sheet: dict[str, Any] | BalanceSheetData | None,
    cash_flow: dict[str, Any] | CashFlowStatementData | None,
    analysis_type: str,
) -> ValidationResult:
    """
    Validate that all required data is present for the requested analysis type.
    
    Args:
        income_statement: Income statement data
        balance_sheet: Balance sheet data
        cash_flow: Cash flow statement data
        analysis_type: Type of analysis requested
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult()
    
    # Get required fields for this analysis type
    requirements = REQUIRED_FIELDS.get(analysis_type.lower(), {})
    
    if not requirements:
        result.add_warning(
            field="analysis_type",
            message=f"Unknown analysis type: {analysis_type}",
            suggestion="Using default comprehensive requirements",
        )
        requirements = REQUIRED_FIELDS.get("comprehensive", {})
    
    # Check income statement requirements
    if "income_statement" in requirements:
        if income_statement is None:
            result.add_error(
                field="income_statement",
                message=f"Income statement is required for {analysis_type} analysis",
                expected="Complete income statement data",
            )
        else:
            is_dict = income_statement.model_dump(by_alias=True) if isinstance(income_statement, IncomeStatementData) else income_statement
            for field in requirements["income_statement"]:
                value, _ = find_field_value(is_dict, field)
                if value is None:
                    result.add_error(
                        field=field,
                        message=f"Field '{field}' is required for {analysis_type} analysis",
                        expected="Numeric value",
                    )
    
    # Check balance sheet requirements
    if "balance_sheet" in requirements:
        if balance_sheet is None:
            result.add_error(
                field="balance_sheet",
                message=f"Balance sheet is required for {analysis_type} analysis",
                expected="Complete balance sheet data",
            )
        else:
            bs_dict = balance_sheet.model_dump(by_alias=True) if isinstance(balance_sheet, BalanceSheetData) else balance_sheet
            for field in requirements["balance_sheet"]:
                value, _ = find_field_value(bs_dict, field)
                if value is None:
                    result.add_warning(
                        field=field,
                        message=f"Field '{field}' is recommended for {analysis_type} analysis",
                        suggestion="Some calculations may be skipped",
                    )
    
    # Check cash flow requirements
    if "cash_flow" in requirements:
        if cash_flow is None:
            result.add_warning(
                field="cash_flow",
                message=f"Cash flow statement is recommended for {analysis_type} analysis",
                suggestion="Cash flow metrics will be skipped",
            )
    
    return result


def validate_statement_set(
    statement_set: FinancialStatementSet,
    analysis_type: str = "comprehensive",
) -> ValidationResult:
    """
    Validate a complete financial statement set.
    
    Args:
        statement_set: Complete set of financial statements
        analysis_type: Type of analysis to validate for
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult()
    
    # Validate individual statements
    is_result = validate_income_statement_schema(statement_set.income_statement)
    result.merge(is_result)
    
    bs_result = validate_balance_sheet_schema(statement_set.balance_sheet)
    result.merge(bs_result)
    
    if statement_set.cash_flow_statement:
        cf_result = validate_cash_flow_schema(statement_set.cash_flow_statement)
        result.merge(cf_result)
    
    # Validate completeness for analysis type
    completeness_result = validate_financial_data_completeness(
        statement_set.income_statement,
        statement_set.balance_sheet,
        statement_set.cash_flow_statement,
        analysis_type,
    )
    result.merge(completeness_result)
    
    return result

```

