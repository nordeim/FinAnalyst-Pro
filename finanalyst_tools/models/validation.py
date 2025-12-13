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

from finanalyst_tools.utils.serialization import to_jsonable


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
    context: dict[str, Any] = field(default_factory=dict)
    
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
        self.context.update(other.context)
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
            "context": self.context,
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
        return to_jsonable({
            "check_name": self.check_name,
            "statement_a": self.statement_a,
            "value_a": self.value_a,
            "statement_b": self.statement_b,
            "value_b": self.value_b,
            "difference": self.difference,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "message": self.message,
        })
    
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
        return to_jsonable({
            "metric_name": self.metric_name,
            "value": self.value,
            "plausible_range": self.plausible_range,
            "is_plausible": self.is_plausible,
            "assessment": self.assessment,
            "severity": self.severity.value,
            "message": self.message,
        })
    
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
                    actual_value=check.value,
                    expected=f"Between {check.plausible_range[0]} and {check.plausible_range[1]}",
                    suggestion="Verify input data accuracy",
                )
        
        return result
