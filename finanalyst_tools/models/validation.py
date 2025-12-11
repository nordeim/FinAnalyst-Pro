# finanalyst_tools/models/validation.py
"""
Models for validation results and issues.

Provides structured types for:
- Validation issues with severity levels
- Validation results with aggregated issues
- Reconciliation check results
- Plausibility check results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any
import json


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Blocking - cannot proceed
    WARNING = "warning"  # Non-blocking - proceed with caution
    INFO = "info"        # Informational only


@dataclass
class ValidationIssue:
    """
    Represents a single validation issue.
    
    Attributes:
        field: The field that has the issue
        message: Description of the issue
        severity: ERROR, WARNING, or INFO
        actual_value: The value that caused the issue
        expected: Description of what was expected
        suggestion: How to fix the issue
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
            result["actual_value"] = str(self.actual_value)
        if self.expected:
            result["expected"] = self.expected
        if self.suggestion:
            result["suggestion"] = self.suggestion
        return result
    
    def __str__(self) -> str:
        icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[self.severity.value]
        return f"{icon} [{self.field}] {self.message}"


@dataclass
class ValidationResult:
    """
    Aggregated result of validation checks.
    
    Contains lists of issues categorized by severity.
    Provides properties for quick status checks.
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
        """Whether analysis can proceed (no errors)."""
        return self.error_count == 0
    
    @property
    def all_issues(self) -> list[ValidationIssue]:
        """All issues combined, sorted by severity."""
        return self.issues + self.warnings + self.info
    
    def add_issue(
        self,
        field: str,
        message: str,
        severity: ValidationSeverity,
        actual_value: Any = None,
        expected: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Add a validation issue."""
        issue = ValidationIssue(
            field=field,
            message=message,
            severity=severity,
            actual_value=actual_value,
            expected=expected,
            suggestion=suggestion,
        )
        
        if severity == ValidationSeverity.ERROR:
            self.issues.append(issue)
            self.is_valid = False
        elif severity == ValidationSeverity.WARNING:
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
        self.add_issue(
            field, message, ValidationSeverity.ERROR,
            actual_value, expected, suggestion
        )
    
    def add_warning(
        self,
        field: str,
        message: str,
        actual_value: Any = None,
        expected: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Convenience method to add a warning."""
        self.add_issue(
            field, message, ValidationSeverity.WARNING,
            actual_value, expected, suggestion
        )
    
    def add_info(self, field: str, message: str) -> None:
        """Convenience method to add an info message."""
        self.add_issue(field, message, ValidationSeverity.INFO)
    
    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """
        Merge another ValidationResult into this one.
        
        Returns a new ValidationResult with combined issues.
        """
        merged = ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            issues=self.issues + other.issues,
            warnings=self.warnings + other.warnings,
            info=self.info + other.info,
            context={**self.context, **other.context},
        )
        return merged
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "can_proceed": self.can_proceed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "issues": [i.to_dict() for i in self.issues],
            "warnings": [w.to_dict() for w in self.warnings],
            "info": [i.to_dict() for i in self.info],
            "context": self.context,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def to_markdown(self) -> str:
        """Format as markdown for display."""
        lines = []
        
        status = "✅ Valid" if self.is_valid else "❌ Invalid"
        lines.append(f"**Validation Status**: {status}")
        lines.append(f"- Errors: {self.error_count}")
        lines.append(f"- Warnings: {self.warning_count}")
        lines.append(f"- Info: {self.info_count}")
        lines.append("")
        
        if self.issues:
            lines.append("### Errors")
            for issue in self.issues:
                lines.append(f"- {issue}")
            lines.append("")
        
        if self.warnings:
            lines.append("### Warnings")
            for warning in self.warnings:
                lines.append(f"- {warning}")
            lines.append("")
        
        if self.info:
            lines.append("### Information")
            for info in self.info:
                lines.append(f"- {info}")
        
        return "\n".join(lines)


@dataclass
class ReconciliationCheck:
    """
    Result of a single reconciliation check.
    
    Compares values between two sources (statements).
    """
    check_name: str
    statement_a: str
    value_a: Decimal
    statement_b: str
    value_b: Decimal
    difference: Decimal
    tolerance: float
    passed: bool
    message: str = ""
    
    def __post_init__(self):
        if not self.message:
            if self.passed:
                self.message = f"{self.check_name}: Values match within tolerance"
            else:
                self.message = (
                    f"{self.check_name}: Mismatch - {self.statement_a} has "
                    f"{self.value_a}, {self.statement_b} has {self.value_b} "
                    f"(difference: {self.difference}, tolerance: {self.tolerance:.2%})"
                )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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


@dataclass
class ReconciliationResult:
    """
    Aggregated result of reconciliation checks.
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
        """Add a reconciliation check."""
        self.checks.append(check)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "all_passed": self.all_passed,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "checks": [check.to_dict() for check in self.checks],
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_markdown(self) -> str:
        """Format as markdown."""
        lines = []
        
        status = "✅ All Passed" if self.all_passed else "❌ Some Failed"
        lines.append(f"**Reconciliation Status**: {status}")
        lines.append(f"- Passed: {self.passed_count}/{len(self.checks)}")
        lines.append("")
        
        if not self.all_passed:
            lines.append("### Failed Checks")
            for check in self.failed_checks:
                lines.append(f"- {check.message}")
        
        return "\n".join(lines)


@dataclass
class PlausibilityCheck:
    """
    Result of a single plausibility check.
    """
    metric_name: str
    value: Decimal
    min_plausible: float
    max_plausible: float
    is_plausible: bool
    severity: ValidationSeverity
    message: str = ""
    
    def __post_init__(self):
        if not self.message:
            if self.is_plausible:
                self.message = f"{self.metric_name}: {self.value} is within plausible range"
            else:
                self.message = (
                    f"{self.metric_name}: {self.value} is outside plausible range "
                    f"({self.min_plausible} to {self.max_plausible})"
                )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "value": float(self.value),
            "min_plausible": self.min_plausible,
            "max_plausible": self.max_plausible,
            "is_plausible": self.is_plausible,
            "severity": self.severity.value,
            "message": self.message,
        }


@dataclass
class PlausibilityResult:
    """
    Aggregated result of plausibility checks.
    """
    checks: list[PlausibilityCheck] = field(default_factory=list)
    
    @property
    def all_plausible(self) -> bool:
        """Whether all values are plausible."""
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
        """Add a plausibility check."""
        self.checks.append(check)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "all_plausible": self.all_plausible,
            "plausible_count": self.plausible_count,
            "implausible_count": self.implausible_count,
            "checks": [check.to_dict() for check in self.checks],
        }
    
    def to_validation_result(self) -> ValidationResult:
        """Convert to ValidationResult for unified handling."""
        result = ValidationResult()
        for check in self.checks:
            if not check.is_plausible:
                result.add_issue(
                    field=check.metric_name,
                    message=check.message,
                    severity=check.severity,
                    actual_value=check.value,
                    expected=f"Between {check.min_plausible} and {check.max_plausible}",
                )
        return result
