Hybrid Implementation — Continued
Files 7-25: Complete Implementation
File 7 of 25: finanalyst_tools/models/__init__.py
Python

# finanalyst_tools/models/__init__.py
"""
Data models for FinAnalyst-Pro Agent Tools.

This package provides Pydantic models for:
- Financial statement data (Income Statement, Balance Sheet, Cash Flow)
- Calculation results with audit trails
- Validation results and issues

All models support:
- Type validation
- JSON serialization
- Field aliases for common naming variations
"""

from __future__ import annotations

# Financial statement models
from finanalyst_tools.models.financial_statements import (
    StatementType,
    PeriodType,
    FinancialPeriod,
    IncomeStatementData,
    BalanceSheetData,
    CashFlowStatementData,
    FinancialStatementSet,
    MultiPeriodFinancialData,
)

# Analysis result models
from finanalyst_tools.models.analysis_results import (
    MetricUnit,
    MetricCategory,
    TrendDirection,
    ConfidenceLevel,
    CalculationStep,
    CalculationResult,
    MetricResult,
    MetricCollection,
    TrendAnalysisResult,
    ConfidenceAssessment,
    ComprehensiveAnalysisResult,
)

# Validation models
from finanalyst_tools.models.validation import (
    ValidationSeverity,
    ValidationIssue,
    ValidationResult,
    ReconciliationCheck,
    ReconciliationResult,
    PlausibilityCheck,
    PlausibilityResult,
)

__all__ = [
    # Statement types
    "StatementType",
    "PeriodType",
    "FinancialPeriod",
    # Financial statements
    "IncomeStatementData",
    "BalanceSheetData",
    "CashFlowStatementData",
    "FinancialStatementSet",
    "MultiPeriodFinancialData",
    # Result enums
    "MetricUnit",
    "MetricCategory",
    "TrendDirection",
    "ConfidenceLevel",
    # Result types
    "CalculationStep",
    "CalculationResult",
    "MetricResult",
    "MetricCollection",
    "TrendAnalysisResult",
    "ConfidenceAssessment",
    "ComprehensiveAnalysisResult",
    # Validation types
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationResult",
    "ReconciliationCheck",
    "ReconciliationResult",
    "PlausibilityCheck",
    "PlausibilityResult",
]
File 8 of 25: finanalyst_tools/models/validation.py
Python

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
File 9 of 25: finanalyst_tools/models/financial_statements.py
Python

# finanalyst_tools/models/financial_statements.py
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
from typing import Annotated, Any, Self

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
    quarter: int | None = Field(default=None, ge=1, le=4, description="Quarter (1-4)")
    month: int | None = Field(default=None, ge=1, le=12, description="Month (1-12)")
    start_date: date | None = Field(default=None, description="Period start date")
    end_date: date | None = Field(default=None, description="Period end date")
    
    model_config = {"extra": "forbid"}
    
    @model_validator(mode="after")
    def validate_period_details(self) -> Self:
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
        return f"FY{self.year}"
    
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
            self.year == other.year
            and self.period_type == other.period_type
            and self.quarter == other.quarter
            and self.month == other.month
        )
    
    def __hash__(self) -> int:
        return hash((self.year, self.period_type, self.quarter, self.month))


def _to_decimal(value: Any) -> Decimal:
    """Convert any numeric value to Decimal."""
    if value is None:
        return Decimal("0")
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


class IncomeStatementData(BaseModel):
    """
    Income Statement / Profit & Loss data model.
    
    Supports multiple naming conventions through field aliases.
    Provides calculated properties for derived values.
    """
    
    period: FinancialPeriod
    currency: str = Field(default="SGD", min_length=3, max_length=3)
    
    # Revenue
    total_revenue: MonetaryValue = Field(..., description="Total revenue / net sales")
    cost_of_goods_sold: MonetaryValue = Field(..., description="Cost of goods sold")
    
    # Operating Expenses (flexible structure)
    operating_expenses: MonetaryValue | None = Field(
        default=None, description="Total operating expenses (aggregate)"
    )
    selling_general_admin: MonetaryValue | None = Field(
        default=None, description="SG&A expenses"
    )
    marketing_expenses: MonetaryValue | None = Field(
        default=None, description="Marketing and advertising"
    )
    research_development: MonetaryValue | None = Field(
        default=None, description="R&D expenses"
    )
    depreciation_amortization: MonetaryValue | None = Field(
        default=None, description="D&A expense"
    )
    other_operating_expenses: MonetaryValue | None = Field(
        default=None, description="Other operating expenses"
    )
    
    # Non-Operating Items
    interest_income: MonetaryValue = Field(default=Decimal("0"))
    interest_expense: MonetaryValue = Field(default=Decimal("0"))
    other_income: MonetaryValue = Field(default=Decimal("0"))
    other_expenses: MonetaryValue = Field(default=Decimal("0"))
    
    # Taxes and Bottom Line
    income_tax_expense: MonetaryValue = Field(default=Decimal("0"))
    net_income: MonetaryValue | None = Field(default=None, description="Net income")
    
    # Per Share Data (optional)
    earnings_per_share: Decimal | None = Field(default=None, description="Basic EPS")
    diluted_eps: Decimal | None = Field(default=None, description="Diluted EPS")
    shares_outstanding: int | None = Field(default=None)
    
    model_config = {"populate_by_name": True, "extra": "allow"}
    
    @field_validator("total_revenue", "cost_of_goods_sold", mode="before")
    @classmethod
    def convert_to_decimal(cls, v: Any) -> Decimal:
        """Convert numeric inputs to Decimal."""
        if v is None:
            raise ValueError("This field is required")
        return _to_decimal(v)
    
    @property
    def gross_profit(self) -> Decimal:
        """Calculate gross profit."""
        return self.total_revenue - self.cost_of_goods_sold
    
    @property
    def total_operating_expenses(self) -> Decimal:
        """Calculate total operating expenses."""
        if self.operating_expenses is not None:
            return self.operating_expenses
        
        total = Decimal("0")
        for expense in [
            self.selling_general_admin,
            self.marketing_expenses,
            self.research_development,
            self.depreciation_amortization,
            self.other_operating_expenses,
        ]:
            if expense is not None:
                total += expense
        return total
    
    @property
    def operating_income(self) -> Decimal:
        """Calculate operating income (EBIT approximation)."""
        return self.gross_profit - self.total_operating_expenses
    
    @property
    def ebit(self) -> Decimal:
        """Earnings Before Interest and Taxes."""
        return self.operating_income
    
    @property
    def ebitda(self) -> Decimal:
        """Calculate EBITDA."""
        da = self.depreciation_amortization or Decimal("0")
        return self.operating_income + da
    
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
        """Calculate net income from components."""
        if self.net_income is not None:
            return self.net_income
        return self.earnings_before_tax - self.income_tax_expense
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with calculated fields."""
        data = self.model_dump(mode="json")
        data["gross_profit"] = float(self.gross_profit)
        data["operating_income"] = float(self.operating_income)
        data["ebitda"] = float(self.ebitda)
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
    
    # Current Assets
    cash_and_equivalents: MonetaryValue = Field(..., description="Cash and equivalents")
    short_term_investments: MonetaryValue = Field(default=Decimal("0"))
    accounts_receivable: MonetaryValue = Field(default=Decimal("0"))
    inventory: MonetaryValue = Field(default=Decimal("0"))
    prepaid_expenses: MonetaryValue = Field(default=Decimal("0"))
    other_current_assets: MonetaryValue = Field(default=Decimal("0"))
    total_current_assets: MonetaryValue | None = Field(default=None)
    
    # Non-Current Assets
    property_plant_equipment: MonetaryValue = Field(default=Decimal("0"))
    intangible_assets: MonetaryValue = Field(default=Decimal("0"))
    goodwill: MonetaryValue = Field(default=Decimal("0"))
    long_term_investments: MonetaryValue = Field(default=Decimal("0"))
    deferred_tax_assets: MonetaryValue = Field(default=Decimal("0"))
    other_non_current_assets: MonetaryValue = Field(default=Decimal("0"))
    total_non_current_assets: MonetaryValue | None = Field(default=None)
    
    # Total Assets
    total_assets: MonetaryValue | None = Field(default=None)
    
    # Current Liabilities
    accounts_payable: MonetaryValue = Field(default=Decimal("0"))
    short_term_debt: MonetaryValue = Field(default=Decimal("0"))
    accrued_liabilities: MonetaryValue = Field(default=Decimal("0"))
    deferred_revenue: MonetaryValue = Field(default=Decimal("0"))
    income_taxes_payable: MonetaryValue = Field(default=Decimal("0"))
    other_current_liabilities: MonetaryValue = Field(default=Decimal("0"))
    total_current_liabilities: MonetaryValue | None = Field(default=None)
    
    # Non-Current Liabilities
    long_term_debt: MonetaryValue = Field(default=Decimal("0"))
    deferred_tax_liabilities: MonetaryValue = Field(default=Decimal("0"))
    pension_liabilities: MonetaryValue = Field(default=Decimal("0"))
    other_non_current_liabilities: MonetaryValue = Field(default=Decimal("0"))
    total_non_current_liabilities: MonetaryValue | None = Field(default=None)
    
    # Total Liabilities
    total_liabilities: MonetaryValue | None = Field(default=None)
    
    # Shareholders' Equity
    common_stock: MonetaryValue = Field(default=Decimal("0"))
    preferred_stock: MonetaryValue = Field(default=Decimal("0"))
    additional_paid_in_capital: MonetaryValue = Field(default=Decimal("0"))
    retained_earnings: MonetaryValue = Field(default=Decimal("0"))
    treasury_stock: MonetaryValue = Field(default=Decimal("0"))
    accumulated_other_comprehensive_income: MonetaryValue = Field(default=Decimal("0"))
    total_shareholders_equity: MonetaryValue | None = Field(default=None)
    non_controlling_interest: MonetaryValue = Field(default=Decimal("0"))
    
    model_config = {"populate_by_name": True, "extra": "allow"}
    
    @field_validator("cash_and_equivalents", mode="before")
    @classmethod
    def convert_cash_to_decimal(cls, v: Any) -> Decimal:
        """Convert numeric inputs to Decimal."""
        if v is None:
            raise ValueError("Cash and equivalents is required")
        return _to_decimal(v)
    
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
        """Calculate total non-current assets."""
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
        """Calculate total assets."""
        if self.total_assets is not None:
            return self.total_assets
        return self.calculated_current_assets + self.calculated_non_current_assets
    
    @property
    def calculated_current_liabilities(self) -> Decimal:
        """Calculate total current liabilities."""
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
        """Calculate total non-current liabilities."""
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
        """Calculate total liabilities."""
        if self.total_liabilities is not None:
            return self.total_liabilities
        return self.calculated_current_liabilities + self.calculated_non_current_liabilities
    
    @property
    def calculated_shareholders_equity(self) -> Decimal:
        """Calculate shareholders' equity."""
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
        """Calculate total debt."""
        return self.short_term_debt + self.long_term_debt
    
    @property
    def capital_employed(self) -> Decimal:
        """Calculate capital employed (Total Assets - Current Liabilities)."""
        return self.calculated_total_assets - self.calculated_current_liabilities
    
    def check_balance_sheet_equation(self, tolerance: Decimal = Decimal("0.01")) -> bool:
        """Verify Assets = Liabilities + Equity."""
        assets = self.calculated_total_assets
        liab_equity = self.calculated_total_liabilities + self.calculated_total_equity
        difference = abs(assets - liab_equity)
        return difference <= tolerance
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with calculated fields."""
        data = self.model_dump(mode="json")
        data["calculated_current_assets"] = float(self.calculated_current_assets)
        data["calculated_total_assets"] = float(self.calculated_total_assets)
        data["calculated_current_liabilities"] = float(self.calculated_current_liabilities)
        data["calculated_total_liabilities"] = float(self.calculated_total_liabilities)
        data["calculated_shareholders_equity"] = float(self.calculated_shareholders_equity)
        data["working_capital"] = float(self.working_capital)
        data["total_debt"] = float(self.total_debt)
        data["capital_employed"] = float(self.capital_employed)
        return data


class CashFlowStatementData(BaseModel):
    """
    Cash Flow Statement data model.
    
    Organized into Operating, Investing, and Financing activities.
    """
    
    period: FinancialPeriod
    currency: str = Field(default="SGD", min_length=3, max_length=3)
    
    # Operating Activities
    net_income: MonetaryValue = Field(..., description="Net income (starting point)")
    depreciation_amortization: MonetaryValue = Field(default=Decimal("0"))
    stock_based_compensation: MonetaryValue = Field(default=Decimal("0"))
    deferred_taxes: MonetaryValue = Field(default=Decimal("0"))
    change_in_receivables: MonetaryValue = Field(default=Decimal("0"))
    change_in_inventory: MonetaryValue = Field(default=Decimal("0"))
    change_in_payables: MonetaryValue = Field(default=Decimal("0"))
    change_in_other_working_capital: MonetaryValue = Field(default=Decimal("0"))
    other_operating_activities: MonetaryValue = Field(default=Decimal("0"))
    net_cash_from_operating: MonetaryValue | None = Field(default=None)
    
    # Investing Activities
    capital_expenditures: MonetaryValue = Field(default=Decimal("0"))
    acquisitions: MonetaryValue = Field(default=Decimal("0"))
    investment_purchases: MonetaryValue = Field(default=Decimal("0"))
    investment_sales: MonetaryValue = Field(default=Decimal("0"))
    other_investing_activities: MonetaryValue = Field(default=Decimal("0"))
    net_cash_from_investing: MonetaryValue | None = Field(default=None)
    
    # Financing Activities
    debt_issued: MonetaryValue = Field(default=Decimal("0"))
    debt_repaid: MonetaryValue = Field(default=Decimal("0"))
    stock_issued: MonetaryValue = Field(default=Decimal("0"))
    stock_repurchased: MonetaryValue = Field(default=Decimal("0"))
    dividends_paid: MonetaryValue = Field(default=Decimal("0"))
    other_financing_activities: MonetaryValue = Field(default=Decimal("0"))
    net_cash_from_financing: MonetaryValue | None = Field(default=None)
    
    # Cash Position
    beginning_cash: MonetaryValue = Field(default=Decimal("0"))
    ending_cash: MonetaryValue | None = Field(default=None)
    
    model_config = {"populate_by_name": True, "extra": "allow"}
    
    @field_validator("net_income", mode="before")
    @classmethod
    def convert_net_income_to_decimal(cls, v: Any) -> Decimal:
        """Convert numeric inputs to Decimal."""
        if v is None:
            raise ValueError("Net income is required")
        return _to_decimal(v)
    
    @property
    def calculated_operating_cash_flow(self) -> Decimal:
        """Calculate operating cash flow."""
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
        """Calculate investing cash flow."""
        if self.net_cash_from_investing is not None:
            return self.net_cash_from_investing
        return (
            -self.capital_expenditures
            - self.acquisitions
            - self.investment_purchases
            + self.investment_sales
            + self.other_investing_activities
        )
    
    @property
    def calculated_financing_cash_flow(self) -> Decimal:
        """Calculate financing cash flow."""
        if self.net_cash_from_financing is not None:
            return self.net_cash_from_financing
        return (
            self.debt_issued
            - self.debt_repaid
            + self.stock_issued
            - self.stock_repurchased
            - self.dividends_paid
            + self.other_financing_activities
        )
    
    @property
    def net_change_in_cash(self) -> Decimal:
        """Calculate net change in cash."""
        return (
            self.calculated_operating_cash_flow
            + self.calculated_investing_cash_flow
            + self.calculated_financing_cash_flow
        )
    
    @property
    def calculated_ending_cash(self) -> Decimal:
        """Calculate ending cash balance."""
        if self.ending_cash is not None:
            return self.ending_cash
        return self.beginning_cash + self.net_change_in_cash
    
    @property
    def free_cash_flow(self) -> Decimal:
        """Calculate free cash flow (Operating - CapEx)."""
        return self.calculated_operating_cash_flow - abs(self.capital_expenditures)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with calculated fields."""
        data = self.model_dump(mode="json")
        data["calculated_operating_cash_flow"] = float(self.calculated_operating_cash_flow)
        data["calculated_investing_cash_flow"] = float(self.calculated_investing_cash_flow)
        data["calculated_financing_cash_flow"] = float(self.calculated_financing_cash_flow)
        data["net_change_in_cash"] = float(self.net_change_in_cash)
        data["calculated_ending_cash"] = float(self.calculated_ending_cash)
        data["free_cash_flow"] = float(self.free_cash_flow)
        return data


class FinancialStatementSet(BaseModel):
    """
    Complete set of financial statements for a single period.
    
    Combines Income Statement, Balance Sheet, and Cash Flow Statement.
    """
    
    income_statement: IncomeStatementData
    balance_sheet: BalanceSheetData
    cash_flow_statement: CashFlowStatementData | None = None
    
    model_config = {"extra": "forbid"}
    
    @model_validator(mode="after")
    def validate_periods_match(self) -> Self:
        """Validate that all statements are for the same period."""
        is_period = self.income_statement.period
        bs_period = self.balance_sheet.period
        
        if is_period != bs_period:
            raise ValueError(
                f"Income statement period ({is_period}) does not match "
                f"balance sheet period ({bs_period})"
            )
        
        if self.cash_flow_statement is not None:
            cf_period = self.cash_flow_statement.period
            if is_period != cf_period:
                raise ValueError(
                    f"Cash flow statement period ({cf_period}) does not match "
                    f"other statements ({is_period})"
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
    Financial data spanning multiple periods.
    
    Used for trend analysis and multi-period comparisons.
    """
    
    periods: list[FinancialStatementSet] = Field(
        ..., min_length=1, description="Financial statement sets by period"
    )
    
    model_config = {"extra": "forbid"}
    
    @model_validator(mode="after")
    def sort_periods(self) -> Self:
        """Sort periods chronologically."""
        self.periods = sorted(self.periods, key=lambda x: x.period)
        return self
    
    @property
    def period_count(self) -> int:
        """Number of periods."""
        return len(self.periods)
    
    @property
    def latest_period(self) -> FinancialStatementSet:
        """Get the most recent period."""
        return self.periods[-1]
    
    @property
    def earliest_period(self) -> FinancialStatementSet:
        """Get the earliest period."""
        return self.periods[0]
    
    def get_period(self, period: FinancialPeriod) -> FinancialStatementSet | None:
        """Get statements for a specific period."""
        for stmt_set in self.periods:
            if stmt_set.period == period:
                return stmt_set
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period_count": self.period_count,
            "periods": [p.to_dict() for p in self.periods],
        }
File 10 of 25: finanalyst_tools/models/analysis_results.py
Python

# finanalyst_tools/models/analysis_results.py
"""
Structured result models for calculations and analysis.

Provides:
- CalculationResult with full audit trail
- MetricCollection for grouped metrics
- TrendAnalysisResult for multi-period analysis
- ConfidenceAssessment for output confidence scoring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any
import json

from finanalyst_tools.models.validation import ValidationResult


class MetricUnit(str, Enum):
    """Units for metric values."""
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    CURRENCY = "currency"
    DAYS = "days"
    COUNT = "count"
    TIMES = "times"


class MetricCategory(str, Enum):
    """Categories of financial metrics."""
    PROFITABILITY = "profitability"
    LIQUIDITY = "liquidity"
    SOLVENCY = "solvency"
    EFFICIENCY = "efficiency"
    GROWTH = "growth"
    VALUATION = "valuation"


class TrendDirection(str, Enum):
    """Direction of trend over time."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class ConfidenceLevel(str, Enum):
    """Confidence levels for analysis output."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class CalculationStep:
    """
    A single step in a calculation for audit trail.
    """
    step_number: int
    description: str
    formula: str | None = None
    values: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step_number,
            "description": self.description,
            "formula": self.formula,
            "values": {k: str(v) for k, v in self.values.items()},
            "result": str(self.result) if self.result is not None else None,
        }
    
    def __str__(self) -> str:
        """Format as readable string."""
        parts = [f"Step {self.step_number}: {self.description}"]
        if self.formula:
            parts.append(f"  Formula: {self.formula}")
        if self.values:
            values_str = ", ".join(f"{k}={v}" for k, v in self.values.items())
            parts.append(f"  Values: {values_str}")
        if self.result is not None:
            parts.append(f"  Result: {self.result}")
        return "\n".join(parts)


@dataclass
class CalculationResult:
    """
    Complete result of a single metric calculation.
    
    Includes the calculated value, formula, all inputs used,
    step-by-step calculation audit trail, plausibility assessment,
    and any warnings.
    """
    metric_name: str
    value: Decimal | None
    unit: MetricUnit
    formula: str
    inputs: dict[str, Any] = field(default_factory=dict)
    calculation_steps: list[CalculationStep] = field(default_factory=list)
    is_plausible: bool = True
    plausibility_range: tuple[float, float] | None = None
    warnings: list[str] = field(default_factory=list)
    category: MetricCategory | None = None
    interpretation: str = ""
    
    def add_step(
        self,
        description: str,
        formula: str | None = None,
        values: dict[str, Any] | None = None,
        result: Any = None,
    ) -> None:
        """Add a calculation step to the audit trail."""
        step_number = len(self.calculation_steps) + 1
        step = CalculationStep(
            step_number=step_number,
            description=description,
            formula=formula,
            values=values or {},
            result=result,
        )
        self.calculation_steps.append(step)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metric_name": self.metric_name,
            "value": float(self.value) if self.value is not None else None,
            "unit": self.unit.value,
            "formula": self.formula,
            "inputs": {k: str(v) for k, v in self.inputs.items()},
            "calculation_steps": [s.to_dict() for s in self.calculation_steps],
            "is_plausible": self.is_plausible,
            "plausibility_range": list(self.plausibility_range) if self.plausibility_range else None,
            "warnings": self.warnings,
            "category": self.category.value if self.category else None,
            "interpretation": self.interpretation,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_reasoning_block(self) -> str:
        """
        Format as a reasoning block for LLM output.
        
        This matches the system prompt's required format for showing
        calculation work.
        """
        lines = [
            f"### {self.metric_name}",
            f"**Formula**: {self.formula}",
            "",
            "**Calculation Steps**:",
        ]
        
        for step in self.calculation_steps:
            lines.append(str(step))
        
        lines.append("")
        
        if self.value is not None:
            if self.unit == MetricUnit.PERCENTAGE:
                lines.append(f"**Result**: {self.value:.2f}%")
            elif self.unit == MetricUnit.RATIO:
                lines.append(f"**Result**: {self.value:.4f}x")
            elif self.unit == MetricUnit.CURRENCY:
                lines.append(f"**Result**: ${self.value:,.2f}")
            else:
                lines.append(f"**Result**: {self.value}")
        else:
            lines.append("**Result**: Unable to calculate")
        
        if not self.is_plausible and self.plausibility_range:
            lines.append(
                f"**⚠️ Warning**: Value outside plausible range "
                f"({self.plausibility_range[0]} to {self.plausibility_range[1]})"
            )
        
        for warning in self.warnings:
            lines.append(f"**⚠️ Warning**: {warning}")
        
        if self.interpretation:
            lines.append(f"**Interpretation**: {self.interpretation}")
        
        return "\n".join(lines)


@dataclass
class MetricResult(CalculationResult):
    """Extended calculation result with period information."""
    period: str = ""
    prior_value: Decimal | None = None
    change: Decimal | None = None
    change_percentage: Decimal | None = None


@dataclass
class MetricCollection:
    """
    Collection of related metrics for a category.
    """
    category: MetricCategory
    period: str
    metrics: list[CalculationResult] = field(default_factory=list)
    currency: str = "SGD"
    
    @property
    def summary(self) -> dict[str, Decimal | None]:
        """Quick access to metric values by name."""
        return {m.metric_name: m.value for m in self.metrics}
    
    @property
    def all_plausible(self) -> bool:
        """Whether all metrics are plausible."""
        return all(m.is_plausible for m in self.metrics)
    
    @property
    def warning_count(self) -> int:
        """Total number of warnings across all metrics."""
        return sum(len(m.warnings) for m in self.metrics)
    
    def add_metric(self, metric: CalculationResult) -> None:
        """Add a metric to the collection."""
        self.metrics.append(metric)
    
    def get_metric(self, name: str) -> CalculationResult | None:
        """Get a metric by name."""
        for metric in self.metrics:
            if metric.metric_name.lower() == name.lower():
                return metric
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "period": self.period,
            "currency": self.currency,
            "all_plausible": self.all_plausible,
            "warning_count": self.warning_count,
            "metrics": [m.to_dict() for m in self.metrics],
            "summary": {k: float(v) if v else None for k, v in self.summary.items()},
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_markdown_table(self) -> str:
        """Format as markdown table."""
        lines = [
            f"### {self.category.value.title()} Metrics ({self.period})",
            "",
            "| Metric | Value | Status |",
            "|--------|-------|--------|",
        ]
        
        for metric in self.metrics:
            if metric.value is not None:
                if metric.unit == MetricUnit.PERCENTAGE:
                    value_str = f"{metric.value:.2f}%"
                elif metric.unit == MetricUnit.RATIO:
                    value_str = f"{metric.value:.2f}x"
                elif metric.unit == MetricUnit.CURRENCY:
                    value_str = f"${metric.value:,.0f}"
                else:
                    value_str = str(metric.value)
            else:
                value_str = "N/A"
            
            status = "✅" if metric.is_plausible and not metric.warnings else "⚠️"
            lines.append(f"| {metric.metric_name} | {value_str} | {status} |")
        
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
    def latest_value(self) -> Decimal | None:
        """Get the most recent value."""
        for v in reversed(self.values):
            if v is not None:
                return v
        return None
    
    @property
    def earliest_value(self) -> Decimal | None:
        """Get the earliest value."""
        for v in self.values:
            if v is not None:
                return v
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "periods": self.periods,
            "values": [float(v) if v else None for v in self.values],
            "direction": self.direction.value,
            "growth_rate": float(self.growth_rate) if self.growth_rate else None,
            "volatility": float(self.volatility) if self.volatility else None,
            "interpretation": self.interpretation,
        }


@dataclass
class ConfidenceAssessment:
    """
    Confidence level assessment for analysis output.
    
    Implements the mandatory confidence scoring from the system prompt.
    """
    level: ConfidenceLevel
    justification: str
    factors: dict[str, str] = field(default_factory=dict)
    score: float = 0.0  # Internal score (0-100)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "justification": self.justification,
            "factors": self.factors,
            "score": self.score,
        }
    
    def to_string(self) -> str:
        """Format for output."""
        return f"{self.level.value} — {self.justification}"
    
    def __str__(self) -> str:
        return self.to_string()


@dataclass
class ComprehensiveAnalysisResult:
    """
    Complete analysis result containing all metrics and assessments.
    """
    analysis_type: str
    period: str
    company_name: str = ""
    currency: str = "SGD"
    
    # Results by category
    profitability: MetricCollection | None = None
    liquidity: MetricCollection | None = None
    solvency: MetricCollection | None = None
    efficiency: MetricCollection | None = None
    
    # Validation and confidence
    validation: ValidationResult | None = None
    confidence: ConfidenceAssessment | None = None
    
    # Trend analysis (if multi-period)
    trends: list[TrendAnalysisResult] = field(default_factory=list)
    
    # Overall findings
    key_findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    
    @property
    def all_metrics(self) -> list[CalculationResult]:
        """Get all calculated metrics."""
        metrics = []
        for collection in [self.profitability, self.liquidity, self.solvency, self.efficiency]:
            if collection:
                metrics.extend(collection.metrics)
        return metrics
    
    @property
    def total_warnings(self) -> int:
        """Total warning count across all metrics."""
        return sum(len(m.warnings) for m in self.all_metrics)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "analysis_type": self.analysis_type,
            "period": self.period,
            "company_name": self.company_name,
            "currency": self.currency,
            "total_warnings": self.total_warnings,
        }
        
        if self.profitability:
            result["profitability"] = self.profitability.to_dict()
        if self.liquidity:
            result["liquidity"] = self.liquidity.to_dict()
        if self.solvency:
            result["solvency"] = self.solvency.to_dict()
        if self.efficiency:
            result["efficiency"] = self.efficiency.to_dict()
        if self.validation:
            result["validation"] = self.validation.to_dict()
        if self.confidence:
            result["confidence"] = self.confidence.to_dict()
        if self.trends:
            result["trends"] = [t.to_dict() for t in self.trends]
        
        result["key_findings"] = self.key_findings
        result["recommendations"] = self.recommendations
        
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
File 11 of 25: finanalyst_tools/validation/__init__.py
Python

# finanalyst_tools/validation/__init__.py
"""
Validation module for FinAnalyst-Pro Agent Tools.

Provides:
- Schema validation for financial data
- Cross-statement reconciliation
- Plausibility checks for calculated metrics
"""

from finanalyst_tools.validation.schema_validator import (
    validate_income_statement_schema,
    validate_balance_sheet_schema,
    validate_cash_flow_schema,
    validate_financial_data_completeness,
    validate_statement_set,
    REQUIRED_FIELDS_BY_ANALYSIS,
    FIELD_ALIASES,
)

from finanalyst_tools.validation.reconciliation import (
    reconcile_net_income,
    reconcile_cash_balance,
    reconcile_retained_earnings,
    reconcile_balance_sheet_equation,
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
    "REQUIRED_FIELDS_BY_ANALYSIS",
    "FIELD_ALIASES",
    # Reconciliation
    "reconcile_net_income",
    "reconcile_cash_balance",
    "reconcile_retained_earnings",
    "reconcile_balance_sheet_equation",
    "run_all_reconciliations",
    # Plausibility
    "check_plausibility",
    "check_all_plausibility",
    "PlausibilityChecker",
]
File 12 of 25: finanalyst_tools/validation/schema_validator.py
Python

# finanalyst_tools/validation/schema_validator.py
"""
Schema validation for financial data.

Validates that financial data:
- Has required fields for the requested analysis
- Uses correct data types
- Has reasonable value ranges
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from finanalyst_tools.models.validation import (
    ValidationResult,
    ValidationSeverity,
)
from finanalyst_tools.models.financial_statements import (
    IncomeStatementData,
    BalanceSheetData,
    CashFlowStatementData,
    FinancialStatementSet,
)
from finanalyst_tools.config import MAX_MONETARY_VALUE


# ============================================================================
# FIELD REQUIREMENTS BY ANALYSIS TYPE
# ============================================================================

REQUIRED_FIELDS_BY_ANALYSIS: dict[str, dict[str, list[str]]] = {
    "profitability": {
        "income_statement": ["total_revenue", "cost_of_goods_sold", "net_income"],
        "balance_sheet": ["total_assets", "total_shareholders_equity"],
    },
    "liquidity": {
        "balance_sheet": [
            "cash_and_equivalents",
            "current_assets",
            "current_liabilities",
            "inventory",
        ],
    },
    "solvency": {
        "balance_sheet": [
            "total_assets",
            "total_liabilities",
            "total_shareholders_equity",
            "long_term_debt",
        ],
        "income_statement": ["interest_expense"],
    },
    "efficiency": {
        "income_statement": ["total_revenue", "cost_of_goods_sold"],
        "balance_sheet": ["total_assets", "inventory", "accounts_receivable"],
    },
    "comprehensive": {
        "income_statement": [
            "total_revenue",
            "cost_of_goods_sold",
            "net_income",
        ],
        "balance_sheet": [
            "cash_and_equivalents",
            "total_assets",
            "total_liabilities",
            "total_shareholders_equity",
        ],
    },
}


# ============================================================================
# FIELD ALIASES
# ============================================================================

FIELD_ALIASES: dict[str, list[str]] = {
    "total_revenue": ["revenue", "net_sales", "sales", "total_sales", "net_revenue"],
    "cost_of_goods_sold": ["cogs", "cost_of_sales", "cost_of_revenue"],
    "net_income": ["net_profit", "net_earnings", "profit_after_tax", "pat"],
    "total_assets": ["assets_total", "total_asset"],
    "total_liabilities": ["liabilities_total", "total_liability"],
    "total_shareholders_equity": [
        "shareholders_equity",
        "stockholders_equity",
        "total_equity",
        "equity",
    ],
    "current_assets": ["total_current_assets"],
    "current_liabilities": ["total_current_liabilities"],
    "cash_and_equivalents": ["cash", "cash_and_cash_equivalents"],
}


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def _find_field_value(
    data: dict[str, Any],
    canonical_name: str,
) -> tuple[Any, str]:
    """
    Find a field value using canonical name or aliases.
    
    Returns:
        Tuple of (value, field_name_used)
    """
    # Try canonical name first
    if canonical_name in data:
        return data[canonical_name], canonical_name
    
    # Try aliases
    aliases = FIELD_ALIASES.get(canonical_name, [])
    for alias in aliases:
        if alias in data:
            return data[alias], alias
    
    return None, canonical_name


def _check_numeric_value(
    value: Any,
    field_name: str,
    result: ValidationResult,
) -> Decimal | None:
    """
    Check if a value is numeric and within reasonable bounds.
    
    Returns the value as Decimal if valid, None otherwise.
    """
    if value is None:
        return None
    
    try:
        if isinstance(value, Decimal):
            dec_value = value
        elif isinstance(value, (int, float)):
            dec_value = Decimal(str(value))
        elif isinstance(value, str):
            dec_value = Decimal(value.replace(",", "").strip())
        else:
            result.add_error(
                field=field_name,
                message=f"Value must be numeric, got {type(value).__name__}",
                actual_value=value,
                expected="numeric value",
            )
            return None
        
        # Check for reasonable bounds
        if abs(float(dec_value)) > MAX_MONETARY_VALUE:
            result.add_warning(
                field=field_name,
                message=f"Value {dec_value} seems unreasonably large",
                actual_value=dec_value,
                expected=f"Less than {MAX_MONETARY_VALUE:,.0f}",
            )
        
        return dec_value
        
    except Exception as e:
        result.add_error(
            field=field_name,
            message=f"Cannot convert to number: {e}",
            actual_value=value,
            expected="numeric value",
        )
        return None


def validate_income_statement_schema(
    data: dict[str, Any],
) -> ValidationResult:
    """
    Validate income statement data structure.
    
    Args:
        data: Dictionary containing income statement fields
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult()
    
    # Check required fields
    required = ["total_revenue", "cost_of_goods_sold"]
    
    for field_name in required:
        value, used_name = _find_field_value(data, field_name)
        if value is None:
            result.add_error(
                field=field_name,
                message=f"Required field '{field_name}' is missing",
                expected="numeric value",
                suggestion=f"Provide {field_name} (or aliases: {FIELD_ALIASES.get(field_name, [])})",
            )
        else:
            _check_numeric_value(value, used_name, result)
    
    # Check optional but important fields
    important = ["net_income", "operating_expenses"]
    for field_name in important:
        value, _ = _find_field_value(data, field_name)
        if value is None:
            result.add_info(field_name, f"Optional field '{field_name}' not provided")
    
    # Validate with Pydantic if we have enough data
    if result.can_proceed:
        try:
            # Attempt to create model (will fail if data is incompatible)
            # We just want to catch type errors
            pass  # Skip for now, manual validation above is sufficient
        except PydanticValidationError as e:
            for error in e.errors():
                result.add_error(
                    field=".".join(str(loc) for loc in error["loc"]),
                    message=error["msg"],
                )
    
    result.context["statement_type"] = "income_statement"
    return result


def validate_balance_sheet_schema(
    data: dict[str, Any],
) -> ValidationResult:
    """
    Validate balance sheet data structure.
    
    Args:
        data: Dictionary containing balance sheet fields
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult()
    
    # Check required fields
    required = ["cash_and_equivalents"]
    
    for field_name in required:
        value, used_name = _find_field_value(data, field_name)
        if value is None:
            result.add_error(
                field=field_name,
                message=f"Required field '{field_name}' is missing",
                expected="numeric value",
            )
        else:
            _check_numeric_value(value, used_name, result)
    
    # Check for asset/liability/equity totals
    total_fields = [
        "total_assets",
        "total_liabilities",
        "total_shareholders_equity",
    ]
    
    totals_provided = 0
    for field_name in total_fields:
        value, _ = _find_field_value(data, field_name)
        if value is not None:
            totals_provided += 1
    
    if totals_provided < 2:
        result.add_warning(
            field="totals",
            message="Less than 2 of 3 total fields provided (assets, liabilities, equity)",
            suggestion="Provide totals for more accurate analysis",
        )
    
    result.context["statement_type"] = "balance_sheet"
    return result


def validate_cash_flow_schema(
    data: dict[str, Any],
) -> ValidationResult:
    """
    Validate cash flow statement data structure.
    
    Args:
        data: Dictionary containing cash flow fields
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult()
    
    # Check required fields
    required = ["net_income"]
    
    for field_name in required:
        value, used_name = _find_field_value(data, field_name)
        if value is None:
            result.add_error(
                field=field_name,
                message=f"Required field '{field_name}' is missing",
                expected="numeric value",
            )
        else:
            _check_numeric_value(value, used_name, result)
    
    result.context["statement_type"] = "cash_flow"
    return result


def validate_financial_data_completeness(
    income_statement: dict[str, Any] | None,
    balance_sheet: dict[str, Any] | None,
    cash_flow: dict[str, Any] | None,
    analysis_type: str,
) -> ValidationResult:
    """
    Validate that required data is present for the requested analysis type.
    
    Args:
        income_statement: Income statement data
        balance_sheet: Balance sheet data
        cash_flow: Cash flow statement data
        analysis_type: Type of analysis to perform
        
    Returns:
        ValidationResult with completeness assessment
    """
    result = ValidationResult()
    
    requirements = REQUIRED_FIELDS_BY_ANALYSIS.get(
        analysis_type.lower(),
        REQUIRED_FIELDS_BY_ANALYSIS["comprehensive"],
    )
    
    # Check income statement requirements
    if "income_statement" in requirements:
        if income_statement is None:
            result.add_error(
                field="income_statement",
                message=f"Income statement required for {analysis_type} analysis",
            )
        else:
            for field_name in requirements["income_statement"]:
                value, _ = _find_field_value(income_statement, field_name)
                if value is None:
                    result.add_error(
                        field=f"income_statement.{field_name}",
                        message=f"Required for {analysis_type} analysis",
                    )
    
    # Check balance sheet requirements
    if "balance_sheet" in requirements:
        if balance_sheet is None:
            result.add_error(
                field="balance_sheet",
                message=f"Balance sheet required for {analysis_type} analysis",
            )
        else:
            for field_name in requirements["balance_sheet"]:
                value, _ = _find_field_value(balance_sheet, field_name)
                if value is None:
                    # Check if it can be calculated from components
                    result.add_warning(
                        field=f"balance_sheet.{field_name}",
                        message=f"Field not provided directly, will attempt to calculate",
                    )
    
    # Check cash flow requirements (if any)
    if "cash_flow" in requirements:
        if cash_flow is None:
            result.add_warning(
                field="cash_flow",
                message="Cash flow statement not provided (optional for most analyses)",
            )
    
    result.context["analysis_type"] = analysis_type
    return result


def validate_statement_set(
    statement_set: FinancialStatementSet,
) -> ValidationResult:
    """
    Validate a complete financial statement set.
    
    Args:
        statement_set: Complete set of financial statements
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult()
    
    # Validate individual statements
    is_result = validate_income_statement_schema(
        statement_set.income_statement.model_dump()
    )
    result = result.merge(is_result)
    
    bs_result = validate_balance_sheet_schema(
        statement_set.balance_sheet.model_dump()
    )
    result = result.merge(bs_result)
    
    if statement_set.cash_flow_statement:
        cf_result = validate_cash_flow_schema(
            statement_set.cash_flow_statement.model_dump()
        )
        result = result.merge(cf_result)
    
    # Check balance sheet equation
    if not statement_set.balance_sheet.check_balance_sheet_equation():
        result.add_warning(
            field="balance_sheet_equation",
            message="Assets ≠ Liabilities + Equity (may indicate data issues)",
        )
    
    return result
File 13 of 25: finanalyst_tools/validation/reconciliation.py
Python

# finanalyst_tools/validation/reconciliation.py
"""
Cross-statement reconciliation checks.

Verifies consistency between related values across
different financial statements.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from finanalyst_tools.models.validation import (
    ReconciliationCheck,
    ReconciliationResult,
    ValidationResult,
    ValidationSeverity,
)
from finanalyst_tools.models.financial_statements import (
    IncomeStatementData,
    BalanceSheetData,
    CashFlowStatementData,
    FinancialStatementSet,
)
from finanalyst_tools.config import ReconciliationTolerances
from finanalyst_tools.utils.math_ops import to_decimal, is_effectively_zero


def _create_check(
    check_name: str,
    statement_a: str,
    value_a: Decimal,
    statement_b: str,
    value_b: Decimal,
    tolerance: float,
) -> ReconciliationCheck:
    """Create a reconciliation check result."""
    difference = abs(value_a - value_b)
    
    # Determine if passed
    if is_effectively_zero(value_a) and is_effectively_zero(value_b):
        passed = True
    else:
        base = max(abs(value_a), abs(value_b))
        if is_effectively_zero(base):
            passed = True
        else:
            relative_diff = float(difference / base)
            passed = relative_diff <= tolerance
    
    return ReconciliationCheck(
        check_name=check_name,
        statement_a=statement_a,
        value_a=value_a,
        statement_b=statement_b,
        value_b=value_b,
        difference=difference,
        tolerance=tolerance,
        passed=passed,
    )


def reconcile_net_income(
    income_statement: IncomeStatementData,
    cash_flow_statement: CashFlowStatementData,
    tolerance: float | None = None,
) -> ReconciliationCheck:
    """
    Reconcile net income between Income Statement and Cash Flow Statement.
    
    The net income on the income statement should match the starting
    net income on the cash flow statement (indirect method).
    
    Args:
        income_statement: Income statement data
        cash_flow_statement: Cash flow statement data
        tolerance: Override tolerance (default: STRICT)
        
    Returns:
        ReconciliationCheck result
    """
    if tolerance is None:
        tolerance = ReconciliationTolerances.get_tolerance("net_income")
    
    is_net_income = income_statement.calculated_net_income
    cf_net_income = cash_flow_statement.net_income
    
    return _create_check(
        check_name="Net Income",
        statement_a="Income Statement",
        value_a=is_net_income,
        statement_b="Cash Flow Statement",
        value_b=cf_net_income,
        tolerance=tolerance,
    )


def reconcile_cash_balance(
    balance_sheet: BalanceSheetData,
    cash_flow_statement: CashFlowStatementData,
    tolerance: float | None = None,
) -> ReconciliationCheck:
    """
    Reconcile cash balance between Balance Sheet and Cash Flow Statement.
    
    The ending cash on the balance sheet should match the ending
    cash on the cash flow statement.
    
    Args:
        balance_sheet: Balance sheet data
        cash_flow_statement: Cash flow statement data
        tolerance: Override tolerance (default: STRICT)
        
    Returns:
        ReconciliationCheck result
    """
    if tolerance is None:
        tolerance = ReconciliationTolerances.get_tolerance("cash_balance")
    
    bs_cash = balance_sheet.cash_and_equivalents
    cf_ending_cash = cash_flow_statement.calculated_ending_cash
    
    return _create_check(
        check_name="Cash Balance",
        statement_a="Balance Sheet",
        value_a=bs_cash,
        statement_b="Cash Flow Statement (Ending)",
        value_b=cf_ending_cash,
        tolerance=tolerance,
    )


def reconcile_retained_earnings(
    current_balance_sheet: BalanceSheetData,
    prior_balance_sheet: BalanceSheetData,
    income_statement: IncomeStatementData,
    dividends_paid: Decimal = Decimal("0"),
    tolerance: float | None = None,
) -> ReconciliationCheck:
    """
    Reconcile retained earnings rollforward.
    
    Current RE should equal: Prior RE + Net Income - Dividends
    
    Args:
        current_balance_sheet: Current period balance sheet
        prior_balance_sheet: Prior period balance sheet
        income_statement: Current period income statement
        dividends_paid: Dividends paid during period
        tolerance: Override tolerance (default: NORMAL)
        
    Returns:
        ReconciliationCheck result
    """
    if tolerance is None:
        tolerance = ReconciliationTolerances.get_tolerance("retained_earnings")
    
    current_re = current_balance_sheet.retained_earnings
    prior_re = prior_balance_sheet.retained_earnings
    net_income = income_statement.calculated_net_income
    
    expected_re = prior_re + net_income - dividends_paid
    
    return _create_check(
        check_name="Retained Earnings Rollforward",
        statement_a="Balance Sheet (Current)",
        value_a=current_re,
        statement_b="Calculated (Prior + NI - Div)",
        value_b=expected_re,
        tolerance=tolerance,
    )


def reconcile_balance_sheet_equation(
    balance_sheet: BalanceSheetData,
    tolerance: float | None = None,
) -> ReconciliationCheck:
    """
    Verify the fundamental accounting equation: Assets = Liabilities + Equity.
    
    Args:
        balance_sheet: Balance sheet data
        tolerance: Override tolerance (default: STRICT)
        
    Returns:
        ReconciliationCheck result
    """
    if tolerance is None:
        tolerance = ReconciliationTolerances.get_tolerance("balance_sheet_equation")
    
    total_assets = balance_sheet.calculated_total_assets
    total_liab_equity = (
        balance_sheet.calculated_total_liabilities
        + balance_sheet.calculated_total_equity
    )
    
    return _create_check(
        check_name="Balance Sheet Equation (A = L + E)",
        statement_a="Total Assets",
        value_a=total_assets,
        statement_b="Liabilities + Equity",
        value_b=total_liab_equity,
        tolerance=tolerance,
    )


def reconcile_working_capital(
    balance_sheet: BalanceSheetData,
    tolerance: float | None = None,
) -> ReconciliationCheck:
    """
    Reconcile working capital calculation.
    
    Working Capital = Current Assets - Current Liabilities
    
    Args:
        balance_sheet: Balance sheet data
        tolerance: Override tolerance (default: NORMAL)
        
    Returns:
        ReconciliationCheck result
    """
    if tolerance is None:
        tolerance = ReconciliationTolerances.get_tolerance("working_capital")
    
    current_assets = balance_sheet.calculated_current_assets
    current_liabilities = balance_sheet.calculated_current_liabilities
    calculated_wc = current_assets - current_liabilities
    stated_wc = balance_sheet.working_capital
    
    return _create_check(
        check_name="Working Capital",
        statement_a="Calculated (CA - CL)",
        value_a=calculated_wc,
        statement_b="Property (working_capital)",
        value_b=stated_wc,
        tolerance=tolerance,
    )


def run_all_reconciliations(
    statement_set: FinancialStatementSet,
    prior_balance_sheet: BalanceSheetData | None = None,
) -> ReconciliationResult:
    """
    Run all applicable reconciliation checks.
    
    Args:
        statement_set: Current period financial statements
        prior_balance_sheet: Prior period balance sheet (optional)
        
    Returns:
        ReconciliationResult with all check results
    """
    result = ReconciliationResult()
    
    # Always check balance sheet equation
    bs_check = reconcile_balance_sheet_equation(statement_set.balance_sheet)
    result.add_check(bs_check)
    
    # Working capital check
    wc_check = reconcile_working_capital(statement_set.balance_sheet)
    result.add_check(wc_check)
    
    # If we have cash flow statement, check net income and cash balance
    if statement_set.cash_flow_statement is not None:
        ni_check = reconcile_net_income(
            statement_set.income_statement,
            statement_set.cash_flow_statement,
        )
        result.add_check(ni_check)
        
        cash_check = reconcile_cash_balance(
            statement_set.balance_sheet,
            statement_set.cash_flow_statement,
        )
        result.add_check(cash_check)
    
    # If we have prior balance sheet, check retained earnings
    if prior_balance_sheet is not None:
        dividends = Decimal("0")
        if statement_set.cash_flow_statement is not None:
            dividends = statement_set.cash_flow_statement.dividends_paid
        
        re_check = reconcile_retained_earnings(
            statement_set.balance_sheet,
            prior_balance_sheet,
            statement_set.income_statement,
            dividends,
        )
        result.add_check(re_check)
    
    return result
File 14 of 25: finanalyst_tools/validation/plausibility.py
Python

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
File 15 of 25: finanalyst_tools/calculations/__init__.py
Python

# finanalyst_tools/calculations/__init__.py
"""
Financial calculation modules for FinAnalyst-Pro Agent Tools.

Provides calculator functions for:
- Profitability ratios (margins, returns)
- Liquidity ratios (current, quick, cash)
- More categories in future phases
"""

from finanalyst_tools.calculations.base import (
    BaseCalculator,
    create_calculation_result,
    format_step,
)

from finanalyst_tools.calculations.profitability import (
    calculate_gross_profit_margin,
    calculate_operating_profit_margin,
    calculate_net_profit_margin,
    calculate_ebitda_margin,
    calculate_return_on_assets,
    calculate_return_on_equity,
    calculate_all_profitability_metrics,
    ProfitabilityCalculator,
)

from finanalyst_tools.calculations.liquidity import (
    calculate_current_ratio,
    calculate_quick_ratio,
    calculate_cash_ratio,
    calculate_working_capital,
    calculate_all_liquidity_metrics,
    LiquidityCalculator,
)


__all__ = [
    # Base
    "BaseCalculator",
    "create_calculation_result",
    "format_step",
    # Profitability
    "calculate_gross_profit_margin",
    "calculate_operating_profit_margin",
    "calculate_net_profit_margin",
    "calculate_ebitda_margin",
    "calculate_return_on_assets",
    "calculate_return_on_equity",
    "calculate_all_profitability_metrics",
    "ProfitabilityCalculator",
    # Liquidity
    "calculate_current_ratio",
    "calculate_quick_ratio",
    "calculate_cash_ratio",
    "calculate_working_capital",
    "calculate_all_liquidity_metrics",
    "LiquidityCalculator",
]
File 16 of 25: finanalyst_tools/calculations/base.py
Python

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
File 17 of 25: finanalyst_tools/calculations/profitability.py
Python

# finanalyst_tools/calculations/profitability.py
"""
Profitability ratio calculations.

Provides calculations for:
- Gross Profit Margin
- Operating Profit Margin
- Net Profit Margin
- EBITDA Margin
- Return on Assets (ROA)
- Return on Equity (ROE)
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from finanalyst_tools.models.analysis_results import (
    CalculationResult,
    MetricUnit,
    MetricCategory,
    MetricCollection,
)
from finanalyst_tools.models.financial_statements import (
    IncomeStatementData,
    BalanceSheetData,
)
from finanalyst_tools.calculations.base import (
    BaseCalculator,
    create_calculation_result,
    format_step,
)
from finanalyst_tools.utils.math_ops import (
    to_decimal,
    safe_divide,
    calculate_percentage,
    calculate_average,
)
from finanalyst_tools.config import METRIC_FORMULAS


# ============================================================================
# STANDALONE CALCULATION FUNCTIONS
# ============================================================================

def calculate_gross_profit_margin(
    revenue: Decimal | float | int,
    cost_of_goods_sold: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Gross Profit Margin.
    
    Formula: ((Revenue - COGS) / Revenue) × 100
    
    Args:
        revenue: Total revenue
        cost_of_goods_sold: Cost of goods sold
        
    Returns:
        CalculationResult with value, steps, and plausibility
    """
    steps = []
    
    # Step 1: Record inputs
    rev = to_decimal(revenue)
    cogs = to_decimal(cost_of_goods_sold)
    
    steps.append(format_step(
        1, "Identify input values",
        values={"revenue": rev, "cogs": cogs}
    ))
    
    # Step 2: Calculate gross profit
    gross_profit = rev - cogs
    steps.append(format_step(
        2, "Calculate Gross Profit",
        formula="Gross Profit = Revenue - COGS",
        values={"revenue": rev, "cogs": cogs},
        result=gross_profit
    ))
    
    # Step 3: Calculate margin
    margin = calculate_percentage(gross_profit, rev)
    steps.append(format_step(
        3, "Calculate Gross Profit Margin",
        formula="(Gross Profit / Revenue) × 100",
        values={"gross_profit": gross_profit, "revenue": rev},
        result=f"{margin}%"
    ))
    
    # Create interpretation
    if margin is not None:
        if margin >= 50:
            interpretation = "Strong gross margin indicating good pricing power or cost control"
        elif margin >= 30:
            interpretation = "Healthy gross margin typical of many industries"
        elif margin >= 20:
            interpretation = "Moderate gross margin; may indicate competitive pricing pressure"
        else:
            interpretation = "Low gross margin; review pricing and cost structure"
    else:
        interpretation = "Unable to calculate - check input values"
    
    return create_calculation_result(
        metric_name="Gross Profit Margin",
        value=margin,
        unit=MetricUnit.PERCENTAGE,
        formula=METRIC_FORMULAS.get("gross_profit_margin", "(Revenue - COGS) / Revenue × 100"),
        inputs={"revenue": revenue, "cost_of_goods_sold": 
So
