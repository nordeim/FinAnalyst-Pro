# finanalyst_tools/validation/schema_validator.py
"""
Schema validation for financial data.
Ensures data completeness and structural correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import ValidationError

from ..models.financial_statements import (
    IncomeStatementData,
    BalanceSheetData,
    CashFlowStatementData,
    FinancialStatementSet,
)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Must be fixed before proceeding
    WARNING = "warning"  # Can proceed but results may be affected
    INFO = "info"        # Informational note


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    field: str
    message: str
    severity: ValidationSeverity
    actual_value: Any = None
    expected: str | None = None


@dataclass
class ValidationResult:
    """Complete validation result."""
    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    info: list[ValidationIssue] = field(default_factory=list)
    
    @property
    def error_count(self) -> int:
        return len(self.issues)
    
    @property
    def warning_count(self) -> int:
        return len(self.warnings)
    
    @property
    def can_proceed(self) -> bool:
        """Check if analysis can proceed (no errors)."""
        return self.error_count == 0
    
    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        if issue.severity == ValidationSeverity.ERROR:
            self.issues.append(issue)
            self.is_valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warnings.append(issue)
        else:
            self.info.append(issue)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "can_proceed": self.can_proceed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "errors": [
                {"field": i.field, "message": i.message, "value": str(i.actual_value)}
                for i in self.issues
            ],
            "warnings": [
                {"field": i.field, "message": i.message, "value": str(i.actual_value)}
                for i in self.warnings
            ],
            "info": [
                {"field": i.field, "message": i.message}
                for i in self.info
            ]
        }


def validate_income_statement_schema(data: dict[str, Any]) -> ValidationResult:
    """
    Validate income statement data against schema.
    
    Args:
        data: Raw income statement data dictionary
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult(is_valid=True)
    
    # Required fields for basic analysis
    required_fields = {
        "total_revenue": ["total_net_sales", "revenue", "sales"],
        "cost_of_goods_sold": ["cost_of_sales", "cogs", "cost_of_revenue"],
    }
    
    # Check required fields
    for canonical_name, aliases in required_fields.items():
        found = False
        for alias in [canonical_name] + aliases:
            if alias in data and data[alias] is not None:
                found = True
                break
        
        if not found:
            result.add_issue(ValidationIssue(
                field=canonical_name,
                message=f"Required field '{canonical_name}' is missing",
                severity=ValidationSeverity.ERROR,
                expected=f"One of: {[canonical_name] + aliases}"
            ))
    
    # Try to parse with Pydantic for detailed validation
    try:
        IncomeStatementData.model_validate(data)
    except ValidationError as e:
        for error in e.errors():
            field_name = ".".join(str(loc) for loc in error["loc"])
            result.add_issue(ValidationIssue(
                field=field_name,
                message=error["msg"],
                severity=ValidationSeverity.ERROR,
                actual_value=error.get("input")
            ))
    
    return result


def validate_balance_sheet_schema(data: dict[str, Any]) -> ValidationResult:
    """
    Validate balance sheet data against schema.
    
    Args:
        data: Raw balance sheet data dictionary
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult(is_valid=True)
    
    # Required fields for basic analysis
    required_fields = ["cash", "total_current_assets", "total_current_liabilities"]
    
    for field_name in required_fields:
        if field_name not in data or data[field_name] is None:
            # Check for common aliases
            aliases = {
                "cash": ["cash_and_equivalents", "cash_and_cash_equivalents"],
                "total_current_assets": ["current_assets"],
                "total_current_liabilities": ["current_liabilities"],
            }
            
            alias_found = False
            for alias in aliases.get(field_name, []):
                if alias in data and data[alias] is not None:
                    alias_found = True
                    break
            
            if not alias_found:
                result.add_issue(ValidationIssue(
                    field=field_name,
                    message=f"Required field '{field_name}' is missing",
                    severity=ValidationSeverity.ERROR
                ))
    
    # Check for total assets (critical for many ratios)
    if "total_assets" not in data or data.get("total_assets") is None:
        # Check if we can derive it
        current = data.get("total_current_assets") or data.get("current_assets")
        non_current = data.get("total_non_current_assets")
        
        if current is None:
            result.add_issue(ValidationIssue(
                field="total_assets",
                message="Total assets not provided and cannot be derived",
                severity=ValidationSeverity.WARNING,
                expected="Provide total_assets or component assets"
            ))
    
    return result


def validate_cash_flow_schema(data: dict[str, Any]) -> ValidationResult:
    """
    Validate cash flow statement data against schema.
    
    Args:
        data: Raw cash flow statement data dictionary
        
    Returns:
        ValidationResult with any issues found
    """
    result = ValidationResult(is_valid=True)
    
    # Net income is required
    if "net_income" not in data or data["net_income"] is None:
        result.add_issue(ValidationIssue(
            field="net_income",
            message="Net income is required for cash flow analysis",
            severity=ValidationSeverity.ERROR
        ))
    
    # Check for major cash flow categories
    cf_categories = [
        ("net_cash_from_operating", "operating cash flow"),
        ("net_cash_from_investing", "investing cash flow"),
        ("net_cash_from_financing", "financing cash flow"),
    ]
    
    for field_name, description in cf_categories:
        if field_name not in data or data.get(field_name) is None:
            result.add_issue(ValidationIssue(
                field=field_name,
                message=f"{description.title()} not provided; will attempt to derive from components",
                severity=ValidationSeverity.INFO
            ))
    
    return result


def validate_financial_data_completeness(
    income_statement: dict[str, Any] | None,
    balance_sheet: dict[str, Any] | None,
    cash_flow: dict[str, Any] | None,
    analysis_type: str = "comprehensive"
) -> ValidationResult:
    """
    Validate completeness of financial data for requested analysis type.
    
    Args:
        income_statement: Income statement data
        balance_sheet: Balance sheet data
        cash_flow: Cash flow statement data
        analysis_type: Type of analysis requested
        
    Returns:
        ValidationResult indicating data readiness
    """
    result = ValidationResult(is_valid=True)
    
    # Define required statements per analysis type
    requirements = {
        "profitability": {"income_statement": True, "balance_sheet": False, "cash_flow": False},
        "liquidity": {"income_statement": False, "balance_sheet": True, "cash_flow": False},
        "solvency": {"income_statement": False, "balance_sheet": True, "cash_flow": False},
        "efficiency": {"income_statement": True, "balance_sheet": True, "cash_flow": False},
        "cash_flow": {"income_statement": False, "balance_sheet": False, "cash_flow": True},
        "comprehensive": {"income_statement": True, "balance_sheet": True, "cash_flow": True},
    }
    
    reqs = requirements.get(analysis_type, requirements["comprehensive"])
    
    if reqs["income_statement"] and not income_statement:
        result.add_issue(ValidationIssue(
            field="income_statement",
            message=f"Income statement required for {analysis_type} analysis",
            severity=ValidationSeverity.ERROR
        ))
    
    if reqs["balance_sheet"] and not balance_sheet:
        result.add_issue(ValidationIssue(
            field="balance_sheet",
            message=f"Balance sheet required for {analysis_type} analysis",
            severity=ValidationSeverity.ERROR
        ))
    
    if reqs["cash_flow"] and not cash_flow:
        result.add_issue(ValidationIssue(
            field="cash_flow_statement",
            message=f"Cash flow statement required for {analysis_type} analysis",
            severity=ValidationSeverity.WARNING  # Often can proceed without
        ))
    
    # Validate individual statements if provided
    if income_statement:
        is_result = validate_income_statement_schema(income_statement)
        for issue in is_result.issues + is_result.warnings + is_result.info:
            issue.field = f"income_statement.{issue.field}"
            result.add_issue(issue)
    
    if balance_sheet:
        bs_result = validate_balance_sheet_schema(balance_sheet)
        for issue in bs_result.issues + bs_result.warnings + bs_result.info:
            issue.field = f"balance_sheet.{issue.field}"
            result.add_issue(issue)
    
    if cash_flow:
        cf_result = validate_cash_flow_schema(cash_flow)
        for issue in cf_result.issues + cf_result.warnings + cf_result.info:
            issue.field = f"cash_flow.{issue.field}"
            result.add_issue(issue)
    
    return result
