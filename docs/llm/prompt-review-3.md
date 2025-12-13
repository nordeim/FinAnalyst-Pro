# finanalyst_tools/validation/utils.py
```py
# finanalyst_tools/validation/utils.py
"""
Utility functions for validation system integration.

Provides:
- Exception to ValidationResult conversion
- ValidationResult to reasoning block formatting
- Cross-system type mapping utilities
- Enhanced error handling and recovery
"""

from __future__ import annotations

from typing import Any, cast
from decimal import Decimal

from finanalyst_tools.models.validation import (
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
)
from finanalyst_tools.exceptions import (
    FinAnalystError,
    CalculationError,
    ValidationError,
    DataError,
    ToolError,
    DivisionByZeroError,
    InvalidInputError,
    SchemaValidationError,
    DataCompletenessError,
    ReconciliationError,
    PlausibilityError,
    DataParsingError,
    MissingDataError,
    ToolNotFoundError,
    ToolExecutionError,
    ToolParameterError,
)


def exception_to_validation_result(
    exc: Exception,
    field: str = "unknown",
    context: str = "general operation"
) -> ValidationResult:
    """
    Convert any exception to a ValidationResult object.
    
    This is the central integration point between the exception hierarchy
    and the validation system.
    
    Args:
        exc: Exception to convert
        field: Field name for validation issue
        context: Context description for error message
        
    Returns:
        ValidationResult with the error
    """
    result = ValidationResult()
    
    # Handle different exception types with appropriate severity mapping
    severity_map = {
        DivisionByZeroError: ValidationSeverity.ERROR,
        InvalidInputError: ValidationSeverity.ERROR,
        SchemaValidationError: ValidationSeverity.ERROR,
        DataCompletenessError: ValidationSeverity.ERROR,
        ReconciliationError: ValidationSeverity.ERROR,
        ToolNotFoundError: ValidationSeverity.ERROR,
        ToolExecutionError: ValidationSeverity.ERROR,
        ToolParameterError: ValidationSeverity.ERROR,
        PlausibilityError: ValidationSeverity.WARNING,
        DataParsingError: ValidationSeverity.ERROR,
        MissingDataError: ValidationSeverity.WARNING,
        ValueError: ValidationSeverity.ERROR,
        TypeError: ValidationSeverity.ERROR,
    }
    
    severity = severity_map.get(type(exc), ValidationSeverity.ERROR)
    
    # Get exception details
    details = {}
    if hasattr(exc, 'details'):
        details = getattr(exc, 'details', {})
    elif hasattr(exc, '__dict__'):
        details = exc.__dict__
    
    # Create validation issue
    issue = ValidationIssue(
        field=field,
        message=str(exc),
        severity=severity,
        actual_value=str(details.get('actual_value', 'unknown')),
        expected=str(details.get('expected', 'valid value')),
        suggestion=get_exception_suggestion(exc, context)
    )
    
    result.add_issue(issue)
    result.context["error_type"] = type(exc).__name__
    result.context["context"] = context
    
    return result


def get_exception_suggestion(exc: Exception, context: str = "general operation") -> str:
    """
    Get a helpful suggestion for resolving an exception.
    
    Args:
        exc: Exception to get suggestion for
        context: Context of the error
        
    Returns:
        Suggestion string
    """
    if isinstance(exc, DivisionByZeroError):
        return "Check denominator values are non-zero before calculation"
    elif isinstance(exc, InvalidInputError):
        if hasattr(exc, 'expected'):
            return f"Provide a value that is: {exc.expected}"
        return "Verify input data format and values"
    elif isinstance(exc, SchemaValidationError):
        return "Check that your data matches the expected schema structure"
    elif isinstance(exc, DataCompletenessError):
        if hasattr(exc, 'missing_fields'):
            return f"Provide the missing fields: {', '.join(exc.missing_fields)}"
        return "Ensure all required data fields are provided"
    elif isinstance(exc, ReconciliationError):
        return "Verify data accuracy across financial statements"
    elif isinstance(exc, PlausibilityError):
        return "Review input data for accuracy - values may be outside normal ranges"
    elif isinstance(exc, ToolNotFoundError):
        if hasattr(exc, 'available_tools'):
            return f"Use one of the available tools: {', '.join(exc.available_tools[:3])}"
        return "Check that the tool name is correct and available"
    elif isinstance(exc, ToolExecutionError):
        return "Check tool parameters and try again"
    elif isinstance(exc, ToolParameterError):
        if hasattr(exc, 'expected_type'):
            return f"Provide a valid {exc.expected_type} value for this parameter"
        return "Check parameter requirements and provide valid values"
    elif isinstance(exc, DataParsingError):
        return "Verify source data format and encoding"
    elif isinstance(exc, MissingDataError):
        if hasattr(exc, 'missing_fields'):
            return f"Provide the missing fields: {', '.join(exc.missing_fields)}"
        return "Ensure all required data is available"
    
    return f"Review the error and try again. Contact support if the issue persists."


def result_to_reasoning_block(result: ValidationResult) -> str:
    """
    Convert ValidationResult to formatted reasoning block.
    
    Args:
        result: Validation result to format
        
    Returns:
        Formatted markdown block
    """
    lines = [
        f"### Validation Result for {result.context.get('context', 'analysis')}",
        "",
        "**Summary**:",
        f"- Status: {'✅ Valid' if result.is_valid else '❌ Invalid'}",
        f"- Errors: {result.error_count}",
        f"- Warnings: {result.warning_count}",
        f"- Info: {result.info_count}",
        "",
    ]
    
    if not result.is_valid:
        lines.append("**Errors**:")
        for issue in result.issues:
            error_icon = "❌ " if issue.severity == ValidationSeverity.ERROR else "⚠️ "
            lines.append(f"  - {error_icon}{issue.field}: {issue.message}")
            if issue.actual_value is not None:
                lines.append(f"    Actual: {issue.actual_value}, Expected: {issue.expected or 'valid value'}")
            if issue.suggestion:
                lines.append(f"    Suggestion: {issue.suggestion}")
        lines.append("")
    
    if result.warning_count > 0:
        lines.append("**Warnings**:")
        for issue in result.warnings:
            lines.append(f"  - ⚠️ {issue.field}: {issue.message}")
            if issue.suggestion:
                lines.append(f"    Suggestion: {issue.suggestion}")
        lines.append("")
    
    if result.info_count > 0:
        lines.append("**Information**:")
        for issue in result.info:
            lines.append(f"  - ℹ️ {issue.field}: {issue.message}")
        lines.append("")
    
    lines.append("**Recommendation**:")
    if result.can_proceed:
        lines.append("✅ Analysis can proceed with the provided data.")
        if result.warning_count > 0:
            lines.append("⚠️ However, please review the warnings for potential data quality issues.")
    else:
        lines.append("❌ Analysis cannot proceed due to validation errors.")
        lines.append("Please correct the errors before continuing.")
    
    return "\n".join(lines)


def merge_validation_results(results: list[ValidationResult]) -> ValidationResult:
    """
    Merge multiple ValidationResult objects into one.
    
    Args:
        results: List of validation results to merge
        
    Returns:
        Merged ValidationResult
    """
    merged = ValidationResult()
    
    for result in results:
        merged = merged.merge(result)
    
    return merged


def is_validation_successful(result: ValidationResult) -> bool:
    """
    Check if validation was successful (no errors).
    
    Args:
        result: ValidationResult to check
        
    Returns:
        True if no errors, False otherwise
    """
    return result.can_proceed


def get_validation_summary(result: ValidationResult) -> dict[str, Any]:
    """
    Get a summary of validation results.
    
    Args:
        result: ValidationResult to summarize
        
    Returns:
        Summary dictionary
    """
    return {
        "is_valid": result.is_valid,
        "can_proceed": result.can_proceed,
        "error_count": result.error_count,
        "warning_count": result.warning_count,
        "info_count": result.info_count,
        "total_issues": result.total_issue_count,
    }

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

# finanalyst_tools/orchestration/pipeline.py
```py
# finanalyst_tools/orchestration/pipeline.py
"""
Analysis pipeline implementing the mandatory 5-phase workflow.

Phases:
1. VALIDATE - Schema validation, completeness check
2. ANALYZE - Identify what to calculate based on data
3. CALCULATE - Execute calculations with audit trail
4. INTERPRET - Add context and insights
5. VERIFY - Pre-delivery checks

This ensures consistent, auditable analysis execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from decimal import Decimal

from finanalyst_tools.models.financial_statements import (
    FinancialStatementSet,
    IncomeStatementData,
    BalanceSheetData,
    CashFlowStatementData,
)
from finanalyst_tools.models.analysis_results import (
    MetricCategory,
    MetricCollection,
    ComprehensiveAnalysisResult,
    CalculationResult,
)
from finanalyst_tools.models.validation import (
    ValidationResult,
    ReconciliationResult,
    PlausibilityResult,
)
from finanalyst_tools.validation.schema_validator import (
    validate_statement_set,
    validate_financial_data_completeness,
)
from finanalyst_tools.validation.reconciliation import run_all_reconciliations
from finanalyst_tools.validation.plausibility import check_all_plausibility
from finanalyst_tools.calculations.profitability import calculate_all_profitability_metrics
from finanalyst_tools.calculations.liquidity import calculate_all_liquidity_metrics
from finanalyst_tools.orchestration.confidence_scorer import calculate_confidence_level
from finanalyst_tools.exceptions import FinAnalystError


class AnalysisPhase(str, Enum):
    """Phases of the analysis pipeline."""
    VALIDATE = "validate"
    ANALYZE = "analyze"
    CALCULATE = "calculate"
    INTERPRET = "interpret"
    VERIFY = "verify"


@dataclass
class AnalysisRequest:
    """
    Request for financial analysis.
    
    Attributes:
        statement_set: Financial statements to analyze
        prior_statement_set: Prior period statements (optional)
        analysis_type: Type of analysis requested
        include_trends: Whether to include trend analysis
        currency: Currency for reporting
    """
    statement_set: FinancialStatementSet
    prior_statement_set: FinancialStatementSet | None = None
    analysis_type: str = "comprehensive"
    include_trends: bool = False
    currency: str = "SGD"


@dataclass
class PipelineState:
    """
    Internal state of the pipeline during execution.
    """
    current_phase: AnalysisPhase = AnalysisPhase.VALIDATE
    validation_result: ValidationResult | None = None
    reconciliation_result: ReconciliationResult | None = None
    plausibility_result: PlausibilityResult | None = None
    metric_collections: list[MetricCollection] = field(default_factory=list)
    all_metrics: list[CalculationResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    phase_completed: dict[AnalysisPhase, bool] = field(default_factory=dict)


class AnalysisPipeline:
    """
    Pipeline for executing financial analysis.
    
    Implements the mandatory 5-phase workflow:
    REQUEST → [1.VALIDATE] → [2.ANALYZE] → [3.CALCULATE] → [4.INTERPRET] → [5.VERIFY] → DELIVER
    """
    
    def __init__(self):
        """Initialize the pipeline."""
        self.state: PipelineState | None = None

    def _require_phase(self, phase: AnalysisPhase) -> None:
        if self.state is None:
            raise FinAnalystError("Pipeline has not been initialized")

        if not self.state.phase_completed.get(phase, False):
            raise FinAnalystError(f"Phase '{phase.value}' is required before continuing")

    def _set_phase_completed(self, phase: AnalysisPhase) -> None:
        if self.state is None:
            raise FinAnalystError("Pipeline has not been initialized")
        self.state.phase_completed[phase] = True
    
    def execute(self, request: AnalysisRequest) -> ComprehensiveAnalysisResult:
        """
        Execute the full analysis pipeline.
        
        Args:
            request: Analysis request with financial data
            
        Returns:
            ComprehensiveAnalysisResult with all analysis outputs
        """
        # Initialize state
        self.state = PipelineState()
        
        try:
            # Phase 1: VALIDATE
            self._phase_validate(request)
            if not self.state.validation_result or not self.state.validation_result.can_proceed:
                return self._create_error_result(request, "Validation failed")

            # Phase 2: ANALYZE
            analysis_plan = self._phase_analyze(request)

            # Phase 3: CALCULATE
            self._phase_calculate(request, analysis_plan)

            # Phase 4: INTERPRET
            self._phase_interpret(request)

            # Phase 5: VERIFY
            self._phase_verify(request)

            # Create final result
            return self._create_result(request)
        except Exception as e:
            if self.state is not None:
                self.state.errors.append(f"Pipeline error in phase '{self.state.current_phase.value}': {str(e)}")
            return self._create_error_result(request, f"Pipeline execution failed: {str(e)}")
    
    def _phase_validate(self, request: AnalysisRequest) -> None:
        """
        Phase 1: VALIDATE
        
        - Schema validation
        - Data completeness check
        - Cross-statement reconciliation
        """
        self.state.current_phase = AnalysisPhase.VALIDATE
        
        # Schema validation
        validation = validate_statement_set(
            request.statement_set,
            request.analysis_type,
        )
        self.state.validation_result = validation
        
        if not validation.can_proceed:
            self.state.errors.append("Schema validation failed")
            self._set_phase_completed(AnalysisPhase.VALIDATE)
            return
        
        # Reconciliation (if cash flow available)
        prior_bs = None
        if request.prior_statement_set:
            prior_bs = request.prior_statement_set.balance_sheet
        
        reconciliation = run_all_reconciliations(
            request.statement_set,
            prior_balance_sheet=prior_bs,
        )
        self.state.reconciliation_result = reconciliation
        
        if not reconciliation.all_passed:
            for check in reconciliation.failed_checks:
                self.state.warnings.append(f"Reconciliation: {check.message}")

        self._set_phase_completed(AnalysisPhase.VALIDATE)
    
    def _phase_analyze(self, request: AnalysisRequest) -> dict[str, bool]:
        """
        Phase 2: ANALYZE
        
        Determine what calculations to perform based on:
        - Analysis type requested
        - Data available
        
        Returns:
            Dictionary of metric categories to calculate
        """
        self._require_phase(AnalysisPhase.VALIDATE)
        if not self.state.validation_result or not self.state.validation_result.can_proceed:
            raise FinAnalystError("Cannot analyze because validation did not pass")

        self.state.current_phase = AnalysisPhase.ANALYZE
        
        analysis_plan = {
            "profitability": False,
            "liquidity": False,
            "solvency": False,
            "efficiency": False,
        }
        
        analysis_type = request.analysis_type.lower()
        
        if analysis_type in ("profitability", "comprehensive"):
            analysis_plan["profitability"] = True
        
        if analysis_type in ("liquidity", "comprehensive"):
            analysis_plan["liquidity"] = True
        
        if analysis_type in ("solvency", "comprehensive"):
            analysis_plan["solvency"] = True
        
        if analysis_type in ("efficiency", "comprehensive"):
            analysis_plan["efficiency"] = True
        
        self._set_phase_completed(AnalysisPhase.ANALYZE)
        return analysis_plan
    
    def _phase_calculate(
        self,
        request: AnalysisRequest,
        analysis_plan: dict[str, bool],
    ) -> None:
        """
        Phase 3: CALCULATE
        
        Execute all planned calculations.
        """
        self._require_phase(AnalysisPhase.ANALYZE)
        self.state.current_phase = AnalysisPhase.CALCULATE
        
        prior_bs = None
        if request.prior_statement_set:
            prior_bs = request.prior_statement_set.balance_sheet
        
        # Profitability metrics
        if analysis_plan.get("profitability"):
            profitability = calculate_all_profitability_metrics(
                income_statement=request.statement_set.income_statement,
                balance_sheet=request.statement_set.balance_sheet,
                prior_balance_sheet=prior_bs,
            )
            self.state.metric_collections.append(profitability)
            self.state.all_metrics.extend(profitability.metrics)
        
        # Liquidity metrics
        if analysis_plan.get("liquidity"):
            liquidity = calculate_all_liquidity_metrics(
                balance_sheet=request.statement_set.balance_sheet,
            )
            self.state.metric_collections.append(liquidity)
            self.state.all_metrics.extend(liquidity.metrics)
        
        # Note: Solvency and Efficiency calculations would be added in Phase 2
        
        self._set_phase_completed(AnalysisPhase.CALCULATE)
    
    def _phase_interpret(self, request: AnalysisRequest) -> None:
        """
        Phase 4: INTERPRET
        
        Add context and insights to calculated metrics.
        """
        self._require_phase(AnalysisPhase.CALCULATE)
        self.state.current_phase = AnalysisPhase.INTERPRET
        
        # Plausibility checks on all metrics
        plausibility = check_all_plausibility(self.state.all_metrics)
        self.state.plausibility_result = plausibility
        
        # Add warnings for implausible values
        for check in plausibility.implausible_checks:
            self.state.warnings.append(f"Plausibility: {check.message}")
        
        self._set_phase_completed(AnalysisPhase.INTERPRET)
    
    def _phase_verify(self, request: AnalysisRequest) -> None:
        """
        Phase 5: VERIFY
        
        Pre-delivery checks:
        - Ensure all requested calculations completed
        - Verify no critical errors
        - Final quality check
        """
        self._require_phase(AnalysisPhase.INTERPRET)
        self.state.current_phase = AnalysisPhase.VERIFY
        
        # Check that calculations were performed
        if not self.state.metric_collections:
            self.state.warnings.append("No metrics were calculated")
        
        # Check for any uncalculable metrics
        uncalculable = [m for m in self.state.all_metrics if m.value is None]
        if uncalculable:
            for m in uncalculable:
                self.state.warnings.append(f"Could not calculate: {m.metric_name}")
        
        self._set_phase_completed(AnalysisPhase.VERIFY)
    
    def _create_result(self, request: AnalysisRequest) -> ComprehensiveAnalysisResult:
        """Create the final analysis result."""
        
        # Calculate confidence
        data_completeness = 1.0
        if self.state.validation_result:
            total_issues = self.state.validation_result.total_issue_count
            data_completeness = max(0.0, 1.0 - (total_issues * 0.1))
        
        confidence = calculate_confidence_level(
            validation_result=self.state.validation_result,
            plausibility_result=self.state.plausibility_result,
            reconciliation_result=self.state.reconciliation_result,
            data_completeness=data_completeness,
        )
        
        # Build result
        result = ComprehensiveAnalysisResult(
            analysis_type=request.analysis_type,
            period=str(request.statement_set.period),
            currency=request.currency,
            metric_collections=self.state.metric_collections,
            confidence=confidence,
        )

        uncalculable_metrics = [m.metric_name for m in self.state.all_metrics if m.value is None]
        result.is_partial = len(uncalculable_metrics) > 0
        result.uncalculable_metrics = uncalculable_metrics
        result.pipeline_errors = list(self.state.errors)
        result.pipeline_warnings = list(self.state.warnings)
        
        # Add validation summary
        if self.state.validation_result:
            result.validation_summary = self.state.validation_result.to_dict()
        
        # Add reconciliation summary
        if self.state.reconciliation_result:
            result.reconciliation_summary = self.state.reconciliation_result.to_dict()
        
        # Add recommendations based on findings
        result.recommendations = self._generate_recommendations()
        
        return result
    
    def _create_error_result(
        self,
        request: AnalysisRequest,
        error_message: str,
    ) -> ComprehensiveAnalysisResult:
        """Create an error result when pipeline fails."""
        result = ComprehensiveAnalysisResult(
            analysis_type=request.analysis_type,
            period=str(request.statement_set.period),
            currency=request.currency,
        )

        if self.state is not None:
            result.pipeline_errors = list(self.state.errors)
            result.pipeline_warnings = list(self.state.warnings)

            if self.state.validation_result:
                result.validation_summary = self.state.validation_result.to_dict()

            if self.state.reconciliation_result:
                result.reconciliation_summary = self.state.reconciliation_result.to_dict()

        if error_message:
            result.pipeline_errors.append(error_message)

        return result
    
    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on analysis findings."""
        recommendations = []
        
        # Based on profitability
        for collection in self.state.metric_collections:
            if collection.category == MetricCategory.PROFITABILITY:
                npm = collection.get_metric("Net Profit Margin")
                if npm and npm.value is not None:
                    if npm.value < Decimal("5"):
                        recommendations.append(
                            "Net profit margin is low (<5%). Consider reviewing cost structure."
                        )
                    elif npm.value > Decimal("20"):
                        recommendations.append(
                            "Strong net profit margin (>20%). Consider reinvestment opportunities."
                        )
        
        # Based on liquidity
        for collection in self.state.metric_collections:
            if collection.category == MetricCategory.LIQUIDITY:
                cr = collection.get_metric("Current Ratio")
                if cr and cr.value is not None:
                    if cr.value < Decimal("1"):
                        recommendations.append(
                            "Current ratio below 1.0 indicates liquidity risk. Review working capital management."
                        )
        
        # Default recommendation if none generated
        if not recommendations:
            recommendations.append(
                "Financial metrics are within normal ranges. Continue monitoring key indicators."
            )
        
        return recommendations

```

# finanalyst_tools/orchestration/__init__.py
```py
# finanalyst_tools/orchestration/__init__.py
"""
Orchestration layer for FinAnalyst-Pro Agent Tools.

This package provides:
- Analysis pipeline with 5-phase workflow
- Confidence scoring for analysis results
- Report generation with mandatory template
"""

from finanalyst_tools.orchestration.pipeline import (
    AnalysisPipeline,
    AnalysisRequest,
    AnalysisPhase,
)

from finanalyst_tools.orchestration.confidence_scorer import (
    calculate_confidence_level,
    ConfidenceScorer,
)

from finanalyst_tools.orchestration.report_generator import (
    generate_financial_report,
    ReportGenerator,
    ReportFormat,
)


__all__ = [
    # Pipeline
    "AnalysisPipeline",
    "AnalysisRequest",
    "AnalysisPhase",
    # Confidence
    "calculate_confidence_level",
    "ConfidenceScorer",
    # Reporting
    "generate_financial_report",
    "ReportGenerator",
    "ReportFormat",
]

```

# finanalyst_tools/orchestration/report_generator.py
```py
# finanalyst_tools/orchestration/report_generator.py
"""
Report generator for financial analysis results.

Generates reports in the mandatory format specified in system prompt:
- Financial Analysis Report header
- Data Validation Summary
- Key Findings
- Detailed Metrics
- Calculation Audit Trail
- Recommendations
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
from datetime import datetime

from finanalyst_tools.models.analysis_results import (
    ComprehensiveAnalysisResult,
    MetricCollection,
    CalculationResult,
    ConfidenceAssessment,
)
from finanalyst_tools.utils.formatting import (
    format_currency,
    format_percentage,
    format_ratio,
    format_markdown_table,
    format_value_with_unit,
)


class ReportFormat(str, Enum):
    """Available report formats."""
    MARKDOWN = "markdown"
    JSON = "json"
    TEXT = "text"


def generate_financial_report(
    analysis_result: ComprehensiveAnalysisResult,
    format: ReportFormat = ReportFormat.MARKDOWN,
    include_audit_trail: bool = True,
) -> str:
    """
    Generate a financial analysis report.
    
    Args:
        analysis_result: Complete analysis result
        format: Output format
        include_audit_trail: Whether to include detailed calculation steps
        
    Returns:
        Formatted report string
    """
    generator = ReportGenerator(
        include_audit_trail=include_audit_trail,
    )
    
    if format == ReportFormat.MARKDOWN:
        return generator.generate_markdown(analysis_result)
    elif format == ReportFormat.JSON:
        return analysis_result.to_json()
    else:
        return generator.generate_text(analysis_result)


class ReportGenerator:
    """
    Generator for financial analysis reports.
    """
    
    def __init__(
        self,
        include_audit_trail: bool = True,
        include_warnings: bool = True,
        company_name: str | None = None,
    ):
        """
        Initialize the report generator.
        
        Args:
            include_audit_trail: Whether to include calculation steps
            include_warnings: Whether to include warning messages
            company_name: Optional company name for report header
        """
        self.include_audit_trail = include_audit_trail
        self.include_warnings = include_warnings
        self.company_name = company_name
    
    def generate_markdown(
        self,
        result: ComprehensiveAnalysisResult,
    ) -> str:
        """
        Generate report in Markdown format.
        
        Follows the mandatory template from system prompt.
        """
        sections = []
        
        # Header
        sections.append(self._generate_header(result))
        
        # Executive Summary with Confidence
        sections.append(self._generate_summary(result))
        
        # Data Validation Summary
        sections.append(self._generate_validation_section(result))
        
        # Key Findings
        sections.append(self._generate_key_findings(result))
        
        # Detailed Metrics
        sections.append(self._generate_metrics_section(result))
        
        # Audit Trail (optional)
        if self.include_audit_trail:
            sections.append(self._generate_audit_trail(result))
        
        # Recommendations
        sections.append(self._generate_recommendations(result))
        
        # Footer
        sections.append(self._generate_footer())
        
        return "\n\n".join(sections)
    
    def generate_text(
        self,
        result: ComprehensiveAnalysisResult,
    ) -> str:
        """Generate report in plain text format."""
        # Simplified version of markdown
        md = self.generate_markdown(result)
        # Remove markdown formatting
        text = md.replace("# ", "").replace("## ", "").replace("### ", "")
        text = text.replace("**", "").replace("*", "")
        text = text.replace("|", " ")
        return text
    
    def _generate_header(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate report header."""
        lines = ["# Financial Analysis Report"]
        
        if self.company_name:
            lines.append(f"\n**Company**: {self.company_name}")
        
        lines.append(f"**Analysis Type**: {result.analysis_type.title()}")
        lines.append(f"**Data Period**: {result.period}")
        lines.append(f"**Currency**: {result.currency}")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        return "\n".join(lines)
    
    def _generate_summary(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate executive summary with confidence level."""
        lines = ["## Executive Summary"]
        
        # Confidence level
        if result.confidence:
            lines.append(f"\n**Confidence Level**: {result.confidence.to_display()}")
        
        # Quick stats
        lines.append(f"\n**Metrics Calculated**: {result.total_metrics}")

        if result.is_partial:
            lines.append("**Result Status**: ⚠️ Partial")
            if result.uncalculable_metrics:
                lines.append(f"**Uncalculable Metrics**: {len(result.uncalculable_metrics)}")

        # Count warnings
        warning_count = len(result.all_warnings)
        if warning_count > 0:
            lines.append(f"**Warnings**: {warning_count}")

        if result.pipeline_warnings:
            lines.append(f"**Pipeline Warnings**: {len(result.pipeline_warnings)}")

        if result.pipeline_errors:
            lines.append(f"**Pipeline Errors**: {len(result.pipeline_errors)}")
        
        return "\n".join(lines)
    
    def _generate_validation_section(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate data validation summary."""
        lines = ["## 1. Data Validation Summary"]
        
        if result.validation_summary:
            is_valid = result.validation_summary.get("is_valid", True)
            error_count = result.validation_summary.get("error_count", 0)
            warning_count = result.validation_summary.get("warning_count", 0)
            
            status = "✅ Passed" if is_valid else "❌ Failed"
            lines.append(f"\n**Status**: {status}")
            
            if error_count > 0:
                lines.append(f"- Errors: {error_count}")
            if warning_count > 0:
                lines.append(f"- Warnings: {warning_count}")
            
            if is_valid and error_count == 0 and warning_count == 0:
                lines.append("- All validation checks passed")
        else:
            lines.append("\n- No validation summary available")

        return "\n".join(lines)

    def _generate_key_findings(self, result: ComprehensiveAnalysisResult) -> str:
        lines = ["## 2. Key Findings"]

        if not result.metric_collections:
            lines.append("\n- No metrics calculated")
            return "\n".join(lines)

        lines.append("")
        shown = 0
        for collection in result.metric_collections:
            if not collection.metrics:
                continue

            lines.append(f"### {collection.category.value.title()}")
            for metric in collection.metrics:
                plausible = "✅" if metric.is_plausible else "⚠️"
                warning_count = len(metric.warnings)
                warning_suffix = f" ({warning_count} warning(s))" if warning_count > 0 else ""
                lines.append(f"- {plausible} **{metric.metric_name}**: {metric.formatted_value}{warning_suffix}")
                shown += 1
                if shown >= 8:
                    break

            if shown >= 8:
                break

        if shown == 0:
            lines.append("\n- No metrics calculated")

        return "\n".join(lines)

    def _generate_metrics_section(self, result: ComprehensiveAnalysisResult) -> str:
        lines = ["## 3. Detailed Metrics"]

        if not result.metric_collections:
            lines.append("\n- No metrics available")
            return "\n".join(lines)

        for collection in result.metric_collections:
            lines.append("")
            lines.append(collection.to_table())

        return "\n".join(lines)

    def _generate_audit_trail(self, result: ComprehensiveAnalysisResult) -> str:
        lines = ["## 4. Calculation Audit Trail"]

        if not result.all_metrics:
            lines.append("\n- No calculations available")
            return "\n".join(lines)

        for metric in result.all_metrics:
            lines.append("")
            lines.append(f"### {metric.metric_name}")
            if metric.calculation_steps:
                lines.append("**Calculation Steps**:")
                for step in metric.calculation_steps:
                    lines.append(f"- {step}")
            if metric.inputs:
                lines.append("**Inputs Used**:")
                for k, v in metric.inputs.items():
                    lines.append(f"- {k}: {v}")
            if metric.warnings:
                lines.append("**Warnings**:")
                for w in metric.warnings:
                    lines.append(f"- {w}")

        return "\n".join(lines)

    def _generate_recommendations(self, result: ComprehensiveAnalysisResult) -> str:
        lines = ["## 5. Recommendations"]

        if result.recommendations:
            lines.append("")
            for rec in result.recommendations:
                lines.append(f"- {rec}")
            return "\n".join(lines)

        lines.append("\n- No recommendations available")
        return "\n".join(lines)

    def _generate_footer(self) -> str:
        return f"---\nReport generated by FinAnalyst-Pro on {datetime.now().strftime('%Y-%m-%d %H:%M')}"

```

# finanalyst_tools/orchestration/confidence_scorer.py
```py
# finanalyst_tools/orchestration/confidence_scorer.py
"""
Confidence scoring for analysis results.

Implements the mandatory confidence level assessment:
- HIGH: Data quality excellent, all checks passed
- MEDIUM: Some warnings but analysis reliable
- LOW: Significant issues, interpret with caution
"""

from __future__ import annotations

from decimal import Decimal

from finanalyst_tools.models.analysis_results import (
    ConfidenceLevel,
    ConfidenceAssessment,
)
from finanalyst_tools.models.validation import (
    ValidationResult,
    ReconciliationResult,
    PlausibilityResult,
)


def _calculate_confidence_assessment(
    validation_result: ValidationResult | None,
    plausibility_result: PlausibilityResult | None,
    reconciliation_result: ReconciliationResult | None,
    data_completeness: float,
    *,
    error_penalty: float,
    warning_penalty: float,
    implausible_penalty: float,
    reconciliation_penalty: float,
    completeness_max_penalty: float,
    high_threshold: float,
    medium_threshold: float,
) -> ConfidenceAssessment:
    score = 100.0
    factors: dict[str, str] = {}

    # Factor 1: Validation issues
    if validation_result:
        error_count = validation_result.error_count
        warning_count = validation_result.warning_count

        if error_count > 0:
            score -= error_count * error_penalty
            factors["validation_errors"] = f"{error_count} error(s) found"

        if warning_count > 0:
            score -= warning_count * warning_penalty
            factors["validation_warnings"] = f"{warning_count} warning(s) found"

    # Factor 2: Plausibility failures
    if plausibility_result:
        implausible = plausibility_result.implausible_count
        if implausible > 0:
            score -= implausible * implausible_penalty
            names = [c.metric_name for c in plausibility_result.implausible_checks[:3]]
            factors["implausible_metrics"] = f"{implausible} metric(s) outside range: {', '.join(names)}"

    # Factor 3: Reconciliation failures
    if reconciliation_result:
        failed = reconciliation_result.failed_count
        if failed > 0:
            score -= failed * reconciliation_penalty
            names = [c.check_name for c in reconciliation_result.failed_checks[:3]]
            factors["reconciliation_failures"] = f"{failed} check(s) failed: {', '.join(names)}"

    # Factor 4: Data completeness
    if data_completeness < 1.0:
        completeness_penalty = (1.0 - data_completeness) * completeness_max_penalty
        score -= completeness_penalty
        factors["data_completeness"] = f"{data_completeness:.0%} of expected data present"

    # Ensure score is in valid range
    score = max(0.0, min(100.0, score))

    # Determine level
    if score >= high_threshold:
        level = ConfidenceLevel.HIGH
    elif score >= medium_threshold:
        level = ConfidenceLevel.MEDIUM
    else:
        level = ConfidenceLevel.LOW

    # Generate justification
    justification = _generate_justification(level, factors, score)

    return ConfidenceAssessment(
        level=level,
        justification=justification,
        factors=factors,
        score=score,
    )


def calculate_confidence_level(
    validation_result: ValidationResult | None = None,
    plausibility_result: PlausibilityResult | None = None,
    reconciliation_result: ReconciliationResult | None = None,
    data_completeness: float = 1.0,
) -> ConfidenceAssessment:
    """Calculate confidence level for analysis results."""
    return _calculate_confidence_assessment(
        validation_result=validation_result,
        plausibility_result=plausibility_result,
        reconciliation_result=reconciliation_result,
        data_completeness=data_completeness,
        error_penalty=20.0,
        warning_penalty=5.0,
        implausible_penalty=10.0,
        reconciliation_penalty=15.0,
        completeness_max_penalty=30.0,
        high_threshold=80.0,
        medium_threshold=50.0,
    )


def _generate_justification(
    level: ConfidenceLevel,
    factors: dict[str, str],
    score: float,
) -> str:
    """Generate human-readable justification for confidence level."""
    
    if level == ConfidenceLevel.HIGH:
        if not factors:
            return "All validation checks passed with no issues"
        return f"Data quality is good with minor observations: {len(factors)} factor(s) noted"
    
    elif level == ConfidenceLevel.MEDIUM:
        factor_summary = "; ".join(factors.values())[:100]
        return f"Analysis reliable with some caveats: {factor_summary}"
    
    else:  # LOW
        factor_summary = "; ".join(factors.values())[:100]
        return f"Significant issues detected: {factor_summary}. Interpret results with caution."


class ConfidenceScorer:
    """
    Class-based confidence scorer with customization options.
    """
    
    def __init__(
        self,
        error_penalty: float = 20.0,
        warning_penalty: float = 5.0,
        implausible_penalty: float = 10.0,
        reconciliation_penalty: float = 15.0,
        high_threshold: float = 80.0,
        medium_threshold: float = 50.0,
    ):
        """
        Initialize with custom scoring parameters.
        
        Args:
            error_penalty: Points deducted per validation error
            warning_penalty: Points deducted per validation warning
            implausible_penalty: Points deducted per implausible metric
            reconciliation_penalty: Points deducted per reconciliation failure
            high_threshold: Minimum score for HIGH confidence
            medium_threshold: Minimum score for MEDIUM confidence
        """
        self.error_penalty = error_penalty
        self.warning_penalty = warning_penalty
        self.implausible_penalty = implausible_penalty
        self.reconciliation_penalty = reconciliation_penalty
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
    
    def calculate(
        self,
        validation_result: ValidationResult | None = None,
        plausibility_result: PlausibilityResult | None = None,
        reconciliation_result: ReconciliationResult | None = None,
        data_completeness: float = 1.0,
    ) -> ConfidenceAssessment:
        """Calculate confidence using instance parameters."""

        return _calculate_confidence_assessment(
            validation_result=validation_result,
            plausibility_result=plausibility_result,
            reconciliation_result=reconciliation_result,
            data_completeness=data_completeness,
            error_penalty=self.error_penalty,
            warning_penalty=self.warning_penalty,
            implausible_penalty=self.implausible_penalty,
            reconciliation_penalty=self.reconciliation_penalty,
            completeness_max_penalty=30.0,
            high_threshold=self.high_threshold,
            medium_threshold=self.medium_threshold,
        )

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

from finanalyst_tools.utils.serialization import to_jsonable


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
        data["gross_profit"] = self.gross_profit
        data["operating_income"] = self.operating_income
        data["ebitda"] = self.ebitda
        data["ebit"] = self.ebit
        data["earnings_before_tax"] = self.earnings_before_tax
        data["calculated_net_income"] = self.calculated_net_income
        return to_jsonable(data)


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
        data["calculated_current_assets"] = self.calculated_current_assets
        data["calculated_non_current_assets"] = self.calculated_non_current_assets
        data["calculated_total_assets"] = self.calculated_total_assets
        data["calculated_current_liabilities"] = self.calculated_current_liabilities
        data["calculated_non_current_liabilities"] = self.calculated_non_current_liabilities
        data["calculated_total_liabilities"] = self.calculated_total_liabilities
        data["calculated_shareholders_equity"] = self.calculated_shareholders_equity
        data["calculated_total_equity"] = self.calculated_total_equity
        data["working_capital"] = self.working_capital
        data["total_debt"] = self.total_debt
        return to_jsonable(data)


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
        data["calculated_operating_cash_flow"] = self.calculated_operating_cash_flow
        data["calculated_investing_cash_flow"] = self.calculated_investing_cash_flow
        data["calculated_financing_cash_flow"] = self.calculated_financing_cash_flow
        data["calculated_net_change"] = self.calculated_net_change
        data["free_cash_flow"] = self.free_cash_flow
        return to_jsonable(data)


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

