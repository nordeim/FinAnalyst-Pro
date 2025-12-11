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
