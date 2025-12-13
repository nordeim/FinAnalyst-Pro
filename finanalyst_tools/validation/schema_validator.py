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
    
    # Required fields for basic balance sheet - strengthened validation
    required = [
        "cash_and_equivalents",
        "total_assets",
        "total_liabilities",
        "total_shareholders_equity",
    ]
    
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
