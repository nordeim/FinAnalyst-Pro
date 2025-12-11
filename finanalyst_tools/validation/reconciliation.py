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
