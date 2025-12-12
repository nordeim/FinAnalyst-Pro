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
