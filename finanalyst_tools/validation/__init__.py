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
