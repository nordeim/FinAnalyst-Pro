"""
Data models for FinAnalyst-Pro tools.

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
    MetricCategory,
    TrendDirection,
    CalculationResult,
    MetricResult,
    TrendAnalysisResult,
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
    # Result types
    "MetricCategory",
    "TrendDirection",
    "CalculationResult",
    "MetricResult",
    "TrendAnalysisResult",
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
