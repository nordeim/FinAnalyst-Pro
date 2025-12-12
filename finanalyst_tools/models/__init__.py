# File: finanalyst_tools/models/__init__.py
"""
Data models for FinAnalyst-Pro Agent Tools.

This package provides Pydantic models for:
- Financial statement data (Income Statement, Balance Sheet, Cash Flow)
- Calculation results with audit trails
- Validation results and issues

All models support:
- Type validation via Pydantic
- JSON serialization
- Field aliases for common naming variations
"""

from finanalyst_tools.models.validation import (
    ValidationSeverity,
    ValidationIssue,
    ValidationResult,
    ReconciliationCheck,
    ReconciliationResult,
    PlausibilityCheck,
    PlausibilityResult,
)

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

from finanalyst_tools.models.analysis_results import (
    MetricUnit,
    MetricCategory,
    TrendDirection,
    ConfidenceLevel,
    CalculationResult,
    MetricResult,
    MetricCollection,
    TrendAnalysisResult,
    ConfidenceAssessment,
    ComprehensiveAnalysisResult,
)


__all__ = [
    # Validation models
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationResult",
    "ReconciliationCheck",
    "ReconciliationResult",
    "PlausibilityCheck",
    "PlausibilityResult",
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
    "MetricUnit",
    "MetricCategory",
    "TrendDirection",
    "ConfidenceLevel",
    "CalculationResult",
    "MetricResult",
    "MetricCollection",
    "TrendAnalysisResult",
    "ConfidenceAssessment",
    "ComprehensiveAnalysisResult",
]
