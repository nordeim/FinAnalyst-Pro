# finanalyst_tools/orchestration/__init__.py
"""
Orchestration layer for FinAnalyst-Pro Agent Tools.

This package provides the mandatory 5-phase processing pipeline:
1. VALIDATE - Data validation and completeness checks
2. ANALYZE - Determine what to calculate
3. CALCULATE - Execute calculations with audit trails
4. INTERPRET - Add context and insights
5. VERIFY - Pre-delivery checks

Also provides:
- Confidence scoring
- Report generation in mandatory format
"""

from finanalyst_tools.orchestration.pipeline import (
    AnalysisPipeline,
    AnalysisRequest,
    AnalysisPhase,
    run_analysis,
)

from finanalyst_tools.orchestration.confidence_scorer import (
    ConfidenceScorer,
    calculate_confidence_level,
)

from finanalyst_tools.orchestration.report_generator import (
    ReportGenerator,
    generate_financial_report,
    ReportFormat,
)


__all__ = [
    # Pipeline
    "AnalysisPipeline",
    "AnalysisRequest",
    "AnalysisPhase",
    "run_analysis",
    # Confidence
    "ConfidenceScorer",
    "calculate_confidence_level",
    # Reporting
    "ReportGenerator",
    "generate_financial_report",
    "ReportFormat",
]
