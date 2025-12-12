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
