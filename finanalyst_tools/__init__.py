# finanalyst_tools/__init__.py
"""
FinAnalyst-Pro Agent Tools - Financial Analysis Toolkit for LLM Agents.

This package provides comprehensive financial analysis capabilities:
- Profitability metrics (margins, ROA, ROE, etc.)
- Liquidity ratios (current, quick, cash ratios)
- 5-phase analysis pipeline (VALIDATE → ANALYZE → CALCULATE → INTERPRET → VERIFY)
- Dual LLM provider support (OpenAI/Anthropic schema generation)

Example Usage:
    from finanalyst_tools import ToolRegistry
    
    registry = ToolRegistry()
    result = registry.execute_tool(
        "analyze_financials",
        statement_set={...},
        analysis_type="comprehensive",
    )
"""

__version__ = "1.0.0"
__author__ = "FinAnalyst-Pro Team"

# Core orchestration
from finanalyst_tools.tool_registry import ToolRegistry
from finanalyst_tools.dispatcher import ToolDispatcher
from finanalyst_tools.orchestration.pipeline import AnalysisPipeline, AnalysisRequest
from finanalyst_tools.orchestration.report_generator import (
    generate_financial_report,
    ReportGenerator,
    ReportFormat,
)
from finanalyst_tools.orchestration.confidence_scorer import calculate_confidence_level

# Models
from finanalyst_tools.models.analysis_results import (
    CalculationResult,
    MetricCollection,
    MetricCategory,
    MetricUnit,
    ConfidenceLevel,
    ComprehensiveAnalysisResult,
)
from finanalyst_tools.models.financial_statements import (
    FinancialStatementSet,
    IncomeStatementData,
    BalanceSheetData,
    CashFlowStatementData,
    FinancialPeriod,
)
from finanalyst_tools.models.validation import (
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
)

# Exceptions
from finanalyst_tools.exceptions import (
    FinAnalystError,
    CalculationError,
    ValidationError,
    DataError,
    ToolError,
)


__all__ = [
    # Version
    "__version__",
    # Core
    "ToolRegistry",
    "ToolDispatcher",
    "AnalysisPipeline",
    "AnalysisRequest",
    "generate_financial_report",
    "ReportGenerator",
    "ReportFormat",
    "calculate_confidence_level",
    # Models - Results
    "CalculationResult",
    "MetricCollection",
    "MetricCategory",
    "MetricUnit",
    "ConfidenceLevel",
    "ComprehensiveAnalysisResult",
    # Models - Financial Statements
    "FinancialStatementSet",
    "IncomeStatementData",
    "BalanceSheetData",
    "CashFlowStatementData",
    "FinancialPeriod",
    # Models - Validation
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    # Exceptions
    "FinAnalystError",
    "CalculationError",
    "ValidationError",
    "DataError",
    "ToolError",
]
