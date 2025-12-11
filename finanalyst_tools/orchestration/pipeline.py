# finanalyst_tools/orchestration/pipeline.py
"""
Analysis pipeline implementing the mandatory 5-phase workflow.

REQUEST → [1.VALIDATE] → [2.ANALYZE] → [3.CALCULATE] → [4.INTERPRET] → [5.VERIFY] → DELIVER
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from decimal import Decimal

from finanalyst_tools.models.financial_statements import (
    FinancialStatementSet,
    IncomeStatementData,
    BalanceSheetData,
    CashFlowStatementData,
)
from finanalyst_tools.models.analysis_results import (
    MetricCategory,
    MetricCollection,
    ComprehensiveAnalysisResult,
    ConfidenceAssessment,
)
from finanalyst_tools.models.validation import (
    ValidationResult,
    ReconciliationResult,
    PlausibilityResult,
)
from finanalyst_tools.validation import (
    validate_statement_set,
    run_all_reconciliations,
    check_all_plausibility,
)
from finanalyst_tools.calculations import (
    calculate_all_profitability_metrics,
    calculate_all_liquidity_metrics,
)
from finanalyst_tools.orchestration.confidence_scorer import calculate_confidence_level
from finanalyst_tools.orchestration.report_generator import generate_financial_report


class AnalysisPhase(str, Enum):
    """Phases of the analysis pipeline."""
    VALIDATE = "validate"
    ANALYZE = "analyze"
    CALCULATE = "calculate"
    INTERPRET = "interpret"
    VERIFY = "verify"
    DELIVER = "deliver"


@dataclass
class AnalysisRequest:
    """
    Request for financial analysis.
    
    Attributes:
        statement_set: Complete financial statements
        analysis_types: List of analysis categories to perform
        prior_statement_set: Prior period for comparisons (optional)
        include_trends: Whether to include trend analysis
        strict_validation: Whether to fail on validation warnings
    """
    statement_set: FinancialStatementSet
    analysis_types: list[str] = field(default_factory=lambda: ["profitability", "liquidity"])
    prior_statement_set: FinancialStatementSet | None = None
    include_trends: bool = False
    strict_validation: bool = False
    
    @property
    def period(self) -> str:
        """Get the analysis period as string."""
        return str(self.statement_set.period)
    
    @property
    def currency(self) -> str:
        """Get the currency."""
        return self.statement_set.currency


@dataclass
class PipelineState:
    """Internal state tracking for the pipeline."""
    current_phase: AnalysisPhase = AnalysisPhase.VALIDATE
    validation_result: ValidationResult | None = None
    reconciliation_result: ReconciliationResult | None = None
    plausibility_result: PlausibilityResult | None = None
    metric_collections: list[MetricCollection] = field(default_factory=list)
    all_metrics: list = field(default_factory=list)
    confidence: ConfidenceAssessment | None = None
    recommendations: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    @property
    def can_proceed(self) -> bool:
        """Check if pipeline can proceed to next phase."""
        if self.validation_result:
            return self.validation_result.can_proceed
        return True


class AnalysisPipeline:
    """
    Orchestrates the 5-phase analysis workflow.
    
    Ensures all mandatory steps are executed in order:
    1. VALIDATE - Cannot be skipped
    2. ANALYZE - Plan calculations
    3. CALCULATE - Execute with audit trails
    4. INTERPRET - Add context
    5. VERIFY - Final checks
    """
    
    def __init__(self):
        self.state = PipelineState()
    
    def execute(self, request: AnalysisRequest) -> ComprehensiveAnalysisResult:
        """
        Execute the full analysis pipeline.
        
        Args:
            request: Analysis request with financial data
            
        Returns:
            ComprehensiveAnalysisResult with all findings
        """
        # Reset state
        self.state = PipelineState()
        
        # Phase 1: VALIDATE (mandatory)
        self._phase_validate(request)
        
        if not self.state.can_proceed:
            return self._create_validation_failure_result(request)
        
        # Phase 2: ANALYZE
        self._phase_analyze(request)
        
        # Phase 3: CALCULATE
        self._phase_calculate(request)
        
        # Phase 4: INTERPRET
        self._phase_interpret(request)
        
        # Phase 5: VERIFY
        self._phase_verify(request)
        
        # DELIVER
        return self._create_result(request)
    
    def _phase_validate(self, request: AnalysisRequest) -> None:
        """
        Phase 1: Validate input data.
        
        - Schema validation
        - Completeness check
        - Cross-statement reconciliation
        """
        self.state.current_phase = AnalysisPhase.VALIDATE
        
        # Schema and completeness validation
        analysis_type = request.analysis_types[0] if request.analysis_types else "comprehensive"
        self.state.validation_result = validate_statement_set(
            request.statement_set,
            analysis_type,
        )
        
        # Cross-statement reconciliation
        prior_bs = request.prior_statement_set.balance_sheet if request.prior_statement_set else None
        self.state.reconciliation_result = run_all_reconciliations(
            request.statement_set,
            prior_balance_sheet=prior_bs,
        )
        
        # Convert reconciliation failures to validation issues
        if not self.state.reconciliation_result.all_passed:
            recon_validation = self.state.reconciliation_result.to_validation_result()
            self.state.validation_result.merge(recon_validation)
    
    def _phase_analyze(self, request: AnalysisRequest) -> None:
        """
        Phase 2: Analyze what calculations to perform.
        
        - Determine available data
        - Plan calculation sequence
        """
        self.state.current_phase = AnalysisPhase.ANALYZE
        # Analysis planning is implicit in the calculation phase
        # Future: Could add more sophisticated dependency analysis
    
    def _phase_calculate(self, request: AnalysisRequest) -> None:
        """
        Phase 3: Execute calculations.
        
        - Run all requested metric calculations
        - Capture audit trails
        """
        self.state.current_phase = AnalysisPhase.CALCULATE
        
        prior_bs = request.prior_statement_set.balance_sheet if request.prior_statement_set else None
        
        for analysis_type in request.analysis_types:
            if analysis_type.lower() == "profitability":
                collection = calculate_all_profitability_metrics(
                    income_statement=request.statement_set.income_statement,
                    balance_sheet=request.statement_set.balance_sheet,
                    prior_balance_sheet=prior_bs,
                )
                self.state.metric_collections.append(collection)
                self.state.all_metrics.extend(collection.metrics)
            
            elif analysis_type.lower() == "liquidity":
                collection = calculate_all_liquidity_metrics(
                    balance_sheet=request.statement_set.balance_sheet,
                )
                self.state.metric_collections.append(collection)
                self.state.all_metrics.extend(collection.metrics)
    
    def _phase_interpret(self, request: AnalysisRequest) -> None:
        """
        Phase 4: Interpret results.
        
        - Run plausibility checks
        - Generate recommendations
        """
        self.state.current_phase = AnalysisPhase.INTERPRET
        
        # Plausibility checks
        self.state.plausibility_result = check_all_plausibility(self.state.all_metrics)
        
        # Generate recommendations based on findings
        self._generate_recommendations(request)
    
    def _phase_verify(self, request: AnalysisRequest) -> None:
        """
        Phase 5: Verify results before delivery.
        
        - Calculate confidence score
        - Final quality checks
        """
        self.state.current_phase = AnalysisPhase.VERIFY
        
        # Calculate data completeness
        total_metrics = len(self.state.all_metrics)
        calculable_metrics = sum(1 for m in self.state.all_metrics if m.value is not None)
        completeness = calculable_metrics / total_metrics if total_metrics > 0 else 0.0
        
        # Calculate confidence
        self.state.confidence = calculate_confidence_level(
            validation_result=self.state.validation_result,
            plausibility_result=self.state.plausibility_result,
            reconciliation_result=self.state.reconciliation_result,
            data_completeness=completeness,
        )
    
    def _generate_recommendations(self, request: AnalysisRequest) -> None:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        for metric in self.state.all_metrics:
            if not metric.is_plausible:
                recommendations.append(
                    f"Review data accuracy for {metric.metric_name} - value appears unusual"
                )
            
            for warning in metric.warnings:
                if "liquidity risk" in warning.lower():
                    recommendations.append(
                        "Consider strategies to improve short-term liquidity position"
                    )
                elif "negative" in warning.lower() and "working capital" in metric.metric_name.lower():
                    recommendations.append(
                        "Address negative working capital to reduce financial risk"
                    )
        
        # Deduplicate
        self.state.recommendations = list(dict.fromkeys(recommendations))
    
    def _create_result(self, request: AnalysisRequest) -> ComprehensiveAnalysisResult:
        """Create the final analysis result."""
        return ComprehensiveAnalysisResult(
            analysis_type=", ".join(request.analysis_types),
            period=request.period,
            currency=request.currency,
            metric_collections=self.state.metric_collections,
            confidence=self.state.confidence,
            validation_summary=self.state.validation_result.to_dict() if self.state.validation_result else {},
            reconciliation_summary=self.state.reconciliation_result.to_dict() if self.state.reconciliation_result else {},
            recommendations=self.state.recommendations,
        )
    
    def _create_validation_failure_result(self, request: AnalysisRequest) -> ComprehensiveAnalysisResult:
        """Create a result for validation failure."""
        result = ComprehensiveAnalysisResult(
            analysis_type=", ".join(request.analysis_types),
            period=request.period,
            currency=request.currency,
            validation_summary=self.state.validation_result.to_dict() if self.state.validation_result else {},
            reconciliation_summary=self.state.reconciliation_result.to_dict() if self.state.reconciliation_result else {},
        )
        result.add_recommendation("Fix validation errors before proceeding with analysis")
        return result


def run_analysis(
    statement_set: FinancialStatementSet,
    analysis_types: list[str] | None = None,
    prior_statement_set: FinancialStatementSet | None = None,
) -> ComprehensiveAnalysisResult:
    """
    Convenience function to run the full analysis pipeline.
    
    Args:
        statement_set: Financial statements to analyze
        analysis_types: Types of analysis to perform (default: profitability, liquidity)
        prior_statement_set: Prior period data for comparisons
        
    Returns:
        ComprehensiveAnalysisResult with all findings
    """
    request = AnalysisRequest(
        statement_set=statement_set,
        analysis_types=analysis_types or ["profitability", "liquidity"],
        prior_statement_set=prior_statement_set,
    )
    
    pipeline = AnalysisPipeline()
    return pipeline.execute(request)
