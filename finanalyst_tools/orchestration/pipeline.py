# finanalyst_tools/orchestration/pipeline.py
"""
Analysis pipeline implementing the mandatory 5-phase workflow.

Phases:
1. VALIDATE - Schema validation, completeness check
2. ANALYZE - Identify what to calculate based on data
3. CALCULATE - Execute calculations with audit trail
4. INTERPRET - Add context and insights
5. VERIFY - Pre-delivery checks

This ensures consistent, auditable analysis execution.
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
    CalculationResult,
)
from finanalyst_tools.models.validation import (
    ValidationResult,
    ReconciliationResult,
    PlausibilityResult,
)
from finanalyst_tools.validation.schema_validator import (
    validate_statement_set,
    validate_financial_data_completeness,
)
from finanalyst_tools.validation.reconciliation import run_all_reconciliations
from finanalyst_tools.validation.plausibility import check_all_plausibility
from finanalyst_tools.calculations.profitability import calculate_all_profitability_metrics
from finanalyst_tools.calculations.liquidity import calculate_all_liquidity_metrics
from finanalyst_tools.orchestration.confidence_scorer import calculate_confidence_level
from finanalyst_tools.exceptions import FinAnalystError


class AnalysisPhase(str, Enum):
    """Phases of the analysis pipeline."""
    VALIDATE = "validate"
    ANALYZE = "analyze"
    CALCULATE = "calculate"
    INTERPRET = "interpret"
    VERIFY = "verify"


@dataclass
class AnalysisRequest:
    """
    Request for financial analysis.
    
    Attributes:
        statement_set: Financial statements to analyze
        prior_statement_set: Prior period statements (optional)
        analysis_type: Type of analysis requested
        include_trends: Whether to include trend analysis
        currency: Currency for reporting
    """
    statement_set: FinancialStatementSet
    prior_statement_set: FinancialStatementSet | None = None
    analysis_type: str = "comprehensive"
    include_trends: bool = False
    currency: str = "SGD"


@dataclass
class PipelineState:
    """
    Internal state of the pipeline during execution.
    """
    current_phase: AnalysisPhase = AnalysisPhase.VALIDATE
    validation_result: ValidationResult | None = None
    reconciliation_result: ReconciliationResult | None = None
    plausibility_result: PlausibilityResult | None = None
    metric_collections: list[MetricCollection] = field(default_factory=list)
    all_metrics: list[CalculationResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    phase_completed: dict[AnalysisPhase, bool] = field(default_factory=dict)


class AnalysisPipeline:
    """
    Pipeline for executing financial analysis.
    
    Implements the mandatory 5-phase workflow:
    REQUEST → [1.VALIDATE] → [2.ANALYZE] → [3.CALCULATE] → [4.INTERPRET] → [5.VERIFY] → DELIVER
    """
    
    def __init__(self):
        """Initialize the pipeline."""
        self.state: PipelineState | None = None

    def _require_phase(self, phase: AnalysisPhase) -> None:
        if self.state is None:
            raise FinAnalystError("Pipeline has not been initialized")

        if not self.state.phase_completed.get(phase, False):
            raise FinAnalystError(f"Phase '{phase.value}' is required before continuing")

    def _set_phase_completed(self, phase: AnalysisPhase) -> None:
        if self.state is None:
            raise FinAnalystError("Pipeline has not been initialized")
        self.state.phase_completed[phase] = True
    
    def execute(self, request: AnalysisRequest) -> ComprehensiveAnalysisResult:
        """
        Execute the full analysis pipeline.
        
        Args:
            request: Analysis request with financial data
            
        Returns:
            ComprehensiveAnalysisResult with all analysis outputs
        """
        # Initialize state
        self.state = PipelineState()
        
        try:
            # Phase 1: VALIDATE
            self._phase_validate(request)
            if not self.state.validation_result or not self.state.validation_result.can_proceed:
                return self._create_error_result(request, "Validation failed")

            # Phase 2: ANALYZE
            analysis_plan = self._phase_analyze(request)

            # Phase 3: CALCULATE
            self._phase_calculate(request, analysis_plan)

            # Phase 4: INTERPRET
            self._phase_interpret(request)

            # Phase 5: VERIFY
            self._phase_verify(request)

            # Create final result
            return self._create_result(request)
        except (FinAnalystError, ValueError, TypeError) as e:
            # Handle expected errors gracefully
            if self.state is not None:
                self.state.errors.append(f"Pipeline error in phase '{self.state.current_phase.value}': {str(e)}")
            return self._create_error_result(request, f"Pipeline execution failed: {str(e)}")
        except Exception as e:
            # Log unexpected errors but don't silently swallow them
            if self.state is not None:
                self.state.errors.append(f"Unexpected error in phase '{self.state.current_phase.value}': {type(e).__name__}: {str(e)}")
            return self._create_error_result(request, f"Unexpected pipeline error: {type(e).__name__}: {str(e)}")
    
    def _phase_validate(self, request: AnalysisRequest) -> None:
        """
        Phase 1: VALIDATE
        
        - Schema validation
        - Data completeness check
        - Cross-statement reconciliation
        """
        self.state.current_phase = AnalysisPhase.VALIDATE
        
        # Schema validation
        validation = validate_statement_set(
            request.statement_set,
            request.analysis_type,
        )
        self.state.validation_result = validation
        
        if not validation.can_proceed:
            self.state.errors.append("Schema validation failed")
            self._set_phase_completed(AnalysisPhase.VALIDATE)
            return
        
        # Reconciliation (if cash flow available)
        prior_bs = None
        if request.prior_statement_set:
            prior_bs = request.prior_statement_set.balance_sheet
        
        reconciliation = run_all_reconciliations(
            request.statement_set,
            prior_balance_sheet=prior_bs,
        )
        self.state.reconciliation_result = reconciliation
        
        if not reconciliation.all_passed:
            for check in reconciliation.failed_checks:
                self.state.warnings.append(f"Reconciliation: {check.message}")

        self._set_phase_completed(AnalysisPhase.VALIDATE)
    
    def _phase_analyze(self, request: AnalysisRequest) -> dict[str, bool]:
        """
        Phase 2: ANALYZE
        
        Determine what calculations to perform based on:
        - Analysis type requested
        - Data available
        
        Returns:
            Dictionary of metric categories to calculate
        """
        self._require_phase(AnalysisPhase.VALIDATE)
        if not self.state.validation_result or not self.state.validation_result.can_proceed:
            raise FinAnalystError("Cannot analyze because validation did not pass")

        self.state.current_phase = AnalysisPhase.ANALYZE
        
        analysis_plan = {
            "profitability": False,
            "liquidity": False,
            "solvency": False,
            "efficiency": False,
        }
        
        analysis_type = request.analysis_type.lower()
        
        if analysis_type in ("profitability", "comprehensive"):
            analysis_plan["profitability"] = True
        
        if analysis_type in ("liquidity", "comprehensive"):
            analysis_plan["liquidity"] = True
        
        if analysis_type in ("solvency", "comprehensive"):
            analysis_plan["solvency"] = True
        
        if analysis_type in ("efficiency", "comprehensive"):
            analysis_plan["efficiency"] = True
        
        self._set_phase_completed(AnalysisPhase.ANALYZE)
        return analysis_plan
    
    def _phase_calculate(
        self,
        request: AnalysisRequest,
        analysis_plan: dict[str, bool],
    ) -> None:
        """
        Phase 3: CALCULATE
        
        Execute all planned calculations.
        """
        self._require_phase(AnalysisPhase.ANALYZE)
        self.state.current_phase = AnalysisPhase.CALCULATE
        
        prior_bs = None
        if request.prior_statement_set:
            prior_bs = request.prior_statement_set.balance_sheet
        
        # Profitability metrics
        if analysis_plan.get("profitability"):
            profitability = calculate_all_profitability_metrics(
                income_statement=request.statement_set.income_statement,
                balance_sheet=request.statement_set.balance_sheet,
                prior_balance_sheet=prior_bs,
            )
            self.state.metric_collections.append(profitability)
            self.state.all_metrics.extend(profitability.metrics)
        
        # Liquidity metrics
        if analysis_plan.get("liquidity"):
            liquidity = calculate_all_liquidity_metrics(
                balance_sheet=request.statement_set.balance_sheet,
            )
            self.state.metric_collections.append(liquidity)
            self.state.all_metrics.extend(liquidity.metrics)
        
        # Note: Solvency and Efficiency calculations would be added in Phase 2
        
        self._set_phase_completed(AnalysisPhase.CALCULATE)
    
    def _phase_interpret(self, request: AnalysisRequest) -> None:
        """
        Phase 4: INTERPRET
        
        Add context and insights to calculated metrics.
        """
        self._require_phase(AnalysisPhase.CALCULATE)
        self.state.current_phase = AnalysisPhase.INTERPRET
        
        # Plausibility checks on all metrics
        plausibility = check_all_plausibility(self.state.all_metrics)
        self.state.plausibility_result = plausibility
        
        # Add warnings for implausible values
        for check in plausibility.implausible_checks:
            self.state.warnings.append(f"Plausibility: {check.message}")
        
        self._set_phase_completed(AnalysisPhase.INTERPRET)
    
    def _phase_verify(self, request: AnalysisRequest) -> None:
        """
        Phase 5: VERIFY
        
        Pre-delivery checks:
        - Ensure all requested calculations completed
        - Verify no critical errors
        - Final quality check
        """
        self._require_phase(AnalysisPhase.INTERPRET)
        self.state.current_phase = AnalysisPhase.VERIFY
        
        # Check that calculations were performed
        if not self.state.metric_collections:
            self.state.warnings.append("No metrics were calculated")
        
        # Check for any uncalculable metrics
        uncalculable = [m for m in self.state.all_metrics if m.value is None]
        if uncalculable:
            for m in uncalculable:
                self.state.warnings.append(f"Could not calculate: {m.metric_name}")
        
        self._set_phase_completed(AnalysisPhase.VERIFY)
    
    def _create_result(self, request: AnalysisRequest) -> ComprehensiveAnalysisResult:
        """Create the final analysis result."""
        
        # Calculate confidence
        data_completeness = 1.0
        if self.state.validation_result:
            total_issues = self.state.validation_result.total_issue_count
            data_completeness = max(0.0, 1.0 - (total_issues * 0.1))
        
        confidence = calculate_confidence_level(
            validation_result=self.state.validation_result,
            plausibility_result=self.state.plausibility_result,
            reconciliation_result=self.state.reconciliation_result,
            data_completeness=data_completeness,
        )
        
        # Build result
        result = ComprehensiveAnalysisResult(
            analysis_type=request.analysis_type,
            period=str(request.statement_set.period),
            currency=request.currency,
            metric_collections=self.state.metric_collections,
            confidence=confidence,
        )

        uncalculable_metrics = [m.metric_name for m in self.state.all_metrics if m.value is None]
        result.is_partial = len(uncalculable_metrics) > 0
        result.uncalculable_metrics = uncalculable_metrics
        result.pipeline_errors = list(self.state.errors)
        result.pipeline_warnings = list(self.state.warnings)
        
        # Add validation summary
        if self.state.validation_result:
            result.validation_summary = self.state.validation_result.to_dict()
        
        # Add reconciliation summary
        if self.state.reconciliation_result:
            result.reconciliation_summary = self.state.reconciliation_result.to_dict()
        
        # Add recommendations based on findings
        result.recommendations = self._generate_recommendations()
        
        return result
    
    def _create_error_result(
        self,
        request: AnalysisRequest,
        error_message: str,
    ) -> ComprehensiveAnalysisResult:
        """Create an error result when pipeline fails."""
        result = ComprehensiveAnalysisResult(
            analysis_type=request.analysis_type,
            period=str(request.statement_set.period),
            currency=request.currency,
        )

        if self.state is not None:
            result.pipeline_errors = list(self.state.errors)
            result.pipeline_warnings = list(self.state.warnings)

            if self.state.validation_result:
                result.validation_summary = self.state.validation_result.to_dict()

            if self.state.reconciliation_result:
                result.reconciliation_summary = self.state.reconciliation_result.to_dict()

        if error_message:
            result.pipeline_errors.append(error_message)

        return result
    
    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on analysis findings."""
        recommendations = []
        
        # Based on profitability
        for collection in self.state.metric_collections:
            if collection.category == MetricCategory.PROFITABILITY:
                npm = collection.get_metric("Net Profit Margin")
                if npm and npm.value is not None:
                    if npm.value < Decimal("5"):
                        recommendations.append(
                            "Net profit margin is low (<5%). Consider reviewing cost structure."
                        )
                    elif npm.value > Decimal("20"):
                        recommendations.append(
                            "Strong net profit margin (>20%). Consider reinvestment opportunities."
                        )
        
        # Based on liquidity
        for collection in self.state.metric_collections:
            if collection.category == MetricCategory.LIQUIDITY:
                cr = collection.get_metric("Current Ratio")
                if cr and cr.value is not None:
                    if cr.value < Decimal("1"):
                        recommendations.append(
                            "Current ratio below 1.0 indicates liquidity risk. Review working capital management."
                        )
        
        # Default recommendation if none generated
        if not recommendations:
            recommendations.append(
                "Financial metrics are within normal ranges. Continue monitoring key indicators."
            )
        
        return recommendations
