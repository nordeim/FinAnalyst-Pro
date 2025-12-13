# finanalyst_tools/orchestration/confidence_scorer.py
"""
Confidence scoring for analysis results.

Implements the mandatory confidence level assessment:
- HIGH: Data quality excellent, all checks passed
- MEDIUM: Some warnings but analysis reliable
- LOW: Significant issues, interpret with caution
"""

from __future__ import annotations

from decimal import Decimal

from finanalyst_tools.models.analysis_results import (
    ConfidenceLevel,
    ConfidenceAssessment,
)
from finanalyst_tools.models.validation import (
    ValidationResult,
    ReconciliationResult,
    PlausibilityResult,
)


def _calculate_confidence_assessment(
    validation_result: ValidationResult | None,
    plausibility_result: PlausibilityResult | None,
    reconciliation_result: ReconciliationResult | None,
    data_completeness: float,
    *,
    error_penalty: float,
    warning_penalty: float,
    implausible_penalty: float,
    reconciliation_penalty: float,
    completeness_max_penalty: float,
    high_threshold: float,
    medium_threshold: float,
) -> ConfidenceAssessment:
    score = 100.0
    factors: dict[str, str] = {}

    # Factor 1: Validation issues
    if validation_result:
        error_count = validation_result.error_count
        warning_count = validation_result.warning_count

        if error_count > 0:
            score -= error_count * error_penalty
            factors["validation_errors"] = f"{error_count} error(s) found"

        if warning_count > 0:
            score -= warning_count * warning_penalty
            factors["validation_warnings"] = f"{warning_count} warning(s) found"

    # Factor 2: Plausibility failures
    if plausibility_result:
        implausible = plausibility_result.implausible_count
        if implausible > 0:
            score -= implausible * implausible_penalty
            names = [c.metric_name for c in plausibility_result.implausible_checks[:3]]
            factors["implausible_metrics"] = f"{implausible} metric(s) outside range: {', '.join(names)}"

    # Factor 3: Reconciliation failures
    if reconciliation_result:
        failed = reconciliation_result.failed_count
        if failed > 0:
            score -= failed * reconciliation_penalty
            names = [c.check_name for c in reconciliation_result.failed_checks[:3]]
            factors["reconciliation_failures"] = f"{failed} check(s) failed: {', '.join(names)}"

    # Factor 4: Data completeness
    if data_completeness < 1.0:
        completeness_penalty = (1.0 - data_completeness) * completeness_max_penalty
        score -= completeness_penalty
        factors["data_completeness"] = f"{data_completeness:.0%} of expected data present"

    # Ensure score is in valid range
    score = max(0.0, min(100.0, score))

    # Determine level
    if score >= high_threshold:
        level = ConfidenceLevel.HIGH
    elif score >= medium_threshold:
        level = ConfidenceLevel.MEDIUM
    else:
        level = ConfidenceLevel.LOW

    # Generate justification
    justification = _generate_justification(level, factors, score)

    return ConfidenceAssessment(
        level=level,
        justification=justification,
        factors=factors,
        score=score,
    )


def calculate_confidence_level(
    validation_result: ValidationResult | None = None,
    plausibility_result: PlausibilityResult | None = None,
    reconciliation_result: ReconciliationResult | None = None,
    data_completeness: float = 1.0,
) -> ConfidenceAssessment:
    """Calculate confidence level for analysis results."""
    return _calculate_confidence_assessment(
        validation_result=validation_result,
        plausibility_result=plausibility_result,
        reconciliation_result=reconciliation_result,
        data_completeness=data_completeness,
        error_penalty=20.0,
        warning_penalty=5.0,
        implausible_penalty=10.0,
        reconciliation_penalty=15.0,
        completeness_max_penalty=30.0,
        high_threshold=80.0,
        medium_threshold=50.0,
    )


def _generate_justification(
    level: ConfidenceLevel,
    factors: dict[str, str],
    score: float,
) -> str:
    """Generate human-readable justification for confidence level."""
    
    if level == ConfidenceLevel.HIGH:
        if not factors:
            return "All validation checks passed with no issues"
        return f"Data quality is good with minor observations: {len(factors)} factor(s) noted"
    
    elif level == ConfidenceLevel.MEDIUM:
        factor_summary = "; ".join(factors.values())[:100]
        return f"Analysis reliable with some caveats: {factor_summary}"
    
    else:  # LOW
        factor_summary = "; ".join(factors.values())[:100]
        return f"Significant issues detected: {factor_summary}. Interpret results with caution."


class ConfidenceScorer:
    """
    Class-based confidence scorer with customization options.
    """
    
    def __init__(
        self,
        error_penalty: float = 20.0,
        warning_penalty: float = 5.0,
        implausible_penalty: float = 10.0,
        reconciliation_penalty: float = 15.0,
        high_threshold: float = 80.0,
        medium_threshold: float = 50.0,
    ):
        """
        Initialize with custom scoring parameters.
        
        Args:
            error_penalty: Points deducted per validation error
            warning_penalty: Points deducted per validation warning
            implausible_penalty: Points deducted per implausible metric
            reconciliation_penalty: Points deducted per reconciliation failure
            high_threshold: Minimum score for HIGH confidence
            medium_threshold: Minimum score for MEDIUM confidence
        """
        self.error_penalty = error_penalty
        self.warning_penalty = warning_penalty
        self.implausible_penalty = implausible_penalty
        self.reconciliation_penalty = reconciliation_penalty
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
    
    def calculate(
        self,
        validation_result: ValidationResult | None = None,
        plausibility_result: PlausibilityResult | None = None,
        reconciliation_result: ReconciliationResult | None = None,
        data_completeness: float = 1.0,
    ) -> ConfidenceAssessment:
        """Calculate confidence using instance parameters."""

        return _calculate_confidence_assessment(
            validation_result=validation_result,
            plausibility_result=plausibility_result,
            reconciliation_result=reconciliation_result,
            data_completeness=data_completeness,
            error_penalty=self.error_penalty,
            warning_penalty=self.warning_penalty,
            implausible_penalty=self.implausible_penalty,
            reconciliation_penalty=self.reconciliation_penalty,
            completeness_max_penalty=30.0,
            high_threshold=self.high_threshold,
            medium_threshold=self.medium_threshold,
        )
