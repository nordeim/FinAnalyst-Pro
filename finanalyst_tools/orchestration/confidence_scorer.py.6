# finanalyst_tools/orchestration/confidence_scorer.py
"""
Confidence scoring for analysis results.

Implements the mandatory confidence level assessment:
- HIGH: Data quality excellent, all checks passed
- MEDIUM: Some warnings but analysis reliable
- LOW: Significant issues, interpret with caution
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from finanalyst_tools.models.analysis_results import (
    ConfidenceLevel,
    ConfidenceAssessment,
    CalculationResult,
)
from finanalyst_tools.models.validation import (
    ValidationResult,
    ReconciliationResult,
    PlausibilityResult,
)


def calculate_confidence_level(
    validation_result: ValidationResult | None = None,
    plausibility_result: PlausibilityResult | None = None,
    reconciliation_result: ReconciliationResult | None = None,
    data_completeness: float = 1.0,
) -> ConfidenceAssessment:
    """
    Calculate confidence level for analysis results.
    
    Scoring factors:
    - Validation warnings: -5 points each
    - Validation errors: -20 points each (should not proceed)
    - Implausible metrics: -10 points each
    - Reconciliation failures: -15 points each
    - Data completeness: Up to -30 points for missing data
    
    Thresholds:
    - HIGH: Score >= 80
    - MEDIUM: Score >= 50
    - LOW: Score < 50
    
    Args:
        validation_result: Schema validation result
        plausibility_result: Plausibility check result
        reconciliation_result: Reconciliation check result
        data_completeness: Fraction of data present (0.0 to 1.0)
        
    Returns:
        ConfidenceAssessment with level and justification
    """
    score = 100.0
    factors: dict[str, str] = {}
    
    # Factor 1: Validation issues
    if validation_result:
        error_count = validation_result.error_count
        warning_count = validation_result.warning_count
        
        if error_count > 0:
            score -= error_count * 20
            factors["validation_errors"] = f"{error_count} error(s) found"
        
        if warning_count > 0:
            score -= warning_count * 5
            factors["validation_warnings"] = f"{warning_count} warning(s) found"
    
    # Factor 2: Plausibility failures
    if plausibility_result:
        implausible = plausibility_result.implausible_count
        if implausible > 0:
            score -= implausible * 10
            names = [c.metric_name for c in plausibility_result.implausible_checks[:3]]
            factors["implausible_metrics"] = f"{implausible} metric(s) outside range: {', '.join(names)}"
    
    # Factor 3: Reconciliation failures
    if reconciliation_result:
        failed = reconciliation_result.failed_count
        if failed > 0:
            score -= failed * 15
            names = [c.check_name for c in reconciliation_result.failed_checks[:3]]
            factors["reconciliation_failures"] = f"{failed} check(s) failed: {', '.join(names)}"
    
    # Factor 4: Data completeness
    if data_completeness < 1.0:
        completeness_penalty = (1.0 - data_completeness) * 30
        score -= completeness_penalty
        factors["data_completeness"] = f"{data_completeness:.0%} of expected data present"
    
    # Ensure score is in valid range
    score = max(0.0, min(100.0, score))
    
    # Determine level
    if score >= 80:
        level = ConfidenceLevel.HIGH
    elif score >= 50:
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
        
        score = 100.0
        factors: dict[str, str] = {}
        
        if validation_result:
            score -= validation_result.error_count * self.error_penalty
            score -= validation_result.warning_count * self.warning_penalty
            if validation_result.error_count:
                factors["errors"] = f"{validation_result.error_count} error(s)"
            if validation_result.warning_count:
                factors["warnings"] = f"{validation_result.warning_count} warning(s)"
        
        if plausibility_result:
            score -= plausibility_result.implausible_count * self.implausible_penalty
            if plausibility_result.implausible_count:
                factors["implausible"] = f"{plausibility_result.implausible_count} metric(s)"
        
        if reconciliation_result:
            score -= reconciliation_result.failed_count * self.reconciliation_penalty
            if reconciliation_result.failed_count:
                factors["reconciliation"] = f"{reconciliation_result.failed_count} failure(s)"
        
        if data_completeness < 1.0:
            score -= (1.0 - data_completeness) * 30
            factors["completeness"] = f"{data_completeness:.0%}"
        
        score = max(0.0, min(100.0, score))
        
        if score >= self.high_threshold:
            level = ConfidenceLevel.HIGH
        elif score >= self.medium_threshold:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW
        
        justification = _generate_justification(level, factors, score)
        
        return ConfidenceAssessment(
            level=level,
            justification=justification,
            factors=factors,
            score=score,
        )
