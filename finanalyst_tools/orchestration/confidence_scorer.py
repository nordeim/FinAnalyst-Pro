# finanalyst_tools/orchestration/confidence_scorer.py
"""
Confidence scoring for analysis results.

Implements the mandatory confidence level assessment:
**Confidence Level**: [HIGH | MEDIUM | LOW] — [Brief justification]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


@dataclass
class ScoringFactors:
    """Factors that influence confidence score."""
    validation_score: float = 100.0
    reconciliation_score: float = 100.0
    plausibility_score: float = 100.0
    completeness_score: float = 100.0
    
    @property
    def total_score(self) -> float:
        """Calculate weighted total score."""
        weights = {
            "validation": 0.30,
            "reconciliation": 0.25,
            "plausibility": 0.25,
            "completeness": 0.20,
        }
        return (
            self.validation_score * weights["validation"] +
            self.reconciliation_score * weights["reconciliation"] +
            self.plausibility_score * weights["plausibility"] +
            self.completeness_score * weights["completeness"]
        )


class ConfidenceScorer:
    """
    Calculator for confidence levels.
    
    Analyzes multiple factors to determine overall confidence:
    - Validation results (errors, warnings)
    - Reconciliation results (cross-statement consistency)
    - Plausibility results (reasonable value ranges)
    - Data completeness (percentage of successful calculations)
    """
    
    # Scoring thresholds
    HIGH_THRESHOLD = 80.0
    MEDIUM_THRESHOLD = 50.0
    
    # Penalty weights
    VALIDATION_ERROR_PENALTY = 25.0
    VALIDATION_WARNING_PENALTY = 5.0
    RECONCILIATION_FAILURE_PENALTY = 15.0
    PLAUSIBILITY_FAILURE_PENALTY = 10.0
    
    def __init__(self):
        self.factors: dict[str, str] = {}
        self.scoring = ScoringFactors()
    
    def calculate(
        self,
        validation_result: ValidationResult | None = None,
        plausibility_result: PlausibilityResult | None = None,
        reconciliation_result: ReconciliationResult | None = None,
        data_completeness: float = 1.0,
    ) -> ConfidenceAssessment:
        """
        Calculate confidence level based on all factors.
        
        Args:
            validation_result: Schema/completeness validation result
            plausibility_result: Plausibility check result
            reconciliation_result: Cross-statement reconciliation result
            data_completeness: Fraction of successful calculations (0.0-1.0)
            
        Returns:
            ConfidenceAssessment with level and justification
        """
        self.factors = {}
        self.scoring = ScoringFactors()
        
        # Score validation
        if validation_result:
            self._score_validation(validation_result)
        
        # Score reconciliation
        if reconciliation_result:
            self._score_reconciliation(reconciliation_result)
        
        # Score plausibility
        if plausibility_result:
            self._score_plausibility(plausibility_result)
        
        # Score completeness
        self._score_completeness(data_completeness)
        
        # Determine level
        total_score = self.scoring.total_score
        level = self._determine_level(total_score)
        justification = self._generate_justification(level, total_score)
        
        return ConfidenceAssessment(
            level=level,
            justification=justification,
            factors=self.factors.copy(),
            score=total_score,
        )
    
    def _score_validation(self, result: ValidationResult) -> None:
        """Score based on validation results."""
        score = 100.0
        
        # Penalize errors heavily
        if result.error_count > 0:
            penalty = result.error_count * self.VALIDATION_ERROR_PENALTY
            score -= penalty
            self.factors["validation_errors"] = f"{result.error_count} error(s) found"
        
        # Penalize warnings lightly
        if result.warning_count > 0:
            penalty = result.warning_count * self.VALIDATION_WARNING_PENALTY
            score -= penalty
            self.factors["validation_warnings"] = f"{result.warning_count} warning(s) found"
        
        self.scoring.validation_score = max(0.0, score)
    
    def _score_reconciliation(self, result: ReconciliationResult) -> None:
        """Score based on reconciliation results."""
        score = 100.0
        
        if result.failed_count > 0:
            penalty = result.failed_count * self.RECONCILIATION_FAILURE_PENALTY
            score -= penalty
            failed_names = [c.check_name for c in result.failed_checks]
            self.factors["reconciliation_failures"] = f"Failed: {', '.join(failed_names)}"
        
        self.scoring.reconciliation_score = max(0.0, score)
    
    def _score_plausibility(self, result: PlausibilityResult) -> None:
        """Score based on plausibility results."""
        score = 100.0
        
        if result.implausible_count > 0:
            penalty = result.implausible_count * self.PLAUSIBILITY_FAILURE_PENALTY
            score -= penalty
            implausible_names = [c.metric_name for c in result.implausible_checks]
            self.factors["implausible_metrics"] = f"Unusual: {', '.join(implausible_names)}"
        
        self.scoring.plausibility_score = max(0.0, score)
    
    def _score_completeness(self, completeness: float) -> None:
        """Score based on data completeness."""
        score = completeness * 100.0
        
        if completeness < 1.0:
            self.factors["data_completeness"] = f"{completeness:.0%} of metrics calculated"
        
        self.scoring.completeness_score = score
    
    def _determine_level(self, score: float) -> ConfidenceLevel:
        """Determine confidence level from score."""
        if score >= self.HIGH_THRESHOLD:
            return ConfidenceLevel.HIGH
        elif score >= self.MEDIUM_THRESHOLD:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW
    
    def _generate_justification(self, level: ConfidenceLevel, score: float) -> str:
        """Generate human-readable justification."""
        if level == ConfidenceLevel.HIGH:
            base = "Data validation passed with minimal issues"
            if not self.factors:
                return base + ", all reconciliations successful, values within expected ranges"
        elif level == ConfidenceLevel.MEDIUM:
            base = "Analysis completed with some concerns"
        else:
            base = "Significant data quality issues detected"
        
        # Add specific concerns
        concerns = []
        if "validation_errors" in self.factors:
            concerns.append(self.factors["validation_errors"])
        if "reconciliation_failures" in self.factors:
            concerns.append(self.factors["reconciliation_failures"])
        if "implausible_metrics" in self.factors:
            concerns.append(self.factors["implausible_metrics"])
        if "data_completeness" in self.factors:
            concerns.append(self.factors["data_completeness"])
        
        if concerns:
            return f"{base}: {'; '.join(concerns)}"
        return base


def calculate_confidence_level(
    validation_result: ValidationResult | None = None,
    plausibility_result: PlausibilityResult | None = None,
    reconciliation_result: ReconciliationResult | None = None,
    data_completeness: float = 1.0,
) -> ConfidenceAssessment:
    """
    Convenience function to calculate confidence level.
    
    Args:
        validation_result: Schema/completeness validation result
        plausibility_result: Plausibility check result
        reconciliation_result: Cross-statement reconciliation result
        data_completeness: Fraction of successful calculations (0.0-1.0)
        
    Returns:
        ConfidenceAssessment with level and justification
    """
    scorer = ConfidenceScorer()
    return scorer.calculate(
        validation_result=validation_result,
        plausibility_result=plausibility_result,
        reconciliation_result=reconciliation_result,
        data_completeness=data_completeness,
    )
