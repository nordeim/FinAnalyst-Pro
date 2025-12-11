# finanalyst_tools/models/analysis_results.py
"""
Structured result models for calculations and analysis.

Provides:
- CalculationResult with full audit trail
- MetricCollection for grouped metrics
- TrendAnalysisResult for multi-period analysis
- ConfidenceAssessment for output confidence scoring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any
import json

from finanalyst_tools.models.validation import ValidationResult


class MetricUnit(str, Enum):
    """Units for metric values."""
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    CURRENCY = "currency"
    DAYS = "days"
    COUNT = "count"
    TIMES = "times"


class MetricCategory(str, Enum):
    """Categories of financial metrics."""
    PROFITABILITY = "profitability"
    LIQUIDITY = "liquidity"
    SOLVENCY = "solvency"
    EFFICIENCY = "efficiency"
    GROWTH = "growth"
    VALUATION = "valuation"


class TrendDirection(str, Enum):
    """Direction of trend over time."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class ConfidenceLevel(str, Enum):
    """Confidence levels for analysis output."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class CalculationStep:
    """
    A single step in a calculation for audit trail.
    """
    step_number: int
    description: str
    formula: str | None = None
    values: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step_number,
            "description": self.description,
            "formula": self.formula,
            "values": {k: str(v) for k, v in self.values.items()},
            "result": str(self.result) if self.result is not None else None,
        }
    
    def __str__(self) -> str:
        """Format as readable string."""
        parts = [f"Step {self.step_number}: {self.description}"]
        if self.formula:
            parts.append(f"  Formula: {self.formula}")
        if self.values:
            values_str = ", ".join(f"{k}={v}" for k, v in self.values.items())
            parts.append(f"  Values: {values_str}")
        if self.result is not None:
            parts.append(f"  Result: {self.result}")
        return "\n".join(parts)


@dataclass
class CalculationResult:
    """
    Complete result of a single metric calculation.
    
    Includes the calculated value, formula, all inputs used,
    step-by-step calculation audit trail, plausibility assessment,
    and any warnings.
    """
    metric_name: str
    value: Decimal | None
    unit: MetricUnit
    formula: str
    inputs: dict[str, Any] = field(default_factory=dict)
    calculation_steps: list[CalculationStep] = field(default_factory=list)
    is_plausible: bool = True
    plausibility_range: tuple[float, float] | None = None
    warnings: list[str] = field(default_factory=list)
    category: MetricCategory | None = None
    interpretation: str = ""
    
    def add_step(
        self,
        description: str,
        formula: str | None = None,
        values: dict[str, Any] | None = None,
        result: Any = None,
    ) -> None:
        """Add a calculation step to the audit trail."""
        step_number = len(self.calculation_steps) + 1
        step = CalculationStep(
            step_number=step_number,
            description=description,
            formula=formula,
            values=values or {},
            result=result,
        )
        self.calculation_steps.append(step)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metric_name": self.metric_name,
            "value": float(self.value) if self.value is not None else None,
            "unit": self.unit.value,
            "formula": self.formula,
            "inputs": {k: str(v) for k, v in self.inputs.items()},
            "calculation_steps": [s.to_dict() for s in self.calculation_steps],
            "is_plausible": self.is_plausible,
            "plausibility_range": list(self.plausibility_range) if self.plausibility_range else None,
            "warnings": self.warnings,
            "category": self.category.value if self.category else None,
            "interpretation": self.interpretation,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_reasoning_block(self) -> str:
        """
        Format as a reasoning block for LLM output.
        
        This matches the system prompt's required format for showing
        calculation work.
        """
        lines = [
            f"### {self.metric_name}",
            f"**Formula**: {self.formula}",
            "",
            "**Calculation Steps**:",
        ]
        
        for step in self.calculation_steps:
            lines.append(str(step))
        
        lines.append("")
        
        if self.value is not None:
            if self.unit == MetricUnit.PERCENTAGE:
                lines.append(f"**Result**: {self.value:.2f}%")
            elif self.unit == MetricUnit.RATIO:
                lines.append(f"**Result**: {self.value:.4f}x")
            elif self.unit == MetricUnit.CURRENCY:
                lines.append(f"**Result**: ${self.value:,.2f}")
            else:
                lines.append(f"**Result**: {self.value}")
        else:
            lines.append("**Result**: Unable to calculate")
        
        if not self.is_plausible and self.plausibility_range:
            lines.append(
                f"**⚠️ Warning**: Value outside plausible range "
                f"({self.plausibility_range[0]} to {self.plausibility_range[1]})"
            )
        
        for warning in self.warnings:
            lines.append(f"**⚠️ Warning**: {warning}")
        
        if self.interpretation:
            lines.append(f"**Interpretation**: {self.interpretation}")
        
        return "\n".join(lines)


@dataclass
class MetricResult(CalculationResult):
    """Extended calculation result with period information."""
    period: str = ""
    prior_value: Decimal | None = None
    change: Decimal | None = None
    change_percentage: Decimal | None = None


@dataclass
class MetricCollection:
    """
    Collection of related metrics for a category.
    """
    category: MetricCategory
    period: str
    metrics: list[CalculationResult] = field(default_factory=list)
    currency: str = "SGD"
    
    @property
    def summary(self) -> dict[str, Decimal | None]:
        """Quick access to metric values by name."""
        return {m.metric_name: m.value for m in self.metrics}
    
    @property
    def all_plausible(self) -> bool:
        """Whether all metrics are plausible."""
        return all(m.is_plausible for m in self.metrics)
    
    @property
    def warning_count(self) -> int:
        """Total number of warnings across all metrics."""
        return sum(len(m.warnings) for m in self.metrics)
    
    def add_metric(self, metric: CalculationResult) -> None:
        """Add a metric to the collection."""
        self.metrics.append(metric)
    
    def get_metric(self, name: str) -> CalculationResult | None:
        """Get a metric by name."""
        for metric in self.metrics:
            if metric.metric_name.lower() == name.lower():
                return metric
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "period": self.period,
            "currency": self.currency,
            "all_plausible": self.all_plausible,
            "warning_count": self.warning_count,
            "metrics": [m.to_dict() for m in self.metrics],
            "summary": {k: float(v) if v else None for k, v in self.summary.items()},
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_markdown_table(self) -> str:
        """Format as markdown table."""
        lines = [
            f"### {self.category.value.title()} Metrics ({self.period})",
            "",
            "| Metric | Value | Status |",
            "|--------|-------|--------|",
        ]
        
        for metric in self.metrics:
            if metric.value is not None:
                if metric.unit == MetricUnit.PERCENTAGE:
                    value_str = f"{metric.value:.2f}%"
                elif metric.unit == MetricUnit.RATIO:
                    value_str = f"{metric.value:.2f}x"
                elif metric.unit == MetricUnit.CURRENCY:
                    value_str = f"${metric.value:,.0f}"
                else:
                    value_str = str(metric.value)
            else:
                value_str = "N/A"
            
            status = "✅" if metric.is_plausible and not metric.warnings else "⚠️"
            lines.append(f"| {metric.metric_name} | {value_str} | {status} |")
        
        return "\n".join(lines)


@dataclass
class TrendAnalysisResult:
    """
    Result of multi-period trend analysis for a metric.
    """
    metric_name: str
    periods: list[str]
    values: list[Decimal | None]
    direction: TrendDirection
    growth_rate: Decimal | None = None  # CAGR or average growth
    volatility: Decimal | None = None   # Standard deviation
    interpretation: str = ""
    
    @property
    def latest_value(self) -> Decimal | None:
        """Get the most recent value."""
        for v in reversed(self.values):
            if v is not None:
                return v
        return None
    
    @property
    def earliest_value(self) -> Decimal | None:
        """Get the earliest value."""
        for v in self.values:
            if v is not None:
                return v
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "periods": self.periods,
            "values": [float(v) if v else None for v in self.values],
            "direction": self.direction.value,
            "growth_rate": float(self.growth_rate) if self.growth_rate else None,
            "volatility": float(self.volatility) if self.volatility else None,
            "interpretation": self.interpretation,
        }


@dataclass
class ConfidenceAssessment:
    """
    Confidence level assessment for analysis output.
    
    Implements the mandatory confidence scoring from the system prompt.
    """
    level: ConfidenceLevel
    justification: str
    factors: dict[str, str] = field(default_factory=dict)
    score: float = 0.0  # Internal score (0-100)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "justification": self.justification,
            "factors": self.factors,
            "score": self.score,
        }
    
    def to_string(self) -> str:
        """Format for output."""
        return f"{self.level.value} — {self.justification}"
    
    def __str__(self) -> str:
        return self.to_string()


@dataclass
class ComprehensiveAnalysisResult:
    """
    Complete analysis result containing all metrics and assessments.
    """
    analysis_type: str
    period: str
    company_name: str = ""
    currency: str = "SGD"
    
    # Results by category
    profitability: MetricCollection | None = None
    liquidity: MetricCollection | None = None
    solvency: MetricCollection | None = None
    efficiency: MetricCollection | None = None
    
    # Validation and confidence
    validation: ValidationResult | None = None
    confidence: ConfidenceAssessment | None = None
    
    # Trend analysis (if multi-period)
    trends: list[TrendAnalysisResult] = field(default_factory=list)
    
    # Overall findings
    key_findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    
    @property
    def all_metrics(self) -> list[CalculationResult]:
        """Get all calculated metrics."""
        metrics = []
        for collection in [self.profitability, self.liquidity, self.solvency, self.efficiency]:
            if collection:
                metrics.extend(collection.metrics)
        return metrics
    
    @property
    def total_warnings(self) -> int:
        """Total warning count across all metrics."""
        return sum(len(m.warnings) for m in self.all_metrics)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "analysis_type": self.analysis_type,
            "period": self.period,
            "company_name": self.company_name,
            "currency": self.currency,
            "total_warnings": self.total_warnings,
        }
        
        if self.profitability:
            result["profitability"] = self.profitability.to_dict()
        if self.liquidity:
            result["liquidity"] = self.liquidity.to_dict()
        if self.solvency:
            result["solvency"] = self.solvency.to_dict()
        if self.efficiency:
            result["efficiency"] = self.efficiency.to_dict()
        if self.validation:
            result["validation"] = self.validation.to_dict()
        if self.confidence:
            result["confidence"] = self.confidence.to_dict()
        if self.trends:
            result["trends"] = [t.to_dict() for t in self.trends]
        
        result["key_findings"] = self.key_findings
        result["recommendations"] = self.recommendations
        
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
