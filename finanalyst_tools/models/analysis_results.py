# File: finanalyst_tools/models/analysis_results.py
"""
Structured result models for calculations and analysis.

Provides:
- CalculationResult: Complete audit trail for single metric
- MetricCollection: Group of related metrics
- TrendAnalysisResult: Multi-period trend analysis output
- ConfidenceAssessment: Confidence level with justification
- ComprehensiveAnalysisResult: Complete analysis output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any
import json

from finanalyst_tools.models.financial_statements import FinancialPeriod
from finanalyst_tools.utils.serialization import to_jsonable


class MetricUnit(str, Enum):
    """Units for financial metrics."""
    PERCENTAGE = "percentage"
    RATIO = "ratio"
    CURRENCY = "currency"
    DAYS = "days"
    COUNT = "count"
    TIMES = "times"


class MetricCategory(str, Enum):
    """Categories for financial metrics."""
    PROFITABILITY = "profitability"
    LIQUIDITY = "liquidity"
    SOLVENCY = "solvency"
    EFFICIENCY = "efficiency"
    GROWTH = "growth"
    VALUATION = "valuation"


class TrendDirection(str, Enum):
    """Direction of a trend."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


class ConfidenceLevel(str, Enum):
    """Confidence levels for analysis results."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class CalculationResult:
    """
    Complete result of a single metric calculation with audit trail.
    
    Provides:
    - The calculated value with unit
    - Formula used
    - All input values
    - Step-by-step calculation log
    - Plausibility assessment
    - Warnings for unusual values
    """
    metric_name: str
    value: Decimal | None
    unit: MetricUnit
    formula: str
    inputs: dict[str, Any] = field(default_factory=dict)
    calculation_steps: list[str] = field(default_factory=list)
    is_plausible: bool = True
    plausibility_range: tuple[float, float] | None = None
    warnings: list[str] = field(default_factory=list)
    category: MetricCategory | None = None
    currency: str = "USD"  # Currency code for CURRENCY unit type
    
    def add_step(self, step: str) -> None:
        """Add a calculation step to the audit trail."""
        step_num = len(self.calculation_steps) + 1
        self.calculation_steps.append(f"Step {step_num}: {step}")
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    @property
    def is_calculable(self) -> bool:
        """Whether the calculation was successful."""
        return self.value is not None
    
    @property
    def formatted_value(self) -> str:
        """Get formatted value based on unit."""
        if self.value is None:
            return "N/A"
        
        if self.unit == MetricUnit.PERCENTAGE:
            return f"{float(self.value):.2f}%"
        elif self.unit == MetricUnit.RATIO:
            return f"{float(self.value):.4f}"
        elif self.unit == MetricUnit.CURRENCY:
            # Use currency-specific symbol, fallback to code
            symbols = {"USD": "$", "SGD": "S$", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥"}
            symbol = symbols.get(self.currency, f"{self.currency} ")
            return f"{symbol}{float(self.value):,.2f}"
        elif self.unit == MetricUnit.DAYS:
            return f"{int(self.value)} days"
        elif self.unit == MetricUnit.TIMES:
            return f"{float(self.value):.2f}x"
        return f"{float(self.value):.2f}"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return to_jsonable({
            "metric_name": self.metric_name,
            "value": self.value,
            "formatted_value": self.formatted_value,
            "unit": self.unit.value,
            "formula": self.formula,
            "inputs": self.inputs,
            "calculation_steps": self.calculation_steps,
            "is_plausible": self.is_plausible,
            "plausibility_range": self.plausibility_range,
            "warnings": self.warnings,
            "category": self.category.value if self.category else None,
        })
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def to_reasoning_block(self) -> str:
        """
        Format as reasoning block for LLM output.
        
        Returns formatted block matching system prompt requirements.
        """
        lines = [
            f"### {self.metric_name}",
            f"**Value**: {self.formatted_value}",
            f"**Formula**: {self.formula}",
            "",
            "**Calculation Steps**:",
        ]
        
        for step in self.calculation_steps:
            lines.append(f"  - {step}")
        
        lines.append("")
        lines.append("**Inputs Used**:")
        for key, val in self.inputs.items():
            if isinstance(val, Decimal):
                lines.append(f"  - {key}: {float(val):,.2f}")
            else:
                lines.append(f"  - {key}: {val}")
        
        if self.warnings:
            lines.append("")
            lines.append("**Warnings**:")
            for warning in self.warnings:
                lines.append(f"  - ⚠️ {warning}")
        
        plausibility_status = "✅ Within range" if self.is_plausible else "⚠️ Outside expected range"
        if self.plausibility_range:
            lines.append(f"\n**Plausibility**: {plausibility_status} ({self.plausibility_range[0]} to {self.plausibility_range[1]})")
        
        return "\n".join(lines)


@dataclass
class MetricResult(CalculationResult):
    """Extended CalculationResult with period information."""
    period: FinancialPeriod | None = None


@dataclass
class MetricCollection:
    """
    Collection of related metrics.
    
    Groups metrics by category with summary statistics.
    """
    category: MetricCategory
    period: FinancialPeriod  # Normalized to FinancialPeriod only
    metrics: list[CalculationResult] = field(default_factory=list)
    
    def add_metric(self, metric: CalculationResult) -> None:
        """Add a metric to the collection."""
        self.metrics.append(metric)
    
    @property
    def metric_count(self) -> int:
        """Number of metrics in collection."""
        return len(self.metrics)
    
    @property
    def calculable_count(self) -> int:
        """Number of successfully calculated metrics."""
        return sum(1 for m in self.metrics if m.is_calculable)
    
    @property
    def plausible_count(self) -> int:
        """Number of plausible metrics."""
        return sum(1 for m in self.metrics if m.is_plausible)
    
    @property
    def summary(self) -> dict[str, Decimal | None]:
        """Quick access to metric values by name."""
        return {m.metric_name: m.value for m in self.metrics}
    
    @property
    def all_warnings(self) -> list[str]:
        """Aggregate all warnings from metrics."""
        warnings = []
        for metric in self.metrics:
            for warning in metric.warnings:
                warnings.append(f"{metric.metric_name}: {warning}")
        return warnings
    
    def get_metric(self, name: str) -> CalculationResult | None:
        """Get a specific metric by name."""
        for metric in self.metrics:
            if metric.metric_name.lower() == name.lower():
                return metric
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "category": self.category.value,
            "period": str(self.period),
            "metric_count": self.metric_count,
            "calculable_count": self.calculable_count,
            "plausible_count": self.plausible_count,
            "metrics": [m.to_dict() for m in self.metrics],
            "warnings": self.all_warnings,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def to_table(self) -> str:
        """Format as Markdown table."""
        lines = [
            f"## {self.category.value.title()} Metrics ({self.period})",
            "",
            "| Metric | Value | Plausible | Warnings |",
            "|--------|-------|-----------|----------|",
        ]
        
        for metric in self.metrics:
            plausible = "✅" if metric.is_plausible else "⚠️"
            warning_count = len(metric.warnings)
            warning_text = f"{warning_count} warning(s)" if warning_count > 0 else "None"
            lines.append(f"| {metric.metric_name} | {metric.formatted_value} | {plausible} | {warning_text} |")
        
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
    def period_count(self) -> int:
        """Number of periods analyzed."""
        return len(self.periods)
    
    @property
    def valid_value_count(self) -> int:
        """Number of periods with valid values."""
        return sum(1 for v in self.values if v is not None)
    
    @property
    def first_value(self) -> Decimal | None:
        """First valid value in series."""
        for v in self.values:
            if v is not None:
                return v
        return None
    
    @property
    def last_value(self) -> Decimal | None:
        """Last valid value in series."""
        for v in reversed(self.values):
            if v is not None:
                return v
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return to_jsonable({
            "metric_name": self.metric_name,
            "periods": self.periods,
            "values": self.values,
            "direction": self.direction.value,
            "growth_rate": self.growth_rate,
            "volatility": self.volatility,
            "interpretation": self.interpretation,
        })
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


@dataclass
class ConfidenceAssessment:
    """
    Confidence level assessment for analysis results.
    
    Implements the mandatory confidence scoring from system prompt.
    """
    level: ConfidenceLevel
    justification: str
    factors: dict[str, str] = field(default_factory=dict)
    score: float = 100.0  # Internal score (0-100)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "level": self.level.value,
            "justification": self.justification,
            "factors": self.factors,
            "score": self.score,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_display(self) -> str:
        """Format for display in report."""
        return f"**{self.level.value}** — {self.justification}"


@dataclass
class ComprehensiveAnalysisResult:
    """
    Complete analysis result combining all components.
    
    This is the top-level result returned by the analysis pipeline.
    """
    analysis_type: str
    period: str
    currency: str
    metric_collections: list[MetricCollection] = field(default_factory=list)
    trend_analyses: list[TrendAnalysisResult] = field(default_factory=list)
    confidence: ConfidenceAssessment | None = None
    validation_summary: dict[str, Any] = field(default_factory=dict)
    reconciliation_summary: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    pipeline_errors: list[str] = field(default_factory=list)
    pipeline_warnings: list[str] = field(default_factory=list)
    is_partial: bool = False
    uncalculable_metrics: list[str] = field(default_factory=list)
    
    @property
    def total_metrics(self) -> int:
        """Total number of metrics calculated."""
        return sum(mc.metric_count for mc in self.metric_collections)
    
    @property
    def all_metrics(self) -> list[CalculationResult]:
        """Flatten all metrics from all collections."""
        metrics = []
        for collection in self.metric_collections:
            metrics.extend(collection.metrics)
        return metrics
    
    @property
    def all_warnings(self) -> list[str]:
        """Aggregate all warnings from all collections."""
        warnings = []
        for collection in self.metric_collections:
            warnings.extend(collection.all_warnings)
        return warnings
    
    def get_metric(self, name: str) -> CalculationResult | None:
        """Get a specific metric by name from any collection."""
        for collection in self.metric_collections:
            metric = collection.get_metric(name)
            if metric:
                return metric
        return None
    
    def add_collection(self, collection: MetricCollection) -> None:
        """Add a metric collection."""
        self.metric_collections.append(collection)
    
    def add_trend(self, trend: TrendAnalysisResult) -> None:
        """Add a trend analysis result."""
        self.trend_analyses.append(trend)
    
    def add_recommendation(self, recommendation: str) -> None:
        """Add a recommendation."""
        self.recommendations.append(recommendation)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "analysis_type": self.analysis_type,
            "period": self.period,
            "currency": self.currency,
            "total_metrics": self.total_metrics,
            "metric_collections": [mc.to_dict() for mc in self.metric_collections],
            "trend_analyses": [ta.to_dict() for ta in self.trend_analyses],
            "confidence": self.confidence.to_dict() if self.confidence else None,
            "validation_summary": self.validation_summary,
            "reconciliation_summary": self.reconciliation_summary,
            "recommendations": self.recommendations,
            "pipeline_errors": self.pipeline_errors,
            "pipeline_warnings": self.pipeline_warnings,
            "is_partial": self.is_partial,
            "uncalculable_metrics": self.uncalculable_metrics,
            "warnings": self.all_warnings,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
