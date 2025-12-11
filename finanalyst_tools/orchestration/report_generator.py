# finanalyst_tools/orchestration/report_generator.py
"""
Report generation in mandatory format.

Generates reports following the exact template specified in the system prompt:

# Financial Analysis Report
**Analysis Type**: ...
**Data Period**: ...
**Confidence Level**: [HIGH | MEDIUM | LOW] — [Brief justification]

## 1. Data Validation Summary
...

## 2. Key Findings
...
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from datetime import datetime

from finanalyst_tools.models.analysis_results import (
    ComprehensiveAnalysisResult,
    ConfidenceAssessment,
    MetricCollection,
    CalculationResult,
)
from finanalyst_tools.utils.formatting import (
    format_markdown_table,
    format_percentage,
    format_ratio,
    format_currency,
)


class ReportFormat(str, Enum):
    """Supported report formats."""
    MARKDOWN = "markdown"
    JSON = "json"
    TEXT = "text"


class ReportGenerator:
    """
    Generator for financial analysis reports.
    
    Creates reports in the mandatory format with all required sections.
    """
    
    def __init__(self, include_audit_trail: bool = True):
        """
        Initialize the report generator.
        
        Args:
            include_audit_trail: Whether to include detailed calculation steps
        """
        self.include_audit_trail = include_audit_trail
    
    def generate(
        self,
        result: ComprehensiveAnalysisResult,
        format: ReportFormat = ReportFormat.MARKDOWN,
    ) -> str:
        """
        Generate a complete financial analysis report.
        
        Args:
            result: The comprehensive analysis result
            format: Output format
            
        Returns:
            Formatted report string
        """
        if format == ReportFormat.MARKDOWN:
            return self._generate_markdown(result)
        elif format == ReportFormat.JSON:
            return result.to_json()
        else:
            return self._generate_text(result)
    
    def _generate_markdown(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate Markdown format report."""
        sections = []
        
        # Header
        sections.append(self._generate_header(result))
        
        # Section 1: Data Validation Summary
        sections.append(self._generate_validation_section(result))
        
        # Section 2: Key Findings
        sections.append(self._generate_findings_section(result))
        
        # Section 3: Detailed Metrics
        sections.append(self._generate_metrics_section(result))
        
        # Section 4: Calculation Audit Trail (optional)
        if self.include_audit_trail:
            sections.append(self._generate_audit_trail_section(result))
        
        # Section 5: Recommendations
        sections.append(self._generate_recommendations_section(result))
        
        # Footer
        sections.append(self._generate_footer())
        
        return "\n\n".join(sections)
    
    def _generate_header(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate report header."""
        lines = [
            "# Financial Analysis Report",
            "",
            f"**Analysis Type**: {result.analysis_type.title()}",
            f"**Data Period**: {result.period}",
            f"**Currency**: {result.currency}",
        ]
        
        if result.confidence:
            lines.append(f"**Confidence Level**: {result.confidence.to_display()}")
        
        return "\n".join(lines)
    
    def _generate_validation_section(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate validation summary section."""
        lines = ["## 1. Data Validation Summary"]
        
        validation = result.validation_summary
        if not validation:
            lines.append("\n✅ No validation performed or all checks passed.")
            return "\n".join(lines)
        
        is_valid = validation.get("is_valid", True)
        error_count = validation.get("error_count", 0)
        warning_count = validation.get("warning_count", 0)
        
        if is_valid and warning_count == 0:
            lines.append("\n✅ All validation checks passed successfully.")
        else:
            status = "❌ Validation failed" if not is_valid else "⚠️ Validation passed with warnings"
            lines.append(f"\n{status}")
            lines.append(f"- Errors: {error_count}")
            lines.append(f"- Warnings: {warning_count}")
            
            # List errors
            errors = validation.get("errors", [])
            if errors:
                lines.append("\n**Errors:**")
                for error in errors[:5]:  # Limit to first 5
                    lines.append(f"- {error.get('field', 'Unknown')}: {error.get('message', '')}")
            
            # List warnings
            warnings = validation.get("warnings", [])
            if warnings:
                lines.append("\n**Warnings:**")
                for warning in warnings[:5]:  # Limit to first 5
                    lines.append(f"- {warning.get('field', 'Unknown')}: {warning.get('message', '')}")
        
        # Reconciliation summary
        recon = result.reconciliation_summary
        if recon:
            all_passed = recon.get("all_passed", True)
            if all_passed:
                lines.append("\n✅ All cross-statement reconciliations passed.")
            else:
                failed_count = recon.get("failed_count", 0)
                lines.append(f"\n⚠️ {failed_count} reconciliation check(s) failed.")
        
        return "\n".join(lines)
    
    def _generate_findings_section(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate key findings section."""
        lines = ["## 2. Key Findings"]
        
        if not result.metric_collections:
            lines.append("\nNo metrics calculated.")
            return "\n".join(lines)
        
        for collection in result.metric_collections:
            lines.append(f"\n### {collection.category.value.title()} Metrics")
            
            # Summary table
            headers = ["Metric", "Value", "Status"]
            rows = []
            
            for metric in collection.metrics:
                status = "✅" if metric.is_plausible else "⚠️"
                if metric.warnings:
                    status = "⚠️"
                rows.append([metric.metric_name, metric.formatted_value, status])
            
            lines.append("")
            lines.append(format_markdown_table(headers, rows))
            
            # Key observations
            observations = self._extract_observations(collection)
            if observations:
                lines.append("\n**Key Observations:**")
                for obs in observations:
                    lines.append(f"- {obs}")
        
        return "\n".join(lines)
    
    def _extract_observations(self, collection: MetricCollection) -> list[str]:
        """Extract key observations from metrics."""
        observations = []
        
        for metric in collection.metrics:
            # Add warnings as observations
            for warning in metric.warnings[:2]:  # Limit per metric
                observations.append(warning)
        
        return observations[:5]  # Limit total
    
    def _generate_metrics_section(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate detailed metrics section."""
        lines = ["## 3. Detailed Metrics"]
        
        for collection in result.metric_collections:
            lines.append(f"\n### {collection.category.value.title()}")
            
            for metric in collection.metrics:
                lines.append(f"\n#### {metric.metric_name}")
                lines.append(f"- **Value**: {metric.formatted_value}")
                lines.append(f"- **Formula**: {metric.formula}")
                
                if metric.plausibility_range:
                    min_val, max_val = metric.plausibility_range
                    lines.append(f"- **Expected Range**: {min_val:.1f} to {max_val:.1f}")
                
                status = "Within expected range" if metric.is_plausible else "Outside expected range"
                lines.append(f"- **Assessment**: {status}")
        
        return "\n".join(lines)
    
    def _generate_audit_trail_section(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate calculation audit trail section."""
        lines = ["## 4. Calculation Audit Trail"]
        
        for collection in result.metric_collections:
            for metric in collection.metrics:
                if metric.calculation_steps:
                    lines.append(f"\n### {metric.metric_name}")
                    lines.append("")
                    for step in metric.calculation_steps:
                        lines.append(f"- {step}")
                    
                    # Show inputs
                    lines.append("\n**Inputs Used:**")
                    for key, value in metric.inputs.items():
                        if value is not None:
                            lines.append(f"- {key}: {value:,.2f}" if isinstance(value, (int, float)) else f"- {key}: {value}")
        
        return "\n".join(lines)
    
    def _generate_recommendations_section(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate recommendations section."""
        lines = ["## 5. Recommendations"]
        
        if not result.recommendations:
            lines.append("\nNo specific recommendations at this time.")
            return "\n".join(lines)
        
        lines.append("")
        for i, rec in enumerate(result.recommendations, 1):
            lines.append(f"{i}. {rec}")
        
        return "\n".join(lines)
    
    def _generate_footer(self) -> str:
        """Generate report footer."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"---\n*Generated by FinAnalyst-Pro Agent Tools | {timestamp}*"
    
    def _generate_text(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate plain text format report."""
        # Simple text version - strip markdown
        md_report = self._generate_markdown(result)
        # Basic markdown stripping
        text = md_report.replace("#", "").replace("**", "").replace("*", "")
        text = text.replace("|", " | ")
        return text


def generate_financial_report(
    result: ComprehensiveAnalysisResult,
    format: ReportFormat = ReportFormat.MARKDOWN,
    include_audit_trail: bool = True,
) -> 
