# finanalyst_tools/orchestration/report_generator.py
"""
Report generator for financial analysis results.

Generates reports in the mandatory format specified in system prompt:
- Financial Analysis Report header
- Data Validation Summary
- Key Findings
- Detailed Metrics
- Calculation Audit Trail
- Recommendations
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
from datetime import datetime

from finanalyst_tools.models.analysis_results import (
    ComprehensiveAnalysisResult,
    MetricCollection,
    CalculationResult,
    ConfidenceAssessment,
)
from finanalyst_tools.utils.formatting import (
    format_currency,
    format_percentage,
    format_ratio,
    format_markdown_table,
    format_value_with_unit,
)


class ReportFormat(str, Enum):
    """Available report formats."""
    MARKDOWN = "markdown"
    JSON = "json"
    TEXT = "text"


def generate_financial_report(
    analysis_result: ComprehensiveAnalysisResult,
    format: ReportFormat = ReportFormat.MARKDOWN,
    include_audit_trail: bool = True,
) -> str:
    """
    Generate a financial analysis report.
    
    Args:
        analysis_result: Complete analysis result
        format: Output format
        include_audit_trail: Whether to include detailed calculation steps
        
    Returns:
        Formatted report string
    """
    generator = ReportGenerator(
        include_audit_trail=include_audit_trail,
    )
    
    if format == ReportFormat.MARKDOWN:
        return generator.generate_markdown(analysis_result)
    elif format == ReportFormat.JSON:
        return analysis_result.to_json()
    else:
        return generator.generate_text(analysis_result)


class ReportGenerator:
    """
    Generator for financial analysis reports.
    """
    
    def __init__(
        self,
        include_audit_trail: bool = True,
        include_warnings: bool = True,
        company_name: str | None = None,
    ):
        """
        Initialize the report generator.
        
        Args:
            include_audit_trail: Whether to include calculation steps
            include_warnings: Whether to include warning messages
            company_name: Optional company name for report header
        """
        self.include_audit_trail = include_audit_trail
        self.include_warnings = include_warnings
        self.company_name = company_name
    
    def generate_markdown(
        self,
        result: ComprehensiveAnalysisResult,
    ) -> str:
        """
        Generate report in Markdown format.
        
        Follows the mandatory template from system prompt.
        """
        sections = []
        
        # Header
        sections.append(self._generate_header(result))
        
        # Executive Summary with Confidence
        sections.append(self._generate_summary(result))
        
        # Data Validation Summary
        sections.append(self._generate_validation_section(result))
        
        # Key Findings
        sections.append(self._generate_key_findings(result))
        
        # Detailed Metrics
        sections.append(self._generate_metrics_section(result))
        
        # Audit Trail (optional)
        if self.include_audit_trail:
            sections.append(self._generate_audit_trail(result))
        
        # Recommendations
        sections.append(self._generate_recommendations(result))
        
        # Footer
        sections.append(self._generate_footer())
        
        return "\n\n".join(sections)
    
    def generate_text(
        self,
        result: ComprehensiveAnalysisResult,
    ) -> str:
        """Generate report in plain text format."""
        # Simplified version of markdown
        md = self.generate_markdown(result)
        # Remove markdown formatting
        text = md.replace("# ", "").replace("## ", "").replace("### ", "")
        text = text.replace("**", "").replace("*", "")
        text = text.replace("|", " ")
        return text
    
    def _generate_header(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate report header."""
        lines = ["# Financial Analysis Report"]
        
        if self.company_name:
            lines.append(f"\n**Company**: {self.company_name}")
        
        lines.append(f"**Analysis Type**: {result.analysis_type.title()}")
        lines.append(f"**Data Period**: {result.period}")
        lines.append(f"**Currency**: {result.currency}")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        return "\n".join(lines)
    
    def _generate_summary(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate executive summary with confidence level."""
        lines = ["## Executive Summary"]
        
        # Confidence level
        if result.confidence:
            lines.append(f"\n**Confidence Level**: {result.confidence.to_display()}")
        
        # Quick stats
        lines.append(f"\n**Metrics Calculated**: {result.total_metrics}")
        
        # Count warnings
        warning_count = len(result.all_warnings)
        if warning_count > 0:
            lines.append(f"**Warnings**: {warning_count}")
        
        return "\n".join(lines)
    
    def _generate_validation_section(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate data validation summary."""
        lines = ["## 1. Data Validation Summary"]
        
        if result.validation_summary:
            is_valid = result.validation_summary.get("is_valid", True)
            error_count = result.validation_summary.get("error_count", 0)
            warning_count = result.validation_summary.get("warning_count", 0)
            
            status = "✅ Passed" if is_valid else "❌ Failed"
            lines.append(f"\n**Status**: {status}")
            
            if error_count > 0:
                lines.append(f"- Errors: {error_count}")
            if warning_count > 0:
                lines.append(f"- Warnings: {warning_count}")
            
            if is_valid and error_count == 0 and warning_count == 0:
                lines.append("- All validation checks passed")
        else:
