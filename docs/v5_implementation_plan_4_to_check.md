# finanalyst_tools/calculations/__init__.py
```py
# finanalyst_tools/calculations/__init__.py
"""
Financial calculation functions for FinAnalyst-Pro Agent Tools.

This package provides calculation functions for:
- Profitability ratios (margins, ROA, ROE, etc.)
- Liquidity ratios (current, quick, cash ratios)
- Base infrastructure for consistent calculation handling

All calculations:
- Use Decimal precision for accuracy
- Return CalculationResult with full audit trail
- Include plausibility checks
- Handle edge cases gracefully
"""

from finanalyst_tools.calculations.base import (
    BaseCalculator,
    create_calculation_result,
    extract_decimal_value,
)

from finanalyst_tools.calculations.profitability import (
    calculate_gross_profit_margin,
    calculate_operating_profit_margin,
    calculate_net_profit_margin,
    calculate_ebitda_margin,
    calculate_return_on_assets,
    calculate_return_on_equity,
    calculate_return_on_capital_employed,
    calculate_all_profitability_metrics,
    ProfitabilityCalculator,
)

from finanalyst_tools.calculations.liquidity import (
    calculate_current_ratio,
    calculate_quick_ratio,
    calculate_cash_ratio,
    calculate_working_capital,
    calculate_all_liquidity_metrics,
    LiquidityCalculator,
)


__all__ = [
    # Base
    "BaseCalculator",
    "create_calculation_result",
    "extract_decimal_value",
    # Profitability
    "calculate_gross_profit_margin",
    "calculate_operating_profit_margin",
    "calculate_net_profit_margin",
    "calculate_ebitda_margin",
    "calculate_return_on_assets",
    "calculate_return_on_equity",
    "calculate_return_on_capital_employed",
    "calculate_all_profitability_metrics",
    "ProfitabilityCalculator",
    # Liquidity
    "calculate_current_ratio",
    "calculate_quick_ratio",
    "calculate_cash_ratio",
    "calculate_working_capital",
    "calculate_all_liquidity_metrics",
    "LiquidityCalculator",
]

```

# finanalyst_tools/dispatcher.py
```py
# finanalyst_tools/dispatcher.py
"""
Tool dispatcher for executing tool calls from LLM.

Provides:
- Parameter validation and type coercion
- Execution timing
- Error handling and formatting
- Structured result formatting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any
import json
import time

from finanalyst_tools.tool_registry import TOOL_REGISTRY, ToolDefinition, ToolParameter
from finanalyst_tools.exceptions import (
    ToolNotFoundError,
    ToolExecutionError,
    ToolParameterError,
    FinAnalystError,
)


@dataclass
class ToolCallResult:
    """
    Result of a tool execution.
    
    Attributes:
        tool_name: Name of the executed tool
        success: Whether execution succeeded
        result: The result (if successful)
        error: Error message (if failed)
        error_details: Additional error context
        execution_time_ms: Execution time in milliseconds
    """
    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    error_details: dict[str, Any] | None = None
    execution_time_ms: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data: dict[str, Any] = {
            "tool_name": self.tool_name,
            "success": self.success,
        }
        
        if self.success:
            if hasattr(self.result, "to_dict"):
                data["result"] = self.result.to_dict()
            else:
                data["result"] = self.result
        else:
            data["error"] = self.error
            if self.error_details:
                data["error_details"] = self.error_details
        
        if self.execution_time_ms is not None:
            data["execution_time_ms"] = round(self.execution_time_ms, 2)
        
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class ToolDispatcher:
    """
    Dispatcher for executing tool calls.
    
    Handles:
    - Tool lookup
    - Parameter validation
    - Type coercion (string → Decimal for numbers)
    - Execution with timing
    - Error handling
    """
    
    def __init__(self):
        """Initialize the dispatcher."""
        self.registry = TOOL_REGISTRY
    
    def execute(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ToolCallResult:
        """
        Execute a tool call.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Dictionary of parameters
            
        Returns:
            ToolCallResult with execution outcome
        """
        start_time = time.perf_counter()
        
        try:
            # Get tool definition
            tool = self.registry.get(tool_name)
            if tool is None:
                available = self.registry.list_tool_names()
                raise ToolNotFoundError(tool_name, available)
            
            # Validate parameters
            validated_params = self._validate_and_coerce_parameters(tool, parameters)
            
            # Execute the tool
            if tool.function is None:
                raise ToolExecutionError(
                    tool_name=tool_name,
                    original_error=ValueError("Tool function not registered"),
                )
            
            result = tool.function(**validated_params)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return ToolCallResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=execution_time,
            )
            
        except FinAnalystError as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                error_details=e.to_dict(),
                execution_time_ms=execution_time,
            )
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=f"Unexpected error: {str(e)}",
                error_details={
                    "error_type": type(e).__name__,
                    "message": str(e),
                },
                execution_time_ms=execution_time,
            )
    
    def _validate_and_coerce_parameters(
        self,
        tool: ToolDefinition,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate and coerce parameters for a tool.
        
        Args:
            tool: Tool definition
            parameters: Raw parameters
            
        Returns:
            Validated and coerced parameters
            
        Raises:
            ToolParameterError: If validation fails
        """
        result = {}
        
        # Check required parameters
        for param in tool.parameters:
            if param.required and param.name not in parameters:
                raise ToolParameterError(
                    tool_name=tool.name,
                    parameter_name=param.name,
                    message="Required parameter is missing",
                    expected_type=param.type,
                )
        
        # Validate and coerce each provided parameter
        for param in tool.parameters:
            if param.name in parameters:
                value = parameters[param.name]
                coerced = self._coerce_parameter(tool.name, param, value)
                result[param.name] = coerced
            elif param.default is not None:
                result[param.name] = param.default
        
        return result
    
    def _coerce_parameter(
        self,
        tool_name: str,
        param: ToolParameter,
        value: Any,
    ) -> Any:
        """
        Coerce a parameter value to the expected type.
        
        Args:
            tool_name: Tool name (for error messages)
            param: Parameter definition
            value: Raw value
            
        Returns:
            Coerced value
            
        Raises:
            ToolParameterError: If coercion fails
        """
        if value is None:
            if param.required:
                raise ToolParameterError(
                    tool_name=tool_name,
                    parameter_name=param.name,
                    message="Value cannot be None",
                    expected_type=param.type,
                )
            return param.default
        
        try:
            if param.type == "number":
                # Convert to Decimal for financial precision
                if isinstance(value, Decimal):
                    return value
                if isinstance(value, (int, float)):
                    return Decimal(str(value))
                if isinstance(value, str):
                    return Decimal(value)
                raise ValueError(f"Cannot convert {type(value).__name__} to number")
                
            elif param.type == "integer":
                return int(value)
                
            elif param.type == "boolean":
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes")
                return bool(value)
                
            elif param.type == "string":
                return str(value)
                
            elif param.type == "object":
                if isinstance(value, dict):
                    return value
                if isinstance(value, str):
                    return json.loads(value)
                raise ValueError("Expected object/dictionary")
                
            elif param.type == "array":
                if isinstance(value, list):
                    return value
                if isinstance(value, str):
                    return json.loads(value)
                raise ValueError("Expected array/list")
                
            else:
                # Unknown type - pass through
                return value
                
        except (ValueError, InvalidOperation, json.JSONDecodeError) as e:
            raise ToolParameterError(
                tool_name=tool_name,
                parameter_name=param.name,
                message=f"Cannot convert to {param.type}: {str(e)}",
                expected_type=param.type,
                actual_value=value,
            )
    
    def list_tools(self) -> list[str]:
        """List all available tool names."""
        return self.registry.list_tool_names()
    
    def get_tool_info(self, tool_name: str) -> dict[str, Any] | None:
        """Get information about a specific tool."""
        tool = self.registry.get(tool_name)
        if tool:
            return tool.to_dict()
        return None


# Global singleton instance
DISPATCHER = ToolDispatcher()


def execute_tool(tool_name: str, parameters: dict[str, Any]) -> ToolCallResult:
    """
    Convenience function to execute a tool.
    
    Args:
        tool_name: Name of the tool
        parameters: Tool parameters
        
    Returns:
        ToolCallResult
    """
    return DISPATCHER.execute(tool_name, parameters)

```

# finanalyst_tools/orchestration/__init__.py
```py
# finanalyst_tools/orchestration/__init__.py
"""
Orchestration layer for FinAnalyst-Pro Agent Tools.

This package provides:
- Analysis pipeline with 5-phase workflow
- Confidence scoring for analysis results
- Report generation with mandatory template
"""

from finanalyst_tools.orchestration.pipeline import (
    AnalysisPipeline,
    AnalysisRequest,
    AnalysisPhase,
)

from finanalyst_tools.orchestration.confidence_scorer import (
    calculate_confidence_level,
    ConfidenceScorer,
)

from finanalyst_tools.orchestration.report_generator import (
    generate_financial_report,
    ReportGenerator,
    ReportFormat,
)


__all__ = [
    # Pipeline
    "AnalysisPipeline",
    "AnalysisRequest",
    "AnalysisPhase",
    # Confidence
    "calculate_confidence_level",
    "ConfidenceScorer",
    # Reporting
    "generate_financial_report",
    "ReportGenerator",
    "ReportFormat",
]

```

# finanalyst_tools/orchestration/confidence_scorer.py
```py
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

```

# finanalyst_tools/orchestration/pipeline.py
```py
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
        
        # Phase 1: VALIDATE
        self._phase_validate(request)
        if not self.state.validation_result.can_proceed:
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
        
        self.state.phase_completed[AnalysisPhase.VALIDATE] = True
    
    def _phase_analyze(self, request: AnalysisRequest) -> dict[str, bool]:
        """
        Phase 2: ANALYZE
        
        Determine what calculations to perform based on:
        - Analysis type requested
        - Data available
        
        Returns:
            Dictionary of metric categories to calculate
        """
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
        
        self.state.phase_completed[AnalysisPhase.ANALYZE] = True
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
        
        self.state.phase_completed[AnalysisPhase.CALCULATE] = True
    
    def _phase_interpret(self, request: AnalysisRequest) -> None:
        """
        Phase 4: INTERPRET
        
        Add context and insights to calculated metrics.
        """
        self.state.current_phase = AnalysisPhase.INTERPRET
        
        # Plausibility checks on all metrics
        plausibility = check_all_plausibility(self.state.all_metrics)
        self.state.plausibility_result = plausibility
        
        # Add warnings for implausible values
        for check in plausibility.implausible_checks:
            self.state.warnings.append(f"Plausibility: {check.message}")
        
        self.state.phase_completed[AnalysisPhase.INTERPRET] = True
    
    def _phase_verify(self, request: AnalysisRequest) -> None:
        """
        Phase 5: VERIFY
        
        Pre-delivery checks:
        - Ensure all requested calculations completed
        - Verify no critical errors
        - Final quality check
        """
        self.state.current_phase = AnalysisPhase.VERIFY
        
        # Check that calculations were performed
        if not self.state.metric_collections:
            self.state.warnings.append("No metrics were calculated")
        
        # Check for any uncalculable metrics
        uncalculable = [m for m in self.state.all_metrics if m.value is None]
        if uncalculable:
            for m in uncalculable:
                self.state.warnings.append(f"Could not calculate: {m.metric_name}")
        
        self.state.phase_completed[AnalysisPhase.VERIFY] = True
    
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
        
        if self.state.validation_result:
            result.validation_summary = self.state.validation_result.to_dict()
        
        result.recommendations = [
            f"Analysis could not be completed: {error_message}",
            "Please address validation errors and retry",
        ]
        
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

```

# finanalyst_tools/orchestration/report_generator.py
```py
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

```

