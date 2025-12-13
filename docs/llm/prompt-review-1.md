Awesome understanding! Now, please meticulously review and analyze the codebase files attached below. Due to the length, I will share the codebase files in this and the next three prompts. Wait for me to tell you that all the codebase files have been uploaded before you begin your consolidated review and analysis.

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

# finanalyst_tools/calculations/base.py
```py
# finanalyst_tools/calculations/base.py
"""
Base classes and utilities for financial calculations.

Provides:
- BaseCalculator abstract class for consistent calculation patterns
- Factory function for creating CalculationResult objects with full compatibility
- Helper functions for extracting values from financial statements
- Advanced decimal handling and validation utilities
- Complete integration with ValidationResult system
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal, getcontext
from typing import Any, Callable, Optional, cast

from finanalyst_tools.config import (
    DECIMAL_PLACES,
    PlausibilityRanges,
    METRIC_FORMULAS,
    METRIC_UNITS,
)
from finanalyst_tools.models.analysis_results import (
    CalculationResult,
    MetricUnit,
    MetricCategory,
    MetricCollection,
)
from finanalyst_tools.models.financial_statements import (
    IncomeStatementData,
    BalanceSheetData,
    CashFlowStatementData,
    FinancialStatementSet,
)
from finanalyst_tools.utils.math_ops import (
    to_decimal,
    safe_divide,
    round_decimal,
    calculate_percentage,
    calculate_average,
    is_effectively_zero,
)
from finanalyst_tools.models.validation import (
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
)
from finanalyst_tools.exceptions import (
    CalculationError,
    DivisionByZeroError,
    InvalidInputError,
)


# Set high precision for financial calculations
getcontext().prec = 28


def get_metric_unit(metric_name: str) -> MetricUnit:
    """
    Get the appropriate unit for a metric.
    
    Args:
        metric_name: Name of the metric
        
    Returns:
        MetricUnit enum value
    """
    unit_str = METRIC_UNITS.get(metric_name.lower(), "ratio")
    unit_map = {
        "percentage": MetricUnit.PERCENTAGE,
        "ratio": MetricUnit.RATIO,
        "currency": MetricUnit.CURRENCY,
        "days": MetricUnit.DAYS,
        "count": MetricUnit.COUNT,
        "times": MetricUnit.TIMES,
    }
    return unit_map.get(unit_str, MetricUnit.RATIO)


def get_metric_formula(metric_name: str) -> str:
    """
    Get the formula for a metric.
    
    Args:
        metric_name: Name of the metric
        
    Returns:
        Formula string
    """
    return METRIC_FORMULAS.get(metric_name.lower(), "N/A")


def _format_reasoning_block(result: CalculationResult) -> str:
    """
    Format a CalculationResult as a reasoning block for LLM output.
    
    This implements the required format from the system prompt.
    
    Args:
        result: Calculation result to format
        
    Returns:
        Formatted markdown block
    """
    lines = [
        f"### {result.metric_name}",
        f"**Value**: {result.formatted_value}",
        f"**Formula**: {result.formula}",
        "",
        "**Calculation Steps**:",
    ]
    
    for step in result.calculation_steps:
        lines.append(f"  - {step}")
    
    lines.append("")
    lines.append("**Inputs Used**:")
    for key, val in result.inputs.items():
        if isinstance(val, Decimal):
            lines.append(f"  - {key}: {float(val):,.2f}")
        else:
            lines.append(f"  - {key}: {val}")
    
    if result.warnings:
        lines.append("")
        lines.append("**Warnings**:")
        for warning in result.warnings:
            lines.append(f"  - ⚠️ {warning}")
    
    plausibility_status = "✅ Within range" if result.is_plausible else "⚠️ Outside expected range"
    if result.plausibility_range:
        lines.append(f"\n**Plausibility**: {plausibility_status} ({result.plausibility_range[0]:.1f} to {result.plausibility_range[1]:.1f})")
    
    return "\n".join(lines)


def create_calculation_result(
    metric_name: str,
    value: Decimal | None,
    formula: str,
    inputs: dict[str, Any],
    steps: list[str],
    category: MetricCategory | None = None,
    warnings: list[str] | None = None,
    unit: MetricUnit | None = None,
) -> CalculationResult:
    """
    Factory function to create a CalculationResult with full compatibility.
    
    This function now creates CalculationResult objects that are fully compatible
    with the system prompt requirements, including proper formatting methods.
    
    Args:
        metric_name: Name of the metric
        value: Calculated value (or None if calculation failed)
        formula: Formula used for calculation
        inputs: Dictionary of input values used
        steps: List of calculation steps for audit trail
        category: Metric category
        warnings: List of warning messages
        unit: Override metric unit if needed
        
    Returns:
        Complete CalculationResult with plausibility assessment
    """
    # Get plausibility range
    plausibility_range = PlausibilityRanges.get_range(metric_name)
    
    # Check plausibility
    is_plausible = True
    warning_list = warnings.copy() if warnings else []
    
    if value is not None and plausibility_range is not None:
        float_value = float(value)
        is_plausible = plausibility_range[0] <= float_value <= plausibility_range[1]
        
        if not is_plausible:
            warning_list.append(
                f"Value {float_value:.2f} is outside typical range "
                f"({plausibility_range[0]:.1f} to {plausibility_range[1]:.1f})"
            )
    
    # Determine unit if not provided
    if unit is None:
        unit = get_metric_unit(metric_name)
    
    # Preserve inputs as-is (Decimal-safe). JSON serialization is handled at output boundaries.
    serializable_inputs = dict(inputs)
    
    # Build result with all required properties
    result = CalculationResult(
        metric_name=metric_name,
        value=value,
        unit=unit,
        formula=formula,
        inputs=serializable_inputs,
        calculation_steps=steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=warning_list,
        category=category,
    )
    
    # Add the to_reasoning_block method dynamically
    result.to_reasoning_block = lambda: _format_reasoning_block(result)  # type: ignore
    
    return result


def convert_exception_to_validation_result(
    exc: Exception,
    field: str = "calculation",
    context: str = "financial calculation"
) -> ValidationResult:
    """
    Convert an exception to a ValidationResult object.
    
    This bridges the exception hierarchy with the validation system.
    
    Args:
        exc: Exception to convert
        field: Field name for validation issue
        context: Context description for error message
        
    Returns:
        ValidationResult with the error
    """
    result = ValidationResult()
    
    # Handle different exception types
    if isinstance(exc, DivisionByZeroError):
        severity = ValidationSeverity.ERROR
        message = f"Division by zero in {context}: {exc.message}"
        suggestion = exc.suggestion or "Check denominator values are non-zero"
    elif isinstance(exc, InvalidInputError):
        severity = ValidationSeverity.ERROR
        message = f"Invalid input in {context}: {exc.message}"
        suggestion = exc.suggestion or "Verify input data format and values"
    elif isinstance(exc, CalculationError):
        severity = ValidationSeverity.ERROR
        message = f"Calculation error in {context}: {exc.message}"
        suggestion = exc.suggestion or "Review calculation inputs and formula"
    else:
        severity = ValidationSeverity.ERROR
        message = f"Unexpected error in {context}: {str(exc)}"
        suggestion = "Contact support for assistance"
    
    # Get details from exception if available
    details = {}
    if hasattr(exc, 'details'):
        details = getattr(exc, 'details', {})
    elif hasattr(exc, '__dict__'):
        details = exc.__dict__
    
    # Create validation issue
    issue = ValidationIssue(
        field=field,
        message=message,
        severity=severity,
        actual_value=str(details.get('actual_value', 'unknown')),
        expected=str(details.get('expected', 'valid numeric value')),
        suggestion=suggestion
    )
    
    result.add_issue(issue)
    return result


def extract_value(
    data: dict[str, Any] | IncomeStatementData | BalanceSheetData | CashFlowStatementData,
    field_name: str,
    default: Decimal | None = None,
) -> Decimal:
    """
    Extract a value from financial data, handling both dict and model inputs.
    
    Args:
        data: Financial data (dict or Pydantic model)
        field_name: Name of the field to extract
        default: Default value if field not found
        
    Returns:
        Decimal value
    """
    if default is None:
        default = Decimal("0")
    
    # Handle different data types
    if hasattr(data, "model_dump"):
        # Pydantic model
        data_dict = data.model_dump(by_alias=False)
    elif hasattr(data, "__dict__"):
        # Regular object
        data_dict = data.__dict__
    else:
        # Dictionary or other type
        data_dict = data
    
    # Try to get the value
    value = data_dict.get(field_name)
    
    if value is None:
        return default
    
    return to_decimal(value, default=default)


def extract_decimal_value(
    value: Any,
    field_name: str = "value",
    default: Decimal | None = None,
) -> Decimal:
    if value is None:
        return default if default is not None else Decimal("0")

    if isinstance(value, Decimal):
        return value

    if isinstance(value, bool):
        raise InvalidInputError(
            f"Invalid boolean for {field_name}",
            field_name=field_name,
            actual_value=value,
            expected="numeric",
        )

    if isinstance(value, int):
        return Decimal(value)

    if isinstance(value, float):
        return Decimal(str(value))

    if isinstance(value, str):
        return Decimal(value)

    raise InvalidInputError(
        f"Cannot convert {type(value).__name__} to Decimal for {field_name}",
        field_name=field_name,
        actual_value=value,
        expected="numeric",
    )


def validate_calculation_inputs(inputs: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate calculation inputs for common issues.
    
    Args:
        inputs: Dictionary of input values
        
    Returns:
        Tuple of (is_valid, list of warnings)
    """
    warnings = []
    is_valid = True
    
    # Check for negative values where they shouldn't be
    for key, value in inputs.items():
        if isinstance(value, (int, float, Decimal)):
            # Check for negative revenue, assets, etc.
            negative_indicators = [
                "revenue", "sales", "assets", "equity", "cash",
                "income", "profit", "margin", "ratio"
            ]
            if any(indicator in key.lower() for indicator in negative_indicators):
                if value < 0:
                    warnings.append(f"Negative value for {key}: {value}")
                    # Don't mark as invalid, but warn
    
    return is_valid, warnings


class BaseCalculator(ABC):
    """
    Abstract base class for financial calculators.
    
    Provides common functionality:
    - Consistent result creation
    - Step logging
    - Plausibility checking
    - Warning accumulation
    - Input validation
    - Exception handling integration
    """
    
    def __init__(self, category: MetricCategory):
        """
        Initialize the calculator.
        
        Args:
            category: The category of metrics this calculator produces
        """
        self.category = category
        self._current_steps: list[str] = []
        self._current_warnings: list[str] = []
        self._current_inputs: dict[str, Any] = {}
        self._step_counter: int = 1
    
    def _reset(self) -> None:
        """Reset calculation state for a new calculation."""
        self._current_steps = []
        self._current_warnings = []
        self._current_inputs = {}
        self._step_counter = 1
    
    def _add_step(self, step_description: str, value: Optional[Any] = None) -> None:
        """
        Add a calculation step with automatic numbering.
        
        Args:
            step_description: Description of the step
            value: Optional value to display
        """
        step_text = f"Step {self._step_counter}: {step_description}"
        if value is not None:
            if isinstance(value, Decimal):
                step_text += f" = {value:,.4f}"
            else:
                step_text += f" = {value}"
        self._current_steps.append(step_text)
        self._step_counter += 1
    
    def _add_warning(self, warning: str) -> None:
        """
        Add a warning message.
        
        Args:
            warning: Warning message to add
        """
        self._current_warnings.append(warning)
    
    def _record_input(self, name: str, value: Any) -> None:
        """
        Record an input value for audit trail.
        
        Args:
            name: Input name
            value: Input value
        """
        self._current_inputs[name] = value
    
    def _validate_inputs(self, inputs: dict[str, Any]) -> None:
        """
        Validate inputs and add warnings if needed.
        
        Args:
            inputs: Input values to validate
        """
        _, warnings = validate_calculation_inputs(inputs)
        for warning in warnings:
            self._add_warning(warning)
    
    def _safe_divide(
        self,
        numerator: Decimal,
        denominator: Decimal,
        description: str,
        precision: int = None,
    ) -> Decimal | None:
        """
        Perform safe division with automatic step logging and error handling.
        
        Args:
            numerator: Numerator value
            denominator: Denominator value
            description: Description of the division
            precision: Decimal places for result (uses config if None)
            
        Returns:
            Result or None if division by zero
        """
        if precision is None:
            precision = DECIMAL_PLACES["ratio"]
        
        if is_effectively_zero(denominator):
            self._add_step(f"{description}: Cannot calculate (denominator is zero)")
            self._add_warning(f"Division by zero in {description}")
            return None
        
        result = safe_divide(numerator, denominator, precision)
        self._add_step(f"{description}", result)
        return result
    
    def _handle_calculation_error(
        self,
        error: Exception,
        metric_name: str,
        formula: str,
        inputs: dict[str, Any],
    ) -> CalculationResult:
        """
        Handle calculation errors gracefully and return a proper result.
        
        Args:
            error: Exception that occurred
            metric_name: Name of the metric
            formula: Formula used
            inputs: Input values
            
        Returns:
            CalculationResult with error information
        """
        # Convert exception to validation result
        validation_result = convert_exception_to_validation_result(
            error,
            field=metric_name,
            context=f"{metric_name} calculation"
        )
        
        # Get unit for the metric
        unit = get_metric_unit(metric_name)
        
        # Create result with error information
        return create_calculation_result(
            metric_name=metric_name,
            value=None,
            formula=formula,
            inputs=inputs,
            steps=self._current_steps,
            category=self.category,
            warnings=[issue.message for issue in validation_result.all_issues] + self._current_warnings,
            unit=unit,
        )
    
    def _create_result(
        self,
        metric_name: str,
        value: Decimal | None,
        formula: str,
        unit: MetricUnit | None = None,
    ) -> CalculationResult:
        """
        Create a calculation result with current state.
        
        Args:
            metric_name: Name of the metric
            value: Calculated value
            formula: Formula used
            unit: Override metric unit
            
        Returns:
            Complete CalculationResult
        """
        return create_calculation_result(
            metric_name=metric_name,
            value=value,
            formula=formula,
            inputs=self._current_inputs.copy(),
            steps=self._current_steps.copy(),
            category=self.category,
            warnings=self._current_warnings.copy(),
            unit=unit,
        )
    
    @abstractmethod
    def calculate_all(
        self,
        statement_set: FinancialStatementSet,
        prior_statement_set: FinancialStatementSet | None = None,
    ) -> MetricCollection:
        """
        Calculate all metrics for this category.
        
        Args:
            statement_set: Current period financial statements
            prior_statement_set: Prior period statements (for averages)
            
        Returns:
            MetricCollection with all calculated metrics
        """
        pass

```

# finanalyst_tools/calculations/liquidity.py
```py
# finanalyst_tools/calculations/liquidity.py
"""
Liquidity ratio calculations.

Provides calculations for:
- Current Ratio
- Quick Ratio (Acid Test)
- Cash Ratio
- Working Capital

All functions return CalculationResult with complete audit trail, detailed step-by-step calculations, and comprehensive validation.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from finanalyst_tools.config import DECIMAL_PLACES, METRIC_FORMULAS
from finanalyst_tools.models.analysis_results import (
    CalculationResult,
    MetricUnit,
    MetricCategory,
    MetricCollection,
)
from finanalyst_tools.models.financial_statements import (
    BalanceSheetData,
    FinancialStatementSet,
)
from finanalyst_tools.utils.math_ops import (
    to_decimal,
    safe_divide,
    round_decimal,
    is_effectively_zero,
)
from finanalyst_tools.calculations.base import (
    BaseCalculator,
    create_calculation_result,
    extract_value,
)


# ============================================================================
# STANDALONE CALCULATION FUNCTIONS
# ============================================================================

def calculate_current_ratio(
    current_assets: Decimal | float | int,
    current_liabilities: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Current Ratio.
    
    Formula: Current Assets / Current Liabilities
    
    Interpretation:
    - < 1.0: May have difficulty meeting short-term obligations
    - 1.0 - 2.0: Generally healthy
    - > 3.0: May have excess assets not being utilized efficiently
    
    Args:
        current_assets: Total current assets
        current_liabilities: Total current liabilities
        
    Returns:
        CalculationResult with current ratio
    """
    steps = []
    warnings = []
    
    # Convert inputs to Decimal
    ca = to_decimal(current_assets)
    cl = to_decimal(current_liabilities)
    
    # Record inputs
    inputs = {
        "current_assets": ca,
        "current_liabilities": cl,
    }
    
    # Step 1: Input validation
    steps.append(f"Step 1: Input Validation")
    steps.append(f"  Current Assets: {ca:,.2f}")
    steps.append(f"  Current Liabilities: {cl:,.2f}")
    
    if ca < 0:
        warnings.append("Negative current assets detected")
    if cl < 0:
        warnings.append("Negative current liabilities detected")
    
    # Step 2: Calculate ratio
    if is_effectively_zero(cl):
        steps.append("Step 2: Cannot calculate ratio (current liabilities is zero)")
        warnings.append("Current liabilities is zero - ratio undefined, indicates no short-term debt")
        value = None
    else:
        ratio = safe_divide(ca, cl, DECIMAL_PLACES["ratio"])
        steps.append(f"Step 2: Calculate Current Ratio = Current Assets / Current Liabilities = {ca:,.2f} / {cl:,.2f} = {ratio:.4f}")
        value = ratio
        
        # Step 3: Interpretation and warnings
        steps.append("Step 3: Ratio Interpretation")
        if ratio < Decimal("1.0"):
            warnings.append(f"Current ratio of {ratio:.2f} is below 1.0, indicating potential liquidity risk")
            warnings.append("Company may have difficulty meeting short-term obligations")
        elif ratio < Decimal("1.2"):
            warnings.append(f"Current ratio of {ratio:.2f} is low - monitor liquidity position closely")
        elif ratio > Decimal("3.0"):
            warnings.append(f"Current ratio of {ratio:.2f} is high, which may indicate inefficient use of assets")
            steps.append("  Note: High current ratio may suggest excess cash or inventory not being deployed effectively")
        else:
            steps.append(f"  Current ratio of {ratio:.2f} is within healthy range (1.2 - 3.0)")
    
    return create_calculation_result(
        metric_name="Current Ratio",
        value=value,
        formula=METRIC_FORMULAS.get("current_ratio", "Current Assets / Current Liabilities"),
        inputs=inputs,
        steps=steps,
        category=MetricCategory.LIQUIDITY,
        warnings=warnings,
        unit=MetricUnit.RATIO,
    )


def calculate_quick_ratio(
    current_assets: Decimal | float | int,
    inventory: Decimal | float | int,
    current_liabilities: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Quick Ratio (Acid Test Ratio).
    
    Formula: (Current Assets - Inventory) / Current Liabilities
    
    More conservative than current ratio as inventory may not be
    quickly convertible to cash.
    
    Args:
        current_assets: Total current assets
        inventory: Inventory value
        current_liabilities: Total current liabilities
        
    Returns:
        CalculationResult with quick ratio
    """
    steps = []
    warnings = []
    
    # Convert inputs to Decimal
    ca = to_decimal(current_assets)
    inv = to_decimal(inventory)
    cl = to_decimal(current_liabilities)
    
    # Record inputs
    inputs = {
        "current_assets": ca,
        "inventory": inv,
        "current_liabilities": cl,
    }
    
    # Step 1: Input validation
    steps.append(f"Step 1: Input Validation")
    steps.append(f"  Current Assets: {ca:,.2f}")
    steps.append(f"  Inventory: {inv:,.2f}")
    steps.append(f"  Current Liabilities: {cl:,.2f}")
    
    if any(x < 0 for x in [ca, inv, cl]):
        warnings.append("Negative value detected in liquidity inputs")
    
    # Step 2: Calculate quick assets
    quick_assets = ca - inv
    steps.append(f"Step 2: Calculate Quick Assets = Current Assets - Inventory = {ca:,.2f} - {inv:,.2f} = {quick_assets:,.2f}")
    inputs["quick_assets"] = quick_assets
    
    # Step 3: Calculate ratio
    if is_effectively_zero(cl):
        steps.append("Step 3: Cannot calculate ratio (current liabilities is zero)")
        warnings.append("Current liabilities is zero - ratio undefined")
        value = None
    else:
        ratio = safe_divide(quick_assets, cl, DECIMAL_PLACES["ratio"])
        steps.append(f"Step 3: Calculate Quick Ratio = Quick Assets / Current Liabilities = {quick_assets:,.2f} / {cl:,.2f} = {ratio:.4f}")
        value = ratio
        
        # Step 4: Interpretation
        steps.append("Step 4: Ratio Interpretation")
        if ratio < Decimal("1.0"):
            warnings.append(f"Quick ratio of {ratio:.2f} is below 1.0, indicating reliance on inventory to meet obligations")
        elif ratio < Decimal("0.5"):
            warnings.append(f"Very low quick ratio of {ratio:.2f} - significant liquidity concern")
        elif ratio > Decimal("2.0"):
            steps.append(f"  Quick ratio of {ratio:.2f} is strong, indicating good immediate liquidity")
        
        # Step 5: Compare to current ratio (if possible)
        if not is_effectively_zero(cl):
            current_ratio = safe_divide(ca, cl, DECIMAL_PLACES["ratio"])
            if current_ratio > 0:
                inventory_impact = current_ratio - ratio
                if inventory_impact > Decimal("0.5"):
                    steps.append(f"Step 5: Comparison Analysis")
                    steps.append(f"  Note: Inventory accounts for {inventory_impact:.2f} of the current ratio difference")
                    steps.append(f"  Current Ratio: {current_ratio:.2f}, Quick Ratio: {ratio:.2f}")
    
    return create_calculation_result(
        metric_name="Quick Ratio",
        value=value,
        formula=METRIC_FORMULAS.get("quick_ratio", "(Current Assets - Inventory) / Current Liabilities"),
        inputs=inputs,
        steps=steps,
        category=MetricCategory.LIQUIDITY,
        warnings=warnings,
        unit=MetricUnit.RATIO,
    )


def calculate_cash_ratio(
    cash_and_equivalents: Decimal | float | int,
    current_liabilities: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Cash Ratio.
    
    Formula: Cash and Equivalents / Current Liabilities
    
    Most conservative liquidity measure - only considers cash
    and cash equivalents.
    
    Args:
        cash_and_equivalents: Cash and cash equivalents
        current_liabilities: Total current liabilities
        
    Returns:
        CalculationResult with cash ratio
    """
    steps = []
    warnings = []
    
    # Convert inputs to Decimal
    cash = to_decimal(cash_and_equivalents)
    cl = to_decimal(current_liabilities)
    
    # Record inputs
    inputs = {
        "cash_and_equivalents": cash,
        "current_liabilities": cl,
    }
    
    # Step 1: Input validation
    steps.append(f"Step 1: Input Validation")
    steps.append(f"  Cash and Equivalents: {cash:,.2f}")
    steps.append(f"  Current Liabilities: {cl:,.2f}")
    
    if cash < 0:
        warnings.append("Negative cash and equivalents detected")
    if cl < 0:
        warnings.append("Negative current liabilities detected")
    
    # Step 2: Calculate ratio
    if is_effectively_zero(cl):
        steps.append("Step 2: Cannot calculate ratio (current liabilities is zero)")
        warnings.append("Current liabilities is zero - ratio undefined")
        value = None
    else:
        ratio = safe_divide(cash, cl, DECIMAL_PLACES["ratio"])
        steps.append(f"Step 2: Calculate Cash Ratio = Cash / Current Liabilities = {cash:,.2f} / {cl:,.2f} = {ratio:.4f}")
        value = ratio
        
        # Step 3: Interpretation
        steps.append("Step 3: Ratio Interpretation")
        if ratio < Decimal("0.2"):
            warnings.append(f"Cash ratio of {ratio:.2f} is below 0.2, which may be low for immediate obligations")
            warnings.append("Company may have limited ability to cover immediate expenses with cash alone")
        elif ratio < Decimal("0.5"):
            steps.append(f"  Cash ratio of {ratio:.2f} is moderate - adequate for most situations")
        elif ratio > Decimal("1.0"):
            warnings.append(f"Cash ratio of {ratio:.2f} is above 1.0, indicating potentially excess cash holdings")
            steps.append("  Note: High cash ratio may indicate conservative management or missed investment opportunities")
    
    return create_calculation_result(
        metric_name="Cash Ratio",
        value=value,
        formula=METRIC_FORMULAS.get("cash_ratio", "Cash and Equivalents / Current Liabilities"),
        inputs=inputs,
        steps=steps,
        category=MetricCategory.LIQUIDITY,
        warnings=warnings,
        unit=MetricUnit.RATIO,
    )


def calculate_working_capital(
    current_assets: Decimal | float | int,
    current_liabilities: Decimal | float | int,
    currency: str = "SGD",
) -> CalculationResult:
    """
    Calculate Working Capital.
    
    Formula: Current Assets - Current Liabilities
    
    This is an absolute amount, not a ratio.
    
    Args:
        current_assets: Total current assets
        current_liabilities: Total current liabilities
        currency: Currency code for the result
        
    Returns:
        CalculationResult with working capital amount
    """
    steps = []
    warnings = []
    
    # Convert inputs to Decimal
    ca = to_decimal(current_assets)
    cl = to_decimal(current_liabilities)
    
    # Record inputs
    inputs = {
        "current_assets": ca,
        "current_liabilities": cl,
        "currency": currency,
    }
    
    # Step 1: Input validation
    steps.append(f"Step 1: Input Validation")
    steps.append(f"  Current Assets: {ca:,.2f}")
    steps.append(f"  Current Liabilities: {cl:,.2f}")
    steps.append(f"  Currency: {currency}")
    
    if ca < 0:
        warnings.append("Negative current assets detected")
    if cl < 0:
        warnings.append("Negative current liabilities detected")
    
    # Step 2: Calculate working capital
    working_capital = ca - cl
    working_capital = round_decimal(working_capital, DECIMAL_PLACES["currency"])
    steps.append(f"Step 2: Calculate Working Capital = Current Assets - Current Liabilities = {ca:,.2f} - {cl:,.2f} = {working_capital:,.2f} {currency}")
    
    # Step 3: Interpretation
    steps.append("Step 3: Working Capital Interpretation")
    if working_capital < 0:
        warnings.append(f"Negative working capital of {working_capital:,.2f} {currency} indicates current liabilities exceed current assets")
        warnings.append("This may indicate liquidity problems or aggressive working capital management")
    elif working_capital == 0:
        warnings.append("Zero working capital - current assets exactly match current liabilities")
    else:
        steps.append(f"  Positive working capital of {working_capital:,.2f} {currency} indicates good short-term financial health")
    
    result = create_calculation_result(
        metric_name="Working Capital",
        value=working_capital,
        formula=METRIC_FORMULAS.get("working_capital", "Current Assets - Current Liabilities"),
        inputs=inputs,
        steps=steps,
        category=MetricCategory.LIQUIDITY,
        warnings=warnings,
        unit=MetricUnit.CURRENCY,
    )
    
    return result


def calculate_all_liquidity_metrics(
    balance_sheet: BalanceSheetData,
) -> MetricCollection:
    """
    Calculate all liquidity metrics.
    
    Args:
        balance_sheet: Balance sheet data
        
    Returns:
        MetricCollection with all liquidity metrics
    """
    collection = MetricCollection(
        category=MetricCategory.LIQUIDITY,
        period=balance_sheet.period,
    )
    
    # Current Ratio
    current = calculate_current_ratio(
        current_assets=balance_sheet.calculated_current_assets,
        current_liabilities=balance_sheet.calculated_current_liabilities,
    )
    collection.add_metric(current)
    
    # Quick Ratio
    quick = calculate_quick_ratio(
        current_assets=balance_sheet.calculated_current_assets,
        inventory=balance_sheet.inventory,
        current_liabilities=balance_sheet.calculated_current_liabilities,
    )
    collection.add_metric(quick)
    
    # Cash Ratio
    cash = calculate_cash_ratio(
        cash_and_equivalents=balance_sheet.cash_and_equivalents,
        current_liabilities=balance_sheet.calculated_current_liabilities,
    )
    collection.add_metric(cash)
    
    # Working Capital
    wc = calculate_working_capital(
        current_assets=balance_sheet.calculated_current_assets,
        current_liabilities=balance_sheet.calculated_current_liabilities,
        currency=balance_sheet.currency,
    )
    collection.add_metric(wc)
    
    return collection


class LiquidityCalculator(BaseCalculator):
    """Class-based calculator for liquidity metrics."""
    
    def __init__(self):
        super().__init__(MetricCategory.LIQUIDITY)
    
    def calculate_all(
        self,
        statement_set: FinancialStatementSet,
        prior_statement_set: FinancialStatementSet | None = None,
    ) -> MetricCollection:
        """
        Calculate all liquidity metrics.
        
        Args:
            statement_set: Current period financial statements
            prior_statement_set: Prior period statements (not used for liquidity)
            
        Returns:
            MetricCollection with all liquidity metrics
        """
        return calculate_all_liquidity_metrics(statement_set.balance_sheet)

```

# finanalyst_tools/calculations/profitability.py
```py
# finanalyst_tools/calculations/profitability.py
"""
Profitability ratio calculations.

Provides calculations for:
- Gross Profit Margin
- Operating Profit Margin
- Net Profit Margin
- EBITDA Margin
- Return on Assets (ROA)
- Return on Equity (ROE)
- Return on Capital Employed (ROCE)

All functions return CalculationResult with complete audit trail, detailed step-by-step calculations, and comprehensive validation.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from finanalyst_tools.config import DECIMAL_PLACES, METRIC_FORMULAS
from finanalyst_tools.models.analysis_results import (
    CalculationResult,
    MetricUnit,
    MetricCategory,
    MetricCollection,
)
from finanalyst_tools.models.financial_statements import (
    IncomeStatementData,
    BalanceSheetData,
    FinancialPeriod,
)
from finanalyst_tools.utils.math_ops import (
    to_decimal,
    safe_divide,
    round_decimal,
    calculate_percentage,
    calculate_average,
    is_effectively_zero,
)
from finanalyst_tools.calculations.base import (
    BaseCalculator,
    create_calculation_result,
    extract_value,
)


# ============================================================================
# STANDALONE CALCULATION FUNCTIONS
# ============================================================================

def calculate_gross_profit_margin(
    revenue: Decimal | float | int,
    cost_of_goods_sold: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Gross Profit Margin.
    
    Formula: (Revenue - COGS) / Revenue × 100
    
    Args:
        revenue: Total revenue
        cost_of_goods_sold: Cost of goods sold 
        
    Returns:
        CalculationResult with gross profit margin percentage
    """
    steps = []
    warnings = []
    
    # Convert inputs to Decimal
    rev = to_decimal(revenue)
    cogs = to_decimal(cost_of_goods_sold)
    
    # Record inputs
    inputs = {
        "revenue": rev,
        "cost_of_goods_sold": cogs,
    }
    
    # Step 1: Validate inputs
    steps.append(f"Step 1: Input Validation")
    steps.append(f"  Revenue: {rev:,.2f}")
    steps.append(f"  COGS: {cogs:,.2f}")
    
    if rev < 0:
        warnings.append("Negative revenue value detected")
    if cogs < 0:
        warnings.append("Negative COGS value detected")
    
    # Step 2: Calculate gross profit
    gross_profit = rev - cogs
    steps.append(f"Step 2: Calculate Gross Profit = Revenue - COGS = {rev:,.2f} - {cogs:,.2f} = {gross_profit:,.2f}")
    
    # Step 3: Check for zero revenue
    if is_effectively_zero(rev):
        steps.append("Step 3: Cannot calculate margin (revenue is zero)")
        warnings.append("Revenue is zero - cannot calculate margin")
        value = None
    else:
        # Step 4: Calculate margin
        margin = calculate_percentage(gross_profit, rev)
        margin = round_decimal(margin, DECIMAL_PLACES["percentage"])
        steps.append(f"Step 4: Calculate Gross Profit Margin = (Gross Profit / Revenue) × 100 = ({gross_profit:,.2f} / {rev:,.2f}) × 100 = {margin:.2f}%")
        value = margin
        
        # Step 5: Add contextual warnings
        if margin < Decimal("0"):
            warnings.append("Negative gross margin indicates selling below cost or data error")
        elif margin > Decimal("100"):
            warnings.append("Gross margin over 100% suggests COGS may be incomplete or negative")
        elif margin < Decimal("20"):
            warnings.append("Low gross margin (<20%) may indicate pricing pressure or high production costs")
        elif margin > Decimal("80"):
            warnings.append("Very high gross margin (>80%) may indicate premium pricing or low production costs")
    
    return create_calculation_result(
        metric_name="Gross Profit Margin",
        value=value,
        formula=METRIC_FORMULAS.get("gross_profit_margin", "(Revenue - COGS) / Revenue × 100"),
        inputs=inputs,
        steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
        unit=MetricUnit.PERCENTAGE,
    )


def calculate_operating_profit_margin(
    revenue: Decimal | float | int,
    cost_of_goods_sold: Decimal | float | int,
    operating_expenses: Decimal | float | int,
    marketing_expenses: Decimal | float | int | None = None,
    include_marketing_in_opex: bool = True,
) -> CalculationResult:
    """
    Calculate Operating Profit Margin.
    
    Formula: (Revenue - COGS - OpEx) / Revenue × 100
    
    Args:
        revenue: Total revenue
        cost_of_goods_sold: Cost of goods sold
        operating_expenses: Operating expenses
        marketing_expenses: Marketing expenses (optional, if tracked separately)
        include_marketing_in_opex: Whether marketing is already included in OpEx
        
    Returns:
        CalculationResult with operating profit margin percentage
    """
    steps = []
    warnings = []
    
    # Convert inputs to Decimal
    rev = to_decimal(revenue)
    cogs = to_decimal(cost_of_goods_sold)
    opex = to_decimal(operating_expenses)
    marketing = to_decimal(marketing_expenses) if marketing_expenses is not None else Decimal("0")
    
    # Record inputs
    inputs = {
        "revenue": rev,
        "cost_of_goods_sold": cogs,
        "operating_expenses": opex,
        "marketing_expenses": marketing if marketing_expenses is not None else None,
        "include_marketing_in_opex": include_marketing_in_opex,
    }
    
    # Step 1: Input validation
    steps.append(f"Step 1: Input Validation")
    steps.append(f"  Revenue: {rev:,.2f}")
    steps.append(f"  COGS: {cogs:,.2f}")
    steps.append(f"  Operating Expenses: {opex:,.2f}")
    
    if marketing_expenses is not None:
        steps.append(f"  Marketing Expenses: {marketing:,.2f}")
        steps.append(f"  Marketing included in OpEx: {include_marketing_in_opex}")
    
    # Validate negative values
    if rev < 0:
        warnings.append("Negative revenue value detected")
    if any(x < 0 for x in [cogs, opex, marketing]):
        warnings.append("Negative expense value detected")
    
    # Step 2: Calculate total operating expenses
    if marketing_expenses is not None and not include_marketing_in_opex:
        total_opex = opex + marketing
        steps.append(f"Step 2: Calculate Total Operating Expenses = OpEx + Marketing = {opex:,.2f} + {marketing:,.2f} = {total_opex:,.2f}")
    else:
        total_opex = opex
        if marketing_expenses is not None:
            steps.append(f"Step 2: Marketing expenses already included in operating expenses")
        else:
            steps.append(f"Step 2: Total Operating Expenses = {total_opex:,.2f}")
    
    # Step 3: Calculate gross profit
    gross_profit = rev - cogs
    steps.append(f"Step 3: Calculate Gross Profit = Revenue - COGS = {rev:,.2f} - {cogs:,.2f} = {gross_profit:,.2f}")
    
    # Step 4: Calculate operating profit
    operating_profit = gross_profit - total_opex
    steps.append(f"Step 4: Calculate Operating Profit = Gross Profit - Total OpEx = {gross_profit:,.2f} - {total_opex:,.2f} = {operating_profit:,.2f}")
    
    # Step 5: Calculate margin
    if is_effectively_zero(rev):
        steps.append("Step 5: Cannot calculate margin (revenue is zero)")
        warnings.append("Revenue is zero - cannot calculate margin")
        value = None
    else:
        margin = calculate_percentage(operating_profit, rev)
        margin = round_decimal(margin, DECIMAL_PLACES["percentage"])
        steps.append(f"Step 5: Calculate Operating Profit Margin = (Operating Profit / Revenue) × 100 = ({operating_profit:,.2f} / {rev:,.2f}) × 100 = {margin:.2f}%")
        value = margin
        
        # Contextual warnings
        if margin < Decimal("-50"):
            warnings.append("Severely negative operating margin (< -50%) indicates significant operational distress")
        elif margin < Decimal("0"):
            warnings.append("Negative operating margin indicates core business operations are unprofitable")
        elif margin > Decimal("30"):
            warnings.append("High operating margin (>30%) suggests strong operational efficiency or premium pricing")
    
    return create_calculation_result(
        metric_name="Operating Profit Margin",
        value=value,
        formula=METRIC_FORMULAS.get("operating_profit_margin", "(Revenue - COGS - OpEx) / Revenue × 100"),
        inputs=inputs,
        steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
        unit=MetricUnit.PERCENTAGE,
    )


def calculate_net_profit_margin(
    revenue: Decimal | float | int,
    net_income: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Net Profit Margin.
    
    Formula: Net Income / Revenue × 100
    
    Args:
        revenue: Total revenue
        net_income: Net income (profit after tax)
        
    Returns:
        CalculationResult with net profit margin percentage
    """
    steps = []
    warnings = []
    
    # Convert inputs to Decimal
    rev = to_decimal(revenue)
    ni = to_decimal(net_income)
    
    # Record inputs
    inputs = {
        "revenue": rev,
        "net_income": ni,
    }
    
    # Step 1: Input validation
    steps.append(f"Step 1: Input Validation")
    steps.append(f"  Revenue: {rev:,.2f}")
    steps.append(f"  Net Income: {ni:,.2f}")
    
    if rev < 0:
        warnings.append("Negative revenue value detected")
    if ni < 0:
        steps.append("  Note: Negative net income indicates net loss")
    
    # Step 2: Calculate margin
    if is_effectively_zero(rev):
        steps.append("Step 2: Cannot calculate margin (revenue is zero)")
        warnings.append("Revenue is zero - cannot calculate margin")
        value = None
    else:
        margin = calculate_percentage(ni, rev)
        margin = round_decimal(margin, DECIMAL_PLACES["percentage"])
        steps.append(f"Step 2: Calculate Net Profit Margin = (Net Income / Revenue) × 100 = ({ni:,.2f} / {rev:,.2f}) × 100 = {margin:.2f}%")
        value = margin
        
        # Contextual warnings and insights
        if margin < Decimal("-100"):
            warnings.append("Net margin below -100% indicates losses exceed revenue - severe financial distress")
        elif margin < Decimal("0"):
            warnings.append("Negative net margin indicates company is operating at a loss")
        elif margin < Decimal("5"):
            warnings.append("Low net margin (<5%) suggests limited profitability or high operating leverage")
        elif margin > Decimal("20"):
            steps.append("  Note: High net margin (>20%) indicates strong profitability and/or efficient operations")
    
    return create_calculation_result(
        metric_name="Net Profit Margin",
        value=value,
        formula=METRIC_FORMULAS.get("net_profit_margin", "Net Income / Revenue × 100"),
        inputs=inputs,
        steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
        unit=MetricUnit.PERCENTAGE,
    )


def calculate_ebitda_margin(
    revenue: Decimal | float | int,
    ebitda: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate EBITDA Margin.
    
    Formula: EBITDA / Revenue × 100
    
    Args:
        revenue: Total revenue
        ebitda: Earnings Before Interest, Taxes, Depreciation, and Amortization
        
    Returns:
        CalculationResult with EBITDA margin percentage
    """
    steps = []
    warnings = []
    
    # Convert inputs to Decimal
    rev = to_decimal(revenue)
    ebitda_val = to_decimal(ebitda)
    
    # Record inputs
    inputs = {
        "revenue": rev,
        "ebitda": ebitda_val,
    }
    
    # Step 1: Input validation
    steps.append(f"Step 1: Input Validation")
    steps.append(f"  Revenue: {rev:,.2f}")
    steps.append(f"  EBITDA: {ebitda_val:,.2f}")
    
    if rev < 0:
        warnings.append("Negative revenue value detected")
    if ebitda_val < 0:
        steps.append("  Note: Negative EBITDA indicates operating losses before non-cash items")
    
    # Step 2: Calculate margin
    if is_effectively_zero(rev):
        steps.append("Step 2: Cannot calculate margin (revenue is zero)")
        warnings.append("Revenue is zero - cannot calculate margin")
        value = None
    else:
        margin = calculate_percentage(ebitda_val, rev)
        margin = round_decimal(margin, DECIMAL_PLACES["percentage"])
        steps.append(f"Step 2: Calculate EBITDA Margin = (EBITDA / Revenue) × 100 = ({ebitda_val:,.2f} / {rev:,.2f}) × 100 = {margin:.2f}%")
        value = margin
        
        # Contextual analysis
        if margin < Decimal("0"):
            warnings.append("Negative EBITDA margin indicates core operations are unprofitable")
        elif margin < Decimal("10"):
            warnings.append("Low EBITDA margin (<10%) suggests limited operating profitability")
        elif margin > Decimal("30"):
            steps.append("  Note: High EBITDA margin (>30%) indicates strong operational cash generation")
    
    return create_calculation_result(
        metric_name="EBITDA Margin",
        value=value,
        formula=METRIC_FORMULAS.get("ebitda_margin", "EBITDA / Revenue × 100"),
        inputs=inputs,
        steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
        unit=MetricUnit.PERCENTAGE,
    )


def calculate_return_on_assets(
    net_income: Decimal | float | int,
    total_assets_beginning: Decimal | float | int,
    total_assets_ending: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Return on Assets (ROA).
    
    Formula: Net Income / Average Total Assets × 100
    
    Args:
        net_income: Net income for the period
        total_assets_beginning: Total assets at period start
        total_assets_ending: Total assets at period end
        
    Returns:
        CalculationResult with ROA percentage
    """
    steps = []
    warnings = []
    
    # Convert inputs to Decimal
    ni = to_decimal(net_income)
    assets_begin = to_decimal(total_assets_beginning)
    assets_end = to_decimal(total_assets_ending)
    
    # Record inputs
    inputs = {
        "net_income": ni,
        "total_assets_beginning": assets_begin,
        "total_assets_ending": assets_end,
    }
    
    # Step 1: Input validation
    steps.append(f"Step 1: Input Validation")
    steps.append(f"  Net Income: {ni:,.2f}")
    steps.append(f"  Total Assets (Beginning): {assets_begin:,.2f}")
    steps.append(f"  Total Assets (Ending): {assets_end:,.2f}")
    
    if any(x < 0 for x in [assets_begin, assets_end]):
        warnings.append("Negative asset value detected")
    
    # Step 2: Calculate average assets
    avg_assets = calculate_average(assets_begin, assets_end)
    steps.append(f"Step 2: Calculate Average Total Assets = (Beginning + Ending) / 2 = ({assets_begin:,.2f} + {assets_end:,.2f}) / 2 = {avg_assets:,.2f}")
    inputs["average_total_assets"] = avg_assets
    
    # Step 3: Calculate ROA
    if is_effectively_zero(avg_assets):
        steps.append("Step 3: Cannot calculate ROA (average assets is zero)")
        warnings.append("Average assets is zero - cannot calculate ROA")
        value = None
    else:
        roa = calculate_percentage(ni, avg_assets)
        roa = round_decimal(roa, DECIMAL_PLACES["percentage"])
        steps.append(f"Step 3: Calculate ROA = (Net Income / Average Assets) × 100 = ({ni:,.2f} / {avg_assets:,.2f}) × 100 = {roa:.2f}%")
        value = roa
        
        # Contextual analysis
        if roa < Decimal("0"):
            warnings.append("Negative ROA indicates the company is destroying asset value")
        elif roa < Decimal("5"):
            warnings.append("Low ROA (<5%) suggests inefficient asset utilization")
        elif roa > Decimal("20"):
            steps.append("  Note: High ROA (>20%) indicates excellent asset utilization and profitability")
    
    return create_calculation_result(
        metric_name="Return on Assets",
        value=value,
        formula=METRIC_FORMULAS.get("roa", "Net Income / Average Total Assets × 100"),
        inputs=inputs,
        steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
        unit=MetricUnit.PERCENTAGE,
    )


def calculate_return_on_equity(
    net_income: Decimal | float | int,
    shareholders_equity_beginning: Decimal | float | int,
    shareholders_equity_ending: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Return on Equity (ROE).
    
    Formula: Net Income / Average Shareholders' Equity × 100
    
    Args:
        net_income: Net income for the period
        shareholders_equity_beginning: Equity at period start
        shareholders_equity_ending: Equity at period end
        
    Returns:
        CalculationResult with ROE percentage
    """
    steps = []
    warnings = []
    
    # Convert inputs to Decimal
    ni = to_decimal(net_income)
    equity_begin = to_decimal(shareholders_equity_beginning)
    equity_end = to_decimal(shareholders_equity_ending)
    
    # Record inputs
    inputs = {
        "net_income": ni,
        "shareholders_equity_beginning": equity_begin,
        "shareholders_equity_ending": equity_end,
    }
    
    # Step 1: Input validation
    steps.append(f"Step 1: Input Validation")
    steps.append(f"  Net Income: {ni:,.2f}")
    steps.append(f"  Shareholders' Equity (Beginning): {equity_begin:,.2f}")
    steps.append(f"  Shareholders' Equity (Ending): {equity_end:,.2f}")
    
    # Check for negative equity
    negative_equity = False
    if equity_begin < 0 or equity_end < 0:
        negative_equity = True
        warnings.append("Negative shareholders' equity indicates accumulated losses exceed invested capital")
    
    # Step 2: Calculate average equity
    avg_equity = calculate_average(equity_begin, equity_end)
    steps.append(f"Step 2: Calculate Average Shareholders' Equity = (Beginning + Ending) / 2 = ({equity_begin:,.2f} + {equity_end:,.2f}) / 2 = {avg_equity:,.2f}")
    inputs["average_shareholders_equity"] = avg_equity
    
    # Step 3: Calculate ROE
    if is_effectively_zero(avg_equity):
        steps.append("Step 3: Cannot calculate ROE (average equity is zero)")
        warnings.append("Average equity is zero - cannot calculate ROE")
        value = None
    else:
        roe = calculate_percentage(ni, avg_equity)
        roe = round_decimal(roe, DECIMAL_PLACES["percentage"])
        steps.append(f"Step 3: Calculate ROE = (Net Income / Average Equity) × 100 = ({ni:,.2f} / {avg_equity:,.2f}) × 100 = {roe:.2f}%")
        value = roe
        
        # Contextual analysis
        if negative_equity:
            steps.append("  Note: ROE interpretation is complex with negative equity")
        
        if roe < Decimal("0") and not negative_equity:
            warnings.append("Negative ROE indicates the company is destroying shareholder value")
        elif roe < Decimal("10") and not negative_equity:
            warnings.append("Low ROE (<10%) suggests limited return on shareholder investment")
        elif roe > Decimal("25") and not negative_equity:
            steps.append("  Note: High ROE (>25%) indicates excellent return on shareholder investment")
        
        # Special case for negative equity
        if negative_equity and roe > Decimal("0"):
            steps.append("  Note: Positive ROE with negative equity indicates recovery from accumulated losses")
    
    return create_calculation_result(
        metric_name="Return on Equity",
        value=value,
        formula=METRIC_FORMULAS.get("roe", "Net Income / Average Shareholders' Equity × 100"),
        inputs=inputs,
        steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
        unit=MetricUnit.PERCENTAGE,
    )


def calculate_return_on_capital_employed(
    ebit: Decimal | float | int,
    total_assets: Decimal | float | int,
    current_liabilities: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Return on Capital Employed (ROCE).
    
    Formula: EBIT / (Total Assets - Current Liabilities) × 100
    
    Args:
        ebit: Earnings Before Interest and Taxes
        total_assets: Total assets
        current_liabilities: Current liabilities
        
    Returns:
        CalculationResult with ROCE percentage
    """
    steps = []
    warnings = []
    
    # Convert inputs to Decimal
    ebit_val = to_decimal(ebit)
    assets = to_decimal(total_assets)
    cl = to_decimal(current_liabilities)
    
    # Record inputs
    inputs = {
        "ebit": ebit_val,
        "total_assets": assets,
        "current_liabilities": cl,
    }
    
    # Step 1: Input validation
    steps.append(f"Step 1: Input Validation")
    steps.append(f"  EBIT: {ebit_val:,.2f}")
    steps.append(f"  Total Assets: {assets:,.2f}")
    steps.append(f"  Current Liabilities: {cl:,.2f}")
    
    if assets < 0:
        warnings.append("Negative total assets detected")
    if cl < 0:
        warnings.append("Negative current liabilities detected")
    
    # Step 2: Calculate capital employed
    capital_employed = assets - cl
    steps.append(f"Step 2: Calculate Capital Employed = Total Assets - Current Liabilities = {assets:,.2f} - {cl:,.2f} = {capital_employed:,.2f}")
    inputs["capital_employed"] = capital_employed
    
    # Step 3: Calculate ROCE
    if is_effectively_zero(capital_employed):
        steps.append("Step 3: Cannot calculate ROCE (capital employed is zero)")
        warnings.append("Capital employed is zero - cannot calculate ROCE")
        value = None
    elif capital_employed < 0:
        steps.append("Step 3: Negative capital employed detected")
        warnings.append("Negative capital employed indicates current liabilities exceed total assets - potential financial distress")
        roce = calculate_percentage(ebit_val, capital_employed)
        roce = round_decimal(roce, DECIMAL_PLACES["percentage"])
        steps.append(f"Step 3 (cont): Calculate ROCE = (EBIT / Capital Employed) × 100 = ({ebit_val:,.2f} / {capital_employed:,.2f}) × 100 = {roce:.2f}%")
        value = roce
    else:
        roce = calculate_percentage(ebit_val, capital_employed)
        roce = round_decimal(roce, DECIMAL_PLACES["percentage"])
        steps.append(f"Step 3: Calculate ROCE = (EBIT / Capital Employed) × 100 = ({ebit_val:,.2f} / {capital_employed:,.2f}) × 100 = {roce:.2f}%")
        value = roce
        
        # Contextual analysis
        if roce < Decimal("0"):
            warnings.append("Negative ROCE indicates the company is not generating returns on its capital")
        elif roce < Decimal("8"):
            warnings.append("Low ROCE (<8%) suggests inefficient capital utilization")
        elif roce > Decimal("20"):
            steps.append("  Note: High ROCE (>20%) indicates excellent capital efficiency and profitability")
    
    return create_calculation_result(
        metric_name="Return on Capital Employed",
        value=value,
        formula=METRIC_FORMULAS.get("roce", "EBIT / (Total Assets - Current Liabilities) × 100"),
        inputs=inputs,
        steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
        unit=MetricUnit.PERCENTAGE,
    )


def calculate_all_profitability_metrics(
    income_statement: IncomeStatementData,
    balance_sheet: BalanceSheetData,
    prior_balance_sheet: BalanceSheetData | None = None,
) -> MetricCollection:
    """
    Calculate all profitability metrics.
    
    Args:
        income_statement: Current period income statement
        balance_sheet: Current period balance sheet
        prior_balance_sheet: Prior period balance sheet (for averages)
        
    Returns:
        MetricCollection with all profitability metrics
    """
    collection = MetricCollection(
        category=MetricCategory.PROFITABILITY,
        period=income_statement.period,
    )
    
    # Use current balance sheet for both if no prior available
    if prior_balance_sheet is None:
        prior_balance_sheet = balance_sheet
    
    # Gross Profit Margin
    gpm = calculate_gross_profit_margin(
        revenue=income_statement.total_revenue,
        cost_of_goods_sold=income_statement.cost_of_goods_sold,
    )
    collection.add_metric(gpm)
    
    # Operating Profit Margin
    opm = calculate_operating_profit_margin(
        revenue=income_statement.total_revenue,
        cost_of_goods_sold=income_statement.cost_of_goods_sold,
        operating_expenses=income_statement.total_operating_expenses,
        marketing_expenses=income_statement.marketing_expenses,
        include_marketing_in_opex=True,
    )
    collection.add_metric(opm)
    
    # Net Profit Margin
    npm = calculate_net_profit_margin(
        revenue=income_statement.total_revenue,
        net_income=income_statement.calculated_net_income,
    )
    collection.add_metric(npm)
    
    # EBITDA Margin
    ebitda_m = calculate_ebitda_margin(
        revenue=income_statement.total_revenue,
        ebitda=income_statement.ebitda,
    )
    collection.add_metric(ebitda_m)
    
    # ROA
    roa = calculate_return_on_assets(
        net_income=income_statement.calculated_net_income,
        total_assets_beginning=prior_balance_sheet.calculated_total_assets,
        total_assets_ending=balance_sheet.calculated_total_assets,
    )
    collection.add_metric(roa)
    
    # ROE
    roe = calculate_return_on_equity(
        net_income=income_statement.calculated_net_income,
        shareholders_equity_beginning=prior_balance_sheet.calculated_shareholders_equity,
        shareholders_equity_ending=balance_sheet.calculated_shareholders_equity,
    )
    collection.add_metric(roe)
    
    # ROCE
    roce = calculate_return_on_capital_employed(
        ebit=income_statement.ebit,
        total_assets=balance_sheet.calculated_total_assets,
        current_liabilities=balance_sheet.calculated_current_liabilities,
    )
    collection.add_metric(roce)
    
    return collection


class ProfitabilityCalculator(BaseCalculator):
    """Class-based calculator for profitability metrics."""
    
    def __init__(self):
        super().__init__(MetricCategory.PROFITABILITY)
    
    def calculate_all(
        self,
        statement_set: FinancialStatementSet,
        prior_statement_set: FinancialStatementSet | None = None,
    ) -> MetricCollection:
        """
        Calculate all profitability metrics.
        
        Args:
            statement_set: Current period financial statements
            prior_statement_set: Prior period statements (for averages)
            
        Returns:
            MetricCollection with all profitability metrics
        """
        prior_bs = prior_statement_set.balance_sheet if prior_statement_set else None
        
        return calculate_all_profitability_metrics(
            income_statement=statement_set.income_statement,
            balance_sheet=statement_set.balance_sheet,
            prior_balance_sheet=prior_bs,
        )

```

# finanalyst_tools/exceptions.py
```py
# File: finanalyst_tools/exceptions.py
"""
Custom exception hierarchy for FinAnalyst-Pro Agent Tools.

Provides specific exception types for different error categories:
- Calculation errors (arithmetic, division by zero, invalid inputs)
- Validation errors (schema, reconciliation, plausibility)
- Data errors (parsing, missing data)
- Tool errors (not found, execution failure)

All exceptions support:
- Serialization to dict/JSON for structured error handling
- Auto-generated error codes for programmatic handling
- Actionable suggestions for LLM consumption
"""

from __future__ import annotations

import json
from typing import Any


# ============================================================================
# BASE EXCEPTION
# ============================================================================

class FinAnalystError(Exception):
    """
    Base exception for all FinAnalyst-Pro errors.
    
    Features:
    - Auto-generated error_code from class name
    - Optional details dictionary for context
    - Optional suggestion for resolution
    - JSON serialization support
    
    All custom exceptions should inherit from this class.
    """
    
    def __init__(
        self, 
        message: str, 
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
        suggestion: str | None = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional context
            error_code: Optional error code (auto-generated if not provided)
            suggestion: Optional actionable suggestion for resolution
            **kwargs: Additional key-value pairs to include in details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.details.update(kwargs)
        self.error_code = error_code or self._generate_error_code()
        self.suggestion = suggestion
    
    def _generate_error_code(self) -> str:
        """Generate error code from class name (CamelCase → SCREAMING_SNAKE)."""
        name = self.__class__.__name__
        result = []
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                result.append("_")
            result.append(char.upper())
        return "".join(result).replace("_ERROR", "")
    
    @property
    def error_type(self) -> str:
        """Get the exception class name."""
        return self.__class__.__name__
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        result = {
            "error_type": self.error_type,
            "error_code": self.error_code,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        if self.suggestion:
            result["suggestion"] = self.suggestion
        return result
    
    def to_json(self) -> str:
        """Convert exception to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def __str__(self) -> str:
        """Format error message with details."""
        parts = [f"[{self.error_code}] {self.message}"]
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)
    
    def __repr__(self) -> str:
        if self.details:
            return f"{self.error_type}({self.message!r}, details={self.details!r})"
        return f"{self.error_type}({self.message!r})"


# ============================================================================
# CALCULATION ERRORS
# ============================================================================

class CalculationError(FinAnalystError):
    """
    Base exception for calculation-related errors.
    
    Raised when a financial calculation cannot be completed
    due to mathematical issues or invalid inputs.
    """
    
    def __init__(
        self,
        message: str,
        metric_name: str | None = None,
        formula: str | None = None,
        suggestion: str | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if metric_name:
            details["metric_name"] = metric_name
        if formula:
            details["formula"] = formula
        super().__init__(
            message, 
            details=details, 
            suggestion=suggestion or "Check input values and try again"
        )


class DivisionByZeroError(CalculationError):
    """
    Raised when a calculation would result in division by zero.
    
    Includes information about the numerator and denominator
    to aid in debugging data issues.
    """
    
    def __init__(
        self,
        numerator: Any,
        denominator_name: str = "denominator",
        metric_name: str | None = None,
        **kwargs: Any
    ) -> None:
        message = f"Cannot divide {numerator} by zero ({denominator_name} is zero)"
        super().__init__(
            message,
            metric_name=metric_name,
            suggestion=f"Ensure {denominator_name} is non-zero before calculation",
            numerator=numerator,
            denominator_name=denominator_name,
            **kwargs
        )


class InvalidInputError(CalculationError):
    """
    Raised when input values are invalid for calculation.
    
    Examples:
    - Negative values where positive required
    - Wrong data types
    - Values outside acceptable ranges
    """
    
    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        actual_value: Any = None,
        expected: str | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if field_name:
            details["field_name"] = field_name
        if actual_value is not None:
            details["actual_value"] = actual_value
        if expected:
            details["expected"] = expected
        
        suggestion = f"Provide a value that is: {expected}" if expected else None
        super().__init__(message, suggestion=suggestion, **details)


# ============================================================================
# VALIDATION ERRORS
# ============================================================================

class ValidationError(FinAnalystError):
    """
    Base exception for validation-related errors.
    
    Raised when data fails validation checks.
    """
    
    def __init__(
        self,
        message: str,
        field: str | None = None,
        validation_type: str | None = None,
        suggestion: str | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if field:
            details["field"] = field
        if validation_type:
            details["validation_type"] = validation_type
        super().__init__(message, details=details, suggestion=suggestion)


class SchemaValidationError(ValidationError):
    """Raised when data doesn't conform to expected schema."""
    
    def __init__(
        self,
        message: str,
        schema_name: str | None = None,
        field_errors: dict[str, str] | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if schema_name:
            details["schema_name"] = schema_name
        if field_errors:
            details["field_errors"] = field_errors
        super().__init__(
            message,
            validation_type="schema",
            suggestion="Verify data structure matches expected schema",
            **details
        )


class DataCompletenessError(ValidationError):
    """Raised when required data is missing for an analysis."""
    
    def __init__(
        self,
        analysis_type: str,
        missing_fields: list[str],
        **kwargs: Any
    ) -> None:
        message = (
            f"Insufficient data for {analysis_type} analysis. "
            f"Missing: {', '.join(missing_fields)}"
        )
        super().__init__(
            message,
            validation_type="completeness",
            suggestion=f"Provide the following fields: {', '.join(missing_fields)}",
            analysis_type=analysis_type,
            missing_fields=missing_fields,
            **kwargs
        )


class ReconciliationError(ValidationError):
    """
    Raised when cross-statement reconciliation fails.
    
    Indicates that values that should match across statements
    are inconsistent beyond acceptable tolerance.
    """
    
    def __init__(
        self,
        message: str,
        check_name: str,
        expected_value: Any,
        actual_value: Any,
        difference: Any = None,
        tolerance: float | None = None,
        **kwargs: Any
    ) -> None:
        details = {
            "check_name": check_name,
            "expected_value": expected_value,
            "actual_value": actual_value,
        }
        if difference is not None:
            details["difference"] = difference
        if tolerance is not None:
            details["tolerance_used"] = tolerance
        details.update(kwargs)
        
        super().__init__(
            message,
            validation_type="reconciliation",
            suggestion="Verify data accuracy or confirm known discrepancy",
            **details
        )


class PlausibilityError(ValidationError):
    """
    Raised when a calculated metric is outside plausible range.
    
    Note: This is typically a warning, not an error, unless explicitly strict.
    """
    
    def __init__(
        self,
        metric_name: str,
        value: float,
        plausible_range: tuple[float, float],
        **kwargs: Any
    ) -> None:
        message = (
            f"{metric_name} value of {value:.2f} is outside the plausible range "
            f"({plausible_range[0]:.2f} to {plausible_range[1]:.2f})"
        )
        super().__init__(
            message,
            validation_type="plausibility",
            suggestion="Verify input data accuracy. This may indicate a data entry error.",
            metric_name=metric_name,
            value=value,
            min_plausible=plausible_range[0],
            max_plausible=plausible_range[1],
            **kwargs
        )


# ============================================================================
# DATA ERRORS
# ============================================================================

class DataError(FinAnalystError):
    """Base exception for data-related errors."""
    pass


class DataParsingError(DataError):
    """
    Raised when data cannot be parsed from input format.
    """
    
    def __init__(
        self,
        message: str,
        source: str | None = None,
        line_number: int | None = None,
        raw_data: str | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if source:
            details["source"] = source
        if line_number is not None:
            details["line_number"] = line_number
        if raw_data:
            details["raw_data"] = raw_data[:500] if len(raw_data) > 500 else raw_data
        super().__init__(
            message, 
            details=details,
            suggestion="Check source format and encoding"
        )


class MissingDataError(DataError):
    """Raised when required data is missing."""
    
    def __init__(
        self,
        message: str,
        missing_fields: list[str] | None = None,
        required_for: str | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if missing_fields:
            details["missing_fields"] = missing_fields
        if required_for:
            details["required_for"] = required_for
        
        suggestion = None
        if missing_fields:
            suggestion = f"Provide the following fields: {', '.join(missing_fields)}"
        
        super().__init__(message, details=details, suggestion=suggestion)


# ============================================================================
# TOOL ERRORS
# ============================================================================

class ToolError(FinAnalystError):
    """Base exception for tool-related errors."""
    
    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        **kwargs: Any
    ) -> None:
        details = kwargs
        if tool_name:
            details["tool_name"] = tool_name
        super().__init__(message, details=details)


class ToolNotFoundError(ToolError):
    """Raised when a requested tool does not exist."""
    
    def __init__(
        self,
        tool_name: str,
        available_tools: list[str] | None = None,
        **kwargs: Any
    ) -> None:
        suggestions = self._find_similar(tool_name, available_tools or [])
        message = f"Tool '{tool_name}' not found"
        if suggestions:
            message += f". Did you mean: {', '.join(suggestions)}?"
        
        super().__init__(
            message,
            tool_name=tool_name,
            suggestion=f"Use one of the available tools",
            available_tools=available_tools[:10] if available_tools else None,
            **kwargs
        )
    
    @staticmethod
    def _find_similar(name: str, available: list[str]) -> list[str]:
        """Find tools with similar names."""
        name_lower = name.lower()
        similar = [
            t for t in available 
            if name_lower in t.lower() or t.lower() in name_lower
        ]
        return similar[:3] if similar else available[:3]


class ToolExecutionError(ToolError):
    """Raised when a tool fails during execution."""
    
    def __init__(
        self,
        tool_name: str,
        original_error: Exception,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> None:
        message = f"Tool '{tool_name}' failed: {str(original_error)}"
        super().__init__(
            message,
            tool_name=tool_name,
            suggestion="Check tool parameters and try again",
            original_error_type=type(original_error).__name__,
            original_error_message=str(original_error),
            parameters=parameters,
            **kwargs
        )
        self.original_error = original_error


class ToolParameterError(ToolError):
    """Raised when tool parameters are invalid."""
    
    def __init__(
        self,
        tool_name: str,
        parameter_name: str,
        message: str,
        expected_type: str | None = None,
        actual_value: Any = None,
        **kwargs: Any
    ) -> None:
        full_message = f"Parameter '{parameter_name}' for tool '{tool_name}': {message}"
        super().__init__(
            full_message,
            tool_name=tool_name,
            suggestion=f"Provide a valid value for '{parameter_name}'",
            parameter_name=parameter_name,
            expected_type=expected_type,
            actual_value=str(actual_value)[:100] if actual_value is not None else None,
            **kwargs
        )

```

# finanalyst_tools/config.py
```py
# File: finanalyst_tools/config.py
"""
Configuration and constants for FinAnalyst-Pro Agent Tools.

This module provides centralized configuration for:
- Decimal precision and rounding modes
- Plausibility ranges for financial metrics
- Reconciliation tolerances
- Currency settings (SGD default for Singapore SMB context)
- Analysis parameters
- Metric formulas and units metadata

All constants use Final for immutability and are fully typed.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, ROUND_HALF_EVEN, Decimal
from enum import Enum
from typing import Final, Any


# ============================================================================
# PRECISION & ROUNDING CONFIGURATION
# ============================================================================

class RoundingMode(str, Enum):
    """
    Supported rounding modes for financial calculations.
    
    STANDARD: Round half up (0.5 → 1) - Most common in financial reporting
    BANKERS: Round half to even (banker's rounding) - Reduces cumulative bias
    """
    STANDARD = "ROUND_HALF_UP"
    BANKERS = "ROUND_HALF_EVEN"
    
    def get_decimal_rounding(self) -> str:
        """Get the decimal module rounding constant."""
        if self == RoundingMode.STANDARD:
            return ROUND_HALF_UP
        return ROUND_HALF_EVEN


# Default decimal precision for different contexts
DECIMAL_PLACES: Final[dict[str, int]] = {
    "currency": 2,       # Monetary values: $1,234.56
    "percentage": 2,     # Percentages: 12.34%
    "ratio": 4,          # Financial ratios: 1.5432
    "shares": 0,         # Share counts: whole numbers
    "growth_rate": 4,    # Growth rates: 0.1234 (12.34%)
    "turnover": 2,       # Turnover ratios: 4.56x
    "days": 0,           # Day counts: whole numbers
}

# Default rounding mode for all calculations
DEFAULT_ROUNDING: Final[RoundingMode] = RoundingMode.STANDARD


# ============================================================================
# PLAUSIBILITY THRESHOLDS
# ============================================================================

class PlausibilityRanges:
    """
    Acceptable ranges for financial ratios and metrics.
    
    Values outside these ranges trigger warnings (not errors) during analysis.
    Ranges are intentionally wide to accommodate various industries and situations
    while catching obvious data errors.
    
    All percentage values are expressed as actual percentages (e.g., 20.0 = 20%).
    All ratios are expressed as decimal values (e.g., 1.5 = 1.5x).
    """
    
    # -------------------------------------------------------------------------
    # PROFITABILITY METRICS (percentages)
    # -------------------------------------------------------------------------
    GROSS_MARGIN: Final[tuple[float, float]] = (-50.0, 95.0)
    OPERATING_MARGIN: Final[tuple[float, float]] = (-100.0, 60.0)
    NET_MARGIN: Final[tuple[float, float]] = (-200.0, 50.0)
    EBITDA_MARGIN: Final[tuple[float, float]] = (-50.0, 70.0)
    ROA: Final[tuple[float, float]] = (-50.0, 40.0)
    ROE: Final[tuple[float, float]] = (-100.0, 80.0)
    ROCE: Final[tuple[float, float]] = (-50.0, 60.0)
    
    # -------------------------------------------------------------------------
    # LIQUIDITY METRICS (ratios)
    # -------------------------------------------------------------------------
    CURRENT_RATIO: Final[tuple[float, float]] = (0.1, 10.0)
    QUICK_RATIO: Final[tuple[float, float]] = (0.05, 8.0)
    CASH_RATIO: Final[tuple[float, float]] = (0.0, 5.0)
    
    # -------------------------------------------------------------------------
    # SOLVENCY METRICS (ratios)
    # -------------------------------------------------------------------------
    DEBT_TO_EQUITY: Final[tuple[float, float]] = (0.0, 10.0)
    DEBT_TO_ASSETS: Final[tuple[float, float]] = (0.0, 1.5)
    INTEREST_COVERAGE: Final[tuple[float, float]] = (-10.0, 100.0)
    EQUITY_RATIO: Final[tuple[float, float]] = (-0.5, 1.0)
    
    # -------------------------------------------------------------------------
    # EFFICIENCY METRICS (ratios/turnover)
    # -------------------------------------------------------------------------
    ASSET_TURNOVER: Final[tuple[float, float]] = (0.1, 5.0)
    INVENTORY_TURNOVER: Final[tuple[float, float]] = (0.5, 50.0)
    RECEIVABLES_TURNOVER: Final[tuple[float, float]] = (1.0, 50.0)
    PAYABLES_TURNOVER: Final[tuple[float, float]] = (1.0, 30.0)
    FIXED_ASSET_TURNOVER: Final[tuple[float, float]] = (0.1, 20.0)
    DAYS_SALES_OUTSTANDING: Final[tuple[float, float]] = (5.0, 180.0)
    DAYS_INVENTORY_OUTSTANDING: Final[tuple[float, float]] = (5.0, 365.0)
    DAYS_PAYABLES_OUTSTANDING: Final[tuple[float, float]] = (5.0, 180.0)
    
    # -------------------------------------------------------------------------
    # GROWTH METRICS (percentages)
    # -------------------------------------------------------------------------
    REVENUE_GROWTH: Final[tuple[float, float]] = (-80.0, 500.0)
    NET_INCOME_GROWTH: Final[tuple[float, float]] = (-500.0, 1000.0)
    ASSET_GROWTH: Final[tuple[float, float]] = (-50.0, 200.0)
    
    @classmethod
    def get_range(cls, metric_name: str) -> tuple[float, float] | None:
        """
        Get plausibility range for a metric by name.
        
        Args:
            metric_name: Name of the metric (case-insensitive, underscores/spaces flexible)
            
        Returns:
            Tuple of (min, max) or None if metric not found
        """
        normalized = metric_name.upper().replace(" ", "_").replace("-", "_")
        # Handle common aliases
        aliases = {
            "GROSS_PROFIT_MARGIN": "GROSS_MARGIN",
            "OPERATING_PROFIT_MARGIN": "OPERATING_MARGIN",
            "NET_PROFIT_MARGIN": "NET_MARGIN",
            "RETURN_ON_ASSETS": "ROA",
            "RETURN_ON_EQUITY": "ROE",
            "RETURN_ON_CAPITAL_EMPLOYED": "ROCE",
        }
        normalized = aliases.get(normalized, normalized)
        return getattr(cls, normalized, None)
    
    @classmethod
    def is_plausible(cls, metric_name: str, value: float) -> bool:
        """
        Check if a metric value is within plausible range.
        
        Args:
            metric_name: Name of the metric
            value: The value to check
            
        Returns:
            True if within range or range not defined, False otherwise
        """
        range_tuple = cls.get_range(metric_name)
        if range_tuple is None:
            return True
        return range_tuple[0] <= value <= range_tuple[1]
    
    @classmethod
    def get_assessment(cls, metric_name: str, value: float) -> str:
        """
        Get a human-readable assessment of a metric value.
        
        Args:
            metric_name: Name of the metric
            value: The value to assess
            
        Returns:
            Assessment string: "within_range", "below_range", "above_range", or "unknown"
        """
        range_tuple = cls.get_range(metric_name)
        if range_tuple is None:
            return "unknown"
        
        if value < range_tuple[0]:
            return "below_range"
        elif value > range_tuple[1]:
            return "above_range"
        return "within_range"


# ============================================================================
# RECONCILIATION TOLERANCES
# ============================================================================

class ReconciliationTolerances:
    """
    Acceptable tolerance levels for cross-statement reconciliation.
    
    Expressed as a proportion (0.01 = 1%) of the larger value being compared.
    Different tolerance levels for different reconciliation contexts.
    """
    
    STRICT: Final[float] = 0.001   # 0.1% - Values that should match exactly
    NORMAL: Final[float] = 0.01    # 1% - Minor rounding differences allowed
    LOOSE: Final[float] = 0.05     # 5% - Derived values with compounding differences
    DEFAULT: Final[float] = NORMAL
    
    # Specific tolerances for different check types
    CHECK_TOLERANCES: Final[dict[str, float]] = {
        "net_income": STRICT,
        "cash_balance": STRICT,
        "retained_earnings": NORMAL,
        "total_assets": STRICT,
        "working_capital": NORMAL,
        "balance_sheet_equation": STRICT,
    }
    
    @classmethod
    def get_tolerance(cls, check_type: str) -> float:
        """Get tolerance for a specific check type."""
        return cls.CHECK_TOLERANCES.get(check_type.lower(), cls.DEFAULT)
    
    @classmethod
    def is_within_tolerance(
        cls, 
        value_a: float, 
        value_b: float, 
        tolerance: float | None = None
    ) -> bool:
        """
        Check if two values are within tolerance of each other.
        
        Args:
            value_a: First value
            value_b: Second value
            tolerance: Tolerance level (proportion). Uses DEFAULT if not specified.
            
        Returns:
            True if values are within tolerance
        """
        if tolerance is None:
            tolerance = cls.DEFAULT
        
        if value_a == 0 and value_b == 0:
            return True
        
        base = max(abs(value_a), abs(value_b))
        if base == 0:
            return True
        
        difference = abs(value_a - value_b)
        return (difference / base) <= tolerance


# ============================================================================
# CURRENCY CONFIGURATION
# ============================================================================

DEFAULT_CURRENCY: Final[str] = "SGD"

SUPPORTED_CURRENCIES: Final[frozenset[str]] = frozenset({
    "SGD", "USD", "EUR", "GBP", "JPY", "CNY", "HKD", "AUD",
    "MYR", "IDR", "THB", "INR", "KRW", "NZD", "PHP", "VND",
})

CURRENCY_SYMBOLS: Final[dict[str, str]] = {
    "SGD": "S$", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥",
    "CNY": "¥", "HKD": "HK$", "AUD": "A$", "MYR": "RM", "IDR": "Rp",
    "THB": "฿", "INR": "₹", "KRW": "₩", "NZD": "NZ$", "PHP": "₱", "VND": "₫",
}

ZERO_DECIMAL_CURRENCIES: Final[frozenset[str]] = frozenset({"JPY", "IDR", "KRW", "VND"})


# ============================================================================
# SINGAPORE SMB CONTEXT
# ============================================================================

class SingaporeConstants:
    """Singapore-specific financial constants and thresholds."""
    
    GST_RATE: Final[Decimal] = Decimal("0.09")  # 9% as of 2024
    
    # SFRS for Small Entities thresholds
    SFRS_SMALL_ENTITY_REVENUE: Final[int] = 10_000_000    # S$10M
    SFRS_SMALL_ENTITY_ASSETS: Final[int] = 10_000_000     # S$10M
    SFRS_SMALL_ENTITY_EMPLOYEES: Final[int] = 50
    
    # SME definition thresholds
    SME_ANNUAL_SALES: Final[int] = 100_000_000  # S$100M
    SME_EMPLOYEES: Final[int] = 200
    
    # Common financial year end months
    COMMON_FYE_MONTHS: Final[list[int]] = [12, 3, 6]
    
    @classmethod
    def calculate_gst_exclusive(cls, gst_inclusive: Decimal | float | int | str) -> Decimal:
        """Convert GST-inclusive amount to GST-exclusive."""
        amount = gst_inclusive if isinstance(gst_inclusive, Decimal) else Decimal(str(gst_inclusive))
        return amount / (Decimal("1") + cls.GST_RATE)
    
    @classmethod
    def calculate_gst_amount(cls, gst_exclusive: Decimal | float | int | str) -> Decimal:
        """Calculate GST on a GST-exclusive amount."""
        amount = gst_exclusive if isinstance(gst_exclusive, Decimal) else Decimal(str(gst_exclusive))
        return amount * cls.GST_RATE


# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================

MIN_PERIODS_FOR_TREND: Final[int] = 3
DEFAULT_FORECAST_PERIODS: Final[int] = 3
MAX_ANALYSIS_PERIODS: Final[int] = 10
DAYS_IN_YEAR: Final[int] = 365
DAYS_IN_MONTH: Final[float] = 30.44
MONTHS_IN_YEAR: Final[int] = 12


# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================

MAX_MONETARY_VALUE: Final[float] = 1e15
ZERO_THRESHOLD: Final[float] = 1e-10


# ============================================================================
# METRIC FORMULAS (from V1 - valuable for documentation)
# ============================================================================

METRIC_FORMULAS: Final[dict[str, str]] = {
    "gross_profit_margin": "(Revenue - COGS) / Revenue × 100",
    "operating_profit_margin": "(Revenue - COGS - OpEx) / Revenue × 100",
    "net_profit_margin": "Net Income / Revenue × 100",
    "ebitda_margin": "EBITDA / Revenue × 100",
    "roa": "Net Income / Average Total Assets × 100",
    "roe": "Net Income / Average Shareholders' Equity × 100",
    "roce": "EBIT / (Total Assets - Current Liabilities) × 100",
    "current_ratio": "Current Assets / Current Liabilities",
    "quick_ratio": "(Current Assets - Inventory) / Current Liabilities",
    "cash_ratio": "Cash and Equivalents / Current Liabilities",
    "working_capital": "Current Assets - Current Liabilities",
    "debt_to_equity": "Total Liabilities / Shareholders' Equity",
    "debt_to_assets": "Total Liabilities / Total Assets",
    "interest_coverage": "EBIT / Interest Expense",
    "asset_turnover": "Revenue / Average Total Assets",
    "inventory_turnover": "COGS / Average Inventory",
    "receivables_turnover": "Revenue / Average Accounts Receivable",
    "days_sales_outstanding": "365 / Receivables Turnover",
    "days_inventory_outstanding": "365 / Inventory Turnover",
}

METRIC_UNITS: Final[dict[str, str]] = {
    "gross_profit_margin": "percentage",
    "operating_profit_margin": "percentage",
    "net_profit_margin": "percentage",
    "ebitda_margin": "percentage",
    "roa": "percentage",
    "roe": "percentage",
    "roce": "percentage",
    "current_ratio": "ratio",
    "quick_ratio": "ratio",
    "cash_ratio": "ratio",
    "working_capital": "currency",
    "debt_to_equity": "ratio",
    "debt_to_assets": "ratio",
    "interest_coverage": "ratio",
    "asset_turnover": "ratio",
    "inventory_turnover": "ratio",
    "receivables_turnover": "ratio",
    "days_sales_outstanding": "days",
    "days_inventory_outstanding": "days",
}


# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

TREND_SYMBOLS: Final[dict[str, str]] = {
    "increasing": "↑",
    "decreasing": "↓",
    "stable": "→",
    "volatile": "↕",
}

STATUS_SYMBOLS: Final[dict[str, str]] = {
    "good": "✅",
    "warning": "⚠️",
    "error": "❌",
    "info": "ℹ️",
    "unknown": "❓",
}

LARGE_NUMBER_SUFFIXES: Final[list[tuple[float, str]]] = [
    (1e12, "T"),
    (1e9, "B"),
    (1e6, "M"),
    (1e3, "K"),
]

```

