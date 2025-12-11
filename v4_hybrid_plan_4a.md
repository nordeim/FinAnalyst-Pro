Hybrid Implementation — Final Batch
Files 15-25: Complete Implementation
File 15 of 25: finanalyst_tools/calculations/__init__.py
Python

# finanalyst_tools/calculations/__init__.py
"""
Financial ratio calculators for FinAnalyst-Pro Agent Tools.

This package provides calculation functions for:
- Profitability ratios (margins, ROA, ROE, ROCE)
- Liquidity ratios (current, quick, cash ratios)
- Solvency ratios (debt ratios, interest coverage) [Phase 2]
- Efficiency ratios (turnover ratios) [Phase 2]

All calculators:
- Use Decimal precision for accuracy
- Return CalculationResult with full audit trail
- Include plausibility checking
- Handle edge cases gracefully
"""

from finanalyst_tools.calculations.base import (
    BaseCalculator,
    create_calculation_result,
    extract_value,
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
    "extract_value",
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
File 16 of 25: finanalyst_tools/calculations/base.py
Python

# finanalyst_tools/calculations/base.py
"""
Base infrastructure for financial calculators.

Provides:
- BaseCalculator abstract class with common functionality
- Helper functions for creating calculation results
- Value extraction utilities
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Callable

from finanalyst_tools.config import (
    PlausibilityRanges,
    METRIC_FORMULAS,
    METRIC_UNITS,
    DECIMAL_PLACES,
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
    FinancialPeriod,
)
from finanalyst_tools.utils.math_ops import (
    to_decimal,
    safe_divide,
    round_decimal,
    calculate_percentage,
    calculate_average,
)


def extract_value(
    data: dict[str, Any] | Any,
    field_name: str,
    default: Decimal | None = None,
) -> Decimal | None:
    """
    Extract a numeric value from data, handling various input types.
    
    Args:
        data: Source data (dict, Pydantic model, or value)
        field_name: Name of field to extract
        default: Default value if field not found
        
    Returns:
        Decimal value or default
    """
    if data is None:
        return default
    
    # If it's a Pydantic model, try to get attribute
    if hasattr(data, field_name):
        value = getattr(data, field_name)
        if value is not None:
            return to_decimal(value)
    
    # If it's a dict, try to get key
    if isinstance(data, dict):
        value = data.get(field_name)
        if value is not None:
            return to_decimal(value)
    
    return default


def get_metric_unit(metric_name: str) -> MetricUnit:
    """Get the appropriate unit for a metric."""
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
    """Get the formula for a metric."""
    return METRIC_FORMULAS.get(metric_name.lower(), "Custom calculation")


def create_calculation_result(
    metric_name: str,
    value: Decimal | None,
    inputs: dict[str, Any],
    calculation_steps: list[str],
    category: MetricCategory | None = None,
    custom_formula: str | None = None,
    custom_unit: MetricUnit | None = None,
) -> CalculationResult:
    """
    Factory function to create a CalculationResult with plausibility check.
    
    Args:
        metric_name: Name of the metric
        value: Calculated value
        inputs: Dictionary of input values used
        calculation_steps: List of calculation step descriptions
        category: Optional metric category
        custom_formula: Optional custom formula (uses default if not provided)
        custom_unit: Optional custom unit (uses default if not provided)
        
    Returns:
        Complete CalculationResult with plausibility assessment
    """
    # Get formula and unit
    formula = custom_formula or get_metric_formula(metric_name)
    unit = custom_unit or get_metric_unit(metric_name)
    
    # Get plausibility range
    plausibility_range = PlausibilityRanges.get_range(metric_name)
    
    # Check plausibility
    is_plausible = True
    warnings = []
    
    if value is not None and plausibility_range is not None:
        float_value = float(value)
        min_val, max_val = plausibility_range
        
        if float_value < min_val:
            is_plausible = False
            warnings.append(
                f"Value {float_value:.2f} is below typical range "
                f"({min_val:.1f} to {max_val:.1f})"
            )
        elif float_value > max_val:
            is_plausible = False
            warnings.append(
                f"Value {float_value:.2f} is above typical range "
                f"({min_val:.1f} to {max_val:.1f})"
            )
    
    # Convert inputs to serializable format
    serializable_inputs = {}
    for key, val in inputs.items():
        if isinstance(val, Decimal):
            serializable_inputs[key] = float(val)
        elif val is None:
            serializable_inputs[key] = None
        else:
            serializable_inputs[key] = val
    
    return CalculationResult(
        metric_name=metric_name,
        value=value,
        unit=unit,
        formula=formula,
        inputs=serializable_inputs,
        calculation_steps=calculation_steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=warnings,
        category=category,
    )


class BaseCalculator(ABC):
    """
    Abstract base class for financial calculators.
    
    Provides common functionality:
    - Step logging
    - Plausibility checking
    - Result formatting
    - Warning accumulation
    """
    
    def __init__(self, category: MetricCategory):
        """
        Initialize the calculator.
        
        Args:
            category: The category of metrics this calculator produces
        """
        self.category = category
        self._steps: list[str] = []
        self._warnings: list[str] = []
        self._inputs: dict[str, Any] = {}
    
    def _reset(self) -> None:
        """Reset internal state for new calculation."""
        self._steps = []
        self._warnings = []
        self._inputs = {}
    
    def _add_step(self, description: str) -> None:
        """Add a calculation step."""
        step_num = len(self._steps) + 1
        self._steps.append(f"Step {step_num}: {description}")
    
    def _add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self._warnings.append(warning)
    
    def _record_input(self, name: str, value: Any) -> None:
        """Record an input value."""
        if isinstance(value, Decimal):
            self._inputs[name] = float(value)
        else:
            self._inputs[name] = value
    
    def _create_result(
        self,
        metric_name: str,
        value: Decimal | None,
        formula: str | None = None,
        unit: MetricUnit | None = None,
    ) -> CalculationResult:
        """Create a calculation result with current state."""
        result = create_calculation_result(
            metric_name=metric_name,
            value=value,
            inputs=self._inputs.copy(),
            calculation_steps=self._steps.copy(),
            category=self.category,
            custom_formula=formula,
            custom_unit=unit,
        )
        
        # Add any accumulated warnings
        for warning in self._warnings:
            result.add_warning(warning)
        
        return result
    
    @abstractmethod
    def calculate_all(
        self,
        data: FinancialStatementSet | dict[str, Any],
        prior_data: FinancialStatementSet | dict[str, Any] | None = None,
    ) -> MetricCollection:
        """
        Calculate all metrics for this category.
        
        Args:
            data: Current period financial data
            prior_data: Prior period data (for averages)
            
        Returns:
            MetricCollection with all calculated metrics
        """
        pass
File 17 of 25: finanalyst_tools/calculations/profitability.py
Python

# finanalyst_tools/calculations/profitability.py
"""
Profitability ratio calculators.

Provides calculations for:
- Gross Profit Margin
- Operating Profit Margin
- Net Profit Margin
- EBITDA Margin
- Return on Assets (ROA)
- Return on Equity (ROE)
- Return on Capital Employed (ROCE)
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
    FinancialStatementSet,
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
        revenue: Total revenue / net sales
        cost_of_goods_sold: Cost of goods sold
        
    Returns:
        CalculationResult with margin percentage
    """
    steps = []
    inputs = {}
    
    # Convert inputs
    rev = to_decimal(revenue)
    cogs = to_decimal(cost_of_goods_sold)
    
    inputs["revenue"] = float(rev)
    inputs["cost_of_goods_sold"] = float(cogs)
    
    steps.append(f"Revenue = {rev:,.2f}")
    steps.append(f"COGS = {cogs:,.2f}")
    
    # Calculate gross profit
    gross_profit = rev - cogs
    steps.append(f"Gross Profit = Revenue - COGS = {rev:,.2f} - {cogs:,.2f} = {gross_profit:,.2f}")
    
    # Calculate margin
    if is_effectively_zero(rev):
        steps.append("Revenue is zero, cannot calculate margin")
        value = None
    else:
        margin = (gross_profit / rev) * 100
        value = round_decimal(margin, DECIMAL_PLACES["percentage"])
        steps.append(f"Gross Profit Margin = (Gross Profit / Revenue) × 100 = ({gross_profit:,.2f} / {rev:,.2f}) × 100 = {value:.2f}%")
    
    return create_calculation_result(
        metric_name="Gross Profit Margin",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.PROFITABILITY,
        custom_formula="(Revenue - COGS) / Revenue × 100",
        custom_unit=MetricUnit.PERCENTAGE,
    )


def calculate_operating_profit_margin(
    revenue: Decimal | float | int,
    cost_of_goods_sold: Decimal | float | int,
    operating_expenses: Decimal | float | int | None = None,
    selling_general_admin: Decimal | float | int | None = None,
    marketing_expenses: Decimal | float | int | None = None,
    research_development: Decimal | float | int | None = None,
    depreciation_amortization: Decimal | float | int | None = None,
    other_operating_expenses: Decimal | float | int | None = None,
) -> CalculationResult:
    """
    Calculate Operating Profit Margin.
    
    Formula: (Revenue - COGS - OpEx) / Revenue × 100
    
    Args:
        revenue: Total revenue
        cost_of_goods_sold: COGS
        operating_expenses: Total OpEx (if aggregated)
        selling_general_admin: SG&A expenses
        marketing_expenses: Marketing expenses
        research_development: R&D expenses
        depreciation_amortization: D&A
        other_operating_expenses: Other OpEx
        
    Returns:
        CalculationResult with margin percentage
    """
    steps = []
    inputs = {}
    
    # Convert inputs
    rev = to_decimal(revenue)
    cogs = to_decimal(cost_of_goods_sold)
    
    inputs["revenue"] = float(rev)
    inputs["cost_of_goods_sold"] = float(cogs)
    
    steps.append(f"Revenue = {rev:,.2f}")
    steps.append(f"COGS = {cogs:,.2f}")
    
    # Calculate total operating expenses
    if operating_expenses is not None:
        opex = to_decimal(operating_expenses)
        inputs["operating_expenses"] = float(opex)
        steps.append(f"Operating Expenses (provided) = {opex:,.2f}")
    else:
        opex = Decimal("0")
        components = [
            ("SG&A", selling_general_admin),
            ("Marketing", marketing_expenses),
            ("R&D", research_development),
            ("D&A", depreciation_amortization),
            ("Other OpEx", other_operating_expenses),
        ]
        
        for name, val in components:
            if val is not None:
                dec_val = to_decimal(val)
                opex += dec_val
                inputs[name.lower().replace("&", "_")] = float(dec_val)
                steps.append(f"{name} = {dec_val:,.2f}")
        
        steps.append(f"Total Operating Expenses = {opex:,.2f}")
    
    # Calculate operating profit
    gross_profit = rev - cogs
    operating_profit = gross_profit - opex
    steps.append(f"Operating Profit = Gross Profit - OpEx = {gross_profit:,.2f} - {opex:,.2f} = {operating_profit:,.2f}")
    
    # Calculate margin
    if is_effectively_zero(rev):
        steps.append("Revenue is zero, cannot calculate margin")
        value = None
    else:
        margin = (operating_profit / rev) * 100
        value = round_decimal(margin, DECIMAL_PLACES["percentage"])
        steps.append(f"Operating Profit Margin = (Operating Profit / Revenue) × 100 = {value:.2f}%")
    
    return create_calculation_result(
        metric_name="Operating Profit Margin",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.PROFITABILITY,
        custom_formula="(Revenue - COGS - OpEx) / Revenue × 100",
        custom_unit=MetricUnit.PERCENTAGE,
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
        net_income: Net income / net profit
        
    Returns:
        CalculationResult with margin percentage
    """
    steps = []
    inputs = {}
    
    rev = to_decimal(revenue)
    ni = to_decimal(net_income)
    
    inputs["revenue"] = float(rev)
    inputs["net_income"] = float(ni)
    
    steps.append(f"Revenue = {rev:,.2f}")
    steps.append(f"Net Income = {ni:,.2f}")
    
    if is_effectively_zero(rev):
        steps.append("Revenue is zero, cannot calculate margin")
        value = None
    else:
        margin = (ni / rev) * 100
        value = round_decimal(margin, DECIMAL_PLACES["percentage"])
        steps.append(f"Net Profit Margin = (Net Income / Revenue) × 100 = ({ni:,.2f} / {rev:,.2f}) × 100 = {value:.2f}%")
    
    result = create_calculation_result(
        metric_name="Net Profit Margin",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.PROFITABILITY,
        custom_formula="Net Income / Revenue × 100",
        custom_unit=MetricUnit.PERCENTAGE,
    )
    
    # Add warning for unusual values
    if value is not None and float(value) >= 100:
        result.add_warning("Net margin >= 100% is unusual - verify data accuracy")
    
    return result


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
        CalculationResult with margin percentage
    """
    steps = []
    inputs = {}
    
    rev = to_decimal(revenue)
    ebitda_val = to_decimal(ebitda)
    
    inputs["revenue"] = float(rev)
    inputs["ebitda"] = float(ebitda_val)
    
    steps.append(f"Revenue = {rev:,.2f}")
    steps.append(f"EBITDA = {ebitda_val:,.2f}")
    
    if is_effectively_zero(rev):
        steps.append("Revenue is zero, cannot calculate margin")
        value = None
    else:
        margin = (ebitda_val / rev) * 100
        value = round_decimal(margin, DECIMAL_PLACES["percentage"])
        steps.append(f"EBITDA Margin = (EBITDA / Revenue) × 100 = ({ebitda_val:,.2f} / {rev:,.2f}) × 100 = {value:.2f}%")
    
    return create_calculation_result(
        metric_name="EBITDA Margin",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.PROFITABILITY,
        custom_formula="EBITDA / Revenue × 100",
        custom_unit=MetricUnit.PERCENTAGE,
    )


def calculate_return_on_assets(
    net_income: Decimal | float | int,
    total_assets_begin: Decimal | float | int,
    total_assets_end: Decimal | float | int | None = None,
) -> CalculationResult:
    """
    Calculate Return on Assets (ROA).
    
    Formula: Net Income / Average Total Assets × 100
    
    Args:
        net_income: Net income for the period
        total_assets_begin: Total assets at beginning of period
        total_assets_end: Total assets at end of period (optional, uses begin if not provided)
        
    Returns:
        CalculationResult with ROA percentage
    """
    steps = []
    inputs = {}
    
    ni = to_decimal(net_income)
    assets_begin = to_decimal(total_assets_begin)
    assets_end = to_decimal(total_assets_end) if total_assets_end is not None else assets_begin
    
    inputs["net_income"] = float(ni)
    inputs["total_assets_begin"] = float(assets_begin)
    inputs["total_assets_end"] = float(assets_end)
    
    steps.append(f"Net Income = {ni:,.2f}")
    steps.append(f"Total Assets (Beginning) = {assets_begin:,.2f}")
    steps.append(f"Total Assets (Ending) = {assets_end:,.2f}")
    
    # Calculate average assets
    avg_assets = (assets_begin + assets_end) / 2
    steps.append(f"Average Total Assets = ({assets_begin:,.2f} + {assets_end:,.2f}) / 2 = {avg_assets:,.2f}")
    
    if is_effectively_zero(avg_assets):
        steps.append("Average assets is zero, cannot calculate ROA")
        value = None
    else:
        roa = (ni / avg_assets) * 100
        value = round_decimal(roa, DECIMAL_PLACES["percentage"])
        steps.append(f"ROA = (Net Income / Average Total Assets) × 100 = ({ni:,.2f} / {avg_assets:,.2f}) × 100 = {value:.2f}%")
    
    return create_calculation_result(
        metric_name="Return on Assets",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.PROFITABILITY,
        custom_formula="Net Income / Average Total Assets × 100",
        custom_unit=MetricUnit.PERCENTAGE,
    )


def calculate_return_on_equity(
    net_income: Decimal | float | int,
    shareholders_equity_begin: Decimal | float | int,
    shareholders_equity_end: Decimal | float | int | None = None,
) -> CalculationResult:
    """
    Calculate Return on Equity (ROE).
    
    Formula: Net Income / Average Shareholders' Equity × 100
    
    Args:
        net_income: Net income for the period
        shareholders_equity_begin: Equity at beginning of period
        shareholders_equity_end: Equity at end of period (optional)
        
    Returns:
        CalculationResult with ROE percentage
    """
    steps = []
    inputs = {}
    warnings = []
    
    ni = to_decimal(net_income)
    equity_begin = to_decimal(shareholders_equity_begin)
    equity_end = to_decimal(shareholders_equity_end) if shareholders_equity_end is not None else equity_begin
    
    inputs["net_income"] = float(ni)
    inputs["shareholders_equity_begin"] = float(equity_begin)
    inputs["shareholders_equity_end"] = float(equity_end)
    
    steps.append(f"Net Income = {ni:,.2f}")
    steps.append(f"Shareholders' Equity (Beginning) = {equity_begin:,.2f}")
    steps.append(f"Shareholders' Equity (Ending) = {equity_end:,.2f}")
    
    # Check for negative equity
    if equity_begin < 0 or equity_end < 0:
        warnings.append("Negative shareholders' equity detected - ROE interpretation may be misleading")
    
    # Calculate average equity
    avg_equity = (equity_begin + equity_end) / 2
    steps.append(f"Average Shareholders' Equity = ({equity_begin:,.2f} + {equity_end:,.2f}) / 2 = {avg_equity:,.2f}")
    
    if is_effectively_zero(avg_equity):
        steps.append("Average equity is zero, cannot calculate ROE")
        value = None
    else:
        roe = (ni / avg_equity) * 100
        value = round_decimal(roe, DECIMAL_PLACES["percentage"])
        steps.append(f"ROE = (Net Income / Average Equity) × 100 = ({ni:,.2f} / {avg_equity:,.2f}) × 100 = {value:.2f}%")
    
    result = create_calculation_result(
        metric_name="Return on Equity",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.PROFITABILITY,
        custom_formula="Net Income / Average Shareholders' Equity × 100",
        custom_unit=MetricUnit.PERCENTAGE,
    )
    
    for warning in warnings:
        result.add_warning(warning)
    
    return result


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
    inputs = {}
    
    ebit_val = to_decimal(ebit)
    assets = to_decimal(total_assets)
    curr_liab = to_decimal(current_liabilities)
    
    inputs["ebit"] = float(ebit_val)
    inputs["total_assets"] = float(assets)
    inputs["current_liabilities"] = float(curr_liab)
    
    steps.append(f"EBIT = {ebit_val:,.2f}")
    steps.append(f"Total Assets = {assets:,.2f}")
    steps.append(f"Current Liabilities = {curr_liab:,.2f}")
    
    # Calculate capital employed
    capital_employed = assets - curr_liab
    steps.append(f"Capital Employed = Total Assets - Current Liabilities = {assets:,.2f} - {curr_liab:,.2f} = {capital_employed:,.2f}")
    
    if is_effectively_zero(capital_employed):
        steps.append("Capital employed is zero, cannot calculate ROCE")
        value = None
    else:
        roce = (ebit_val / capital_employed) * 100
        value = round_decimal(roce, DECIMAL_PLACES["percentage"])
        steps.append(f"ROCE = (EBIT / Capital Employed) × 100 = ({ebit_val:,.2f} / {capital_employed:,.2f}) × 100 = {value:.2f}%")
    
    return create_calculation_result(
        metric_name="Return on Capital Employed",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.PROFITABILITY,
        custom_formula="EBIT / (Total Assets - Current Liabilities) × 100",
        custom_unit=MetricUnit.PERCENTAGE,
    )


def calculate_all_profitability_metrics(
    income_statement: IncomeStatementData,
    balance_sheet: BalanceSheetData,
    prior_balance_sheet: BalanceSheetData | None = None,
) -> MetricCollection:
    """
    Calculate all profitability metrics from financial statements.
    
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
    
    # Get prior period values for averages
    prior_assets = prior_balance_sheet.calculated_total_assets if prior_balance_sheet else None
    prior_equity = prior_balance_sheet.calculated_shareholders_equity if prior_balance_sheet else None
    
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
        operating_expenses=income_statement.operating_expenses,
        selling_general_admin=income_statement.selling_general_admin,
        marketing_expenses=income_statement.marketing_expenses,
        research_development=income_statement.research_development,
        depreciation_amortization=income_statement.depreciation_amortization,
        other_operating_expenses=income_statement.other_operating_expenses,
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
        total_assets_begin=prior_assets or balance_sheet.calculated_total_assets,
        total_assets_end=balance_sheet.calculated_total_assets,
    )
    collection.add_metric(roa)
    
    # ROE
    roe = calculate_return_on_equity(
        net_income=income_statement.calculated_net_income,
        shareholders_equity_begin=prior_equity or balance_sheet.calculated_shareholders_equity,
        shareholders_equity_end=balance_sheet.calculated_shareholders_equity,
    )
    collection.add_metric(roe)
    
    # ROCE
    roce = calculate_return_on_capital_employed(
        ebit=income_statement.operating_income,
        total_assets=balance_sheet.calculated_total_assets,
        current_liabilities=balance_sheet.calculated_current_liabilities,
    )
    collection.add_metric(roce)
    
    return collection


class ProfitabilityCalculator(BaseCalculator):
    """Class-based profitability calculator."""
    
    def __init__(self):
        super().__init__(category=MetricCategory.PROFITABILITY)
    
    def calculate_all(
        self,
        data: FinancialStatementSet | dict[str, Any],
        prior_data: FinancialStatementSet | dict[str, Any] | None = None,
    ) -> MetricCollection:
        """Calculate all profitability metrics."""
        if isinstance(data, FinancialStatementSet):
            prior_bs = prior_data.balance_sheet if isinstance(prior_data, FinancialStatementSet) else None
            return calculate_all_profitability_metrics(
                income_statement=data.income_statement,
                balance_sheet=data.balance_sheet,
                prior_balance_sheet=prior_bs,
            )
        else:
            raise ValueError("Dictionary input not yet supported - use FinancialStatementSet")
File 18 of 25: finanalyst_tools/calculations/liquidity.py
Python

# finanalyst_tools/calculations/liquidity.py
"""
Liquidity ratio calculators.

Provides calculations for:
- Current Ratio
- Quick Ratio (Acid Test)
- Cash Ratio
- Working Capital
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from finanalyst_tools.config import DECIMAL_PLACES
from finanalyst_tools.models.analysis_results import (
    CalculationResult,
    MetricUnit,
    MetricCategory,
    MetricCollection,
)
from finanalyst_tools.models.financial_statements import (
    BalanceSheetData,
    FinancialStatementSet,
    FinancialPeriod,
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
    
    Args:
        current_assets: Total current assets
        current_liabilities: Total current liabilities
        
    Returns:
        CalculationResult with ratio value
    """
    steps = []
    inputs = {}
    warnings = []
    
    ca = to_decimal(current_assets)
    cl = to_decimal(current_liabilities)
    
    inputs["current_assets"] = float(ca)
    inputs["current_liabilities"] = float(cl)
    
    steps.append(f"Current Assets = {ca:,.2f}")
    steps.append(f"Current Liabilities = {cl:,.2f}")
    
    if is_effectively_zero(cl):
        steps.append("Current liabilities is zero, cannot calculate ratio")
        value = None
    else:
        ratio = ca / cl
        value = round_decimal(ratio, DECIMAL_PLACES["ratio"])
        steps.append(f"Current Ratio = Current Assets / Current Liabilities = {ca:,.2f} / {cl:,.2f} = {value:.4f}")
        
        # Add interpretation
        if float(value) < 1.0:
            warnings.append("Current ratio < 1.0 indicates potential liquidity risk - current liabilities exceed current assets")
        elif float(value) < 1.5:
            warnings.append("Current ratio between 1.0 and 1.5 is acceptable but warrants monitoring")
        elif float(value) > 3.0:
            warnings.append("Current ratio > 3.0 may indicate inefficient use of working capital")
    
    result = create_calculation_result(
        metric_name="Current Ratio",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.LIQUIDITY,
        custom_formula="Current Assets / Current Liabilities",
        custom_unit=MetricUnit.RATIO,
    )
    
    for warning in warnings:
        result.add_warning(warning)
    
    return result


def calculate_quick_ratio(
    current_assets: Decimal | float | int,
    inventory: Decimal | float | int,
    current_liabilities: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Quick Ratio (Acid Test Ratio).
    
    Formula: (Current Assets - Inventory) / Current Liabilities
    
    Args:
        current_assets: Total current assets
        inventory: Inventory value
        current_liabilities: Total current liabilities
        
    Returns:
        CalculationResult with ratio value
    """
    steps = []
    inputs = {}
    warnings = []
    
    ca = to_decimal(current_assets)
    inv = to_decimal(inventory)
    cl = to_decimal(current_liabilities)
    
    inputs["current_assets"] = float(ca)
    inputs["inventory"] = float(inv)
    inputs["current_liabilities"] = float(cl)
    
    steps.append(f"Current Assets = {ca:,.2f}")
    steps.append(f"Inventory = {inv:,.2f}")
    steps.append(f"Current Liabilities = {cl:,.2f}")
    
    # Calculate quick assets
    quick_assets = ca - inv
    steps.append(f"Quick Assets = Current Assets - Inventory = {ca:,.2f} - {inv:,.2f} = {quick_assets:,.2f}")
    
    if is_effectively_zero(cl):
        steps.append("Current liabilities is zero, cannot calculate ratio")
        value = None
    else:
        ratio = quick_assets / cl
        value = round_decimal(ratio, DECIMAL_PLACES["ratio"])
        steps.append(f"Quick Ratio = Quick Assets / Current Liabilities = {quick_assets:,.2f} / {cl:,.2f} = {value:.4f}")
        
        # Add interpretation
        if float(value) < 0.5:
            warnings.append("Quick ratio < 0.5 indicates significant liquidity risk")
        elif float(value) < 1.0:
            warnings.append("Quick ratio < 1.0 suggests reliance on inventory to meet short-term obligations")
    
    result = create_calculation_result(
        metric_name="Quick Ratio",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.LIQUIDITY,
        custom_formula="(Current Assets - Inventory) / Current Liabilities",
        custom_unit=MetricUnit.RATIO,
    )
    
    for warning in warnings:
        result.add_warning(warning)
    
    return result


def calculate_cash_ratio(
    cash_and_equivalents: Decimal | float | int,
    current_liabilities: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Cash Ratio.
    
    Formula: Cash and Equivalents / Current Liabilities
    
    Args:
        cash_and_equivalents: Cash and cash equivalents
        current_liabilities: Total current liabilities
        
    Returns:
        CalculationResult with ratio value
    """
    steps = []
    inputs = {}
    warnings = []
    
    cash = to_decimal(cash_and_equivalents)
    cl = to_decimal(current_liabilities)
    
    inputs["cash_and_equivalents"] = float(cash)
    inputs["current_liabilities"] = float(cl)
    
    steps.append(f"Cash and Equivalents = {cash:,.2f}")
    steps.append(f"Current Liabilities = {cl:,.2f}")
    
    if is_effectively_zero(cl):
        steps.append("Current liabilities is zero, cannot calculate ratio")
        value = None
    else:
        ratio = cash / cl
        value = round_decimal(ratio, DECIMAL_PLACES["ratio"])
        steps.append(f"Cash Ratio = Cash / Current Liabilities = {cash:,.2f} / {cl:,.2f} = {value:.4f}")
        
        # Add interpretation
        if float(value) < 0.1:
            warnings.append("Cash ratio < 0.1 indicates very limited immediate liquidity")
        elif float(value) > 1.0:
            warnings.append("Cash ratio > 1.0 may indicate excess cash that could be deployed for growth")
    
    result = create_calculation_result(
        metric_name="Cash Ratio",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.LIQUIDITY,
        custom_formula="Cash and Equivalents / Current Liabilities",
        custom_unit=MetricUnit.RATIO,
    )
    
    for warning in warnings:
        result.add_warning(warning)
    
    return result


def calculate_working_capital(
    current_assets: Decimal | float | int,
    current_liabilities: Decimal | float | int,
    currency: str = "SGD",
) -> CalculationResult:
    """
    Calculate Working Capital.
    
    Formula: Current Assets - Current Liabilities
    
    Args:
        current_assets: Total current assets
        current_liabilities: Total current liabilities
        currency: Currency code for display
        
    Returns:
        CalculationResult with currency value
    """
    steps = []
    inputs = {}
    warnings = []
    
    ca = to_decimal(current_assets)
    cl = to_decimal(current_liabilities)
    
    inputs["current_assets"] = float(ca)
    inputs["current_liabilities"] = float(cl)
    inputs["currency"] = currency
    
    steps.append(f"Current Assets = {ca:,.2f}")
    steps.append(f"Current Liabilities = {cl:,.2f}")
    
    # Calculate working capital
    wc = ca - cl
    value = round_decimal(wc, DECIMAL_PLACES["currency"])
    steps.append(f"Working Capital = Current Assets - Current Liabilities = {ca:,.2f} - {cl:,.2f} = {value:,.2f}")
    
    # Add interpretation
    if float(value) < 0:
        warnings.append("Negative working capital indicates current liabilities exceed current assets - liquidity risk")
    elif is_effectively_zero(value):
        warnings.append("Zero working capital indicates no buffer for unexpected expenses")
    
    result = create_calculation_result(
        metric_name="Working Capital",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.LIQUIDITY,
        custom_formula="Current Assets - Current Liabilities",
        custom_unit=MetricUnit.CURRENCY,
    )
    
    for warning in warnings:
        result.add_warning(warning)
    
    return result


def calculate_all_liquidity_metrics(
    balance_sheet: BalanceSheetData,
) -> MetricCollection:
    """
    Calculate all liquidity metrics from balance sheet.
    
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
    """Class-based liquidity calculator."""
    
    def __init__(self):
        super().__init__(category=MetricCategory.LIQUIDITY)
    
    def calculate_all(
        self,
        data: FinancialStatementSet | dict[str, Any],
        prior_data: FinancialStatementSet | dict[str, Any] | None = None,
    ) -> MetricCollection:
        """Calculate all liquidity metrics."""
        if isinstance(data, FinancialStatementSet):
            return calculate_all_liquidity_metrics(
                balance_sheet=data.balance_sheet,
            )
        else:
            raise ValueError("Dictionary input not yet supported - use FinancialStatementSet")
File 19 of 25: finanalyst_tools/tool_registry.py
Python

# finanalyst_tools/tool_registry.py
"""
Central registry of all tools with metadata for LLM function calling.

Provides:
- Tool definitions with parameters and descriptions
- OpenAI function calling schema generation
- Anthropic tool schema generation
- Tool discovery and lookup
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import json


class ToolCategory(str, Enum):
    """Categories for organizing tools."""
    VALIDATION = "validation"
    PROFITABILITY = "profitability"
    LIQUIDITY = "liquidity"
    SOLVENCY = "solvency"
    EFFICIENCY = "efficiency"
    GROWTH = "growth"
    VALUATION = "valuation"
    TREND = "trend"
    FORECAST = "forecast"
    RECONCILIATION = "reconciliation"
    REPORTING = "reporting"
    UTILITY = "utility"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "number", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None
    
    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolDefinition:
    """Complete definition of a tool for LLM function calling."""
    name: str
    description: str
    category: ToolCategory
    parameters: list[ToolParameter]
    returns: str
    function: Callable[..., Any]
    example: str | None = None
    
    def to_openai_schema(self) -> dict[str, Any]:
        """
        Convert to OpenAI function calling schema.
        
        Returns:
            Dictionary in OpenAI function format
        """
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
    
    def to_anthropic_schema(self) -> dict[str, Any]:
        """
        Convert to Anthropic tool schema.
        
        Returns:
            Dictionary in Anthropic tool format
        """
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for documentation."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                }
                for p in self.parameters
            ],
            "returns": self.returns,
            "example": self.example,
        }


class ToolRegistry:
    """
    Central registry for all available tools.
    
    Provides methods for:
    - Registering tools
    - Looking up tools by name
    - Listing tools by category
    - Generating LLM-compatible schemas
    """
    
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._register_all_tools()
    
    def register(self, tool: ToolDefinition) -> None:
        """
        Register a tool.
        
        Args:
            tool: Tool definition to register
        """
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> ToolDefinition | None:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool definition or None if not found
        """
        return self._tools.get(name)
    
    def list_tools(
        self,
        category: ToolCategory | None = None,
    ) -> list[ToolDefinition]:
        """
        List all tools, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of matching tool definitions
        """
        if category is None:
            return list(self._tools.values())
        return [t for t in self._tools.values() if t.category == category]
    
    def get_tool_names(
        self,
        category: ToolCategory | None = None,
    ) -> list[str]:
        """Get list of tool names."""
        tools = self.list_tools(category)
        return [t.name for t in tools]
    
    def get_openai_tools(
        self,
        categories: list[ToolCategory] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get OpenAI function calling schemas for tools.
        
        Args:
            categories: Optional list of categories to include
            
        Returns:
            List of OpenAI function schemas
        """
        tools = self._tools.values()
        if categories:
            tools = [t for t in tools if t.category in categories]
        return [t.to_openai_schema() for t in tools]
    
    def get_anthropic_tools(
        self,
        categories: list[ToolCategory] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get Anthropic tool schemas.
        
        Args:
            categories: Optional list of categories to include
            
        Returns:
            List of Anthropic tool schemas
        """
        tools = self._tools.values()
        if categories:
            tools = [t for t in tools if t.category in categories]
        return [t.to_anthropic_schema() for t in tools]
    
    def get_tool_descriptions(self) -> str:
        """
        Get formatted descriptions of all tools for prompt inclusion.
        
        Returns:
            Markdown-formatted tool descriptions
        """
        lines = ["# Available Financial Analysis Tools\n"]
        
        # Group by category
        categories: dict[ToolCategory, list[ToolDefinition]] = {}
        for tool in self._tools.values():
            if tool.category not in categories:
                categories[tool.category] = []
            categories[tool.category].append(tool)
        
        for category in sorted(categories.keys(), key=lambda c: c.value):
            lines.append(f"\n## {category.value.title()}\n")
            for tool in sorted(categories[category], key=lambda t: t.name):
                lines.append(f"### `{tool.name}`")
                lines.append(f"{tool.description}\n")
                if tool.parameters:
                    lines.append("**Parameters:**")
                    for param in tool.parameters:
                        req = "(required)" if param.required else "(optional)"
                        lines.append(f"- `{param.name}` ({param.type}) {req}: {param.description}")
                lines.append(f"\n**Returns:** {tool.returns}\n")
        
        return "\n".join(lines)
    
    def _register_all_tools(self) -> None:
        """Register all Phase 1 tools."""
        # Import here to avoid circular imports
        from finanalyst_tools.calculations.profitability import (
            calculate_gross_profit_margin,
            calculate_operating_profit_margin,
            calculate_net_profit_margin,
            calculate_ebitda_margin,
            calculate_return_on_assets,
            calculate_return_on_equity,
            calculate_return_on_capital_employed,
        )
        from finanalyst_tools.calculations.liquidity import (
            calculate_current_ratio,
            calculate_quick_ratio,
            calculate_cash_ratio,
            calculate_working_capital,
        )
        from finanalyst_tools.validation.schema_validator import (
            validate_financial_data_completeness,
        )
        
        # Validation Tools
        self.register(ToolDefinition(
            name="validate_financial_data",
            description="Validate that financial data is complete and properly structured for the requested analysis type",
            category=ToolCategory.VALIDATION,
            parameters=[
                ToolParameter("income_statement", "object", "Income statement data", required=False),
                ToolParameter("balance_sheet", "object", "Balance sheet data", required=False),
                ToolParameter("cash_flow", "object", "Cash flow statement data", required=False),
                ToolParameter("analysis_type", "string", "Type of analysis to validate for", 
                            required=True, enum=["profitability", "liquidity", "solvency", "efficiency", "comprehensive"]),
            ],
            returns="ValidationResult with any issues found",
            function=validate_financial_data_completeness,
            example='validate_financial_data(income_statement={...}, balance_sheet={...}, analysis_type="profitability")',
        ))
        
        # Profitability Tools
        self.register(ToolDefinition(
            name="calculate_gross_profit_margin",
            description="Calculate Gross Profit Margin: (Revenue - COGS) / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("revenue", "number", "Total revenue or net sales"),
                ToolParameter("cost_of_goods_sold", "number", "Cost of goods sold"),
            ],
            returns="CalculationResult with margin percentage and audit trail",
            function=calculate_gross_profit_margin,
            example="calculate_gross_profit_margin(revenue=1000000, cost_of_goods_sold=600000)",
        ))
        
        self.register(ToolDefinition(
            name="calculate_operating_profit_margin",
            description="Calculate Operating Profit Margin: (Revenue - COGS - OpEx) / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("revenue", "number", "Total revenue"),
                ToolParameter("cost_of_goods_sold", "number", "Cost of goods sold"),
                ToolParameter("operating_expenses", "number", "Total operating expenses", required=False),
                ToolParameter("selling_general_admin", "number", "SG&A expenses", required=False),
                ToolParameter("marketing_expenses", "number", "Marketing expenses", required=False),
                ToolParameter("research_development", "number", "R&D expenses", required=False),
                ToolParameter("depreciation_amortization", "number", "D&A", required=False),
            ],
            returns="CalculationResult with margin percentage and audit trail",
            function=calculate_operating_profit_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_net_profit_margin",
            description="Calculate Net Profit Margin: Net Income / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("revenue", "number", "Total revenue"),
                ToolParameter("net_income", "number", "Net income / net profit"),
            ],
            returns="CalculationResult with margin percentage and audit trail",
            function=calculate_net_profit_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_ebitda_margin",
            description="Calculate EBITDA Margin: EBITDA / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("revenue", "number", "Total revenue"),
                ToolParameter("ebitda", "number", "EBITDA"),
            ],
            returns="CalculationResult with margin percentage and audit trail",
            function=calculate_ebitda_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_return_on_assets",
            description="Calculate ROA: Net Income / Average Total Assets × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("net_income", "number", "Net income for the period"),
                ToolParameter("total_assets_begin", "number", "Total assets at period start"),
                ToolParameter("total_assets_end", "number", "Total assets at period end", required=False),
            ],
            returns="CalculationResult with ROA percentage and audit trail",
            function=calculate_return_on_assets,
        ))
        
        self.register(ToolDefinition(
            name="calculate_return_on_equity",
            description="Calculate ROE: Net Income / Average Shareholders' Equity × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("net_income", "number", "Net income for the period"),
                ToolParameter("shareholders_equity_begin", "number", "Equity at period start"),
                ToolParameter("shareholders_equity_end", "number", "Equity at period end", required=False),
            ],
            returns="CalculationResult with ROE percentage and audit trail",
            function=calculate_return_on_equity,
        ))
        
        self.register(ToolDefinition(
            name="calculate_return_on_capital_employed",
            description="Calculate ROCE: EBIT / (Total Assets - Current Liabilities) × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter("ebit", "number", "Earnings Before Interest and Taxes"),
                ToolParameter("total_assets", "number", "Total assets"),
                ToolParameter("current_liabilities", "number", "Current liabilities"),
            ],
            returns="CalculationResult with ROCE percentage and audit trail",
            function=calculate_return_on_capital_employed,
        ))
        
        # Liquidity Tools
        self.register(ToolDefinition(
            name="calculate_current_ratio",
            description="Calculate Current Ratio: Current Assets / Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter("current_assets", "number", "Total current assets"),
                ToolParameter("current_liabilities", "number", "Total current liabilities"),
            ],
            returns="CalculationResult with ratio value and interpretation",
            function=calculate_current_ratio,
        ))
        
        self.register(ToolDefinition(
            name="calculate_quick_ratio",
            description="Calculate Quick Ratio: (Current Assets - Inventory) / Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter("current_assets", "number", "Total current assets"),
                ToolParameter("inventory", "number", "Inventory value"),
                ToolParameter("current_liabilities", "number", "Total current liabilities"),
            ],
            returns="CalculationResult with ratio value and interpretation",
            function=calculate_quick_ratio,
        ))
        
        self.register(ToolDefinition(
            name="calculate_cash_ratio",
            description="Calculate Cash Ratio: Cash and Equivalents / Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter("cash_and_equivalents", "number", "Cash and cash equivalents"),
                ToolParameter("current_liabilities", "number", "Total current liabilities"),
            ],
            returns="CalculationResult with ratio value and interpretation",
            function=calculate_cash_ratio,
        ))
        
        self.register(ToolDefinition(
            name="calculate_working_capital",
            description="Calculate Working Capital: Current Assets - Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter("current_assets", "number", "Total current assets"),
                ToolParameter("current_liabilities", "number", "Total current liabilities"),
                ToolParameter("currency", "string", "Currency code", required=False, default="SGD"),
            ],
            returns="CalculationResult with currency value",
            function=calculate_working_capital,
        ))


# Global singleton instance
TOOL_REGISTRY = ToolRegistry()
File 20 of 25: finanalyst_tools/dispatcher.py
Python

# finanalyst_tools/dispatcher.py
"""
Tool call dispatcher for executing tools from LLM requests.

Provides:
- Parameter validation and type coercion
- Execution timing
- Error handling and formatting
- Structured result output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any
import json
import time

from finanalyst_tools.tool_registry import TOOL_REGISTRY, ToolDefinition
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
        success: Whether execution was successful
        result: The result data (if successful)
        error: Error message (if failed)
        error_details: Additional error context
        execution_time_ms: Execution time in milliseconds
    """
    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    error_details: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data: dict[str, Any] = {
            "tool_name": self.tool_name,
            "success": self.success,
        }
        
        if self.success:
            # Serialize result
            if hasattr(self.result, "to_dict"):
                data["result"] = self.result.to_dict()
            elif isinstance(self.result, Decimal):
                data["result"] = float(self.result)
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
    
    def __str__(self) -> str:
        if self.success:
            return f"✅ {self.tool_name}: Success ({self.execution_time_ms:.1f}ms)"
        return f"❌ {self.tool_name}: {self.error}"


class ToolDispatcher:
    """
    Dispatcher for executing tool calls.
    
    Handles:
    - Tool lookup
    - Parameter validation and coercion
    - Execution with timing
    - Error handling
    """
    
    def __init__(self):
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
            parameters: Parameters to pass to the tool
            
        Returns:
            ToolCallResult with execution outcome
        """
        start_time = time.perf_counter()
        
        try:
            # Get tool definition
            tool = self.registry.get(tool_name)
            if tool is None:
                available = self.registry.get_tool_names()
                raise ToolNotFoundError(tool_name, available_tools=available)
            
            # Validate and coerce parameters
            validated_params = self._validate_and_coerce_parameters(tool, parameters)
            
            # Execute the tool function
            result = tool.function(**validated_params)
            
            # Calculate execution time
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return ToolCallResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=execution_time,
            )
            
        except ToolNotFoundError as e:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                error_details=e.to_dict(),
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )
            
        except ToolParameterError as e:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                error_details=e.to_dict(),
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )
            
        except FinAnalystError as e:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                error_details=e.to_dict(),
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )
            
        except Exception as e:
            # Wrap unexpected errors
            wrapped = ToolExecutionError(tool_name, e, parameters)
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=str(wrapped),
                error_details=wrapped.to_dict(),
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )
    
    def _validate_and_coerce_parameters(
        self,
        tool: ToolDefinition,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate parameters against tool definition and coerce types.
        
        Args:
            tool: Tool definition
            parameters: Raw parameters from caller
            
        Returns:
            Validated and coerced parameters
            
        Raises:
            ToolParameterError: If validation fails
        """
        result = {}
        
        for param in tool.parameters:
            value = parameters.get(param.name)
            
            # Check required parameters
            if value is None:
                if param.required:
                    raise ToolParameterError(
                        tool_name=tool.name,
                        parameter_name=param.name,
                        message="Required parameter is missing",
                        expected_type=param.type,
                    )
                elif param.default is not None:
                    value = param.default
                else:
                    continue  # Optional with no default, skip
            
            # Coerce types
            try:
                coerced = self._coerce_value(value, param.type, param.name, tool.name)
                result[param.name] = coerced
            except (ValueError, TypeError, InvalidOperation) as e:
                raise ToolParameterError(
                    tool_name=tool.name,
                    parameter_name=param.name,
                    message=f"Cannot convert value to {param.type}: {e}",
                    expected_type=param.type,
                    actual_value=value,
                )
        
        return result
    
    def _coerce_value(
        self,
        value: Any,
        expected_type: str,
        param_name: str,
        tool_name: str,
    ) -> Any:
        """
        Coerce a value to the expected type.
        
        Args:
            value: Value to coerce
            expected_type: Expected type string
            param_name: Parameter name (for error messages)
            tool_name: Tool name (for error messages)
            
        Returns:
            Coerced value
        """
        if expected_type == "number":
            if isinstance(value, (int, float, Decimal)):
                return Decimal(str(value))
            if isinstance(value, str):
                # Remove currency symbols and commas
                cleaned = value.replace("$", "").replace(",", "").replace("S$", "").strip()
                return Decimal(cleaned)
            raise ValueError(f"Cannot convert {type(value).__name__} to number")
        
        elif expected_type == "integer":
            if isinstance(value, int):
                return value
            if isinstance(value, (float, Decimal)):
                return int(value)
            if isinstance(value, str):
                return int(float(value))
            raise ValueError(f"Cannot convert {type(value).__name__} to integer")
        
        elif expected_type == "string":
            return str(value)
        
        elif expected_type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "yes", "1")
            return bool(value)
        
        elif expected_type == "object":
            if isinstance(value, dict):
                return value
            raise ValueError(f"Expected object, got {type(value).__name__}")
        
        elif expected_type == "array":
            if isinstance(value, list):
                return value
            raise ValueError(f"Expected array, got {type(value).__name__}")
        
        # Unknown type, return as-is
        return value
    
    def get_available_tools(self) -> list[str]:
        """Get list of available tool names."""
        return self.registry.get_tool_names()
    
    def get_tool_info(self, tool_name: str) -> dict[str, Any] | None:
        """Get information about a specific tool."""
        tool = self.registry.get(tool_name)
        if tool:
            return tool.to_dict()
        return None


def execute_tool(
    tool_name: str,
    parameters: dict[str, Any],
) -> ToolCallResult:
    """
    Convenience function to execute a tool.
    
    Args:
        tool_name: Name of the tool to execute
        parameters: Parameters to pass to the tool
        
    Returns:
        ToolCallResult with execution outcome
    """
    dispatcher = ToolDispatcher()
    return dispatcher.execute(tool_name, parameters)


# Global dispatcher instance
DISPATCHER = ToolDispatcher()
File 21 of 25: finanalyst_tools/orchestration/__init__.py
Python

# finanalyst_tools/orchestration/__init__.py
"""
Orchestration layer for FinAnalyst-Pro Agent Tools.

This package provides the mandatory 5-phase processing pipeline:
1. VALIDATE - Data validation and completeness checks
2. ANALYZE - Determine what to calculate
3. CALCULATE - Execute calculations with audit trails
4. INTERPRET - Add context and insights
5. VERIFY - Pre-delivery checks

Also provides:
- Confidence scoring
- Report generation in mandatory format
"""

from finanalyst_tools.orchestration.pipeline import (
    AnalysisPipeline,
    AnalysisRequest,
    AnalysisPhase,
    run_analysis,
)

from finanalyst_tools.orchestration.confidence_scorer import (
    ConfidenceScorer,
    calculate_confidence_level,
)

from finanalyst_tools.orchestration.report_generator import (
    ReportGenerator,
    generate_financial_report,
    ReportFormat,
)


__all__ = [
    # Pipeline
    "AnalysisPipeline",
    "AnalysisRequest",
    "AnalysisPhase",
    "run_analysis",
    # Confidence
    "ConfidenceScorer",
    "calculate_confidence_level",
    # Reporting
    "ReportGenerator",
    "generate_financial_report",
    "ReportFormat",
]
File 22 of 25: finanalyst_tools/orchestration/pipeline.py
Python

# finanalyst_tools/orchestration/pipeline.py
"""
Analysis pipeline implementing the mandatory 5-phase workflow.

REQUEST → [1.VALIDATE] → [2.ANALYZE] → [3.CALCULATE] → [4.INTERPRET] → [5.VERIFY] → DELIVER
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
    ConfidenceAssessment,
)
from finanalyst_tools.models.validation import (
    ValidationResult,
    ReconciliationResult,
    PlausibilityResult,
)
from finanalyst_tools.validation import (
    validate_statement_set,
    run_all_reconciliations,
    check_all_plausibility,
)
from finanalyst_tools.calculations import (
    calculate_all_profitability_metrics,
    calculate_all_liquidity_metrics,
)
from finanalyst_tools.orchestration.confidence_scorer import calculate_confidence_level
from finanalyst_tools.orchestration.report_generator import generate_financial_report


class AnalysisPhase(str, Enum):
    """Phases of the analysis pipeline."""
    VALIDATE = "validate"
    ANALYZE = "analyze"
    CALCULATE = "calculate"
    INTERPRET = "interpret"
    VERIFY = "verify"
    DELIVER = "deliver"


@dataclass
class AnalysisRequest:
    """
    Request for financial analysis.
    
    Attributes:
        statement_set: Complete financial statements
        analysis_types: List of analysis categories to perform
        prior_statement_set: Prior period for comparisons (optional)
        include_trends: Whether to include trend analysis
        strict_validation: Whether to fail on validation warnings
    """
    statement_set: FinancialStatementSet
    analysis_types: list[str] = field(default_factory=lambda: ["profitability", "liquidity"])
    prior_statement_set: FinancialStatementSet | None = None
    include_trends: bool = False
    strict_validation: bool = False
    
    @property
    def period(self) -> str:
        """Get the analysis period as string."""
        return str(self.statement_set.period)
    
    @property
    def currency(self) -> str:
        """Get the currency."""
        return self.statement_set.currency


@dataclass
class PipelineState:
    """Internal state tracking for the pipeline."""
    current_phase: AnalysisPhase = AnalysisPhase.VALIDATE
    validation_result: ValidationResult | None = None
    reconciliation_result: ReconciliationResult | None = None
    plausibility_result: PlausibilityResult | None = None
    metric_collections: list[MetricCollection] = field(default_factory=list)
    all_metrics: list = field(default_factory=list)
    confidence: ConfidenceAssessment | None = None
    recommendations: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    @property
    def can_proceed(self) -> bool:
        """Check if pipeline can proceed to next phase."""
        if self.validation_result:
            return self.validation_result.can_proceed
        return True


class AnalysisPipeline:
    """
    Orchestrates the 5-phase analysis workflow.
    
    Ensures all mandatory steps are executed in order:
    1. VALIDATE - Cannot be skipped
    2. ANALYZE - Plan calculations
    3. CALCULATE - Execute with audit trails
    4. INTERPRET - Add context
    5. VERIFY - Final checks
    """
    
    def __init__(self):
        self.state = PipelineState()
    
    def execute(self, request: AnalysisRequest) -> ComprehensiveAnalysisResult:
        """
        Execute the full analysis pipeline.
        
        Args:
            request: Analysis request with financial data
            
        Returns:
            ComprehensiveAnalysisResult with all findings
        """
        # Reset state
        self.state = PipelineState()
        
        # Phase 1: VALIDATE (mandatory)
        self._phase_validate(request)
        
        if not self.state.can_proceed:
            return self._create_validation_failure_result(request)
        
        # Phase 2: ANALYZE
        self._phase_analyze(request)
        
        # Phase 3: CALCULATE
        self._phase_calculate(request)
        
        # Phase 4: INTERPRET
        self._phase_interpret(request)
        
        # Phase 5: VERIFY
        self._phase_verify(request)
        
        # DELIVER
        return self._create_result(request)
    
    def _phase_validate(self, request: AnalysisRequest) -> None:
        """
        Phase 1: Validate input data.
        
        - Schema validation
        - Completeness check
        - Cross-statement reconciliation
        """
        self.state.current_phase = AnalysisPhase.VALIDATE
        
        # Schema and completeness validation
        analysis_type = request.analysis_types[0] if request.analysis_types else "comprehensive"
        self.state.validation_result = validate_statement_set(
            request.statement_set,
            analysis_type,
        )
        
        # Cross-statement reconciliation
        prior_bs = request.prior_statement_set.balance_sheet if request.prior_statement_set else None
        self.state.reconciliation_result = run_all_reconciliations(
            request.statement_set,
            prior_balance_sheet=prior_bs,
        )
        
        # Convert reconciliation failures to validation issues
        if not self.state.reconciliation_result.all_passed:
            recon_validation = self.state.reconciliation_result.to_validation_result()
            self.state.validation_result.merge(recon_validation)
    
    def _phase_analyze(self, request: AnalysisRequest) -> None:
        """
        Phase 2: Analyze what calculations to perform.
        
        - Determine available data
        - Plan calculation sequence
        """
        self.state.current_phase = AnalysisPhase.ANALYZE
        # Analysis planning is implicit in the calculation phase
        # Future: Could add more sophisticated dependency analysis
    
    def _phase_calculate(self, request: AnalysisRequest) -> None:
        """
        Phase 3: Execute calculations.
        
        - Run all requested metric calculations
        - Capture audit trails
        """
        self.state.current_phase = AnalysisPhase.CALCULATE
        
        prior_bs = request.prior_statement_set.balance_sheet if request.prior_statement_set else None
        
        for analysis_type in request.analysis_types:
            if analysis_type.lower() == "profitability":
                collection = calculate_all_profitability_metrics(
                    income_statement=request.statement_set.income_statement,
                    balance_sheet=request.statement_set.balance_sheet,
                    prior_balance_sheet=prior_bs,
                )
                self.state.metric_collections.append(collection)
                self.state.all_metrics.extend(collection.metrics)
            
            elif analysis_type.lower() == "liquidity":
                collection = calculate_all_liquidity_metrics(
                    balance_sheet=request.statement_set.balance_sheet,
                )
                self.state.metric_collections.append(collection)
                self.state.all_metrics.extend(collection.metrics)
    
    def _phase_interpret(self, request: AnalysisRequest) -> None:
        """
        Phase 4: Interpret results.
        
        - Run plausibility checks
        - Generate recommendations
        """
        self.state.current_phase = AnalysisPhase.INTERPRET
        
        # Plausibility checks
        self.state.plausibility_result = check_all_plausibility(self.state.all_metrics)
        
        # Generate recommendations based on findings
        self._generate_recommendations(request)
    
    def _phase_verify(self, request: AnalysisRequest) -> None:
        """
        Phase 5: Verify results before delivery.
        
        - Calculate confidence score
        - Final quality checks
        """
        self.state.current_phase = AnalysisPhase.VERIFY
        
        # Calculate data completeness
        total_metrics = len(self.state.all_metrics)
        calculable_metrics = sum(1 for m in self.state.all_metrics if m.value is not None)
        completeness = calculable_metrics / total_metrics if total_metrics > 0 else 0.0
        
        # Calculate confidence
        self.state.confidence = calculate_confidence_level(
            validation_result=self.state.validation_result,
            plausibility_result=self.state.plausibility_result,
            reconciliation_result=self.state.reconciliation_result,
            data_completeness=completeness,
        )
    
    def _generate_recommendations(self, request: AnalysisRequest) -> None:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        for metric in self.state.all_metrics:
            if not metric.is_plausible:
                recommendations.append(
                    f"Review data accuracy for {metric.metric_name} - value appears unusual"
                )
            
            for warning in metric.warnings:
                if "liquidity risk" in warning.lower():
                    recommendations.append(
                        "Consider strategies to improve short-term liquidity position"
                    )
                elif "negative" in warning.lower() and "working capital" in metric.metric_name.lower():
                    recommendations.append(
                        "Address negative working capital to reduce financial risk"
                    )
        
        # Deduplicate
        self.state.recommendations = list(dict.fromkeys(recommendations))
    
    def _create_result(self, request: AnalysisRequest) -> ComprehensiveAnalysisResult:
        """Create the final analysis result."""
        return ComprehensiveAnalysisResult(
            analysis_type=", ".join(request.analysis_types),
            period=request.period,
            currency=request.currency,
            metric_collections=self.state.metric_collections,
            confidence=self.state.confidence,
            validation_summary=self.state.validation_result.to_dict() if self.state.validation_result else {},
            reconciliation_summary=self.state.reconciliation_result.to_dict() if self.state.reconciliation_result else {},
            recommendations=self.state.recommendations,
        )
    
    def _create_validation_failure_result(self, request: AnalysisRequest) -> ComprehensiveAnalysisResult:
        """Create a result for validation failure."""
        result = ComprehensiveAnalysisResult(
            analysis_type=", ".join(request.analysis_types),
            period=request.period,
            currency=request.currency,
            validation_summary=self.state.validation_result.to_dict() if self.state.validation_result else {},
            reconciliation_summary=self.state.reconciliation_result.to_dict() if self.state.reconciliation_result else {},
        )
        result.add_recommendation("Fix validation errors before proceeding with analysis")
        return result


def run_analysis(
    statement_set: FinancialStatementSet,
    analysis_types: list[str] | None = None,
    prior_statement_set: FinancialStatementSet | None = None,
) -> ComprehensiveAnalysisResult:
    """
    Convenience function to run the full analysis pipeline.
    
    Args:
        statement_set: Financial statements to analyze
        analysis_types: Types of analysis to perform (default: profitability, liquidity)
        prior_statement_set: Prior period data for comparisons
        
    Returns:
        ComprehensiveAnalysisResult with all findings
    """
    request = AnalysisRequest(
        statement_set=statement_set,
        analysis_types=analysis_types or ["profitability", "liquidity"],
        prior_statement_set=prior_statement_set,
    )
    
    pipeline = AnalysisPipeline()
    return pipeline.execute(request)
File 23 of 25: finanalyst_tools/orchestration/confidence_scorer.py
Python

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
File 24 of 25: finanalyst_tools/orchestration/report_generator.py
Python

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
