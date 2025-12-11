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
