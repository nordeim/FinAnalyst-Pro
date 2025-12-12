# finanalyst_tools/calculations/liquidity.py
"""
Liquidity ratio calculations.

Provides functions to calculate:
- Current Ratio
- Quick Ratio (Acid Test)
- Cash Ratio
- Working Capital

All functions return CalculationResult with complete audit trail.
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
from finanalyst_tools.models.financial_statements import BalanceSheetData
from finanalyst_tools.utils.math_ops import (
    to_decimal,
    safe_divide,
    round_decimal,
    is_effectively_zero,
)
from finanalyst_tools.calculations.base import (
    BaseCalculator,
    create_calculation_result,
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
    - > 3.0: May indicate inefficient use of assets
    
    Args:
        current_assets: Total current assets
        current_liabilities: Total current liabilities
        
    Returns:
        CalculationResult with current ratio
    """
    ca = to_decimal(current_assets)
    cl = to_decimal(current_liabilities)
    
    steps = []
    warnings = []
    inputs = {"current_assets": ca, "current_liabilities": cl}
    
    # Calculate ratio
    if is_effectively_zero(cl):
        steps.append("Step 1: Cannot calculate ratio (current liabilities is zero)")
        warnings.append("Current liabilities is zero - ratio undefined")
        value = None
    else:
        ratio = ca / cl
        ratio = round_decimal(ratio, DECIMAL_PLACES["ratio"])
        steps.append(f"Step 1: Current Ratio = Current Assets / Current Liabilities = {ca:,.2f} / {cl:,.2f} = {ratio:.4f}")
        value = ratio
        
        # Interpretation warnings
        if ratio < Decimal("1.0"):
            warnings.append(f"Current ratio of {ratio:.2f} is below 1.0, indicating potential liquidity risk")
        elif ratio > Decimal("3.0"):
            warnings.append(f"Current ratio of {ratio:.2f} is above 3.0, which may indicate inefficient asset utilization")
    
    return create_calculation_result(
        metric_name="Current Ratio",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.LIQUIDITY,
        warnings=warnings,
        unit=MetricUnit.RATIO,
        formula=METRIC_FORMULAS.get("current_ratio", "Current Assets / Current Liabilities"),
    )


def calculate_quick_ratio(
    current_assets: Decimal | float | int,
    inventory: Decimal | float | int,
    current_liabilities: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Quick Ratio (Acid Test).
    
    Formula: (Current Assets - Inventory) / Current Liabilities
    
    More conservative than current ratio as it excludes inventory,
    which may not be quickly convertible to cash.
    
    Args:
        current_assets: Total current assets
        inventory: Inventory value
        current_liabilities: Total current liabilities
        
    Returns:
        CalculationResult with quick ratio
    """
    ca = to_decimal(current_assets)
    inv = to_decimal(inventory)
    cl = to_decimal(current_liabilities)
    
    steps = []
    warnings = []
    inputs = {
        "current_assets": ca,
        "inventory": inv,
        "current_liabilities": cl,
    }
    
    # Step 1: Calculate quick assets
    quick_assets = ca - inv
    steps.append(f"Step 1: Quick Assets = Current Assets - Inventory = {ca:,.2f} - {inv:,.2f} = {quick_assets:,.2f}")
    
    # Step 2: Calculate ratio
    if is_effectively_zero(cl):
        steps.append("Step 2: Cannot calculate ratio (current liabilities is zero)")
        warnings.append("Current liabilities is zero - ratio undefined")
        value = None
    else:
        ratio = quick_assets / cl
        ratio = round_decimal(ratio, DECIMAL_PLACES["ratio"])
        steps.append(f"Step 2: Quick Ratio = Quick Assets / Current Liabilities = {quick_assets:,.2f} / {cl:,.2f} = {ratio:.4f}")
        value = ratio
        
        if ratio < Decimal("1.0"):
            warnings.append(f"Quick ratio of {ratio:.2f} is below 1.0, indicating limited liquid assets")
    
    inputs["quick_assets"] = quick_assets
    
    return create_calculation_result(
        metric_name="Quick Ratio",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.LIQUIDITY,
        warnings=warnings,
        unit=MetricUnit.RATIO,
        formula=METRIC_FORMULAS.get("quick_ratio", "(Current Assets - Inventory) / Current Liabilities"),
    )


def calculate_cash_ratio(
    cash_and_equivalents: Decimal | float | int,
    current_liabilities: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Cash Ratio.
    
    Formula: Cash and Equivalents / Current Liabilities
    
    Most conservative liquidity measure - only considers
    the most liquid assets.
    
    Args:
        cash_and_equivalents: Cash and cash equivalents
        current_liabilities: Total current liabilities
        
    Returns:
        CalculationResult with cash ratio
    """
    cash = to_decimal(cash_and_equivalents)
    cl = to_decimal(current_liabilities)
    
    steps = []
    warnings = []
    inputs = {"cash_and_equivalents": cash, "current_liabilities": cl}
    
    if is_effectively_zero(cl):
        steps.append("Step 1: Cannot calculate ratio (current liabilities is zero)")
        warnings.append("Current liabilities is zero - ratio undefined")
        value = None
    else:
        ratio = cash / cl
        ratio = round_decimal(ratio, DECIMAL_PLACES["ratio"])
        steps.append(f"Step 1: Cash Ratio = Cash / Current Liabilities = {cash:,.2f} / {cl:,.2f} = {ratio:.4f}")
        value = ratio
        
        if ratio < Decimal("0.2"):
            warnings.append(f"Cash ratio of {ratio:.2f} is below 0.2, which may be low for immediate obligations")
        elif ratio > Decimal("1.0"):
            warnings.append(f"Cash ratio of {ratio:.2f} is above 1.0, indicating potentially excess cash holdings")
    
    return create_calculation_result(
        metric_name="Cash Ratio",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.LIQUIDITY,
        warnings=warnings,
        unit=MetricUnit.RATIO,
        formula=METRIC_FORMULAS.get("cash_ratio", "Cash and Equivalents / Current Liabilities"),
    )


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
        CalculationResult with working capital (absolute amount)
    """
    ca = to_decimal(current_assets)
    cl = to_decimal(current_liabilities)
    
    steps = []
    warnings = []
    inputs = {
        "current_assets": ca,
        "current_liabilities": cl,
        "currency": currency,
    }
    
    # Calculate working capital
    wc = ca - cl
    wc = round_decimal(wc, DECIMAL_PLACES["currency"])
    steps.append(f"Step 1: Working Capital = Current Assets - Current Liabilities = {ca:,.2f} - {cl:,.2f} = {wc:,.2f}")
    
    if wc < Decimal("0"):
        warnings.append(f"Negative working capital of {wc:,.2f} indicates current liabilities exceed current assets")
    
    return create_calculation_result(
        metric_name="Working Capital",
        value=wc,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.LIQUIDITY,
        warnings=warnings,
        unit=MetricUnit.CURRENCY,
        formula=METRIC_FORMULAS.get("working_capital", "Current Assets - Current Liabilities"),
    )


def calculate_all_liquidity_metrics(
    balance_sheet: BalanceSheetData,
) -> MetricCollection:
    """
    Calculate all liquidity metrics from a balance sheet.
    
    Args:
        balance_sheet: Balance sheet data
        
    Returns:
        MetricCollection containing all liquidity metrics
    """
    collection = MetricCollection(
        category=MetricCategory.LIQUIDITY,
        period=balance_sheet.period,
    )
    
    current_assets = balance_sheet.calculated_current_assets
    current_liabilities = balance_sheet.calculated_current_liabilities
    
    # Current Ratio
    cr = calculate_current_ratio(
        current_assets=current_assets,
        current_liabilities=current_liabilities,
    )
    collection.add_metric(cr)
    
    # Quick Ratio
    qr = calculate_quick_ratio(
        current_assets=current_assets,
        inventory=balance_sheet.inventory,
        current_liabilities=current_liabilities,
    )
    collection.add_metric(qr)
    
    # Cash Ratio
    cash_r = calculate_cash_ratio(
        cash_and_equivalents=balance_sheet.cash_and_equivalents,
        current_liabilities=current_liabilities,
    )
    collection.add_metric(cash_r)
    
    # Working Capital
    wc = calculate_working_capital(
        current_assets=current_assets,
        current_liabilities=current_liabilities,
        currency=balance_sheet.currency,
    )
    collection.add_metric(wc)
    
    return collection


# ============================================================================
# CLASS-BASED CALCULATOR
# ============================================================================

class LiquidityCalculator(BaseCalculator):
    """
    Class-based liquidity calculator with stateful operations.
    """
    
    def calculate(
        self,
        balance_sheet: BalanceSheetData,
    ) -> MetricCollection:
        """
        Calculate all liquidity metrics.
        
        Args:
            balance_sheet: Balance sheet data
            
        Returns:
            MetricCollection with all liquidity metrics
        """
        return calculate_all_liquidity_metrics(balance_sheet)
