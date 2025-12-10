# finanalyst_tools/calculations/liquidity.py
"""
Liquidity ratio calculations.
"""

from __future__ import annotations

from decimal import Decimal

from ..config import PlausibilityRanges
from ..utils.math_ops import safe_divide, to_decimal
from .profitability import CalculationResult


def calculate_current_ratio(
    current_assets: Decimal | float,
    current_liabilities: Decimal | float
) -> CalculationResult:
    """
    Calculate Current Ratio.
    
    Formula: Current Assets / Current Liabilities
    
    Interpretation:
    - > 2.0: Strong liquidity, possibly inefficient asset use
    - 1.5 - 2.0: Healthy liquidity
    - 1.0 - 1.5: Adequate but monitor closely
    - < 1.0: Potential liquidity problems
    
    Args:
        current_assets: Total current assets
        current_liabilities: Total current liabilities
        
    Returns:
        CalculationResult with current ratio
    """
    ca = to_decimal(current_assets)
    cl = to_decimal(current_liabilities)
    
    steps = [
        f"Step 1: Current Assets = {ca:,.2f}",
        f"Step 2: Current Liabilities = {cl:,.2f}",
    ]
    
    ratio = safe_divide(ca, cl, precision=2)
    
    warnings = []
    is_plausible = True
    plausibility_range = PlausibilityRanges.CURRENT_RATIO
    
    if ratio is not None:
        steps.append(f"Step 3: Current Ratio = {ca:,.2f} / {cl:,.2f} = {ratio:.2f}")
        
        # Interpretation
        if ratio < 1:
            warnings.append(
                f"Current ratio below 1.0 indicates current liabilities exceed current assets. "
                f"Potential short-term liquidity risk."
            )
        
        if not (plausibility_range[0] <= float(ratio) <= plausibility_range[1]):
            is_plausible = False
            warnings.append(
                f"Current ratio {ratio:.2f} is outside typical range "
                f"({plausibility_range[0]} to {plausibility_range[1]})"
            )
    else:
        steps.append("Step 3: Cannot calculate - current liabilities is zero")
        warnings.append("Current liabilities is zero; cannot calculate ratio")
    
    return CalculationResult(
        metric_name="Current Ratio",
        value=ratio,
        unit="ratio",
        formula="Current Assets / Current Liabilities",
        inputs={"current_assets": ca, "current_liabilities": cl},
        calculation_steps=steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=warnings,
    )


def calculate_quick_ratio(
    current_assets: Decimal | float,
    inventory: Decimal | float,
    current_liabilities: Decimal | float
) -> CalculationResult:
    """
    Calculate Quick Ratio (Acid Test).
    
    Formula: (Current Assets - Inventory) / Current Liabilities
    
    More conservative than current ratio as inventory may not be easily liquidated.
    
    Args:
        current_assets: Total current assets
        inventory: Total inventory
        current_liabilities: Total current liabilities
        
    Returns:
        CalculationResult with quick ratio
    """
    ca = to_decimal(current_assets)
    inv = to_decimal(inventory)
    cl = to_decimal(current_liabilities)
    
    quick_assets = ca - inv
    
    steps = [
        f"Step 1: Current Assets = {ca:,.2f}, Inventory = {inv:,.2f}",
        f"Step 2: Quick Assets = {ca:,.2f} - {inv:,.2f} = {quick_assets:,.2f}",
        f"Step 3: Current Liabilities = {cl:,.2f}",
    ]
    
    ratio = safe_divide(quick_assets, cl, precision=2)
    
    warnings = []
    is_plausible = True
    plausibility_range = PlausibilityRanges.QUICK_RATIO
    
    if ratio is not None:
        steps.append(f"Step 4: Quick Ratio = {quick_assets:,.2f} / {cl:,.2f} = {ratio:.2f}")
        
        if ratio < 1:
            warnings.append(
                f"Quick ratio below 1.0 indicates the company may struggle to meet "
                f"short-term obligations without selling inventory."
            )
        
        if not (plausibility_range[0] <= float(ratio) <= plausibility_range[1]):
            is_plausible = False
            warnings.append(
                f"Quick ratio {ratio:.2f} is outside typical range "
                f"({plausibility_range[0]} to {plausibility_range[1]})"
            )
    else:
        steps.append("Step 4: Cannot calculate - current liabilities is zero")
        warnings.append("Current liabilities is zero; cannot calculate ratio")
    
    return CalculationResult(
        metric_name="Quick Ratio (Acid Test)",
        value=ratio,
        unit="ratio",
        formula="(Current Assets - Inventory) / Current Liabilities",
        inputs={
            "current_assets": ca,
            "inventory": inv,
            "quick_assets": quick_assets,
            "current_liabilities": cl,
        },
        calculation_steps=steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=warnings,
    )


def calculate_cash_ratio(
    cash_and_equivalents: Decimal | float,
    current_liabilities: Decimal | float
) -> CalculationResult:
    """
    Calculate Cash Ratio.
    
    Formula: Cash and Cash Equivalents / Current Liabilities
    
    Most conservative liquidity measure - only considers cash.
    
    Args:
        cash_and_equivalents: Cash and cash equivalents
        current_liabilities: Total current liabilities
        
    Returns:
        CalculationResult with cash ratio
    """
    cash = to_decimal(cash_and_equivalents)
    cl = to_decimal(current_liabilities)
    
    steps = [
        f"Step 1: Cash and Equivalents = {cash:,.2f}",
        f"Step 2: Current Liabilities = {cl:,.2f}",
    ]
    
    ratio = safe_divide(cash, cl, precision=2)
    
    warnings = []
    is_plausible = True
    plausibility_range = PlausibilityRanges.CASH_RATIO
    
    if ratio is not None:
        steps.append(f"Step 3: Cash Ratio = {cash:,.2f} / {cl:,.2f} = {ratio:.2f}")
        
        if not (plausibility_range[0] <= float(ratio) <= plausibility_range[1]):
            is_plausible = False
            warnings.append(
                f"Cash ratio {ratio:.2f} is outside typical range "
                f"({plausibility_range[0]} to {plausibility_range[1]})"
            )
    else:
        steps.append("Step 3: Cannot calculate - current liabilities is zero")
        warnings.append("Current liabilities is zero; cannot calculate ratio")
    
    return CalculationResult(
        metric_name="Cash Ratio",
        value=ratio,
        unit="ratio",
        formula="Cash and Equivalents / Current Liabilities",
        inputs={"cash_and_equivalents": cash, "current_liabilities": cl},
        calculation_steps=steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=warnings,
    )


def calculate_working_capital(
    current_assets: Decimal | float,
    current_liabilities: Decimal | float
) -> CalculationResult:
    """
    Calculate Working Capital.
    
    Formula: Current Assets - Current Liabilities
    
    Args:
        current_assets: Total current assets
        current_liabilities: Total current liabilities
        
    Returns:
        CalculationResult with working capital amount
    """
    ca = to_decimal(current_assets)
    cl = to_decimal(current_liabilities)
    
    working_capital = ca - cl
    
    steps = [
        f"Step 1: Current Assets = {ca:,.2f}",
        f"Step 2: Current Liabilities = {cl:,.2f}",
        f"Step 3: Working Capital = {ca:,.2f} - {cl:,.2f} = {working_capital:,.2f}",
    ]
    
    warnings = []
    if working_capital < 0:
        warnings.append(
            f"Negative working capital ({working_capital:,.2f}) indicates "
            f"current liabilities exceed current assets. Review short-term financing needs."
        )
    
    return CalculationResult(
        metric_name="Working Capital",
        value=working_capital,
        unit="currency",
        formula="Current Assets - Current Liabilities",
        inputs={"current_assets": ca, "current_liabilities": cl},
        calculation_steps=steps,
        is_plausible=True,  # Any value is technically valid
        plausibility_range=None,
        warnings=warnings,
    )
