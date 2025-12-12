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
        "current_assets": float(ca),
        "current_liabilities": float(cl),
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
        "current_assets": float(ca),
        "inventory": float(inv),
        "current_liabilities": float(cl),
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
    inputs["quick_assets"] = float(quick_assets)
    
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
        "cash_and_equivalents": float(cash),
        "current_liabilities": float(cl),
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
        "current_assets": float(ca),
        "current_liabilities": float(cl),
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
