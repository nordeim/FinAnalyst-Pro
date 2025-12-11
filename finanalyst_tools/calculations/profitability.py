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
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from finanalyst_tools.models.analysis_results import (
    CalculationResult,
    MetricUnit,
    MetricCategory,
    MetricCollection,
)
from finanalyst_tools.models.financial_statements import (
    IncomeStatementData,
    BalanceSheetData,
)
from finanalyst_tools.calculations.base import (
    BaseCalculator,
    create_calculation_result,
    format_step,
)
from finanalyst_tools.utils.math_ops import (
    to_decimal,
    safe_divide,
    calculate_percentage,
    calculate_average,
)
from finanalyst_tools.config import METRIC_FORMULAS


# ============================================================================
# STANDALONE CALCULATION FUNCTIONS
# ============================================================================

def calculate_gross_profit_margin(
    revenue: Decimal | float | int,
    cost_of_goods_sold: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Gross Profit Margin.
    
    Formula: ((Revenue - COGS) / Revenue) × 100
    
    Args:
        revenue: Total revenue
        cost_of_goods_sold: Cost of goods sold
        
    Returns:
        CalculationResult with value, steps, and plausibility
    """
    steps = []
    
    # Step 1: Record inputs
    rev = to_decimal(revenue)
    cogs = to_decimal(cost_of_goods_sold)
    
    steps.append(format_step(
        1, "Identify input values",
        values={"revenue": rev, "cogs": cogs}
    ))
    
    # Step 2: Calculate gross profit
    gross_profit = rev - cogs
    steps.append(format_step(
        2, "Calculate Gross Profit",
        formula="Gross Profit = Revenue - COGS",
        values={"revenue": rev, "cogs": cogs},
        result=gross_profit
    ))
    
    # Step 3: Calculate margin
    margin = calculate_percentage(gross_profit, rev)
    steps.append(format_step(
        3, "Calculate Gross Profit Margin",
        formula="(Gross Profit / Revenue) × 100",
        values={"gross_profit": gross_profit, "revenue": rev},
        result=f"{margin}%"
    ))
    
    # Create interpretation
    if margin is not None:
        if margin >= 50:
            interpretation = "Strong gross margin indicating good pricing power or cost control"
        elif margin >= 30:
            interpretation = "Healthy gross margin typical of many industries"
        elif margin >= 20:
            interpretation = "Moderate gross margin; may indicate competitive pricing pressure"
        else:
            interpretation = "Low gross margin; review pricing and cost structure"
    else:
        interpretation = "Unable to calculate - check input values"
    
    return create_calculation_result(
        metric_name="Gross Profit Margin",
        value=margin,
        unit=MetricUnit.PERCENTAGE,
        formula=METRIC_FORMULAS.get("gross_profit_margin", "(Revenue - COGS) / Revenue × 100"),
        inputs={"revenue": revenue, "cost_of_goods_sold": 
