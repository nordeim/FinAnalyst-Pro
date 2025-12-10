# finanalyst_tools/calculations/profitability.py
"""
Profitability ratio calculations.
All functions return structured results with audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from ..config import PlausibilityRanges, DECIMAL_PLACES
from ..models.financial_statements import IncomeStatementData, BalanceSheetData
from ..utils.math_ops import (
    safe_divide,
    calculate_percentage,
    round_decimal,
    to_decimal,
)


@dataclass
class CalculationResult:
    """Structured result from a financial calculation."""
    
    metric_name: str
    value: Decimal | None
    unit: str  # "percentage", "ratio", "currency", "days"
    formula: str
    inputs: dict[str, Any]
    calculation_steps: list[str]
    is_plausible: bool
    plausibility_range: tuple[float, float] | None
    warnings: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metric_name": self.metric_name,
            "value": float(self.value) if self.value is not None else None,
            "unit": self.unit,
            "formula": self.formula,
            "inputs": {k: float(v) if isinstance(v, Decimal) else v for k, v in self.inputs.items()},
            "calculation_steps": self.calculation_steps,
            "is_plausible": self.is_plausible,
            "plausibility_range": self.plausibility_range,
            "warnings": self.warnings,
        }


def calculate_gross_profit_margin(
    revenue: Decimal | float,
    cost_of_goods_sold: Decimal | float
) -> CalculationResult:
    """
    Calculate Gross Profit Margin.
    
    Formula: (Revenue - COGS) / Revenue × 100
    
    Args:
        revenue: Total revenue/sales
        cost_of_goods_sold: Cost of goods sold
        
    Returns:
        CalculationResult with gross profit margin percentage
    """
    rev = to_decimal(revenue)
    cogs = to_decimal(cost_of_goods_sold)
    
    steps = [
        f"Step 1: Identify values → Revenue = {rev:,.2f}, COGS = {cogs:,.2f}",
        f"Step 2: Calculate Gross Profit = Revenue - COGS",
        f"Step 3: Gross Profit = {rev:,.2f} - {cogs:,.2f} = {rev - cogs:,.2f}",
    ]
    
    gross_profit = rev - cogs
    margin = calculate_percentage(gross_profit, rev)
    
    warnings = []
    is_plausible = True
    plausibility_range = PlausibilityRanges.GROSS_MARGIN
    
    if margin is not None:
        steps.append(f"Step 4: Gross Margin = ({gross_profit:,.2f} / {rev:,.2f}) × 100 = {margin:.2f}%")
        
        if not (plausibility_range[0] <= float(margin) <= plausibility_range[1]):
            is_plausible = False
            warnings.append(
                f"Gross margin {margin:.2f}% is outside typical range "
                f"({plausibility_range[0]}% to {plausibility_range[1]}%)"
            )
    else:
        steps.append("Step 4: Cannot calculate - revenue is zero")
        warnings.append("Revenue is zero; cannot calculate margin")
    
    return CalculationResult(
        metric_name="Gross Profit Margin",
        value=margin,
        unit="percentage",
        formula="(Revenue - COGS) / Revenue × 100",
        inputs={"revenue": rev, "cost_of_goods_sold": cogs, "gross_profit": gross_profit},
        calculation_steps=steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=warnings,
    )


def calculate_operating_profit_margin(
    revenue: Decimal | float,
    cost_of_goods_sold: Decimal | float,
    operating_expenses: Decimal | float,
    marketing_expenses: Decimal | float = Decimal("0"),
    include_marketing_in_opex: bool = False
) -> CalculationResult:
    """
    Calculate Operating Profit Margin (EBIT Margin).
    
    Formula: Operating Profit / Revenue × 100
    Where: Operating Profit = Revenue - COGS - Operating Expenses [- Marketing if separate]
    
    Args:
        revenue: Total revenue/sales
        cost_of_goods_sold: Cost of goods sold
        operating_expenses: Operating expenses
        marketing_expenses: Marketing expenses (if tracked separately)
        include_marketing_in_opex: If True, marketing is already in operating_expenses
        
    Returns:
        CalculationResult with operating profit margin percentage
    """
    rev = to_decimal(revenue)
    cogs = to_decimal(cost_of_goods_sold)
    opex = to_decimal(operating_expenses)
    marketing = to_decimal(marketing_expenses)
    
    gross_profit = rev - cogs
    
    if include_marketing_in_opex:
        total_opex = opex
        formula_note = "(Marketing included in OpEx)"
    else:
        total_opex = opex + marketing
        formula_note = "(Marketing added separately)"
    
    operating_profit = gross_profit - total_opex
    
    steps = [
        f"Step 1: Revenue = {rev:,.2f}, COGS = {cogs:,.2f}",
        f"Step 2: Gross Profit = {rev:,.2f} - {cogs:,.2f} = {gross_profit:,.2f}",
        f"Step 3: Total Operating Expenses = {total_opex:,.2f} {formula_note}",
        f"Step 4: Operating Profit = {gross_profit:,.2f} - {total_opex:,.2f} = {operating_profit:,.2f}",
    ]
    
    margin = calculate_percentage(operating_profit, rev)
    
    warnings = []
    is_plausible = True
    plausibility_range = PlausibilityRanges.OPERATING_MARGIN
    
    if margin is not None:
        steps.append(f"Step 5: Operating Margin = ({operating_profit:,.2f} / {rev:,.2f}) × 100 = {margin:.2f}%")
        
        if not (plausibility_range[0] <= float(margin) <= plausibility_range[1]):
            is_plausible = False
            warnings.append(
                f"Operating margin {margin:.2f}% is outside typical range "
                f"({plausibility_range[0]}% to {plausibility_range[1]}%)"
            )
    else:
        steps.append("Step 5: Cannot calculate - revenue is zero")
        warnings.append("Revenue is zero; cannot calculate margin")
    
    return CalculationResult(
        metric_name="Operating Profit Margin",
        value=margin,
        unit="percentage",
        formula="(Revenue - COGS - OpEx) / Revenue × 100",
        inputs={
            "revenue": rev,
            "cost_of_goods_sold": cogs,
            "operating_expenses": opex,
            "marketing_expenses": marketing,
            "gross_profit": gross_profit,
            "operating_profit": operating_profit,
        },
        calculation_steps=steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=warnings,
    )


def calculate_net_profit_margin(
    revenue: Decimal | float,
    net_income: Decimal | float
) -> CalculationResult:
    """
    Calculate Net Profit Margin.
    
    Formula: Net Income / Revenue × 100
    
    Args:
        revenue: Total revenue/sales
        net_income: Net income (after taxes)
        
    Returns:
        CalculationResult with net profit margin percentage
    """
    rev = to_decimal(revenue)
    net = to_decimal(net_income)
    
    steps = [
        f"Step 1: Identify values → Revenue = {rev:,.2f}, Net Income = {net:,.2f}",
    ]
    
    margin = calculate_percentage(net, rev)
    
    warnings = []
    is_plausible = True
    plausibility_range = PlausibilityRanges.NET_MARGIN
    
    if margin is not None:
        steps.append(f"Step 2: Net Margin = ({net:,.2f} / {rev:,.2f}) × 100 = {margin:.2f}%")
        
        if not (plausibility_range[0] <= float(margin) <= plausibility_range[1]):
            is_plausible = False
            warnings.append(
                f"Net margin {margin:.2f}% is outside typical range "
                f"({plausibility_range[0]}% to {plausibility_range[1]}%)"
            )
        
        # Special check for impossibly high margins
        if float(margin) >= 100:
            warnings.append(
                f"CRITICAL: Net margin ≥ 100% indicates data error. "
                f"Net income ({net:,.2f}) cannot exceed revenue ({rev:,.2f})"
            )
            is_plausible = False
    else:
        steps.append("Step 2: Cannot calculate - revenue is zero")
        warnings.append("Revenue is zero; cannot calculate margin")
    
    return CalculationResult(
        metric_name="Net Profit Margin",
        value=margin,
        unit="percentage",
        formula="Net Income / Revenue × 100",
        inputs={"revenue": rev, "net_income": net},
        calculation_steps=steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=warnings,
    )


def calculate_return_on_assets(
    net_income: Decimal | float,
    total_assets_beginning: Decimal | float,
    total_assets_ending: Decimal | float
) -> CalculationResult:
    """
    Calculate Return on Assets (ROA).
    
    Formula: Net Income / Average Total Assets × 100
    
    Args:
        net_income: Net income for the period
        total_assets_beginning: Total assets at start of period
        total_assets_ending: Total assets at end of period
        
    Returns:
        CalculationResult with ROA percentage
    """
    net = to_decimal(net_income)
    assets_begin = to_decimal(total_assets_beginning)
    assets_end = to_decimal(total_assets_ending)
    
    avg_assets = (assets_begin + assets_end) / 2
    
    steps = [
        f"Step 1: Net Income = {net:,.2f}",
        f"Step 2: Beginning Assets = {assets_begin:,.2f}, Ending Assets = {assets_end:,.2f}",
        f"Step 3: Average Assets = ({assets_begin:,.2f} + {assets_end:,.2f}) / 2 = {avg_assets:,.2f}",
    ]
    
    roa = calculate_percentage(net, avg_assets)
    
    warnings = []
    is_plausible = True
    plausibility_range = PlausibilityRanges.ROA
    
    if roa is not None:
        steps.append(f"Step 4: ROA = ({net:,.2f} / {avg_assets:,.2f}) × 100 = {roa:.2f}%")
        
        if not (plausibility_range[0] <= float(roa) <= plausibility_range[1]):
            is_plausible = False
            warnings.append(
                f"ROA {roa:.2f}% is outside typical range "
                f"({plausibility_range[0]}% to {plausibility_range[1]}%)"
            )
    else:
        steps.append("Step 4: Cannot calculate - average assets is zero")
        warnings.append("Average assets is zero; cannot calculate ROA")
    
    return CalculationResult(
        metric_name="Return on Assets (ROA)",
        value=roa,
        unit="percentage",
        formula="Net Income / Average Total Assets × 100",
        inputs={
            "net_income": net,
            "total_assets_beginning": assets_begin,
            "total_assets_ending": assets_end,
            "average_total_assets": avg_assets,
        },
        calculation_steps=steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=warnings,
    )


def calculate_return_on_equity(
    net_income: Decimal | float,
    shareholders_equity_beginning: Decimal | float,
    shareholders_equity_ending: Decimal | float
) -> CalculationResult:
    """
    Calculate Return on Equity (ROE).
    
    Formula: Net Income / Average Shareholders' Equity × 100
    
    Args:
        net_income: Net income for the period
        shareholders_equity_beginning: Equity at start of period
        shareholders_equity_ending: Equity at end of period
        
    Returns:
        CalculationResult with ROE percentage
    """
    net = to_decimal(net_income)
    equity_begin = to_decimal(shareholders_equity_beginning)
    equity_end = to_decimal(shareholders_equity_ending)
    
    avg_equity = (equity_begin + equity_end) / 2
    
    steps = [
        f"Step 1: Net Income = {net:,.2f}",
        f"Step 2: Beginning Equity = {equity_begin:,.2f}, Ending Equity = {equity_end:,.2f}",
        f"Step 3: Average Equity = ({equity_begin:,.2f} + {equity_end:,.2f}) / 2 = {avg_equity:,.2f}",
    ]
    
    roe = calculate_percentage(net, avg_equity)
    
    warnings = []
    is_plausible = True
    plausibility_range = PlausibilityRanges.ROE
    
    if roe is not None:
        steps.append(f"Step 4: ROE = ({net:,.2f} / {avg_equity:,.2f}) × 100 = {roe:.2f}%")
        
        if not (plausibility_range[0] <= float(roe) <= plausibility_range[1]):
            is_plausible = False
            warnings.append(
                f"ROE {roe:.2f}% is outside typical range "
                f"({plausibility_range[0]}% to {plausibility_range[1]}%)"
            )
    else:
        steps.append("Step 4: Cannot calculate - average equity is zero")
        warnings.append("Average shareholders' equity is zero; cannot calculate ROE")
    
    return CalculationResult(
        metric_name="Return on Equity (ROE)",
        value=roe,
        unit="percentage",
        formula="Net Income / Average Shareholders' Equity × 100",
        inputs={
            "net_income": net,
            "shareholders_equity_beginning": equity_begin,
            "shareholders_equity_ending": equity_end,
            "average_shareholders_equity": avg_equity,
        },
        calculation_steps=steps,
        is_plausible=is_plausible,
        plausibility_range=plausibility_range,
        warnings=warnings,
    )
