# finanalyst_tools/calculations/profitability.py
"""
Profitability ratio calculations.

Provides functions to calculate:
- Gross Profit Margin
- Operating Profit Margin
- Net Profit Margin
- EBITDA Margin
- Return on Assets (ROA)
- Return on Equity (ROE)
- Return on Capital Employed (ROCE)

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
    extract_decimal_value,
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
    rev = to_decimal(revenue)
    cogs = to_decimal(cost_of_goods_sold)
    
    steps = []
    warnings = []
    inputs = {"revenue": rev, "cost_of_goods_sold": cogs}
    
    # Step 1: Calculate gross profit
    gross_profit = rev - cogs
    steps.append(f"Step 1: Calculate Gross Profit = Revenue - COGS = {rev:,.2f} - {cogs:,.2f} = {gross_profit:,.2f}")
    
    # Step 2: Calculate margin
    if is_effectively_zero(rev):
        steps.append("Step 2: Cannot calculate margin (revenue is zero)")
        warnings.append("Revenue is zero - cannot calculate margin")
        value = None
    else:
        margin = (gross_profit / rev) * 100
        margin = round_decimal(margin, DECIMAL_PLACES["percentage"])
        steps.append(f"Step 2: Calculate Margin = (Gross Profit / Revenue) × 100 = ({gross_profit:,.2f} / {rev:,.2f}) × 100 = {margin:.2f}%")
        value = margin
        
        # Validation warnings
        if margin < Decimal("0"):
            warnings.append("Negative gross margin indicates selling below cost")
        elif margin > Decimal("100"):
            warnings.append("Gross margin > 100% is unusual - verify COGS data")
    
    return create_calculation_result(
        metric_name="Gross Profit Margin",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
        unit=MetricUnit.PERCENTAGE,
        formula=METRIC_FORMULAS.get("gross_profit_margin", "(Revenue - COGS) / Revenue × 100"),
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
        marketing_expenses: Marketing expenses (optional, may be separate)
        include_marketing_in_opex: Whether marketing is already included in OpEx
        
    Returns:
        CalculationResult with operating profit margin percentage
    """
    rev = to_decimal(revenue)
    cogs = to_decimal(cost_of_goods_sold)
    opex = to_decimal(operating_expenses)
    mktg = to_decimal(marketing_expenses) if marketing_expenses else Decimal("0")
    
    steps = []
    warnings = []
    inputs = {
        "revenue": rev,
        "cost_of_goods_sold": cogs,
        "operating_expenses": opex,
    }
    
    if marketing_expenses is not None:
        inputs["marketing_expenses"] = mktg
        inputs["include_marketing_in_opex"] = include_marketing_in_opex
    
    # Step 1: Calculate total operating expenses
    if marketing_expenses is not None and not include_marketing_in_opex:
        total_opex = opex + mktg
        steps.append(f"Step 1: Total OpEx = OpEx + Marketing = {opex:,.2f} + {mktg:,.2f} = {total_opex:,.2f}")
    else:
        total_opex = opex
        steps.append(f"Step 1: Total OpEx = {total_opex:,.2f}")
    
    # Step 2: Calculate gross profit
    gross_profit = rev - cogs
    steps.append(f"Step 2: Gross Profit = Revenue - COGS = {rev:,.2f} - {cogs:,.2f} = {gross_profit:,.2f}")
    
    # Step 3: Calculate operating profit
    operating_profit = gross_profit - total_opex
    steps.append(f"Step 3: Operating Profit = Gross Profit - OpEx = {gross_profit:,.2f} - {total_opex:,.2f} = {operating_profit:,.2f}")
    
    # Step 4: Calculate margin
    if is_effectively_zero(rev):
        steps.append("Step 4: Cannot calculate margin (revenue is zero)")
        warnings.append("Revenue is zero - cannot calculate margin")
        value = None
    else:
        margin = (operating_profit / rev) * 100
        margin = round_decimal(margin, DECIMAL_PLACES["percentage"])
        steps.append(f"Step 4: Operating Margin = (Operating Profit / Revenue) × 100 = ({operating_profit:,.2f} / {rev:,.2f}) × 100 = {margin:.2f}%")
        value = margin
        
        if margin < Decimal("-50"):
            warnings.append("Operating margin below -50% indicates significant operational losses")
    
    return create_calculation_result(
        metric_name="Operating Profit Margin",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
        unit=MetricUnit.PERCENTAGE,
        formula=METRIC_FORMULAS.get("operating_profit_margin", "(Revenue - COGS - OpEx) / Revenue × 100"),
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
        net_income: Net income (bottom line)
        
    Returns:
        CalculationResult with net profit margin percentage
    """
    rev = to_decimal(revenue)
    ni = to_decimal(net_income)
    
    steps = []
    warnings = []
    inputs = {"revenue": rev, "net_income": ni}
    
    # Step 1: Calculate margin
    if is_effectively_zero(rev):
        steps.append("Step 1: Cannot calculate margin (revenue is zero)")
        warnings.append("Revenue is zero - cannot calculate margin")
        value = None
    else:
        margin = (ni / rev) * 100
        margin = round_decimal(margin, DECIMAL_PLACES["percentage"])
        steps.append(f"Step 1: Net Margin = (Net Income / Revenue) × 100 = ({ni:,.2f} / {rev:,.2f}) × 100 = {margin:.2f}%")
        value = margin
        
        if margin < Decimal("-100"):
            warnings.append("Net margin below -100% indicates severe losses exceeding revenue")
        elif margin > Decimal("50"):
            warnings.append("Net margin above 50% is exceptional - verify data accuracy")
    
    return create_calculation_result(
        metric_name="Net Profit Margin",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
        unit=MetricUnit.PERCENTAGE,
        formula=METRIC_FORMULAS.get("net_profit_margin", "Net Income / Revenue × 100"),
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
        ebitda: EBITDA (Earnings Before Interest, Taxes, Depreciation, Amortization)
        
    Returns:
        CalculationResult with EBITDA margin percentage
    """
    rev = to_decimal(revenue)
    ebitda_val = to_decimal(ebitda)
    
    steps = []
    warnings = []
    inputs = {"revenue": rev, "ebitda": ebitda_val}
    
    if is_effectively_zero(rev):
        steps.append("Step 1: Cannot calculate margin (revenue is zero)")
        warnings.append("Revenue is zero - cannot calculate margin")
        value = None
    else:
        margin = (ebitda_val / rev) * 100
        margin = round_decimal(margin, DECIMAL_PLACES["percentage"])
        steps.append(f"Step 1: EBITDA Margin = (EBITDA / Revenue) × 100 = ({ebitda_val:,.2f} / {rev:,.2f}) × 100 = {margin:.2f}%")
        value = margin
    
    return create_calculation_result(
        metric_name="EBITDA Margin",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
        unit=MetricUnit.PERCENTAGE,
        formula=METRIC_FORMULAS.get("ebitda_margin", "EBITDA / Revenue × 100"),
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
    ni = to_decimal(net_income)
    assets_begin = to_decimal(total_assets_beginning)
    assets_end = to_decimal(total_assets_ending)
    
    steps = []
    warnings = []
    inputs = {
        "net_income": ni,
        "total_assets_beginning": assets_begin,
        "total_assets_ending": assets_end,
    }
    
    # Step 1: Calculate average assets
    avg_assets = (assets_begin + assets_end) / 2
    steps.append(f"Step 1: Average Total Assets = (Beginning + Ending) / 2 = ({assets_begin:,.2f} + {assets_end:,.2f}) / 2 = {avg_assets:,.2f}")
    
    # Step 2: Calculate ROA
    if is_effectively_zero(avg_assets):
        steps.append("Step 2: Cannot calculate ROA (average assets is zero)")
        warnings.append("Average assets is zero - cannot calculate ROA")
        value = None
    else:
        roa = (ni / avg_assets) * 100
        roa = round_decimal(roa, DECIMAL_PLACES["percentage"])
        steps.append(f"Step 2: ROA = (Net Income / Avg Assets) × 100 = ({ni:,.2f} / {avg_assets:,.2f}) × 100 = {roa:.2f}%")
        value = roa
        
        if roa > Decimal("40"):
            warnings.append("ROA above 40% is exceptional - verify data")
    
    inputs["average_total_assets"] = avg_assets
    
    return create_calculation_result(
        metric_name="Return on Assets",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
        unit=MetricUnit.PERCENTAGE,
        formula=METRIC_FORMULAS.get("roa", "Net Income / Average Total Assets × 100"),
    )


def calculate_return_on_equity(
    net_income: Decimal | float | int,
    equity_beginning: Decimal | float | int,
    equity_ending: Decimal | float | int,
) -> CalculationResult:
    """
    Calculate Return on Equity (ROE).
    
    Formula: Net Income / Average Shareholders' Equity × 100
    
    Args:
        net_income: Net income for the period
        equity_beginning: Shareholders' equity at period start
        equity_ending: Shareholders' equity at period end
        
    Returns:
        CalculationResult with ROE percentage
    """
    ni = to_decimal(net_income)
    equity_begin = to_decimal(equity_beginning)
    equity_end = to_decimal(equity_ending)
    
    steps = []
    warnings = []
    inputs = {
        "net_income": ni,
        "equity_beginning": equity_begin,
        "equity_ending": equity_end,
    }
    
    # Check for negative equity
    if equity_begin < 0 or equity_end < 0:
        warnings.append("Negative equity detected - ROE interpretation may be misleading")
    
    # Step 1: Calculate average equity
    avg_equity = (equity_begin + equity_end) / 2
    steps.append(f"Step 1: Average Equity = (Beginning + Ending) / 2 = ({equity_begin:,.2f} + {equity_end:,.2f}) / 2 = {avg_equity:,.2f}")
    
    # Step 2: Calculate ROE
    if is_effectively_zero(avg_equity):
        steps.append("Step 2: Cannot calculate ROE (average equity is zero)")
        warnings.append("Average equity is zero - cannot calculate ROE")
        value = None
    else:
        roe = (ni / avg_equity) * 100
        roe = round_decimal(roe, DECIMAL_PLACES["percentage"])
        steps.append(f"Step 2: ROE = (Net Income / Avg Equity) × 100 = ({ni:,.2f} / {avg_equity:,.2f}) × 100 = {roe:.2f}%")
        value = roe
        
        if roe > Decimal("80"):
            warnings.append("ROE above 80% may indicate high leverage or low equity base")
    
    inputs["average_shareholders_equity"] = avg_equity
    
    return create_calculation_result(
        metric_name="Return on Equity",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
        unit=MetricUnit.PERCENTAGE,
        formula=METRIC_FORMULAS.get("roe", "Net Income / Average Shareholders' Equity × 100"),
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
    ebit_val = to_decimal(ebit)
    assets = to_decimal(total_assets)
    curr_liab = to_decimal(current_liabilities)
    
    steps = []
    warnings = []
    inputs = {
        "ebit": ebit_val,
        "total_assets": assets,
        "current_liabilities": curr_liab,
    }
    
    # Step 1: Calculate capital employed
    capital_employed = assets - curr_liab
    steps.append(f"Step 1: Capital Employed = Total Assets - Current Liabilities = {assets:,.2f} - {curr_liab:,.2f} = {capital_employed:,.2f}")
    
    # Step 2: Calculate ROCE
    if is_effectively_zero(capital_employed):
        steps.append("Step 2: Cannot calculate ROCE (capital employed is zero)")
        warnings.append("Capital employed is zero - cannot calculate ROCE")
        value = None
    elif capital_employed < 0:
        steps.append("Step 2: Negative capital employed - ROCE may be misleading")
        warnings.append("Negative capital employed indicates current liabilities exceed total assets")
        roce = (ebit_val / capital_employed) * 100
        roce = round_decimal(roce, DECIMAL_PLACES["percentage"])
        steps.append(f"Step 2 (cont): ROCE = (EBIT / Capital Employed) × 100 = ({ebit_val:,.2f} / {capital_employed:,.2f}) × 100 = {roce:.2f}%")
        value = roce
    else:
        roce = (ebit_val / capital_employed) * 100
        roce = round_decimal(roce, DECIMAL_PLACES["percentage"])
        steps.append(f"Step 2: ROCE = (EBIT / Capital Employed) × 100 = ({ebit_val:,.2f} / {capital_employed:,.2f}) × 100 = {roce:.2f}%")
        value = roce
    
    inputs["capital_employed"] = capital_employed
    
    return create_calculation_result(
        metric_name="Return on Capital Employed",
        value=value,
        inputs=inputs,
        calculation_steps=steps,
        category=MetricCategory.PROFITABILITY,
        warnings=warnings,
        unit=MetricUnit.PERCENTAGE,
        formula=METRIC_FORMULAS.get("roce", "EBIT / (Total Assets - Current Liabilities) × 100"),
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
        prior_balance_sheet: Prior period balance sheet (for average calculations)
        
    Returns:
        MetricCollection containing all profitability metrics
    """
    collection = MetricCollection(
        category=MetricCategory.PROFITABILITY,
        period=income_statement.period,
    )
    
    # Use current period values if no prior period
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
        equity_beginning=prior_balance_sheet.calculated_shareholders_equity,
        equity_ending=balance_sheet.calculated_shareholders_equity,
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


# ============================================================================
# CLASS-BASED CALCULATOR
# ============================================================================

class ProfitabilityCalculator(BaseCalculator):
    """
    Class-based profitability calculator with stateful operations.
    """
    
    def calculate(
        self,
        income_statement: IncomeStatementData,
        balance_sheet: BalanceSheetData,
        prior_balance_sheet: BalanceSheetData | None = None,
    ) -> MetricCollection:
        """
        Calculate all profitability metrics.
        
        Args:
            income_statement: Current period income statement
            balance_sheet: Current period balance sheet
            prior_balance_sheet: Prior period balance sheet
            
        Returns:
            MetricCollection with all profitability metrics
        """
        return calculate_all_profitability_metrics(
            income_statement=income_statement,
            balance_sheet=balance_sheet,
            prior_balance_sheet=prior_balance_sheet,
        )
