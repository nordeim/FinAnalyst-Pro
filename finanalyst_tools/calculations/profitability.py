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
