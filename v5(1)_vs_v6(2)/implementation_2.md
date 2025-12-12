# finanalyst_tools/calculations/profitability.py
```6
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

```

# finanalyst_tools/calculations/__init__.py
```6
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

# finanalyst_tools/calculations/liquidity.py
```6
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

```

# finanalyst_tools/calculations/base.py
```6
# finanalyst_tools/calculations/base.py
"""
Base calculation infrastructure for FinAnalyst-Pro Agent Tools.

Provides:
- BaseCalculator class with common functionality
- Factory functions for creating calculation results
- Utility functions for extracting and validating inputs
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any

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


def extract_decimal_value(
    value: Any,
    field_name: str = "value",
    default: Decimal | None = None,
) -> Decimal:
    """
    Extract a Decimal value from various input types.
    
    Args:
        value: Input value (int, float, Decimal, str, or None)
        field_name: Name of the field (for error messages)
        default: Default value if conversion fails
        
    Returns:
        Decimal value
    """
    if value is None:
        if default is not None:
            return default
        return Decimal("0")
    
    return to_decimal(value, default=default or Decimal("0"))


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
    Get the formula string for a metric.
    
    Args:
        metric_name: Name of the metric
        
    Returns:
        Formula string
    """
    return METRIC_FORMULAS.get(metric_name.lower(), "Custom calculation")


def create_calculation_result(
    metric_name: str,
    value: Decimal | None,
    inputs: dict[str, Any],
    calculation_steps: list[str],
    category: MetricCategory | None = None,
    warnings: list[str] | None = None,
    unit: MetricUnit | None = None,
    formula: str | None = None,
) -> CalculationResult:
    """
    Factory function to create a CalculationResult with plausibility check.
    
    Args:
        metric_name: Name of the metric
        value: Calculated value
        inputs: Dictionary of input values used
        calculation_steps: List of calculation step descriptions
        category: Metric category
        warnings: List of warning messages
        unit: Override unit type
        formula: Override formula string
        
    Returns:
        Complete CalculationResult with plausibility assessment
    """
    # Get unit and formula if not provided
    if unit is None:
        unit = get_metric_unit(metric_name)
    if formula is None:
        formula = get_metric_formula(metric_name)
    
    # Get plausibility range
    plausibility_range = PlausibilityRanges.get_range(metric_name)
    
    # Check plausibility
    is_plausible = True
    result_warnings = list(warnings) if warnings else []
    
    if value is not None and plausibility_range is not None:
        float_value = float(value)
        min_val, max_val = plausibility_range
        
        if float_value < min_val:
            is_plausible = False
            result_warnings.append(
                f"Value {float_value:.2f} is below typical range ({min_val:.1f} to {max_val:.1f})"
            )
        elif float_value > max_val:
            is_plausible = False
            result_warnings.append(
                f"Value {float_value:.2f} is above typical range ({min_val:.1f} to {max_val:.1f})"
            )
    
    # Convert inputs to serializable format
    serializable_inputs = {}
    for key, val in inputs.items():
        if isinstance(val, Decimal):
            serializable_inputs[key] = float(val)
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
        warnings=result_warnings,
        category=category,
    )


class BaseCalculator(ABC):
    """
    Abstract base class for financial calculators.
    
    Provides common functionality for:
    - Input extraction and validation
    - Result creation with audit trail
    - Plausibility checking
    - Warning accumulation
    """
    
    def __init__(self):
        """Initialize the calculator."""
        self._warnings: list[str] = []
        self._steps: list[str] = []
    
    def reset(self) -> None:
        """Reset the calculator state for a new calculation."""
        self._warnings = []
        self._steps = []
    
    def add_step(self, description: str, value: Any = None) -> None:
        """
        Add a calculation step to the audit trail.
        
        Args:
            description: Description of the step
            value: Optional value to include
        """
        step_num = len(self._steps) + 1
        if value is not None:
            if isinstance(value, Decimal):
                self._steps.append(f"Step {step_num}: {description} = {float(value):,.4f}")
            else:
                self._steps.append(f"Step {step_num}: {description} = {value}")
        else:
            self._steps.append(f"Step {step_num}: {description}")
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self._warnings.append(warning)
    
    def get_steps(self) -> list[str]:
        """Get the calculation steps."""
        return self._steps.copy()
    
    def get_warnings(self) -> list[str]:
        """Get the warning messages."""
        return self._warnings.copy()
    
    def create_result(
        self,
        metric_name: str,
        value: Decimal | None,
        inputs: dict[str, Any],
        category: MetricCategory | None = None,
        unit: MetricUnit | None = None,
        formula: str | None = None,
    ) -> CalculationResult:
        """
        Create a calculation result with current steps and warnings.
        
        Args:
            metric_name: Name of the metric
            value: Calculated value
            inputs: Dictionary of input values
            category: Metric category
            unit: Override unit type
            formula: Override formula string
            
        Returns:
            Complete CalculationResult
        """
        return create_calculation_result(
            metric_name=metric_name,
            value=value,
            inputs=inputs,
            calculation_steps=self.get_steps(),
            category=category,
            warnings=self.get_warnings(),
            unit=unit,
            formula=formula,
        )
    
    @abstractmethod
    def calculate(self, *args, **kwargs) -> CalculationResult:
        """
        Perform the calculation.
        
        Must be implemented by subclasses.
        """
        pass
    
    def safe_divide_with_step(
        self,
        numerator: Decimal,
        denominator: Decimal,
        description: str,
        precision: int = DECIMAL_PLACES["ratio"],
    ) -> Decimal | None:
        """
        Perform safe division and add calculation step.
        
        Args:
            numerator: The dividend
            denominator: The divisor
            description: Description of what's being calculated
            precision: Decimal places for result
            
        Returns:
            Result or None if division by zero
        """
        if is_effectively_zero(denominator):
            self.add_step(f"{description}: Cannot calculate (denominator is zero)")
            self.add_warning(f"Division by zero in {description}")
            return None
        
        result = safe_divide(numerator, denominator, precision=precision)
        self.add_step(f"{description}", result)
        return result
    
    def calculate_average_with_step(
        self,
        value1: Decimal,
        value2: Decimal,
        description: str,
    ) -> Decimal:
        """
        Calculate average and add calculation step.
        
        Args:
            value1: First value
            value2: Second value
            description: Description of what's being averaged
            
        Returns:
            Average value
        """
        result = calculate_average(value1, value2)
        self.add_step(f"Calculate average {description}", result)
        return result

```

# finanalyst_tools/orchestration/report_generator.py
```6
# finanalyst_tools/orchestration/report_generator.py
"""
Report generator for financial analysis results.

Generates reports in the mandatory format specified in system prompt:
- Financial Analysis Report header
- Data Validation Summary
- Key Findings
- Detailed Metrics
- Calculation Audit Trail
- Recommendations
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
from datetime import datetime

from finanalyst_tools.models.analysis_results import (
    ComprehensiveAnalysisResult,
    MetricCollection,
    CalculationResult,
    ConfidenceAssessment,
)
from finanalyst_tools.utils.formatting import (
    format_currency,
    format_percentage,
    format_ratio,
    format_markdown_table,
    format_value_with_unit,
)


class ReportFormat(str, Enum):
    """Available report formats."""
    MARKDOWN = "markdown"
    JSON = "json"
    TEXT = "text"


def generate_financial_report(
    analysis_result: ComprehensiveAnalysisResult,
    format: ReportFormat = ReportFormat.MARKDOWN,
    include_audit_trail: bool = True,
) -> str:
    """
    Generate a financial analysis report.
    
    Args:
        analysis_result: Complete analysis result
        format: Output format
        include_audit_trail: Whether to include detailed calculation steps
        
    Returns:
        Formatted report string
    """
    generator = ReportGenerator(
        include_audit_trail=include_audit_trail,
    )
    
    if format == ReportFormat.MARKDOWN:
        return generator.generate_markdown(analysis_result)
    elif format == ReportFormat.JSON:
        return analysis_result.to_json()
    else:
        return generator.generate_text(analysis_result)


class ReportGenerator:
    """
    Generator for financial analysis reports.
    """
    
    def __init__(
        self,
        include_audit_trail: bool = True,
        include_warnings: bool = True,
        company_name: str | None = None,
    ):
        """
        Initialize the report generator.
        
        Args:
            include_audit_trail: Whether to include calculation steps
            include_warnings: Whether to include warning messages
            company_name: Optional company name for report header
        """
        self.include_audit_trail = include_audit_trail
        self.include_warnings = include_warnings
        self.company_name = company_name
    
    def generate_markdown(
        self,
        result: ComprehensiveAnalysisResult,
    ) -> str:
        """
        Generate report in Markdown format.
        
        Follows the mandatory template from system prompt.
        """
        sections = []
        
        # Header
        sections.append(self._generate_header(result))
        
        # Executive Summary with Confidence
        sections.append(self._generate_summary(result))
        
        # Data Validation Summary
        sections.append(self._generate_validation_section(result))
        
        # Key Findings
        sections.append(self._generate_key_findings(result))
        
        # Detailed Metrics
        sections.append(self._generate_metrics_section(result))
        
        # Audit Trail (optional)
        if self.include_audit_trail:
            sections.append(self._generate_audit_trail(result))
        
        # Recommendations
        sections.append(self._generate_recommendations(result))
        
        # Footer
        sections.append(self._generate_footer())
        
        return "\n\n".join(sections)
    
    def generate_text(
        self,
        result: ComprehensiveAnalysisResult,
    ) -> str:
        """Generate report in plain text format."""
        # Simplified version of markdown
        md = self.generate_markdown(result)
        # Remove markdown formatting
        text = md.replace("# ", "").replace("## ", "").replace("### ", "")
        text = text.replace("**", "").replace("*", "")
        text = text.replace("|", " ")
        return text
    
    def _generate_header(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate report header."""
        lines = ["# Financial Analysis Report"]
        
        if self.company_name:
            lines.append(f"\n**Company**: {self.company_name}")
        
        lines.append(f"**Analysis Type**: {result.analysis_type.title()}")
        lines.append(f"**Data Period**: {result.period}")
        lines.append(f"**Currency**: {result.currency}")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        return "\n".join(lines)
    
    def _generate_summary(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate executive summary with confidence level."""
        lines = ["## Executive Summary"]
        
        # Confidence level
        if result.confidence:
            lines.append(f"\n**Confidence Level**: {result.confidence.to_display()}")
        
        # Quick stats
        lines.append(f"\n**Metrics Calculated**: {result.total_metrics}")
        
        # Count warnings
        warning_count = len(result.all_warnings)
        if warning_count > 0:
            lines.append(f"**Warnings**: {warning_count}")
        
        return "\n".join(lines)
    
    def _generate_validation_section(self, result: ComprehensiveAnalysisResult) -> str:
        """Generate data validation summary."""
        lines = ["## 1. Data Validation Summary"]
        
        if result.validation_summary:
            is_valid = result.validation_summary.get("is_valid", True)
            error_count = result.validation_summary.get("error_count", 0)
            warning_count = result.validation_summary.get("warning_count", 0)
            
            status = "✅ Passed" if is_valid else "❌ Failed"
            lines.append(f"\n**Status**: {status}")
            
            if error_count > 0:
                lines.append(f"- Errors: {error_count}")
            if warning_count > 0:
                lines.append(f"- Warnings: {warning_count}")
            
            if is_valid and error_count == 0 and warning_count == 0:
                lines.append("- All validation checks passed")
        else:
            

```

# finanalyst_tools/orchestration/__init__.py
```6
# finanalyst_tools/orchestration/__init__.py
"""
Orchestration layer for FinAnalyst-Pro Agent Tools.

This package provides:
- Analysis pipeline with 5-phase workflow
- Confidence scoring for analysis results
- Report generation with mandatory template
"""

from finanalyst_tools.orchestration.pipeline import (
    AnalysisPipeline,
    AnalysisRequest,
    AnalysisPhase,
)

from finanalyst_tools.orchestration.confidence_scorer import (
    calculate_confidence_level,
    ConfidenceScorer,
)

from finanalyst_tools.orchestration.report_generator import (
    generate_financial_report,
    ReportGenerator,
    ReportFormat,
)


__all__ = [
    # Pipeline
    "AnalysisPipeline",
    "AnalysisRequest",
    "AnalysisPhase",
    # Confidence
    "calculate_confidence_level",
    "ConfidenceScorer",
    # Reporting
    "generate_financial_report",
    "ReportGenerator",
    "ReportFormat",
]

```

# finanalyst_tools/orchestration/confidence_scorer.py
```6
# finanalyst_tools/orchestration/confidence_scorer.py
"""
Confidence scoring for analysis results.

Implements the mandatory confidence level assessment:
- HIGH: Data quality excellent, all checks passed
- MEDIUM: Some warnings but analysis reliable
- LOW: Significant issues, interpret with caution
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

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


def calculate_confidence_level(
    validation_result: ValidationResult | None = None,
    plausibility_result: PlausibilityResult | None = None,
    reconciliation_result: ReconciliationResult | None = None,
    data_completeness: float = 1.0,
) -> ConfidenceAssessment:
    """
    Calculate confidence level for analysis results.
    
    Scoring factors:
    - Validation warnings: -5 points each
    - Validation errors: -20 points each (should not proceed)
    - Implausible metrics: -10 points each
    - Reconciliation failures: -15 points each
    - Data completeness: Up to -30 points for missing data
    
    Thresholds:
    - HIGH: Score >= 80
    - MEDIUM: Score >= 50
    - LOW: Score < 50
    
    Args:
        validation_result: Schema validation result
        plausibility_result: Plausibility check result
        reconciliation_result: Reconciliation check result
        data_completeness: Fraction of data present (0.0 to 1.0)
        
    Returns:
        ConfidenceAssessment with level and justification
    """
    score = 100.0
    factors: dict[str, str] = {}
    
    # Factor 1: Validation issues
    if validation_result:
        error_count = validation_result.error_count
        warning_count = validation_result.warning_count
        
        if error_count > 0:
            score -= error_count * 20
            factors["validation_errors"] = f"{error_count} error(s) found"
        
        if warning_count > 0:
            score -= warning_count * 5
            factors["validation_warnings"] = f"{warning_count} warning(s) found"
    
    # Factor 2: Plausibility failures
    if plausibility_result:
        implausible = plausibility_result.implausible_count
        if implausible > 0:
            score -= implausible * 10
            names = [c.metric_name for c in plausibility_result.implausible_checks[:3]]
            factors["implausible_metrics"] = f"{implausible} metric(s) outside range: {', '.join(names)}"
    
    # Factor 3: Reconciliation failures
    if reconciliation_result:
        failed = reconciliation_result.failed_count
        if failed > 0:
            score -= failed * 15
            names = [c.check_name for c in reconciliation_result.failed_checks[:3]]
            factors["reconciliation_failures"] = f"{failed} check(s) failed: {', '.join(names)}"
    
    # Factor 4: Data completeness
    if data_completeness < 1.0:
        completeness_penalty = (1.0 - data_completeness) * 30
        score -= completeness_penalty
        factors["data_completeness"] = f"{data_completeness:.0%} of expected data present"
    
    # Ensure score is in valid range
    score = max(0.0, min(100.0, score))
    
    # Determine level
    if score >= 80:
        level = ConfidenceLevel.HIGH
    elif score >= 50:
        level = ConfidenceLevel.MEDIUM
    else:
        level = ConfidenceLevel.LOW
    
    # Generate justification
    justification = _generate_justification(level, factors, score)
    
    return ConfidenceAssessment(
        level=level,
        justification=justification,
        factors=factors,
        score=score,
    )


def _generate_justification(
    level: ConfidenceLevel,
    factors: dict[str, str],
    score: float,
) -> str:
    """Generate human-readable justification for confidence level."""
    
    if level == ConfidenceLevel.HIGH:
        if not factors:
            return "All validation checks passed with no issues"
        return f"Data quality is good with minor observations: {len(factors)} factor(s) noted"
    
    elif level == ConfidenceLevel.MEDIUM:
        factor_summary = "; ".join(factors.values())[:100]
        return f"Analysis reliable with some caveats: {factor_summary}"
    
    else:  # LOW
        factor_summary = "; ".join(factors.values())[:100]
        return f"Significant issues detected: {factor_summary}. Interpret results with caution."


class ConfidenceScorer:
    """
    Class-based confidence scorer with customization options.
    """
    
    def __init__(
        self,
        error_penalty: float = 20.0,
        warning_penalty: float = 5.0,
        implausible_penalty: float = 10.0,
        reconciliation_penalty: float = 15.0,
        high_threshold: float = 80.0,
        medium_threshold: float = 50.0,
    ):
        """
        Initialize with custom scoring parameters.
        
        Args:
            error_penalty: Points deducted per validation error
            warning_penalty: Points deducted per validation warning
            implausible_penalty: Points deducted per implausible metric
            reconciliation_penalty: Points deducted per reconciliation failure
            high_threshold: Minimum score for HIGH confidence
            medium_threshold: Minimum score for MEDIUM confidence
        """
        self.error_penalty = error_penalty
        self.warning_penalty = warning_penalty
        self.implausible_penalty = implausible_penalty
        self.reconciliation_penalty = reconciliation_penalty
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
    
    def calculate(
        self,
        validation_result: ValidationResult | None = None,
        plausibility_result: PlausibilityResult | None = None,
        reconciliation_result: ReconciliationResult | None = None,
        data_completeness: float = 1.0,
    ) -> ConfidenceAssessment:
        """Calculate confidence using instance parameters."""
        
        score = 100.0
        factors: dict[str, str] = {}
        
        if validation_result:
            score -= validation_result.error_count * self.error_penalty
            score -= validation_result.warning_count * self.warning_penalty
            if validation_result.error_count:
                factors["errors"] = f"{validation_result.error_count} error(s)"
            if validation_result.warning_count:
                factors["warnings"] = f"{validation_result.warning_count} warning(s)"
        
        if plausibility_result:
            score -= plausibility_result.implausible_count * self.implausible_penalty
            if plausibility_result.implausible_count:
                factors["implausible"] = f"{plausibility_result.implausible_count} metric(s)"
        
        if reconciliation_result:
            score -= reconciliation_result.failed_count * self.reconciliation_penalty
            if reconciliation_result.failed_count:
                factors["reconciliation"] = f"{reconciliation_result.failed_count} failure(s)"
        
        if data_completeness < 1.0:
            score -= (1.0 - data_completeness) * 30
            factors["completeness"] = f"{data_completeness:.0%}"
        
        score = max(0.0, min(100.0, score))
        
        if score >= self.high_threshold:
            level = ConfidenceLevel.HIGH
        elif score >= self.medium_threshold:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW
        
        justification = _generate_justification(level, factors, score)
        
        return ConfidenceAssessment(
            level=level,
            justification=justification,
            factors=factors,
            score=score,
        )

```

# finanalyst_tools/orchestration/pipeline.py
```6
# finanalyst_tools/orchestration/pipeline.py
"""
Analysis pipeline implementing the mandatory 5-phase workflow.

Phases:
1. VALIDATE - Schema validation, completeness check
2. ANALYZE - Identify what to calculate based on data
3. CALCULATE - Execute calculations with audit trail
4. INTERPRET - Add context and insights
5. VERIFY - Pre-delivery checks

This ensures consistent, auditable analysis execution.
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
    CalculationResult,
)
from finanalyst_tools.models.validation import (
    ValidationResult,
    ReconciliationResult,
    PlausibilityResult,
)
from finanalyst_tools.validation.schema_validator import (
    validate_statement_set,
    validate_financial_data_completeness,
)
from finanalyst_tools.validation.reconciliation import run_all_reconciliations
from finanalyst_tools.validation.plausibility import check_all_plausibility
from finanalyst_tools.calculations.profitability import calculate_all_profitability_metrics
from finanalyst_tools.calculations.liquidity import calculate_all_liquidity_metrics
from finanalyst_tools.orchestration.confidence_scorer import calculate_confidence_level


class AnalysisPhase(str, Enum):
    """Phases of the analysis pipeline."""
    VALIDATE = "validate"
    ANALYZE = "analyze"
    CALCULATE = "calculate"
    INTERPRET = "interpret"
    VERIFY = "verify"


@dataclass
class AnalysisRequest:
    """
    Request for financial analysis.
    
    Attributes:
        statement_set: Financial statements to analyze
        prior_statement_set: Prior period statements (optional)
        analysis_type: Type of analysis requested
        include_trends: Whether to include trend analysis
        currency: Currency for reporting
    """
    statement_set: FinancialStatementSet
    prior_statement_set: FinancialStatementSet | None = None
    analysis_type: str = "comprehensive"
    include_trends: bool = False
    currency: str = "SGD"


@dataclass
class PipelineState:
    """
    Internal state of the pipeline during execution.
    """
    current_phase: AnalysisPhase = AnalysisPhase.VALIDATE
    validation_result: ValidationResult | None = None
    reconciliation_result: ReconciliationResult | None = None
    plausibility_result: PlausibilityResult | None = None
    metric_collections: list[MetricCollection] = field(default_factory=list)
    all_metrics: list[CalculationResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    phase_completed: dict[AnalysisPhase, bool] = field(default_factory=dict)


class AnalysisPipeline:
    """
    Pipeline for executing financial analysis.
    
    Implements the mandatory 5-phase workflow:
    REQUEST → [1.VALIDATE] → [2.ANALYZE] → [3.CALCULATE] → [4.INTERPRET] → [5.VERIFY] → DELIVER
    """
    
    def __init__(self):
        """Initialize the pipeline."""
        self.state: PipelineState | None = None
    
    def execute(self, request: AnalysisRequest) -> ComprehensiveAnalysisResult:
        """
        Execute the full analysis pipeline.
        
        Args:
            request: Analysis request with financial data
            
        Returns:
            ComprehensiveAnalysisResult with all analysis outputs
        """
        # Initialize state
        self.state = PipelineState()
        
        # Phase 1: VALIDATE
        self._phase_validate(request)
        if not self.state.validation_result.can_proceed:
            return self._create_error_result(request, "Validation failed")
        
        # Phase 2: ANALYZE
        analysis_plan = self._phase_analyze(request)
        
        # Phase 3: CALCULATE
        self._phase_calculate(request, analysis_plan)
        
        # Phase 4: INTERPRET
        self._phase_interpret(request)
        
        # Phase 5: VERIFY
        self._phase_verify(request)
        
        # Create final result
        return self._create_result(request)
    
    def _phase_validate(self, request: AnalysisRequest) -> None:
        """
        Phase 1: VALIDATE
        
        - Schema validation
        - Data completeness check
        - Cross-statement reconciliation
        """
        self.state.current_phase = AnalysisPhase.VALIDATE
        
        # Schema validation
        validation = validate_statement_set(
            request.statement_set,
            request.analysis_type,
        )
        self.state.validation_result = validation
        
        if not validation.can_proceed:
            self.state.errors.append("Schema validation failed")
            return
        
        # Reconciliation (if cash flow available)
        prior_bs = None
        if request.prior_statement_set:
            prior_bs = request.prior_statement_set.balance_sheet
        
        reconciliation = run_all_reconciliations(
            request.statement_set,
            prior_balance_sheet=prior_bs,
        )
        self.state.reconciliation_result = reconciliation
        
        if not reconciliation.all_passed:
            for check in reconciliation.failed_checks:
                self.state.warnings.append(f"Reconciliation: {check.message}")
        
        self.state.phase_completed[AnalysisPhase.VALIDATE] = True
    
    def _phase_analyze(self, request: AnalysisRequest) -> dict[str, bool]:
        """
        Phase 2: ANALYZE
        
        Determine what calculations to perform based on:
        - Analysis type requested
        - Data available
        
        Returns:
            Dictionary of metric categories to calculate
        """
        self.state.current_phase = AnalysisPhase.ANALYZE
        
        analysis_plan = {
            "profitability": False,
            "liquidity": False,
            "solvency": False,
            "efficiency": False,
        }
        
        analysis_type = request.analysis_type.lower()
        
        if analysis_type in ("profitability", "comprehensive"):
            analysis_plan["profitability"] = True
        
        if analysis_type in ("liquidity", "comprehensive"):
            analysis_plan["liquidity"] = True
        
        if analysis_type in ("solvency", "comprehensive"):
            analysis_plan["solvency"] = True
        
        if analysis_type in ("efficiency", "comprehensive"):
            analysis_plan["efficiency"] = True
        
        self.state.phase_completed[AnalysisPhase.ANALYZE] = True
        return analysis_plan
    
    def _phase_calculate(
        self,
        request: AnalysisRequest,
        analysis_plan: dict[str, bool],
    ) -> None:
        """
        Phase 3: CALCULATE
        
        Execute all planned calculations.
        """
        self.state.current_phase = AnalysisPhase.CALCULATE
        
        prior_bs = None
        if request.prior_statement_set:
            prior_bs = request.prior_statement_set.balance_sheet
        
        # Profitability metrics
        if analysis_plan.get("profitability"):
            profitability = calculate_all_profitability_metrics(
                income_statement=request.statement_set.income_statement,
                balance_sheet=request.statement_set.balance_sheet,
                prior_balance_sheet=prior_bs,
            )
            self.state.metric_collections.append(profitability)
            self.state.all_metrics.extend(profitability.metrics)
        
        # Liquidity metrics
        if analysis_plan.get("liquidity"):
            liquidity = calculate_all_liquidity_metrics(
                balance_sheet=request.statement_set.balance_sheet,
            )
            self.state.metric_collections.append(liquidity)
            self.state.all_metrics.extend(liquidity.metrics)
        
        # Note: Solvency and Efficiency calculations would be added in Phase 2
        
        self.state.phase_completed[AnalysisPhase.CALCULATE] = True
    
    def _phase_interpret(self, request: AnalysisRequest) -> None:
        """
        Phase 4: INTERPRET
        
        Add context and insights to calculated metrics.
        """
        self.state.current_phase = AnalysisPhase.INTERPRET
        
        # Plausibility checks on all metrics
        plausibility = check_all_plausibility(self.state.all_metrics)
        self.state.plausibility_result = plausibility
        
        # Add warnings for implausible values
        for check in plausibility.implausible_checks:
            self.state.warnings.append(f"Plausibility: {check.message}")
        
        self.state.phase_completed[AnalysisPhase.INTERPRET] = True
    
    def _phase_verify(self, request: AnalysisRequest) -> None:
        """
        Phase 5: VERIFY
        
        Pre-delivery checks:
        - Ensure all requested calculations completed
        - Verify no critical errors
        - Final quality check
        """
        self.state.current_phase = AnalysisPhase.VERIFY
        
        # Check that calculations were performed
        if not self.state.metric_collections:
            self.state.warnings.append("No metrics were calculated")
        
        # Check for any uncalculable metrics
        uncalculable = [m for m in self.state.all_metrics if m.value is None]
        if uncalculable:
            for m in uncalculable:
                self.state.warnings.append(f"Could not calculate: {m.metric_name}")
        
        self.state.phase_completed[AnalysisPhase.VERIFY] = True
    
    def _create_result(self, request: AnalysisRequest) -> ComprehensiveAnalysisResult:
        """Create the final analysis result."""
        
        # Calculate confidence
        data_completeness = 1.0
        if self.state.validation_result:
            total_issues = self.state.validation_result.total_issue_count
            data_completeness = max(0.0, 1.0 - (total_issues * 0.1))
        
        confidence = calculate_confidence_level(
            validation_result=self.state.validation_result,
            plausibility_result=self.state.plausibility_result,
            reconciliation_result=self.state.reconciliation_result,
            data_completeness=data_completeness,
        )
        
        # Build result
        result = ComprehensiveAnalysisResult(
            analysis_type=request.analysis_type,
            period=str(request.statement_set.period),
            currency=request.currency,
            metric_collections=self.state.metric_collections,
            confidence=confidence,
        )
        
        # Add validation summary
        if self.state.validation_result:
            result.validation_summary = self.state.validation_result.to_dict()
        
        # Add reconciliation summary
        if self.state.reconciliation_result:
            result.reconciliation_summary = self.state.reconciliation_result.to_dict()
        
        # Add recommendations based on findings
        result.recommendations = self._generate_recommendations()
        
        return result
    
    def _create_error_result(
        self,
        request: AnalysisRequest,
        error_message: str,
    ) -> ComprehensiveAnalysisResult:
        """Create an error result when pipeline fails."""
        result = ComprehensiveAnalysisResult(
            analysis_type=request.analysis_type,
            period=str(request.statement_set.period),
            currency=request.currency,
        )
        
        if self.state.validation_result:
            result.validation_summary = self.state.validation_result.to_dict()
        
        result.recommendations = [
            f"Analysis could not be completed: {error_message}",
            "Please address validation errors and retry",
        ]
        
        return result
    
    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on analysis findings."""
        recommendations = []
        
        # Based on profitability
        for collection in self.state.metric_collections:
            if collection.category == MetricCategory.PROFITABILITY:
                npm = collection.get_metric("Net Profit Margin")
                if npm and npm.value is not None:
                    if npm.value < Decimal("5"):
                        recommendations.append(
                            "Net profit margin is low (<5%). Consider reviewing cost structure."
                        )
                    elif npm.value > Decimal("20"):
                        recommendations.append(
                            "Strong net profit margin (>20%). Consider reinvestment opportunities."
                        )
        
        # Based on liquidity
        for collection in self.state.metric_collections:
            if collection.category == MetricCategory.LIQUIDITY:
                cr = collection.get_metric("Current Ratio")
                if cr and cr.value is not None:
                    if cr.value < Decimal("1"):
                        recommendations.append(
                            "Current ratio below 1.0 indicates liquidity risk. Review working capital management."
                        )
        
        # Default recommendation if none generated
        if not recommendations:
            recommendations.append(
                "Financial metrics are within normal ranges. Continue monitoring key indicators."
            )
        
        return recommendations

```

# finanalyst_tools/dispatcher.py
```6
# finanalyst_tools/dispatcher.py
"""
Tool dispatcher for executing tool calls from LLM.

Provides:
- Parameter validation and type coercion
- Execution timing
- Error handling and formatting
- Structured result formatting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any
import json
import time

from finanalyst_tools.tool_registry import TOOL_REGISTRY, ToolDefinition, ToolParameter
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
        success: Whether execution succeeded
        result: The result (if successful)
        error: Error message (if failed)
        error_details: Additional error context
        execution_time_ms: Execution time in milliseconds
    """
    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    error_details: dict[str, Any] | None = None
    execution_time_ms: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data: dict[str, Any] = {
            "tool_name": self.tool_name,
            "success": self.success,
        }
        
        if self.success:
            if hasattr(self.result, "to_dict"):
                data["result"] = self.result.to_dict()
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


class ToolDispatcher:
    """
    Dispatcher for executing tool calls.
    
    Handles:
    - Tool lookup
    - Parameter validation
    - Type coercion (string → Decimal for numbers)
    - Execution with timing
    - Error handling
    """
    
    def __init__(self):
        """Initialize the dispatcher."""
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
            parameters: Dictionary of parameters
            
        Returns:
            ToolCallResult with execution outcome
        """
        start_time = time.perf_counter()
        
        try:
            # Get tool definition
            tool = self.registry.get(tool_name)
            if tool is None:
                available = self.registry.list_tool_names()
                raise ToolNotFoundError(tool_name, available)
            
            # Validate parameters
            validated_params = self._validate_and_coerce_parameters(tool, parameters)
            
            # Execute the tool
            if tool.function is None:
                raise ToolExecutionError(
                    tool_name=tool_name,
                    original_error=ValueError("Tool function not registered"),
                )
            
            result = tool.function(**validated_params)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            return ToolCallResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=execution_time,
            )
            
        except FinAnalystError as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                error_details=e.to_dict(),
                execution_time_ms=execution_time,
            )
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=f"Unexpected error: {str(e)}",
                error_details={
                    "error_type": type(e).__name__,
                    "message": str(e),
                },
                execution_time_ms=execution_time,
            )
    
    def _validate_and_coerce_parameters(
        self,
        tool: ToolDefinition,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate and coerce parameters for a tool.
        
        Args:
            tool: Tool definition
            parameters: Raw parameters
            
        Returns:
            Validated and coerced parameters
            
        Raises:
            ToolParameterError: If validation fails
        """
        result = {}
        
        # Check required parameters
        for param in tool.parameters:
            if param.required and param.name not in parameters:
                raise ToolParameterError(
                    tool_name=tool.name,
                    parameter_name=param.name,
                    message="Required parameter is missing",
                    expected_type=param.type,
                )
        
        # Validate and coerce each provided parameter
        for param in tool.parameters:
            if param.name in parameters:
                value = parameters[param.name]
                coerced = self._coerce_parameter(tool.name, param, value)
                result[param.name] = coerced
            elif param.default is not None:
                result[param.name] = param.default
        
        return result
    
    def _coerce_parameter(
        self,
        tool_name: str,
        param: ToolParameter,
        value: Any,
    ) -> Any:
        """
        Coerce a parameter value to the expected type.
        
        Args:
            tool_name: Tool name (for error messages)
            param: Parameter definition
            value: Raw value
            
        Returns:
            Coerced value
            
        Raises:
            ToolParameterError: If coercion fails
        """
        if value is None:
            if param.required:
                raise ToolParameterError(
                    tool_name=tool_name,
                    parameter_name=param.name,
                    message="Value cannot be None",
                    expected_type=param.type,
                )
            return param.default
        
        try:
            if param.type == "number":
                # Convert to Decimal for financial precision
                if isinstance(value, Decimal):
                    return value
                if isinstance(value, (int, float)):
                    return Decimal(str(value))
                if isinstance(value, str):
                    return Decimal(value)
                raise ValueError(f"Cannot convert {type(value).__name__} to number")
                
            elif param.type == "integer":
                return int(value)
                
            elif param.type == "boolean":
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes")
                return bool(value)
                
            elif param.type == "string":
                return str(value)
                
            elif param.type == "object":
                if isinstance(value, dict):
                    return value
                if isinstance(value, str):
                    return json.loads(value)
                raise ValueError("Expected object/dictionary")
                
            elif param.type == "array":
                if isinstance(value, list):
                    return value
                if isinstance(value, str):
                    return json.loads(value)
                raise ValueError("Expected array/list")
                
            else:
                # Unknown type - pass through
                return value
                
        except (ValueError, InvalidOperation, json.JSONDecodeError) as e:
            raise ToolParameterError(
                tool_name=tool_name,
                parameter_name=param.name,
                message=f"Cannot convert to {param.type}: {str(e)}",
                expected_type=param.type,
                actual_value=value,
            )
    
    def list_tools(self) -> list[str]:
        """List all available tool names."""
        return self.registry.list_tool_names()
    
    def get_tool_info(self, tool_name: str) -> dict[str, Any] | None:
        """Get information about a specific tool."""
        tool = self.registry.get(tool_name)
        if tool:
            return tool.to_dict()
        return None


# Global singleton instance
DISPATCHER = ToolDispatcher()


def execute_tool(tool_name: str, parameters: dict[str, Any]) -> ToolCallResult:
    """
    Convenience function to execute a tool.
    
    Args:
        tool_name: Name of the tool
        parameters: Tool parameters
        
    Returns:
        ToolCallResult
    """
    return DISPATCHER.execute(tool_name, parameters)

```

# finanalyst_tools/tool_registry.py
```6
# finanalyst_tools/tool_registry.py
"""
Central registry of all tools for LLM function calling.

Provides:
- Tool definitions with complete metadata
- OpenAI and Anthropic schema generation
- Tool discovery and filtering
- Parameter validation schemas
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
    CASH_FLOW = "cash_flow"
    RECONCILIATION = "reconciliation"
    FORMATTING = "formatting"
    UTILITY = "utility"
    ANALYSIS = "analysis"


@dataclass
class ToolParameter:
    """
    Definition of a tool parameter.
    
    Attributes:
        name: Parameter name
        type: Parameter type (string, number, integer, boolean, object, array)
        description: Human-readable description
        required: Whether the parameter is required
        default: Default value if not provided
        enum: List of allowed values (for string enums)
    """
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None
    
    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI parameter schema."""
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema
    
    def to_anthropic_schema(self) -> dict[str, Any]:
        """Convert to Anthropic parameter schema."""
        # Anthropic uses similar format to OpenAI
        return self.to_openai_schema()


@dataclass
class ToolDefinition:
    """
    Complete definition of a tool.
    
    Attributes:
        name: Tool name (function name for calling)
        description: Human-readable description
        category: Tool category
        parameters: List of parameters
        returns: Description of return value
        example: Example usage
        function: Reference to the actual function
    """
    name: str
    description: str
    category: ToolCategory
    parameters: list[ToolParameter]
    returns: str
    example: str = ""
    function: Callable | None = None
    
    @property
    def required_parameters(self) -> list[str]:
        """Get list of required parameter names."""
        return [p.name for p in self.parameters if p.required]
    
    @property
    def optional_parameters(self) -> list[str]:
        """Get list of optional parameter names."""
        return [p.name for p in self.parameters if not p.required]
    
    def to_openai_schema(self) -> dict[str, Any]:
        """
        Convert to OpenAI function calling schema.
        
        Returns:
            Dictionary in OpenAI function format
        """
        properties = {}
        for param in self.parameters:
            properties[param.name] = param.to_openai_schema()
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": self.required_parameters,
                },
            },
        }
    
    def to_anthropic_schema(self) -> dict[str, Any]:
        """
        Convert to Anthropic tool use schema.
        
        Returns:
            Dictionary in Anthropic tool format
        """
        properties = {}
        for param in self.parameters:
            properties[param.name] = param.to_anthropic_schema()
        
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": self.required_parameters,
            },
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    
    Provides:
    - Tool registration and lookup
    - Filtering by category
    - Schema generation for LLM integration
    """
    
    def __init__(self):
        """Initialize the registry."""
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
            ToolDefinition or None if not found
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
            List of tool definitions
        """
        if category is None:
            return list(self._tools.values())
        return [t for t in self._tools.values() if t.category == category]
    
    def list_tool_names(
        self,
        category: ToolCategory | None = None,
    ) -> list[str]:
        """
        List tool names, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tool names
        """
        return [t.name for t in self.list_tools(category)]
    
    def get_openai_tools(
        self,
        categories: list[ToolCategory] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get tools in OpenAI function calling format.
        
        Args:
            categories: Optional list of categories to include
            
        Returns:
            List of tool schemas
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
        Get tools in Anthropic tool use format.
        
        Args:
            categories: Optional list of categories to include
            
        Returns:
            List of tool schemas
        """
        tools = self._tools.values()
        if categories:
            tools = [t for t in tools if t.category in categories]
        return [t.to_anthropic_schema() for t in tools]
    
    def get_tool_descriptions(self) -> str:
        """
        Get human-readable descriptions of all tools.
        
        Returns:
            Formatted string with tool descriptions
        """
        lines = ["# Available Tools\n"]
        
        # Group by category
        by_category: dict[ToolCategory, list[ToolDefinition]] = {}
        for tool in self._tools.values():
            if tool.category not in by_category:
                by_category[tool.category] = []
            by_category[tool.category].append(tool)
        
        for category in ToolCategory:
            if category in by_category:
                lines.append(f"\n## {category.value.title()}\n")
                for tool in by_category[category]:
                    lines.append(f"### {tool.name}")
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
        
        # Import calculation functions
        from finanalyst_tools.calculations.profitability import (
            calculate_gross_profit_margin,
            calculate_operating_profit_margin,
            calculate_net_profit_margin,
            calculate_ebitda_margin,
            calculate_return_on_assets,
            calculate_return_on_equity,
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
        
        # ================================================================
        # VALIDATION TOOLS
        # ================================================================
        
        self.register(ToolDefinition(
            name="validate_financial_data",
            description="Validate that financial data is complete and correct for a specific analysis type",
            category=ToolCategory.VALIDATION,
            parameters=[
                ToolParameter(
                    name="income_statement",
                    type="object",
                    description="Income statement data with revenue, COGS, expenses, and net income",
                    required=False,
                ),
                ToolParameter(
                    name="balance_sheet",
                    type="object",
                    description="Balance sheet data with assets, liabilities, and equity",
                    required=False,
                ),
                ToolParameter(
                    name="cash_flow",
                    type="object",
                    description="Cash flow statement data",
                    required=False,
                ),
                ToolParameter(
                    name="analysis_type",
                    type="string",
                    description="Type of analysis to validate for",
                    required=True,
                    enum=["profitability", "liquidity", "solvency", "efficiency", "comprehensive"],
                ),
            ],
            returns="ValidationResult with any issues found",
            example="validate_financial_data(income_statement={...}, balance_sheet={...}, analysis_type='profitability')",
            function=validate_financial_data_completeness,
        ))
        
        # ================================================================
        # PROFITABILITY TOOLS
        # ================================================================
        
        self.register(ToolDefinition(
            name="calculate_gross_profit_margin",
            description="Calculate gross profit margin: (Revenue - COGS) / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter(
                    name="revenue",
                    type="number",
                    description="Total revenue / net sales",
                    required=True,
                ),
                ToolParameter(
                    name="cost_of_goods_sold",
                    type="number",
                    description="Cost of goods sold / cost of sales",
                    required=True,
                ),
            ],
            returns="CalculationResult with gross profit margin percentage and calculation steps",
            example="calculate_gross_profit_margin(revenue=1000000, cost_of_goods_sold=600000)",
            function=calculate_gross_profit_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_operating_profit_margin",
            description="Calculate operating profit margin: (Revenue - COGS - OpEx) / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter(
                    name="revenue",
                    type="number",
                    description="Total revenue",
                    required=True,
                ),
                ToolParameter(
                    name="cost_of_goods_sold",
                    type="number",
                    description="Cost of goods sold",
                    required=True,
                ),
                ToolParameter(
                    name="operating_expenses",
                    type="number",
                    description="Total operating expenses",
                    required=True,
                ),
                ToolParameter(
                    name="marketing_expenses",
                    type="number",
                    description="Marketing expenses (if separate from OpEx)",
                    required=False,
                ),
            ],
            returns="CalculationResult with operating margin percentage and calculation steps",
            example="calculate_operating_profit_margin(revenue=1000000, cost_of_goods_sold=600000, operating_expenses=200000)",
            function=calculate_operating_profit_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_net_profit_margin",
            description="Calculate net profit margin: Net Income / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter(
                    name="revenue",
                    type="number",
                    description="Total revenue",
                    required=True,
                ),
                ToolParameter(
                    name="net_income",
                    type="number",
                    description="Net income (bottom line profit)",
                    required=True,
                ),
            ],
            returns="CalculationResult with net margin percentage and calculation steps",
            example="calculate_net_profit_margin(revenue=1000000, net_income=100000)",
            function=calculate_net_profit_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_ebitda_margin",
            description="Calculate EBITDA margin: EBITDA / Revenue × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter(
                    name="revenue",
                    type="number",
                    description="Total revenue",
                    required=True,
                ),
                ToolParameter(
                    name="ebitda",
                    type="number",
                    description="EBITDA (Earnings Before Interest, Taxes, Depreciation, Amortization)",
                    required=True,
                ),
            ],
            returns="CalculationResult with EBITDA margin percentage",
            example="calculate_ebitda_margin(revenue=1000000, ebitda=250000)",
            function=calculate_ebitda_margin,
        ))
        
        self.register(ToolDefinition(
            name="calculate_return_on_assets",
            description="Calculate ROA: Net Income / Average Total Assets × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter(
                    name="net_income",
                    type="number",
                    description="Net income for the period",
                    required=True,
                ),
                ToolParameter(
                    name="total_assets_beginning",
                    type="number",
                    description="Total assets at beginning of period",
                    required=True,
                ),
                ToolParameter(
                    name="total_assets_ending",
                    type="number",
                    description="Total assets at end of period",
                    required=True,
                ),
            ],
            returns="CalculationResult with ROA percentage",
            example="calculate_return_on_assets(net_income=100000, total_assets_beginning=500000, total_assets_ending=550000)",
            function=calculate_return_on_assets,
        ))
        
        self.register(ToolDefinition(
            name="calculate_return_on_equity",
            description="Calculate ROE: Net Income / Average Shareholders' Equity × 100",
            category=ToolCategory.PROFITABILITY,
            parameters=[
                ToolParameter(
                    name="net_income",
                    type="number",
                    description="Net income for the period",
                    required=True,
                ),
                ToolParameter(
                    name="equity_beginning",
                    type="number",
                    description="Shareholders' equity at beginning of period",
                    required=True,
                ),
                ToolParameter(
                    name="equity_ending",
                    type="number",
                    description="Shareholders' equity at end of period",
                    required=True,
                ),
            ],
            returns="CalculationResult with ROE percentage",
            example="calculate_return_on_equity(net_income=100000, equity_beginning=300000, equity_ending=350000)",
            function=calculate_return_on_equity,
        ))
        
        # ================================================================
        # LIQUIDITY TOOLS
        # ================================================================
        
        self.register(ToolDefinition(
            name="calculate_current_ratio",
            description="Calculate current ratio: Current Assets / Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter(
                    name="current_assets",
                    type="number",
                    description="Total current assets",
                    required=True,
                ),
                ToolParameter(
                    name="current_liabilities",
                    type="number",
                    description="Total current liabilities",
                    required=True,
                ),
            ],
            returns="CalculationResult with current ratio and interpretation",
            example="calculate_current_ratio(current_assets=500000, current_liabilities=300000)",
            function=calculate_current_ratio,
        ))
        
        self.register(ToolDefinition(
            name="calculate_quick_ratio",
            description="Calculate quick ratio (acid test): (Current Assets - Inventory) / Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter(
                    name="current_assets",
                    type="number",
                    description="Total current assets",
                    required=True,
                ),
                ToolParameter(
                    name="inventory",
                    type="number",
                    description="Inventory value",
                    required=True,
                ),
                ToolParameter(
                    name="current_liabilities",
                    type="number",
                    description="Total current liabilities",
                    required=True,
                ),
            ],
            returns="CalculationResult with quick ratio",
            example="calculate_quick_ratio(current_assets=500000, inventory=150000, current_liabilities=300000)",
            function=calculate_quick_ratio,
        ))
        
        self.register(ToolDefinition(
            name="calculate_cash_ratio",
            description="Calculate cash ratio: Cash and Equivalents / Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter(
                    name="cash_and_equivalents",
                    type="number",
                    description="Cash and cash equivalents",
                    required=True,
                ),
                ToolParameter(
                    name="current_liabilities",
                    type="number",
                    description="Total current liabilities",
                    required=True,
                ),
            ],
            returns="CalculationResult with cash ratio",
            example="calculate_cash_ratio(cash_and_equivalents=100000, current_liabilities=300000)",
            function=calculate_cash_ratio,
        ))
        
        self.register(ToolDefinition(
            name="calculate_working_capital",
            description="Calculate working capital: Current Assets - Current Liabilities",
            category=ToolCategory.LIQUIDITY,
            parameters=[
                ToolParameter(
                    name="current_assets",
                    type="number",
                    description="Total current assets",
                    required=True,
                ),
                ToolParameter(
                    name="current_liabilities",
                    type="number",
                    description="Total current liabilities",
                    required=True,
                ),
            ],
            returns="CalculationResult with working capital amount",
            example="calculate_working_capital(current_assets=500000, current_liabilities=300000)",
            function=calculate_working_capital,
        ))


# Global singleton instance
TOOL_REGISTRY = ToolRegistry()

```

