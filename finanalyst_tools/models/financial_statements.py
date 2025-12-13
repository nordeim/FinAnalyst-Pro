# File: finanalyst_tools/models/financial_statements.py
"""
Pydantic models for financial statement data structures.

These models provide:
- Strong type validation for financial data
- Field aliases for common naming variations
- Calculated properties for derived values
- JSON serialization for LLM communication
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator, model_validator

from finanalyst_tools.utils.serialization import to_jsonable


class StatementType(str, Enum):
    """Types of financial statements."""
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"


class PeriodType(str, Enum):
    """Financial reporting period types."""
    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    TTM = "trailing_twelve_months"


# Type aliases for documentation
MonetaryValue = Annotated[Decimal, Field(description="Monetary value in base currency")]
Percentage = Annotated[Decimal, Field(description="Percentage value")]
Ratio = Annotated[Decimal, Field(description="Ratio value")]


class FinancialPeriod(BaseModel):
    """
    Represents a financial reporting period.
    
    Examples:
        - Annual: FinancialPeriod(year=2023, period_type=PeriodType.ANNUAL)
        - Quarterly: FinancialPeriod(year=2023, period_type=PeriodType.QUARTERLY, quarter=2)
    """
    
    year: int = Field(..., ge=1900, le=2100, description="Fiscal year")
    period_type: PeriodType = Field(default=PeriodType.ANNUAL)
    quarter: int | None = Field(default=None, ge=1, le=4, description="Quarter number (1-4)")
    month: int | None = Field(default=None, ge=1, le=12, description="Month number (1-12)")
    start_date: date | None = Field(default=None, description="Period start date")
    end_date: date | None = Field(default=None, description="Period end date")
    
    model_config = {"extra": "forbid"}
    
    @model_validator(mode="after")
    def validate_period_details(self) -> "FinancialPeriod":
        """Validate period-specific fields."""
        if self.period_type == PeriodType.QUARTERLY and self.quarter is None:
            raise ValueError("Quarter must be specified for quarterly periods")
        if self.period_type == PeriodType.MONTHLY and self.month is None:
            raise ValueError("Month must be specified for monthly periods")
        return self
    
    def __str__(self) -> str:
        """Human-readable period representation."""
        if self.period_type == PeriodType.QUARTERLY:
            return f"Q{self.quarter} {self.year}"
        if self.period_type == PeriodType.MONTHLY:
            return f"{self.year}-{self.month:02d}"
        if self.period_type == PeriodType.TTM:
            return f"TTM {self.year}"
        return str(self.year)
    
    def __lt__(self, other: "FinancialPeriod") -> bool:
        """Enable sorting by period."""
        if self.year != other.year:
            return self.year < other.year
        self_sub = self.quarter or self.month or 0
        other_sub = other.quarter or other.month or 0
        return self_sub < other_sub
    
    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, FinancialPeriod):
            return False
        return (
            self.year == other.year and
            self.period_type == other.period_type and
            self.quarter == other.quarter and
            self.month == other.month
        )
    
    def __hash__(self) -> int:
        """Make hashable for use in sets/dicts."""
        return hash((self.year, self.period_type, self.quarter, self.month))


class IncomeStatementData(BaseModel):
    """
    Income Statement / Profit & Loss data model.
    
    Supports multiple naming conventions through field aliases.
    Provides calculated properties for derived values.
    """
    
    period: FinancialPeriod
    currency: str = Field(default="SGD", min_length=3, max_length=3)
    
    # ─────────────────────────────────────────────────────────────────────
    # Revenue
    # ─────────────────────────────────────────────────────────────────────
    total_revenue: MonetaryValue = Field(
        ...,
        alias="revenue",
        description="Total revenue / net sales"
    )
    cost_of_goods_sold: MonetaryValue = Field(
        ...,
        alias="cogs",
        description="Cost of goods sold / cost of sales"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Operating Expenses
    # ─────────────────────────────────────────────────────────────────────
    operating_expenses: MonetaryValue | None = Field(
        default=None,
        alias="opex",
        description="Total operating expenses"
    )
    selling_general_admin: MonetaryValue | None = Field(
        default=None,
        alias="sga",
        description="Selling, General & Administrative expenses"
    )
    marketing_expenses: MonetaryValue | None = Field(
        default=None,
        description="Marketing and advertising expenses"
    )
    research_development: MonetaryValue | None = Field(
        default=None,
        alias="r_and_d",
        description="Research & Development expenses"
    )
    depreciation_amortization: MonetaryValue | None = Field(
        default=None,
        alias="d_and_a",
        description="Depreciation and amortization"
    )
    other_operating_expenses: MonetaryValue | None = Field(
        default=None,
        description="Other operating expenses"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Non-Operating Items
    # ─────────────────────────────────────────────────────────────────────
    interest_income: MonetaryValue = Field(default=Decimal("0"))
    interest_expense: MonetaryValue = Field(default=Decimal("0"))
    other_income: MonetaryValue = Field(default=Decimal("0"))
    other_expenses: MonetaryValue = Field(default=Decimal("0"))
    
    # ─────────────────────────────────────────────────────────────────────
    # Taxes and Bottom Line
    # ─────────────────────────────────────────────────────────────────────
    income_tax_expense: MonetaryValue = Field(default=Decimal("0"))
    net_income: MonetaryValue | None = Field(default=None)
    
    # ─────────────────────────────────────────────────────────────────────
    # Per Share Data
    # ─────────────────────────────────────────────────────────────────────
    earnings_per_share: Decimal | None = Field(default=None, alias="eps")
    diluted_eps: Decimal | None = Field(default=None)
    shares_outstanding: int | None = Field(default=None)
    
    model_config = {
        "populate_by_name": True,
        "extra": "allow",
    }
    
    @field_validator("total_revenue", "cost_of_goods_sold", mode="before")
    @classmethod
    def convert_to_decimal(cls, v: Any) -> Decimal:
        """Convert numeric inputs to Decimal."""
        if v is None:
            raise ValueError("This field is required")
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))
    
    @property
    def gross_profit(self) -> Decimal:
        """Calculate gross profit."""
        return self.total_revenue - self.cost_of_goods_sold
    
    @property
    def total_operating_expenses(self) -> Decimal:
        """Calculate total operating expenses from components or aggregate."""
        if self.operating_expenses is not None:
            return self.operating_expenses
        
        total = Decimal("0")
        expense_fields = [
            self.selling_general_admin,
            self.marketing_expenses,
            self.research_development,
            self.depreciation_amortization,
            self.other_operating_expenses,
        ]
        for expense in expense_fields:
            if expense is not None:
                total += expense
        return total
    
    @property
    def operating_income(self) -> Decimal:
        """Calculate operating income (EBIT approximation)."""
        return self.gross_profit - self.total_operating_expenses
    
    @property
    def ebitda(self) -> Decimal:
        """Calculate EBITDA."""
        da = self.depreciation_amortization or Decimal("0")
        return self.operating_income + da
    
    @property
    def ebit(self) -> Decimal:
        """Calculate EBIT (same as operating_income)."""
        return self.operating_income
    
    @property
    def earnings_before_tax(self) -> Decimal:
        """Calculate earnings before tax."""
        return (
            self.operating_income
            + self.interest_income
            - self.interest_expense
            + self.other_income
            - self.other_expenses
        )
    
    @property
    def calculated_net_income(self) -> Decimal:
        """Calculate net income from components if not provided."""
        if self.net_income is not None:
            return self.net_income
        return self.earnings_before_tax - self.income_tax_expense
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with calculated fields included."""
        data = self.model_dump(by_alias=False)
        data["gross_profit"] = self.gross_profit
        data["operating_income"] = self.operating_income
        data["ebitda"] = self.ebitda
        data["ebit"] = self.ebit
        data["earnings_before_tax"] = self.earnings_before_tax
        data["calculated_net_income"] = self.calculated_net_income
        return to_jsonable(data)


class BalanceSheetData(BaseModel):
    """
    Balance Sheet data model.
    
    Organized into Current Assets, Non-Current Assets, Current Liabilities,
    Non-Current Liabilities, and Shareholders' Equity sections.
    """
    
    period: FinancialPeriod
    currency: str = Field(default="SGD", min_length=3, max_length=3)
    
    # ─────────────────────────────────────────────────────────────────────
    # Current Assets
    # ─────────────────────────────────────────────────────────────────────
    cash_and_equivalents: MonetaryValue = Field(
        ...,
        alias="cash",
        description="Cash and cash equivalents"
    )
    short_term_investments: MonetaryValue = Field(default=Decimal("0"))
    accounts_receivable: MonetaryValue = Field(default=Decimal("0"), alias="ar")
    inventory: MonetaryValue = Field(default=Decimal("0"))
    prepaid_expenses: MonetaryValue = Field(default=Decimal("0"))
    other_current_assets: MonetaryValue = Field(default=Decimal("0"))
    total_current_assets: MonetaryValue | None = Field(default=None)
    
    # ─────────────────────────────────────────────────────────────────────
    # Non-Current Assets
    # ─────────────────────────────────────────────────────────────────────
    property_plant_equipment: MonetaryValue = Field(default=Decimal("0"), alias="ppe")
    intangible_assets: MonetaryValue = Field(default=Decimal("0"))
    goodwill: MonetaryValue = Field(default=Decimal("0"))
    long_term_investments: MonetaryValue = Field(default=Decimal("0"))
    deferred_tax_assets: MonetaryValue = Field(default=Decimal("0"))
    other_non_current_assets: MonetaryValue = Field(default=Decimal("0"))
    total_non_current_assets: MonetaryValue | None = Field(default=None)
    
    total_assets: MonetaryValue | None = Field(default=None)
    
    # ─────────────────────────────────────────────────────────────────────
    # Current Liabilities
    # ─────────────────────────────────────────────────────────────────────
    accounts_payable: MonetaryValue = Field(default=Decimal("0"), alias="ap")
    short_term_debt: MonetaryValue = Field(default=Decimal("0"))
    accrued_liabilities: MonetaryValue = Field(default=Decimal("0"))
    deferred_revenue: MonetaryValue = Field(default=Decimal("0"))
    income_taxes_payable: MonetaryValue = Field(default=Decimal("0"))
    other_current_liabilities: MonetaryValue = Field(default=Decimal("0"))
    total_current_liabilities: MonetaryValue | None = Field(default=None)
    
    # ─────────────────────────────────────────────────────────────────────
    # Non-Current Liabilities
    # ─────────────────────────────────────────────────────────────────────
    long_term_debt: MonetaryValue = Field(default=Decimal("0"))
    deferred_tax_liabilities: MonetaryValue = Field(default=Decimal("0"))
    pension_liabilities: MonetaryValue = Field(default=Decimal("0"))
    other_non_current_liabilities: MonetaryValue = Field(default=Decimal("0"))
    total_non_current_liabilities: MonetaryValue | None = Field(default=None)
    
    total_liabilities: MonetaryValue | None = Field(default=None)
    
    # ─────────────────────────────────────────────────────────────────────
    # Shareholders' Equity
    # ─────────────────────────────────────────────────────────────────────
    common_stock: MonetaryValue = Field(default=Decimal("0"))
    preferred_stock: MonetaryValue = Field(default=Decimal("0"))
    additional_paid_in_capital: MonetaryValue = Field(default=Decimal("0"), alias="apic")
    retained_earnings: MonetaryValue = Field(default=Decimal("0"))
    treasury_stock: MonetaryValue = Field(default=Decimal("0"))
    accumulated_other_comprehensive_income: MonetaryValue = Field(default=Decimal("0"), alias="aoci")
    total_shareholders_equity: MonetaryValue | None = Field(default=None)
    non_controlling_interest: MonetaryValue = Field(default=Decimal("0"))
    
    model_config = {
        "populate_by_name": True,
        "extra": "allow",
    }
    
    @field_validator("cash_and_equivalents", mode="before")
    @classmethod
    def convert_to_decimal(cls, v: Any) -> Decimal:
        """Convert numeric inputs to Decimal."""
        if v is None:
            raise ValueError("Cash and equivalents is required")
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))
    
    @property
    def calculated_current_assets(self) -> Decimal:
        """Calculate total current assets from components."""
        if self.total_current_assets is not None:
            return self.total_current_assets
        return (
            self.cash_and_equivalents
            + self.short_term_investments
            + self.accounts_receivable
            + self.inventory
            + self.prepaid_expenses
            + self.other_current_assets
        )
    
    @property
    def calculated_non_current_assets(self) -> Decimal:
        """Calculate total non-current assets from components."""
        if self.total_non_current_assets is not None:
            return self.total_non_current_assets
        return (
            self.property_plant_equipment
            + self.intangible_assets
            + self.goodwill
            + self.long_term_investments
            + self.deferred_tax_assets
            + self.other_non_current_assets
        )
    
    @property
    def calculated_total_assets(self) -> Decimal:
        """Calculate total assets from components."""
        if self.total_assets is not None:
            return self.total_assets
        return self.calculated_current_assets + self.calculated_non_current_assets
    
    @property
    def calculated_current_liabilities(self) -> Decimal:
        """Calculate total current liabilities from components."""
        if self.total_current_liabilities is not None:
            return self.total_current_liabilities
        return (
            self.accounts_payable
            + self.short_term_debt
            + self.accrued_liabilities
            + self.deferred_revenue
            + self.income_taxes_payable
            + self.other_current_liabilities
        )
    
    @property
    def calculated_non_current_liabilities(self) -> Decimal:
        """Calculate total non-current liabilities from components."""
        if self.total_non_current_liabilities is not None:
            return self.total_non_current_liabilities
        return (
            self.long_term_debt
            + self.deferred_tax_liabilities
            + self.pension_liabilities
            + self.other_non_current_liabilities
        )
    
    @property
    def calculated_total_liabilities(self) -> Decimal:
        """Calculate total liabilities from components."""
        if self.total_liabilities is not None:
            return self.total_liabilities
        return self.calculated_current_liabilities + self.calculated_non_current_liabilities
    
    @property
    def calculated_shareholders_equity(self) -> Decimal:
        """Calculate shareholders' equity from components."""
        if self.total_shareholders_equity is not None:
            return self.total_shareholders_equity
        return (
            self.common_stock
            + self.preferred_stock
            + self.additional_paid_in_capital
            + self.retained_earnings
            - self.treasury_stock
            + self.accumulated_other_comprehensive_income
        )
    
    @property
    def calculated_total_equity(self) -> Decimal:
        """Calculate total equity including non-controlling interest."""
        return self.calculated_shareholders_equity + self.non_controlling_interest
    
    @property
    def working_capital(self) -> Decimal:
        """Calculate working capital."""
        return self.calculated_current_assets - self.calculated_current_liabilities
    
    @property
    def total_debt(self) -> Decimal:
        """Calculate total debt (short-term + long-term)."""
        return self.short_term_debt + self.long_term_debt
    
    def check_balance_sheet_equation(self, tolerance: Decimal = Decimal("0.01")) -> bool:
        """Verify Assets = Liabilities + Equity."""
        assets = self.calculated_total_assets
        liab_equity = self.calculated_total_liabilities + self.calculated_total_equity
        difference = abs(assets - liab_equity)
        return difference <= tolerance
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with calculated fields included."""
        data = self.model_dump(by_alias=False)
        data["calculated_current_assets"] = self.calculated_current_assets
        data["calculated_non_current_assets"] = self.calculated_non_current_assets
        data["calculated_total_assets"] = self.calculated_total_assets
        data["calculated_current_liabilities"] = self.calculated_current_liabilities
        data["calculated_non_current_liabilities"] = self.calculated_non_current_liabilities
        data["calculated_total_liabilities"] = self.calculated_total_liabilities
        data["calculated_shareholders_equity"] = self.calculated_shareholders_equity
        data["calculated_total_equity"] = self.calculated_total_equity
        data["working_capital"] = self.working_capital
        data["total_debt"] = self.total_debt
        return to_jsonable(data)


class CashFlowStatementData(BaseModel):
    """
    Cash Flow Statement data model.
    
    Organized into Operating, Investing, and Financing activities.
    """
    
    period: FinancialPeriod
    currency: str = Field(default="SGD", min_length=3, max_length=3)
    
    # ─────────────────────────────────────────────────────────────────────
    # Operating Activities
    # ─────────────────────────────────────────────────────────────────────
    net_income: MonetaryValue = Field(..., description="Net income (starting point)")
    depreciation_amortization: MonetaryValue = Field(default=Decimal("0"))
    stock_based_compensation: MonetaryValue = Field(default=Decimal("0"))
    deferred_taxes: MonetaryValue = Field(default=Decimal("0"))
    
    # Working capital changes
    change_in_receivables: MonetaryValue = Field(default=Decimal("0"))
    change_in_inventory: MonetaryValue = Field(default=Decimal("0"))
    change_in_payables: MonetaryValue = Field(default=Decimal("0"))
    change_in_other_working_capital: MonetaryValue = Field(default=Decimal("0"))
    other_operating_activities: MonetaryValue = Field(default=Decimal("0"))
    net_cash_from_operating: MonetaryValue | None = Field(default=None, alias="cfo")
    
    # ─────────────────────────────────────────────────────────────────────
    # Investing Activities
    # ─────────────────────────────────────────────────────────────────────
    capital_expenditures: MonetaryValue = Field(default=Decimal("0"), alias="capex")
    acquisitions: MonetaryValue = Field(default=Decimal("0"))
    investment_purchases: MonetaryValue = Field(default=Decimal("0"))
    investment_sales: MonetaryValue = Field(default=Decimal("0"))
    other_investing_activities: MonetaryValue = Field(default=Decimal("0"))
    net_cash_from_investing: MonetaryValue | None = Field(default=None, alias="cfi")
    
    # ─────────────────────────────────────────────────────────────────────
    # Financing Activities
    # ─────────────────────────────────────────────────────────────────────
    debt_issued: MonetaryValue = Field(default=Decimal("0"))
    debt_repaid: MonetaryValue = Field(default=Decimal("0"))
    shares_issued: MonetaryValue = Field(default=Decimal("0"))
    shares_repurchased: MonetaryValue = Field(default=Decimal("0"))
    dividends_paid: MonetaryValue = Field(default=Decimal("0"))
    other_financing_activities: MonetaryValue = Field(default=Decimal("0"))
    net_cash_from_financing: MonetaryValue | None = Field(default=None, alias="cff")
    
    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────
    beginning_cash: MonetaryValue | None = Field(default=None)
    ending_cash: MonetaryValue | None = Field(default=None)
    net_change_in_cash: MonetaryValue | None = Field(default=None)
    
    model_config = {
        "populate_by_name": True,
        "extra": "allow",
    }
    
    @field_validator("net_income", mode="before")
    @classmethod
    def convert_to_decimal(cls, v: Any) -> Decimal:
        """Convert numeric inputs to Decimal."""
        if v is None:
            raise ValueError("Net income is required")
        if isinstance(v, Decimal):
            return v
        return Decimal(str(v))
    
    @property
    def calculated_operating_cash_flow(self) -> Decimal:
        """Calculate operating cash flow from components."""
        if self.net_cash_from_operating is not None:
            return self.net_cash_from_operating
        return (
            self.net_income
            + self.depreciation_amortization
            + self.stock_based_compensation
            + self.deferred_taxes
            - self.change_in_receivables
            - self.change_in_inventory
            + self.change_in_payables
            + self.change_in_other_working_capital
            + self.other_operating_activities
        )
    
    @property
    def calculated_investing_cash_flow(self) -> Decimal:
        """Calculate investing cash flow from components."""
        if self.net_cash_from_investing is not None:
            return self.net_cash_from_investing
        return (
            -abs(self.capital_expenditures)  # CapEx is usually an outflow
            - abs(self.acquisitions)
            - self.investment_purchases
            + self.investment_sales
            + self.other_investing_activities
        )
    
    @property
    def calculated_financing_cash_flow(self) -> Decimal:
        """Calculate financing cash flow from components."""
        if self.net_cash_from_financing is not None:
            return self.net_cash_from_financing
        return (
            self.debt_issued
            - self.debt_repaid
            + self.shares_issued
            - self.shares_repurchased
            - abs(self.dividends_paid)
            + self.other_financing_activities
        )
    
    @property
    def calculated_net_change(self) -> Decimal:
        """Calculate net change in cash."""
        if self.net_change_in_cash is not None:
            return self.net_change_in_cash
        return (
            self.calculated_operating_cash_flow
            + self.calculated_investing_cash_flow
            + self.calculated_financing_cash_flow
        )
    
    @property
    def free_cash_flow(self) -> Decimal:
        """Calculate free cash flow (CFO - CapEx)."""
        return self.calculated_operating_cash_flow - abs(self.capital_expenditures)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with calculated fields included."""
        data = self.model_dump(by_alias=False)
        data["calculated_operating_cash_flow"] = self.calculated_operating_cash_flow
        data["calculated_investing_cash_flow"] = self.calculated_investing_cash_flow
        data["calculated_financing_cash_flow"] = self.calculated_financing_cash_flow
        data["calculated_net_change"] = self.calculated_net_change
        data["free_cash_flow"] = self.free_cash_flow
        return to_jsonable(data)


class FinancialStatementSet(BaseModel):
    """
    A complete set of financial statements for a single period.
    
    Combines Income Statement, Balance Sheet, and Cash Flow Statement.
    """
    
    income_statement: IncomeStatementData
    balance_sheet: BalanceSheetData
    cash_flow_statement: CashFlowStatementData | None = None
    
    @model_validator(mode="after")
    def validate_period_consistency(self) -> "FinancialStatementSet":
        """Ensure all statements are for the same period."""
        is_period = self.income_statement.period
        bs_period = self.balance_sheet.period
        
        if is_period != bs_period:
            raise ValueError(
                f"Period mismatch: Income Statement is for {is_period}, "
                f"Balance Sheet is for {bs_period}"
            )
        
        if self.cash_flow_statement:
            cf_period = self.cash_flow_statement.period
            if is_period != cf_period:
                raise ValueError(
                    f"Period mismatch: Income Statement is for {is_period}, "
                    f"Cash Flow is for {cf_period}"
                )
        
        return self
    
    @property
    def period(self) -> FinancialPeriod:
        """Get the period for this statement set."""
        return self.income_statement.period
    
    @property
    def currency(self) -> str:
        """Get the currency for this statement set."""
        return self.income_statement.currency
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "period": str(self.period),
            "currency": self.currency,
            "income_statement": self.income_statement.to_dict(),
            "balance_sheet": self.balance_sheet.to_dict(),
        }
        if self.cash_flow_statement:
            result["cash_flow_statement"] = self.cash_flow_statement.to_dict()
        return result


class MultiPeriodFinancialData(BaseModel):
    """
    Financial data spanning multiple periods for trend analysis.
    """
    
    periods: list[FinancialStatementSet] = Field(
        ...,
        min_length=1,
        description="List of financial statement sets, one per period"
    )
    
    @model_validator(mode="after")
    def sort_by_period(self) -> "MultiPeriodFinancialData":
        """Sort periods chronologically."""
        self.periods.sort(key=lambda x: x.period)
        return self
    
    @property
    def period_count(self) -> int:
        """Number of periods in the data."""
        return len(self.periods)
    
    @property
    def earliest_period(self) -> FinancialPeriod:
        """Get the earliest period."""
        return self.periods[0].period
    
    @property
    def latest_period(self) -> FinancialPeriod:
        """Get the latest period."""
        return self.periods[-1].period
    
    @property
    def currency(self) -> str:
        """Get the currency (from first period)."""
        return self.periods[0].currency
    
    def get_period(self, period: FinancialPeriod) -> FinancialStatementSet | None:
        """Get statement set for a specific period."""
        for stmt_set in self.periods:
            if stmt_set.period == period:
                return stmt_set
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period_count": self.period_count,
            "earliest_period": str(self.earliest_period),
            "latest_period": str(self.latest_period),
            "currency": self.currency,
            "periods": [p.to_dict() for p in self.periods],
        }
