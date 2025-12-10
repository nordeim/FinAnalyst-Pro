# finanalyst_tools/models/financial_statements.py
"""
Pydantic models for financial statement data structures.
Provides validation, serialization, and type safety.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator, model_validator


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


# Type aliases for clarity
MonetaryValue = Annotated[Decimal, Field(description="Monetary value in base currency")]
Percentage = Annotated[Decimal, Field(ge=-1000, le=1000, description="Percentage value")]
Ratio = Annotated[Decimal, Field(description="Financial ratio")]


class FinancialPeriod(BaseModel):
    """Represents a financial reporting period."""
    
    year: int = Field(..., ge=1900, le=2100, description="Fiscal year")
    period_type: PeriodType = Field(default=PeriodType.ANNUAL)
    quarter: int | None = Field(default=None, ge=1, le=4, description="Quarter number if quarterly")
    start_date: date | None = Field(default=None, description="Period start date")
    end_date: date | None = Field(default=None, description="Period end date")
    
    @model_validator(mode="after")
    def validate_quarter(self) -> "FinancialPeriod":
        if self.period_type == PeriodType.QUARTERLY and self.quarter is None:
            raise ValueError("Quarter must be specified for quarterly periods")
        return self
    
    def __str__(self) -> str:
        if self.period_type == PeriodType.QUARTERLY:
            return f"Q{self.quarter} {self.year}"
        return str(self.year)


class IncomeStatementData(BaseModel):
    """Income Statement / Profit & Loss data model."""
    
    period: FinancialPeriod
    currency: str = Field(default="SGD", min_length=3, max_length=3)
    
    # Revenue
    total_revenue: MonetaryValue = Field(..., alias="total_net_sales")
    cost_of_goods_sold: MonetaryValue = Field(..., alias="cost_of_sales")
    
    # Operating expenses (can be broken down or aggregated)
    operating_expenses: MonetaryValue | None = None
    marketing_expenses: MonetaryValue | None = None
    research_development: MonetaryValue | None = None
    general_administrative: MonetaryValue | None = None
    depreciation_amortization: MonetaryValue | None = None
    
    # Other income/expenses
    interest_income: MonetaryValue = Field(default=Decimal("0"))
    interest_expense: MonetaryValue = Field(default=Decimal("0"))
    other_income: MonetaryValue = Field(default=Decimal("0"))
    other_expenses: MonetaryValue = Field(default=Decimal("0"))
    
    # Taxes and bottom line
    income_tax_expense: MonetaryValue = Field(default=Decimal("0"), alias="taxes")
    net_income: MonetaryValue | None = None
    
    # Per share data
    earnings_per_share: Decimal | None = None
    diluted_eps: Decimal | None = None
    shares_outstanding: int | None = None
    
    model_config = {"populate_by_name": True}
    
    @property
    def gross_profit(self) -> Decimal:
        """Calculate gross profit."""
        return self.total_revenue - self.cost_of_goods_sold
    
    @property
    def total_operating_expenses(self) -> Decimal:
        """Calculate total operating expenses from components or use provided value."""
        if self.operating_expenses is not None:
            base = self.operating_expenses
        else:
            base = Decimal("0")
        
        # Add any separately listed expenses
        if self.marketing_expenses:
            base += self.marketing_expenses
        if self.research_development:
            base += self.research_development
        if self.general_administrative:
            base += self.general_administrative
            
        return base
    
    @property
    def operating_income(self) -> Decimal:
        """Calculate operating income (EBIT approximation)."""
        return self.gross_profit - self.total_operating_expenses
    
    @property
    def calculated_net_income(self) -> Decimal:
        """Calculate net income if not provided."""
        if self.net_income is not None:
            return self.net_income
        
        ebt = (
            self.operating_income 
            + self.interest_income 
            - self.interest_expense
            + self.other_income
            - self.other_expenses
        )
        return ebt - self.income_tax_expense


class BalanceSheetData(BaseModel):
    """Balance Sheet data model."""
    
    period: FinancialPeriod
    currency: str = Field(default="SGD", min_length=3, max_length=3)
    
    # Current Assets
    cash_and_equivalents: MonetaryValue = Field(..., alias="cash")
    accounts_receivable: MonetaryValue = Field(default=Decimal("0"))
    inventory: MonetaryValue = Field(default=Decimal("0"), alias="inventories")
    prepaid_expenses: MonetaryValue = Field(default=Decimal("0"))
    other_current_assets: MonetaryValue = Field(default=Decimal("0"))
    total_current_assets: MonetaryValue | None = Field(default=None, alias="current_assets")
    
    # Non-Current Assets
    property_plant_equipment: MonetaryValue = Field(default=Decimal("0"))
    intangible_assets: MonetaryValue = Field(default=Decimal("0"))
    long_term_investments: MonetaryValue = Field(default=Decimal("0"))
    other_non_current_assets: MonetaryValue = Field(default=Decimal("0"))
    total_non_current_assets: MonetaryValue | None = None
    
    # Total Assets
    total_assets: MonetaryValue | None = None
    
    # Current Liabilities
    accounts_payable: MonetaryValue = Field(default=Decimal("0"))
    short_term_debt: MonetaryValue = Field(default=Decimal("0"))
    accrued_liabilities: MonetaryValue = Field(default=Decimal("0"))
    deferred_revenue: MonetaryValue = Field(default=Decimal("0"))
    other_current_liabilities: MonetaryValue = Field(default=Decimal("0"))
    total_current_liabilities: MonetaryValue | None = Field(default=None, alias="current_liabilities")
    
    # Non-Current Liabilities
    long_term_debt: MonetaryValue = Field(default=Decimal("0"))
    deferred_tax_liabilities: MonetaryValue = Field(default=Decimal("0"))
    other_non_current_liabilities: MonetaryValue = Field(default=Decimal("0"))
    total_non_current_liabilities: MonetaryValue | None = None
    
    # Total Liabilities
    total_liabilities: MonetaryValue | None = None
    
    # Shareholders' Equity
    common_stock: MonetaryValue = Field(default=Decimal("0"))
    retained_earnings: MonetaryValue = Field(default=Decimal("0"))
    additional_paid_in_capital: MonetaryValue = Field(default=Decimal("0"))
    treasury_stock: MonetaryValue = Field(default=Decimal("0"))
    total_shareholders_equity: MonetaryValue | None = Field(default=None, alias="shareholder_equity")
    
    model_config = {"populate_by_name": True}
    
    @property
    def calculated_current_assets(self) -> Decimal:
        """Calculate total current assets if not provided."""
        if self.total_current_assets is not None:
            return self.total_current_assets
        return (
            self.cash_and_equivalents
            + self.accounts_receivable
            + self.inventory
            + self.prepaid_expenses
            + self.other_current_assets
        )
    
    @property
    def calculated_total_assets(self) -> Decimal:
        """Calculate total assets if not provided."""
        if self.total_assets is not None:
            return self.total_assets
        
        non_current = self.total_non_current_assets or (
            self.property_plant_equipment
            + self.intangible_assets
            + self.long_term_investments
            + self.other_non_current_assets
        )
        return self.calculated_current_assets + non_current
    
    @property
    def calculated_current_liabilities(self) -> Decimal:
        """Calculate total current liabilities if not provided."""
        if self.total_current_liabilities is not None:
            return self.total_current_liabilities
        return (
            self.accounts_payable
            + self.short_term_debt
            + self.accrued_liabilities
            + self.deferred_revenue
            + self.other_current_liabilities
        )
    
    @property
    def calculated_total_liabilities(self) -> Decimal:
        """Calculate total liabilities if not provided."""
        if self.total_liabilities is not None:
            return self.total_liabilities
        
        non_current = self.total_non_current_liabilities or (
            self.long_term_debt
            + self.deferred_tax_liabilities
            + self.other_non_current_liabilities
        )
        return self.calculated_current_liabilities + non_current
    
    @property
    def calculated_equity(self) -> Decimal:
        """Calculate shareholders' equity if not provided."""
        if self.total_shareholders_equity is not None:
            return self.total_shareholders_equity
        return (
            self.common_stock
            + self.retained_earnings
            + self.additional_paid_in_capital
            - self.treasury_stock
        )


class CashFlowStatementData(BaseModel):
    """Cash Flow Statement data model."""
    
    period: FinancialPeriod
    currency: str = Field(default="SGD", min_length=3, max_length=3)
    
    # Operating Activities
    net_income: MonetaryValue = Field(...)
    depreciation_amortization: MonetaryValue = Field(default=Decimal("0"))
    stock_based_compensation: MonetaryValue = Field(default=Decimal("0"))
    changes_in_working_capital: MonetaryValue = Field(default=Decimal("0"))
    other_operating_activities: MonetaryValue = Field(default=Decimal("0"))
    net_cash_from_operating: MonetaryValue | None = None
    
    # Investing Activities
    capital_expenditures: MonetaryValue = Field(default=Decimal("0"), alias="purchase_of_equipment")
    acquisitions: MonetaryValue = Field(default=Decimal("0"))
    investment_purchases: MonetaryValue = Field(default=Decimal("0"))
    investment_sales: MonetaryValue = Field(default=Decimal("0"))
    other_investing_activities: MonetaryValue = Field(default=Decimal("0"))
    net_cash_from_investing: MonetaryValue | None = None
    
    # Financing Activities
    debt_issued: MonetaryValue = Field(default=Decimal("0"))
    debt_repaid: MonetaryValue = Field(default=Decimal("0"))
    dividends_paid: MonetaryValue = Field(default=Decimal("0"))
    stock_issued: MonetaryValue = Field(default=Decimal("0"))
    stock_repurchased: MonetaryValue = Field(default=Decimal("0"))
    other_financing_activities: MonetaryValue = Field(default=Decimal("0"))
    net_cash_from_financing: MonetaryValue | None = None
    
    # Net change
    net_change_in_cash: MonetaryValue | None = Field(default=None, alias="net_cash")
    beginning_cash: MonetaryValue | None = None
    ending_cash: MonetaryValue | None = None
    
    model_config = {"populate_by_name": True}
    
    @property
    def calculated_operating_cash_flow(self) -> Decimal:
        """Calculate operating cash flow if not provided."""
        if self.net_cash_from_operating is not None:
            return self.net_cash_from_operating
        return (
            self.net_income
            + self.depreciation_amortization
            + self.stock_based_compensation
            + self.changes_in_working_capital
            + self.other_operating_activities
        )


class FinancialStatementSet(BaseModel):
    """Complete set of financial statements for a period."""
    
    company_name: str | None = None
    period: FinancialPeriod
    income_statement: IncomeStatementData
    balance_sheet: BalanceSheetData
    cash_flow_statement: CashFlowStatementData | None = None
    
    @model_validator(mode="after")
    def validate_periods_match(self) -> "FinancialStatementSet":
        """Ensure all statements are for the same period."""
        periods = [self.income_statement.period, self.balance_sheet.period]
        if self.cash_flow_statement:
            periods.append(self.cash_flow_statement.period)
        
        if not all(p.year == self.period.year for p in periods):
            raise ValueError("All statements must be for the same period")
        return self


class MultiPeriodFinancialData(BaseModel):
    """Financial data across multiple periods for trend analysis."""
    
    company_name: str | None = None
    currency: str = Field(default="SGD")
    periods: list[FinancialStatementSet] = Field(..., min_length=1)
    
    @field_validator("periods")
    @classmethod
    def sort_periods(cls, v: list[FinancialStatementSet]) -> list[FinancialStatementSet]:
        """Sort periods chronologically."""
        return sorted(v, key=lambda x: x.period.year)
    
    @property
    def years(self) -> list[int]:
        """Get list of years in the dataset."""
        return [p.period.year for p in self.periods]
    
    def get_period(self, year: int) -> FinancialStatementSet | None:
        """Get financial statements for a specific year."""
        for p in self.periods:
            if p.period.year == year:
                return p
        return None
