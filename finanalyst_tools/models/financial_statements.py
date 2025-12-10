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
MonetaryValue = Annotated[
    Decimal, 
    Field(description="Monetary value in base currency")
]


class FinancialPeriod(BaseModel):
    """
    Represents a financial reporting period.
    
    Examples:
        - Annual: FinancialPeriod(year=2023, period_type=PeriodType.ANNUAL)
        - Quarterly: FinancialPeriod(year=2023, period_type=PeriodType.QUARTERLY, quarter=2)
    """
    
    year: int = Field(..., ge=1900, le=2100, description="Fiscal year")
    period_type: PeriodType = Field(default=PeriodType.ANNUAL)
    quarter: int | None = Field(
        default=None, 
        ge=1, 
        le=4, 
        description="Quarter number (1-4) if quarterly"
    )
    month: int | None = Field(
        default=None,
        ge=1,
        le=12,
        description="Month number (1-12) if monthly"
    )
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
        # Same year - compare by quarter/month
        self_sub = self.quarter or self.month or 0
        other_sub = other.quarter or other.month or 0
        return self_sub < other_sub


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
        description="Total revenue / net sales"
    )
    cost_of_goods_sold: MonetaryValue = Field(
        ...,
        description="Cost of goods sold / cost of sales"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Operating Expenses (flexible structure)
    # ─────────────────────────────────────────────────────────────────────
    operating_expenses: MonetaryValue | None = Field(
        default=None,
        description="Total operating expenses (if provided as aggregate)"
    )
    selling_general_admin: MonetaryValue | None = Field(
        default=None,
        description="Selling, General & Administrative expenses"
    )
    marketing_expenses: MonetaryValue | None = Field(
        default=None,
        description="Marketing and advertising expenses"
    )
    research_development: MonetaryValue | None = Field(
        default=None,
        description="Research & Development expenses"
    )
    depreciation_amortization: MonetaryValue | None = Field(
        default=None,
        description="Depreciation and amortization"
    )
    other_operating_expenses: MonetaryValue | None = Field(
        default=None,
        description="Other operating expenses"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Non-Operating Items
    # ─────────────────────────────────────────────────────────────────────
    interest_income: MonetaryValue = Field(
        default=Decimal("0"),
        description="Interest and investment income"
    )
    interest_expense: MonetaryValue = Field(
        default=Decimal("0"),
        description="Interest expense on debt"
    )
    other_income: MonetaryValue = Field(
        default=Decimal("0"),
        description="Other non-operating income"
    )
    other_expenses: MonetaryValue = Field(
        default=Decimal("0"),
        description="Other non-operating expenses"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Taxes and Bottom Line
    # ─────────────────────────────────────────────────────────────────────
    income_tax_expense: MonetaryValue = Field(
        default=Decimal("0"),
        description="Income tax expense"
    )
    net_income: MonetaryValue | None = Field(
        default=None,
        description="Net income / net profit"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Per Share Data (optional)
    # ─────────────────────────────────────────────────────────────────────
    earnings_per_share: Decimal | None = Field(
        default=None,
        description="Basic earnings per share"
    )
    diluted_eps: Decimal | None = Field(
        default=None,
        description="Diluted earnings per share"
    )
    shares_outstanding: int | None = Field(
        default=None,
        description="Weighted average shares outstanding"
    )
    
    model_config = {
        "populate_by_name": True,
        "extra": "allow",  # Allow additional fields
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
        """
        Calculate total operating expenses.
        
        Uses provided aggregate if available, otherwise sums components.
        """
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
        """Calculate EBITDA (operating income + D&A)."""
        da = self.depreciation_amortization or Decimal("0")
        return self.operating_income + da
    
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
        """
        Calculate net income from components.
        
        Returns provided net_income if available, otherwise calculates.
        """
        if self.net_income is not None:
            return self.net_income
        return self.earnings_before_tax - self.income_tax_expense
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with calculated fields included."""
        data = self.model_dump()
        data["gross_profit"] = float(self.gross_profit)
        data["operating_income"] = float(self.operating_income)
        data["ebitda"] = float(self.ebitda)
        data["calculated_net_income"] = float(self.calculated_net_income)
        return data


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
        description="Cash and cash equivalents"
    )
    short_term_investments: MonetaryValue = Field(
        default=Decimal("0"),
        description="Short-term investments"
    )
    accounts_receivable: MonetaryValue = Field(
        default=Decimal("0"),
        description="Accounts receivable, net"
    )
    inventory: MonetaryValue = Field(
        default=Decimal("0"),
        description="Inventories"
    )
    prepaid_expenses: MonetaryValue = Field(
        default=Decimal("0"),
        description="Prepaid expenses and other current assets"
    )
    other_current_assets: MonetaryValue = Field(
        default=Decimal("0"),
        description="Other current assets"
    )
    total_current_assets: MonetaryValue | None = Field(
        default=None,
        description="Total current assets (if provided)"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Non-Current Assets
    # ─────────────────────────────────────────────────────────────────────
    property_plant_equipment: MonetaryValue = Field(
        default=Decimal("0"),
        description="Property, plant and equipment, net"
    )
    intangible_assets: MonetaryValue = Field(
        default=Decimal("0"),
        description="Intangible assets (including goodwill)"
    )
    goodwill: MonetaryValue = Field(
        default=Decimal("0"),
        description="Goodwill (if separate from intangibles)"
    )
    long_term_investments: MonetaryValue = Field(
        default=Decimal("0"),
        description="Long-term investments"
    )
    deferred_tax_assets: MonetaryValue = Field(
        default=Decimal("0"),
        description="Deferred tax assets"
    )
    other_non_current_assets: MonetaryValue = Field(
        default=Decimal("0"),
        description="Other non-current assets"
    )
    total_non_current_assets: MonetaryValue | None = Field(
        default=None,
        description="Total non-current assets"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Total Assets
    # ─────────────────────────────────────────────────────────────────────
    total_assets: MonetaryValue | None = Field(
        default=None,
        description="Total assets"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Current Liabilities
    # ─────────────────────────────────────────────────────────────────────
    accounts_payable: MonetaryValue = Field(
        default=Decimal("0"),
        description="Accounts payable"
    )
    short_term_debt: MonetaryValue = Field(
        default=Decimal("0"),
        description="Short-term debt / current portion of long-term debt"
    )
    accrued_liabilities: MonetaryValue = Field(
        default=Decimal("0"),
        description="Accrued liabilities"
    )
    deferred_revenue: MonetaryValue = Field(
        default=Decimal("0"),
        description="Deferred revenue / unearned revenue"
    )
    income_taxes_payable: MonetaryValue = Field(
        default=Decimal("0"),
        description="Income taxes payable"
    )
    other_current_liabilities: MonetaryValue = Field(
        default=Decimal("0"),
        description="Other current liabilities"
    )
    total_current_liabilities: MonetaryValue | None = Field(
        default=None,
        description="Total current liabilities"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Non-Current Liabilities
    # ─────────────────────────────────────────────────────────────────────
    long_term_debt: MonetaryValue = Field(
        default=Decimal("0"),
        description="Long-term debt"
    )
    deferred_tax_liabilities: MonetaryValue = Field(
        default=Decimal("0"),
        description="Deferred tax liabilities"
    )
    pension_liabilities: MonetaryValue = Field(
        default=Decimal("0"),
        description="Pension and post-retirement obligations"
    )
    other_non_current_liabilities: MonetaryValue = Field(
        default=Decimal("0"),
        description="Other non-current liabilities"
    )
    total_non_current_liabilities: MonetaryValue | None = Field(
        default=None,
        description="Total non-current liabilities"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Total Liabilities
    # ─────────────────────────────────────────────────────────────────────
    total_liabilities: MonetaryValue | None = Field(
        default=None,
        description="Total liabilities"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Shareholders' Equity
    # ─────────────────────────────────────────────────────────────────────
    common_stock: MonetaryValue = Field(
        default=Decimal("0"),
        description="Common stock / share capital"
    )
    preferred_stock: MonetaryValue = Field(
        default=Decimal("0"),
        description="Preferred stock"
    )
    additional_paid_in_capital: MonetaryValue = Field(
        default=Decimal("0"),
        description="Additional paid-in capital"
    )
    retained_earnings: MonetaryValue = Field(
        default=Decimal("0"),
        description="Retained earnings"
    )
    treasury_stock: MonetaryValue = Field(
        default=Decimal("0"),
        description="Treasury stock (contra-equity)"
    )
    accumulated_other_comprehensive_income: MonetaryValue = Field(
        default=Decimal("0"),
        description="Accumulated other comprehensive income/loss"
    )
    total_shareholders_equity: MonetaryValue | None = Field(
        default=None,
        description="Total shareholders' equity"
    )
    
    # Non-controlling interest (for consolidated statements)
    non_controlling_interest: MonetaryValue = Field(
        default=Decimal("0"),
        description="Non-controlling / minority interest"
    )
    
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
        """
        Verify Assets = Liabilities + Equity.
        
        Args:
            tolerance: Acceptable difference (default 0.01 = 1 cent)
            
        Returns:
            True if equation balances within tolerance
        """
        assets = self.calculated_total_assets
        liab_equity = self.calculated_total_liabilities + self.calculated_total_equity
        difference = abs(assets - liab_equity)
        return difference <= tolerance


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
    net_income: MonetaryValue = Field(
        ...,
        description="Net income (starting point for indirect method)"
    )
    depreciation_amortization: MonetaryValue = Field(
        default=Decimal("0"),
        description="Depreciation and amortization"
    )
    stock_based_compensation: MonetaryValue = Field(
        default=Decimal("0"),
        description="Stock-based compensation expense"
    )
    deferred_taxes: MonetaryValue = Field(
        default=Decimal("0"),
        description="Deferred income taxes"
    )
    
    # Working capital changes
    change_in_receivables: MonetaryValue = Field(
        default=Decimal("0"),
        description="Change in accounts receivable"
    )
    change_in_inventory: MonetaryValue = Field(
        default=Decimal("0"),
        description="Change in inventories"
    )
    change_in_payables: MonetaryValue = Field(
        default=Decimal("0"),
        description="Change in accounts payable"
    )
    change_in_other_working_capital: MonetaryValue = Field(
        default=Decimal("0"),
        description="Changes in other working capital items"
    )
    other_operating_activities: MonetaryValue = Field(
        default=Decimal("0"),
        description="Other operating activities"
    )
    net_cash_from_operating: MonetaryValue | None = Field(
        default=None,
        description="Net cash from operating activities"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Investing Activities
    # ─────────────────────────────────────────────────────────────────────
    capital_expenditures: MonetaryValue = Field(
        default=Decimal("0"),
        description="Capital expenditures (purchases of PP&E)"
    )
    acquisitions: MonetaryValue = Field(
        default=Decimal("0"),
        description="Acquisitions, net of cash acquired"
    )
    investment_purchases: MonetaryValue = Field(
        default=Decimal("0"),
        description="Purchases of investments"
    )
    investment_sales: MonetaryValue = Field(
        default=Decimal("0"),
        description="Sales/maturities of investments"
    )
    other_investing_activities: MonetaryValue = Field(
        default=Decimal("0"),
        description="Other investing activities"
    )
    net_cash_from_investing: MonetaryValue | None = Field(
        default=None,
        description="Net cash from investing activities"
    )
    
    # ─────────────────────────────────────────────────────────────────────
    # Financing Activities
    # ─────────────────────────────────────────────────────────────────────
    debt_issued: MonetaryValue = Field(
        default=Decimal("0"),
        description="Proceeds from debt issuance"
    
