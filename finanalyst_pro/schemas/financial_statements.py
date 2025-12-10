# ============================================================================
# FILE: finanalyst_pro/schemas/financial_statements.py
# PURPOSE: Canonical data models for financial statements
# ============================================================================

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Period(BaseModel):
    """Financial reporting period."""
    
    start_date: date
    end_date: date
    fiscal_year: int
    quarter: Literal[1, 2, 3, 4] | None = None
    period_type: Literal["annual", "quarterly", "monthly", "ytd"] = "annual"
    
    @property
    def label(self) -> str:
        """Human-readable period label."""
        if self.period_type == "annual":
            return f"FY{self.fiscal_year}"
        elif self.quarter:
            return f"Q{self.quarter} FY{self.fiscal_year}"
        return f"{self.start_date} to {self.end_date}"


class IncomeStatementLine(BaseModel):
    """Single line item from income statement."""
    
    period: Period
    
    # Revenue
    total_revenue: Decimal | None = Field(None, alias="total_net_sales")
    cost_of_goods_sold: Decimal | None = Field(None, alias="cost_of_sales")
    
    # Operating expenses (detailed)
    selling_general_admin: Decimal | None = None
    marketing_expense: Decimal | None = None
    research_development: Decimal | None = None
    depreciation_amortization: Decimal | None = None
    other_operating_expense: Decimal | None = None
    total_operating_expenses: Decimal | None = None
    
    # Other income/expense
    interest_income: Decimal | None = None
    interest_expense: Decimal | None = None
    other_income: Decimal | None = None
    other_expense: Decimal | None = None
    
    # Tax and profit
    income_tax_expense: Decimal | None = None
    net_income: Decimal | None = None
    
    # Per share data
    earnings_per_share_basic: Decimal | None = None
    earnings_per_share_diluted: Decimal | None = None
    shares_outstanding: Decimal | None = None
    
    @field_validator("*", mode="before")
    @classmethod
    def convert_to_decimal(cls, v):
        """Convert numeric strings to Decimal."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        if isinstance(v, str):
            # Remove currency symbols and commas
            cleaned = v.replace("$", "").replace(",", "").replace(" ", "")
            if cleaned.startswith("(") and cleaned.endswith(")"):
                cleaned = "-" + cleaned[1:-1]
            try:
                return Decimal(cleaned)
            except Exception:
                return None
        return v


class BalanceSheetLine(BaseModel):
    """Single line item from balance sheet."""
    
    period: Period
    
    # Current Assets
    cash_and_equivalents: Decimal | None = None
    short_term_investments: Decimal | None = None
    accounts_receivable: Decimal | None = None
    inventory: Decimal | None = None
    prepaid_expenses: Decimal | None = None
    other_current_assets: Decimal | None = None
    total_current_assets: Decimal | None = None
    
    # Non-current Assets
    property_plant_equipment: Decimal | None = None
    accumulated_depreciation: Decimal | None = None
    net_ppe: Decimal | None = None
    intangible_assets: Decimal | None = None
    goodwill: Decimal | None = None
    long_term_investments: Decimal | None = None
    other_non_current_assets: Decimal | None = None
    total_non_current_assets: Decimal | None = None
    
    # Total Assets
    total_assets: Decimal | None = None
    
    # Current Liabilities
    accounts_payable: Decimal | None = None
    short_term_debt: Decimal | None = None
    current_portion_long_term_debt: Decimal | None = None
    accrued_expenses: Decimal | None = None
    deferred_revenue: Decimal | None = None
    other_current_liabilities: Decimal | None = None
    total_current_liabilities: Decimal | None = None
    
    # Non-current Liabilities
    long_term_debt: Decimal | None = None
    deferred_tax_liabilities: Decimal | None = None
    other_non_current_liabilities: Decimal | None = None
    total_non_current_liabilities: Decimal | None = None
    
    # Total Liabilities
    total_liabilities: Decimal | None = None
    
    # Shareholders' Equity
    common_stock: Decimal | None = None
    retained_earnings: Decimal | None = None
    additional_paid_in_capital: Decimal | None = None
    treasury_stock: Decimal | None = None
    accumulated_other_comprehensive_income: Decimal | None = None
    total_shareholders_equity: Decimal | None = None
    
    @field_validator("*", mode="before")
    @classmethod
    def convert_to_decimal(cls, v):
        """Convert numeric strings to Decimal."""
        if v is None or isinstance(v, Period):
            return v
        if isinstance(v, (int, float)):
            return Decimal(str(v))
        if isinstance(v, str):
            cleaned = v.replace("$", "").replace(",", "").replace(" ", "")
            if cleaned.startswith("(") and cleaned.endswith(")"):
                cleaned = "-" + cleaned[1:-1]
            try:
                return Decimal(cleaned)
            except Exception:
                return None
        return v


class CashFlowLine(BaseModel):
    """Single line item from cash flow statement."""
    
    period: Period
    
    # Operating Activities
    net_income: Decimal | None = None
    depreciation_amortization: Decimal | None = None
    stock_based_compensation: Decimal | None = None
    deferred_taxes: Decimal | None = None
    changes_in_working_capital: Decimal | None = None
    change_in_receivables: Decimal | None = None
    change_in_inventory: Decimal | None = None
    change_in_payables: Decimal | None = None
    other_operating_activities: Decimal | None = None
    net_cash_from_operating: Decimal | None = None
    
    # Investing Activities
    capital_expenditures: Decimal | None = None
    acquisitions: Decimal | None = None
    purchases_of_investments: Decimal | None = None
    sales_of_investments: Decimal | None = None
    other_investing_activities: Decimal | None = None
    net_cash_from_investing: Decimal | None = None
    
    # Financing Activities
    debt_issued: Decimal | None = None
    debt_repaid: Decimal | None = None
    equity_issued: Decimal | None = None
    dividends_paid: Decimal | None = None
    share_repurchases: Decimal | None = None
    other_financing_activities: Decimal | None = None
    net_cash_from_financing: Decimal | None = None
    
    # Net Change
    net_change_in_cash: Decimal | None = None
    beginning_cash: Decimal | None = None
    ending_cash: Decimal | None = None


class FinancialStatements(BaseModel):
    """Complete set of financial statements for analysis."""
    
    company_name: str | None = None
    currency: str = "SGD"
    reporting_standard: str | None = None  # e.g., "SFRS", "IFRS", "GAAP"
    
    income_statements: list[IncomeStatementLine] = Field(default_factory=list)
    balance_sheets: list[BalanceSheetLine] = Field(default_factory=list)
    cash_flow_statements: list[CashFlowLine] = Field(default_factory=list)
    
    # Metadata
    source_file: str | None = None
    extraction_timestamp: str | None = None
    extraction_confidence: float | None = None
    notes: list[str] = Field(default_factory=list)
    
    def get_periods(self) -> list[Period]:
        """Get all unique periods across statements."""
        periods = set()
        for stmt in self.income_statements:
            periods.add((stmt.period.fiscal_year, stmt.period.quarter))
        for stmt in self.balance_sheets:
            periods.add((stmt.period.fiscal_year, stmt.period.quarter))
        for stmt in self.cash_flow_statements:
            periods.add((stmt.period.fiscal_year, stmt.period.quarter))
        return sorted(periods, key=lambda x: (x[0], x[1] or 0))
    
    def get_income_statement(self, fiscal_year: int) -> IncomeStatementLine | None:
        """Get income statement for a specific year."""
        for stmt in self.income_statements:
            if stmt.period.fiscal_year == fiscal_year:
                return stmt
        return None
    
    def get_balance_sheet(self, fiscal_year: int) -> BalanceSheetLine | None:
        """Get balance sheet for a specific year."""
        for stmt in self.balance_sheets:
            if stmt.period.fiscal_year == fiscal_year:
                return stmt
        return None
    
    def get_cash_flow(self, fiscal_year: int) -> CashFlowLine | None:
        """Get cash flow statement for a specific year."""
        for stmt in self.cash_flow_statements:
            if stmt.period.fiscal_year == fiscal_year:
                return stmt
        return None
