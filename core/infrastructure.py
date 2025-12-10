# ============================================================================
# FILE: finanalyst_pro/core/infrastructure.py
# PURPOSE: Shared infrastructure for all tools
# ============================================================================

from __future__ import annotations

import logging
import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, Generic
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict

# ============================================================================
# CONFIGURATION
# ============================================================================

class ToolConfig(BaseModel):
    """Global configuration for FinAnalyst-Pro tools."""
    
    model_config = ConfigDict(frozen=True)
    
    # Decimal precision for financial calculations
    decimal_places: int = 2
    percentage_decimal_places: int = 2
    
    # Currency settings
    default_currency: str = "SGD"
    currency_symbol: str = "$"
    
    # Audit settings
    enable_audit_log: bool = True
    audit_log_path: Path = Path("./logs/audit.jsonl")
    
    # Memory settings
    memory_backend: str = "sqlite"  # "sqlite" | "redis" | "memory"
    memory_db_path: Path = Path("./data/memory.db")
    
    # Plausibility ranges for validation
    plausibility_ranges: dict[str, tuple[float, float]] = Field(default_factory=lambda: {
        "gross_margin": (-50.0, 95.0),
        "operating_margin": (-100.0, 80.0),
        "net_margin": (-200.0, 50.0),
        "current_ratio": (0.1, 10.0),
        "quick_ratio": (0.05, 8.0),
        "debt_to_equity": (0.0, 20.0),
        "asset_turnover": (0.01, 10.0),
        "inventory_turnover": (0.1, 100.0),
        "roe": (-100.0, 100.0),
        "roa": (-50.0, 50.0),
    })


# Global config instance
CONFIG = ToolConfig()


# ============================================================================
# ENUMS
# ============================================================================

class ConfidenceLevel(str, Enum):
    """Confidence level for analysis results."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class AnalysisType(str, Enum):
    """Types of financial analysis supported."""
    INCOME_STATEMENT = "income_statement"
    CASH_FLOW = "cash_flow"
    BALANCE_SHEET = "balance_sheet"
    PROFITABILITY = "profitability"
    LIQUIDITY = "liquidity"
    SOLVENCY = "solvency"
    EFFICIENCY = "efficiency"
    COMPREHENSIVE = "comprehensive"
    TREND = "trend"
    FORECAST = "forecast"


class ValidationStatus(str, Enum):
    """Status of data validation checks."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================================
# BASE SCHEMAS
# ============================================================================

class ToolResult(BaseModel, Generic[TypeVar("T")]):
    """Standardized result wrapper for all tool outputs."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    success: bool
    tool_name: str
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: Any | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float | None = None
    
    def to_llm_context(self) -> str:
        """Format result for LLM consumption."""
        if self.success:
            return json.dumps({
                "status": "success",
                "tool": self.tool_name,
                "data": self.data,
                "warnings": self.warnings if self.warnings else None,
            }, indent=2, default=str)
        else:
            return json.dumps({
                "status": "error",
                "tool": self.tool_name,
                "error": self.error,
            }, indent=2)


class AuditLogEntry(BaseModel):
    """Audit log entry for compliance and debugging."""
    
    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tool_name: str
    operation: str
    input_hash: str  # SHA-256 of inputs for traceability without storing PII
    success: bool
    execution_time_ms: float
    user_session_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None


# ============================================================================
# DECIMAL UTILITIES
# ============================================================================

class FinancialDecimal:
    """Precise decimal operations for financial calculations."""
    
    @staticmethod
    def from_value(value: int | float | str | Decimal) -> Decimal:
        """Convert any numeric value to Decimal."""
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))
    
    @staticmethod
    def round_currency(value: Decimal, places: int = 2) -> Decimal:
        """Round to currency precision."""
        quantize_str = "0." + "0" * places
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
    
    @staticmethod
    def round_percentage(value: Decimal, places: int = 2) -> Decimal:
        """Round percentage to specified precision."""
        quantize_str = "0." + "0" * places
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
    
    @staticmethod
    def safe_divide(
        numerator: Decimal, 
        denominator: Decimal, 
        default: Decimal | None = None
    ) -> Decimal | None:
        """Safely divide, returning default if denominator is zero."""
        if denominator == 0:
            return default
        return numerator / denominator
    
    @staticmethod
    def percentage(value: Decimal, total: Decimal) -> Decimal | None:
        """Calculate percentage of total."""
        result = FinancialDecimal.safe_divide(value * 100, total)
        if result is not None:
            return FinancialDecimal.round_percentage(result)
        return None


# ============================================================================
# AUDIT LOGGER
# ============================================================================

class AuditLogger:
    """Centralized audit logging for compliance."""
    
    def __init__(self, config: ToolConfig = CONFIG):
        self.config = config
        self.logger = logging.getLogger("finanalyst.audit")
        
        if config.enable_audit_log:
            config.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log(self, entry: AuditLogEntry) -> None:
        """Write audit log entry."""
        if not self.config.enable_audit_log:
            return
        
        with open(self.config.audit_log_path, "a") as f:
            f.write(entry.model_dump_json() + "\n")
    
    @staticmethod
    def hash_input(data: Any) -> str:
        """Create SHA-256 hash of input data for audit trail."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]


# Global audit logger instance
AUDIT = AuditLogger()
