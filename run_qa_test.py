import json
import sys
from decimal import Decimal
from finanalyst_tools.tool_registry import ToolRegistry

# Initialize the Registry
registry = ToolRegistry()

# ==========================================
# 1. TEST DATASETS
# ==========================================

SCENARIOS = {
    "SCENARIO_A_HAPPY_PATH": {
        "description": "Perfectly balanced Singapore SMB data",
        "statement_set": {
            "income_statement": {
                "period": {"year": 2023, "period_type": "annual"},
                "currency": "SGD",
                "total_revenue": 5000000,
                "cost_of_goods_sold": 2000000,
                "operating_expenses": 1500000,
                "net_income": 1200000,
                "interest_expense": 50000,
                "income_tax_expense": 250000
            },
            "balance_sheet": {
                "period": {"year": 2023, "period_type": "annual"},
                "currency": "SGD",
                "cash_and_equivalents": 800000,
                "accounts_receivable": 400000,
                "inventory": 300000,
                "total_current_assets": 1500000,
                "property_plant_equipment": 1000000,
                "total_non_current_assets": 1000000,
                "total_assets": 2500000,  # Matches sum
                
                "accounts_payable": 200000,
                "short_term_debt": 100000,
                "total_current_liabilities": 300000,
                "long_term_debt": 500000,
                "total_non_current_liabilities": 500000,
                "total_liabilities": 800000,
                
                "common_stock": 500000,
                "retained_earnings": 1200000,
                "total_shareholders_equity": 1700000  # Liab (800k) + Equity (1.7M) = 2.5M
            },
            "cash_flow_statement": {
                "period": {"year": 2023, "period_type": "annual"},
                "currency": "SGD",
                "net_income": 1200000,  # Matches IS
                "ending_cash": 800000   # Matches BS
            }
        }
    },

    "SCENARIO_B_STRESS_TEST": {
        "description": "Reconciliation failures and Implausible Data",
        "statement_set": {
            "income_statement": {
                "period": {"year": 2023, "period_type": "annual"},
                "currency": "USD",
                "total_revenue": "100000",
                "cost_of_goods_sold": "-5000", # IMPLAUSIBLE: Negative COGS
                "net_income": "50000"
            },
            "balance_sheet": {
                "period": {"year": 2023, "period_type": "annual"},
                "currency": "USD",
                "cash_and_equivalents": -5000, # WARNING: Negative Cash
                "total_assets": 100000,
                "total_liabilities": 20000,
                "total_shareholders_equity": 20000 
                # ERROR: 20k + 20k = 40k != 100k (Reconciliation Fail)
            }
        }
    },

    "SCENARIO_C_BLOCKING_FAIL": {
        "description": "Missing mandatory schema fields",
        "statement_set": {
            "income_statement": {
                "period": {"year": 2023, "period_type": "annual"},
                "currency": "EUR",
                "total_revenue": 500000
                # MISSING: cost_of_goods_sold
            },
            "balance_sheet": {
                "period": {"year": 2023, "period_type": "annual"},
                "currency": "EUR",
                "cash_and_equivalents": 50000
                # MISSING: total_assets, total_liabilities
            }
        }
    }
}

# ==========================================
# 2. EXECUTION RUNNER
# ==========================================

def run_test():
    print("🚀 STARTING FINANALYST-PRO QUALITY ASSURANCE SUITE\n")
    print("="*60)

    for name, data in SCENARIOS.items():
        print(f"\n🧪 RUNNING: {name}")
        print(f"📝 Desc: {data['description']}")
        print("-" * 60)
        
        try:
            # We call the tool exactly as the LLM Dispatcher would
            report = registry.execute_tool(
                "analyze_financials",
                statement_set=data["statement_set"],
                analysis_type="comprehensive",
                include_audit_trail=True
            )
            
            # Print a preview of the report (First 20 lines + Recommendations)
            lines = report.split('\n')
            
            print("--- OUTPUT PREVIEW (HEADER/SUMMARY) ---")
            for line in lines[:15]:
                print(line)
                
            # Check for specific keywords to verify logic
            if "Confidence Level: **HIGH**" in report:
                print("\n✅ VERIFIED: High Confidence achieved.")
            elif "Confidence Level: **LOW**" in report or "Confidence Level: **MEDIUM**" in report:
                 print("\n⚠️ VERIFIED: Confidence Penalty applied.")
            
            if "Calculation Steps" in report:
                 print("✅ VERIFIED: Audit trail generated.")
                 
            if "Validation failed" in report:
                 print("✅ VERIFIED: Pipeline halted on Validation Error (As Expected).")

        except Exception as e:
            print(f"❌ CRITICAL EXECUTION ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("🏁 QA SUITE COMPLETE")

if __name__ == "__main__":
    run_test()
