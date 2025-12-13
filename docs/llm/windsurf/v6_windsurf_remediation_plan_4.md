I’m going to finish **Milestone 5** by making pipeline outcomes *explicit* in the returned `ComprehensiveAnalysisResult`: capture pipeline **errors**, **warnings**, and whether the output is **partial** (e.g., some metrics uncalculable). I’ll first inspect the existing result model to extend it in a backward-compatible way, then wire those fields in [AnalysisPipeline._create_result()](cci:1://file:///home/project/FinAnalyst-Pro/finanalyst_tools/orchestration/pipeline.py:318:4-354:21) / [_create_error_result()](cci:1://file:///home/project/FinAnalyst-Pro/finanalyst_tools/orchestration/pipeline.py:356:4-376:21), and finally rerun the venv `compileall` gate.
You’re right to flag that tool error: I tried to read past the end of [report_generator.py](cci:7://file:///home/project/FinAnalyst-Pro/finanalyst_tools/orchestration/report_generator.py:0:0-0:0)—it’s only ~200 lines in this repo, so there’s nothing beyond that offset.

Next I’ll complete **Milestone 5** by making failure/partial semantics explicit in the returned [ComprehensiveAnalysisResult](cci:2://file:///home/project/FinAnalyst-Pro/finanalyst_tools/models/analysis_results.py:350:0-426:64) (add `pipeline_errors`, `pipeline_warnings`, `is_partial`, and `uncalculable_metrics`), wire those from [AnalysisPipeline](cci:2://file:///home/project/FinAnalyst-Pro/finanalyst_tools/orchestration/pipeline.py:94:0-412:30), and update the report summary to surface them. Then I’ll rerun the venv `compileall` gate.
I’m going to inspect [orchestration/pipeline.py](cci:7://file:///home/project/FinAnalyst-Pro/finanalyst_tools/orchestration/pipeline.py:0:0-0:0) around [_create_error_result()](cci:1://file:///home/project/FinAnalyst-Pro/finanalyst_tools/orchestration/pipeline.py:356:4-376:21)—the patch output suggests there may be a stray `]` or malformed block causing the Pyright “Expected expression” error. Then I’ll fix that function cleanly, ensure pipeline error/partial fields are set correctly, and re-run the venv `compileall` gate.







I’m going to move on to **Milestone 6 (confidence scoring consolidation)** by inspecting `orchestration/confidence_scorer.py` and its call sites, then refactor so there’s exactly one canonical scoring entry point (used by the pipeline and any tools) with consistent justification generation. I’ll keep it backward-compatible by leaving existing names as thin wrappers if needed, and then I’ll re-run the venv `compileall` gate.

I’ll now consolidate confidence scoring by extracting a single internal scorer function in [orchestration/confidence_scorer.py](cci:7://file:///home/project/FinAnalyst-Pro/finanalyst_tools/orchestration/confidence_scorer.py:0:0-0:0) and making both the functional API ([calculate_confidence_level](cci:1://file:///home/project/FinAnalyst-Pro/finanalyst_tools/orchestration/confidence_scorer.py:27:0-114:5)) and the class API ([ConfidenceScorer.calculate](cci:1://file:///home/project/FinAnalyst-Pro/finanalyst_tools/orchestration/confidence_scorer.py:170:4-220:9)) delegate to it. This removes duplicate logic while keeping the pipeline’s current confidence output behavior stable. Then I’ll re-run the venv `compileall` gate.






I’ll proceed with **Milestone 7 (agent surface cleanup)** by (1) inventorying what tools are currently registered/exposed to the LLM, (2) identifying which should be *internal-only* (low-level metric calculators) vs *agent-facing* (high-level orchestration), and then (3) implementing a minimal mechanism to hide internal tools from [get_openai_tools()](cci:1://file:///home/project/FinAnalyst-Pro/finanalyst_tools/tool_registry.py:476:4-494:52) / [get_anthropic_tools()](cci:1://file:///home/project/FinAnalyst-Pro/finanalyst_tools/tool_registry.py:496:4-514:55) while keeping them executable for internal use. I’ll then add a single high-level “analyze” tool if the codebase already has an orchestration entrypoint.

I’m going to implement Milestone 7 by adding a simple `expose_to_llm` flag to [ToolDefinition](cci:2://file:///home/project/FinAnalyst-Pro/finanalyst_tools/tool_registry.py:101:0-350:75) and filtering [get_openai_tools()](cci:1://file:///home/project/FinAnalyst-Pro/finanalyst_tools/tool_registry.py:476:4-494:52) / [get_anthropic_tools()](cci:1://file:///home/project/FinAnalyst-Pro/finanalyst_tools/tool_registry.py:496:4-514:55) to only publish tools marked public. Then I’ll create a single high-level `analyze_financials` tool (pipeline-backed) and mark the low-level `calculate_*` tools as internal-only. I’ll first search for existing model constructors (`from_dict` / `parse`) to safely build `FinancialStatementSet` from raw dict inputs.
I’ll now finish **Milestone 7** by (1) adding an `expose_to_llm` flag to [ToolDefinition](cci:2://file:///home/project/FinAnalyst-Pro/finanalyst_tools/tool_registry.py:101:0-350:75), (2) filtering schema generation so only public tools are shown to the LLM, and (3) registering a single high-level `analyze_financials` tool that runs [AnalysisPipeline](cci:2://file:///home/project/FinAnalyst-Pro/finanalyst_tools/orchestration/pipeline.py:94:0-406:30) from raw statement dicts (parsed via the existing Pydantic models). Low-level `calculate_*` tools will remain executable but will be hidden from the LLM tool list. I’ll read the [FinancialStatementSet](cci:2://file:///home/project/FinAnalyst-Pro/finanalyst_tools/models/financial_statements.py:616:0-669:21) model section to implement the parsing correctly.

I’m going to implement Milestone 7 now by adding an `expose_to_llm` flag to [ToolDefinition](cci:2://file:///home/project/FinAnalyst-Pro/finanalyst_tools/tool_registry.py:101:0-350:75), filtering schema generation so only public tools are shown to the LLM, and registering a single public `analyze_financials` tool that runs [AnalysisPipeline](cci:2://file:///home/project/FinAnalyst-Pro/finanalyst_tools/orchestration/pipeline.py:94:0-406:30) end-to-end (parsing inputs via `FinancialStatementSet.model_validate`). I’ll keep low-level `calculate_*` tools executable but hidden.
I’m going to fix the syntax errors introduced in [tool_registry.py](cci:7://file:///home/project/FinAnalyst-Pro/finanalyst_tools/tool_registry.py:0:0-0:0) (the new [analyze_financials](cci:1://file:///home/project/FinAnalyst-Pro/finanalyst_tools/tool_registry.py:603:8-866:30) helper currently has an unclosed [generate_financial_report(](cci:1://file:///home/project/FinAnalyst-Pro/finanalyst_tools/orchestration/report_generator.py:42:0-67:55) call, which breaks the module). I’ll inspect the exact broken region, patch it to a valid function body, then re-run the venv `compileall` gate.






I’m going to verify the [tool_registry.py](cci:7://file:///home/project/FinAnalyst-Pro/finanalyst_tools/tool_registry.py:0:0-0:0) agent-surface changes are fully consistent: ensure [get_anthropic_tools()](cci:1://file:///home/project/FinAnalyst-Pro/finanalyst_tools/tool_registry.py:497:4-515:55) is also filtering by `expose_to_llm`, confirm only [analyze_financials](cci:1://file:///home/project/FinAnalyst-Pro/finanalyst_tools/tool_registry.py:603:8-650:13) + `validate_financial_data` are exposed, and then mark Milestone 7 complete.






## Done — remediation plan executed through Milestone 7

### What I completed since your last “continue” (high impact items)
- **Milestone 5 (phase gating + failure/partial semantics)** is now complete:
  - [AnalysisPipeline](cci:2://file:///home/project/FinAnalyst-Pro/finanalyst_tools/orchestration/pipeline.py:94:0-423:30) enforces phase order and returns structured error results on failure.
  - [ComprehensiveAnalysisResult](cci:2://file:///home/project/FinAnalyst-Pro/finanalyst_tools/models/analysis_results.py:350:0-434:64) now explicitly carries:
    - `pipeline_errors`
    - `pipeline_warnings`
    - `is_partial`
    - `uncalculable_metrics`
  - Report executive summary now surfaces partial/error/warning counts.

- **Milestone 6 (confidence scoring consolidation)** is complete:
  - Removed duplicate logic by introducing a single internal implementation ([_calculate_confidence_assessment](cci:1://file:///home/project/FinAnalyst-Pro/finanalyst_tools/orchestration/confidence_scorer.py:25:0-96:5)).
  - Both [calculate_confidence_level()](cci:1://file:///home/project/FinAnalyst-Pro/finanalyst_tools/orchestration/confidence_scorer.py:99:0-118:5) and [ConfidenceScorer.calculate()](cci:1://file:///home/project/FinAnalyst-Pro/finanalyst_tools/orchestration/confidence_scorer.py:174:4-195:9) delegate to it.
  - **No behavior change intended**, just DRY + consistent output.

- **Milestone 7 (agent surface cleanup)** is complete:
  - Added `ToolDefinition.expose_to_llm`.
  - [get_openai_tools()](cci:1://file:///home/project/FinAnalyst-Pro/finanalyst_tools/tool_registry.py:477:4-495:52) and [get_anthropic_tools()](cci:1://file:///home/project/FinAnalyst-Pro/finanalyst_tools/tool_registry.py:497:4-515:55) now **only expose** tools with `expose_to_llm=True`.
  - Registered a single high-level **public** tool: [analyze_financials](cci:1://file:///home/project/FinAnalyst-Pro/finanalyst_tools/tool_registry.py:603:8-866:30) (pipeline-backed).
  - Kept `validate_financial_data` public.
  - Marked low-level `calculate_*` tools as `expose_to_llm=False` (still executable internally).
  - Fixed the temporary syntax break in [tool_registry.py](cci:7://file:///home/project/FinAnalyst-Pro/finanalyst_tools/tool_registry.py:0:0-0:0) introduced during the tool addition.

### Verification gates
- **Venv compile gate passed** after fixes:
  - `. /opt/venv/bin/activate && python -m compileall -q finanalyst_tools` ✅

## Status
- All milestones **1–7 are completed**.
- No remaining in-progress items in the plan.

If you want, the next logical step is a **quick end-to-end sanity run** using [analyze_financials](cci:1://file:///home/project/FinAnalyst-Pro/finanalyst_tools/tool_registry.py:603:8-866:30) with a minimal `statement_set` payload to confirm:
- tool boundary coercion behaves (Decimal-safe)
- pipeline executes
- report renders and includes new status fields
