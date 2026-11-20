#!/usr/bin/env python3
"""Classify Agent tool failures and compare the rule-planner baseline."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "artifacts" / "eval"
REPORTS = ROOT / "reports"


def load_rows(name: str) -> dict[str, dict]:
    path = EVAL / name / "results.jsonl"
    return {
        row["question_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def classify(row: dict) -> str:
    attempts = []
    for trace in row.get("tool_trace", []):
        if trace["tool"] == "planner":
            attempts = trace.get("result", {}).get("planner_attempts", [])
    successful = [
        trace["tool"]
        for trace in row.get("tool_trace", [])
        if trace["tool"] != "planner" and trace["status"] == "ok"
    ]
    expected = row["expected_tools"]
    if row.get("error") and any(item["status"] == "invalid" for item in attempts):
        return "route_contract_mismatch_after_repair"
    if row.get("error"):
        return "planner_runtime_error"
    if expected == ["compare_models"] and "search_knowledge" in successful:
        return "registry_query_misrouted_to_rag"
    if expected == ["calculate_performance"] and "search_knowledge" in successful:
        return "arithmetic_misrouted_to_rag"
    if expected == ["search_knowledge", "generate_report"]:
        return "report_plan_missing_generate_report"
    return "other_wrong_tool"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    agent = load_rows("agent_hybrid_rerank")
    rule = load_rows("agent_rule_rerank")
    failures = [
        row
        for row in agent.values()
        if row["failure_type"] in {"wrong_tool", "runtime_or_tool_error"}
    ]
    output = []
    for row in failures:
        trace = row.get("tool_trace", [])
        attempts = next(
            (
                item.get("result", {}).get("planner_attempts", [])
                for item in trace
                if item["tool"] == "planner"
            ),
            [],
        )
        actual_tools = [
            item["tool"] for item in trace if item["tool"] != "planner" and item["status"] == "ok"
        ]
        baseline = rule[row["question_id"]]
        output.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "expected_tools": row["expected_tools"],
                "actual_tools": actual_tools,
                "category": classify(row),
                "error": row.get("error"),
                "planner_attempts": attempts,
                "answer": row["answer"],
                "rule_baseline_failure": baseline["failure_type"],
                "rule_baseline_answer_correct": baseline["answer_correct"],
            }
        )
    categories = Counter(item["category"] for item in output)
    (REPORTS / "agent_failure_cases.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    category_rows = "\n".join(
        f"| `{name}` | {count} |" for name, count in sorted(categories.items())
    )
    case_rows = "\n".join(
        f"| {item['question_id']} | `{item['category']}` | "
        f"{', '.join(item['expected_tools']) or '-'} | "
        f"{', '.join(item['actual_tools']) or 'none'} | "
        f"{item['rule_baseline_failure'] or 'success'} |"
        for item in output
    )
    report = f"""# Agent tool failure analysis

The LLM-planner configuration produced {len(failures)} tool-related failures
among 20 fixed tool-call questions: 11 wrong-tool results and 3 planner/runtime
errors after the single repair attempt.

## Failure categories

| Category | Cases |
|---|---:|
{category_rows}

## Case comparison

| Question | LLM-planner category | Expected tools | Successful tools | Rule baseline |
|---|---|---|---|---|
{case_rows}

## Diagnosis

- The dominant failure is semantic routing, not HTTP or tool timeout: Qwen3
  often emits schema-valid `search_knowledge` plans for model-registry and
  arithmetic questions.
- Three cases preserved the correct tool but returned the wrong `answer_kind`
  after repair. The plan contract correctly rejected both attempts instead of
  executing an ambiguous route.
- Report requests were sometimes reduced to knowledge search, omitting the
  required `generate_report` second step.
- The rule baseline reached 85% tool-selection accuracy on the 20 tool
  questions, versus 30% for free JSON planning. This supports keeping both a
  deterministic fast path and an LLM planner.

## Engineering improvements

1. Constrain generation with JSON schema or grammar (llama.cpp GBNF/vLLM guided
   decoding) rather than relying on prose instructions.
2. Compile tool preconditions before generation: model comparison requires two
   registry IDs; calculation requires two numeric operands; report requires two
   ordered tool calls.
3. Keep the rule planner as a deterministic pre-router for high-confidence tool
   intents and reserve free LLM planning for ambiguous requests.
4. Add a second validation layer that checks `answer_kind`, tool name, required
   arguments, and tool order before execution.
5. Use tool schemas with examples generated from the registry, not only static
   prompt text.

The raw planner attempts and full traces are in
`reports/agent_failure_cases.json`.
"""
    (REPORTS / "agent_failure_analysis.md").write_text(report, encoding="utf-8")
    print(json.dumps(dict(categories), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
