# Agent tool failure analysis

The LLM-planner configuration produced 14 tool-related failures
among 20 fixed tool-call questions: 11 wrong-tool results and 3 planner/runtime
errors after the single repair attempt.

## Failure categories

| Category | Cases |
|---|---:|
| `arithmetic_misrouted_to_rag` | 2 |
| `registry_query_misrouted_to_rag` | 7 |
| `report_plan_missing_generate_report` | 2 |
| `route_contract_mismatch_after_repair` | 3 |

## Case comparison

| Question | LLM-planner category | Expected tools | Successful tools | Rule baseline |
|---|---|---|---|---|
| q081 | `registry_query_misrouted_to_rag` | compare_models | search_knowledge | success |
| q082 | `registry_query_misrouted_to_rag` | compare_models | search_knowledge | success |
| q084 | `registry_query_misrouted_to_rag` | compare_models | search_knowledge | success |
| q085 | `registry_query_misrouted_to_rag` | compare_models | search_knowledge | wrong_tool |
| q086 | `route_contract_mismatch_after_repair` | compare_models | none | success |
| q088 | `registry_query_misrouted_to_rag` | compare_models | search_knowledge | success |
| q089 | `registry_query_misrouted_to_rag` | compare_models | search_knowledge | success |
| q090 | `registry_query_misrouted_to_rag` | compare_models | search_knowledge | success |
| q094 | `arithmetic_misrouted_to_rag` | calculate_performance | search_knowledge | wrong_tool |
| q095 | `arithmetic_misrouted_to_rag` | calculate_performance | search_knowledge | wrong_tool |
| q096 | `route_contract_mismatch_after_repair` | calculate_performance | none | success |
| q097 | `route_contract_mismatch_after_repair` | calculate_performance | none | success |
| q099 | `report_plan_missing_generate_report` | search_knowledge, generate_report | search_knowledge | citation_missing |
| q100 | `report_plan_missing_generate_report` | search_knowledge, generate_report | search_knowledge | retrieval_miss |

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
