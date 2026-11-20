from __future__ import annotations

import pytest

from llm_bench.json_utils import OutputFormatError, parse_json_object
from llm_bench.service import IdempotencyCache


def test_model_json_output_parser_accepts_object_and_fence() -> None:
    assert parse_json_object('{"answer": "ok"}') == {"answer": "ok"}
    assert parse_json_object('```json\n{"answer": "ok"}\n```') == {"answer": "ok"}


def test_no_answer_with_hallucinated_identifier_is_incorrect() -> None:
    from llm_bench.evaluation import answer_correct

    question = {"expected_behavior": "no_answer", "gold_groups": []}
    answer = "It invokes `secret_function`. I cannot determine the rest."
    assert answer_correct(question, answer) is False


@pytest.mark.parametrize(
    "text",
    [
        '{"answer": "truncated"',
        "[1, 2, 3]",
        '```json\n{"answer": 1}\n',
        "",
    ],
)
def test_malformed_model_json_is_rejected(text: str) -> None:
    with pytest.raises(OutputFormatError):
        parse_json_object(text)


def test_idempotency_cache_replays_same_key() -> None:
    cache = IdempotencyCache(ttl_s=30)
    assert cache.get("/ask", "request-1") is None
    payload = {"answer": "cached"}
    cache.set("/ask", "request-1", payload)
    assert cache.get("/ask", "request-1") == payload
    assert cache.get("/agent", "request-1") is None


def test_grounded_tool_marks_invalid_citation_id() -> None:
    from llm_bench.llm import LLMResponse
    from llm_bench.qa import grounded_from_tool
    from llm_bench.retrieval import Hit

    class InvalidCitationLLM:
        def generate(self, *_, **__):
            return LLMResponse(
                text="The answer cites a missing source. [E99]",
                latency_s=0.01,
                ttft_s=0.005,
                prompt_tokens=10,
                completion_tokens=8,
                timings={},
            )

    hit = Hit(
        chunk_id="doc-a:1",
        document_id="doc-a",
        title="Policy",
        section="Timeouts",
        text="Tools stop after two timeouts.",
        source_url="https://example.test/a",
        license="MIT",
        score=1.0,
        rank=1,
    )
    result = grounded_from_tool(InvalidCitationLLM(), "timeouts?", "fake", [hit])
    assert result.error == "Invalid citation ids: [99]"
    assert result.citations == []


def test_faithfulness_detects_negation_reversal() -> None:
    from llm_bench.evaluation import faithfulness_metrics
    from llm_bench.qa import AnswerResult

    result = AnswerResult(
        answer="vLLM guarantees deterministic outputs by default.",
        mode="hybrid_rerank",
        model_id="fake",
        retrieved=[
            {
                "document_id": "doc-a",
                "section": "Default",
                "text": "vLLM does not guarantee deterministic outputs by default.",
            }
        ],
        citations=[
            {
                "evidence_id": "E1",
                "document_id": "doc-a",
                "section": "Default",
            }
        ],
    )
    overlap, reversal = faithfulness_metrics(result)
    assert overlap is not None
    assert overlap > 0
    assert reversal is True
    _, skipped = faithfulness_metrics(result, {"category": "false_premise"})
    assert skipped is False

    refusal = AnswerResult(
        answer="I cannot determine this from the provided evidence.",
        mode="hybrid_rerank",
        model_id="fake",
        retrieved=result.retrieved,
        citations=result.citations,
    )
    _, skipped = faithfulness_metrics(
        refusal,
        {"category": "unanswerable", "expected_behavior": "no_answer"},
    )
    assert skipped is False
