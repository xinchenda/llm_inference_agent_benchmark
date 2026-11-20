from __future__ import annotations

from llm_bench.evaluation import failure_type
from llm_bench.qa import AnswerResult


def answer(mode: str, error: str | None = None) -> AnswerResult:
    return AnswerResult(answer="answer", mode=mode, model_id="fake", error=error)


def question(expected_behavior: str = "answer") -> dict:
    return {"expected_behavior": expected_behavior}


def test_direct_miss_is_answer_incorrect_not_retrieval_miss() -> None:
    result = answer("direct")
    assert (
        failure_type(
            {**question(), "expected_docs": ["doc-a"]},
            result,
            correct=False,
            recall=0.0,
            citation_support=None,
            tool_correct=0.0,
            negation_reversal=False,
        )
        == "answer_incorrect"
    )


def test_direct_unanswerable_hallucination_has_specific_label() -> None:
    result = answer("direct")
    assert (
        failure_type(
            question("no_answer"),
            result,
            correct=False,
            recall=None,
            citation_support=None,
            tool_correct=None,
            negation_reversal=False,
        )
        == "hallucinated_when_unanswerable"
    )


def test_rag_tool_miss_is_not_wrong_tool() -> None:
    result = answer("hybrid_rerank")
    assert (
        failure_type(
            {**question(), "expected_docs": ["doc-a"]},
            result,
            correct=False,
            recall=0.0,
            citation_support=0.0,
            tool_correct=0.0,
            negation_reversal=False,
        )
        == "retrieval_miss"
    )


def test_agent_tool_miss_is_wrong_tool() -> None:
    result = answer("agent")
    assert (
        failure_type(
            question(),
            result,
            correct=False,
            recall=None,
            citation_support=None,
            tool_correct=0.0,
            negation_reversal=False,
        )
        == "wrong_tool"
    )


def test_agent_tool_success_does_not_require_document_citation() -> None:
    result = answer("agent")
    assert (
        failure_type(
            {**question(), "expected_docs": []},
            result,
            correct=True,
            recall=None,
            citation_support=0.0,
            tool_correct=1.0,
            negation_reversal=False,
        )
        is None
    )


def test_direct_citation_support_does_not_create_rag_failure() -> None:
    result = answer("direct")
    assert (
        failure_type(
            question(),
            result,
            correct=True,
            recall=None,
            citation_support=0.0,
            tool_correct=None,
            negation_reversal=False,
        )
        is None
    )
