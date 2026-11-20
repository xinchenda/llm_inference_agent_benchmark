from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from llm_bench.llm import LLMResponse, LocalLLM
from llm_bench.retrieval import Hit, KnowledgeIndex, deduplicate_documents

CITATION_PATTERN = re.compile(r"\[E(\d+)\]")

GROUNDED_SYSTEM = """You are a careful technical assistant.
Answer only from the supplied evidence. Cite every factual paragraph with [E1], [E2], etc.
If the evidence directly answers the question, answer confidently.
Only when no supplied evidence supports the answer, say
"I cannot determine this from the provided evidence."
Never combine a supported answer with a no-answer disclaimer; use only one conclusion.
When evidence conflicts, report both values, identify versions if present,
and do not silently choose one.
Correct any false premise in the question. Be concise and specific."""

DIRECT_SYSTEM = """You are a concise technical assistant. Answer the question directly.
If the premise is false, correct it. If the answer is not knowable,
say so instead of inventing facts."""


@dataclass
class AnswerResult:
    answer: str
    mode: str
    model_id: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    latency_s: float = 0.0
    ttft_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_per_second: float | None = None
    error: str | None = None
    tool_trace: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _response_fields(response: LLMResponse) -> dict[str, Any]:
    return {
        "latency_s": response.latency_s,
        "ttft_s": response.ttft_s,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "tokens_per_second": response.tokens_per_second,
    }


def direct_answer(
    llm: LocalLLM,
    question: str,
    model_id: str,
    max_tokens: int = 180,
) -> AnswerResult:
    try:
        response = llm.generate(
            DIRECT_SYSTEM,
            question,
            max_tokens=max_tokens,
        )
        return AnswerResult(
            answer=response.text,
            mode="direct",
            model_id=model_id,
            **_response_fields(response),
        )
    except Exception as exc:
        return AnswerResult(
            answer="",
            mode="direct",
            model_id=model_id,
            error=f"{type(exc).__name__}: {exc}",
        )


def retrieve(
    index: KnowledgeIndex,
    question: str,
    mode: str,
    retrieval_config: dict[str, Any],
) -> list[Hit]:
    candidates = index.search(
        question,
        mode=mode,
        top_k=20,
        bm25_k=int(retrieval_config["bm25_k"]),
        vector_k=int(retrieval_config["vector_k"]),
        rerank_k=int(retrieval_config["rerank_k"]),
        rrf_k=int(retrieval_config["rrf_k"]),
    )
    return deduplicate_documents(candidates, limit=5)


def build_evidence_prompt(question: str, hits: list[Hit], max_chars: int = 6500) -> str:
    blocks: list[str] = []
    remaining = max_chars
    for number, hit in enumerate(hits, 1):
        header = f"[E{number}] {hit.title} | {hit.section} | {hit.document_id} | {hit.license}"
        available = remaining - len(header) - 10
        if available < 100:
            break
        body = hit.text[:available]
        block = f"{header}\n{body}"
        blocks.append(block)
        remaining -= len(block)
    evidence = "\n\n".join(blocks) if blocks else "(No evidence retrieved.)"
    return f"""Question:
{question}

Evidence:
{evidence}

Answer the question in at most 120 words. Use only the evidence and include citations."""


def grounded_answer(
    llm: LocalLLM,
    index: KnowledgeIndex,
    question: str,
    model_id: str,
    mode: str,
    retrieval_config: dict[str, Any],
    max_tokens: int = 180,
) -> AnswerResult:
    try:
        hits = retrieve(index, question, mode, retrieval_config)
        prompt = build_evidence_prompt(question, hits)
        response = llm.generate(GROUNDED_SYSTEM, prompt, max_tokens=max_tokens)
        citations = [
            {
                "evidence_id": f"E{number}",
                "document_id": hit.document_id,
                "title": hit.title,
                "section": hit.section,
                "source_url": hit.source_url,
            }
            for number, hit in enumerate(hits, 1)
            if f"[E{number}]" in response.text
        ]
        invalid_citations = sorted(
            {
                int(number)
                for number in CITATION_PATTERN.findall(response.text)
                if int(number) < 1 or int(number) > len(hits)
            }
        )
        return AnswerResult(
            answer=response.text,
            mode=mode,
            model_id=model_id,
            citations=citations,
            retrieved=[hit.as_dict() for hit in hits],
            error=(f"Invalid citation ids: {invalid_citations}" if invalid_citations else None),
            **_response_fields(response),
        )
    except Exception as exc:
        return AnswerResult(
            answer="",
            mode=mode,
            model_id=model_id,
            error=f"{type(exc).__name__}: {exc}",
        )


def grounded_from_tool(
    llm: LocalLLM,
    question: str,
    model_id: str,
    hits: list[Hit],
    max_tokens: int = 180,
) -> AnswerResult:
    prompt = build_evidence_prompt(question, hits)
    try:
        response = llm.generate(GROUNDED_SYSTEM, prompt, max_tokens=max_tokens)
        citations = [
            {
                "evidence_id": f"E{number}",
                "document_id": hit.document_id,
                "title": hit.title,
                "section": hit.section,
                "source_url": hit.source_url,
            }
            for number, hit in enumerate(hits, 1)
            if f"[E{number}]" in response.text
        ]
        invalid_citations = sorted(
            {
                int(number)
                for number in CITATION_PATTERN.findall(response.text)
                if int(number) < 1 or int(number) > len(hits)
            }
        )
        return AnswerResult(
            answer=response.text,
            mode="agent",
            model_id=model_id,
            citations=citations,
            retrieved=[hit.as_dict() for hit in hits],
            error=(f"Invalid citation ids: {invalid_citations}" if invalid_citations else None),
            **_response_fields(response),
        )
    except Exception as exc:
        return AnswerResult(
            answer="",
            mode="agent",
            model_id=model_id,
            error=f"{type(exc).__name__}: {exc}",
        )
