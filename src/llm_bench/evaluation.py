from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from llm_bench.agent import ControlledAgent
from llm_bench.config import PROJECT_ROOT, load_models, resolve_path
from llm_bench.llm import LlamaServer, LocalLLM, choose_device
from llm_bench.qa import AnswerResult, direct_answer, grounded_answer
from llm_bench.retrieval import KnowledgeIndex, stem_term, tokenize

REFUSAL_MARKERS = (
    "cannot determine",
    "cannot be determined",
    "cannot answer",
    "cannot be answered",
    "not enough evidence",
    "not enough information",
    "insufficient information",
    "not provided",
    "not directly provided",
    "not explicitly provided",
    "not explicitly stated",
    "does not specify",
    "no direct information",
    "no specific information",
    "no evidence",
    "unknown",
    "unable to determine",
    "can't determine",
    "does not provide",
)
NEGATION_PATTERN = re.compile(
    r"\b(?:no|not|never|cannot|can't|doesn't|isn't|aren't|without|none)\b",
    re.IGNORECASE,
)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
RETRIEVAL_MODES = {"vector", "hybrid", "hybrid_rerank"}


def is_retrieval_mode(mode: str) -> bool:
    return mode in RETRIEVAL_MODES or mode.startswith("rag_")


def is_direct_mode(mode: str) -> bool:
    return mode == "direct" or mode.startswith("direct_")


def load_questions(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    questions = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return questions[:limit] if limit else questions


def _text_contains(text: str, alternatives: list[str]) -> bool:
    lowered = text.lower()
    return any(alternative.lower() in lowered for alternative in alternatives)


def answer_correct(question: dict[str, Any], answer: str) -> bool:
    if not answer.strip():
        return False
    if question["expected_behavior"] == "no_answer":
        lowered = answer.lower()
        marker_positions = [lowered.find(marker) for marker in REFUSAL_MARKERS if marker in lowered]
        if not marker_positions:
            return False
        prefix = answer[: min(marker_positions)]
        # A refusal that first asserts a concrete code identifier or number is
        # treated as a hallucinated answer rather than a clean abstention.
        if re.search(r"`[^`]+`|\b\d{2,}\b", prefix):
            return False
        return True
    return all(_text_contains(answer, group) for group in question["gold_groups"])


def recall_at_5(question: dict[str, Any], retrieved: list[dict[str, Any]]) -> float | None:
    expected = set(question["expected_docs"])
    if not expected:
        return None
    found = {item["document_id"] for item in retrieved[:5]}
    return len(expected & found) / len(expected)


def citation_metrics(
    question: dict[str, Any], result: AnswerResult
) -> tuple[float | None, float | None, float | None]:
    if not result.citations:
        return (None, 0.0, None)
    expected = set(question["expected_docs"])
    cited = [item["document_id"] for item in result.citations]
    precision = sum(document in expected for document in cited) / len(cited)
    support = float(bool(expected & set(cited))) if expected else None
    valid = float(all("[E" in f"[{item['evidence_id']}]" for item in result.citations))
    return precision, support, valid


def tool_selection_correct(question: dict[str, Any], result: AnswerResult) -> float | None:
    expected = question["expected_tools"]
    if not expected:
        return None
    actual = [item["tool"] for item in result.tool_trace if item["status"] == "ok"]
    if expected == ["compare_models"]:
        return float("compare_models" in actual)
    if expected == ["calculate_performance"]:
        return float("calculate_performance" in actual)
    return float(all(tool in actual for tool in expected))


def faithfulness_metrics(
    result: AnswerResult,
    question: dict[str, Any] | None = None,
) -> tuple[float | None, bool]:
    if not result.answer or not result.retrieved:
        return None, False
    cited_keys = {(item.get("document_id"), item.get("section")) for item in result.citations}
    cited_evidence = [
        item
        for item in result.retrieved
        if cited_keys and (item.get("document_id"), item.get("section")) in cited_keys
    ]
    evidence = "\n".join(str(item.get("text", "")) for item in cited_evidence if item.get("text"))
    if not evidence:
        return None, False
    answer_terms = {stem_term(term) for term in tokenize(result.answer)}
    evidence_terms = {stem_term(term) for term in tokenize(evidence)}
    overlap = len(answer_terms & evidence_terms) / len(answer_terms) if answer_terms else 0.0
    evidence_sentences = [item for item in SENTENCE_PATTERN.split(evidence) if item]
    negation_reversal = False
    for answer_sentence in SENTENCE_PATTERN.split(result.answer):
        answer_sentence_terms = {stem_term(term) for term in tokenize(answer_sentence)}
        if len(answer_sentence_terms) < 3:
            continue
        answer_negated = bool(NEGATION_PATTERN.search(answer_sentence))
        for evidence_sentence in evidence_sentences:
            evidence_sentence_terms = {stem_term(term) for term in tokenize(evidence_sentence)}
            shared = answer_sentence_terms & evidence_sentence_terms
            union = answer_sentence_terms | evidence_sentence_terms
            similarity = len(shared) / len(union) if union else 0.0
            if len(shared) >= 4 and similarity >= 0.5:
                evidence_negated = bool(NEGATION_PATTERN.search(evidence_sentence))
                if answer_negated != evidence_negated:
                    negation_reversal = True
                    break
        if negation_reversal:
            break
    if question is not None and (
        question.get("category") == "false_premise"
        or (
            question.get("expected_behavior") == "no_answer"
            and _text_contains(result.answer, list(REFUSAL_MARKERS))
        )
    ):
        negation_reversal = False
    return float(min(1.0, overlap)), negation_reversal


def failure_type(
    question: dict[str, Any],
    result: AnswerResult,
    correct: bool,
    recall: float | None,
    citation_support: float | None,
    tool_correct: float | None,
    negation_reversal: bool,
) -> str | None:
    direct_mode = is_direct_mode(result.mode)
    retrieval_mode = is_retrieval_mode(result.mode)
    agent_mode = result.mode == "agent"

    if direct_mode:
        if question["expected_behavior"] == "no_answer" and not correct:
            return "hallucinated_when_unanswerable"
        return None if correct else "answer_incorrect"
    if result.error:
        return "runtime_or_tool_error"
    if agent_mode and tool_correct is not None and tool_correct == 0:
        return "wrong_tool"
    if (
        (retrieval_mode or agent_mode)
        and question["expected_docs"]
        and recall is not None
        and recall == 0
    ):
        return "retrieval_miss"
    if (retrieval_mode or agent_mode) and question["expected_docs"] and not result.citations:
        return "citation_missing"
    if negation_reversal:
        return "negation_reversal"
    if question["expected_behavior"] == "no_answer" and not correct:
        return "hallucinated_when_unanswerable"
    if not correct:
        return "answer_incorrect"
    if (retrieval_mode or agent_mode) and question["expected_docs"] and citation_support == 0:
        return "citation_unsupported"
    return None


def score_result(question: dict[str, Any], result: AnswerResult) -> dict[str, Any]:
    correct = answer_correct(question, result.answer)
    recall = recall_at_5(question, result.retrieved)
    citation_precision, citation_support, citation_valid = citation_metrics(question, result)
    tool_correct = tool_selection_correct(question, result)
    faithfulness_overlap, negation_reversal = faithfulness_metrics(result, question)
    success = correct
    if question["expected_tools"]:
        success = success and tool_correct == 1.0
    if (is_retrieval_mode(result.mode) or result.mode == "agent") and question["expected_docs"]:
        success = success and recall is not None and recall > 0 and citation_support == 1.0
    if result.mode in {"agent", "vector", "hybrid", "hybrid_rerank"}:
        success = success and not negation_reversal
    return {
        "question_id": question["id"],
        "category": question["category"],
        "mode": result.mode,
        "model_id": result.model_id,
        "question": question["question"],
        "expected_docs": question["expected_docs"],
        "expected_behavior": question["expected_behavior"],
        "expected_tools": question["expected_tools"],
        "answer": result.answer,
        "answer_correct": correct,
        "task_success": bool(success),
        "recall_at_5": recall,
        "citation_precision": citation_precision,
        "citation_support": citation_support,
        "citation_valid": citation_valid,
        "faithfulness_overlap": faithfulness_overlap,
        "negation_reversal": negation_reversal,
        "tool_selection_correct": tool_correct,
        "failure_type": failure_type(
            question,
            result,
            correct,
            recall,
            citation_support,
            tool_correct,
            negation_reversal,
        ),
        "citations": result.citations,
        "retrieved_document_ids": [item["document_id"] for item in result.retrieved[:5]],
        "tool_trace": result.tool_trace,
        "latency_s": result.latency_s,
        "ttft_s": result.ttft_s,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "tokens_per_second": result.tokens_per_second,
        "error": result.error,
        "human_review": question["human_review"],
    }


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, q))


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str, subset: list[dict[str, Any]] | None = None) -> float | None:
        selected = subset if subset is not None else records
        values = [row[key] for row in selected if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    latencies = [row["latency_s"] for row in records if row["latency_s"] > 0]
    ttfts = [row["ttft_s"] for row in records if row.get("ttft_s") is not None]
    by_category: dict[str, Any] = {}
    for category in sorted({row["category"] for row in records}):
        subset = [row for row in records if row["category"] == category]
        by_category[category] = {
            "questions": len(subset),
            "answer_accuracy": mean("answer_correct", subset),
            "task_success_rate": mean("task_success", subset),
            "recall_at_5": mean("recall_at_5", subset),
            "citation_precision": mean("citation_precision", subset),
            "citation_support_rate": mean("citation_support", subset),
            "faithfulness_overlap": mean("faithfulness_overlap", subset),
            "negation_reversal_rate": mean("negation_reversal", subset),
            "tool_selection_accuracy": mean("tool_selection_correct", subset),
        }
    return {
        "questions": len(records),
        "answer_accuracy": mean("answer_correct"),
        "task_success_rate": mean("task_success"),
        "recall_at_5": mean("recall_at_5"),
        "citation_precision": mean("citation_precision"),
        "citation_support_rate": mean("citation_support"),
        "faithfulness_overlap": mean("faithfulness_overlap"),
        "negation_reversal_rate": mean("negation_reversal"),
        "tool_selection_accuracy": mean("tool_selection_correct"),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p95_s": percentile(ttfts, 95),
        "failure_types": dict(
            Counter(row["failure_type"] for row in records if row["failure_type"])
        ),
        "by_category": by_category,
    }


class EvaluationRunner:
    def __init__(
        self,
        config: dict[str, Any],
        index: KnowledgeIndex,
        questions: list[dict[str, Any]],
        output_dir: Path,
    ) -> None:
        self.config = config
        self.index = index
        self.questions = questions
        self.output_dir = output_dir
        self.model_registry = load_models()["models"]
        self.concurrency = int(config["eval"]["concurrency"])

    async def _run_rag_question(
        self,
        llm: LocalLLM,
        question: dict[str, Any],
        mode: str,
        model_id: str,
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        async with semaphore:
            result = await asyncio.to_thread(
                grounded_answer,
                llm,
                self.index,
                question["question"],
                model_id,
                mode,
                self.config["retrieval"],
            )
            return score_result(question, result)

    async def _run_direct_question(
        self,
        llm: LocalLLM,
        question: dict[str, Any],
        model_id: str,
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        async with semaphore:
            result = await asyncio.to_thread(
                direct_answer,
                llm,
                question["question"],
                model_id,
                int(self.config["llm"]["max_tokens"]),
            )
            return score_result(question, result)

    async def _run_agent_question(
        self,
        agent: ControlledAgent,
        question: dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        async with semaphore:
            result = await agent.run(question["question"])
            return score_result(question, result)

    async def _evaluate_with_server(
        self,
        server: LlamaServer,
        llm: LocalLLM,
        model_id: str,
        configurations: list[str],
    ) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        semaphore = asyncio.Semaphore(self.concurrency)
        for configuration in configurations:
            print(f"evaluating {configuration} with {model_id}", flush=True)
            started = datetime.now(UTC)
            if configuration.startswith("direct_"):
                records = await asyncio.gather(
                    *[
                        self._run_direct_question(llm, question, model_id, semaphore)
                        for question in self.questions
                    ]
                )
            elif configuration.startswith("rag_"):
                mode = configuration.removeprefix("rag_")
                records = await asyncio.gather(
                    *[
                        self._run_rag_question(
                            llm,
                            question,
                            mode,
                            model_id,
                            semaphore,
                        )
                        for question in self.questions
                    ]
                )
            elif configuration in {"agent_hybrid_rerank", "agent_rule_rerank"}:
                planner_mode = "rule" if configuration == "agent_rule_rerank" else "llm"
                agent = ControlledAgent(
                    llm=llm,
                    index=self.index,
                    models=self.model_registry,
                    retrieval_config=self.config["retrieval"],
                    artifact_dir=self.output_dir / "agent_reports",
                    model_id=model_id,
                    max_steps=int(self.config["agent"]["max_steps"]),
                    max_tool_calls=int(self.config["agent"]["max_tool_calls"]),
                    tool_timeout_s=float(self.config["agent"]["tool_timeout_s"]),
                    planner_mode=planner_mode,
                    planner_timeout_s=float(self.config["agent"].get("planner_timeout_s", 30)),
                )
                records = await asyncio.gather(
                    *[
                        self._run_agent_question(agent, question, semaphore)
                        for question in self.questions
                    ]
                )
            else:
                raise ValueError(f"Unknown evaluation configuration: {configuration}")

            directory = self.output_dir / configuration
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "results.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary = {
                "configuration": configuration,
                "model_id": model_id,
                "started_at": started.isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
                "server_cold_start_s": server.cold_start_s,
                "server_rss_mb": server.rss_mb,
                **summarize(records),
            }
            (directory / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            summaries.append(summary)
        return summaries

    async def run(self) -> list[dict[str, Any]]:
        requested = list(self.config["eval"]["configurations"])
        groups: list[tuple[str, list[str]]] = [
            ("qwen3_0_6b_q8", ["direct_qwen3_0_6b_q8"]),
            (
                "qwen3_1_7b_q4",
                [
                    "direct_qwen3_1_7b_q4",
                    "rag_vector",
                    "rag_hybrid",
                    "rag_hybrid_rerank",
                    "agent_hybrid_rerank",
                    "agent_rule_rerank",
                ],
            ),
            ("qwen3_1_7b_q8", ["direct_qwen3_1_7b_q8"]),
        ]
        summaries: list[dict[str, Any]] = []
        for model_id, configurations in groups:
            configurations = [item for item in configurations if item in requested]
            if not configurations:
                continue
            model = self.model_registry[model_id]
            model_path = PROJECT_ROOT / model["file"]
            server = LlamaServer(
                model_path=model_path,
                port=8080,
                context_size=int(self.config["llm"]["context_size"]),
                parallel=self.concurrency,
                log_path=self.output_dir / "logs" / f"llama-{model_id}.log",
            )
            server.start()
            llm = LocalLLM(
                base_url=server.base_url,
                model_name=model_id,
                api_key=str(self.config["llm"]["api_key"]),
                timeout_s=float(self.config["llm"]["request_timeout_s"]),
                temperature=float(self.config["llm"]["temperature"]),
                top_p=float(self.config["llm"]["top_p"]),
                max_tokens=int(self.config["llm"]["max_tokens"]),
            )
            try:
                summaries.extend(
                    await self._evaluate_with_server(
                        server=server,
                        llm=llm,
                        model_id=model_id,
                        configurations=configurations,
                    )
                )
            finally:
                server.stop()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "all_summaries.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summaries


async def run_evaluation(config: dict[str, Any]) -> list[dict[str, Any]]:
    index_dir = resolve_path(config, "paths.index")
    questions_path = resolve_path(config, "paths.questions")
    output_dir = resolve_path(config, "paths.artifacts") / "eval"
    device = choose_device(str(config["retrieval"]["device"]))
    index = KnowledgeIndex.load(index_dir, device=device)
    questions = load_questions(
        questions_path,
        limit=int(config["eval"]["max_questions"]),
    )
    runner = EvaluationRunner(
        config=config,
        index=index,
        questions=questions,
        output_dir=output_dir,
    )
    return await runner.run()


def collect_badcases(output_dir: Path, limit: int = 12) -> list[dict[str, Any]]:
    preferred = output_dir / "agent_hybrid_rerank" / "results.jsonl"
    if not preferred.exists():
        return []
    rows = [
        json.loads(line)
        for line in preferred.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failures = [row for row in rows if row["failure_type"]]
    return failures[:limit]


def recompute_text_metrics(
    eval_dir: Path,
    index: KnowledgeIndex,
    questions_path: Path,
) -> list[dict[str, Any]]:
    """Recompute lexical faithfulness and failure labels without rerunning models."""
    questions = {item["id"]: item for item in load_questions(questions_path)}
    by_document: dict[str, list[Any]] = {}
    for chunk in index.chunks:
        by_document.setdefault(chunk.document_id, []).append(chunk)
    summaries: list[dict[str, Any]] = []
    for results_path in sorted(eval_dir.glob("*/results.jsonl")):
        configuration = results_path.parent.name
        summary_path = results_path.parent / "summary.json"
        records = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in records:
            question = questions[row["question_id"]]
            mode = row.get("mode")
            if not mode:
                if configuration.startswith("direct"):
                    mode = "direct"
                elif configuration == "agent_hybrid_rerank":
                    mode = "agent"
                elif configuration == "agent_rule_rerank":
                    mode = "agent"
                elif configuration.startswith("rag_"):
                    mode = configuration.removeprefix("rag_")
                else:
                    mode = configuration
            evidence_chunks = []
            citation_sections = {
                (citation["document_id"], citation["section"])
                for citation in row.get("citations", [])
            }
            for document_id in row.get("retrieved_document_ids", []):
                matching = [
                    chunk
                    for chunk in by_document.get(document_id, [])
                    if (document_id, chunk.section) in citation_sections
                ]
                evidence_chunks.extend(matching or by_document.get(document_id, [])[:1])
            result = AnswerResult(
                answer=row["answer"],
                mode=mode,
                model_id=row.get("model_id", ""),
                citations=row.get("citations", []),
                retrieved=[
                    {
                        "document_id": chunk.document_id,
                        "section": chunk.section,
                        "text": chunk.text,
                    }
                    for chunk in evidence_chunks
                ],
                error=row.get("error"),
            )
            overlap, reversal = faithfulness_metrics(result, question)
            correct = answer_correct(question, row["answer"])
            citation_precision, citation_support, citation_valid = citation_metrics(
                question,
                result,
            )
            row["mode"] = mode
            row["answer_correct"] = correct
            row["citation_precision"] = citation_precision
            row["citation_support"] = citation_support
            row["citation_valid"] = citation_valid
            row["faithfulness_overlap"] = overlap
            row["negation_reversal"] = reversal
            row["failure_type"] = failure_type(
                question,
                result,
                correct,
                row.get("recall_at_5"),
                citation_support,
                row.get("tool_selection_correct"),
                reversal,
            )
            success = correct
            if question["expected_tools"]:
                success = success and row.get("tool_selection_correct") == 1.0
            if (is_retrieval_mode(mode) or mode == "agent") and question["expected_docs"]:
                success = success and (row.get("recall_at_5") or 0) > 0 and citation_support == 1.0
            if is_retrieval_mode(mode) or mode == "agent":
                success = success and not reversal
            row["task_success"] = bool(success)
        with results_path.open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if summary_path.exists():
            old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            common = {
                key: old_summary[key]
                for key in (
                    "configuration",
                    "model_id",
                    "started_at",
                    "finished_at",
                    "server_cold_start_s",
                    "server_rss_mb",
                )
                if key in old_summary
            }
            summary = {**common, **summarize(records)}
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            summaries.append(summary)
    (eval_dir / "all_summaries.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summaries
