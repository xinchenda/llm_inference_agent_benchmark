from __future__ import annotations

import time

from fastapi.testclient import TestClient

from llm_bench.service import create_app


def test_service_replays_idempotent_request(toy_index, fake_llm, monkeypatch) -> None:
    monkeypatch.setattr(
        "llm_bench.service.load_models",
        lambda: {
            "models": {
                "fake": {
                    "display_name": "Fake",
                    "quantization": "none",
                    "context_length": 1024,
                    "file": "models/fake.gguf",
                    "license": "MIT",
                }
            }
        },
    )
    config = {
        "llm": {"active_model": "fake"},
        "retrieval": {
            "bm25_k": 20,
            "vector_k": 20,
            "rerank_k": 8,
            "rrf_k": 60,
        },
        "agent": {
            "max_steps": 4,
            "max_tool_calls": 4,
            "tool_timeout_s": 1,
            "planner_mode": "rule",
        },
        "service": {"request_timeout_s": 2},
        "paths": {"artifacts": "artifacts"},
    }
    app = create_app(config, fake_llm, toy_index, default_model_id="fake")
    # The model registry is intentionally loaded from disk in production. This test
    # exercises health and validation without starting llama.cpp.
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["backend"] == "ok"
    invalid = client.post(
        "/search",
        json={"query": "timeout", "mode": "not-a-mode"},
    )
    assert invalid.status_code == 422
    headers = {"Idempotency-Key": "same-request"}
    first = client.post(
        "/ask",
        json={"question": "What is the timeout policy?", "mode": "hybrid_rerank"},
        headers=headers,
    )
    second = client.post(
        "/ask",
        json={"question": "What is the timeout policy?", "mode": "hybrid_rerank"},
        headers=headers,
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.headers["x-idempotent-replay"] == "true"
    assert fake_llm.calls == 1


def test_service_exposes_metrics(toy_index, fake_llm) -> None:
    config = {
        "llm": {"active_model": "fake"},
        "retrieval": {
            "bm25_k": 20,
            "vector_k": 20,
            "rerank_k": 8,
            "rrf_k": 60,
        },
        "agent": {
            "max_steps": 4,
            "max_tool_calls": 4,
            "tool_timeout_s": 1,
            "planner_mode": "rule",
        },
        "service": {"request_timeout_s": 2},
        "paths": {"artifacts": "artifacts"},
    }
    client = TestClient(create_app(config, fake_llm, toy_index, "fake"))
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "llm_bench_index_chunks 2" in response.text


def test_ask_returns_504_when_request_times_out(toy_index, fake_llm, monkeypatch) -> None:
    monkeypatch.setattr(
        "llm_bench.service.load_models",
        lambda: {"models": {"fake": {"file": "models/fake.gguf"}}},
    )

    def slow_answer(**_):
        time.sleep(0.2)

    monkeypatch.setattr("llm_bench.service.grounded_answer", slow_answer)
    config = {
        "llm": {"active_model": "fake"},
        "retrieval": {
            "bm25_k": 20,
            "vector_k": 20,
            "rerank_k": 8,
            "rrf_k": 60,
        },
        "agent": {
            "max_steps": 4,
            "max_tool_calls": 4,
            "tool_timeout_s": 1,
            "planner_mode": "rule",
        },
        "service": {"request_timeout_s": 0.02},
        "paths": {"artifacts": "artifacts"},
    }
    client = TestClient(create_app(config, fake_llm, toy_index, "fake"))
    response = client.post(
        "/ask",
        json={"question": "What is the timeout policy?", "mode": "hybrid_rerank"},
    )
    assert response.status_code == 504


def test_agent_returns_502_when_tool_fails(toy_index, monkeypatch) -> None:
    from llm_bench.llm import LLMResponse

    class BadPlanLLM:
        def generate(self, *_, **__):
            return LLMResponse(
                text=(
                    '{"tool_calls":[{"tool":"search_knowledge",'
                    '"arguments":{}}],"answer_kind":"grounded"}'
                ),
                latency_s=0.01,
                ttft_s=0.005,
                prompt_tokens=10,
                completion_tokens=5,
                timings={},
            )

        def health(self):
            return {"data": []}

    monkeypatch.setattr(
        "llm_bench.service.load_models",
        lambda: {
            "models": {
                "fake": {
                    "file": "models/fake.gguf",
                    "display_name": "Fake",
                    "quantization": "none",
                    "context_length": 1024,
                    "license": "MIT",
                }
            }
        },
    )
    config = {
        "llm": {"active_model": "fake"},
        "retrieval": {
            "bm25_k": 20,
            "vector_k": 20,
            "rerank_k": 8,
            "rrf_k": 60,
        },
        "agent": {
            "max_steps": 4,
            "max_tool_calls": 4,
            "tool_timeout_s": 1,
            "planner_mode": "llm",
            "planner_timeout_s": 1,
        },
        "service": {"request_timeout_s": 2},
        "paths": {"artifacts": "artifacts"},
    }
    client = TestClient(create_app(config, BadPlanLLM(), toy_index, "fake"))
    response = client.post("/agent", json={"question": "Search the timeout policy"})
    assert response.status_code == 502
    assert response.json()["detail"]["error"]
