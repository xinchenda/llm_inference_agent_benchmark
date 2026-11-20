from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from llm_bench.agent import ControlledAgent
from llm_bench.config import load_models, resolve_path
from llm_bench.llm import LocalLLM
from llm_bench.qa import grounded_answer
from llm_bench.retrieval import KnowledgeIndex, deduplicate_documents

logger = logging.getLogger("llm_bench.service")
logger.setLevel(logging.INFO)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    mode: str = Field(default="hybrid_rerank", pattern="^(vector|hybrid|hybrid_rerank)$")
    model_id: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    mode: str = Field(default="hybrid_rerank", pattern="^(vector|hybrid|hybrid_rerank)$")


class AgentRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    model_id: str | None = None


class RateLimiter:
    def __init__(self, limit: int, window_s: float) -> None:
        self.limit = limit
        self.window_s = window_s
        self.events: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            queue = self.events[key]
            while queue and queue[0] < now - self.window_s:
                queue.popleft()
            if len(queue) >= self.limit:
                return False
            queue.append(now)
            return True


class IdempotencyCache:
    def __init__(self, ttl_s: float = 300, max_entries: int = 200) -> None:
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._items: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def get(self, path: str, key: str | None) -> dict[str, Any] | None:
        if not key:
            return None
        now = time.monotonic()
        with self._lock:
            item = self._items.get((path, key))
            if not item or item[0] < now:
                self._items.pop((path, key), None)
                return None
            return item[1]

    def set(self, path: str, key: str | None, value: dict[str, Any]) -> None:
        if not key:
            return
        now = time.monotonic()
        with self._lock:
            if len(self._items) >= self.max_entries:
                oldest = min(self._items, key=lambda item: self._items[item][0])
                self._items.pop(oldest, None)
            self._items[(path, key)] = (now + self.ttl_s, value)


def create_app(
    config: dict[str, Any],
    llm: LocalLLM,
    index: KnowledgeIndex,
    default_model_id: str | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Local LLM RAG Agent Benchmark",
        version="0.1.0",
        description="Local inference, cited RAG, controlled Agent tools, and metrics.",
    )
    active_model = default_model_id or str(config["llm"]["active_model"])
    registry = load_models()["models"]
    limiter = RateLimiter(limit=30, window_s=60)
    expected_token = os.getenv("LLM_BENCH_API_TOKEN", "")
    idempotency = IdempotencyCache()
    request_timeout_s = float(config.get("service", {}).get("request_timeout_s", 90))
    counters = {"requests": 0, "errors": 0, "rate_limited": 0}
    counters_lock = threading.Lock()

    def authenticate(authorization: str | None = Header(default=None)) -> None:
        if not expected_token:
            return
        if authorization != f"Bearer {expected_token}":
            raise HTTPException(status_code=401, detail="Invalid bearer token")

    @app.middleware("http")
    async def observability(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        client = request.client.host if request.client else "unknown"
        if request.url.path not in {"/health", "/ready", "/metrics"} and not limiter.allow(client):
            with counters_lock:
                counters["rate_limited"] += 1
            return PlainTextResponse("Rate limit exceeded", status_code=429)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            with counters_lock:
                counters["errors"] += 1
            logger.exception(
                json.dumps(
                    {
                        "request_id": request_id,
                        "path": request.url.path,
                        "status": 500,
                    }
                )
            )
            raise
        elapsed = time.perf_counter() - started
        with counters_lock:
            counters["requests"] += 1
            if response.status_code >= 400:
                counters["errors"] += 1
        response.headers["x-request-id"] = request_id
        response.headers["x-process-time-ms"] = f"{elapsed * 1000:.2f}"
        if getattr(request.state, "idempotent_replay", False):
            response.headers["x-idempotent-replay"] = "true"
        logger.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(elapsed * 1000, 2),
                }
            )
        )
        return response

    @app.get("/health", dependencies=[Depends(authenticate)])
    def health() -> dict[str, Any]:
        try:
            llm.health()
            backend = "ok"
        except Exception as exc:
            backend = f"error: {type(exc).__name__}"
        return {
            "status": "ok" if backend == "ok" else "degraded",
            "backend": backend,
            "model_id": active_model,
            "documents": len({chunk.document_id for chunk in index.chunks}),
            "chunks": len(index.chunks),
        }

    @app.get("/ready", dependencies=[Depends(authenticate)])
    def ready() -> dict[str, Any]:
        llm.health()
        return {"ready": True, "index_chunks": len(index.chunks)}

    @app.get("/models", dependencies=[Depends(authenticate)])
    def models() -> dict[str, Any]:
        return {"active": active_model, "models": registry}

    @app.post("/search", dependencies=[Depends(authenticate)])
    def search(request: SearchRequest) -> dict[str, Any]:
        hits = index.search(
            request.query,
            mode=request.mode,
            top_k=20,
            bm25_k=int(config["retrieval"]["bm25_k"]),
            vector_k=int(config["retrieval"]["vector_k"]),
            rerank_k=int(config["retrieval"]["rerank_k"]),
            rrf_k=int(config["retrieval"]["rrf_k"]),
        )
        unique = deduplicate_documents(hits, 5)
        return {"mode": request.mode, "hits": [hit.as_dict() for hit in unique]}

    @app.post("/ask", dependencies=[Depends(authenticate)])
    async def ask(
        request: AskRequest,
        raw_request: Request,
        idempotency_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        cached = idempotency.get("/ask", idempotency_key)
        if cached is not None:
            raw_request.state.idempotent_replay = True
            return cached
        model_id = request.model_id or active_model
        if model_id not in registry:
            raise HTTPException(status_code=400, detail="Unknown model_id")
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    grounded_answer,
                    llm=llm,
                    index=index,
                    question=request.question,
                    model_id=model_id,
                    mode=request.mode,
                    retrieval_config=config["retrieval"],
                ),
                timeout=request_timeout_s,
            )
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Request timed out") from exc
        if result.error:
            raise HTTPException(status_code=502, detail=result.error)
        payload = result.as_dict()
        idempotency.set("/ask", idempotency_key, payload)
        return payload

    @app.post("/agent", dependencies=[Depends(authenticate)])
    async def agent(
        request: AgentRequest,
        raw_request: Request,
        idempotency_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        cached = idempotency.get("/agent", idempotency_key)
        if cached is not None:
            raw_request.state.idempotent_replay = True
            return cached
        model_id = request.model_id or active_model
        if model_id not in registry:
            raise HTTPException(status_code=400, detail="Unknown model_id")
        controlled = ControlledAgent(
            llm=llm,
            index=index,
            models=registry,
            retrieval_config=config["retrieval"],
            artifact_dir=resolve_path(config, "paths.artifacts") / "service_reports",
            model_id=model_id,
            max_steps=int(config["agent"]["max_steps"]),
            max_tool_calls=int(config["agent"]["max_tool_calls"]),
            tool_timeout_s=float(config["agent"]["tool_timeout_s"]),
            planner_mode=str(config["agent"].get("planner_mode", "llm")),
            planner_timeout_s=float(config["agent"].get("planner_timeout_s", 30)),
        )
        try:
            result = await asyncio.wait_for(
                controlled.run(request.question),
                timeout=request_timeout_s,
            )
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Request timed out") from exc
        payload = result.as_dict()
        failed_tool = next(
            (
                item
                for item in result.tool_trace
                if item["tool"] != "planner" and item["status"] in {"error", "timeout", "skipped"}
            ),
            None,
        )
        if result.error or failed_tool:
            raise HTTPException(
                status_code=502,
                detail={
                    "error": result.error or failed_tool.get("error"),
                    "result": payload,
                },
            )
        idempotency.set("/agent", idempotency_key, payload)
        return payload

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        with counters_lock:
            snapshot = dict(counters)
        lines = [
            "# HELP llm_bench_requests_total HTTP requests handled",
            "# TYPE llm_bench_requests_total counter",
            f"llm_bench_requests_total {snapshot['requests']}",
            "# HELP llm_bench_errors_total HTTP or inference errors",
            "# TYPE llm_bench_errors_total counter",
            f"llm_bench_errors_total {snapshot['errors']}",
            "# HELP llm_bench_rate_limited_total Requests rejected by the local limiter",
            "# TYPE llm_bench_rate_limited_total counter",
            f"llm_bench_rate_limited_total {snapshot['rate_limited']}",
            "# HELP llm_bench_index_chunks Indexed text chunks",
            "# TYPE llm_bench_index_chunks gauge",
            f"llm_bench_index_chunks {len(index.chunks)}",
        ]
        return "\n".join(lines) + "\n"

    return app
