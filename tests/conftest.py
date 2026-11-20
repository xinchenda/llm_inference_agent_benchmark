from __future__ import annotations

import numpy as np
import pytest

from llm_bench.llm import LLMResponse
from llm_bench.retrieval import Chunk, KnowledgeIndex


class FakeEmbedder:
    def encode(self, texts, **_: object):
        vectors = []
        for text in texts:
            if "timeout" in text.lower() or "tool" in text.lower():
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class FakeReranker:
    def predict(self, pairs, **_: object):
        scores = []
        for query, text in pairs:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            scores.append(float(len(query_words & text_words)))
        return np.asarray(scores, dtype=np.float32)


class FakeLLM:
    def __init__(self, text: str = "The tool stops after two timeouts. [E1]") -> None:
        self.text = text
        self.calls = 0

    def generate(self, *_: object, **__: object) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text=self.text,
            latency_s=0.01,
            ttft_s=0.005,
            prompt_tokens=20,
            completion_tokens=10,
            timings={},
        )

    def health(self):
        return {"data": [{"id": "fake"}]}


@pytest.fixture
def toy_index() -> KnowledgeIndex:
    chunks = [
        Chunk(
            chunk_id="doc-a:1",
            document_id="doc-a",
            title="Agent Policy",
            section="Timeouts",
            text="The agent stops tools after two timeouts and rejects duplicate calls.",
            source_url="https://example.test/a",
            license="MIT",
            ordinal=0,
        ),
        Chunk(
            chunk_id="doc-b:1",
            document_id="doc-b",
            title="Background Jobs",
            section="Queues",
            text="Heavy background jobs should use a queue and multiple processes.",
            source_url="https://example.test/b",
            license="MIT",
            ordinal=0,
        ),
    ]
    index = KnowledgeIndex(
        chunks=chunks,
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        embedding_model="fake",
        reranker_model="fake-reranker",
        device="cpu",
    )
    index._embedder = FakeEmbedder()
    index._reranker = FakeReranker()
    return index


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
