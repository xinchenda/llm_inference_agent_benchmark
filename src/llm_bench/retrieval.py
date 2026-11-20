from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+-]*")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(text):
        normalized = token.lower().rstrip(".,;:!?")
        if normalized and normalized not in STOP_WORDS:
            tokens.append(normalized)
    return tokens


def stem_term(term: str) -> str:
    """Small deterministic stemmer for overlap and negation checks."""
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 4 and term.endswith("ing"):
        return term[:-3]
    if len(term) > 3 and term.endswith("ed"):
        return term[:-2]
    if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    section: str
    text: str
    source_url: str
    license: str
    ordinal: int


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    document_id: str
    title: str
    section: str
    text: str
    source_url: str
    license: str
    score: float
    rank: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _paragraphs(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line.rstrip())
        elif current:
            paragraphs.append("\n".join(current).strip())
            current = []
    if current:
        paragraphs.append("\n".join(current).strip())
    return paragraphs


def _window_paragraphs(
    paragraphs: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if not paragraphs:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for paragraph in paragraphs:
        if current and current_length + len(paragraph) + 2 > chunk_size:
            chunks.append("\n\n".join(current))
            overlap: list[str] = []
            overlap_length = 0
            for previous in reversed(current):
                if overlap_length + len(previous) > chunk_overlap:
                    break
                overlap.insert(0, previous)
                overlap_length += len(previous)
            current = overlap
            current_length = overlap_length
        current.append(paragraph)
        current_length += len(paragraph) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_markdown(
    path: Path,
    metadata: dict[str, Any],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    document_title = metadata.get("title") or path.stem
    section = "Introduction"
    sections: list[tuple[str, list[str]]] = []
    body: list[str] = []
    seen_title = False
    for line in lines:
        heading = HEADING_PATTERN.match(line)
        if heading:
            title = heading.group(2).strip()
            if not seen_title and len(heading.group(1)) == 1:
                document_title = title
                seen_title = True
                continue
            if body:
                sections.append((section, body))
                body = []
            section = title
            continue
        body.append(line)
    if body:
        sections.append((section, body))

    document_id = metadata.get("document_id") or path.stem
    chunks: list[Chunk] = []
    for section_name, section_lines in sections:
        for text in _window_paragraphs(_paragraphs(section_lines), chunk_size, chunk_overlap):
            if len(text.strip()) < 80:
                continue
            ordinal = len(chunks)
            digest = hashlib.sha1(
                f"{document_id}:{section_name}:{ordinal}:{text}".encode()
            ).hexdigest()[:12]
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}:{digest}",
                    document_id=document_id,
                    title=str(document_title),
                    section=section_name,
                    text=text,
                    source_url=str(metadata.get("source_url", "")),
                    license=str(metadata.get("license", "Project-authored")),
                    ordinal=ordinal,
                )
            )
    return chunks


class KnowledgeIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        embedding_model: str,
        reranker_model: str | None,
        device: str = "cpu",
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk count and embedding count differ")
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.device = device
        self._embedder = None
        self._reranker = None
        self._model_lock = threading.RLock()
        self._bm25 = BM25Okapi([tokenize(chunk.text) for chunk in chunks])

    @property
    def embedder(self):
        with self._model_lock:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(
                    self.embedding_model_name,
                    device=self.device,
                )
        return self._embedder

    @property
    def reranker(self):
        if self.reranker_model_name is None:
            return None
        with self._model_lock:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(self.reranker_model_name, device=self.device)
        return self._reranker

    def _dense_ranking(self, query: str) -> list[tuple[int, float]]:
        with self._model_lock:
            query_vector = self.embedder.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
        scores = self.embeddings @ query_vector.astype(np.float32)
        order = np.argsort(-scores)
        return [(int(index), float(scores[index])) for index in order]

    def _sparse_ranking(self, query: str) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)
        return [(int(index), float(scores[index])) for index in order]

    def search(
        self,
        query: str,
        mode: str = "hybrid_rerank",
        top_k: int = 5,
        bm25_k: int = 20,
        vector_k: int = 20,
        rerank_k: int = 8,
        rrf_k: int = 60,
    ) -> list[Hit]:
        if not self.chunks:
            return []
        dense = self._dense_ranking(query)
        if mode == "vector":
            ranked = dense[:top_k]
            return [self._hit(index, score, rank) for rank, (index, score) in enumerate(ranked, 1)]

        sparse = self._sparse_ranking(query)
        fusion: dict[int, float] = {}
        for rank, (index, _) in enumerate(sparse[:bm25_k], 1):
            fusion[index] = fusion.get(index, 0.0) + 1.0 / (rrf_k + rank)
        for rank, (index, _) in enumerate(dense[:vector_k], 1):
            fusion[index] = fusion.get(index, 0.0) + 1.0 / (rrf_k + rank)
        fused = sorted(fusion.items(), key=lambda item: (-item[1], item[0]))

        if mode == "hybrid":
            return [
                self._hit(index, score, rank)
                for rank, (index, score) in enumerate(fused[:top_k], 1)
            ]
        if mode != "hybrid_rerank":
            raise ValueError(f"Unknown retrieval mode: {mode}")
        if self.reranker is None:
            raise RuntimeError("Reranking requested without a reranker model")
        candidates = fused[: max(rerank_k, top_k)]
        pairs = [(query, self.chunks[index].text) for index, _ in candidates]
        with self._model_lock:
            scores = self.reranker.predict(pairs, show_progress_bar=False)
        reranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0][0]),
        )
        return [
            self._hit(candidate[0], float(score), rank)
            for rank, (candidate, score) in enumerate(reranked[:top_k], 1)
        ]

    def _hit(self, index: int, score: float, rank: int) -> Hit:
        chunk = self.chunks[index]
        return Hit(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            title=chunk.title,
            section=chunk.section,
            text=chunk.text,
            source_url=chunk.source_url,
            license=chunk.license,
            score=score,
            rank=rank,
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "index_meta.json").write_text(
            json.dumps(
                {
                    "embedding_model": self.embedding_model_name,
                    "reranker_model": self.reranker_model_name,
                    "device": self.device,
                    "chunks": len(self.chunks),
                    "dimensions": int(self.embeddings.shape[1]),
                    "documents": len({chunk.document_id for chunk in self.chunks}),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path, device: str = "cpu") -> KnowledgeIndex:
        chunks = [
            Chunk(**json.loads(line))
            for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        embeddings = np.load(directory / "embeddings.npy")
        meta = json.loads((directory / "index_meta.json").read_text(encoding="utf-8"))
        return cls(
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=meta["embedding_model"],
            reranker_model=meta.get("reranker_model"),
            device=device,
        )


def build_index(
    corpus_dir: Path,
    index_dir: Path,
    manifest_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    reranker_model: str | None,
    device: str,
) -> KnowledgeIndex:
    manifest_records: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        manifest_records = {
            item["file"]: item for item in json.loads(manifest_path.read_text(encoding="utf-8"))
        }
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        metadata = manifest_records.get(
            path.name,
            {
                "document_id": path.stem,
                "title": path.stem.replace("__", " ").replace("_", " ").title(),
                "source_url": "",
                "license": "Project-authored",
            },
        )
        chunks.extend(
            chunk_markdown(
                path,
                metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    if len({chunk.document_id for chunk in chunks}) < 50:
        raise RuntimeError("Corpus must contain at least 50 documents")

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(embedding_model, device=device)
    embeddings = embedder.encode(
        [chunk.text for chunk in chunks],
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    index = KnowledgeIndex(
        chunks=chunks,
        embeddings=np.asarray(embeddings),
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        device=device,
    )
    index.save(index_dir)
    return index


def deduplicate_documents(hits: Iterable[Hit], limit: int | None = None) -> list[Hit]:
    seen: set[str] = set()
    result: list[Hit] = []
    for hit in hits:
        if hit.document_id in seen:
            continue
        seen.add(hit.document_id)
        result.append(hit)
        if limit and len(result) >= limit:
            break
    return result
