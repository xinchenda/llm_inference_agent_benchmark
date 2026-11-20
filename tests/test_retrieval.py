from __future__ import annotations

from pathlib import Path

from llm_bench.retrieval import chunk_markdown, deduplicate_documents


def test_chunk_long_document_preserves_section_and_overlap(tmp_path: Path) -> None:
    path = tmp_path / "long.md"
    paragraphs = "\n\n".join(
        f"Paragraph {index}. " + ("technical evidence " * 30) for index in range(40)
    )
    path.write_text(f"# Long Guide\n\n## Details\n\n{paragraphs}\n", encoding="utf-8")
    chunks = chunk_markdown(
        path,
        metadata={
            "document_id": "long-doc",
            "title": "Long Guide",
            "source_url": "https://example.test/long",
            "license": "MIT",
        },
        chunk_size=700,
        chunk_overlap=120,
    )
    assert len(chunks) > 5
    assert all(chunk.document_id == "long-doc" for chunk in chunks)
    assert all(chunk.section == "Details" for chunk in chunks)
    assert len([chunk for chunk in chunks if "Paragraph 39" in chunk.text]) >= 1


def test_retrieval_returns_document_level_hits(toy_index) -> None:
    hits = toy_index.search("tool timeout policy", mode="hybrid_rerank", top_k=5)
    unique = deduplicate_documents(hits, limit=5)
    assert unique[0].document_id == "doc-a"
    assert unique[0].section == "Timeouts"
