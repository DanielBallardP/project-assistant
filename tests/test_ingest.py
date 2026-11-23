import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from project_assistant.rag.models import Chunk, Document
from scripts import ingest


def make_document(doc_id: str, revision: str) -> Document:
    return Document(
        doc_id=doc_id,
        title="Shared report",
        typ="Report",
        projekt_id=None,
        abteilung="Quality",
        vertraulichkeit="intern",
        revision=revision,
        datum="2026-11-22",
        body=f"Body {revision}",
    )


@pytest.mark.parametrize("reset", [False, True])
def test_ingest_continues_after_parse_failure_and_reports_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    reset: bool,
):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    for name in ("bad.md", "revision-a.md", "revision-b.md"):
        (docs_dir / name).write_text("placeholder", encoding="utf-8")

    documents = {
        "revision-a.md": make_document("DOC-A", "A"),
        "revision-b.md": make_document("DOC-B", "B"),
    }

    def parse(path: Path) -> Document:
        if path.name == "bad.md":
            raise ValueError("invalid front matter")
        return documents[path.name]

    chunks = [
        Chunk(
            chunk_id=f"{document.doc_id}#0",
            doc_id=document.doc_id,
            heading="Shared report",
            text=f"Embedded {document.doc_id}",
            metadata={"doc_id": document.doc_id, "title": document.title},
        )
        for document in documents.values()
    ]
    chunks_by_doc_id = {chunk.doc_id: chunk for chunk in chunks}

    class FakeEmbeddings:
        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            assert texts == [chunk.text for chunk in chunks]
            return [[0.1, 0.2], [0.3, 0.4]]

    class FakeStore:
        def __init__(self) -> None:
            self.upsert_call = None

        def upsert(self, received_chunks, embeddings) -> None:
            self.upsert_call = (received_chunks, embeddings)

    fake_store = FakeStore()
    deleted_collections = []
    monkeypatch.setattr(ingest, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(
        ingest,
        "get_settings",
        lambda: SimpleNamespace(
            chroma_path="chroma",
            collection_name="docs",
            embedding_model="test-model",
        ),
    )
    monkeypatch.setattr(ingest, "parse_document", parse)
    monkeypatch.setattr(
        ingest,
        "chunk_document",
        lambda document: [chunks_by_doc_id[document.doc_id]],
    )
    monkeypatch.setattr(ingest, "get_embedding_client", lambda: FakeEmbeddings())
    monkeypatch.setattr(
        ingest,
        "_delete_collection_if_exists",
        lambda path, name: deleted_collections.append((path, name)),
    )
    monkeypatch.setattr(
        ingest,
        "_create_store",
        lambda path, name, model: fake_store,
    )

    asyncio.run(ingest.main(reset=reset))

    assert deleted_collections == ([("chroma", "docs")] if reset else [])
    received_chunks, embeddings = fake_store.upsert_call
    assert [chunk.doc_id for chunk in received_chunks] == ["DOC-A", "DOC-B"]
    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]

    captured = capsys.readouterr()
    assert "bad.md: failed to parse: invalid front matter" in captured.err
    assert "Documents: 2 (1 failed)" in captured.out
    assert "Chunks: 2" in captured.out
    assert "Current: 1; superseded: 1" in captured.out
    assert "Vector dimension: 2" in captured.out
    assert "Elapsed: " in captured.out
