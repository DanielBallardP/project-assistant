import asyncio

from project_assistant.rag.store import Hit
from scripts import search


def test_parse_args_reads_query_role_and_k(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["search.py", "Frage", "--role", "mitarbeiter", "--k", "4"],
    )

    arguments = search.parse_args()

    assert (arguments.query, arguments.role, arguments.k) == (
        "Frage",
        "mitarbeiter",
        4,
    )


def test_main_prints_doc_id_score_and_first_80_characters(monkeypatch, capsys):
    settings = type(
        "Settings",
        (),
        {
            "chroma_path": "data/chroma",
            "collection_name": "docs",
            "embedding_model": "test-model",
        },
    )()
    hit = Hit(
        chunk_id="DOC-1#0",
        doc_id="DOC-1",
        title="Report",
        text="x" * 100,
        score=0.876543,
        metadata={},
    )
    received_store = {}

    class FakeStore:
        def __init__(self, *, path: str, collection: str, embedding_model: str):
            received_store.update(
                path=path,
                collection=collection,
                embedding_model=embedding_model,
            )

    class FakeEmbeddings:
        pass

    async def fake_retrieve(query, *, embeddings, store, role, k):
        assert query == "Frage"
        assert isinstance(embeddings, FakeEmbeddings)
        assert role == "mitarbeiter"
        assert k == 4
        return [hit]

    monkeypatch.setattr(search, "get_settings", lambda: settings)
    monkeypatch.setattr(search, "get_embedding_client", FakeEmbeddings)
    monkeypatch.setattr(search, "VectorStore", FakeStore)
    monkeypatch.setattr(search, "retrieve", fake_retrieve)

    asyncio.run(search.main("Frage", "mitarbeiter", 4))

    assert received_store == {
        "path": "data/chroma",
        "collection": "docs",
        "embedding_model": "test-model",
    }
    assert capsys.readouterr().out == f"DOC-1\t0.8765\t{'x' * 80}\n"
