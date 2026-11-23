import importlib
import sys
from types import SimpleNamespace

import pytest

from project_assistant.rag.models import Chunk


class FakeCollection:
    def __init__(self, name: str, metadata: dict) -> None:
        self.name = name
        self.metadata = metadata
        self.id = "collection-id"
        self.upsert_calls: list[dict] = []
        self.query_calls: list[dict] = []
        self.query_result: dict = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    def upsert(self, **kwargs) -> None:
        self.upsert_calls.append(kwargs)

    def query(self, **kwargs) -> dict:
        self.query_calls.append(kwargs)
        return self.query_result


class FakeClient:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.created: list[FakeCollection] = []
        self.collection = FakeCollection("docs", {})
        self.existing_metadata: dict | None = None

    def get_or_create_collection(self, name: str, metadata: dict) -> FakeCollection:
        self.collection.name = name
        self.collection.metadata = self.existing_metadata or metadata
        return self.collection

    def delete_collection(self, name: str) -> None:
        self.deleted.append(name)

    def create_collection(self, name: str, metadata: dict) -> FakeCollection:
        collection = FakeCollection(name, metadata)
        self.created.append(collection)
        return collection


@pytest.fixture
def vector_store_env(monkeypatch: pytest.MonkeyPatch):
    client = FakeClient()
    fake_chromadb = SimpleNamespace(
        PersistentClient=lambda *, path: client,
    )
    module_name = "project_assistant.rag.store"
    previous_module = sys.modules.pop(module_name, None)
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb)
    store_module = importlib.import_module(module_name)

    store = store_module.VectorStore(
        path="unused",
        collection="docs",
        embedding_model="test-model",
    )
    yield store, client

    sys.modules.pop(module_name, None)
    if previous_module is not None:
        sys.modules[module_name] = previous_module


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="DOC-1",
        heading="Heading",
        text=text,
        metadata={"doc_id": "DOC-1", "title": "Title"},
    )


def test_reset_recreates_collection_with_metadata_and_new_handle(vector_store_env):
    store, client = vector_store_env

    store.reset()

    assert client.deleted == ["docs"]
    assert len(client.created) == 1
    assert client.created[0].metadata == {
        "hnsw:space": "cosine",
        "embedding_model": "test-model",
    }
    assert store._col is client.created[0]


def test_upsert_sends_ids_embeddings_text_and_metadata(vector_store_env):
    store, client = vector_store_env
    chunks = [make_chunk("DOC-1#0", "First"), make_chunk("DOC-1#1", "Second")]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    store.upsert(chunks, embeddings)

    assert client.collection.upsert_calls == [
        {
            "ids": ["DOC-1#0", "DOC-1#1"],
            "embeddings": embeddings,
            "documents": ["First", "Second"],
            "metadatas": [
                {"doc_id": "DOC-1", "title": "Title"},
                {"doc_id": "DOC-1", "title": "Title"},
            ],
        }
    ]


def test_upsert_rejects_mismatched_chunk_and_embedding_counts(vector_store_env):
    store, client = vector_store_env

    with pytest.raises(ValueError, match="number of chunks"):
        store.upsert([make_chunk("DOC-1#0", "First")], [])

    assert client.collection.upsert_calls == []


@pytest.mark.parametrize(
    "stored_metadata",
    [
        {"hnsw:space": "cosine", "embedding_model": "different-model"},
        {"hnsw:space": "cosine"},
    ],
)
def test_constructor_rejects_missing_or_different_embedding_model(
    vector_store_env, stored_metadata
):
    store, client = vector_store_env
    client.existing_metadata = stored_metadata

    with pytest.raises(ValueError, match="Rebuild the collection with --reset"):
        store.__class__(
            path="unused",
            collection="docs",
            embedding_model="test-model",
        )


def test_query_maps_results_to_hits_and_converts_distances(vector_store_env):
    store, client = vector_store_env
    client.collection.query_result = {
        "ids": [["DOC-1#0", "DOC-1#1"]],
        "documents": [["First", "Second"]],
        "metadatas": [
            [
                {"doc_id": "DOC-1", "title": "Title"},
                {"doc_id": "DOC-2", "title": "Other title"},
            ]
        ],
        "distances": [[0.0, 0.25]],
    }

    hits = store.query(
        embedding=[0.1, 0.2],
        k=2,
        where={"is_current": True},
    )

    assert client.collection.query_calls == [
        {
            "query_embeddings": [[0.1, 0.2]],
            "n_results": 2,
            "where": {"is_current": True},
            "include": ["documents", "metadatas", "distances"],
        }
    ]
    assert [
        (hit.chunk_id, hit.doc_id, hit.title, hit.text, hit.score) for hit in hits
    ] == [
        ("DOC-1#0", "DOC-1", "Title", "First", 1.0),
        ("DOC-1#1", "DOC-2", "Other title", "Second", 0.75),
    ]
    assert hits[0].metadata == {"doc_id": "DOC-1", "title": "Title"}


def test_query_returns_empty_list_when_collection_has_no_matches(vector_store_env):
    store, _client = vector_store_env

    assert store.query(embedding=[0.1, 0.2], k=4, where=None) == []


def test_query_rejects_non_positive_result_count(vector_store_env):
    store, client = vector_store_env

    with pytest.raises(ValueError, match="k must be greater than zero"):
        store.query(embedding=[0.1, 0.2], k=0, where=None)

    assert client.collection.query_calls == []
