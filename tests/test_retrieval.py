import asyncio

import pytest

from project_assistant.rag.retrieval import retrieve
from project_assistant.rag.store import Hit


class FakeEmbeddings:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [0.1, 0.2]


class FakeStore:
    def __init__(self) -> None:
        self.query_calls: list[dict] = []
        self.hits = [
            Hit(
                chunk_id="DOC-1#0",
                doc_id="DOC-1",
                title="Report",
                text="Relevant text",
                score=0.9,
                metadata={"vertraulichkeit": "intern"},
            )
        ]

    def query(self, *, embedding: list[float], k: int, where: dict) -> list[Hit]:
        self.query_calls.append({"embedding": embedding, "k": k, "where": where})
        return self.hits


@pytest.mark.parametrize(
    ("role", "include_superseded", "expected_where"),
    [
        (
            "mitarbeiter",
            False,
            {
                "$and": [
                    {"vertraulichkeit": {"$in": ["intern"]}},
                    {"is_current": True},
                ]
            },
        ),
        (
            "projektleitung",
            False,
            {
                "$and": [
                    {"vertraulichkeit": {"$in": ["intern", "vertraulich"]}},
                    {"is_current": True},
                ]
            },
        ),
        (
            "projektleitung",
            True,
            {"vertraulichkeit": {"$in": ["intern", "vertraulich"]}},
        ),
    ],
)
def test_retrieve_embeds_query_and_applies_access_and_revision_filters(
    role: str, include_superseded: bool, expected_where: dict
):
    embeddings = FakeEmbeddings()
    store = FakeStore()

    hits = asyncio.run(
        retrieve(
            "What is the report status?",
            embeddings=embeddings,
            store=store,
            role=role,
            k=3,
            include_superseded=include_superseded,
        )
    )

    assert embeddings.queries == ["What is the report status?"]
    assert store.query_calls == [
        {"embedding": [0.1, 0.2], "k": 3, "where": expected_where}
    ]
    assert hits == store.hits


def test_retrieve_rejects_unknown_role_before_embedding_or_querying():
    embeddings = FakeEmbeddings()
    store = FakeStore()

    with pytest.raises(ValueError, match="Unknown role"):
        asyncio.run(
            retrieve(
                "Any query",
                embeddings=embeddings,
                store=store,
                role="unknown",
            )
        )

    assert embeddings.queries == []
    assert store.query_calls == []
