import asyncio
from types import SimpleNamespace

import pytest

from project_assistant import llm
from project_assistant.llm import OpenAICompatibleEmbeddings


class FakeEmbeddingsResource:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        texts = kwargs["input"]
        assert isinstance(texts, list)

        data = [
            SimpleNamespace(index=index, embedding=[float(index + 1)])
            for index in range(len(texts) - 1, -1, -1)
        ]
        return SimpleNamespace(data=data)


class FakeAsyncOpenAI:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsResource()


def test_embeddings_batch_and_preserve_input_order() -> None:
    async def run_test() -> None:
        embeddings = OpenAICompatibleEmbeddings(
            base_url="http://test/v1",
            model="test-model",
            batch_size=2,
        )
        embeddings._client = FakeAsyncOpenAI()

        result = await embeddings.embed_documents(["a", "b", "c"])

        assert result == [[1.0], [2.0], [1.0]]
        assert embeddings._client.embeddings.calls == [
            {"model": "test-model", "input": ["a", "b"]},
            {"model": "test-model", "input": ["c"]},
        ]

    asyncio.run(run_test())


def test_embedding_prefixes_are_applied() -> None:
    async def run_test() -> None:
        embeddings = OpenAICompatibleEmbeddings(
            base_url="http://test/v1",
            model="test-model",
            query_prefix="query: ",
            document_prefix="document: ",
            batch_size=2,
        )
        embeddings._client = FakeAsyncOpenAI()

        await embeddings.embed_documents(["one", "two"])
        query = await embeddings.embed_query("question")

        calls = embeddings._client.embeddings.calls
        assert calls[0]["input"] == ["document: one", "document: two"]
        assert calls[1]["input"] == ["query: question"]
        assert query == [1.0]

    asyncio.run(run_test())


def test_empty_document_list_does_not_call_api() -> None:
    async def run_test() -> None:
        embeddings = OpenAICompatibleEmbeddings(
            base_url="http://test/v1",
            model="test-model",
        )
        embeddings._client = FakeAsyncOpenAI()

        assert await embeddings.embed_documents([]) == []
        assert embeddings._client.embeddings.calls == []

    asyncio.run(run_test())


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        OpenAICompatibleEmbeddings(
            base_url="http://test/v1",
            model="test-model",
            batch_size=0,
        )


def test_get_embedding_client_uses_settings_and_returns_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeAsyncOpenAI()
    client_options: dict[str, str] = {}

    def create_fake_client(**kwargs: str) -> FakeAsyncOpenAI:
        client_options.update(kwargs)
        return fake_client

    monkeypatch.setattr(llm, "AsyncOpenAI", create_fake_client)
    monkeypatch.setattr(
        llm,
        "get_settings",
        lambda: SimpleNamespace(
            llm_base_url="http://embedding.test/v1",
            embedding_model="configured-model",
            embedding_query_prefix="query: ",
            embedding_document_prefix="document: ",
        ),
    )
    llm.get_embedding_client.cache_clear()

    try:
        embeddings = llm.get_embedding_client()

        async def run_test() -> None:
            documents = await embeddings.embed_documents(["alpha", "beta"])
            query = await embeddings.embed_query("question")

            assert documents == [[1.0], [2.0]]
            assert query == [1.0]

        asyncio.run(run_test())

        assert client_options == {
            "base_url": "http://embedding.test/v1",
            "api_key": "not-needed",
        }
        assert fake_client.embeddings.calls == [
            {
                "model": "configured-model",
                "input": ["document: alpha", "document: beta"],
            },
            {"model": "configured-model", "input": ["query: question"]},
        ]
    finally:
        llm.get_embedding_client.cache_clear()
