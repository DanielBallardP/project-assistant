from collections.abc import AsyncIterator
from importlib import import_module

import httpx
import openai
from fastapi.testclient import TestClient

from project_assistant.config import Settings, get_settings
from project_assistant.llm import StreamDelta, get_embedding_client, get_llm_client
from project_assistant.main import create_app, get_http_client, get_vector_store_factory
from project_assistant.rag.store import Hit

main_module = import_module("project_assistant.main")


class FakeEmbeddings:
    async def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2]


class FakeVectorStore:
    def __init__(self) -> None:
        self.query_args: dict | None = None

    def query(self, *, embedding: list[float], k: int, where: dict) -> list[Hit]:
        self.query_args = {"embedding": embedding, "k": k, "where": where}
        return [
            Hit(
                chunk_id="DOC-1#0",
                doc_id="DOC-1",
                title="Test document",
                text="Test context.",
                score=0.9,
                metadata={
                    "doc_id": "DOC-1",
                    "title": "Test document",
                    "revision": "A",
                    "datum": "21.10.2026",
                    "vertraulichkeit": "intern",
                    "is_current": True,
                },
            )
        ]


def install_fake_rag(app):
    store = FakeVectorStore()
    app.dependency_overrides[get_embedding_client] = FakeEmbeddings
    app.dependency_overrides[get_vector_store_factory] = lambda: lambda: store
    return store


def test_health_returns_ok():
    app = create_app()
    test_settings = Settings(
        app_name="Test Project Assistant",
        llm_base_url="http://testserver:8000",
        llm_model="test-model",
        embedding_model="test-embedding-model",
        reasoning_level="test-reasoning-level",
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app_name": "Test Project Assistant",
        "llm_base_url": "http://testserver:8000",
        "llm_model": "test-model",
        "embedding_model": "test-embedding-model",
        "reasoning_level": "test-reasoning-level",
    }


def test_health_llm_returns_model_ids_without_server():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url == "http://llm.test/v1/models"
        return httpx.Response(
            200,
            json={"data": [{"id": "test-model-1"}, {"id": "test-model-2"}]},
        )

    async def override_http_client() -> AsyncIterator[httpx.AsyncClient]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            yield client

    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(
        llm_base_url="http://llm.test/v1"
    )
    app.dependency_overrides[get_http_client] = override_http_client
    client = TestClient(app)

    response = client.get("/health/llm")

    assert response.status_code == 200
    assert response.json() == ["test-model-1", "test-model-2"]


def test_chat_returns_answer_from_llm(fake_llm):
    app = create_app()
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    client = TestClient(app)

    response = client.post("/chat", json={"message": "Hallo, wie geht es dir?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "Dies ist eine simulierte Antwort."}
    assert fake_llm.received[0].role == "system"
    assert fake_llm.received[-1].role == "user"


def test_chat_with_empty_message_returns_422(fake_llm):
    app = create_app()
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    client = TestClient(app)

    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_chat_stream_returns_sse_events(fake_llm):
    app = create_app()
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    store = install_fake_rag(app)
    client = TestClient(app)

    response = client.post("/chat/stream", json={"message": "Hallo"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: sources" in response.text
    assert '"doc_id": "DOC-1"' in response.text
    assert 'data: {"token": "Dies"}' in response.text
    assert "event: done\ndata: {}\n\n" in response.text
    assert store.query_args == {
        "embedding": [0.1, 0.2],
        "k": 4,
        "where": {
            "$and": [
                {"vertraulichkeit": {"$in": ["intern"]}},
                {"is_current": True},
            ]
        },
    }
    assert fake_llm.received[0].role == "system"
    assert "Test context." in fake_llm.received[-1].content
    assert fake_llm.received[-1].role == "user"


def test_chat_stream_uses_demo_role_header(fake_llm):
    app = create_app()
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    store = install_fake_rag(app)
    client = TestClient(app)

    response = client.post(
        "/chat/stream",
        json={"message": "Hallo"},
        headers={"X-Demo-Role": "projektleitung"},
    )

    assert response.status_code == 200
    assert store.query_args is not None
    assert store.query_args["where"] == {
        "$and": [
            {"vertraulichkeit": {"$in": ["intern", "vertraulich"]}},
            {"is_current": True},
        ]
    }


def test_chat_stream_rejects_unknown_demo_role(fake_llm):
    app = create_app()
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    store = install_fake_rag(app)
    client = TestClient(app)

    response = client.post(
        "/chat/stream",
        json={"message": "Hallo"},
        headers={"X-Demo-Role": "admin"},
    )

    assert response.status_code == 422
    assert store.query_args is None
    assert fake_llm.received == []


def test_chat_stream_emits_generic_error_event_when_retrieval_fails(
    fake_llm, monkeypatch
):
    async def failing_retrieve(*args, **kwargs):
        raise RuntimeError("private vector-store details")

    monkeypatch.setattr(main_module, "retrieve", failing_retrieve)
    app = create_app()
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    install_fake_rag(app)
    client = TestClient(app)

    response = client.post("/chat/stream", json={"message": "Hallo"})

    assert response.status_code == 200
    assert (
        'event: error\ndata: {"error": "Document retrieval failed. Please try again."}'
        in response.text
    )
    assert "private vector-store details" not in response.text
    assert "event: sources" not in response.text
    assert "event: done" not in response.text
    assert fake_llm.received == []


def test_chat_stream_maps_store_creation_failure_to_generic_error_event(fake_llm):
    def failing_store_factory():
        raise RuntimeError("private collection configuration")

    app = create_app()
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    app.dependency_overrides[get_embedding_client] = FakeEmbeddings
    app.dependency_overrides[get_vector_store_factory] = lambda: failing_store_factory
    client = TestClient(app)

    response = client.post("/chat/stream", json={"message": "Hallo"})

    assert response.status_code == 200
    assert "Document retrieval failed. Please try again." in response.text
    assert "private collection configuration" not in response.text
    assert "event: sources" not in response.text
    assert "event: done" not in response.text
    assert fake_llm.received == []


def test_root_returns_html(fake_llm):
    app = create_app()
    app.dependency_overrides[get_llm_client] = lambda: fake_llm
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_chat_stream_returns_reasoning_events():
    class ReasoningLLM:
        async def chat(self, messages):
            return ""

        async def chat_stream(self, messages):
            yield StreamDelta(event="reasoning", text="Ich prüfe")
            yield StreamDelta(event="message", text="Die Antwort")

    app = create_app()
    app.dependency_overrides[get_llm_client] = lambda: ReasoningLLM()
    install_fake_rag(app)
    client = TestClient(app)

    response = client.post("/chat/stream", json={"message": "Hallo"})

    assert response.status_code == 200
    assert 'event: reasoning\ndata: {"reasoning": "Ich prüfe"}' in response.text
    assert 'data: {"token": "Die Antwort"}' in response.text


def test_chat_stream_emits_error_event_when_llm_fails():
    class FailingStreamLLM:
        async def chat(self, messages):
            return ""

        async def chat_stream(self, messages):
            raise openai.APIConnectionError(
                request=httpx.Request("POST", "http://llm.test/v1/chat/completions")
            )
            yield ""

    app = create_app()
    app.dependency_overrides[get_llm_client] = lambda: FailingStreamLLM()
    install_fake_rag(app)
    client = TestClient(app)

    response = client.post("/chat/stream", json={"message": "Hallo"})

    assert response.status_code == 200
    assert "event: error" in response.text
    assert (
        'data: {"error": "The language model could not generate a response."}'
        in response.text
    )
    assert "Connection error." not in response.text
    assert "event: done" not in response.text
