from collections.abc import AsyncIterator

import httpx
import openai
from fastapi.testclient import TestClient

from project_assistant.config import Settings, get_settings
from project_assistant.llm import get_llm_client
from project_assistant.main import create_app, get_http_client


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
    client = TestClient(app)

    response = client.post("/chat/stream", json={"message": "Hallo"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'data: {"token": "Dies"}' in response.text
    assert "event: done\ndata: {}\n\n" in response.text
    assert fake_llm.received[0].role == "system"
    assert fake_llm.received[-1].role == "user"


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
    client = TestClient(app)

    response = client.post("/chat/stream", json={"message": "Hallo"})

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "LLM server unreachable" in response.text
    assert "event: done" not in response.text
