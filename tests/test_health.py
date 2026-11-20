from collections.abc import AsyncIterator

import httpx
from fastapi.testclient import TestClient

from project_assistant.config import Settings, get_settings
from project_assistant.llm import FakeLLM, get_llm_client
from project_assistant.main import create_app, get_http_client


def test_health_returns_ok():
    app = create_app()
    test_settings = Settings(
        app_name="Test Project Assistant",
        llm_base_url="http://testserver:8000",
        llm_model="test-model",
        embedding_model="test-embedding-model",
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


def test_chat_returns_answer_from_llm():
    fake = FakeLLM(answer="Dies ist eine simulierte Antwort.")
    app = create_app()
    app.dependency_overrides[get_llm_client] = lambda: fake
    client = TestClient(app)

    response = client.post("/chat", json={"message": "Hallo, wie geht es dir?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "Dies ist eine simulierte Antwort."}
    assert fake.received[0].role == "system"
    maxMessageNumber = len(fake.received)
    assert fake.received[maxMessageNumber - 1].role == "user"


def test_chat_with_empty_message_returns_422():
    fake = FakeLLM(answer="Dies ist eine simulierte Antwort.")
    app = create_app()
    app.dependency_overrides[get_llm_client] = lambda: fake
    client = TestClient(app)

    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422
