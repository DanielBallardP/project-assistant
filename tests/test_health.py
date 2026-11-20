from fastapi.testclient import TestClient
from project_assistent.config import Settings, get_settings
from project_assistent.main import create_app


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
