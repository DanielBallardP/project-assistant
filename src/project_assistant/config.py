from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PA_",
        extra="ignore",
    )

    app_name: str = "Project Assistant"
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str = "google/gemma-4-12b-qat"
    reasoning_level: str = "high"
    answer_system_prompt: str | None = None

    embedding_model: str = "text-embedding-bge-m3"
    embedding_query_prefix: str = ""
    embedding_document_prefix: str = ""

    chroma_path: str = "data/chroma"
    collection_name: str = "docs"

    chunk_min_chars: int = 250
    chunk_max_chars: int = 900


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached settings.

    FastAPI calls this dependency function for every request, so caching avoids
    re-reading .env and re-validating the settings on each call.
    Dependency overrides are keyed by this function object, allowing tests to
    replace get_settings with a custom Settings instance.
    """
    return Settings()
