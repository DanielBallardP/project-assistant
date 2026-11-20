from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PA_",
        extra="ignore",
    )

    app_name: str = "Project Assistant"
    llm_base_url: str = "http://localhost:8000"
    llm_model: str = "text-embedding-bge-m3"
    embedding_model: str = "text-embedding-bge-m3"


@lru_cache
def get_settings() -> Settings:
    return Settings()
