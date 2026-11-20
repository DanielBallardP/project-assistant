from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from project_assistent.config import Settings, get_settings


class HealthResponse(BaseModel):
    status: str
    app_name: str
    llm_base_url: str
    llm_model: str
    embedding_model: str


def create_app() -> FastAPI:
    app = FastAPI(title="Project Assistant")

    @app.get("/health", response_model=HealthResponse)
    def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
        return HealthResponse(
            status="ok",
            app_name=settings.app_name,
            llm_base_url=settings.llm_base_url,
            llm_model=settings.llm_model,
            embedding_model=settings.embedding_model,
        )

    return app


app = create_app()
