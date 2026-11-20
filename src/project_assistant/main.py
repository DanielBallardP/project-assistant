from collections.abc import AsyncIterator
from typing import Annotated

import httpx
import openai
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from project_assistant import prompts
from project_assistant.config import Settings, get_settings
from project_assistant.llm import ChatMessage, LLMClient, get_llm_client


async def get_http_client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        yield client


class HealthResponse(BaseModel):
    status: str
    app_name: str
    llm_base_url: str
    llm_model: str
    embedding_model: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str


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

    @app.get("/health/llm", response_model=list[str])
    async def health_llm(
        settings: Annotated[Settings, Depends(get_settings)],
        client: Annotated[httpx.AsyncClient, Depends(get_http_client)],
    ) -> list[str]:
        try:
            response = await client.get(f"{settings.llm_base_url}/models")
            response.raise_for_status()

        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=503, detail=f"LLM server unreachable: {exc}"
            ) from exc

        return [model["id"] for model in response.json()["data"]]

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        request: ChatRequest,
        llm: Annotated[LLMClient, Depends(get_llm_client)],
    ) -> ChatResponse:
        try:
            llm_messages = [
                {"role": "system", "content": prompts.SYSTEM_PROMPT},
                {"role": "user", "content": request.message},
            ]
            llm_response = await llm.chat([ChatMessage(**msg) for msg in llm_messages])

        except openai.APIConnectionError as exc:
            raise HTTPException(
                status_code=503, detail=f"LLM server unreachable: {exc}"
            ) from exc

        return ChatResponse(answer=llm_response)

    return app


app = create_app()
