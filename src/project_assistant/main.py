import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

import httpx
import openai
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from project_assistant import prompts
from project_assistant.config import Settings, get_settings
from project_assistant.llm import ChatMessage, LLMClient, StreamDelta, get_llm_client

STATIC_DIR = Path(__file__).parent / "static"


async def get_http_client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        yield client


class HealthResponse(BaseModel):
    status: str
    app_name: str
    llm_base_url: str
    llm_model: str
    embedding_model: str
    reasoning_level: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str


def create_app() -> FastAPI:
    app = FastAPI(title="Project Assistant")

    @app.get("/health")
    def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
        return HealthResponse(
            status="ok",
            app_name=settings.app_name,
            llm_base_url=settings.llm_base_url,
            llm_model=settings.llm_model,
            embedding_model=settings.embedding_model,
            reasoning_level=settings.reasoning_level,
        )

    @app.get("/health/llm")
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

    @app.post("/chat")
    async def chat(
        request: ChatRequest,
        llm: Annotated[LLMClient, Depends(get_llm_client)],
    ) -> ChatResponse:
        messages = [
            ChatMessage(role="system", content=prompts.SYSTEM_PROMPT),
            ChatMessage(role="user", content=request.message),
        ]
        try:
            answer = (await llm.chat(messages)).strip()
        except openai.APIConnectionError as exc:
            raise HTTPException(
                status_code=503, detail=f"LLM server unreachable: {exc}"
            ) from exc
        except openai.APIStatusError as exc:
            raise HTTPException(
                status_code=502, detail=f"LLM server error: {exc}"
            ) from exc
        except openai.APIError as exc:
            raise HTTPException(
                status_code=502, detail=f"LLM server error: {exc}"
            ) from exc
        return ChatResponse(answer=answer)

    @app.post("/chat/stream")
    async def chat_stream(
        request: ChatRequest,
        llm: Annotated[LLMClient, Depends(get_llm_client)],
    ) -> StreamingResponse:
        messages = [
            ChatMessage(role="system", content=prompts.SYSTEM_PROMPT),
            ChatMessage(role="user", content=request.message),
        ]

        def sse(event: str | None, data: dict) -> str:
            """Format a Server-Sent Event (SSE) string.

            If event is None, the default "message" event type is used.
            """
            event_line = f"event: {event}\n" if event else ""
            data_line = f"data: {json.dumps(data, ensure_ascii=False)}\n"
            return f"{event_line}{data_line}\n"

        async def event_source() -> AsyncIterator[str]:
            try:
                async for delta in llm.chat_stream(messages):
                    if isinstance(delta, str):
                        delta = StreamDelta(event="message", text=delta)
                    elif isinstance(delta, tuple) and len(delta) == 2:
                        event, text = delta
                        delta = StreamDelta(event=event, text=text)
                    elif isinstance(delta, dict):
                        delta = StreamDelta(
                            event=delta.get("event", "message"),
                            text=delta.get("text", ""),
                        )
                    if not delta.text:
                        continue
                    if delta.event == "reasoning":
                        yield sse(
                            event="reasoning",
                            data={"reasoning": delta.text},
                        )
                    else:
                        # JSON escaping keeps newline characters inside one SSE data line.
                        yield f"data: {json.dumps({'token': delta.text}, ensure_ascii=False)}\n\n"
            except openai.APIError as exc:
                detail = f"LLM server error: {exc}"
                yield sse(event="error", data={"error": detail})
                return

            yield sse(event="done", data={})

        return StreamingResponse(event_source(), media_type="text/event-stream")

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()
