import json
import logging
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Annotated, Literal

import httpx
import openai
from chromadb.errors import ChromaError
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from project_assistant import prompts
from project_assistant.config import Settings, get_settings
from project_assistant.llm import (
    ChatMessage,
    EmbeddingClient,
    LLMClient,
    StreamDelta,
    get_embedding_client,
    get_llm_client,
)
from project_assistant.rag.answering import build_messages, to_sources
from project_assistant.rag.retrieval import retrieve
from project_assistant.rag.store import VectorStore

STATIC_DIR = Path(__file__).parent / "static"
logger = logging.getLogger(__name__)


async def get_http_client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        yield client


def get_vector_store_factory(
    settings: Annotated[Settings, Depends(get_settings)],
) -> Callable[[], VectorStore]:
    return lambda: VectorStore(
        settings.chroma_path, settings.collection_name, settings.embedding_model
    )


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
        embeddings: Annotated[EmbeddingClient, Depends(get_embedding_client)],
        store_factory: Annotated[
            Callable[[], VectorStore], Depends(get_vector_store_factory)
        ],
        # Temporary only: a client-sent role is not security. Replace with the
        # role from the Entra token at later stage.
        role: Annotated[
            Literal["mitarbeiter", "projektleitung"],
            Header(alias="X-Demo-Role"),
        ] = "mitarbeiter",
    ) -> StreamingResponse:
        def sse(event: str | None, data: dict) -> str:
            """Format a Server-Sent Event (SSE) string.

            If event is None, the default "message" event type is used.
            """
            event_line = f"event: {event}\n" if event else ""
            data_line = f"data: {json.dumps(data, ensure_ascii=False)}\n"
            return f"{event_line}{data_line}\n"

        async def event_source() -> AsyncIterator[str]:
            try:
                store = store_factory()
                hits = await retrieve(
                    request.message,
                    embeddings=embeddings,
                    store=store,
                    role=role,
                )
            except (
                ChromaError,
                IndexError,
                KeyError,
                openai.APIError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                logger.exception("Document retrieval failed")
                yield sse(
                    event="error",
                    data={"error": "Document retrieval failed. Please try again."},
                )
                return

            messages = build_messages(request.message, hits, get_settings().answer_system_prompt)
            yield sse(
                event="sources",
                data={"sources": [source.model_dump() for source in to_sources(hits)]},
            )

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
            except openai.APIError:
                logger.exception("LLM streaming failed")
                yield sse(
                    event="error",
                    data={"error": "The language model could not generate a response."},
                )
                return

            yield sse(event="done", data={})

        return StreamingResponse(event_source(), media_type="text/event-stream")

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()
