from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Protocol

from openai import AsyncOpenAI
from pydantic import BaseModel

from project_assistant.config import get_settings


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class StreamDelta:
    event: Literal["message", "reasoning"]
    text: str


class EmbeddingClient(Protocol):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


class OpenAICompatibleEmbeddings:
    def __init__(
        self,
        base_url: str,
        model: str,
        query_prefix: str = "",
        document_prefix: str = "",
        api_key: str = "not-needed",
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._batch_size = batch_size

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float] | None] = [None] * len(texts)

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = await self._client.embeddings.create(
                model=self._model,
                input=batch,
            )

            for item in response.data:
                index = item.index
                if index < 0 or index >= len(batch):
                    raise RuntimeError(
                        f"Embedding response index {index} is outside the batch"
                    )
                vectors[start + index] = list(item.embedding)

        if any(vector is None for vector in vectors):
            raise RuntimeError(
                "Embedding response did not contain a vector for every input"
            )

        return [vector for vector in vectors if vector is not None]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed([f"{self._document_prefix}{text}" for text in texts])

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed([f"{self._query_prefix}{text}"])
        return vectors[0]


@lru_cache
def get_embedding_client() -> EmbeddingClient:
    """Return a shared embedding client so requests reuse AsyncOpenAI's HTTP connection pool.

    FastAPI calls dependencies for each request; caching avoids constructing a
    new client and connection pool every time. If LLM settings change at
    runtime, clear this cache and the settings cache before creating a client.
    """
    settings = get_settings()

    return OpenAICompatibleEmbeddings(
        base_url=settings.llm_base_url,
        model=settings.embedding_model,
        query_prefix=settings.embedding_query_prefix,
        document_prefix=settings.embedding_document_prefix,
    )


class LLMClient(Protocol):
    async def chat(self, messages: list[ChatMessage]) -> str: ...
    def chat_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncIterator[StreamDelta]: ...


class OpenAICompatibleClient:
    """Talks to any OpenAI-compatible server (LM Studio now, Azure OpenAI later)."""

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed") -> None:
        # LM Studio ignores the key, but the SDK requires a non-empty one.
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    async def chat(self, messages: list[ChatMessage]) -> str:
        response = await self._client.chat.completions.create(
            model=self._model, messages=[message.model_dump() for message in messages]
        )

        return response.choices[0].message.content or ""

    async def chat_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncIterator[StreamDelta]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=[message.model_dump() for message in messages],
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if not reasoning:
                reasoning = getattr(delta, "reasoning", None)
            if reasoning:
                yield StreamDelta(event="reasoning", text=reasoning)

            content = delta.content
            if content:
                yield StreamDelta(event="message", text=content)


@lru_cache
def get_llm_client() -> LLMClient:
    """Return a shared client so requests reuse AsyncOpenAI's HTTP connection pool.

    FastAPI calls dependencies for each request; caching avoids constructing a
    new client and connection pool every time. If LLM settings change at
    runtime, clear this cache and the settings cache before creating a client.
    """
    settings = get_settings()

    return OpenAICompatibleClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )
