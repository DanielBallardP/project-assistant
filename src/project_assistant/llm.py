from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Literal, Protocol

from openai import AsyncOpenAI
from pydantic import BaseModel

from project_assistant.config import get_settings


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMClient(Protocol):
    async def chat(self, messages: list[ChatMessage]) -> str: ...
    def chat_stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]: ...

    # chat_stream is a plain def because it returns an AsyncIterator when called.
    # async def is used for chat(), which awaits one complete result. Declaring
    # chat_stream as async def would instead make the caller await a coroutine.


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

    async def chat_stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=[message.model_dump() for message in messages],
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content
            if content:
                yield content


@lru_cache
def get_llm_client() -> LLMClient:
    """Return a shared client so requests reuse AsyncOpenAI's HTTP connection pool.

    FastAPI calls dependencies for each request; caching avoids constructing a
    new client and connection pool every time. If LLM settings change at
    runtime, clear this cache and the settings cache before creating a client.
    """
    settings = get_settings()

    return OpenAICompatibleClient(
        base_url=settings.llm_base_url, model=settings.llm_model
    )
