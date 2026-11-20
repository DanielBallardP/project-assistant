from typing import Literal, Protocol

from openai import AsyncOpenAI
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMClient(Protocol):
    async def chat(self, messages: list[ChatMessage]) -> str: ...


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


class FakeLLM:
    """Deterministic LLM client for tests and local development."""

    def __init__(self, answer: str = "Dies ist eine simulierte Antwort.") -> None:
        self.answer = answer
        self.received: list[ChatMessage] = []

    async def chat(self, messages: list[ChatMessage]) -> str:
        self.received = messages
        return self.answer


def get_llm_client() -> LLMClient:
    from project_assistant.config import get_settings

    settings = get_settings()
    # Not caching with lru_cache because the settings may change during runtime, and we want to reflect those changes.
    return OpenAICompatibleClient(
        base_url=settings.llm_base_url, model=settings.llm_model
    )
