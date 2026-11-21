import asyncio
from collections.abc import AsyncIterator

import pytest

from project_assistant.llm import ChatMessage, StreamDelta


class FakeLLM:
    def __init__(self, answer: str = "Dies ist eine simulierte Antwort.") -> None:
        self.answer = answer
        self.received: list[ChatMessage] = []

    async def chat(self, messages: list[ChatMessage]) -> str:
        self.received = messages
        return self.answer

    async def chat_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncIterator[StreamDelta]:
        self.received = messages
        for word in self.answer.split():
            yield StreamDelta(event="message", text=word)
            await asyncio.sleep(0)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(answer="  Dies ist eine simulierte Antwort.  \n")
