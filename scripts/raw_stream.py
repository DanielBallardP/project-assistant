"""Usage: uv run python scripts/raw_stream.py "Welchen Umsatz hatte die Firma 2025?" [role]"""
import asyncio
import sys

from openai import AsyncOpenAI

from project_assistant.config import get_settings
from project_assistant.llm import get_embedding_client
from project_assistant.rag.answering import build_messages
from project_assistant.rag.retrieval import retrieve
from project_assistant.rag.store import VectorStore


async def main(question: str, role: str) -> None:
    settings = get_settings()
    store = VectorStore(settings.chroma_path, settings.collection_name, settings.embedding_model)
    hits = await retrieve(question, embeddings=get_embedding_client(), store=store, role=role, k=4)
    messages = build_messages(question, hits)

    client = AsyncOpenAI(base_url=settings.llm_base_url, api_key="not-needed")
    stream = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[m.model_dump() for m in messages],
        stream=True,
    )
    i = 0
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = (delta.model_extra or {}).get("reasoning_content")
        if delta.content or reasoning:
            i += 1
            print(f"{i:>3} content={ascii(delta.content):<40} reasoning={ascii(reasoning)}")


asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "mitarbeiter"))