"""Usage: uv run python scripts/smoke_llm.py [--runs 5]"""
import argparse
import asyncio
import re

from openai import AsyncOpenAI

from project_assistant.config import get_settings
from project_assistant.rag.answering import ANSWER_SYSTEM

ABSTAIN = "Dazu finde ich nichts in den Dokumenten."
CONTEXT = (
    "[BP-2043-1105 | Baubesprechung P-2043 vom 05.11.2026 | Revision A | 2026-11-05]\n"
    "Die Rohrlieferung verzögert sich um drei Wochen."
)
QUESTION = "Welchen Umsatz hatte die Firma 2025?"
SUSPICIOUS = re.compile(r"</?think>|</?thought>|[\U0001D400-\U0001D7FF]|<\|[a-z_]+\|>")


async def one_run(client: AsyncOpenAI, model: str) -> str:
    stream = await client.chat.completions.create(
        model=model,
        stream=True,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"Kontext:\n{CONTEXT}\n\nFrage:\n{QUESTION}"},
        ],
    )
    parts: list[str] = []
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            parts.append(chunk.choices[0].delta.content)
    return "".join(parts)


async def main(runs: int) -> None:
    settings = get_settings()
    client = AsyncOpenAI(base_url=settings.llm_base_url, api_key="not-needed")
    clean = exact = 0
    for n in range(1, runs + 1):
        text = await one_run(client, settings.llm_model)
        is_clean = not SUSPICIOUS.search(text)
        is_exact = text.strip() == ABSTAIN
        clean += is_clean
        exact += is_exact
        print(f"run {n}: clean={is_clean} exact_abstain={is_exact} raw={ascii(text)[:90]}")
    print(f"\n{settings.llm_model}: clean {clean}/{runs}, exact abstain {exact}/{runs}")


parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=5)
asyncio.run(main(parser.parse_args().runs))