"""Usage: uv run python scripts/benchmark.py <model-id>"""

import asyncio
import json
import sys
import time

from openai import AsyncOpenAI

BASE_URL = "http://localhost:1234/v1"

PROMPTS = [
    "Erkläre in drei Sätzen, wofür ein Prüfprotokoll im Rohrleitungsbau dient.",
    (
        "Fasse in zwei Sätzen zusammen: Die Druckprüfung der Leitung 4711 ergab bei 1,5-fachem "
        "Betriebsdruck keinen Druckabfall. Zwei Flanschverbindungen wurden nachgezogen."
    ),
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_open_items",
            "description": "Liefert die offenen Punkte eines Projekts.",
            "parameters": {
                "type": "object",
                "properties": {"project_id": {"type": "string"}},
                "required": ["project_id"],
            },
        },
    }
]


async def bench_stream(client: AsyncOpenAI, model: str, prompt: str) -> dict:
    started = time.perf_counter()
    ttft_answer: float | None = None
    answer_parts: list[str] = []

    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    async for chunk in stream:
        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content
        if content:
            if ttft_answer is None:
                ttft_answer = time.perf_counter() - started
            answer_parts.append(content)

    total = time.perf_counter() - started
    answer = "".join(answer_parts)
    generation_time = total - (ttft_answer or total)
    chars_per_s = len(answer) / generation_time if generation_time > 0 else 0.0

    return {
        "ttft_answer": ttft_answer or 0.0,
        "total": total,
        "chars_per_s": chars_per_s,
    }


async def bench_tool_call(client: AsyncOpenAI, model: str) -> bool:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Welche offenen Punkte hat Projekt P-2041?"}
        ],
        tools=TOOLS,
    )

    message = response.choices[0].message
    tool_calls = message.tool_calls or []

    for tool_call in tool_calls:
        if tool_call.type != "function":
            continue
        if tool_call.function.name != "get_open_items":
            continue

        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            continue

        if isinstance(arguments, dict):
            project_id = arguments.get("project_id")
            if isinstance(project_id, str) and project_id:
                return True

    return False


async def main(model: str) -> None:
    client = AsyncOpenAI(base_url=BASE_URL, api_key="not-needed")
    try:
        # Warm-up the model and discard the result.
        await bench_stream(client, model, "Sag Hallo.")

        for index, prompt in enumerate(PROMPTS, start=1):
            result = await bench_stream(client, model, prompt)
            print(
                f"prompt {index}: "
                f"ttft_answer={result['ttft_answer']:.3f}s "
                f"total={result['total']:.3f}s "
                f"chars_per_s={result['chars_per_s']:.1f}"
            )

        tool_call_ok = await bench_tool_call(client, model)
        print("tool_call get_open_items: " + ("PASS" if tool_call_ok else "FAIL"))
    finally:
        await client.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <model-id>", file=sys.stderr)
        raise SystemExit(2)

    asyncio.run(main(sys.argv[1]))
