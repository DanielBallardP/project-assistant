"""Usage: uv run python scripts/probe_tools.py [--only N]

Checks whether the local model calls our two tools correctly in a multi-turn loop,
before any LangGraph code exists.
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from openai import AsyncOpenAI

from project_assistant.config import get_settings
from project_assistant.llm import get_embedding_client
from project_assistant.rag.answering import build_context
from project_assistant.rag.retrieval import retrieve
from project_assistant.rag.store import VectorStore
from project_assistant.sql_tool import QueryResult, SqlToolError, run_query

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "project.db"
MAX_STEPS = 6

SYSTEM = (
    "Du bist der Projektassistent eines Anlagenbau-Unternehmens. "
    "Antworte auf Deutsch und kurz. Nutze Werkzeuge, statt zu raten. "
    "Zahlen und Listen holst du mit query_database, "
    "Verfahren, Prüfergebnisse und Revisionen mit search_documents. "
    "Wenn eine Frage beides braucht, nutze beide Werkzeuge. "
    "Nenne bei Dokumentwissen die Dokument-ID in eckigen Klammern, "
    "zum Beispiel [PP-2041-A]. Stichtag für Datumsfragen ist der 20.11.2026.\n\n"
    "Datenbank (SQLite, Datumsangaben sind Text im Format YYYY-MM-DD):\n"
    "kunden(id, name, ort)\n"
    "projekte(id, name, kunde_id, status, projektleiter, start_datum, "
    "ende_geplant, budget_eur)\n"
    "stunden(id, projekt_id, mitarbeiter, datum, stunden, taetigkeit)\n"
    "material(id, projekt_id, bezeichnung, menge, einheit, kosten_eur, lieferant)\n"
    "offene_punkte(id, projekt_id, beschreibung, prioritaet, status, "
    "faellig_am, verantwortlich)"
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the project documents (test reports, work procedure "
                "specifications, quality manuals). Use for questions about "
                "procedures, test results, weld seams, pressures and revisions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query in German"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": (
                "Run ONE read-only SQLite SELECT query on the project database "
                "(customers, projects, booked hours, material, open items). "
                "Use for numbers, sums and lists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A single SELECT statement"}
                },
                "required": ["sql"],
            },
        },
    },
]

# (question, tools that must be called, regex the final answer must match)
CASES = [
    (
        "Wie viele Stunden hat Peter Gruber auf P-2041 gebucht?",
        {"query_database"},
        r"54[.,]5",
    ),
    (
        "Welchen Prüfdruck hatte Strang A?",
        {"search_documents"},
        r"24",
    ),
    (
        "Welchen Prüfdruck hatte Strang A und wie viele Stunden hat "
        "Peter Gruber auf P-2041 gebucht?",
        {"search_documents", "query_database"},
        r"24.*54[.,]5|54[.,]5.*24",
    ),
]


def format_result(result: QueryResult) -> str:
    lines = [" | ".join(result.columns)]
    lines += [
        " | ".join("NULL" if value is None else str(value) for value in row)
        for row in result.rows
    ]
    if result.truncated:
        lines.append("... (truncated)")
    return "\n".join(lines)


async def search_documents(query: str) -> str:
    settings = get_settings()
    store = VectorStore(
        settings.chroma_path, settings.collection_name, settings.embedding_model
    )
    hits = await retrieve(
        query, embeddings=get_embedding_client(), store=store, role="mitarbeiter"
    )
    return build_context(hits) or "No documents found."


async def query_database(sql: str) -> str:
    try:
        result = await asyncio.to_thread(run_query, sql, db_path=DB_PATH)
    except SqlToolError as exc:
        return f"ERROR: {exc}"  # the model can read this and correct itself
    return format_result(result)


async def execute_tool(name: str, raw_arguments: str) -> tuple[str, bool]:
    """Return (result text, whether the arguments were valid)."""
    try:
        arguments = json.loads(raw_arguments or "{}")
        if name == "search_documents":
            return await search_documents(arguments["query"]), True
        if name == "query_database":
            return await query_database(arguments["sql"]), True
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return f"ERROR: invalid tool arguments ({exc!r})", False
    return f"ERROR: unknown tool {name!r}", False


async def run_case(
    client: AsyncOpenAI,
    model: str,
    question: str,
    expected_tools: set[str],
    answer_pattern: str,
) -> bool:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    called: list[str] = []
    bad_arguments = 0
    answer = ""
    started = time.monotonic()

    for _ in range(MAX_STEPS):
        response = await client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS
        )
        message = response.choices[0].message
        if not message.tool_calls:
            answer = (message.content or "").strip()
            break

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                ],
            }
        )
        for call in message.tool_calls:
            result, valid = await execute_tool(
                call.function.name, call.function.arguments
            )
            called.append(call.function.name)
            bad_arguments += not valid
            print(f"  tool {call.function.name} {call.function.arguments}")
            print(f"    -> {result[:120]!r}")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )

    seconds = time.monotonic() - started
    ok = (
        set(called) == expected_tools
        and bad_arguments == 0
        and bool(re.search(answer_pattern, answer, re.S))
    )
    print(f"  answer: {answer or '(none)'}")
    print(
        f"  {'PASS' if ok else 'FAIL'}  tools={sorted(set(called))} "
        f"expected={sorted(expected_tools)} bad_args={bad_arguments} {seconds:.0f}s"
    )
    return ok


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, help="run only case N (1-based)")
    args = parser.parse_args()

    settings = get_settings()
    client = AsyncOpenAI(
        base_url=settings.llm_base_url, api_key="not-needed", timeout=600
    )
    results = []
    for number, (question, expected_tools, pattern) in enumerate(CASES, start=1):
        if args.only and args.only != number:
            continue
        print(f"\n=== Q{number}: {question}")
        results.append(
            await run_case(client, settings.llm_model, question, expected_tools, pattern)
        )
    print(f"\n{sum(results)}/{len(results)} passed")


if __name__ == "__main__":
    asyncio.run(main())
    