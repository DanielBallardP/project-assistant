"""Usage: uv run python scripts/generate_docs.py [--only DOC_ID] [--force]"""

import argparse
import asyncio
import time
from pathlib import Path

import yaml

from project_assistant.config import get_settings
from project_assistant.llm import ChatMessage, OpenAICompatibleClient

ROOT = Path(__file__).resolve().parent.parent
FACTS_FILE = ROOT / "data" / "facts.yaml"
OUT_DIR = ROOT / "data" / "docs"

SYSTEM_PROMPT_GENERATOR = (
    "Du bist ein technischer Redakteur. "
    "Schreibe ein realistisches internes Dokument des angegebenen Typs in Markdown "
    "(Überschriften, eine Tabelle, wo es passt, Daten im Format TT.MM.JJJJ); "
    "Verwende NUR die bereitgestellten Fakten und erfinde KEINE weiteren Zahlen, Namen oder Daten; "
    "Gib nur das Dokument aus, keine Kommentare."
)


def build_user_prompt(brief: dict) -> str:
    lines = [
        f"Typ: {brief['typ']}",
        f"Titel: {brief['title']}",
        f"Datum: {brief['datum']}",
        f"Revision: {brief['revision']}",
        "Fakten:",
    ]
    lines.extend(f"- {fact}" for fact in brief["facts"])
    return "\n".join(lines)


def front_matter(brief: dict) -> str:
    metadata = {
        "doc_id": brief["doc_id"],
        "title": brief["title"],
        "typ": brief["typ"],
        "projekt_id": brief["projekt_id"],
        "abteilung": brief["abteilung"],
        "vertraulichkeit": brief["vertraulichkeit"],
        "revision": brief["revision"],
        "datum": brief["datum"],
    }
    yaml_block = yaml.safe_dump(
        metadata,
        allow_unicode=True,
        sort_keys=False,
    ).strip()
    return f"---\n{yaml_block}\n---"


def missing_facts(text: str, brief: dict) -> list[str]:
    normalized_text = text.casefold()
    return [
        fact
        for fact in brief["must_contain"]
        if fact.casefold() not in normalized_text
    ]


def clean_body(text: str) -> str:
    body = text.strip()
    lines = body.splitlines()
    if (
        len(lines) >= 2
        and lines[0].strip().casefold() == "```markdown"
        and lines[-1].strip() == "```"
    ):
        body = "\n".join(lines[1:-1]).strip()
    return body


async def generate(
    client: OpenAICompatibleClient, brief: dict, max_attempts: int = 3
) -> str:
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT_GENERATOR),
        ChatMessage(role="user", content=build_user_prompt(brief)),
    ]

    for attempt in range(max_attempts):
        text = await client.chat(messages)
        missing = missing_facts(text, brief)
        if not missing:
            return text

        if attempt == max_attempts - 1:
            raise RuntimeError(
                f'Dokument "{brief["doc_id"]}" enthielt nach {max_attempts} '
                f"Versuchen nicht alle Pflichtangaben: {', '.join(missing)}"
            )

        messages.append(ChatMessage(role="assistant", content=text))
        messages.append(
            ChatMessage(
                role="user",
                content=(
                    "Korrektur erforderlich. Das Dokument muss zusätzlich folgende "
                    f"Angaben enthalten: {', '.join(missing)}"
                ),
            )
        )

    raise RuntimeError(
        f'Dokument "{brief["doc_id"]}" konnte nicht erstellt werden '
        "(max_attempts muss mindestens 1 sein)"
    )


async def main(only: str | None, force: bool) -> None:
    settings = get_settings()
    client = OpenAICompatibleClient(settings.llm_base_url, settings.llm_model)
    briefs = yaml.safe_load(FACTS_FILE.read_text(encoding="utf-8"))["briefs"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed_ids = []
    for brief in briefs:
        if only is not None and brief["doc_id"] != only:
            continue

        output_file = OUT_DIR / f"{brief['doc_id']}.md"
        if output_file.exists() and not force:
            print(f"{brief['doc_id']}: übersprungen ({output_file})")
            continue

        started = time.perf_counter()
        try:
            body = clean_body(await generate(client, brief))
        except RuntimeError as error:
            failed_ids.append(brief["doc_id"])
            print(f'{brief["doc_id"]}: fehlgeschlagen: {error}')
            continue
        elapsed = time.perf_counter() - started
        output_file.write_text(
            front_matter(brief) + "\n" + body + "\n",
            encoding="utf-8",
        )
        print(f"{brief['doc_id']}: geschrieben ({elapsed:.2f}s)")

    summary = ", ".join(failed_ids) if failed_ids else "keine"
    print(f"Fehlgeschlagene Dokumente: {summary}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.only, args.force))
