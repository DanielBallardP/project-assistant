import re
from datetime import date, datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from project_assistant.rag.models import Document


def parse_document(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---(?:\n|$)", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"{path.name}: missing or incomplete YAML-Frontmatter")

    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as error:
        raise ValueError(f"{path.name}: invalid YAML-Frontmatter") from error

    if not isinstance(meta, dict):
        raise ValueError(  # noqa: TRY004 - parse failures use a consistent ValueError
            f"{path.name}: incomplete YAML-Frontmatter"
        )

    required = {
        "doc_id",
        "title",
        "typ",
        "projekt_id",
        "abteilung",
        "vertraulichkeit",
        "revision",
        "datum",
    }
    if not required.issubset(meta):
        raise ValueError(f"{path.name}: incomplete YAML-Frontmatter")

    string_fields = required - {"projekt_id"}
    for field in string_fields:
        value = meta[field]
        if isinstance(value, (date, datetime)):
            meta[field] = value.isoformat()
        elif not isinstance(value, str):
            meta[field] = str(value)

    body = text[match.end() :].strip()
    try:
        return Document(**meta, body=body)
    except ValidationError as error:
        raise ValueError(
            f"{path.name}: incomplete oder invalid YAML-Frontmatter"
        ) from error


def mark_current(docs: list[Document]) -> list[Document]:
    """Goes through all documents and marks the latest Revision as current
    in alphabetical order.

    e.g. out of three Revisions A, B, C of the same document
    Revision C will be marked as current
    """
    highest_revisions: dict[tuple[str, str], str] = {}
    for doc in docs:
        family = (doc.typ, doc.title)
        highest_revisions[family] = max(
            highest_revisions.get(family, doc.revision),
            doc.revision,
        )

    return [
        doc.model_copy(
            update={
                "is_current": doc.revision == highest_revisions[(doc.typ, doc.title)]
            }
        )
        for doc in docs
    ]
