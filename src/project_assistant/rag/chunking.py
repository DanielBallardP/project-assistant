import re

from project_assistant.config import get_settings
from project_assistant.rag.models import Chunk, Document


def _is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def _is_table_separator(line: str) -> bool:
    cells = line.strip().strip("|").split("|")
    return len(cells) >= 2 and all(
        re.fullmatch(r":?-{3,}:?", cell.strip()) is not None for cell in cells
    )


def _is_table_block(block: str) -> bool:
    lines = block.splitlines()
    return _is_table_line(lines[0]) or any(_is_table_separator(line) for line in lines)


def _blocks(lines: list[str]) -> list[str]:
    result: list[str] = []
    current: list[str] = []
    current_is_table: bool | None = None
    table_lines = [_is_table_line(line) for line in lines]

    for index, line in enumerate(lines):
        if not _is_table_separator(line) or index == 0:
            continue
        if "|" not in lines[index - 1] or not lines[index - 1].strip():
            continue

        table_lines[index - 1] = True
        table_lines[index] = True
        row_index = index + 1
        while (
            row_index < len(lines)
            and lines[row_index].strip()
            and "|" in lines[row_index]
        ):
            table_lines[row_index] = True
            row_index += 1

    def flush() -> None:
        if current:
            result.append("\n".join(current).strip())
            current.clear()

    for line, is_table in zip(lines, table_lines, strict=True):
        if not line.strip():
            flush()
            current_is_table = None
            continue

        if current and is_table != current_is_table:
            flush()
        current.append(line.rstrip())
        current_is_table = is_table

    flush()
    return result


def _split_prose(text: str, max_chars: int) -> list[str]:
    words = text.split()
    parts: list[str] = []
    current = ""

    for word in words:
        while len(word) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.append(word[:max_chars])
            word = word[max_chars:]

        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars:
            parts.append(current)
            current = word
        else:
            current = candidate

    if current:
        parts.append(current)
    return parts


def _sections(body: str, default_heading: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    heading = default_heading
    lines: list[str] = []

    for line in body.splitlines():
        if line.startswith("## "):
            if lines:
                sections.append((heading, lines))
            heading = line[3:].strip()
            lines = [line]
        else:
            lines.append(line)

    if lines:
        sections.append((heading, lines))
    return sections or [(default_heading, [])]


def chunk_document(
    doc: Document, max_chars: int | None = None, min_chars: int | None = None
) -> list[Chunk]:
    """Split a Markdown document into heading-aware chunks without splitting tables.

    max_chars / min_chars default to the values from Settings
    (PA_CHUNK_MAX_CHARS / PA_CHUNK_MIN_CHARS) when not passed explicitly.
    """
    settings = None
    if max_chars is None:
        settings = get_settings()
        max_chars = settings.chunk_max_chars
    if min_chars is None:
        if settings is None:
            settings = get_settings()
        min_chars = settings.chunk_min_chars

    if max_chars < 1:
        raise ValueError("max_chars must be greater than zero")
    if min_chars < 0:
        raise ValueError("min_chars cannot be negative")
    if min_chars > max_chars:
        raise ValueError("min_chars cannot be greater than max_chars")

    sections = [
        (heading, lines.copy()) for heading, lines in _sections(doc.body, doc.title)
    ]
    for index in range(len(sections) - 1, -1, -1):
        heading, lines = sections[index]
        if len("\n".join(lines).strip()) >= min_chars or len(sections) == 1:
            continue

        if index > 0:
            previous_heading, previous_lines = sections[index - 1]
            sections[index - 1] = (
                previous_heading,
                previous_lines + [""] + lines,
            )
            del sections[index]
        else:
            next_heading, next_lines = sections[index + 1]
            sections[index + 1] = (next_heading, lines + [""] + next_lines)
            del sections[index]

    section_chunks: list[tuple[str, str]] = []
    for heading, lines in sections:
        content_blocks = _blocks(lines)
        split_blocks: list[str] = []
        for block in content_blocks:
            if _is_table_block(block):
                if len(block) > max_chars:
                    raise ValueError(
                        f"Table in {doc.doc_id} section {heading!r} exceeds "
                        f"max_chars ({len(block)} > {max_chars})"
                    )
                split_blocks.append(block)
            elif len(block) <= max_chars:
                split_blocks.append(block)
            else:
                split_blocks.extend(_split_prose(block, max_chars))

        current = ""
        for block in split_blocks:
            candidate = f"{current}\n\n{block}".strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    section_chunks.append((heading, current))
                current = block
        if current:
            section_chunks.append((heading, current))

    if not section_chunks:
        section_chunks.append((doc.title, ""))

    merged: list[tuple[str, str]] = []
    for heading, content in section_chunks:
        if merged:
            previous_heading, previous_content = merged[-1]
            combined = f"{previous_content}\n\n{content}"
            if len(combined) <= max_chars:
                merged[-1] = (previous_heading, combined)
                continue
        merged.append((heading, content))

    context_header = f"{doc.title} | {doc.typ} {doc.doc_id}, Revision {doc.revision}"
    if doc.projekt_id is not None:
        context_header += f" | Projekt {doc.projekt_id}"

    metadata: dict[str, str | int | float | bool] = {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "typ": doc.typ,
        "abteilung": doc.abteilung,
        "vertraulichkeit": doc.vertraulichkeit,
        "revision": doc.revision,
        "datum": doc.datum,
        "is_current": doc.is_current,
    }
    if doc.projekt_id is not None:
        metadata["projekt_id"] = doc.projekt_id

    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#{index}",
            doc_id=doc.doc_id,
            heading=heading,
            text=f"{context_header}\n\n{content}",
            metadata={**metadata, "heading": heading},
        )
        for index, (heading, content) in enumerate(merged)
    ]
