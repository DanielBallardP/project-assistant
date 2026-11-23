from unittest.mock import Mock

from project_assistant.rag import chunking
from project_assistant.rag.chunking import chunk_document
from project_assistant.rag.models import Document


def make_document(body: str) -> Document:
    return Document(
        doc_id="SYNTH-001",
        title="Synthetic report",
        typ="Test report",
        projekt_id="P-2041",
        abteilung="Quality",
        vertraulichkeit="intern",
        revision="B",
        datum="2026-11-22",
        body=body,
        is_current=True,
    )


def test_long_markdown_chunks_respect_content_limit_and_have_sequential_ids():
    paragraph = " ".join(["synthetic"] * 38)
    document = make_document(
        "# Synthetic report\n\n"
        "## Introduction\n\n"
        f"{paragraph}\n\n"
        f"{paragraph}\n\n"
        "## Findings\n\n"
        f"{paragraph}\n\n"
        f"{paragraph}\n"
    )
    max_chars = 220
    header = "Synthetic report | Test report SYNTH-001, Revision B | Projekt P-2041"

    chunks = chunk_document(document, max_chars=max_chars, min_chars=100)

    assert len(chunks) > 1
    assert [chunk.chunk_id for chunk in chunks] == [
        f"SYNTH-001#{index}" for index in range(len(chunks))
    ]
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    for chunk in chunks:
        assert chunk.text.startswith(f"{header}\n\n")
        content = chunk.text.removeprefix(f"{header}\n\n")
        assert len(content) <= max_chars


def test_table_is_kept_whole_when_surrounding_section_is_split():
    table = (
        "| Parameter | Value |\n"
        "|---|---|\n"
        "| Pressure | 24 bar |\n"
        "| Duration | 60 minutes |\n"
        "| Result | Passed |\n"
    )
    paragraph = " ".join(["supporting"] * 40)
    document = make_document(
        "# Synthetic report\n\n"
        "## Test results\n\n"
        f"{paragraph}\n\n"
        f"{table}\n"
        f"{paragraph}\n"
    )

    chunks = chunk_document(document, max_chars=220, min_chars=50)
    chunks_containing_table = [
        chunk for chunk in chunks if table.rstrip() in chunk.text
    ]

    assert len(chunks) > 1
    assert len(chunks_containing_table) == 1


def test_pipe_less_markdown_table_is_kept_whole():
    table = (
        "Parameter | Value\n"
        "--- | ---\n"
        "Pressure | 24 bar\n"
        "Duration | 60 minutes\n"
        "Result | Passed"
    )
    paragraph = " ".join(["supporting"] * 40)
    document = make_document(
        f"## Test results\n\n{paragraph}\n\n{table}\n\n{paragraph}"
    )

    chunks = chunk_document(document, max_chars=220, min_chars=50)
    chunks_containing_table = [chunk for chunk in chunks if table in chunk.text]

    assert len(chunks) > 1
    assert len(chunks_containing_table) == 1


def test_short_document_produces_exactly_one_chunk():
    document = make_document("# Synthetic report\n\nA brief note.")

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "SYNTH-001#0"
    assert chunks[0].text.endswith("# Synthetic report\n\nA brief note.")


def test_short_section_merges_with_neighbor_before_splitting():
    document = make_document(
        "## Note\n\nShort.\n\n## Details\n\n" + " ".join(["detail"] * 50)
    )

    chunks = chunk_document(document, max_chars=200, min_chars=100)

    first_content = chunks[0].text.split("\n\n", maxsplit=1)[1]
    assert chunks[0].heading == "Details"
    assert "## Note" in first_content
    assert "## Details" in first_content
    assert all(len(chunk.text.split("\n\n", maxsplit=1)[1]) <= 200 for chunk in chunks)


def test_explicit_sizes_do_not_load_settings(monkeypatch):
    settings_lookup = Mock(side_effect=AssertionError("settings should not load"))
    monkeypatch.setattr(chunking, "get_settings", settings_lookup)

    chunks = chunk_document(
        make_document("# Synthetic report\n\nA brief note."),
        max_chars=100,
        min_chars=20,
    )

    settings_lookup.assert_not_called()
    assert len(chunks) == 1
