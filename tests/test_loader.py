from pathlib import Path

import pytest

from project_assistant.rag.loader import mark_current, parse_document
from project_assistant.rag.models import Document


def make_document(**overrides) -> Document:
    values = {
        "doc_id": "DOC-A",
        "title": "Shared title",
        "typ": "Report",
        "projekt_id": None,
        "abteilung": "Quality",
        "vertraulichkeit": "intern",
        "revision": "A",
        "datum": "2026-11-22",
        "body": "Body",
    }
    values.update(overrides)
    return Document(**values)


def test_parse_document_reads_front_matter_and_body(tmp_path: Path):
    path = tmp_path / "report.md"
    path.write_text(
        "---\n"
        "doc_id: DOC-A\n"
        "title: Report title\n"
        "typ: Report\n"
        "projekt_id: null\n"
        "abteilung: Quality\n"
        "vertraulichkeit: intern\n"
        "revision: A\n"
        "datum: '2026-11-22'\n"
        "---\n"
        "\n"
        "# Heading\n"
        "\n"
        "Content\n",
        encoding="utf-8",
    )

    document = parse_document(path)

    assert document.doc_id == "DOC-A"
    assert document.datum == "2026-11-22"
    assert document.body == "# Heading\n\nContent"


@pytest.mark.parametrize(
    "text",
    [
        "# No front matter\n",
        "---\ndoc_id: DOC-A\n",
        "---\ndoc_id: DOC-A\n---\n",
    ],
)
def test_parse_document_rejects_missing_or_incomplete_front_matter(
    tmp_path: Path, text: str
):
    path = tmp_path / "broken.md"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="broken\\.md"):
        parse_document(path)


def test_parse_document_rejects_incomplete_front_matter(
    tmp_path: Path,
):
    path = tmp_path / "incomplete_frontmatter.md"
    path.write_text(
        "---\ndoc_id: DOC-A\n---\nBody",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incomplete YAML-Frontmatter"):
        parse_document(path)


def test_parse_document_rejects_invalid_front_matter(
    tmp_path: Path,
):
    path = tmp_path / "invalid_frontmatter.md"
    path.write_text(
        "---\ndoc_id: [\n---\nBody",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid YAML-Frontmatter"):
        parse_document(path)


def test_mark_current_marks_only_highest_revision_and_returns_copies():
    revision_a = make_document(
        doc_id="WPS-014-A",
        title="WPS-014",
        typ="Schweißanweisung",
        revision="A",
    )
    revision_b = make_document(
        doc_id="WPS-014-B",
        title="WPS-014",
        typ="Schweißanweisung",
        revision="B",
    )
    unrelated = make_document(
        doc_id="OTHER",
        title="Other",
        typ="Report",
        revision="Z",
    )

    result = mark_current([revision_a, revision_b, unrelated])

    assert result[0].is_current is False
    assert result[1].is_current is True
    assert result[2].is_current is True
    assert result[0] is not revision_a
