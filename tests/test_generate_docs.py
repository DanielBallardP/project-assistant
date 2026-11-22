import asyncio
from types import SimpleNamespace

from scripts import generate_docs
from scripts.generate_docs import missing_facts


def test_missing_facts_returns_only_required_strings_absent_from_text():
    brief = {"must_contain": ["24 bar", "60 Minuten", "bestanden"]}

    assert missing_facts("Prüfdruck 24 bar; Ergebnis bestanden.", brief) == [
        "60 Minuten"
    ]


def test_missing_facts_returns_empty_list_when_all_required_strings_are_present():
    brief = {"must_contain": ["24 bar", "bestanden"]}

    assert missing_facts("24 bar Prüfdruck; Ergebnis bestanden.", brief) == []


def test_missing_facts_ignores_capitalization():
    brief = {"must_contain": ["BESTANDEN", "24 BAR"]}

    assert missing_facts("Ergebnis bestanden bei 24 bar.", brief) == []


def test_missing_facts_normalizes_nonbreaking_whitespace():
    brief = {
        "must_contain": [
            "Prüfdruck\u00a024 bar",
            "Haltedauer\u202f60 Minuten",
        ]
    }

    assert (
        missing_facts("Prüfdruck\u202f 24 bar; Haltedauer\u00a060 Minuten.", brief)
        == []
    )


def test_clean_body_strips_whitespace_and_markdown_fence():
    assert generate_docs.clean_body(" \n```markdown\n# Bericht\n\nInhalt\n```\n ") == (
        "# Bericht\n\nInhalt"
    )


def test_main_continues_after_generation_failure_and_reports_failed_ids(
    tmp_path, monkeypatch, capsys
):
    briefs = [
        {
            "doc_id": doc_id,
            "title": f"Dokument {doc_id}",
            "typ": "Bericht",
            "projekt_id": None,
            "abteilung": "Qualität",
            "vertraulichkeit": "intern",
            "revision": "A",
            "datum": "2026-11-21",
            "facts": [],
            "must_contain": [],
        }
        for doc_id in ("FAIL", "SUCCESS")
    ]
    facts_file = tmp_path / "facts.yaml"
    facts_file.write_text(
        "briefs:\n"
        + "".join(
            f"  - doc_id: {brief['doc_id']}\n"
            f"    title: {brief['title']}\n"
            "    typ: Bericht\n"
            "    projekt_id: null\n"
            "    abteilung: Qualität\n"
            "    vertraulichkeit: intern\n"
            "    revision: A\n"
            "    datum: '2026-11-21'\n"
            "    facts: []\n"
            "    must_contain: []\n"
            for brief in briefs
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(generate_docs, "FACTS_FILE", facts_file)
    monkeypatch.setattr(generate_docs, "OUT_DIR", tmp_path / "docs")
    monkeypatch.setattr(
        generate_docs,
        "get_settings",
        lambda: SimpleNamespace(llm_base_url="http://unused", llm_model="unused"),
    )
    monkeypatch.setattr(
        generate_docs, "OpenAICompatibleClient", lambda *_args: object()
    )

    async def fake_generate(_client, brief):
        if brief["doc_id"] == "FAIL":
            raise RuntimeError("required facts are missing")
        return " \n```markdown\n# Generated\n``` \n"

    monkeypatch.setattr(generate_docs, "generate", fake_generate)

    asyncio.run(generate_docs.main(only=None, force=False))

    assert not (tmp_path / "docs" / "FAIL.md").exists()
    generated = (tmp_path / "docs" / "SUCCESS.md").read_text(encoding="utf-8")
    assert generated.endswith("---\n# Generated\n")
    assert "```markdown" not in generated
    output = capsys.readouterr().out
    assert "FAIL: fehlgeschlagen: required facts are missing" in output
    assert "Fehlgeschlagene Dokumente: FAIL" in output
