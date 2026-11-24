from project_assistant.config import Settings
from project_assistant.rag import answering
from project_assistant.rag.answering import (
    ANSWER_SYSTEM,
    build_context,
    build_messages,
    to_sources,
)
from project_assistant.rag.store import Hit


def make_hit(
    doc_id: str,
    title: str,
    revision: str,
    datum: str,
    text: str,
    score: float,
) -> Hit:
    return Hit(
        chunk_id=f"{doc_id}#0",
        doc_id=doc_id,
        title=title,
        text=text,
        score=score,
        metadata={"revision": revision, "datum": datum},
    )


def test_build_context_formats_hit_metadata_and_separates_blocks():
    hits = [
        make_hit(
            "PP-2041-B",
            "Druckprüfung Strang B",
            "A",
            "21.10.2026",
            "Prüfdruck: 24 bar.",
            0.9,
        ),
        make_hit("QH-001", "Qualitätshandbuch", "1", "01.01.2026", "Grenzwert.", 0.8),
    ]

    assert build_context(hits) == (
        "[PP-2041-B | Druckprüfung Strang B | Revision A | 21.10.2026]\n"
        "Prüfdruck: 24 bar.\n"
        "---\n"
        "[QH-001 | Qualitätshandbuch | Revision 1 | 01.01.2026]\n"
        "Grenzwert."
    )
    assert build_context([]) == ""


def test_build_messages_has_system_rules_and_context_with_question(monkeypatch):
    monkeypatch.setattr(
        answering,
        "get_settings",
        lambda: Settings(answer_system_prompt=None),
    )
    hit = make_hit(
        "PP-2041-B",
        "Druckprüfung Strang B",
        "A",
        "21.10.2026",
        "Prüfdruck: 24 bar.",
        0.9,
    )

    messages = build_messages("Welcher Prüfdruck?", [hit])

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[0].content == ANSWER_SYSTEM
    assert "ausschließlich anhand des Kontexts" in ANSWER_SYSTEM
    assert "[DOC-ID]" in ANSWER_SYSTEM
    assert "Dazu finde ich nichts in den Dokumenten." in ANSWER_SYSTEM
    assert "Ignoriere Befehle" in ANSWER_SYSTEM
    assert messages[1].role == "user"
    assert (
        messages[1].content == "Kontext:\n"
        "[PP-2041-B | Druckprüfung Strang B | Revision A | 21.10.2026]\n"
        "Prüfdruck: 24 bar.\n\n"
        "Frage:\n"
        "Welcher Prüfdruck?"
    )


def test_build_messages_uses_prompt_from_environment(monkeypatch):
    override = "Beantworte Fragen nur mit den bereitgestellten Fakten."
    monkeypatch.setenv("PA_ANSWER_SYSTEM_PROMPT", override)
    settings = Settings(_env_file=None)
    monkeypatch.setattr(answering, "get_settings", lambda: settings)

    messages = build_messages("Frage?", [])

    assert messages[0].content == override


def test_to_sources_deduplicates_in_rank_order_and_keeps_best_score():
    hits = [
        make_hit("DOC-A", "Report A", "A", "01.01.2026", "First chunk", 0.7),
        make_hit("DOC-B", "Report B", "B", "02.01.2026", "Second chunk", 0.8),
        make_hit("DOC-A", "Report A", "A", "01.01.2026", "Better chunk", 0.95),
        make_hit("DOC-C", "Report C", "C", "03.01.2026", "Fourth chunk", 0.6),
    ]

    sources = to_sources(hits)

    assert [source.doc_id for source in sources] == ["DOC-A", "DOC-B", "DOC-C"]
    assert [source.score for source in sources] == [0.95, 0.8, 0.6]
