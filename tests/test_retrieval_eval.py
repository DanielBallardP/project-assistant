import pytest
from pydantic import ValidationError

from evals.retrieval_eval import Question, load_questions, score


def test_real_questions_file_is_valid():
    assert load_questions()


def test_score_hits_when_all_expected_sources_present():
    q = Question(id="t", category="combined", question="?", expected_sources=["A", "B"])
    result = score(q, ["X", "B", "A"])
    assert result.hit and result.first_rank == 2


def score_misses_when_an_expected_source_is_absent():
    q = Question(id="t", category="combined", question="?", expected_sources=["A", "B"])
    result = score(q, ["X", "A"])
    assert not result.hit and result.missing == ["B"]


def test_score_fails_when_a_forbidden_source_leaks():
    q = Question(id="t", category="permission", question="?", forbidden_sources=["LB"])
    assert score(q, ["LB"]).leaked == ["LB"]
    assert score(q, ["PB"]).hit


def test_unknown_key_and_unknown_category_are_rejected():
    with pytest.raises(ValidationError):
        Question(id="t", category="doc", question="?", forbiden_sources=["x"])
    with pytest.raises(ValidationError):
        Question(id="t", category="docg", question="?")
