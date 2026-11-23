"""Usage: uv run python evals/retrieval_eval.py [--k 4] [--only q03] [--min-hit-rate 0.8]"""

import argparse
import asyncio
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from project_assistant.config import get_settings
from project_assistant.llm import get_embedding_client
from project_assistant.rag.retrieval import retrieve
from project_assistant.rag.store import VectorStore

QUESTIONS_FILE = Path(__file__).resolve().parent / "questions.yaml"
RETRIEVAL_CATEGORIES = {"doc", "combined", "outdated_revision", "permission"}


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a misspelled key fails loudly

    id: str
    category: Literal[
        "doc", "sql", "combined", "outdated_revision", "permission", "unanswerable"
    ]
    question: str
    user_role: Literal["mitarbeiter", "projektleitung"] = "projektleitung"
    expected_contains: list[str] = []
    forbidden_contains: list[str] = []
    expected_sources: list[str] = []
    forbidden_sources: list[str] = []
    expected_behavior: Literal["abstain"] | None = None
    expected_sql: str | None = None


class Result(BaseModel):
    question_id: str
    category: str
    hit: bool
    first_rank: int | None
    retrieved: list[str]
    missing: list[str]
    leaked: list[str]


def load_questions(path: Path = QUESTIONS_FILE) -> list[Question]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    questions: list[Question] = []
    for index, item in enumerate(raw, start=1):
        try:
            question = Question(**item)
        except ValidationError as exc:
            raise ValueError(
                f"questions.yaml entry {index} ({item.get('id', '?')}): {exc}"
            ) from exc
        if question.category in RETRIEVAL_CATEGORIES and not (
            question.expected_sources or question.forbidden_sources
        ):
            raise ValueError(
                f"{question.id}: a retrieval question needs a source assertion"
            )
        questions.append(question)
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("questions.yaml contains duplicate ids")
    return questions


def unique_in_order(doc_ids: list[str]) -> list[str]:
    """Top-k are CHUNKS; several can come from one document. Rank documents instead."""
    return list(dict.fromkeys(doc_ids))


def score(q: Question, retrieved: list[str]) -> Result:
    missing = [s for s in q.expected_sources if s not in retrieved]
    leaked = [s for s in q.forbidden_sources if s in retrieved]
    ranks = [retrieved.index(s) + 1 for s in q.expected_sources if s in retrieved]
    return Result(
        question_id=q.id,
        category=q.category,
        hit=not missing and not leaked,
        first_rank=min(ranks) if ranks else None,
        retrieved=retrieved,
        missing=missing,
        leaked=leaked,
    )


def print_report(results: list[Result]) -> None:
    print(f"{'id':<5} {'category':<18} {'result':<6} {'rank':<5} retrieved")
    print("-" * 90)
    for r in results:
        status = "PASS" if r.hit else "FAIL"
        rank = str(r.first_rank) if r.first_rank else "-"
        notes = ""
        if r.missing:
            notes += f"  MISSING: {', '.join(r.missing)}"
        if r.leaked:
            notes += f"  LEAKED: {', '.join(r.leaked)}"
        print(
            f"{r.question_id:<5} {r.category:<18} {status:<6} {rank:<5} {', '.join(r.retrieved)}{notes}"
        )

    by_category: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r.hit)
    print()
    for category, hits in sorted(by_category.items()):
        print(f"{category:<18} {sum(hits)}/{len(hits)}")
    total = sum(r.hit for r in results)
    print(f"{'overall':<18} {total}/{len(results)}  ({total / len(results):.0%})")


async def run(k: int, only: str | None) -> list[Result]:
    settings = get_settings()
    store = VectorStore(
        settings.chroma_path, settings.collection_name, settings.embedding_model
    )
    embeddings = get_embedding_client()
    questions = [
        q
        for q in load_questions()
        if q.category in RETRIEVAL_CATEGORIES and (only is None or q.id == only)
    ]
    results: list[Result] = []
    for q in questions:
        hits = await retrieve(
            q.question, embeddings=embeddings, store=store, role=q.user_role, k=k
        )
        results.append(score(q, unique_in_order([h.doc_id for h in hits])))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--only")
    parser.add_argument("--min-hit-rate", type=float, default=0.0)
    args = parser.parse_args()

    all_results = asyncio.run(run(args.k, args.only))
    if not all_results:
        sys.exit("No retrieval questions selected.")
    print_report(all_results)
    rate = sum(r.hit for r in all_results) / len(all_results)
    sys.exit(0 if rate >= args.min_hit_rate else 1)
