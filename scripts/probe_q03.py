"""Usage: uv run python scripts/probe_q03.py"""
import asyncio

from project_assistant.config import get_settings
from project_assistant.llm import get_embedding_client
from project_assistant.rag.retrieval import retrieve
from project_assistant.rag.store import VectorStore

TARGETS = ["PP-2041-B", "BP-2041-1109"]
QUERIES = {
    "A original, compound, undicht": "Welche Schweißnaht war undicht und wer ist für die Nachschweißung zuständig?",
    "B single, Leckage": "An welcher Schweißnaht trat die Leckage auf?",
    "C single, Nachschweißung": "Wer führt die Nachschweißung durch?",
    "D compound, Leckage": "Welche Schweißnaht hatte eine Leckage und wer ist für die Nachschweißung zuständig?",
    "E single, undicht": "Welche Schweißnaht war undicht?",
}


async def main() -> None:
    settings = get_settings()
    store = VectorStore(settings.chroma_path, settings.collection_name, settings.embedding_model)
    embeddings = get_embedding_client()
    for label, query in QUERIES.items():
        hits = await retrieve(query, embeddings=embeddings, store=store, role="projektleitung", k=15)
        best: dict[str, tuple[int, float]] = {}  # doc_id -> (rank, score) of first appearance
        for hit in hits:
            best.setdefault(hit.doc_id, (len(best) + 1, hit.score))
        parts = []
        for target in TARGETS:
            rank, score = best.get(target, (None, None))
            parts.append(f"{target}: " + (f"rank {rank} ({score:.3f})" if rank else "not in top 15"))
        top = next(iter(best))
        print(f"{label:<32} | {' | '.join(parts)} | top1: {top} ({best[top][1]:.3f})")


asyncio.run(main())