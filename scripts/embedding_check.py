"""Usage: uv run python scripts/embedding_check.py --model text-embedding-bge-m3
   uv run python scripts/embedding_check.py --model text-embedding-nomic-embed-text-v1.5 \
       --query-prefix "search_query: " --document-prefix "search_document: " """

import argparse
import asyncio
from time import perf_counter

import numpy as np

from project_assistant.config import get_settings
from project_assistant.llm import OpenAICompatibleEmbeddings

CASES = [
    {
        "query": "Welchen Prüfdruck hatte Strang A?",
        "candidates": {
            "relevant (Strang A)": "Prüfmedium Wasser, Betriebsdruck 16 bar, Prüfdruck 24 bar, Haltedauer 60 Minuten, Druckabfall 0 bar, bestanden. Strang A.",
            "hard near-miss (Strang B)": "Prüfmedium Wasser, Betriebsdruck 16 bar, Prüfdruck 24 bar, Haltedauer 60 Minuten, Druckabfall 0,4 bar, nicht bestanden. Strang B.",
            "same topic, general rule": "Der Prüfdruck beträgt das 1,5-fache des Betriebsdrucks, die Haltedauer mindestens 60 Minuten.",
            "irrelevant": "Die Baustellenzufahrt Abschnitt Süd muss mit der Gemeinde Eisenau abgestimmt werden.",
        },
    },
    {
        "query": "Wie hoch ist die maximale Zwischenlagentemperatur nach WPS-014?",
        "candidates": {
            "WPS-014 Rev A (outdated)": "WPS-014 Revision A, WIG-Schweißen 1.4571. Zwischenlagentemperatur maximal 200 °C. Gültig bis 30.04.2026.",
            "WPS-014 Rev B (current)": "WPS-014 Revision B, WIG-Schweißen 1.4571. Zwischenlagentemperatur maximal 150 °C. Ersetzt Revision A.",
            "irrelevant": "Die Rohrlieferung verzögert sich um drei Wochen, neuer Liefertermin 30.11.2026.",
        },
    },
]


def cosine(a: list[float], b: list[float]) -> float:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(
            f"Embedding dimensions do not match: {left.size} and {right.size}"
        )

    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector")

    return float(np.dot(left, right) / denominator)


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Compare embedding similarity for query/candidate examples."
    )
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--query-prefix", default=settings.embedding_query_prefix)
    parser.add_argument("--document-prefix", default=settings.embedding_document_prefix)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = get_settings()
    embeddings = OpenAICompatibleEmbeddings(
        base_url=settings.llm_base_url,
        model=args.model,
        query_prefix=args.query_prefix,
        document_prefix=args.document_prefix,
    )

    for case_number, case in enumerate(CASES, start=1):
        started = perf_counter()
        labels = list(case["candidates"])
        query_vector = await embeddings.embed_query(case["query"])
        candidate_vectors = await embeddings.embed_documents(
            list(case["candidates"].values())
        )
        similarities = [
            (label, cosine(query_vector, vector))
            for label, vector in zip(labels, candidate_vectors, strict=True)
        ]
        elapsed = perf_counter() - started

        print(f"Case {case_number}: {case['query']}")
        print(f"Vector dimension: {len(query_vector)}")
        print(f"Time: {elapsed:.3f}s")
        print(f"{'Candidate':<36} Similarity")
        print("-" * 50)
        for label, similarity in sorted(
            similarities, key=lambda result: result[1], reverse=True
        ):
            print(f"{label:<36} {similarity:.6f}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
