"""Usage: uv run python scripts/search.py "Frage" --role mitarbeiter --k 4"""

import argparse
import asyncio

from project_assistant.config import get_settings
from project_assistant.llm import get_embedding_client
from project_assistant.rag.retrieval import ROLE_CLEARANCE, retrieve
from project_assistant.rag.store import VectorStore


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search indexed project documents.")
    parser.add_argument("query", help="Question or search phrase.")
    parser.add_argument(
        "--role",
        required=True,
        choices=ROLE_CLEARANCE,
        help="Role used to filter documents by confidentiality.",
    )
    parser.add_argument(
        "--k",
        type=_positive_int,
        default=4,
        help="Maximum number of hits to return (default: 4).",
    )
    return parser.parse_args()


async def main(query: str, role: str, k: int) -> None:
    settings = get_settings()
    store = VectorStore(
        path=settings.chroma_path,
        collection=settings.collection_name,
        embedding_model=settings.embedding_model,
    )
    hits = await retrieve(
        query,
        embeddings=get_embedding_client(),
        store=store,
        role=role,
        k=k,
    )

    for hit in hits:
        preview = hit.text[:80].replace("\n", " ")
        print(f"{hit.doc_id}\t{hit.score:.4f}\t{preview}")


if __name__ == "__main__":
    arguments = parse_args()
    asyncio.run(main(arguments.query, arguments.role, arguments.k))
