"""Usage: uv run python scripts/ingest.py [--reset]"""

import argparse
import asyncio
import sys
from pathlib import Path
from time import perf_counter

from project_assistant.config import get_settings
from project_assistant.llm import get_embedding_client
from project_assistant.rag.chunking import chunk_document
from project_assistant.rag.loader import mark_current, parse_document
from project_assistant.rag.models import Chunk, Document

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"


def _delete_collection_if_exists(path: str, collection_name: str) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=path)
    collection_names = {
        getattr(collection, "name", collection)
        for collection in client.list_collections()
    }
    if collection_name in collection_names:
        client.delete_collection(collection_name)


def _create_store(path: str, collection_name: str, embedding_model: str):
    from project_assistant.rag.store import VectorStore

    return VectorStore(path, collection_name, embedding_model)


async def main(reset: bool = False) -> None:
    started = perf_counter()
    settings = get_settings()

    documents: list[Document] = []
    failed_files: list[str] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        try:
            documents.append(parse_document(path))
        except (OSError, ValueError) as error:
            failed_files.append(path.name)
            print(f"{path.name}: failed to parse: {error}", file=sys.stderr)

    documents = mark_current(documents)
    chunks: list[Chunk] = [
        chunk for document in documents for chunk in chunk_document(document)
    ]
    embeddings = await get_embedding_client().embed_documents(
        [chunk.text for chunk in chunks]
    )

    if reset:
        _delete_collection_if_exists(settings.chroma_path, settings.collection_name)

    store = _create_store(
        settings.chroma_path,
        settings.collection_name,
        settings.embedding_model,
    )
    store.upsert(chunks, embeddings)

    current_count = sum(document.is_current for document in documents)
    superseded_count = len(documents) - current_count
    vector_dimension = len(embeddings[0]) if embeddings else "n/a"
    elapsed = perf_counter() - started

    print(f"Documents: {len(documents)} ({len(failed_files)} failed)")
    print(f"Chunks: {len(chunks)}")
    print(f"Current: {current_count}; superseded: {superseded_count}")
    print(f"Vector dimension: {vector_dimension}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest Markdown documents into Chroma."
    )
    parser.add_argument("--reset", action="store_true", help="Rebuild the collection.")
    arguments = parser.parse_args()
    asyncio.run(main(reset=arguments.reset))
