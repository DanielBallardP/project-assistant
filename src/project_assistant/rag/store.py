import chromadb
from pydantic import BaseModel

from project_assistant.rag.models import Chunk


class Hit(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    metadata: dict


class VectorStore:
    def __init__(self, path: str, collection: str, embedding_model: str) -> None:
        self._client = chromadb.PersistentClient(path=path)
        self._collection_metadata = {
            "hnsw:space": "cosine",
            "embedding_model": embedding_model,
        }
        self._col = self._client.get_or_create_collection(
            collection,
            metadata=self._collection_metadata.copy(),
        )
        stored_model = (self._col.metadata or {}).get("embedding_model")
        if stored_model != embedding_model:
            raise ValueError(
                f"Collection {collection!r} uses embedding model "
                f"{stored_model!r}, but {embedding_model!r} is configured. "
                "Rebuild the collection with --reset."
            )

    def reset(self) -> None:
        collection_name = self._col.name
        self._client.delete_collection(collection_name)
        self._col = self._client.create_collection(
            collection_name,
            metadata=self._collection_metadata.copy(),
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("The number of chunks must match the number of embeddings")
        if not chunks:
            return

        self._col.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
        )

    def query(self, embedding: list[float], k: int, where: dict | None) -> list[Hit]:
        if k < 1:
            raise ValueError("k must be greater than zero")

        results = self._col.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        if not results["ids"]:
            raise RuntimeError("Chroma query returned no result groups")

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        hits: list[Hit] = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            if text is None or metadata is None:
                raise RuntimeError(
                    f"Chroma query omitted document content for chunk {chunk_id!r}"
                )

            doc_id = metadata.get("doc_id")
            title = metadata.get("title")
            if not isinstance(doc_id, str) or not isinstance(title, str):
                raise TypeError(
                    f"Chroma metadata is missing doc_id or title for chunk {chunk_id!r}"
                )

            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    title=title,
                    text=text,
                    score=1 - distance,
                    metadata=metadata,
                )
            )
        return hits
