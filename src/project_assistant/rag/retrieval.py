from project_assistant.llm import EmbeddingClient
from project_assistant.rag.store import Hit, VectorStore

ROLE_CLEARANCE = {
    "mitarbeiter": ["intern"],
    "projektleitung": ["intern", "vertraulich"],
}


async def retrieve(
    query: str,
    *,
    embeddings: EmbeddingClient,
    store: VectorStore,
    role: str,
    k: int = 4,
    include_superseded: bool = False,
) -> list[Hit]:
    if role not in ROLE_CLEARANCE:
        raise ValueError(f"Unknown role: {role}")

    conditions = [{"vertraulichkeit": {"$in": ROLE_CLEARANCE[role]}}]
    if not include_superseded:
        conditions.append({"is_current": True})

    where_filter = {"$and": conditions} if len(conditions) > 1 else conditions[0]
    query_vector = await embeddings.embed_query(query)
    return store.query(embedding=query_vector, k=k, where=where_filter)
