import tempfile

import chromadb

client = chromadb.PersistentClient(path=tempfile.mkdtemp())
col = client.get_or_create_collection("smoke", metadata={"hnsw:space": "cosine"})
col.add(
    ids=["a", "b"],
    embeddings=[[1.0, 0.0], [0.0, 1.0]],
    metadatas=[{"k": "x"}, {"k": "y"}],
    documents=["A", "B"],
)
print(col.query(query_embeddings=[[1.0, 0.1]], n_results=1, where={"k": "x"}))
