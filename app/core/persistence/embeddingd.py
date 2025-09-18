"""
Purpose: Vectorize JDs / answers / question bank for retrieval-augmented flows.
Why: Better context selection, semantic search, deduplication.

What is inside (future):

embed_texts(list[str]) -> list[vector]

VectorIndex.add(ids, vectors), search(query_vec, k)

Pluggable backends (FAISS, Chroma, pgvector).

Testing: Deterministic embeddings via mocks; search returns expected neighbors.
"""
