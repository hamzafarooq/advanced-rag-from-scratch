"""Semantic search primitives: cosine, euclidean, and FAISS-backed search.

Used by the Chapter 4 walkthrough notebook. Kept dependency-light so it can be
imported into Chapter 6 too.
"""

from __future__ import annotations

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_embedding_model(
    name: str = "nomic-ai/nomic-embed-text-v1.5",
) -> SentenceTransformer:
    return SentenceTransformer(name, trust_remote_code=True)


def get_embeddings(texts, model: SentenceTransformer) -> np.ndarray:
    return model.encode(texts, show_progress_bar=False).astype("float32")


def cosine_search(query_embedding: np.ndarray, doc_embeddings: np.ndarray, k: int):
    q = query_embedding.reshape(-1)
    dots = np.dot(doc_embeddings, q)
    q_norm = np.linalg.norm(q)
    d_norms = np.linalg.norm(doc_embeddings, axis=1)
    sims = dots / (q_norm * d_norms)
    top = np.argsort(-sims)[:k]
    return top, sims[top]


def euclidean_search(query_embedding: np.ndarray, doc_embeddings: np.ndarray, k: int):
    q = query_embedding.reshape(-1) / np.linalg.norm(query_embedding)
    docs = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    distances = np.linalg.norm(docs - q, axis=1)
    top = np.argsort(distances)[:k]
    return top, distances[top]


def build_faiss_cosine_index(doc_embeddings: np.ndarray) -> faiss.Index:
    """IndexFlatIP on L2-normalized embeddings == exact cosine similarity."""
    embeddings = doc_embeddings.astype("float32")
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search_faiss_index(query_embedding: np.ndarray, index: faiss.Index, k: int = 5):
    q = query_embedding / np.linalg.norm(query_embedding)
    distances, indices = index.search(q.astype("float32"), k)
    return distances, indices


def build_faiss_indices(doc_embeddings: np.ndarray):
    """Three FAISS indices used in the chapter to compare exact vs. approximate search."""
    d = doc_embeddings.shape[1]
    embeddings = doc_embeddings.astype("float32")

    flat = faiss.IndexFlatL2(d)
    flat.add(embeddings)

    hnsw = faiss.IndexHNSWFlat(d, 32)
    hnsw.add(embeddings)

    nlist, m, bits = 100, 8, 8
    ivfpq = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, nlist, m, bits)
    ivfpq.train(embeddings)
    ivfpq.add(embeddings)

    return {"flat": flat, "hnsw": hnsw, "ivfpq": ivfpq}
