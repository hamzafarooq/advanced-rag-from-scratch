"""Qdrant helpers for Chapter 6a (hotels) and 6b (research papers)."""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient, models

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "chapter_04_semantic_search"))
from search import get_embeddings  # noqa: E402


def make_in_memory_client() -> QdrantClient:
    """A throwaway Qdrant client that lives in RAM. Wiped when the process ends."""
    return QdrantClient(":memory:")


def reset_collection(
    client: QdrantClient, name: str, *, vector_size: int, distance=models.Distance.COSINE
) -> None:
    """Idempotent: deletes the collection if it exists and creates a fresh one."""
    if client.collection_exists(name):
        client.delete_collection(collection_name=name)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=vector_size, distance=distance),
    )


# ---------- Chapter 6a: hotel reviews ----------

def upsert_hotel_reviews(
    client: QdrantClient, collection_name: str, reviews_df, embeddings: np.ndarray
) -> None:
    points = [
        models.PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "hotel_name": reviews_df.iloc[i]["hotel_name"],
                "review_text": reviews_df.iloc[i]["review_text"],
                "locality": reviews_df.iloc[i]["locality"],
            },
        )
        for i in range(len(embeddings))
    ]
    client.upsert(collection_name=collection_name, wait=True, points=points)


def search_hotels(
    query: str,
    embed_model,
    client: QdrantClient,
    *,
    collection_name: str = "hotel_reviews",
    city: str | None = None,
    k: int = 10,
):
    query_vec = get_embeddings([query], embed_model)[0]

    query_filter = None
    if city:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="locality", match=models.MatchValue(value=city))]
        )

    return client.query_points(
        collection_name=collection_name,
        query=query_vec.tolist(),
        query_filter=query_filter,
        limit=k,
    ).points


# ---------- Chapter 6b: research papers ----------

def upsert_text_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[dict],
    embeddings: list[np.ndarray],
) -> None:
    """Each chunk is a dict {'page_content': str, 'metadata': dict}."""
    points = [
        models.PointStruct(
            id=str(uuid4()),
            vector=np.asarray(embeddings[i]).tolist(),
            payload={
                "metadata": chunk["metadata"],
                "content": chunk["page_content"],
            },
        )
        for i, chunk in enumerate(chunks)
    ]
    client.upload_points(collection_name=collection_name, points=points)


def query_papers(
    query: str,
    embed_fn,
    client: QdrantClient,
    *,
    collection_name: str = "research_collection",
    limit: int = 5,
):
    query_vec = embed_fn(query)
    return client.query_points(
        collection_name=collection_name,
        query=query_vec.tolist() if hasattr(query_vec, "tolist") else list(query_vec),
        limit=limit,
    ).points
