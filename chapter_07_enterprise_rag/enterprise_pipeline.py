"""Unified Enterprise RAG pipeline (Chapter 7, §7.5).

Wires the agentic router, semantic cache, and query rewriter / decomposer
into a single async entry-point. The order is:

1. Cache check (paraphrase-aware)
2. Time-sensitivity bypass (always go to web for "today", "latest", etc.)
3. Routing
4. Rewriting
5. Decomposition
6. Per-sub-query retrieval
7. Grounded synthesis
8. Cache store
"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from agentic_router import (
    COLLECTIONS,
    get_text_embedding,
    rag_formatted_response,
    route_query,
    search_web,
)
from query_rewriter import decompose_query, rewrite_query
from semantic_cache import SemanticCaching, is_time_sensitive


async def enterprise_rag_pipeline(
    user_query: str,
    cache: SemanticCaching,
    qdrant: AsyncQdrantClient,
    *,
    conversation_history: list[tuple[str, str]] | None = None,
) -> dict:
    result = {
        "query": user_query,
        "rewritten_query": None,
        "sub_queries": None,
        "route": None,
        "reason": None,
        "cache_hit": False,
        "time_sensitive": False,
        "answer": None,
    }

    # 1. Cache check
    hit, cached_answer, embedding, _sim, _idx = cache.check_cache(user_query)
    if hit:
        result["cache_hit"] = True
        result["answer"] = cached_answer
        return result

    # 2. Time-sensitivity bypass
    if is_time_sensitive(user_query):
        result["time_sensitive"] = True
        result["answer"] = rag_formatted_response(user_query, search_web(user_query))
        return result

    # 3. Route
    route = route_query(user_query)
    result["route"] = route["action"]
    result["reason"] = route["reason"]

    # Trivial direct answers from the router
    if route.get("answer"):
        result["answer"] = route["answer"]
        cache.add_to_cache(user_query, result["answer"], embedding)
        return result

    # 4. Rewrite
    rewritten = rewrite_query(user_query, conversation_history)
    result["rewritten_query"] = rewritten

    # 5. Decompose
    sub_queries = decompose_query(rewritten)
    result["sub_queries"] = sub_queries

    # 6. Retrieve per sub-query
    action = route["action"]
    all_context: list[str] = []

    if action == "WEB_SEARCH":
        for sq in sub_queries:
            all_context.extend(search_web(sq))
    else:
        collection = COLLECTIONS[action]
        for sq in sub_queries:
            sq_embedding = get_text_embedding(sq)
            hits = await qdrant.query_points(
                collection_name=collection, query=sq_embedding, limit=3
            )
            all_context.extend(p.payload["content"] for p in hits.points)

    # 7. Synthesize grounded answer
    result["answer"] = rag_formatted_response(user_query, all_context)

    # 8. Cache
    cache.add_to_cache(user_query, result["answer"], embedding)

    return result
