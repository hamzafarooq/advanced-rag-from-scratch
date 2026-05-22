"""Memory-augmented Enterprise RAG pipeline (Chapter 8, §8.5).

Extends the Chapter 7 pipeline with two new layers:
1. Short-term ConversationMemory — passes recent + relevant turns to the rewriter.
2. LongTermMemory recall — prepends known user facts to the synthesis prompt.

The Chapter 7 rewriter changes by exactly one function argument: it now receives
memory context alongside the raw query. Everything else is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

from qdrant_client import AsyncQdrantClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapter_07_enterprise_rag"))
from agentic_router import (
    COLLECTIONS,
    get_text_embedding,
    rag_formatted_response,
    route_query,
    search_web,
)
from query_rewriter import decompose_query, rewrite_query
from semantic_cache import SemanticCaching, is_time_sensitive

from conversation_memory import ConversationMemory, LongTermMemory, MemoryFactExtractor


def _rewrite_with_memory(user_query: str, memory: ConversationMemory) -> str:
    context = memory.get_context_for_rewriter(user_query)
    # Pass memory context as a single-item history tuple so the Ch7 rewriter
    # sees it without needing a structural change to its signature.
    history = [("context", context)] if context else None
    return rewrite_query(user_query, history)


async def memory_rag_pipeline(
    user_query: str,
    cache: SemanticCaching,
    qdrant: AsyncQdrantClient,
    short_term: ConversationMemory,
    long_term: LongTermMemory,
    user_id: str = "anonymous",
) -> dict:
    result = {
        "query": user_query,
        "rewritten_query": None,
        "sub_queries": None,
        "route": None,
        "reason": None,
        "cache_hit": False,
        "time_sensitive": False,
        "memory_facts": [],
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
        answer = rag_formatted_response(user_query, search_web(user_query))
        result["answer"] = answer
        turn = short_term.add_turn(user_query, answer)
        long_term.extract_and_store(user_id, turn)
        return result

    # 3. Route
    route = route_query(user_query)
    result["route"] = route["action"]
    result["reason"] = route["reason"]

    if route.get("answer"):
        result["answer"] = route["answer"]
        cache.add_to_cache(user_query, result["answer"], embedding)
        return result

    # 4. Rewrite with short-term memory context
    rewritten = _rewrite_with_memory(user_query, short_term)
    result["rewritten_query"] = rewritten

    # 5. Recall long-term facts and inject into synthesis
    facts = long_term.recall(user_id, rewritten)
    result["memory_facts"] = [f.content for f in facts]

    # 6. Decompose
    sub_queries = decompose_query(rewritten)
    result["sub_queries"] = sub_queries

    # 7. Retrieve per sub-query
    action = route["action"]
    all_context: list[str] = list(result["memory_facts"])  # seed with user facts

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

    # 8. Synthesize
    answer = rag_formatted_response(user_query, all_context)
    result["answer"] = answer

    # 9. Cache and record turn
    cache.add_to_cache(user_query, answer, embedding)
    turn = short_term.add_turn(user_query, answer)
    long_term.extract_and_store(user_id, turn)

    return result
