"""Production RAG pipeline — full stack (Chapter 9, §9.5).

Extends the Chapter 8 memory pipeline with:
1. Input guardrail  — inspects query before any processing (local Ollama)
2. Access control   — filters collection routes by user role
3. Output guardrail — checks answer for hallucination / policy issues
4. Provenance log   — records sources and context for every answer

This is the end-state pipeline the book has been building toward.
"""

from __future__ import annotations

import sys
from pathlib import Path

from qdrant_client import AsyncQdrantClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapter_07_enterprise_rag"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapter_08_memory_and_local_models"))

from agentic_router import (
    COLLECTIONS,
    get_text_embedding,
    rag_formatted_response,
    route_query,
    search_web,
)
from query_rewriter import decompose_query
from semantic_cache import SemanticCaching, is_time_sensitive
from conversation_memory import ConversationMemory, LongTermMemory
from pipeline_ch8 import _rewrite_with_memory

from access_control import AccessControl, ProvenanceTracker, SourceChunk
from guardrails import InputGuardrail, OutputGuardrail


# Module-level singletons — instantiate once per process
_input_guardrail = InputGuardrail()
_output_guardrail = OutputGuardrail()
_access_control = AccessControl()
_provenance = ProvenanceTracker()


async def production_rag_pipeline(
    user_query: str,
    cache: SemanticCaching,
    qdrant: AsyncQdrantClient,
    short_term: ConversationMemory,
    long_term: LongTermMemory,
    *,
    user_id: str = "anonymous",
    role: str = "readonly",
) -> dict:
    result = {
        "query": user_query,
        "user_id": user_id,
        "role": role,
        "rewritten_query": None,
        "sub_queries": None,
        "route": None,
        "reason": None,
        "cache_hit": False,
        "time_sensitive": False,
        "memory_facts": [],
        "guardrail_input": None,
        "guardrail_output": None,
        "sources": [],
        "provenance_id": None,
        "answer": None,
        "blocked": False,
    }

    # 1. Input guardrail — runs locally, never sends untrusted content to a remote API
    input_check = _input_guardrail.check(user_query)
    result["guardrail_input"] = {"passed": input_check.passed, "reason": input_check.reason}
    if not input_check.passed:
        result["blocked"] = True
        result["answer"] = (
            f"Your query was blocked by the input guardrail ({input_check.category}). "
            "Please rephrase and try again."
        )
        return result

    # 2. Cache check
    hit, cached_answer, embedding, _sim, _idx = cache.check_cache(user_query)
    if hit:
        result["cache_hit"] = True
        result["answer"] = cached_answer
        return result

    # 3. Time-sensitivity bypass
    if is_time_sensitive(user_query):
        result["time_sensitive"] = True
        answer = rag_formatted_response(user_query, search_web(user_query))
        result["answer"] = answer
        turn = short_term.add_turn(user_query, answer)
        long_term.extract_and_store(user_id, turn)
        return result

    # 4. Route + access control
    route = route_query(user_query)
    permitted_action = _access_control.filter_route(role, route["action"])
    result["route"] = permitted_action
    result["reason"] = route["reason"] + (
        f" [downgraded from {route['action']} due to role permissions]"
        if permitted_action != route["action"] else ""
    )

    if route.get("answer"):
        result["answer"] = route["answer"]
        cache.add_to_cache(user_query, result["answer"], embedding)
        return result

    # 5. Rewrite with short-term memory
    rewritten = _rewrite_with_memory(user_query, short_term)
    result["rewritten_query"] = rewritten

    # 6. Long-term memory recall
    facts = long_term.recall(user_id, rewritten)
    result["memory_facts"] = [f.content for f in facts]

    # 7. Decompose
    sub_queries = decompose_query(rewritten)
    result["sub_queries"] = sub_queries

    # 8. Retrieve per sub-query
    action = permitted_action
    all_context: list[str] = list(result["memory_facts"])
    source_chunks: list[SourceChunk] = []

    if action == "WEB_SEARCH":
        for sq in sub_queries:
            snippets = search_web(sq)
            all_context.extend(snippets)
            for s in snippets:
                source_chunks.append(SourceChunk(
                    chunk_id="web", collection="web", content_preview=s[:200], score=0.0
                ))
    else:
        collection = COLLECTIONS[action]
        for sq in sub_queries:
            sq_embedding = get_text_embedding(sq)
            hits = await qdrant.query_points(
                collection_name=collection, query=sq_embedding, limit=3
            )
            for p in hits.points:
                all_context.append(p.payload["content"])
                source_chunks.append(SourceChunk(
                    chunk_id=str(p.id),
                    collection=collection,
                    content_preview=p.payload["content"][:200],
                    score=p.score if hasattr(p, "score") else 0.0,
                ))

    result["sources"] = [
        {"collection": s.collection, "preview": s.content_preview, "score": s.score}
        for s in source_chunks
    ]

    # 9. Synthesize
    answer = rag_formatted_response(user_query, all_context)

    # 10. Output guardrail
    output_check = _output_guardrail.check(user_query, all_context, answer)
    result["guardrail_output"] = {"passed": output_check.passed, "reason": output_check.reason}
    if not output_check.passed:
        result["blocked"] = True
        result["answer"] = (
            "The generated answer was blocked by the output guardrail "
            f"({output_check.category}: {output_check.reason}). "
            "Please try a more specific question."
        )
        return result

    result["answer"] = answer

    # 11. Cache + memory update
    cache.add_to_cache(user_query, answer, embedding)
    turn = short_term.add_turn(user_query, answer)
    long_term.extract_and_store(user_id, turn)

    # 12. Provenance record
    rec = _provenance.record(
        user_id=user_id,
        role=role,
        query=user_query,
        rewritten_query=rewritten,
        answer=answer,
        sources=source_chunks,
        guardrail_passed=True,
    )
    result["provenance_id"] = rec.record_id

    return result


def explain_answer(provenance_id: str) -> str:
    return _provenance.explain(provenance_id)
