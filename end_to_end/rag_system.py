"""End-to-end RAG system — complete stack (Chapters 3–9).

Single entry-point that wires every layer built across the book:

  Ch3/4  Chunking + Nomic embeddings + Qdrant ingestion
  Ch5    OpenAI LLM synthesis
  Ch6    Basic RAG pipeline
  Ch7    Agentic routing, semantic cache, query rewriting/decomposition
  Ch8    Short-term + long-term conversation memory, local Ollama model
  Ch9    Input/output guardrails, role-based access control, provenance

Public API
----------
  RAGSystem(role, user_id)     — construct a session
  .ingest(paths)               — load documents into Qdrant
  .chat(query)    -> RAGResponse
  .explain(id)    -> str       — why did the system say that?
  .history()      -> list      — all turns this session
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Path wiring — import from each chapter directory
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent

for _chapter in [
    "chapter_07_enterprise_rag",
    "chapter_08_memory_and_local_models",
    "chapter_09_guardrails_and_production",
    "chapter_06_rag",
]:
    _p = str(_ROOT / _chapter)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Ch7 — routing, cache, web search, synthesis
from agentic_router import (  # noqa: E402
    COLLECTIONS,
    get_qdrant_async_client,
    get_text_embedding,
    rag_formatted_response,
    route_query,
    search_web,
)
from query_rewriter import decompose_query  # noqa: E402
from semantic_cache import SemanticCaching, is_time_sensitive  # noqa: E402
from ingest import ingest_all  # noqa: E402

# Ch8 — memory + local LLM
from conversation_memory import ConversationMemory, LongTermMemory  # noqa: E402
from pipeline_ch8 import _rewrite_with_memory  # noqa: E402

# Ch9 — guardrails, access control, provenance
from guardrails import InputGuardrail, OutputGuardrail  # noqa: E402
from access_control import (  # noqa: E402
    AccessControl,
    ProvenanceTracker,
    SourceChunk,
    ROLE_PERMISSIONS,
)


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    query: str
    answer: str
    blocked: bool = False
    block_reason: str = ""
    route: str = ""
    rewritten_query: str = ""
    sub_queries: list[str] = field(default_factory=list)
    cache_hit: bool = False
    time_sensitive: bool = False
    memory_facts: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    provenance_id: str | None = None

    def __str__(self) -> str:
        if self.blocked:
            return f"[BLOCKED] {self.block_reason}"
        lines = [self.answer]
        if self.sources:
            lines.append(f"\nSources: {[s['collection'] for s in self.sources]}")
        if self.provenance_id:
            lines.append(f"Provenance: {self.provenance_id}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RAGSystem
# ---------------------------------------------------------------------------

class RAGSystem:
    """
    Single-session RAG system. Instantiate once per user session.

    Parameters
    ----------
    role : str
        One of 'admin', 'analyst', 'developer', 'readonly'.
        Controls which Qdrant collections the session may query.
    user_id : str
        Stable identifier for this user (used for long-term memory recall).
    cache_path : str
        Path for the semantic cache JSON file. Defaults to a temp file.
    memory_window : int
        Number of recent turns kept verbatim in short-term memory.
    """

    def __init__(
        self,
        role: str = "admin",
        user_id: str = "anonymous",
        cache_path: str = "cache.json",
        memory_window: int = 6,
    ) -> None:
        self.role = role
        self.user_id = user_id

        # One Qdrant client shared for the session
        self._qdrant = get_qdrant_async_client()

        # Ch7 — semantic cache
        self._cache = SemanticCaching(json_file=cache_path)

        # Ch8 — memory layers
        self._short_term = ConversationMemory(window_size=memory_window)
        self._long_term = LongTermMemory()

        # Ch9 — guardrails, access, provenance
        self._input_guard = InputGuardrail()
        self._output_guard = OutputGuardrail()
        self._access = AccessControl()
        self._provenance = ProvenanceTracker()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, paths: list[str | Path] | None = None) -> None:
        """
        Ingest documents into Qdrant.

        Passing no paths uses the default corpus shipped with the repo
        (OpenAI agents guide + Uber/Lyft 10-Ks, same as Chapters 7-9).
        Pass a list of PDF/HTM paths to ingest your own documents.
        """
        if paths:
            self._ingest_custom(paths)
        else:
            asyncio.run(ingest_all(self._qdrant))
            print("Default corpus ingested (OpenAI guide + 10-K filings).")

    def _ingest_custom(self, paths: list[str | Path]) -> None:
        import sys
        from pathlib import Path as _Path
        sys.path.insert(0, str(_ROOT / "chapter_07_enterprise_rag"))
        from ingest import chunk_documents, build_collection
        from agentic_router import COLLECTIONS, EMBEDDING_DIM

        resolved = [_Path(p) for p in paths]
        chunks = chunk_documents(resolved)
        # All custom documents go into a 'custom_data' collection
        import asyncio as _asyncio
        from qdrant_client import models
        async def _build():
            col = "custom_data"
            if not await self._qdrant.collection_exists(col):
                await self._qdrant.create_collection(
                    col,
                    vectors_config=models.VectorParams(
                        size=EMBEDDING_DIM, distance=models.Distance.COSINE
                    ),
                )
            await build_collection(self._qdrant, col, chunks)
        _asyncio.run(_build())
        print(f"Custom corpus ingested: {len(chunks)} chunks.")

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(self, query: str) -> RAGResponse:
        """Run the full pipeline for a single user turn. Synchronous wrapper."""
        return asyncio.run(self._chat_async(query))

    async def _chat_async(self, query: str) -> RAGResponse:
        resp = RAGResponse(query=query, answer="")

        # 1. Input guardrail
        input_check = self._input_guard.check(query)
        if not input_check.passed:
            resp.blocked = True
            resp.block_reason = f"{input_check.category}: {input_check.reason}"
            resp.answer = (
                f"Your query was blocked ({input_check.category}). "
                "Please rephrase and try again."
            )
            return resp

        # 2. Semantic cache
        hit, cached_answer, embedding, _sim, _idx = self._cache.check_cache(query)
        if hit:
            resp.cache_hit = True
            resp.answer = cached_answer
            return resp

        # 3. Time-sensitivity bypass
        if is_time_sensitive(query):
            resp.time_sensitive = True
            answer = rag_formatted_response(query, search_web(query))
            resp.answer = answer
            turn = self._short_term.add_turn(query, answer)
            self._long_term.extract_and_store(self.user_id, turn)
            return resp

        # 4. Route + access control
        route = route_query(query)
        permitted_action = self._access.filter_route(self.role, route["action"])
        resp.route = permitted_action
        if route.get("answer"):
            resp.answer = route["answer"]
            self._cache.add_to_cache(query, resp.answer, embedding)
            return resp

        # 5. Rewrite with short-term memory context
        rewritten = _rewrite_with_memory(query, self._short_term)
        resp.rewritten_query = rewritten

        # 6. Long-term memory recall
        facts = self._long_term.recall(self.user_id, rewritten)
        resp.memory_facts = [f.content for f in facts]

        # 7. Decompose
        sub_queries = decompose_query(rewritten)
        resp.sub_queries = sub_queries

        # 8. Retrieve
        all_context: list[str] = list(resp.memory_facts)
        source_chunks: list[SourceChunk] = []

        if permitted_action == "WEB_SEARCH":
            for sq in sub_queries:
                snippets = search_web(sq)
                all_context.extend(snippets)
                for s in snippets:
                    source_chunks.append(SourceChunk("web", "web", s[:200], 0.0))
        else:
            collection = COLLECTIONS.get(permitted_action, "custom_data")
            for sq in sub_queries:
                sq_vec = get_text_embedding(sq)
                hits = await self._qdrant.query_points(
                    collection_name=collection, query=sq_vec, limit=3
                )
                for p in hits.points:
                    all_context.append(p.payload["content"])
                    source_chunks.append(SourceChunk(
                        str(p.id), collection,
                        p.payload["content"][:200],
                        p.score if hasattr(p, "score") else 0.0,
                    ))

        resp.sources = [
            {"collection": s.collection, "preview": s.content_preview, "score": s.score}
            for s in source_chunks
        ]

        # 9. Synthesize
        answer = rag_formatted_response(query, all_context)

        # 10. Output guardrail
        out_check = self._output_guard.check(query, all_context, answer)
        if not out_check.passed:
            resp.blocked = True
            resp.block_reason = f"{out_check.category}: {out_check.reason}"
            resp.answer = (
                f"The answer was blocked by the output guardrail ({out_check.category}). "
                "Please try a more specific question."
            )
            return resp

        resp.answer = answer

        # 11. Cache + memory
        self._cache.add_to_cache(query, answer, embedding)
        turn = self._short_term.add_turn(query, answer)
        self._long_term.extract_and_store(self.user_id, turn)

        # 12. Provenance
        rec = self._provenance.record(
            user_id=self.user_id,
            role=self.role,
            query=query,
            rewritten_query=rewritten,
            answer=answer,
            sources=source_chunks,
            guardrail_passed=True,
        )
        resp.provenance_id = rec.record_id

        return resp

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def explain(self, provenance_id: str) -> str:
        """Return a human-readable breakdown of what sourced an answer."""
        return self._provenance.explain(provenance_id)

    def history(self) -> list[dict]:
        """Return all turns this session as plain dicts."""
        return [
            {
                "turn": i + 1,
                "user": t.user_msg,
                "assistant": t.assistant_msg[:200] + "...",
                "timestamp": t.timestamp.isoformat(),
            }
            for i, t in enumerate(self._short_term._turns)
        ]

    def clear_cache(self) -> None:
        self._cache.clear_cache()

    def __repr__(self) -> str:
        return (
            f"RAGSystem(role={self.role!r}, user_id={self.user_id!r}, "
            f"turns={len(self._short_term)})"
        )
