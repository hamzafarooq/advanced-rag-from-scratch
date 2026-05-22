"""Role-based access control and provenance tracking (Chapter 9, §9.4).

Two responsibilities:
- AccessControl  — decides which Qdrant collections a user may query,
  based on a simple role -> collection allowlist.
- ProvenanceTracker — records which source chunks contributed to each answer,
  so users can ask "why did you say that?" and get a real answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------

# Maps role name -> set of allowed collection names (from Ch7 COLLECTIONS dict).
# Add roles and extend collections as the system grows.
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin":      {"opnai_data", "10k_data"},
    "analyst":    {"10k_data"},
    "developer":  {"opnai_data"},
    "readonly":   {},   # web search only; no internal collections
}


class AccessDenied(Exception):
    pass


class AccessControl:
    """Enforces collection-level access by user role."""

    def __init__(self, role_map: dict[str, set[str]] | None = None) -> None:
        self._roles = role_map or ROLE_PERMISSIONS

    def get_allowed_collections(self, role: str) -> set[str]:
        return self._roles.get(role, set())

    def check(self, role: str, collection: str) -> None:
        """Raise AccessDenied if the role cannot query collection."""
        allowed = self.get_allowed_collections(role)
        if collection not in allowed:
            raise AccessDenied(
                f"Role '{role}' is not permitted to access collection '{collection}'. "
                f"Allowed: {sorted(allowed) or ['none (web search only)']}"
            )

    def filter_route(self, role: str, action: str) -> str:
        """Downgrade a router action if the role lacks permission for the collection.

        WEB_SEARCH is always allowed. For collection-backed routes, falls back
        to WEB_SEARCH if the role cannot access the collection.
        """
        if action == "WEB_SEARCH":
            return action
        from chapter_07_enterprise_rag.agentic_router import COLLECTIONS  # noqa
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapter_07_enterprise_rag"))
        from agentic_router import COLLECTIONS
        collection = COLLECTIONS.get(action)
        if collection and collection not in self.get_allowed_collections(role):
            return "WEB_SEARCH"
        return action


# ---------------------------------------------------------------------------
# Provenance tracking
# ---------------------------------------------------------------------------

@dataclass
class SourceChunk:
    chunk_id: str
    collection: str
    content_preview: str   # first 200 chars
    score: float


@dataclass
class ProvenanceRecord:
    record_id: str
    user_id: str
    role: str
    query: str
    rewritten_query: str | None
    answer_preview: str    # first 300 chars
    sources: list[SourceChunk]
    guardrail_passed: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ProvenanceTracker:
    """In-memory provenance log. Swap for a database in production."""

    def __init__(self) -> None:
        self._records: list[ProvenanceRecord] = []

    def record(
        self,
        *,
        user_id: str,
        role: str,
        query: str,
        rewritten_query: str | None,
        answer: str,
        sources: list[SourceChunk],
        guardrail_passed: bool,
    ) -> ProvenanceRecord:
        rec = ProvenanceRecord(
            record_id=str(uuid4()),
            user_id=user_id,
            role=role,
            query=query,
            rewritten_query=rewritten_query,
            answer_preview=answer[:300],
            sources=sources,
            guardrail_passed=guardrail_passed,
        )
        self._records.append(rec)
        return rec

    def get_user_history(self, user_id: str, *, limit: int = 20) -> list[ProvenanceRecord]:
        user_recs = [r for r in self._records if r.user_id == user_id]
        return user_recs[-limit:]

    def explain(self, record_id: str) -> str:
        """Return a human-readable explanation of why the answer was generated."""
        rec = next((r for r in self._records if r.record_id == record_id), None)
        if not rec:
            return f"No record found for id {record_id}"

        lines = [
            f"Query: {rec.query}",
            f"Rewritten: {rec.rewritten_query or '(unchanged)'}",
            f"Role: {rec.role}",
            f"Guardrail passed: {rec.guardrail_passed}",
            f"\nSources used ({len(rec.sources)}):",
        ]
        for i, src in enumerate(rec.sources, 1):
            lines.append(f"  [{i}] {src.collection} (score={src.score:.3f}): {src.content_preview[:120]}...")
        return "\n".join(lines)
