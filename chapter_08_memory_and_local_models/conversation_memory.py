"""Short-term and long-term conversation memory (Chapter 8, §8.2–§8.3).

Two layers:
- ConversationMemory  — in-session, recent window + semantic retrieval over
  turns that have aged out of the window.
- LongTermMemory      — cross-session, distilled facts extracted from turns and
  persisted per user in a separate vector store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import uuid4

from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    turn_id: str
    user_msg: str
    assistant_msg: str
    timestamp: datetime
    embedding: list[float] | None = None  # filled lazily when turn ages out of window


@dataclass
class MemoryFact:
    fact_id: str
    user_id: str
    category: str          # e.g. "interest", "preference", "role"
    content: str           # distilled fact, one sentence
    source_turn_id: str
    created_at: datetime
    expires_at: datetime
    embedding: list[float] | None = None


# ---------------------------------------------------------------------------
# Tiny in-memory vector store (replace with Qdrant/Chroma in production)
# ---------------------------------------------------------------------------

class _VectorStore:
    """Minimal cosine-search store backed by a plain list."""

    def __init__(self) -> None:
        self._items: list = []

    def upsert(self, item) -> None:
        self._items.append(item)

    def is_populated(self) -> bool:
        return len(self._items) > 0

    def search(self, query_vec: list[float], *, k: int, exclude_ids: set[str] | None = None) -> list:
        import numpy as np

        if not self._items:
            return []
        exclude_ids = exclude_ids or set()
        q = np.array(query_vec, dtype="float32")
        scored = []
        for item in self._items:
            if getattr(item, "turn_id", getattr(item, "fact_id", None)) in exclude_ids:
                continue
            if item.embedding is None:
                continue
            v = np.array(item.embedding, dtype="float32")
            score = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9))
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]

    def search_user(
        self,
        user_id: str,
        query_vec: list[float],
        *,
        k: int,
        exclude_expired: bool = True,
    ) -> list[MemoryFact]:
        now = datetime.utcnow()
        candidates = [
            f for f in self._items
            if isinstance(f, MemoryFact)
            and f.user_id == user_id
            and (not exclude_expired or f.expires_at > now)
        ]
        if not candidates:
            return []
        import numpy as np
        q = np.array(query_vec, dtype="float32")
        scored = []
        for fact in candidates:
            if fact.embedding is None:
                continue
            v = np.array(fact.embedding, dtype="float32")
            score = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9))
            scored.append((score, fact))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:k]]


# ---------------------------------------------------------------------------
# Short-term memory
# ---------------------------------------------------------------------------

_ENCODER: SentenceTransformer | None = None

def _get_encoder() -> SentenceTransformer:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    return _ENCODER


def _embed(text: str) -> list[float]:
    enc = _get_encoder()
    return enc.encode([text], normalize_embeddings=True).astype("float32")[0].tolist()


def _format_turn_for_embedding(turn: Turn) -> str:
    return f"User: {turn.user_msg}\nAssistant: {turn.assistant_msg}"


class ConversationMemory:
    """Per-session memory: recent window + semantic retrieval over older turns."""

    def __init__(self, window_size: int = 6) -> None:
        self._store = _VectorStore()
        self._turns: list[Turn] = []
        self._window_size = window_size

    def add_turn(self, user_msg: str, assistant_msg: str) -> Turn:
        # Truncate very long assistant messages so they don't bloat the window
        truncated = assistant_msg
        if len(assistant_msg) > 2000:
            truncated = assistant_msg[:2000] + "\n[truncated]"

        turn = Turn(
            turn_id=str(uuid4()),
            user_msg=user_msg,
            assistant_msg=truncated,
            timestamp=datetime.utcnow(),
        )
        self._turns.append(turn)

        # Embed the turn that just fell out of the recent window
        if len(self._turns) > self._window_size:
            old = self._turns[-(self._window_size + 1)]
            if old.embedding is None:
                old.embedding = _embed(_format_turn_for_embedding(old))
                self._store.upsert(old)
        return turn

    def get_recent(self, n: int | None = None) -> list[Turn]:
        n = n if n is not None else self._window_size
        return self._turns[-n:]

    def get_relevant(self, query: str, k: int = 3) -> list[Turn]:
        if not self._store.is_populated():
            return []
        query_vec = _embed(query)
        recent_ids = {t.turn_id for t in self.get_recent()}
        hits = self._store.search(query_vec, k=k + self._window_size, exclude_ids=recent_ids)
        return hits[:k]

    def get_context_for_rewriter(self, query: str) -> str:
        recent = self.get_recent()
        relevant = self.get_relevant(query)
        return _format_context(recent=recent, relevant=relevant)

    def __len__(self) -> int:
        return len(self._turns)


def _format_context(*, recent: list[Turn], relevant: list[Turn]) -> str:
    parts: list[str] = []
    if recent:
        parts.append("=== Recent conversation ===")
        for t in recent:
            parts.append(f"User: {t.user_msg}")
            parts.append(f"Assistant: {t.assistant_msg}")
    if relevant:
        parts.append("\n=== Earlier in this conversation (semantically related) ===")
        for t in sorted(relevant, key=lambda x: x.timestamp):
            parts.append(f"User: {t.user_msg}")
            parts.append(f"Assistant: {t.assistant_msg}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Long-term memory
# ---------------------------------------------------------------------------

_LONG_STORE = _VectorStore()  # shared across sessions; swap for a real DB


class LongTermMemory:
    """Cross-session memory: distilled facts extracted per user and recalled by query."""

    # TTLs by category
    _TTL: dict[str, timedelta] = {
        "role": timedelta(days=180),
        "preference": timedelta(days=90),
        "interest": timedelta(days=30),
        "project": timedelta(days=14),
    }
    _DEFAULT_TTL = timedelta(days=60)

    def __init__(self, extractor: "MemoryFactExtractor | None" = None) -> None:
        self._store = _LONG_STORE
        self._extractor = extractor or MemoryFactExtractor()

    def extract_and_store(self, user_id: str, turn: Turn) -> list[MemoryFact]:
        candidates = self._extractor.extract(user_id, turn)
        kept = []
        for fact in candidates:
            fact.embedding = _embed(fact.content)
            # Deduplicate: if a newer fact in the same category exists, supersede
            self._supersede(user_id, fact.category)
            self._store.upsert(fact)
            kept.append(fact)
        return kept

    def recall(self, user_id: str, query: str, k: int = 3) -> list[MemoryFact]:
        query_vec = _embed(query)
        return self._store.search_user(user_id, query_vec, k=k, exclude_expired=True)

    def _supersede(self, user_id: str, category: str) -> None:
        now = datetime.utcnow()
        for fact in self._store._items:
            if (
                isinstance(fact, MemoryFact)
                and fact.user_id == user_id
                and fact.category == category
                and fact.expires_at > now
            ):
                fact.expires_at = now  # mark expired


# ---------------------------------------------------------------------------
# Fact extractor (LLM-backed, runs asynchronously after each turn)
# ---------------------------------------------------------------------------

class MemoryFactExtractor:
    """Uses an LLM to decide what (if anything) in a turn is worth keeping long-term."""

    def __init__(self, llm_model: str = "gpt-4o-mini") -> None:
        self._llm_model = llm_model

    def extract(self, user_id: str, turn: Turn) -> list[MemoryFact]:
        import json
        import re

        from agentic_router import get_openai_client  # reuse Ch7 client

        prompt = f"""\
You are a paranoid note-taker reviewing a single conversation turn.
Your job: decide if the user revealed anything DURABLE about themselves.

Categories: role, preference, interest, project
Default answer: empty list.

Rules:
- Only extract facts directly supported by a quote from the turn.
- Prefer fewer, more precise facts over many vague ones.
- Ignore facts that are specific to this query only (not durable).
- Return ONLY a JSON array. Each element: {{"category": "...", "content": "one sentence fact"}}.

User message: {turn.user_msg}
Assistant response (first 300 chars): {turn.assistant_msg[:300]}
"""
        client = get_openai_client()
        try:
            resp = client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
            )
            text = resp.choices[0].message.content.strip()
            match = re.search(r"\[.*\]", text, re.DOTALL)
            items = json.loads(match.group()) if match else []
        except Exception:
            items = []

        now = datetime.utcnow()
        facts = []
        for item in items:
            category = item.get("category", "interest")
            ttl = LongTermMemory._TTL.get(category, LongTermMemory._DEFAULT_TTL)
            facts.append(
                MemoryFact(
                    fact_id=str(uuid4()),
                    user_id=user_id,
                    category=category,
                    content=item.get("content", ""),
                    source_turn_id=turn.turn_id,
                    created_at=now,
                    expires_at=now + ttl,
                )
            )
        return facts
