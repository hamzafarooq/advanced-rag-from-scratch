"""FAISS-backed semantic cache for the Enterprise RAG pipeline (Chapter 7, §7.3).

Caches answers keyed on the *meaning* of the question (via embeddings), so a
paraphrased question hits the cache too. Time-sensitive questions are detected
upfront and bypass the cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


TIME_SENSITIVE_KEYWORDS = [
    "today", "tonight", "now", "currently", "current",
    "latest", "recent", "recently", "right now",
    "at the moment", "at present", "as of now",
    "this week", "this month", "this year",
    "this quarter", "this season", "this morning",
    "this afternoon", "this evening", "this weekend",
    "yesterday", "tomorrow", "last week", "last month",
    "last year", "upcoming", "live", "breaking",
    "just happened", "what time", "what day", "what date",
    "happening now", "events today", "news today",
    "news this week", "stock price", "share price",
    "weather", "forecast", "temperature",
    "real-time", "realtime", "schedule today",
    "outage", "down right now",
]


def is_time_sensitive(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in TIME_SENSITIVE_KEYWORDS)


class SemanticCaching:
    """Persistent semantic cache over a FAISS L2 index."""

    def __init__(
        self,
        json_file: str = "cache.json",
        threshold: float = 0.2,
        embedding_dim: int = 768,
        encoder_name: str = "nomic-ai/nomic-embed-text-v1.5",
        clear_on_init: bool = False,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.json_file = Path(json_file)
        self.euclidean_threshold = threshold
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.encoder = SentenceTransformer(encoder_name, trust_remote_code=True)
        self.cache: dict = {"questions": [], "embeddings": [], "response_text": []}

        if clear_on_init:
            self.clear_cache()
        else:
            self.load_cache()

    def load_cache(self) -> None:
        try:
            with self.json_file.open("r") as f:
                self.cache = json.load(f)
            if self.cache["embeddings"]:
                vectors = np.array(self.cache["embeddings"], dtype=np.float32)
                self.index.add(vectors)
        except FileNotFoundError:
            self.cache = {"questions": [], "embeddings": [], "response_text": []}

    def check_cache(self, question: str):
        embedding = self.encoder.encode([question], normalize_embeddings=True).astype(
            "float32"
        )
        if self.index.ntotal == 0:
            return False, None, embedding, None, None

        distances, indices = self.index.search(embedding, 1)
        idx = int(indices[0][0])
        dist = float(distances[0][0])
        if idx != -1 and dist <= self.euclidean_threshold:
            return (
                True,
                self.cache["response_text"][idx],
                embedding,
                1.0 - dist,
                idx,
            )
        return False, None, embedding, None, None

    def add_to_cache(self, question: str, answer: str, embedding) -> None:
        self.cache["questions"].append(question)
        self.cache["embeddings"].append(embedding[0].tolist())
        self.cache["response_text"].append(answer)
        self.index.add(embedding)
        self.save_cache()

    def save_cache(self) -> None:
        with self.json_file.open("w") as f:
            json.dump(self.cache, f)

    def clear_cache(self) -> None:
        self.cache = {"questions": [], "embeddings": [], "response_text": []}
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.save_cache()
