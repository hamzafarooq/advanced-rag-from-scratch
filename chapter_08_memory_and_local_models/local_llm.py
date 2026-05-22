"""Ollama-backed local LLM client (Chapter 8, §8.4).

Provides a drop-in wrapper around Ollama's OpenAI-compatible HTTP endpoint.
Used for tasks that must not leave the machine — fact extraction from
conversation turns, input inspection before guardrails, and any step where
the data is too sensitive for a third-party API call.

Prerequisites:
    brew install ollama
    ollama pull llama3.2        # 3 B, fast on CPU
    ollama pull nomic-embed-text  # embeddings (optional, if you want fully local)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
_OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


@lru_cache(maxsize=1)
def get_ollama_client() -> OpenAI:
    """Return an OpenAI client pointed at the local Ollama server."""
    return OpenAI(base_url=_OLLAMA_BASE_URL, api_key="ollama")


def local_chat(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """Single-turn chat completion via Ollama.

    Returns the assistant message string, or raises if Ollama is unreachable.
    """
    client = get_ollama_client()
    model = model or _OLLAMA_DEFAULT_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def is_ollama_available(model: str | None = None) -> bool:
    """Return True if Ollama is running and the target model is loaded."""
    try:
        result = local_chat("ping", model=model, max_tokens=5)
        return isinstance(result, str)
    except Exception:
        return False
