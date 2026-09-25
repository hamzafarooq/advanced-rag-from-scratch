"""Thin wrapper around an OpenAI-compatible chat completions endpoint.

Everything that identifies the provider lives in the environment, not the code:

    LLM_BASE_URL   endpoint (default: OpenAI's https://api.openai.com/v1)
    LLM_MODEL      model name (default: gpt-5)
    LLM_API_KEY    API key (falls back to OPENAI_API_KEY)

Because OpenRouter (chapter 6) and a local Ollama server (chapter 8) both speak
the same OpenAI-compatible protocol, this one client becomes either of them by
changing `LLM_BASE_URL` and `LLM_MODEL` -- no code changes.

The original Colab notebooks read the API key from `google.colab.userdata`.
For local conda use we read it from a `.env` file at the repo root via
`python-dotenv` instead.
"""

from __future__ import annotations

import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load .env from repo root (one level up from this chapter folder)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-5")


def get_client() -> OpenAI:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No API key. Copy .env.example to .env at the repo root and set "
            "LLM_API_KEY (or OPENAI_API_KEY)."
        )
    return OpenAI(api_key=api_key, base_url=os.getenv("LLM_BASE_URL"))


# Kept so older notebooks that import get_openai_client keep working
get_openai_client = get_client


def generate_text(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float | None = None,
    max_tokens: int = 2000,
    client: OpenAI | None = None,
) -> str:
    # temperature=None uses the provider default; reasoning models such as
    # gpt-5 reject any other value, so only pass it when explicitly set
    client = client or get_client()
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_tokens,
        **kwargs,
    )
    return completion.choices[0].message.content
