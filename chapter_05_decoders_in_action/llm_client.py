"""Thin wrapper around the OpenAI Chat Completions API.

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


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Copy .env.example to .env at the repo root and add your key."
        )
    return OpenAI(api_key=api_key)


def generate_text(
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.2,
    max_tokens: int = 1000,
    client: OpenAI | None = None,
) -> str:
    client = client or get_openai_client()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content
