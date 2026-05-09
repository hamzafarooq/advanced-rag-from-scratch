"""End-to-end RAG pipeline for hotel reviews (Chapter 6a).

Composes the FAISS search from Chapter 4 with an OpenRouter-hosted LLM
to produce grounded, cited answers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

# Reach into Chapter 4 for the search helpers
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "chapter_04_semantic_search"))
from search import get_embeddings, search_faiss_index  # noqa: E402

load_dotenv(_REPO_ROOT / ".env")


def get_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPEN_ROUTER_API_KEY not set. Copy .env.example to .env at the repo root and add your key."
        )
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def search_hotels_by_query(query, model, faiss_index, df, k: int = 25):
    """Run a FAISS search and return results as a list of dicts the LLM can read."""
    query_embedding = get_embeddings([query], model)
    distances, indices = search_faiss_index(query_embedding, faiss_index, k=k)

    return [
        {
            "rank": rank,
            "hotel_name": df.iloc[idx]["hotel_name"],
            "review_text": df.iloc[idx]["review_text"],
            "cosine_similarity": float(dist),
        }
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1)
    ]


def generate_answer(
    query: str,
    model,
    faiss_index,
    df,
    *,
    k: int = 25,
    llm_model: str = "qwen/qwen3-8b",
    client: OpenAI | None = None,
    stream: bool = True,
):
    """Retrieve, then prompt an LLM to synthesize a grounded markdown answer."""
    client = client or get_openrouter_client()
    sources = search_hotels_by_query(query, model, faiss_index, df, k=k)

    prompt = f"""\
Based on the following query from a user, generate a small grounded answer
focused on the original query and the retrieved context. The answer should
be in paragraphs. Remove special characters and \\n; keep the output clean.
Cite sources inline as [1][2]. Start directly with the answer — no salutations.

###########
query:
"{query}"

########
context:
{sources}
#####

Return in Markdown format.
"""

    if stream:
        stream_resp = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        output = ""
        for chunk in stream_resp:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                output += delta
        print()
        return output, sources

    completion = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content, sources
