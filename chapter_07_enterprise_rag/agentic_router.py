"""Agentic query router (Chapter 7, §7.2).

A single GPT-4o call classifies each user query into one of three routes:
- OPENAI_QUERY -> Qdrant collection 'opnai_data'
- 10K_DOCUMENT_QUERY -> Qdrant collection '10k_data'
- WEB_SEARCH -> live web (SerpAPI)

Trivial questions are short-circuited with a direct answer.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing from .env")
    return OpenAI(api_key=api_key)


def get_qdrant_async_client() -> AsyncQdrantClient:
    """In-memory Qdrant by default; remote when QDRANT_URL is set."""
    url = os.getenv("QDRANT_URL", ":memory:")
    api_key = os.getenv("QDRANT_API_KEY")
    if url == ":memory:":
        return AsyncQdrantClient(":memory:")
    return AsyncQdrantClient(url=url, api_key=api_key)


COLLECTIONS = {
    "OPENAI_QUERY": "opnai_data",
    "10K_DOCUMENT_QUERY": "10k_data",
}

EMBEDDING_DIM = 768  # nomic-embed-text-v1.5


@lru_cache(maxsize=1)
def _get_encoder() -> SentenceTransformer:
    """Lazy-load and cache the local Nomic encoder. Used for both retrieval and the cache."""
    return SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)


def get_text_embedding(text: str) -> list[float]:
    """Embed a query with the local Nomic model — same encoder used at ingestion."""
    encoder = _get_encoder()
    vector = encoder.encode([text], normalize_embeddings=True).astype("float32")[0]
    return vector.tolist()


def route_query(user_query: str, *, llm_model: str = "gpt-4o") -> dict:
    client = get_openai_client()
    router_system_prompt = f"""\
As a professional query router, classify user input into one of three categories:

1. "OPENAI_QUERY": Questions about OpenAI documentation -- agents, tools, APIs,
   models, embeddings, guardrails, the Responses API, or Assistants API.
2. "10K_DOCUMENT_QUERY": Questions about company financials, 10-K annual reports,
   Uber or Lyft revenue, operating costs, or filing disclosures.
3. "WEB_SEARCH": Everything else -- general knowledge, technology trends,
   comparisons, or anything not in the internal document collections.

Always respond in this exact JSON format:
{{
    "action": "OPENAI_QUERY" or "10K_DOCUMENT_QUERY" or "WEB_SEARCH",
    "reason": "one sentence justification for the routing decision",
    "answer": "AT MOST 5 words if trivially obvious, else leave empty"
}}

User: {user_query}
"""
    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "system", "content": router_system_prompt}],
        )
        text = response.choices[0].message.content
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group())
    except (OpenAIError, json.JSONDecodeError, AttributeError) as err:
        return {"action": "WEB_SEARCH", "reason": f"Routing error: {err}", "answer": ""}


def rag_formatted_response(
    user_query: str, context: list, *, llm_model: str = "gpt-4o"
) -> str:
    client = get_openai_client()
    rag_prompt = f"""\
Based on the given context, answer the user query: {user_query}
Context: {context}
Use numbered citations [1][2][3] referencing the context chunks.
Begin directly with the answer.
"""
    response = client.chat.completions.create(
        model=llm_model, messages=[{"role": "system", "content": rag_prompt}]
    )
    return response.choices[0].message.content


async def retrieve_and_respond(
    user_query: str, action: str, qdrant: AsyncQdrantClient, *, k: int = 3
) -> str:
    query_embedding = get_text_embedding(user_query)
    hits = await qdrant.query_points(
        collection_name=COLLECTIONS[action], query=query_embedding, limit=k
    )
    contents = [point.payload["content"] for point in hits.points]
    return rag_formatted_response(user_query, contents)


def search_web(user_query: str, *, max_results: int = 5) -> list:
    """SerpAPI-backed web search. Returns a list of snippet strings.

    If SERPAPI_KEY is missing, returns a tagged stub so the rest of the
    pipeline still flows during local demos.
    """
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return [
            f"[stub] Web search disabled: set SERPAPI_KEY in .env to enable. "
            f"Query was: {user_query}"
        ]
    try:
        import requests

        params = {"q": user_query, "api_key": api_key, "engine": "google"}
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        organic = data.get("organic_results", [])[:max_results]
        return [item.get("snippet", "") for item in organic if item.get("snippet")]
    except Exception as err:
        return [f"[error] SerpAPI failed ({err}). Query was: {user_query}"]


async def handle_query(user_query: str, qdrant: AsyncQdrantClient) -> str:
    route = route_query(user_query)
    print(f"Route: {route['action']}\nReason: {route['reason']}")

    if route.get("answer"):
        return route["answer"]

    if route["action"] == "WEB_SEARCH":
        return rag_formatted_response(user_query, search_web(user_query))

    return await retrieve_and_respond(user_query, route["action"], qdrant)
