"""Query rewriting and decomposition (Chapter 7, §7.4).

`rewrite_query` polishes a single user query — expands abbreviations, fills
in vague references using recent conversation history, adds clearly-implied
domain context.

`decompose_query` breaks a compound question into 2-4 atomic sub-queries
that can each be routed and retrieved independently.
"""

from __future__ import annotations

import json
import re

from agentic_router import get_openai_client


def rewrite_query(
    user_query: str,
    conversation_history: list[tuple[str, str]] | None = None,
    *,
    llm_model: str = "gpt-4o",
) -> str:
    history_context = ""
    if conversation_history:
        history_context = "\n".join(
            f"Q: {q}\nA: {a[:200]}..." for q, a in conversation_history[-3:]
        )

    rewrite_prompt = f"""\
You are a search query optimizer. Rewrite the user's query to make it more
precise and retrieval-friendly.

Rules:
1. Expand abbreviations ("Q3" -> "third quarter", "rev" -> "revenue")
2. Replace vague references with specific terms using conversation history
3. Add relevant domain context (year, company name) when clearly implied
4. Do NOT add constraints the user did not express
5. Return ONLY the rewritten query, no explanation

Conversation history:
{history_context if history_context else "None"}

Original query: {user_query}
Rewritten query:
"""
    client = get_openai_client()
    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": rewrite_prompt}],
        max_tokens=200,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def decompose_query(user_query: str, *, llm_model: str = "gpt-4o") -> list[str]:
    decompose_prompt = f"""\
Analyze the following query and determine if it contains multiple distinct
information needs. If it does, break it into 2-4 focused atomic sub-queries.
If it is already a single focused question, return it unchanged.

Rules:
- Each sub-query must be independently answerable
- Sub-queries should not overlap or repeat each other
- Preserve specific entities (company names, time periods)
- Return ONLY a JSON array of strings

Query: {user_query}
"""
    client = get_openai_client()
    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": decompose_prompt}],
        temperature=0,
    )
    try:
        text = response.choices[0].message.content.strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        return [user_query]
