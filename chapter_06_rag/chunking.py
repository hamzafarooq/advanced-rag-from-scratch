"""Recursive text-chunking helper used in Chapter 6b.

Splits long documents on a hierarchy of separators (paragraph → line → space → punctuation),
preserving metadata on every resulting chunk. Inspired by LangChain's RecursiveCharacterTextSplitter
but kept dependency-free.
"""

from __future__ import annotations

DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ".", ",", "，", "、", "．", "。"]


def simple_recursive_split(
    doc: dict,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    separators: list[str] | None = None,
) -> list[dict]:
    text = doc["page_content"]
    metadata = doc["metadata"]
    seps = separators or DEFAULT_SEPARATORS

    def split(t: str, sep_idx: int = 0) -> list[str]:
        if len(t) <= chunk_size:
            return [t]
        for j in range(sep_idx, len(seps)):
            sep = seps[j]
            if sep and sep in t:
                parts = t.split(sep)
                chunks, current = [], ""
                for part in parts:
                    part = part + sep
                    if len(current) + len(part) <= chunk_size:
                        current += part
                    else:
                        if current:
                            chunks.append(current)
                        current = part
                if current:
                    chunks.append(current)
                # Recurse with the NEXT separator so we always make progress
                final = []
                for c in chunks:
                    final.extend(split(c, j + 1) if len(c) > chunk_size else [c])
                return final
        # No separator helped — hard cut
        step = max(1, chunk_size - chunk_overlap)
        return [t[i : i + chunk_size] for i in range(0, len(t), step)]

    pieces = split(text)

    # Apply overlap by sliding the tail of chunk N onto the head of chunk N+1
    if chunk_overlap and len(pieces) > 1:
        overlapped = [pieces[0]]
        for i in range(1, len(pieces)):
            tail = pieces[i - 1][-chunk_overlap:]
            overlapped.append(tail + pieces[i])
        pieces = overlapped

    return [{"page_content": p, "metadata": metadata} for p in pieces]
