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

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    def hard_cut(t: str) -> list[str]:
        step = chunk_size - chunk_overlap if chunk_overlap else chunk_size
        return [t[i : i + chunk_size] for i in range(0, len(t), step)]

    def split(t: str) -> list[str]:
        if len(t) <= chunk_size:
            return [t]
        for sep in seps:
            if sep and sep in t:
                parts = t.split(sep)
                chunks, current = [], ""
                for part in parts:
                    part = part + sep
                    if len(part) > chunk_size:
                        if current:
                            chunks.append(current)
                            current = ""
                        chunks.extend(hard_cut(part))
                        continue
                    if len(current) + len(part) <= chunk_size:
                        current += part
                    else:
                        if current:
                            chunks.append(current)
                        current = part
                if current:
                    chunks.append(current)
                # Recurse if any single chunk is still too big
                final = []
                for c in chunks:
                    final.extend(split(c) if len(c) > chunk_size and c != t else [c])
                return final
        # No separator helped — hard cut
        return hard_cut(t)

    pieces = split(text)

    # Apply overlap by sliding the tail of chunk N onto the head of chunk N+1
    if chunk_overlap and len(pieces) > 1:
        overlapped = [pieces[0]]
        for i in range(1, len(pieces)):
            tail = pieces[i - 1][-chunk_overlap:]
            overlapped.append(tail + pieces[i])
        pieces = overlapped

    return [{"page_content": p, "metadata": metadata} for p in pieces]
