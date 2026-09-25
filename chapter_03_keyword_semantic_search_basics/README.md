# Chapter 3 — Keyword Search & Semantic Search

Companion code for **Chapter 3** of *Build an Advanced RAG Application (From Scratch)*.

## What this chapter builds

Starting from a small corpus of reviews for the (fictional) Travelle hotel, we walk
through the two foundational text-retrieval approaches — and the context-length
problem that motivates chunking:

- Building an inverted index and ranking with TF-IDF
- Why exact token matching breaks, and why stemming/lemmatization can't fully fix it
- Chunking: context limits and retrieval precision
- Encoding the corpus with `nomic-embed-text-v1.5` sentence embeddings
- Cosine similarity search — keyword vs. semantic, side by side

## Files

| File | Purpose |
|------|---------|
| [notebook.ipynb](notebook.ipynb) | Walkthrough — read top-to-bottom |
| [hotel_page.txt](hotel_page.txt) | Long hotel page used for the chunking demo |

## Run it

From the **repo root**:

```bash
conda activate advanced-rag
cd chapter_03_keyword_semantic_search_basics
jupyter lab notebook.ipynb
```

No API keys needed — all models download automatically from HuggingFace on first run.
