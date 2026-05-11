# Chapter 3 — Keyword Search & Semantic Search

Companion code for **Chapter 3** of *Build an Advanced RAG Application (From Scratch)*.

## What this chapter builds

Starting from a handful of sample sentences, we walk through the two foundational
text-retrieval approaches — and the context-length problem that motivates both:

- Why transformer models need chunking (the 512-token BERT limit)
- Building an inverted index and ranking with TF-IDF
- Encoding documents with `all-MiniLM-L6-v2` sentence embeddings
- Cosine similarity search — exact token matching vs. semantic meaning

## Files

| File | Purpose |
|------|---------|
| [notebook.ipynb](notebook.ipynb) | Walkthrough — read top-to-bottom |
| [bond_article.txt](bond_article.txt) | Sample long article for the context-length demo |

## Run it

From the **repo root**:

```bash
conda activate advanced-rag
cd chapter_03_basic_keyword_semantic
jupyter lab notebook.ipynb
```

No API keys needed — all models download automatically from HuggingFace on first run.
