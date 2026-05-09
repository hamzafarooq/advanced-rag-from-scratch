# Chapter 4 — Semantic Search from Scratch

Companion code for **Chapter 4** of *Build an Advanced RAG Application (From Scratch)*.

## What this chapter builds

Starting from a corpus of hotel reviews, we walk from a hand-rolled cosine search through to a production-grade FAISS index — covering:

- Loading the `traversaal-ai-hackathon/hotel_datasets` dataset
- Generating embeddings with `nomic-embed-text-v1.5`
- Cosine similarity in pure NumPy
- Normalized-Euclidean distance (and why it ranks identically)
- FAISS `IndexFlatIP` for exact cosine search at scale
- Comparing **Flat** vs. **HNSW** vs. **IVF-PQ** index types

## Files

| File | Purpose |
|------|---------|
| [notebook.ipynb](notebook.ipynb) | Walkthrough — read top-to-bottom |
| [data_loader.py](data_loader.py) | `load_paris_reviews()` and friends |
| [search.py](search.py) | Cosine, Euclidean, and FAISS helpers — reusable in later chapters |

## Run it

From the **repo root**:

```bash
conda activate advanced-rag
cd chapter_04_semantic_search
jupyter lab notebook.ipynb
```

No API keys are needed for this chapter — everything runs locally on CPU or GPU.
