# Chapter 6 — Retrieval-Augmented Generation (RAG)

Companion code for **Chapter 6** of *Build an Advanced RAG Application (From Scratch)*.

This chapter wires retrieval (Ch. 4) and generation (Ch. 5) together end-to-end, on two different corpora.

## Notebooks

| Notebook | What it covers |
|----------|----------------|
| [06a_rag_pipeline.ipynb](06a_rag_pipeline.ipynb) | Hotels — FAISS + LLM, then upgrade to **Qdrant** with city-level metadata filtering |
| [06b_research_papers.ipynb](06b_research_papers.ipynb) | Research papers — chunking, `transformers` directly, year filters, full RAG |

## Reusable modules

| File | Purpose |
|------|---------|
| [rag_pipeline.py](rag_pipeline.py) | `search_hotels_by_query`, `generate_answer` (FAISS + OpenRouter, streaming) |
| [qdrant_helpers.py](qdrant_helpers.py) | Collection setup, upserts, filtered search for both corpora |
| [chunking.py](chunking.py) | `simple_recursive_split` — dependency-free recursive text splitter |

## Run it

```bash
conda activate advanced-rag
cd chapter_06_rag
jupyter lab 06a_rag_pipeline.ipynb     # then 06b
```

### API keys (copy `.env.example` → `.env` at the repo root)

| Key | Used by |
|-----|---------|
| `OPEN_ROUTER_API_KEY` | both notebooks (LLM generation) |
| `KAGGLE_USERNAME` + `KAGGLE_KEY` | 6b only (downloads the DBLP papers dataset) |

OpenRouter (https://openrouter.ai) is a unified gateway over many LLMs — Qwen, GPT, Claude, Mistral — with one API key. The notebooks default to a small free Qwen model; swap `llm_model=` for any model name OpenRouter exposes.
