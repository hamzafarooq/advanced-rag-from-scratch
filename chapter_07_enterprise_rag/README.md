# Chapter 7 — Enterprise RAG: Agentic Routing, Semantic Caching, and Query Rewriting

Companion code for **Chapter 7** of *Build an Advanced RAG Application (From Scratch)*.

Plain RAG starts to crack in enterprise settings — multiple knowledge bases, paraphrased questions hammering the LLM, vague or compound queries that no single search nails. This chapter fixes all three.

> [!NOTE]
> Source code adapted from the original Colab at [hamzafarooq/multi-agent-course/Module_3_Agentic_RAG](https://github.com/hamzafarooq/multi-agent-course/tree/main/Module_3_Agentic_RAG). The PDFs in `data/` are the same corpora that course used.

## What this chapter builds

| Pillar | What it does | Module |
|--------|--------------|--------|
| **Agentic Router** | LLM-based intent classifier that dispatches each query to the right knowledge base (or web) | [agentic_router.py](agentic_router.py) |
| **Semantic Cache** | FAISS-backed cache keyed on Nomic embeddings — paraphrases hit | [semantic_cache.py](semantic_cache.py) |
| **Query Rewriter / Decomposer** | Polishes vague queries; splits compound ones | [query_rewriter.py](query_rewriter.py) |
| **Enterprise RAG pipeline** | All three plus a time-sensitivity bypass, in one orchestrated async function | [enterprise_pipeline.py](enterprise_pipeline.py) |
| **Ingestion** | PyMuPDF -> recursive chunker -> Nomic -> Qdrant | [ingest.py](ingest.py) |

## Data

All five source documents are shipped in [`data/`](data/) so the notebook is fully self-contained:

| File | Source | Routes to |
|------|--------|-----------|
| `openai_agents_guide.pdf` | OpenAI's *A Practical Guide to Building Agents* | `opnai_data` |
| `lyft_10k_2020.pdf` | Lyft Inc. Form 10-K, fiscal 2020 | `10k_data` |
| `lyft_10k_2021.pdf` | Lyft Inc. Form 10-K, fiscal 2021 | `10k_data` |
| `lyft_10k_2022.pdf` | Lyft Inc. Form 10-K, fiscal 2022 | `10k_data` |
| `uber_10k_2021.htm` | Uber Technologies Form 10-K from SEC EDGAR, fiscal 2021 | `10k_data` |

**Pipeline**: PyMuPDF text extraction -> `simple_recursive_split` (`chunk_size=2048`, `overlap=50`, reused from Chapter 6) -> `nomic-embed-text-v1.5` (768-dim, normalized) -> Qdrant `cosine` distance.

## Run it

```bash
conda activate advanced-rag
cd chapter_07_enterprise_rag
jupyter lab notebook.ipynb
```

The first cell after setup ingests all five documents into an in-memory Qdrant client. Embedding takes ~2 minutes on CPU, much faster on GPU/MPS. To persist across sessions, set `QDRANT_URL` to a real cluster.

### API keys (copy `.env.example` -> `.env` at the repo root)

| Key | Required? | Used for |
|-----|-----------|----------|
| `OPENAI_API_KEY` | yes | LLM calls (router, rewriter, decomposer, response generator) |
| `QDRANT_URL` | optional | Remote Qdrant URL. Falls back to `:memory:` when missing |
| `QDRANT_API_KEY` | optional | Cloud Qdrant auth |
| `SERPAPI_KEY` | optional | Real Google search for the `WEB_SEARCH` route. Without it, `search_web` returns a clearly-tagged stub |
