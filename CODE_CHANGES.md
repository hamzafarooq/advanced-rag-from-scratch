# Code changes log

Per-chapter record of changes to the companion code during the RAG-pivot revision.
Newest first. The manuscript drafts themselves live outside this repo; this file
tracks only what changed under the chapter code folders.

## Chapter 4 — `chapter_04_semantic_search/` (2026-07-22, commit f5c83d2)

Chapter: "Building the RAG retriever."

**Changed**
- `notebook.ipynb` — re-executed with real outputs. Four changes:
  1. Task prefixes: reviews encoded with `search_document:` and the query with
     `search_query:`, matching the nomic setup in chapters 2 and 3. (Retrieval
     scores shift slightly vs. a prefix-less encode; all chapter numbers were
     verified against the prefixed version.)
  2. New cross-encoder reranking section using
     `cross-encoder/ms-marco-MiniLM-L-6-v2` over the bi-encoder's top-50.
  3. New packaged `retrieve(query, k, depth)` function returning ranked hotels —
     the search layer chapters 5 and 6 build on.
  4. New index/embedding persistence cell (`np.save` + `faiss.write_index`, then
     load), so a restart skips re-encoding.
- `README.md` — feature list updated to include the cross-encoder, hotel
  aggregation, and persistence.

**Unchanged**
- `search.py`, `data_loader.py` — the chapter's listings match these as they stand.

**Dependencies**
- No new packages. `CrossEncoder` ships inside `sentence-transformers`, already
  used by the book. `faiss`, `torch`, `datasets` already present.

**Generated artifacts (gitignored)**
- `*.npy`, `*.faiss` produced by the persistence cell.

**Notes**
- macOS: the notebook's first cell sets `KMP_DUPLICATE_LIB_OK=TRUE` and
  `OMP_NUM_THREADS=1` before importing torch, to avoid a FAISS/torch OpenMP clash
  during index training.
