# Build an Advanced RAG Application (From Scratch) — Companion Code

<p align="center">
  <img src="book-cover.png" alt="Build an Advanced RAG Application (From Scratch)" width="320"/>
</p>

This is the official companion code for **[Build an Advanced RAG Application (From Scratch)](https://www.manning.com/books/build-an-advanced-rag-application-from-scratch)** by **Hamza Farooq**, published by **Manning** (MEAP).

The book teaches you to build search and Retrieval-Augmented Generation (RAG) systems **without leaning on Langchain or LlamaIndex** — so you understand every layer.

## Repo layout

Each chapter lives in its own folder with a teaching notebook plus reusable `.py` modules.

| Folder | Chapter | What you'll build |
|--------|---------|-------------------|
| [chapter_04_semantic_search/](chapter_04_semantic_search/) | **4** Semantic Search from Scratch | Cosine search → FAISS, Flat / HNSW / IVF-PQ |
| [chapter_05_decoders_in_action/](chapter_05_decoders_in_action/) | **5** Decoders in Action | Prompting styles: basic, structured, few-shot, CoT |
| [chapter_06_rag/](chapter_06_rag/) | **6** Retrieval-Augmented Generation | Full RAG pipeline on hotels (FAISS + Qdrant) and research papers |

Chapters 1–3, 7, and 8 in the book are conceptual / not yet covered here. Stay tuned.

The original Colab notebooks (pre-cleanup) are preserved in [archive_original_notebooks/](archive_original_notebooks/) for reference.

## Setup

### 1. Clone and create the conda env

```bash
git clone <this-repo>
cd advanced-rag-from-scratch

conda create -n advanced-rag python=3.11 -y
conda activate advanced-rag
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# then edit .env and fill in your real keys
```

You only need keys for chapters that use them:

| Key | Required for |
|-----|--------------|
| `OPENAI_API_KEY` | Chapter 5 |
| `OPEN_ROUTER_API_KEY` | Chapter 6 (LLM generation — sign up at https://openrouter.ai) |
| `KAGGLE_USERNAME` + `KAGGLE_KEY` | Chapter 6b (dataset download — get a token at https://www.kaggle.com/settings) |

Chapter 4 needs no API keys — it runs locally.

### 3. Register the env as a Jupyter kernel

```bash
python -m ipykernel install --user --name advanced-rag --display-name "Python (advanced-rag)"
jupyter lab
```

Then in any chapter notebook, select the **Python (advanced-rag)** kernel.

## What you'll learn

**Chapter 4** — How vectors and similarity actually work, written out long-form in NumPy before we hand it to FAISS.

**Chapter 5** — How prompt structure changes output quality, and why Chain-of-Thought is the right prompt for analyzing retrieval results.

**Chapter 6** — A complete RAG loop, twice: once on hotel reviews (with city-level metadata filtering) and once on research papers (with chunking and year-level filtering). Both use the same building blocks; only the corpus and the metadata change.

## License

See [LICENSE](LICENSE).

---

*Hamza Farooq — Founder, [Traversaal.ai](https://traversaal.ai). Adjunct Professor, UCLA & Stanford.*
