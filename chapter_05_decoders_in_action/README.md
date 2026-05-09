# Chapter 5 — Decoders in Action

Companion code for **Chapter 5** of *Build an Advanced RAG Application (From Scratch)*.

## What this chapter builds

Four prompting styles, side by side, on the same model:

- **Basic** — minimal ask
- **Structured** — explicit sections and constraints
- **Few-shot** — show examples instead of describing the format
- **Chain-of-Thought** — make the model reason step by step

The chapter ends with a CoT prompt that **analyzes hotel search results** — that's the exact seam Chapter 6 plugs retrieval into.

## Files

| File | Purpose |
|------|---------|
| [notebook.ipynb](notebook.ipynb) | Walkthrough — read top-to-bottom |
| [llm_client.py](llm_client.py) | OpenAI client + `generate_text` helper |
| [prompts.py](prompts.py) | Prompt templates: basic / structured / few-shot / CoT |

## Run it

```bash
conda activate advanced-rag
cd chapter_05_decoders_in_action
jupyter lab notebook.ipynb
```

### API key

Copy `.env.example` at the repo root to `.env` and set:

```
OPENAI_API_KEY=sk-...
```

`llm_client.py` loads the key automatically via `python-dotenv`.
