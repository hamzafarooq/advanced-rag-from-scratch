# Chapter 5 — Generation: putting decoders to work

Companion code for **Chapter 5** of *Build an Advanced RAG Application (From Scratch)*.

## What this chapter builds

Chapter 4's retriever hands back ranked hotels; this chapter puts a decoder on top of it to write the grounded answer.

- **The autoregressive loop** — watch GPT-2 predict one token at a time (5.1)
- **Decoding algorithms** — greedy, beam search, temperature / top-k / top-p, with real reproductions of their failure modes (5.2)
- **Prompting** — basic vs structured vs few-shot vs chain-of-thought on GPT-5 (5.3)
- **Grounded, cited generation** — the prompt pattern that answers only from retrieved reviews, with citations (5.3.4)
- **Failure modes** — ignoring the context, over-trusting it, and refusal, each reproduced live (5.5)

## Files

| File | Purpose |
|------|---------|
| [notebook.ipynb](notebook.ipynb) | Walkthrough — read top-to-bottom |
| [llm_client.py](llm_client.py) | Env-driven OpenAI-compatible client + `generate_text` |
| [decoding.py](decoding.py) | GPT-2 decoding demos: distributions, greedy, beam, sampling |
| [prompts.py](prompts.py) | Prompt templates: basic / structured / few-shot / CoT |

## Run it

```bash
conda activate advanced-rag
cd chapter_05_decoders_in_action
jupyter lab notebook.ipynb
```

Sections 1–2 run locally on GPT-2 (no key needed). Sections 3–6 call the live API.

### API configuration

Copy `.env.example` at the repo root to `.env` and set:

```
OPENAI_API_KEY=sk-...
```

Optional overrides (all read by `llm_client.py`):

```
LLM_BASE_URL=   # any OpenAI-compatible endpoint; unset = OpenAI
LLM_MODEL=      # default gpt-5
LLM_API_KEY=    # falls back to OPENAI_API_KEY
```

The same client becomes OpenRouter (chapter 6) or a local Ollama server (chapter 8) by changing `LLM_BASE_URL` — the client is the swap seam.
