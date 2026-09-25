"""Decoding-algorithm demos on GPT-2, a small open model whose distribution
we can inspect directly. Hosted APIs hide the probability distribution (and
reasoning models such as gpt-5 refuse logprobs entirely), so the mechanics in
chapter 5 sections 5.1-5.2 run locally instead.

GPT-2 is tiny (124M parameters) and runs on any laptop CPU. The algorithms are
identical to what runs inside a frontier model; only the scale differs.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_tok = None
_lm = None


def load_gpt2():
    global _tok, _lm
    if _lm is None:
        _tok = AutoTokenizer.from_pretrained("gpt2")
        _lm = AutoModelForCausalLM.from_pretrained("gpt2")
        _lm.eval()
    return _tok, _lm


def next_token_distribution(prompt: str, top_n: int = 5):
    """One forward pass: the probability of every possible next token."""
    tok, lm = load_gpt2()
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = lm(ids).logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, top_n)
    return [(tok.decode(i), p.item()) for p, i in zip(top.values, top.indices)]


def greedy_by_hand(prompt: str, steps: int = 6, show_top: int = 3):
    """The autoregressive loop written out: predict, pick the argmax, append."""
    tok, lm = load_gpt2()
    cur = tok(prompt, return_tensors="pt").input_ids
    for step in range(steps):
        with torch.no_grad():
            logits = lm(cur).logits[0, -1]
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, show_top)
        cands = ", ".join(f"{tok.decode(i)!r} {p.item():.3f}"
                          for p, i in zip(top.values, top.indices))
        pick = top.indices[0].reshape(1, 1)
        print(f"step {step + 1}: picks {tok.decode(pick[0])!r:12s} from [{cands}]")
        cur = torch.cat([cur, pick], dim=1)
    return tok.decode(cur[0])


def generate(prompt: str, max_new_tokens: int = 45, seed: int | None = None,
             **kwargs) -> str:
    """Thin wrapper over model.generate so the notebook stays terse.

    kwargs pass straight through: do_sample, num_beams, temperature,
    top_k, top_p.
    """
    tok, lm = load_gpt2()
    if seed is not None:
        torch.manual_seed(seed)
    ids = tok(prompt, return_tensors="pt").input_ids
    out = lm.generate(ids, max_new_tokens=max_new_tokens,
                      pad_token_id=tok.eos_token_id, **kwargs)
    return tok.decode(out[0])


def nucleus_size(prompt: str, p: float = 0.9) -> int:
    """How many tokens it takes to cover probability mass p at this position."""
    tok, lm = load_gpt2()
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = lm(ids).logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    sorted_p, _ = torch.sort(probs, descending=True)
    csum = torch.cumsum(sorted_p, dim=0)
    return int((csum < p).sum().item()) + 1
