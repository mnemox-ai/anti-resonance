# anti-resonance

Counteract positive recall bias in memory-augmented AI systems. Zero dependencies.

## Install

```bash
pip install anti-resonance
```

## Quick Start

```python
from anti_resonance import ensure_negative_ratio

# Your RAG/memory system returned 10 results — mostly wins
results = [
    {"text": "Trade won +$150", "pnl": 150, "score": 0.9},
    {"text": "Trade won +$80",  "pnl": 80,  "score": 0.8},
    {"text": "Trade won +$200", "pnl": 200, "score": 0.85},
    {"text": "Trade won +$50",  "pnl": 50,  "score": 0.7},
    {"text": "Trade won +$120", "pnl": 120, "score": 0.75},
]

# Losses from your full memory pool
candidates = [
    {"text": "Trade lost -$90 (SL hit in ranging market)",  "pnl": -90,  "score": 0.6},
    {"text": "Trade lost -$200 (gap against position)",     "pnl": -200, "score": 0.4},
    {"text": "Trade lost -$45 (spread widened at news)",    "pnl": -45,  "score": 0.5},
]

# Force at least 20% losses into the context window
balanced = ensure_negative_ratio(
    results, candidates,
    min_negative_ratio=0.2,
    score_key="score",         # score-aware: replaces lowest-score wins
)
```

The agent now sees both wins AND losses before making its next decision.

## The Problem: Parametric-External Memory Resonance

When an LLM-based agent retrieves memories to inform a decision, two bias sources **resonate** with each other:

1. **Parametric bias** (inside the LLM): Language models trained on internet text have a built-in optimism bias. Success stories, best practices, and positive outcomes are overrepresented in training data.

2. **External memory bias** (in the retrieval system): Similarity-based retrieval (vector search, embedding lookup) clusters positive outcomes together. Winning trades look alike ("strong momentum", "breakout confirmed"). Losing trades are diverse and scattered ("gap down", "spread spike", "regime change") — they don't cluster, so they rank low in similarity search.

When the LLM's parametric optimism meets a one-sided positive context window, the biases **amplify** each other. The model becomes dangerously overconfident because both its training and its retrieved evidence agree.

This is **Parametric-External Memory Resonance** — a feedback loop between the model's internal beliefs and the retrieval system's structural bias toward positive outcomes.

### The Fix

Break the resonance by forcing counterexamples into the context window. `anti-resonance` does this at the post-retrieval stage:

```
Query → Retrieval → [anti-resonance] → LLM
                          ↑
                    Force 20% negatives
                    into the result set
```

The algorithm is simple:
1. Count negative items in top-K results
2. If ratio >= threshold → return unchanged
3. Otherwise → replace **lowest-scoring positives** with **highest-scoring negatives** from the candidate pool
4. Re-sort by score

Score-aware replacement (v0.2.0) ensures you sacrifice the least-relevant positive items and inject the most-relevant negative ones.

## API

### `ensure_negative_ratio(results, candidates, min_negative_ratio=0.2, negative_classifier=None, score_key=None)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results` | `list[dict]` | required | Top-K retrieval results |
| `candidates` | `list[dict]` | required | Full candidate pool |
| `min_negative_ratio` | `float` | `0.2` | Minimum negative fraction [0.0, 1.0] |
| `negative_classifier` | `Callable` | `None` | Returns `True` for negative items. Default: `item["pnl"] < 0` |
| `score_key` | `str` | `None` | Dict key for relevance score. Enables score-aware replacement |

**Returns:** New list (same length) with at least `min_negative_ratio` negatives.

### `ensure_negative_balance`

Alias for `ensure_negative_ratio` (backward compatibility with TradeMemory).

## Use Cases

### Trading Memory
```python
balanced = ensure_negative_ratio(
    recalled_trades, all_trades,
    min_negative_ratio=0.2,
    score_key="owm_score",
)
# Agent sees losing trades alongside winners before deciding
```

### RAG Pipelines
```python
balanced = ensure_negative_ratio(
    search_results, full_corpus,
    negative_classifier=lambda doc: doc["sentiment"] < 0,
    score_key="relevance",
)
# LLM sees failure cases alongside success stories
```

### Customer Support
```python
balanced = ensure_negative_ratio(
    similar_tickets, all_tickets,
    negative_classifier=lambda t: t["resolution"] == "unresolved",
)
# Agent considers unresolved similar cases, not just resolved ones
```

## Related Work

- **FinMem** (Yu et al., 2024) — *FinMem: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design*. Identified the echo chamber problem in LLM trading agents where memory retrieval reinforces existing beliefs. [arXiv:2311.13743](https://arxiv.org/abs/2311.13743)

- **TradeMemory Protocol** — Open-source trading memory layer where anti-resonance was first implemented. The `ensure_negative_balance()` function in TradeMemory's hybrid recall pipeline is the production origin of this package. [GitHub](https://github.com/mnemox-ai/tradememory-protocol)

- **Building Anti-Resonance for AI Trading Agents** (Sean, 2026) — DEV.to article explaining why RAG-based trading agents develop confirmation bias and how to fix it. [DEV.to](https://dev.to/mnemox)

## License

MIT
