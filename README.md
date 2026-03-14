# anti-resonance

Inject counterexamples into similarity-based retrieval results to counteract positive clustering bias.

## Install

```bash
pip install anti-resonance
```

## Quick Start

```python
from anti_resonance import ensure_negative_ratio

# Your RAG pipeline returned 5 docs — all success stories
results = [
    {"text": "Deploy with zero downtime using blue-green strategy", "sentiment": 0.8},
    {"text": "Rolling deployments ensure continuous availability", "sentiment": 0.7},
    {"text": "Canary releases reduce blast radius of changes", "sentiment": 0.6},
    {"text": "Feature flags enable safe production testing", "sentiment": 0.9},
    {"text": "CI/CD pipelines automate reliable deployments", "sentiment": 0.7},
]

# Failure cases from your full corpus
candidates = [
    {"text": "Production outage caused by missing DB migration", "sentiment": -0.8},
    {"text": "Memory leak went undetected for 3 weeks post-deploy", "sentiment": -0.6},
    {"text": "Rollback failed due to incompatible schema change", "sentiment": -0.9},
]

balanced = ensure_negative_ratio(
    results,
    candidates,
    min_negative_ratio=0.2,
    negative_classifier=lambda doc: doc["sentiment"] < 0,
)

# balanced now contains at least 20% failure cases (1 out of 5)
for doc in balanced:
    print(f"[{'NEG' if doc['sentiment'] < 0 else 'POS'}] {doc['text']}")
```

Output:

```
[NEG] Production outage caused by missing DB migration
[POS] Rolling deployments ensure continuous availability
[POS] Canary releases reduce blast radius of changes
[POS] Feature flags enable safe production testing
[POS] CI/CD pipelines automate reliable deployments
```

## API Reference

### `ensure_negative_ratio(results, candidates, min_negative_ratio=0.2, negative_classifier=None)`

Replaces items in `results` with negative items from `candidates` until `min_negative_ratio` is met.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `results` | `list[dict]` | required | Current result list |
| `candidates` | `list[dict]` | required | Pool of replacement candidates |
| `min_negative_ratio` | `float` | `0.2` | Minimum fraction of results that must be negative |
| `negative_classifier` | `Callable[[dict], bool]` or `None` | `None` | Returns `True` if an item is negative. Defaults to `item["pnl"] < 0` |

**Returns:** New list (same length as `results`) with at least `min_negative_ratio` negatives, if enough candidates exist.

**Behavior:**

- If `results` already meets the ratio, returns a copy unchanged.
- If `results` is empty, returns `[]`.
- If no negative candidates exist, returns a copy unchanged.
- If not enough negative candidates to reach the ratio, injects as many as available.
- Replaces positive items starting from index 0.

## Why

Similarity-based retrieval (vector search, embedding lookup, RAG) has a structural bias: **positive resonance**. When you search for "deployment best practices", cosine similarity returns the 10 most similar documents — which are overwhelmingly success stories. Failures, warnings, and edge cases are dissimilar to the query *and* to each other, so they rank low.

This isn't a bug in the retrieval system. It's a property of how similarity works. Similar things cluster together. Positive outcomes tend to be described in similar language ("worked well", "improved performance", "reduced errors"). Negative outcomes are diverse and scattered ("crashed", "data loss", "timeout", "wrong schema").

The result: your LLM sees a one-sided context window and generates confident, optimistic answers that miss critical failure modes.

`anti-resonance` fixes this at the result-set level. After retrieval, before passing context to the LLM, inject a minimum ratio of counterexamples. The classifier is pluggable — use sentiment, labels, scores, whatever your domain requires.

## Origin

This pattern was extracted from [TradeMemory](https://github.com/mnemox-ai/tradememory-protocol), where similarity-based trade recall consistently surfaced winning trades and buried losses — producing dangerously overconfident strategy adjustments.

`anti-resonance` is the domain-agnostic, zero-dependency extraction of that fix. No numpy, no torch, no frameworks. One function, one idea.

## License

MIT
