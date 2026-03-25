"""Core anti-resonance logic: ensure result sets contain sufficient negative samples.

Implements score-aware replacement: replaces the lowest-scoring positive items
with the highest-scoring negative candidates from the full pool.
"""

import math
from typing import Callable, List, Optional


def _default_negative_classifier(item: dict) -> bool:
    return item.get("pnl", 0) < 0


def ensure_negative_ratio(
    results: List[dict],
    candidates: List[dict],
    min_negative_ratio: float = 0.2,
    negative_classifier: Optional[Callable[[dict], bool]] = None,
    score_key: Optional[str] = None,
) -> List[dict]:
    """Replace positive results with negative candidates to meet min_negative_ratio.

    When ``score_key`` is provided, uses **score-aware replacement**: replaces the
    lowest-scoring positive items with the highest-scoring negative candidates.
    Without ``score_key``, replaces positives from the front of the list.

    This is the core anti-resonance algorithm. It counteracts the "positive
    resonance" effect where similarity-based retrieval over-represents positive
    outcomes because successful examples cluster in embedding space while
    failures are diverse and scattered.

    Args:
        results: Current result list (top-K from retrieval).
        candidates: Full candidate pool (superset of results).
        min_negative_ratio: Minimum fraction of results that must be negative.
            Default 0.2 (20%). Range [0.0, 1.0].
        negative_classifier: Callable returning True if item is a negative
            outcome. Defaults to ``item['pnl'] < 0``.
        score_key: Optional dict key for relevance/quality score. When set,
            enables score-aware replacement (lowest-score positives replaced
            first, highest-score negatives injected first).

    Returns:
        New list (same length as ``results``) with at least
        ``min_negative_ratio`` negatives, if enough candidates exist.
        Original list is never mutated.
    """
    if not results:
        return []

    classifier = negative_classifier or _default_negative_classifier

    num_negatives = sum(1 for r in results if classifier(r))
    current_ratio = num_negatives / len(results)

    if current_ratio >= min_negative_ratio:
        return list(results)

    # Find negative candidates not already in results
    result_ids = set(id(r) for r in results)
    spare_negatives = [c for c in candidates if classifier(c) and id(c) not in result_ids]

    if not spare_negatives:
        return list(results)

    needed = math.ceil(min_negative_ratio * len(results)) - num_negatives

    if score_key:
        # Score-aware: replace lowest-scoring positives with highest-scoring negatives
        spare_negatives.sort(key=lambda x: x.get(score_key, 0), reverse=True)

        positives = [
            (i, r) for i, r in enumerate(results) if not classifier(r)
        ]
        positives.sort(key=lambda x: x[1].get(score_key, 0))  # lowest score first

        out = list(results)
        swapped = 0
        for neg in spare_negatives:
            if swapped >= needed:
                break
            if not positives:
                break
            idx, _victim = positives.pop(0)
            out[idx] = neg
            swapped += 1

        # Re-sort by score (highest first)
        out.sort(key=lambda x: x.get(score_key, 0), reverse=True)
        return out
    else:
        # Index-based: replace positives from front
        out = list(results)
        ci = 0
        for i in range(len(out)):
            if ci >= needed or ci >= len(spare_negatives):
                break
            if not classifier(out[i]):
                out[i] = spare_negatives[ci]
                ci += 1
        return out


# Alias for TradeMemory compatibility
ensure_negative_balance = ensure_negative_ratio
