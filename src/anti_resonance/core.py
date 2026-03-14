"""Core anti-resonance logic: ensure result sets contain sufficient negative samples."""

import math
from typing import Callable, List, Optional


def _default_negative_classifier(item: dict) -> bool:
    return item.get("pnl", 0) < 0


def ensure_negative_ratio(
    results: List[dict],
    candidates: List[dict],
    min_negative_ratio: float = 0.2,
    negative_classifier: Optional[Callable[[dict], bool]] = None,
) -> List[dict]:
    """Replace positive results with negative candidates to meet min_negative_ratio.

    Args:
        results: Current result list.
        candidates: Pool of replacement candidates.
        min_negative_ratio: Minimum fraction of results that must be negative.
        negative_classifier: Callable returning True if item is negative.
            Defaults to ``item['pnl'] < 0``.

    Returns:
        New list with at least *min_negative_ratio* negatives (if enough candidates).
    """
    if not results:
        return []

    classifier = negative_classifier or _default_negative_classifier

    num_negatives = sum(1 for r in results if classifier(r))
    current_ratio = num_negatives / len(results)

    if current_ratio >= min_negative_ratio:
        return list(results)

    neg_candidates = [c for c in candidates if classifier(c)]

    if not neg_candidates:
        return list(results)

    needed = math.ceil(min_negative_ratio * len(results)) - num_negatives
    out = list(results)
    ci = 0

    for i in range(len(out)):
        if ci >= needed or ci >= len(neg_candidates):
            break
        if not classifier(out[i]):
            out[i] = neg_candidates[ci]
            ci += 1

    return out
