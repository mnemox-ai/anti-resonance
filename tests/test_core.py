"""Tests for anti_resonance.core.ensure_negative_ratio."""

from anti_resonance.core import ensure_negative_ratio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pos(pnl: float = 100.0) -> dict:
    return {"pnl": pnl, "symbol": "XAUUSD"}


def _neg(pnl: float = -50.0) -> dict:
    return {"pnl": pnl, "symbol": "XAUUSD"}


# ---------------------------------------------------------------------------
# (1) 10 positive results + negative candidates → at least 20% negative
# ---------------------------------------------------------------------------

class TestBasicReplacement:
    def test_all_positive_gets_negatives_injected(self):
        results = [_pos() for _ in range(10)]
        candidates = [_neg(-10), _neg(-20), _neg(-30), _neg(-40), _neg(-50)]

        out = ensure_negative_ratio(results, candidates)

        neg_count = sum(1 for r in out if r["pnl"] < 0)
        assert len(out) == 10
        assert neg_count / len(out) >= 0.2

    def test_replaced_items_come_from_candidates(self):
        results = [_pos() for _ in range(10)]
        candidates = [_neg(-10), _neg(-20), _neg(-30)]

        out = ensure_negative_ratio(results, candidates)

        negatives_in_out = [r for r in out if r["pnl"] < 0]
        for neg in negatives_in_out:
            assert neg in candidates


# ---------------------------------------------------------------------------
# (2) Already meets ratio → no replacement
# ---------------------------------------------------------------------------

class TestAlreadySatisfied:
    def test_ratio_already_met_returns_same_content(self):
        results = [_pos(), _pos(), _pos(), _neg(), _neg()]  # 40% negative
        candidates = [_neg(-999)]

        out = ensure_negative_ratio(results, candidates, min_negative_ratio=0.2)

        # Should not inject the -999 candidate
        assert all(r["pnl"] != -999 for r in out)
        assert out == results

    def test_exact_ratio_not_replaced(self):
        # Exactly 20% negative
        results = [_pos()] * 8 + [_neg()] * 2
        candidates = [_neg(-999)]

        out = ensure_negative_ratio(results, candidates, min_negative_ratio=0.2)

        assert all(r["pnl"] != -999 for r in out)


# ---------------------------------------------------------------------------
# (3) Empty results → empty return
# ---------------------------------------------------------------------------

class TestEmptyResults:
    def test_empty_results_returns_empty(self):
        assert ensure_negative_ratio([], [_neg()]) == []

    def test_empty_results_with_empty_candidates(self):
        assert ensure_negative_ratio([], []) == []


# ---------------------------------------------------------------------------
# (4) No negative candidates → return original results
# ---------------------------------------------------------------------------

class TestNoNegativeCandidates:
    def test_all_positive_candidates_returns_original(self):
        results = [_pos() for _ in range(5)]
        candidates = [_pos(200), _pos(300)]

        out = ensure_negative_ratio(results, candidates)

        assert out == results

    def test_empty_candidates_returns_original(self):
        results = [_pos() for _ in range(5)]

        out = ensure_negative_ratio(results, [])

        assert out == results


# ---------------------------------------------------------------------------
# (5) min_negative_ratio=0.5
# ---------------------------------------------------------------------------

class TestHighRatio:
    def test_half_negative_required(self):
        results = [_pos() for _ in range(10)]
        candidates = [_neg(-i) for i in range(1, 11)]

        out = ensure_negative_ratio(results, candidates, min_negative_ratio=0.5)

        neg_count = sum(1 for r in out if r["pnl"] < 0)
        assert len(out) == 10
        assert neg_count / len(out) >= 0.5

    def test_not_enough_candidates_for_high_ratio(self):
        results = [_pos() for _ in range(10)]
        candidates = [_neg(-1), _neg(-2)]  # Only 2 negatives, need 5

        out = ensure_negative_ratio(results, candidates, min_negative_ratio=0.5)

        neg_count = sum(1 for r in out if r["pnl"] < 0)
        # Should inject as many as possible (2), but can't reach 50%
        assert neg_count == 2
        assert len(out) == 10


# ---------------------------------------------------------------------------
# (6) Custom negative_classifier (sentiment < 0 instead of pnl)
# ---------------------------------------------------------------------------

class TestCustomClassifier:
    def test_sentiment_based_classifier(self):
        classifier = lambda item: item.get("sentiment", 0) < 0

        results = [
            {"sentiment": 0.8, "text": "great"},
            {"sentiment": 0.5, "text": "ok"},
            {"sentiment": 0.9, "text": "excellent"},
            {"sentiment": 0.3, "text": "fine"},
            {"sentiment": 0.7, "text": "good"},
        ]
        candidates = [
            {"sentiment": -0.6, "text": "bad"},
            {"sentiment": -0.9, "text": "terrible"},
            {"sentiment": -0.3, "text": "meh"},
        ]

        out = ensure_negative_ratio(
            results, candidates, min_negative_ratio=0.2, negative_classifier=classifier
        )

        neg_count = sum(1 for r in out if classifier(r))
        assert neg_count / len(out) >= 0.2

    def test_custom_classifier_ignores_pnl(self):
        """Items with negative pnl but positive sentiment should NOT be treated as negative."""
        classifier = lambda item: item.get("sentiment", 0) < 0

        results = [
            {"pnl": -100, "sentiment": 0.5},  # negative pnl, positive sentiment
            {"pnl": 200, "sentiment": 0.8},
            {"pnl": 300, "sentiment": 0.9},
            {"pnl": -50, "sentiment": 0.6},   # negative pnl, positive sentiment
            {"pnl": 100, "sentiment": 0.7},
        ]
        candidates = [
            {"pnl": 500, "sentiment": -0.8},  # positive pnl, negative sentiment
        ]

        out = ensure_negative_ratio(
            results, candidates, min_negative_ratio=0.2, negative_classifier=classifier
        )

        neg_count = sum(1 for r in out if classifier(r))
        assert neg_count / len(out) >= 0.2
        # The injected item has positive pnl but negative sentiment
        injected = [r for r in out if r.get("sentiment", 0) < 0]
        assert len(injected) >= 1
        assert injected[0]["pnl"] == 500  # proves it used sentiment, not pnl
