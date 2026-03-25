"""Tests for anti_resonance.core."""

from anti_resonance.core import ensure_negative_ratio, ensure_negative_balance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pos(pnl: float = 100.0, score: float = 0.8) -> dict:
    return {"pnl": pnl, "score": score, "symbol": "XAUUSD"}


def _neg(pnl: float = -50.0, score: float = 0.5) -> dict:
    return {"pnl": pnl, "score": score, "symbol": "XAUUSD"}


# ---------------------------------------------------------------------------
# (1) Basic replacement (no score_key)
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

    def test_original_not_mutated(self):
        results = [_pos() for _ in range(5)]
        original_len = len(results)
        candidates = [_neg()]

        ensure_negative_ratio(results, candidates)

        assert len(results) == original_len
        assert all(r["pnl"] > 0 for r in results)


# ---------------------------------------------------------------------------
# (2) Already meets ratio
# ---------------------------------------------------------------------------

class TestAlreadySatisfied:
    def test_ratio_already_met_returns_same_content(self):
        results = [_pos(), _pos(), _pos(), _neg(), _neg()]  # 40% negative
        candidates = [_neg(-999)]

        out = ensure_negative_ratio(results, candidates, min_negative_ratio=0.2)

        assert all(r["pnl"] != -999 for r in out)
        assert out == results

    def test_exact_ratio_not_replaced(self):
        results = [_pos()] * 8 + [_neg()] * 2
        candidates = [_neg(-999)]

        out = ensure_negative_ratio(results, candidates, min_negative_ratio=0.2)

        assert all(r["pnl"] != -999 for r in out)


# ---------------------------------------------------------------------------
# (3) Empty/edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_results_returns_empty(self):
        assert ensure_negative_ratio([], [_neg()]) == []

    def test_empty_results_with_empty_candidates(self):
        assert ensure_negative_ratio([], []) == []

    def test_single_result_positive(self):
        results = [_pos()]
        candidates = [_neg()]
        out = ensure_negative_ratio(results, candidates, min_negative_ratio=0.5)
        # ceil(0.5 * 1) = 1, so the single positive should be replaced
        assert len(out) == 1
        assert out[0]["pnl"] < 0

    def test_single_result_negative(self):
        results = [_neg()]
        candidates = [_pos()]
        out = ensure_negative_ratio(results, candidates, min_negative_ratio=0.2)
        assert out == results


# ---------------------------------------------------------------------------
# (4) No negative candidates
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
# (5) High ratio
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
        assert neg_count == 2
        assert len(out) == 10


# ---------------------------------------------------------------------------
# (6) Custom classifier
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
        classifier = lambda item: item.get("sentiment", 0) < 0

        results = [
            {"pnl": -100, "sentiment": 0.5},
            {"pnl": 200, "sentiment": 0.8},
            {"pnl": 300, "sentiment": 0.9},
            {"pnl": -50, "sentiment": 0.6},
            {"pnl": 100, "sentiment": 0.7},
        ]
        candidates = [
            {"pnl": 500, "sentiment": -0.8},
        ]

        out = ensure_negative_ratio(
            results, candidates, min_negative_ratio=0.2, negative_classifier=classifier
        )

        neg_count = sum(1 for r in out if classifier(r))
        assert neg_count / len(out) >= 0.2
        injected = [r for r in out if r.get("sentiment", 0) < 0]
        assert len(injected) >= 1
        assert injected[0]["pnl"] == 500


# ---------------------------------------------------------------------------
# (7) Score-aware replacement (NEW in v0.2.0)
# ---------------------------------------------------------------------------

class TestScoreAwareReplacement:
    def test_lowest_score_positive_replaced_first(self):
        """With score_key, the lowest-scoring positive should be replaced first."""
        results = [
            _pos(100, score=0.9),  # high score — keep
            _pos(80, score=0.7),   # medium — keep
            _pos(60, score=0.3),   # LOW score — replace this
            _pos(50, score=0.5),   # medium — keep
            _pos(40, score=0.2),   # LOWEST — replace this
        ]
        candidates = [
            _neg(-10, score=0.6),
            _neg(-20, score=0.4),
        ]

        out = ensure_negative_ratio(
            results, candidates, min_negative_ratio=0.2, score_key="score"
        )

        neg_count = sum(1 for r in out if r["pnl"] < 0)
        assert neg_count >= 1
        # The highest-scoring items should survive
        scores = [r["score"] for r in out if r["pnl"] > 0]
        assert 0.9 in scores  # highest positive kept
        assert 0.7 in scores  # second highest kept

    def test_highest_score_negative_injected_first(self):
        """With score_key, the highest-scoring negative candidate should be injected first."""
        results = [_pos(100, score=0.1) for _ in range(5)]
        candidates = [
            _neg(-10, score=0.3),  # lower
            _neg(-20, score=0.8),  # HIGHEST — inject this first
            _neg(-30, score=0.5),  # medium
        ]

        out = ensure_negative_ratio(
            results, candidates, min_negative_ratio=0.2, score_key="score"
        )

        negatives = [r for r in out if r["pnl"] < 0]
        assert len(negatives) >= 1
        # The highest-score negative should be present
        assert any(r["score"] == 0.8 for r in negatives)

    def test_score_aware_re_sorts_by_score(self):
        """Output should be sorted by score descending when score_key is set."""
        results = [
            _pos(100, score=0.5),
            _pos(80, score=0.9),
            _pos(60, score=0.1),
        ]
        candidates = [_neg(-10, score=0.7)]

        out = ensure_negative_ratio(
            results, candidates, min_negative_ratio=0.3, score_key="score"
        )

        scores = [r["score"] for r in out]
        assert scores == sorted(scores, reverse=True)

    def test_score_aware_without_score_key_falls_back(self):
        """Without score_key, should still work (index-based replacement)."""
        results = [_pos(100, score=0.9), _pos(80, score=0.1)]
        candidates = [_neg(-10, score=0.5)]

        out = ensure_negative_ratio(results, candidates, min_negative_ratio=0.5)

        neg_count = sum(1 for r in out if r["pnl"] < 0)
        assert neg_count >= 1


# ---------------------------------------------------------------------------
# (8) Alias compatibility
# ---------------------------------------------------------------------------

class TestAlias:
    def test_ensure_negative_balance_is_alias(self):
        assert ensure_negative_balance is ensure_negative_ratio
