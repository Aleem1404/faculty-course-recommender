from __future__ import annotations

import math
import pytest
import numpy as np

try:
    from m6.ir_metrics import (
        dcg_at_k,
        idcg_at_k,
        ndcg_at_k,
        precision_at_k,
        reciprocal_rank,
        average_precision,
        hit_rate_at_k,
        mean_relevance,
        evaluate_ranking,
        aggregate_query_metrics,
    )
    from m6.statistical_reliability import (
        cohens_kappa,
        fleiss_kappa,
        interpret_kappa,
        bootstrap_confidence_interval,
        paired_significance_test,
    )
except ImportError:
    from src.m6.ir_metrics import (
        dcg_at_k,
        idcg_at_k,
        ndcg_at_k,
        precision_at_k,
        reciprocal_rank,
        average_precision,
        hit_rate_at_k,
        mean_relevance,
        evaluate_ranking,
        aggregate_query_metrics,
    )
    from src.m6.statistical_reliability import (
        cohens_kappa,
        fleiss_kappa,
        interpret_kappa,
        bootstrap_confidence_interval,
        paired_significance_test,
    )


class TestIRMetrics:
    def test_dcg_calculation(self):
        rels = [2, 1, 0, 0, 0]
        # rank 1: (2^2 - 1)/log2(2) = 3 / 1 = 3.0
        # rank 2: (2^1 - 1)/log2(3) = 1 / 1.58496 = 0.63093
        expected = 3.0 + (1.0 / math.log2(3))
        assert math.isclose(dcg_at_k(rels, k=5), expected, rel_tol=1e-4)

    def test_ndcg_perfect_and_zero(self):
        perfect = [2, 2, 1, 0, 0]
        assert math.isclose(ndcg_at_k(perfect, k=5), 1.0, rel_tol=1e-4)

        all_zeros = [0, 0, 0, 0, 0]
        assert ndcg_at_k(all_zeros, k=5) == 0.0

        empty = []
        assert ndcg_at_k(empty, k=5) == 0.0

    def test_ndcg_suboptimal_ordering(self):
        suboptimal = [0, 1, 2, 0, 0]
        score = ndcg_at_k(suboptimal, k=5)
        assert 0.0 < score < 1.0

    def test_precision_at_k(self):
        rels = [2, 1, 0, 2, 0]
        assert precision_at_k(rels, k=5, threshold=1) == 0.6
        assert precision_at_k(rels, k=5, threshold=2) == 0.4
        assert precision_at_k(rels, k=1, threshold=2) == 1.0

    def test_reciprocal_rank(self):
        assert reciprocal_rank([2, 0, 0], threshold=2) == 1.0
        assert math.isclose(reciprocal_rank([0, 0, 2, 0], threshold=2), 1.0 / 3.0)
        assert reciprocal_rank([0, 0, 0], threshold=2) == 0.0

    def test_average_precision(self):
        assert math.isclose(average_precision([1, 0, 1], k=3, threshold=1), 5.0 / 6.0, rel_tol=1e-4)
        assert average_precision([0, 0, 0], k=3, threshold=1) == 0.0

    def test_hit_rate(self):
        assert hit_rate_at_k([0, 2, 0], k=3, threshold=2) == 1.0
        assert hit_rate_at_k([0, 1, 0], k=3, threshold=2) == 0.0

    def test_evaluate_ranking_battery(self):
        res = evaluate_ranking([2, 1, 0, 0, 0], k=5)
        assert "ndcg_at_5" in res
        assert "precision_at_1_strict" in res
        assert "mrr_strict" in res
        assert "map_at_5_strict" in res
        assert res["precision_at_1_strict"] == 1.0


class TestStatisticalReliability:
    def test_cohens_kappa_perfect_agreement(self):
        r1 = [0, 1, 2, 1, 0, 2]
        r2 = [0, 1, 2, 1, 0, 2]
        res = cohens_kappa(r1, r2)
        assert res["kappa"] == 1.0
        assert res["observed_agreement"] == 1.0
        assert "Almost Perfect" in res["interpretation"]

    def test_cohens_kappa_weighted(self):
        r1 = [0, 1, 2, 2]
        r2 = [0, 2, 1, 2]
        unweighted = cohens_kappa(r1, r2, weights="unweighted")
        linear = cohens_kappa(r1, r2, weights="linear")
        quadratic = cohens_kappa(r1, r2, weights="quadratic")
        assert linear["kappa"] >= unweighted["kappa"]

    def test_fleiss_kappa(self):
        matrix = [
            [3, 0, 0],
            [0, 3, 0],
            [0, 0, 3],
            [3, 0, 0],
        ]
        res = fleiss_kappa(matrix)
        assert res["kappa"] == 1.0

    def test_bootstrap_confidence_interval(self):
        values = [0.8, 0.85, 0.9, 0.82, 0.88, 0.84, 0.86]
        res = bootstrap_confidence_interval(values, num_bootstraps=500, confidence_level=0.95, seed=42)
        assert res["ci_lower"] <= res["mean"] <= res["ci_upper"]
        assert 0.75 <= res["ci_lower"] < 0.95

    def test_paired_significance_test(self):
        scores_a = [0.2, 0.3, 0.25, 0.22, 0.28, 0.31, 0.24, 0.29]
        scores_b = [0.7, 0.75, 0.8, 0.72, 0.78, 0.82, 0.74, 0.79]
        res = paired_significance_test(scores_a, scores_b, alpha=0.05)
        assert res["is_significant"] is True
        assert res["primary_p_value"] < 0.01
        assert res["mean_b"] > res["mean_a"]
        assert res["cohens_d"] > 2.0
