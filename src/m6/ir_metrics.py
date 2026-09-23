from __future__ import annotations

import math
from typing import Any, Sequence


def dcg_at_k(
    relevances: Sequence[int | float],
    k: int = 5,
    exponential: bool = True,
) -> float:
    """Computes Discounted Cumulative Gain at rank k."""
    top_k = relevances[:k]
    score = 0.0

    for rank, rel in enumerate(top_k, start=1):
        if rel <= 0:
            continue
        gain = (2.0**rel) - 1.0 if exponential else float(rel)
        discount = math.log2(rank + 1.0)
        score += gain / discount

    return score


def idcg_at_k(
    relevances: Sequence[int | float],
    k: int = 5,
    exponential: bool = True,
) -> float:
    """Computes Ideal Discounted Cumulative Gain at rank k."""
    sorted_relevances = sorted(relevances, reverse=True)
    return dcg_at_k(sorted_relevances, k=k, exponential=exponential)


def ndcg_at_k(
    relevances: Sequence[int | float],
    k: int = 5,
    exponential: bool = True,
) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank k (in [0.0, 1.0])."""
    ideal = idcg_at_k(relevances, k=k, exponential=exponential)
    if ideal <= 0.0:
        return 0.0
    actual = dcg_at_k(relevances, k=k, exponential=exponential)
    return min(1.0, max(0.0, actual / ideal))


def precision_at_k(
    relevances: Sequence[int | float],
    k: int = 5,
    threshold: int | float = 1,
) -> float:
    """Computes Precision at rank k given a relevance threshold."""
    if k <= 0:
        return 0.0
    top_k = list(relevances[:k])
    if not top_k:
        return 0.0

    relevant_count = sum(1 for rel in top_k if rel >= threshold)
    return relevant_count / float(len(top_k))


def reciprocal_rank(
    relevances: Sequence[int | float],
    threshold: int | float = 1,
    max_rank: int = 5,
) -> float:
    """Computes Reciprocal Rank (1/rank of first item meeting relevance threshold)."""
    for rank, rel in enumerate(relevances[:max_rank], start=1):
        if rel >= threshold:
            return 1.0 / float(rank)
    return 0.0


def average_precision(
    relevances: Sequence[int | float],
    k: int = 5,
    threshold: int | float = 1,
) -> float:
    """Computes Average Precision at rank k."""
    top_k = list(relevances[:k])
    if not top_k:
        return 0.0

    relevant_so_far = 0
    precision_sum = 0.0

    for rank, rel in enumerate(top_k, start=1):
        if rel >= threshold:
            relevant_so_far += 1
            precision_sum += relevant_so_far / float(rank)

    total_relevant = sum(1 for rel in top_k if rel >= threshold)
    if total_relevant == 0:
        return 0.0

    return precision_sum / float(total_relevant)


def hit_rate_at_k(
    relevances: Sequence[int | float],
    k: int = 5,
    threshold: int | float = 1,
) -> float:
    """Returns 1.0 if at least one relevant item appears in top k, else 0.0."""
    top_k = relevances[:k]
    return 1.0 if any(rel >= threshold for rel in top_k) else 0.0


def mean_relevance(relevances: Sequence[int | float], k: int = 5) -> float:
    """Computes the arithmetic mean of relevance scores in the top k."""
    top_k = list(relevances[:k])
    if not top_k:
        return 0.0
    return sum(top_k) / float(len(top_k))


def evaluate_ranking(
    relevances: Sequence[int | float],
    k: int = 5,
) -> dict[str, float]:
    """Computes a complete battery of IR metrics for a single query."""
    return {
        "ndcg_at_5": round(ndcg_at_k(relevances, k=k), 6),
        "precision_at_1_relaxed": round(
            precision_at_k(relevances, k=1, threshold=1), 6
        ),
        "precision_at_1_strict": round(
            precision_at_k(relevances, k=1, threshold=2), 6
        ),
        "precision_at_5_relaxed": round(
            precision_at_k(relevances, k=k, threshold=1), 6
        ),
        "precision_at_5_strict": round(
            precision_at_k(relevances, k=k, threshold=2), 6
        ),
        "mrr_relaxed": round(
            reciprocal_rank(relevances, threshold=1, max_rank=k), 6
        ),
        "mrr_strict": round(
            reciprocal_rank(relevances, threshold=2, max_rank=k), 6
        ),
        "map_at_5_relaxed": round(
            average_precision(relevances, k=k, threshold=1), 6
        ),
        "map_at_5_strict": round(
            average_precision(relevances, k=k, threshold=2), 6
        ),
        "hit_rate_relaxed": round(
            hit_rate_at_k(relevances, k=k, threshold=1), 6
        ),
        "hit_rate_strict": round(
            hit_rate_at_k(relevances, k=k, threshold=2), 6
        ),
        "mean_relevance": round(
            mean_relevance(relevances, k=k), 6
        ),
    }


def aggregate_query_metrics(
    query_evaluations: Sequence[dict[str, float]],
) -> dict[str, float]:
    """Aggregates per-query metrics into macro-averaged system performance scores."""
    if not query_evaluations:
        return {}

    metric_keys = list(query_evaluations[0].keys())
    aggregated: dict[str, float] = {}

    for key in metric_keys:
        values = [q[key] for q in query_evaluations if key in q]
        if values:
            aggregated[f"mean_{key}"] = round(sum(values) / len(values), 6)
        else:
            aggregated[f"mean_{key}"] = 0.0

    return aggregated
