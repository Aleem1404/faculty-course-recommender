"""M6: Formal Information Retrieval & Statistical Reliability Benchmark Framework.

Exposes standard IR evaluation metrics (NDCG@K, P@K, MRR, MAP),
inter-rater agreement statistics (Cohen's Kappa, Fleiss' Kappa),
bootstrap confidence intervals, and paired significance hypothesis testing.
"""

try:
    from m6.ir_metrics import (
        aggregate_query_metrics,
        average_precision,
        dcg_at_k,
        evaluate_ranking,
        hit_rate_at_k,
        idcg_at_k,
        mean_relevance,
        ndcg_at_k,
        precision_at_k,
        reciprocal_rank,
    )
    from m6.statistical_reliability import (
        bootstrap_confidence_interval,
        cohens_kappa,
        fleiss_kappa,
        interpret_kappa,
        paired_significance_test,
    )
    from m6.consolidated_benchmark import (
        ConsolidatedBenchmark,
        MODEL_REGISTRY,
    )
    from m6.pipeline import run_m6_pipeline
except ImportError:
    from src.m6.ir_metrics import (
        aggregate_query_metrics,
        average_precision,
        dcg_at_k,
        evaluate_ranking,
        hit_rate_at_k,
        idcg_at_k,
        mean_relevance,
        ndcg_at_k,
        precision_at_k,
        reciprocal_rank,
    )
    from src.m6.statistical_reliability import (
        bootstrap_confidence_interval,
        cohens_kappa,
        fleiss_kappa,
        interpret_kappa,
        paired_significance_test,
    )
    from src.m6.consolidated_benchmark import (
        ConsolidatedBenchmark,
        MODEL_REGISTRY,
    )
    from src.m6.pipeline import run_m6_pipeline

__all__ = [
    "dcg_at_k",
    "idcg_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "average_precision",
    "hit_rate_at_k",
    "mean_relevance",
    "evaluate_ranking",
    "aggregate_query_metrics",
    "interpret_kappa",
    "cohens_kappa",
    "fleiss_kappa",
    "bootstrap_confidence_interval",
    "paired_significance_test",
    "ConsolidatedBenchmark",
    "MODEL_REGISTRY",
    "run_m6_pipeline",
]
