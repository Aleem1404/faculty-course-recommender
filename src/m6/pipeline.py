from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict
import numpy as np

try:
    from m6.consolidated_benchmark import ConsolidatedBenchmark
except ImportError:
    from src.m6.consolidated_benchmark import ConsolidatedBenchmark


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_m6_pipeline(
    project_root: Path | None = None,
    output_dir: Path | None = None,
    num_bootstraps: int = 2000,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Executes the end-to-end M6 Information Retrieval & Statistical Reliability benchmark."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    if output_dir is None:
        output_dir = project_root / "data" / "outputs" / "m6"

    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = ConsolidatedBenchmark(project_root)
    results = benchmark.run_full_benchmark(
        num_bootstraps=num_bootstraps,
        alpha=alpha,
    )

    # 1. Save Full JSON Benchmark Report
    json_path = output_dir / "consolidated_ir_benchmark.json"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, cls=NpEncoder),
        encoding="utf-8",
    )

    # 2. Save CSV Summary Table
    csv_summary_path = output_dir / "consolidated_ir_benchmark_table.csv"
    models_summary = results["model_performance_summary"]

    fieldnames = [
        "model_id",
        "model_name",
        "ndcg_at_5",
        "ndcg_at_5_95_ci",
        "p_at_1_strict",
        "p_at_1_strict_95_ci",
        "p_at_5_strict",
        "p_at_5_strict_95_ci",
        "mrr_strict",
        "mrr_strict_95_ci",
        "map_at_5_strict",
        "map_at_5_strict_95_ci",
        "mean_relevance",
    ]

    csv_rows = []
    for model_id, data in models_summary.items():
        macro = data["macro_metrics"]
        cis = data["confidence_intervals_95"]

        def format_ci(ci_dict: Dict[str, float]) -> str:
            return f"[{ci_dict['ci_lower']:.4f}, {ci_dict['ci_upper']:.4f}]"

        csv_rows.append({
            "model_id": model_id,
            "model_name": data["display_name"],
            "ndcg_at_5": f"{macro['mean_ndcg_at_5']:.4f}",
            "ndcg_at_5_95_ci": format_ci(cis["ndcg_at_5"]),
            "p_at_1_strict": f"{macro['mean_precision_at_1_strict']:.4f}",
            "p_at_1_strict_95_ci": format_ci(cis["precision_at_1_strict"]),
            "p_at_5_strict": f"{macro['mean_precision_at_5_strict']:.4f}",
            "p_at_5_strict_95_ci": format_ci(cis["precision_at_5_strict"]),
            "mrr_strict": f"{macro['mean_mrr_strict']:.4f}",
            "mrr_strict_95_ci": format_ci(cis["mrr_strict"]),
            "map_at_5_strict": f"{macro['mean_map_at_5_strict']:.4f}",
            "map_at_5_strict_95_ci": format_ci(cis["map_at_5_strict"]),
            "mean_relevance": f"{macro['mean_mean_relevance']:.4f}",
        })

    with csv_summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # 3. Save Pairwise Statistical Significance Matrix CSV
    csv_pairwise_path = output_dir / "pairwise_statistical_significance.csv"
    pairwise_rows = []

    for pair in results["pairwise_statistical_significance"]:
        m_a = pair["model_a"]
        m_b = pair["model_b"]
        for metric_name, test in pair["metric_tests"].items():
            pairwise_rows.append({
                "comparison": f"{m_a} -> {m_b}",
                "model_a": m_a,
                "model_b": m_b,
                "metric": metric_name,
                "mean_a": test["mean_a"],
                "mean_b": test["mean_b"],
                "mean_difference": test["mean_difference"],
                "percentage_gain": f"{test['percentage_gain']}%",
                "test_used": test["primary_test"],
                "p_value": test["primary_p_value"],
                "cohens_d": test["cohens_d"],
                "is_significant": test["is_significant"],
            })

    pair_fieldnames = [
        "comparison",
        "model_a",
        "model_b",
        "metric",
        "mean_a",
        "mean_b",
        "mean_difference",
        "percentage_gain",
        "test_used",
        "p_value",
        "cohens_d",
        "is_significant",
    ]

    with csv_pairwise_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=pair_fieldnames)
        writer.writeheader()
        writer.writerows(pairwise_rows)

    # 4. Save Inter-Rater Reliability JSON
    reliability_path = output_dir / "inter_rater_reliability.json"
    reliability_path.write_text(
        json.dumps(
            results["inter_rater_reliability"], indent=2, ensure_ascii=False, cls=NpEncoder
        ),
        encoding="utf-8",
    )

    return results
