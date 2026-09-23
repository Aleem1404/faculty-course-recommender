from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to sys.path for direct script execution
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from m6.pipeline import run_m6_pipeline


def print_table_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(f" {title}")
    print("=" * 90)


def main() -> None:
    print("Executing M6: Formal Information Retrieval & Statistical Reliability Benchmark...")
    results = run_m6_pipeline(
        project_root=project_root,
        num_bootstraps=2000,
        alpha=0.05,
    )

    models_summary = results["model_performance_summary"]
    pairwise = results["pairwise_statistical_significance"]
    reliability = results["inter_rater_reliability"]
    meta = results["benchmark_metadata"]

    # 1. Consolidated IR Benchmark Table
    print_table_header(
        f"CONSOLIDATED INFORMATION RETRIEVAL BENCHMARK (N={meta['evaluation_modules_evaluated']} Taught Modules, B={meta['bootstraps_run']} Bootstraps)"
    )
    header = (
        f"{'Model Architecture':<38} | {'NDCG@5 (95% CI)':<22} | {'P@1 Strict':<10} | {'P@5 Strict':<10} | {'MRR Strict':<10} | {'MAP@5':<10}"
    )
    print(header)
    print("-" * 115)

    for model_id, data in models_summary.items():
        name = data["display_name"]
        macro = data["macro_metrics"]
        cis = data["confidence_intervals_95"]
        ndcg_str = f"{macro['mean_ndcg_at_5']:.4f} [{cis['ndcg_at_5']['ci_lower']:.3f}-{cis['ndcg_at_5']['ci_upper']:.3f}]"
        print(
            f"{name:<38} | {ndcg_str:<22} | {macro['mean_precision_at_1_strict']:<10.4f} | {macro['mean_precision_at_5_strict']:<10.4f} | {macro['mean_mrr_strict']:<10.4f} | {macro['mean_map_at_5_strict']:<10.4f}"
        )

    # 2. Pairwise Hypothesis Testing Table
    print_table_header("PAIRWISE STATISTICAL SIGNIFICANCE & EFFECT SIZES (alpha=0.05)")
    pair_header = (
        f"{'Ablation Transition':<34} | {'Metric':<18} | {'Mean A -> B':<16} | {'Gain (%)':<10} | {'p-value':<10} | {'Cohen d':<8} | {'Sig?'}"
    )
    print(pair_header)
    print("-" * 115)

    for pair in pairwise:
        m_a = pair["model_a"].replace("_TFIDF", "").replace("_Semantic", "").replace("_LoadBalanced", "")
        m_b = pair["model_b"].replace("_TFIDF", "").replace("_Semantic", "").replace("_LoadBalanced", "")
        trans_name = f"{m_a} -> {m_b}"
        for metric, test in pair["metric_tests"].items():
            diff_str = f"{test['mean_a']:.3f} -> {test['mean_b']:.3f}"
            sig_str = "YES (p<0.05)" if test["is_significant"] else "No (p>=0.05)"
            p_val_str = f"{test['primary_p_value']:.4f}" if test['primary_p_value'] >= 0.0001 else "<0.0001"
            print(
                f"{trans_name:<34} | {metric:<18} | {diff_str:<16} | {test['percentage_gain']:>8.1f}% | {p_val_str:<10} | {test['cohens_d']:>7.2f} | {sig_str}"
            )
        print("-" * 115)

    # 3. Inter-Rater Reliability Table
    print_table_header("INTER-RATER RELIABILITY & AGREEMENT ANALYSIS")
    if "cohens_kappa_unweighted" in reliability:
        ku = reliability["cohens_kappa_unweighted"]
        kl = reliability["cohens_kappa_linear"]
        kq = reliability["cohens_kappa_quadratic"]
        fl = reliability["fleiss_kappa"]
        print(f"Sample Size (Double-Coded Cases) : {reliability['cases_evaluated']}")
        print(f"Cohen's Kappa (Unweighted)       : {ku['kappa']:.4f}  ({ku['interpretation']}) [Po={ku['observed_agreement']:.3f}, Pe={ku['expected_agreement']:.3f}]")
        print(f"Cohen's Kappa (Linear Weighted)  : {kl['kappa']:.4f}  ({kl['interpretation']})")
        print(f"Cohen's Kappa (Quadratic Weighted: {kq['kappa']:.4f}  ({kq['interpretation']})")
        print(f"Fleiss' Kappa (Multi-Rater Matrix: {fl['kappa']:.4f}  ({fl['interpretation']}) [Po={fl['observed_agreement']:.3f}, Pe={fl['expected_agreement']:.3f}]")

    print_table_header("M6 BENCHMARK COMPLETE")
    print(f"Output files written to: {project_root / 'data' / 'outputs' / 'm6'}\n")


if __name__ == "__main__":
    main()
