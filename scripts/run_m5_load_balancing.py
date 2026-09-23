from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from m5.capacity_policy import CapacityPolicy
from m5.pipeline import M5Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run M5: Multi-Objective Load-Balanced and Fair Academic Allocation."
    )
    parser.add_argument(
        "--max-primary-modules",
        type=int,
        default=3,
        help="Default maximum primary modules allocated per faculty member (default: 3).",
    )
    parser.add_argument(
        "--min-competency",
        type=float,
        default=0.15,
        help="Minimum relevance score threshold for qualification (default: 0.15).",
    )
    args = parser.parse_args()

    policy = CapacityPolicy(
        default_max_primary_modules=args.max_primary_modules,
        min_competency_threshold=args.min_competency,
    )

    pipeline = M5Pipeline(project_root=PROJECT_ROOT, policy=policy)
    summary = pipeline.run()

    print("\n" + "=" * 70)
    print("      M5 LOAD-BALANCED & FAIR ALLOCATION EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total Modules Processed : {summary['total_catalog_modules']}")
    print(f"Taught Modules Balanced : {summary['total_taught_modules']}")
    print(f"Exempt Modules          : {summary['exempt_modules']}")
    print("-" * 70)
    
    comp = summary["comparison_metrics"]
    delta = comp["delta_improvements"]
    b_base = comp["baseline_greedy_m4"]
    b_m5 = comp["m5_load_balanced"]

    print(f"Gini Coefficient        : {b_base['gini_coefficient']:.4f} (Greedy M4) -> {b_m5['gini_coefficient']:.4f} (M5 Balanced) [Reduced by {delta['gini_reduction_pct']}%]")
    print(f"Faculty Utilization     : {b_base['faculty_utilization_pct']}% -> {b_m5['faculty_utilization_pct']}% [+{delta['faculty_utilization_increase_pct']}% active faculty]")
    print(f"Max Individual Workload : {b_base['max_workload']} modules -> {b_m5['max_workload']} modules (Cap enforced)")
    print(f"Top 10 Concentration    : {b_base['top_10_faculty_concentration_pct']}% -> {b_m5['top_10_faculty_concentration_pct']}%")
    print(f"Average Relevance Score : {b_base['average_relevance_score']:.4f} -> {b_m5['average_relevance_score']:.4f} ({delta['relevance_retention_pct']}% retention)")
    print(f"Reallocated Modules     : {delta['reallocated_modules_count']} ({delta['reallocated_modules_pct']}%)")
    print("=" * 70)
    print(f"Outputs written to: {summary['outputs']['jsonl_recommendations']}")
    print(f"Audit CSV sheet:    {summary['outputs']['csv_audit_sheet']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
