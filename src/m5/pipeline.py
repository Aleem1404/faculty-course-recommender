from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from m5.capacity_policy import CapacityPolicy
from m5.explanation_generator import M5ExplanationGenerator
from m5.fairness_metrics import FairnessMetricsCalculator
from m5.load_balancer import M5LoadBalancer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("m5.pipeline")


class M5Pipeline:
    """End-to-end pipeline orchestrating M5 Load-Balanced and Fair Academic Allocation."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        policy: Optional[CapacityPolicy] = None,
    ):
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.policy = policy or CapacityPolicy()
        self.balancer = M5LoadBalancer(self.policy)

    def load_m4_v2_recommendations(self, path: Optional[Path] = None) -> list[dict[str, Any]]:
        """Loads unified recommendations output from M4-v2."""
        target_path = (
            path
            or self.project_root
            / "data"
            / "outputs"
            / "m4_v2"
            / "unified_recommendations.jsonl"
        )
        if not target_path.exists():
            raise FileNotFoundError(f"M4-v2 recommendations not found at {target_path}")

        records: list[dict[str, Any]] = []
        with target_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        LOGGER.info(f"Loaded {len(records)} module recommendations from {target_path.name}")
        return records

    def load_faculty_metadata(self, path: Optional[Path] = None) -> dict[str, dict[str, Any]]:
        """Loads faculty profiles to extract position, department, and academic seniority."""
        target_path = (
            path
            or self.project_root
            / "data"
            / "processed"
            / "core"
            / "staff.jsonl"
        )
        if not target_path.exists():
            LOGGER.warning(f"Staff metadata not found at {target_path}; proceeding with empty metadata.")
            return {}

        meta: dict[str, dict[str, Any]] = {}
        with target_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    staff_id = row.get("staff_id")
                    if staff_id:
                        meta[staff_id] = row

        LOGGER.info(f"Loaded metadata for {len(meta)} faculty members.")
        return meta

    def run(
        self,
        output_dir: Optional[Path] = None,
    ) -> dict[str, Any]:
        """Executes full optimization, fairness audit, and exports results."""
        out_dir = output_dir or self.project_root / "data" / "outputs" / "m5"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load inputs
        module_records = self.load_m4_v2_recommendations()
        staff_meta = self.load_faculty_metadata()
        all_faculty_ids = list(staff_meta.keys()) if staff_meta else []

        # 2. Extract baseline unconstrained greedy allocations
        greedy_allocations = []
        for m in module_records:
            if m.get("module_classification") != "independent_project_placement":
                top_recs = m.get("layer_1_primary_delivery", {}).get("top_5_internal_recommendations", [])
                if top_recs:
                    greedy_allocations.append({
                        "module_id": m.get("module_id"),
                        "staff_id": top_recs[0].get("staff_id"),
                        "m3_kg_score": float(top_recs[0].get("m3_kg_score") or top_recs[0].get("semantic_score") or 0.0),
                    })

        # Evaluate Baseline Greedy Fairness
        baseline_report = FairnessMetricsCalculator.evaluate_allocations(
            allocations=greedy_allocations,
            all_faculty_pool=all_faculty_ids,
        )

        # 3. Run M5 Load Balancing Optimization
        balanced_results = self.balancer.balance_allocations(module_records, staff_meta)

        # Extract balanced allocations for metric evaluation
        balanced_eval_records = []
        for r in balanced_results:
            if r.module_classification != "independent_project_placement":
                balanced_eval_records.append({
                    "module_id": r.module_id,
                    "staff_id": r.assigned_staff_id,
                    "m3_kg_score": r.assigned_score,
                })

        balanced_report = FairnessMetricsCalculator.evaluate_allocations(
            allocations=balanced_eval_records,
            all_faculty_pool=all_faculty_ids,
        )

        # 4. Generate Unified Output Records with Explanations & Dual Layers
        unified_m5_records: list[dict[str, Any]] = []
        csv_rows: list[dict[str, Any]] = []

        for idx, (m, b_res) in enumerate(zip(module_records, balanced_results)):
            explanation_payload = M5ExplanationGenerator.generate_explanation(b_res)

            # Build updated layer 1 with M5 allocation
            orig_layer_1 = m.get("layer_1_primary_delivery", {})
            orig_layer_2 = m.get("layer_2_intelligent_collaboration", {})

            m5_layer_1 = {
                "lead_staff_id": b_res.assigned_staff_id,
                "lead_staff_name": b_res.assigned_staff_name,
                "lead_department": b_res.assigned_department,
                "supervision_role": orig_layer_1.get("supervision_role", "Primary Module Leader"),
                "assigned_relevance_score": b_res.assigned_score,
                "rank_shift_from_greedy": b_res.rank_shift,
                "is_capacity_constrained": b_res.is_displaced_due_to_capacity,
                "m5_workload_policy_explanation": explanation_payload,
                "original_greedy_recommendations": b_res.all_top_candidates,
            }

            unified_record = {
                "module_id": b_res.module_id,
                "module_code": b_res.module_code,
                "module_title": b_res.module_title,
                "module_classification": b_res.module_classification,
                "collaboration_eligible": m.get("collaboration_eligible", True),
                "module_departments": m.get("module_departments", []),
                "module_colleges": m.get("module_colleges", []),
                "layer_1_primary_delivery": orig_layer_1,
                "m5_balanced_primary_delivery": m5_layer_1,
                "layer_2_intelligent_collaboration": orig_layer_2,
            }

            unified_m5_records.append(unified_record)

            # Build row for CSV case study audit
            csv_rows.append({
                "module_code": b_res.module_code,
                "module_title": b_res.module_title,
                "classification": b_res.module_classification,
                "m5_assigned_staff": b_res.assigned_staff_name,
                "m5_assigned_dept": b_res.assigned_department,
                "m5_assigned_score": round(b_res.assigned_score, 4),
                "greedy_original_staff": b_res.all_top_candidates[0].get("full_name") if b_res.all_top_candidates else "N/A",
                "greedy_original_score": round(b_res.original_greedy_score, 4),
                "rank_shift": b_res.rank_shift,
                "capacity_displaced": "YES" if b_res.is_displaced_due_to_capacity else "NO",
                "policy_flag": explanation_payload["policy_flag"],
                "explanation_summary": explanation_payload["summary"],
                "full_rationale": explanation_payload["allocation_rationale"],
            })

        # 5. Write JSONL unified recommendations
        jsonl_path = out_dir / "m5_balanced_recommendations.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for rec in unified_m5_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # 6. Write CSV review sheet
        csv_path = out_dir / "m5_allocation_audit_sheet.csv"
        if csv_rows:
            with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)

        # 7. Write Comparison & Evaluation Summary
        displaced_modules_count = sum(1 for r in balanced_results if r.is_displaced_due_to_capacity)
        total_taught_modules = len(balanced_eval_records)

        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_version": "M5 (Multi-Objective Load-Balanced & Fair Allocation)",
            "total_catalog_modules": len(module_records),
            "total_taught_modules": total_taught_modules,
            "exempt_modules": len(module_records) - total_taught_modules,
            "capacity_policy": {
                "default_max_primary_modules": self.policy.default_max_primary_modules,
                "min_competency_threshold": self.policy.min_competency_threshold,
            },
            "comparison_metrics": {
                "baseline_greedy_m4": baseline_report.to_dict(),
                "m5_load_balanced": balanced_report.to_dict(),
                "delta_improvements": {
                    "gini_reduction": round(baseline_report.gini_coefficient - balanced_report.gini_coefficient, 4),
                    "gini_reduction_pct": round(
                        ((baseline_report.gini_coefficient - balanced_report.gini_coefficient) / baseline_report.gini_coefficient) * 100.0,
                        2
                    ) if baseline_report.gini_coefficient else 0.0,
                    "faculty_utilization_increase_pct": round(
                        balanced_report.faculty_utilization_pct - baseline_report.faculty_utilization_pct, 2
                    ),
                    "top_10_concentration_reduction_pct": round(
                        baseline_report.top_10_faculty_concentration_pct - balanced_report.top_10_faculty_concentration_pct, 2
                    ),
                    "relevance_tradeoff_cost": round(
                        baseline_report.average_relevance_score - balanced_report.average_relevance_score, 4
                    ),
                    "relevance_retention_pct": round(
                        (balanced_report.average_relevance_score / baseline_report.average_relevance_score) * 100.0, 2
                    ) if baseline_report.average_relevance_score else 100.0,
                    "reallocated_modules_count": displaced_modules_count,
                    "reallocated_modules_pct": round((displaced_modules_count / total_taught_modules * 100.0), 2) if total_taught_modules else 0.0,
                }
            },
            "outputs": {
                "jsonl_recommendations": str(jsonl_path),
                "csv_audit_sheet": str(csv_path),
            }
        }

        summary_path = out_dir / "m5_fairness_evaluation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        LOGGER.info(f"M5 Execution Complete. Summary saved to {summary_path.name}")
        return summary
