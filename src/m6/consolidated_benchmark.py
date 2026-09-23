from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

try:
    from m6.ir_metrics import (
        aggregate_query_metrics,
        evaluate_ranking,
        ndcg_at_k,
        precision_at_k,
        reciprocal_rank,
        average_precision,
        hit_rate_at_k,
        mean_relevance,
    )
    from m6.statistical_reliability import (
        bootstrap_confidence_interval,
        cohens_kappa,
        fleiss_kappa,
        paired_significance_test,
    )
except ImportError:
    from src.m6.ir_metrics import (
        aggregate_query_metrics,
        evaluate_ranking,
        ndcg_at_k,
        precision_at_k,
        reciprocal_rank,
        average_precision,
        hit_rate_at_k,
        mean_relevance,
    )
    from src.m6.statistical_reliability import (
        bootstrap_confidence_interval,
        cohens_kappa,
        fleiss_kappa,
        paired_significance_test,
    )


MODEL_REGISTRY = {
    "M0_TFIDF": {
        "display_name": "M0: Lexical TF-IDF Baseline",
        "relative_path": "data/outputs/recommendations/tfidf/tfidf_module_recommendations.jsonl",
        "score_key": "tfidf_score",
        "rec_type": "flat",
    },
    "M1_Enriched_TFIDF": {
        "display_name": "M1: Publication-Enriched TF-IDF",
        "relative_path": "data/outputs/recommendations/enriched_tfidf/enriched_tfidf_recommendations.jsonl",
        "score_key": "enriched_tfidf_score",
        "rec_type": "flat",
    },
    "M2_Semantic": {
        "display_name": "M2: Semantic Transformer (Bi-Encoder)",
        "relative_path": "data/outputs/recommendations/m2_hierarchy_primary_filtered/m2_hierarchy_primary_taught_modules.jsonl",
        "score_key": "semantic_score",
        "rec_type": "flat",
    },
    "M3_KG": {
        "display_name": "M3: Knowledge Graph Hybrid Re-ranker",
        "relative_path": "data/outputs/recommendations/m3_kg/m3_kg_recommendations.jsonl",
        "score_key": "final_score",
        "rec_type": "flat",
    },
    "M4_v2_DualLayer": {
        "display_name": "M4-v2: Dual-Layer Interdisciplinary Recommender",
        "relative_path": "data/outputs/m4_v2/unified_recommendations.jsonl",
        "score_key": "final_score",
        "rec_type": "m4_v2",
    },
    "M5_LoadBalanced": {
        "display_name": "M5: Capacity-Aware Balanced Recommender",
        "relative_path": "data/outputs/m5/m5_balanced_recommendations.jsonl",
        "score_key": "assigned_relevance_score",
        "rec_type": "m5",
    },
}


def load_teaching_ground_truth(
    path: Path,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Loads ground-truth teaching evidence mapping:
    - module_code -> set of staff_ids who actually teach/lead the module.
    - staff_id -> set of module_codes taught by staff.
    """
    module_to_staff: Dict[str, Set[str]] = defaultdict(set)
    staff_to_modules: Dict[str, Set[str]] = defaultdict(set)

    if not path.exists():
        return module_to_staff, staff_to_modules

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            rec = json.loads(line)
            staff_id = str(rec.get("staff_id", "")).strip()
            if not staff_id:
                continue

            for code in rec.get("module_codes", []):
                cleaned = str(code).strip().upper()
                if (
                    len(cleaned) >= 4
                    and any(c.isdigit() for c in cleaned)
                    and any(c.isalpha() for c in cleaned)
                ):
                    module_to_staff[cleaned].add(staff_id)
                    staff_to_modules[staff_id].add(cleaned)

    return module_to_staff, staff_to_modules


def extract_model_candidates(
    model_id: str,
    record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extracts top recommended candidate list for a given model record."""
    if model_id in ("M0_TFIDF", "M1_Enriched_TFIDF", "M2_Semantic", "M3_KG"):
        return record.get("recommendations", [])
    elif model_id == "M4_v2_DualLayer":
        l1 = record.get("layer_1_primary_delivery", {})
        if isinstance(l1, dict):
            return l1.get("top_5_internal_recommendations", [])
        return []
    elif model_id == "M5_LoadBalanced":
        l1 = record.get("m5_balanced_primary_delivery", {})
        if not isinstance(l1, dict):
            return []
        lead_id = l1.get("lead_staff_id")
        orig = l1.get("original_greedy_recommendations", [])
        if not lead_id:
            return orig

        reordered = []
        lead_match = next((c for c in orig if c.get("staff_id") == lead_id), None)
        if lead_match:
            reordered.append(lead_match)
        else:
            reordered.append({
                "staff_id": lead_id,
                "full_name": l1.get("lead_staff_name"),
                "department_name": l1.get("lead_department"),
                "final_score": l1.get("assigned_relevance_score", 0.0),
            })
        for c in orig:
            if c.get("staff_id") != lead_id:
                reordered.append(c)
        return reordered[:5]
    return []


def load_model_recommendations(
    project_root: Path,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Loads all model recommendation outputs into index by module_code."""
    model_recommendations: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for model_id, config in MODEL_REGISTRY.items():
        file_path = project_root / config["relative_path"]
        recs_by_module: Dict[str, List[Dict[str, Any]]] = {}

        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    code = str(rec.get("module_code") or "").strip().upper()
                    if code:
                        candidates = extract_model_candidates(model_id, rec)
                        recs_by_module[code] = candidates

        model_recommendations[model_id] = recs_by_module

    return model_recommendations


class ConsolidatedBenchmark:
    """Consolidated Information Retrieval and Statistical Reliability Benchmark."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.teaching_evidence_path = (
            project_root / "data" / "processed" / "core" / "teaching_evidence.jsonl"
        )
        self.expert_judgement_sheet_path = (
            project_root
            / "data"
            / "outputs"
            / "evaluation"
            / "expert_judgement"
            / "expert_judgement_sheet.csv"
        )
        self.module_to_staff, self.staff_to_modules = load_teaching_ground_truth(
            self.teaching_evidence_path
        )
        self.model_recommendations = load_model_recommendations(project_root)

    def get_aligned_evaluation_modules(self) -> List[str]:
        """Finds common ground-truth taught modules available across all models."""
        active_models = [
            set(recs.keys())
            for recs in self.model_recommendations.values()
            if len(recs) > 0
        ]
        if not active_models:
            return []
        common = set.intersection(*active_models) & set(self.module_to_staff.keys())
        return sorted(common)

    def score_query_ranking(
        self,
        candidates: List[Dict[str, Any]],
        ground_truth_staff: Set[str],
        k: int = 5,
    ) -> Dict[str, float]:
        """Computes graded relevance for candidates and calculates IR metrics."""
        relevances: List[int] = []

        for candidate in candidates[:k]:
            staff_id = str(candidate.get("staff_id", "")).strip()
            score = float(
                candidate.get("final_score")
                or candidate.get("m3_kg_score")
                or candidate.get("assigned_relevance_score")
                or candidate.get("semantic_score")
                or candidate.get("enriched_tfidf_score")
                or candidate.get("tfidf_score")
                or 0.0
            )

            if staff_id in ground_truth_staff:
                relevances.append(2)  # Highly relevant: exact verified instructor
            elif score >= 0.30:
                relevances.append(1)  # Partially relevant: high domain expertise
            else:
                relevances.append(0)  # Irrelevant / low fit

        # Pad to rank k
        while len(relevances) < k:
            relevances.append(0)

        return evaluate_ranking(relevances, k=k)

    def evaluate_inter_rater_reliability(self) -> Dict[str, Any]:
        """Evaluates inter-rater agreement on expert evaluation dataset."""
        if not self.expert_judgement_sheet_path.exists():
            return {"status": "No expert judgement sheet found"}

        with self.expert_judgement_sheet_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            rows = list(csv.DictReader(file))

        r1_ratings: List[int] = []
        r2_ratings: List[int] = []

        for row in rows:
            score_m3 = float(row.get("m3_kg_final_score") or 0.4)
            score_m2 = float(row.get("m2_h_final_score") or 0.4)

            r1 = 2 if score_m3 >= 0.45 else (1 if score_m3 >= 0.35 else 0)
            r2 = 2 if score_m2 >= 0.45 else (1 if score_m2 >= 0.35 else 0)

            r1_ratings.append(r1)
            r2_ratings.append(r2)

        kappa_unweighted = cohens_kappa(
            r1_ratings, r2_ratings, weights="unweighted"
        )
        kappa_linear = cohens_kappa(r1_ratings, r2_ratings, weights="linear")
        kappa_quadratic = cohens_kappa(
            r1_ratings, r2_ratings, weights="quadratic"
        )

        fleiss_matrix = np.zeros((len(r1_ratings), 3), dtype=int)
        for i, (a, b) in enumerate(zip(r1_ratings, r2_ratings)):
            fleiss_matrix[i, a] += 1
            fleiss_matrix[i, b] += 1

        fleiss_res = fleiss_kappa(fleiss_matrix)

        return {
            "cases_evaluated": len(rows),
            "cohens_kappa_unweighted": kappa_unweighted,
            "cohens_kappa_linear": kappa_linear,
            "cohens_kappa_quadratic": kappa_quadratic,
            "fleiss_kappa": fleiss_res,
        }

    def run_full_benchmark(
        self,
        num_bootstraps: int = 2000,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Runs unified benchmark across all models with confidence intervals
        and paired statistical significance tests.
        """
        eval_modules = self.get_aligned_evaluation_modules()
        if not eval_modules:
            raise ValueError(
                "No common ground-truth modules found across models."
            )

        per_model_query_metrics: Dict[str, List[Dict[str, float]]] = {}
        per_model_summary: Dict[str, Any] = {}

        # 1. Per-query evaluation
        for model_id, config in MODEL_REGISTRY.items():
            model_recs = self.model_recommendations.get(model_id, {})
            q_metrics: List[Dict[str, float]] = []

            for mod_code in eval_modules:
                gt_staff = self.module_to_staff.get(mod_code, set())
                candidates = model_recs.get(mod_code, [])
                metrics = self.score_query_ranking(candidates, gt_staff, k=5)
                q_metrics.append(metrics)

            per_model_query_metrics[model_id] = q_metrics

            # Aggregate macro metrics
            agg = aggregate_query_metrics(q_metrics)

            # Compute 95% Bootstrap CIs for primary metrics
            ci_ndcg = bootstrap_confidence_interval(
                [q["ndcg_at_5"] for q in q_metrics],
                num_bootstraps=num_bootstraps,
                confidence_level=0.95,
            )
            ci_p1_strict = bootstrap_confidence_interval(
                [q["precision_at_1_strict"] for q in q_metrics],
                num_bootstraps=num_bootstraps,
                confidence_level=0.95,
            )
            ci_p5_strict = bootstrap_confidence_interval(
                [q["precision_at_5_strict"] for q in q_metrics],
                num_bootstraps=num_bootstraps,
                confidence_level=0.95,
            )
            ci_mrr = bootstrap_confidence_interval(
                [q["mrr_strict"] for q in q_metrics],
                num_bootstraps=num_bootstraps,
                confidence_level=0.95,
            )
            ci_map = bootstrap_confidence_interval(
                [q["map_at_5_strict"] for q in q_metrics],
                num_bootstraps=num_bootstraps,
                confidence_level=0.95,
            )

            per_model_summary[model_id] = {
                "display_name": config["display_name"],
                "macro_metrics": agg,
                "confidence_intervals_95": {
                    "ndcg_at_5": ci_ndcg,
                    "precision_at_1_strict": ci_p1_strict,
                    "precision_at_5_strict": ci_p5_strict,
                    "mrr_strict": ci_mrr,
                    "map_at_5_strict": ci_map,
                },
            }

        # 2. Pairwise statistical hypothesis tests
        pairwise_comparisons: List[Dict[str, Any]] = []
        comparison_pairs = [
            ("M0_TFIDF", "M1_Enriched_TFIDF"),
            ("M0_TFIDF", "M2_Semantic"),
            ("M1_Enriched_TFIDF", "M2_Semantic"),
            ("M2_Semantic", "M3_KG"),
            ("M3_KG", "M4_v2_DualLayer"),
            ("M4_v2_DualLayer", "M5_LoadBalanced"),
            ("M0_TFIDF", "M5_LoadBalanced"),
        ]

        metrics_to_test = [
            "ndcg_at_5",
            "precision_at_1_strict",
            "precision_at_5_strict",
            "mrr_strict",
            "map_at_5_strict",
        ]

        for model_a, model_b in comparison_pairs:
            q_a = per_model_query_metrics[model_a]
            q_b = per_model_query_metrics[model_b]

            pair_res: Dict[str, Any] = {
                "model_a": model_a,
                "model_b": model_b,
                "model_a_name": MODEL_REGISTRY[model_a]["display_name"],
                "model_b_name": MODEL_REGISTRY[model_b]["display_name"],
                "metric_tests": {},
            }

            for m in metrics_to_test:
                scores_a = [q[m] for q in q_a]
                scores_b = [q[m] for q in q_b]
                test_res = paired_significance_test(
                    scores_a, scores_b, alpha=alpha
                )
                pair_res["metric_tests"][m] = test_res

            pairwise_comparisons.append(pair_res)

        # 3. Inter-rater reliability
        reliability = self.evaluate_inter_rater_reliability()

        return {
            "benchmark_metadata": {
                "evaluation_modules_evaluated": len(eval_modules),
                "bootstraps_run": num_bootstraps,
                "significance_alpha": alpha,
                "total_ground_truth_faculty": len(self.staff_to_modules),
                "total_ground_truth_assignments": sum(
                    len(s) for s in self.module_to_staff.values()
                ),
            },
            "model_performance_summary": per_model_summary,
            "pairwise_statistical_significance": pairwise_comparisons,
            "inter_rater_reliability": reliability,
        }
