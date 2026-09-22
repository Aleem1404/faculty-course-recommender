from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def check_recommendations(module_code: str = "CS5707") -> None:
    project_root = Path(__file__).resolve().parents[1]

    models = {
        "M0 (TF-IDF Baseline)": project_root / "data" / "outputs" / "recommendations" / "tfidf" / "tfidf_module_recommendations.jsonl",
        "M1 (Enriched TF-IDF)": project_root / "data" / "outputs" / "recommendations" / "enriched_tfidf" / "enriched_tfidf_recommendations.jsonl",
        "M2-H (Semantic Hierarchy)": project_root / "data" / "outputs" / "recommendations" / "m2_hierarchy_primary" / "m2_hierarchy_primary_recommendations.jsonl",
        "M3-KG (Knowledge Graph)": project_root / "data" / "outputs" / "recommendations" / "m3_kg" / "m3_kg_recommendations.jsonl",
    }

    comparison_path = (
        project_root
        / "data"
        / "outputs"
        / "evaluation"
        / "m2_vs_m3"
        / "m2_h_vs_m3_kg_module_comparisons.json"
    )

    print(f"\n=======================================================")
    print(f"   FACULTY RECOMMENDATIONS & METRICS FOR: {module_code.upper()}")
    print(f"=======================================================")

    m2_ranks: dict[str, int] = {}

    for model_name, path in models.items():
        print(f"\n{'='*25} {model_name} {'='*25}")
        if not path.exists():
            print(f"File not found: {path}")
            continue

        found = False
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)
                if str(record.get("module_code", "")).upper() == module_code.upper():
                    found = True
                    print(f"Module: {record.get('module_code')} - {record.get('module_title')}")
                    if "decision_status" in record:
                        print(f"Decision Status: {record.get('decision_status')}")

                    recs = record.get("recommendations", [])[:5]
                    for rec in recs:
                        rank = rec.get("rank")
                        staff_id = rec.get("staff_id")
                        name = rec.get("full_name")
                        dept = rec.get("department_name")
                        role = rec.get("recommendation_role", "candidate")

                        if "M2-H" in model_name:
                            m2_ranks[staff_id] = rank

                        print(f"\n  [Rank {rank}] {name} ({dept}) | Role: {role}")

                        # Scores
                        if "tfidf_score" in rec:
                            print(f"    - TF-IDF Score: {rec['tfidf_score']:.6f}")
                        if "enriched_tfidf_score" in rec:
                            print(f"    - Enriched TF-IDF Score: {rec['enriched_tfidf_score']:.6f}")
                            if "field_scores" in rec:
                                print(f"      Field Scores: {rec['field_scores']}")
                        if "semantic_score" in rec and "M2-H" in model_name:
                            print(f"    - Semantic Score: {rec['semantic_score']:.6f}")
                            if "field_scores" in rec:
                                print(f"      Field Scores: {rec['field_scores']}")
                            if "field_contributions" in rec:
                                print(f"      Contributions: {rec['field_contributions']}")

                        # M3 KG Details
                        if "m3_kg_score" in rec:
                            kg_score = rec["m3_kg_score"]
                            components = rec.get("m3_kg_components", {})
                            sem_s = components.get("semantic_score", 0.0)
                            graph_s = components.get("graph_support_score", 0.0)
                            sem_w = components.get("semantic_weight", 0.8)
                            graph_w = components.get("graph_weight", 0.2)

                            prev_rank = m2_ranks.get(staff_id)
                            rank_shift_str = ""
                            if prev_rank is not None:
                                if prev_rank > rank:
                                    rank_shift_str = f" [PROMOTED: Rank {prev_rank} -> {rank}]"
                                elif prev_rank < rank:
                                    rank_shift_str = f" [DEMOTED: Rank {prev_rank} -> {rank}]"
                                else:
                                    rank_shift_str = f" [UNCHANGED: Rank {rank}]"

                            print(f"    - Final M3-KG Score: {kg_score:.6f}{rank_shift_str}")
                            print(f"      Formula: ({sem_w} * {sem_s:.6f}) + ({graph_w} * {graph_s:.2f})")
                            print(f"      Graph Support Score: {graph_s:.2f}")

                        # Explanations
                        explanation = rec.get("explanation")
                        if explanation:
                            print(f"    - Explanation: \"{explanation}\"")

                        shared_topics = rec.get("shared_topics") or rec.get("graph_features", {}).get("shared_topics", [])
                        if shared_topics:
                            print(f"    - Shared Topics in Knowledge Graph: {', '.join(shared_topics)}")

                        # Evidence Terms preview
                        gate_diag = rec.get("gate", {}).get("diagnostics", {})
                        preview_terms = gate_diag.get("evidence_terms_preview", [])
                        if preview_terms:
                            print(f"    - Matched Evidence Terms ({len(preview_terms)} terms): {', '.join(preview_terms[:6])}...")

                    break

        if not found:
            print(f"No recommendations found for module {module_code}")

    # Comparative evaluation metrics if available
    if comparison_path.exists():
        try:
            comparisons = json.loads(comparison_path.read_text(encoding="utf-8"))
            module_comp = next((m for m in comparisons if str(m.get("module_code", "")).upper() == module_code.upper()), None)
            if module_comp:
                print(f"\n{'='*25} M2-H vs M3-KG Comparison Metrics {'='*25}")
                print(f"Candidate Count: {module_comp.get('candidate_count_m3')}")
                print(f"Candidates with Graph Support: {module_comp.get('candidates_with_graph_support')}")
                print(f"Rank Changes Count: {module_comp.get('changed_rank_count')} / {module_comp.get('candidate_count_m3')}")
                print(f"Top 1 Faculty Changed: {module_comp.get('top_rank_changed')}")
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Check faculty recommendations across models M0 to M3 with full explanations and metrics.")
    parser.add_argument("--module", default="CS5707", help="Module code to inspect (default: CS5707)")
    args = parser.parse_args()

    check_recommendations(args.module)


if __name__ == "__main__":
    main()
