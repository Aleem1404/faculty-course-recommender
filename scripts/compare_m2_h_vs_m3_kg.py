from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def write_json(
    path: Path,
    data: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return round(float(mean(values)), 6)


def recommendation_index(
    recommendations: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}

    for recommendation in recommendations:
        staff_id = recommendation.get("staff_id")

        if not staff_id:
            continue

        index[str(staff_id)] = recommendation

    return index


def compare_module_results(
    m2_result: dict[str, Any],
    m3_result: dict[str, Any],
) -> dict[str, Any]:
    m2_recommendations = m2_result.get(
        "recommendations",
        [],
    )
    m3_recommendations = m3_result.get(
        "recommendations",
        [],
    )

    m2_index = recommendation_index(
        m2_recommendations
    )
    m3_index = recommendation_index(
        m3_recommendations
    )

    all_staff_ids = sorted(
        set(m2_index).union(set(m3_index))
    )

    rank_changes: list[dict[str, Any]] = []
    changed_ranks = 0
    candidates_with_graph_support = 0

    for staff_id in all_staff_ids:
        m2_candidate = m2_index.get(staff_id)
        m3_candidate = m3_index.get(staff_id)

        m2_rank = (
            m2_candidate.get("rank")
            if m2_candidate
            else None
        )
        m3_rank = (
            m3_candidate.get("rank")
            if m3_candidate
            else None
        )

        semantic_score = 0.0
        m3_kg_score = 0.0
        graph_support_score = 0.0
        shared_topic_count = 0
        graph_explanation = ""

        if m2_candidate is not None:
            semantic_score = float(
                m2_candidate.get(
                    "semantic_score",
                    0.0,
                )
            )

        if m3_candidate is not None:
            semantic_score = float(
                m3_candidate.get(
                    "semantic_score",
                    semantic_score,
                )
            )
            m3_kg_score = float(
                m3_candidate.get(
                    "m3_kg_score",
                    0.0,
                )
            )
            graph_support_score = float(
                m3_candidate
                .get("m3_kg_components", {})
                .get(
                    "graph_support_score",
                    0.0,
                )
            )
            shared_topic_count = int(
                m3_candidate
                .get("graph_features", {})
                .get(
                    "shared_topic_count",
                    0,
                )
            )
            graph_explanation = str(
                m3_candidate
                .get("graph_explanation", {})
                .get(
                    "explanation_text",
                    "",
                )
            )

        if shared_topic_count > 0:
            candidates_with_graph_support += 1

        if m2_rank != m3_rank:
            changed_ranks += 1

        rank_changes.append({
            "staff_id": staff_id,
            "full_name": (
                (m3_candidate or m2_candidate or {})
                .get("full_name", "")
            ),
            "m2_rank": m2_rank,
            "m3_rank": m3_rank,
            "rank_changed": m2_rank != m3_rank,
            "semantic_score": round(
                semantic_score,
                6,
            ),
            "m3_kg_score": round(
                m3_kg_score,
                6,
            ),
            "graph_support_score": round(
                graph_support_score,
                6,
            ),
            "shared_topic_count": (
                shared_topic_count
            ),
            "graph_explanation": (
                graph_explanation
            ),
        })

    m2_top_staff = (
        str(m2_recommendations[0]["staff_id"])
        if m2_recommendations
        else None
    )
    m3_top_staff = (
        str(m3_recommendations[0]["staff_id"])
        if m3_recommendations
        else None
    )

    return {
        "module_id": m2_result.get("module_id"),
        "module_code": m2_result.get(
            "module_code"
        ),
        "module_title": m2_result.get(
            "module_title"
        ),
        "decision_status_m2": m2_result.get(
            "decision_status"
        ),
        "decision_status_m3": m3_result.get(
            "decision_status"
        ),
        "m2_top_staff_id": m2_top_staff,
        "m3_top_staff_id": m3_top_staff,
        "top_rank_changed": (
            m2_top_staff != m3_top_staff
        ),
        "candidate_count_m2": len(
            m2_recommendations
        ),
        "candidate_count_m3": len(
            m3_recommendations
        ),
        "changed_rank_count": changed_ranks,
        "candidates_with_graph_support": (
            candidates_with_graph_support
        ),
        "rank_changes": rank_changes,
    }


def flatten_case_rows(
    module_comparisons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for module_result in module_comparisons:
        for candidate in module_result.get(
            "rank_changes",
            [],
        ):
            rows.append({
                "module_id": module_result.get(
                    "module_id"
                ),
                "module_code": module_result.get(
                    "module_code"
                ),
                "module_title": module_result.get(
                    "module_title"
                ),
                "m2_top_staff_id": module_result.get(
                    "m2_top_staff_id"
                ),
                "m3_top_staff_id": module_result.get(
                    "m3_top_staff_id"
                ),
                "top_rank_changed": (
                    module_result.get(
                        "top_rank_changed"
                    )
                ),
                "staff_id": candidate.get(
                    "staff_id"
                ),
                "full_name": candidate.get(
                    "full_name"
                ),
                "m2_rank": candidate.get(
                    "m2_rank"
                ),
                "m3_rank": candidate.get(
                    "m3_rank"
                ),
                "rank_changed": candidate.get(
                    "rank_changed"
                ),
                "semantic_score": candidate.get(
                    "semantic_score"
                ),
                "m3_kg_score": candidate.get(
                    "m3_kg_score"
                ),
                "graph_support_score": candidate.get(
                    "graph_support_score"
                ),
                "shared_topic_count": candidate.get(
                    "shared_topic_count"
                ),
                "graph_explanation": candidate.get(
                    "graph_explanation"
                ),
            })

    return rows


def build_summary(
    m2_results: list[dict[str, Any]],
    m3_results: list[dict[str, Any]],
    module_comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    compared_modules = len(module_comparisons)
    top_rank_changed_modules = sum(
        1
        for result in module_comparisons
        if result.get("top_rank_changed")
    )

    total_rank_changes = sum(
        result.get("changed_rank_count", 0)
        for result in module_comparisons
    )

    total_candidates_with_graph_support = sum(
        result.get(
            "candidates_with_graph_support",
            0,
        )
        for result in module_comparisons
    )

    m2_semantic_scores: list[float] = []
    m3_semantic_scores: list[float] = []
    m3_hybrid_scores: list[float] = []
    m3_graph_scores: list[float] = []

    for m2_result in m2_results:
        for recommendation in m2_result.get(
            "recommendations",
            [],
        ):
            m2_semantic_scores.append(
                float(
                    recommendation.get(
                        "semantic_score",
                        0.0,
                    )
                )
            )

    for m3_result in m3_results:
        for recommendation in m3_result.get(
            "recommendations",
            [],
        ):
            m3_semantic_scores.append(
                float(
                    recommendation.get(
                        "semantic_score",
                        0.0,
                    )
                )
            )
            m3_hybrid_scores.append(
                float(
                    recommendation.get(
                        "m3_kg_score",
                        0.0,
                    )
                )
            )
            m3_graph_scores.append(
                float(
                    recommendation
                    .get(
                        "m3_kg_components",
                        {},
                    )
                    .get(
                        "graph_support_score",
                        0.0,
                    )
                )
            )

    return {
        "m2_module_results": len(m2_results),
        "m3_module_results": len(m3_results),
        "compared_modules": compared_modules,
        "top_rank_changed_modules": (
            top_rank_changed_modules
        ),
        "top_rank_changed_percentage": round(
            (
                top_rank_changed_modules
                / compared_modules
            ) * 100,
            2,
        )
        if compared_modules
        else 0.0,
        "total_rank_changes": total_rank_changes,
        "total_candidates_with_shared_topics": (
            total_candidates_with_graph_support
        ),
        "average_m2_semantic_score": safe_mean(
            m2_semantic_scores
        ),
        "average_m3_semantic_score": safe_mean(
            m3_semantic_scores
        ),
        "average_m3_kg_score": safe_mean(
            m3_hybrid_scores
        ),
        "average_m3_graph_support_score": safe_mean(
            m3_graph_scores
        ),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    m2_results_path = (
        project_root
        / "data"
        / "outputs"
        / "recommendations"
        / "m2_hierarchy_primary_filtered"
        / "m2_hierarchy_primary_taught_modules.jsonl"
    )

    m3_results_path = (
        project_root
        / "data"
        / "outputs"
        / "recommendations"
        / "m3_kg"
        / "m3_kg_recommendations.jsonl"
    )

    output_dir = (
        project_root
        / "data"
        / "outputs"
        / "evaluation"
        / "m2_vs_m3"
    )

    summary_output_path = (
        output_dir
        / "m2_h_vs_m3_kg_summary.json"
    )

    case_rows_output_path = (
        output_dir
        / "m2_h_vs_m3_kg_case_rows.csv"
    )

    module_comparisons_output_path = (
        output_dir
        / "m2_h_vs_m3_kg_module_comparisons.json"
    )

    m2_results = load_jsonl(m2_results_path)
    m3_results = load_jsonl(m3_results_path)

    m2_by_module_id = {
        str(record["module_id"]): record
        for record in m2_results
        if record.get("module_id")
    }

    m3_by_module_id = {
        str(record["module_id"]): record
        for record in m3_results
        if record.get("module_id")
    }

    shared_module_ids = sorted(
        set(m2_by_module_id).intersection(
            set(m3_by_module_id)
        )
    )

    module_comparisons = [
        compare_module_results(
            m2_result=m2_by_module_id[module_id],
            m3_result=m3_by_module_id[module_id],
        )
        for module_id in shared_module_ids
    ]

    summary = build_summary(
        m2_results=m2_results,
        m3_results=m3_results,
        module_comparisons=module_comparisons,
    )

    summary.update({
        "m2_results_path": str(
            m2_results_path
        ),
        "m3_results_path": str(
            m3_results_path
        ),
        "module_comparisons_output_path": str(
            module_comparisons_output_path
        ),
        "case_rows_output_path": str(
            case_rows_output_path
        ),
    })

    case_rows = flatten_case_rows(
        module_comparisons
    )

    write_json(
        summary_output_path,
        summary,
    )
    write_json(
        module_comparisons_output_path,
        module_comparisons,
    )
    write_csv(
        case_rows_output_path,
        case_rows,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()