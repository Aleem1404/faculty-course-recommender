from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

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
        path.write_text(
            "",
            encoding="utf-8",
        )
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


def safe_mean(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return round(
        float(mean(values)),
        6,
    )


def top_recommendation(
    recommendations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not recommendations:
        return None

    ordered = sorted(
        recommendations,
        key=lambda item: (
            item.get("rank", 999999),
            str(item.get("staff_id", "")),
        ),
    )

    return ordered[0]


def build_chart_data(
    comparison_summary: dict[str, Any],
    case_study_summary: dict[str, Any],
    expert_summary: dict[str, Any],
    m2_results: list[dict[str, Any]],
    m3_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compared_modules = int(
        comparison_summary.get(
            "compared_modules",
            0,
        )
    )
    top_rank_changed = int(
        comparison_summary.get(
            "top_rank_changed_modules",
            0,
        )
    )
    top_rank_unchanged = max(
        0,
        compared_modules - top_rank_changed,
    )

    shared_topic_recommendations = int(
        comparison_summary.get(
            "total_candidates_with_shared_topics",
            0,
        )
    )

    total_m3_recommendations = sum(
        len(
            result.get(
                "recommendations",
                [],
            )
        )
        for result in m3_results
    )
    no_shared_topic_recommendations = max(
        0,
        total_m3_recommendations
        - shared_topic_recommendations,
    )

    case_counts = expert_summary.get(
        "case_type_counts",
        {},
    )

    m2_top_scores: list[float] = []
    m3_top_scores: list[float] = []
    m3_top_graph_scores: list[float] = []

    m3_by_module_id = {
        str(record["module_id"]): record
        for record in m3_results
        if record.get("module_id")
    }

    for m2_result in m2_results:
        module_id = str(
            m2_result.get("module_id", "")
        )
        m3_result = m3_by_module_id.get(
            module_id
        )

        if m3_result is None:
            continue

        m2_top = top_recommendation(
            m2_result.get(
                "recommendations",
                [],
            )
        )
        m3_top = top_recommendation(
            m3_result.get(
                "recommendations",
                [],
            )
        )

        if m2_top is not None:
            m2_top_scores.append(
                float(
                    m2_top.get(
                        "semantic_score",
                        0.0,
                    )
                )
            )

        if m3_top is not None:
            m3_top_scores.append(
                float(
                    m3_top.get(
                        "m3_kg_score",
                        0.0,
                    )
                )
            )
            m3_top_graph_scores.append(
                float(
                    m3_top.get(
                        "m3_kg_components",
                        {},
                    ).get(
                        "graph_support_score",
                        0.0,
                    )
                )
            )

    rows = [
        {
            "chart_group": "top_rank_change",
            "label": "Changed",
            "value": top_rank_changed,
        },
        {
            "chart_group": "top_rank_change",
            "label": "Unchanged",
            "value": top_rank_unchanged,
        },
        {
            "chart_group": "shared_topics",
            "label": "Shared topics",
            "value": shared_topic_recommendations,
        },
        {
            "chart_group": "shared_topics",
            "label": "No shared topics",
            "value": no_shared_topic_recommendations,
        },
        {
            "chart_group": "case_types",
            "label": "Changed top",
            "value": int(
                case_counts.get(
                    "changed_top_rank",
                    0,
                )
            ),
        },
        {
            "chart_group": "case_types",
            "label": "Strong graph",
            "value": int(
                case_counts.get(
                    "strong_graph_evidence",
                    0,
                )
            ),
        },
        {
            "chart_group": "case_types",
            "label": "Stable explained",
            "value": int(
                case_counts.get(
                    "stable_but_explained",
                    0,
                )
            ),
        },
        {
            "chart_group": "score_means",
            "label": "M2-H top",
            "value": safe_mean(
                m2_top_scores
            ),
        },
        {
            "chart_group": "score_means",
            "label": "M3-KG top",
            "value": safe_mean(
                m3_top_scores
            ),
        },
        {
            "chart_group": "score_means",
            "label": "Graph support",
            "value": safe_mean(
                m3_top_graph_scores
            ),
        },
    ]

    return rows


def build_summary(
    chart_data_rows: list[dict[str, Any]],
    comparison_summary: dict[str, Any],
    expert_summary: dict[str, Any],
) -> dict[str, Any]:
    chart_groups = sorted({
        row["chart_group"]
        for row in chart_data_rows
    })

    return {
        "chart_data_rows": len(
            chart_data_rows
        ),
        "chart_groups": chart_groups,
        "compared_modules": comparison_summary.get(
            "compared_modules",
            0,
        ),
        "top_rank_changed_modules": comparison_summary.get(
            "top_rank_changed_modules",
            0,
        ),
        "selected_review_cases": expert_summary.get(
            "selected_review_cases",
            0,
        ),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    comparison_summary_path = (
        project_root
        / "data"
        / "outputs"
        / "evaluation"
        / "m2_vs_m3"
        / "m2_h_vs_m3_kg_summary.json"
    )

    case_study_summary_path = (
        project_root
        / "data"
        / "outputs"
        / "evaluation"
        / "case_studies"
        / "m3_case_studies_summary.json"
    )

    expert_summary_path = (
        project_root
        / "data"
        / "outputs"
        / "evaluation"
        / "expert_judgement"
        / "expert_judgement_summary.json"
    )

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
        / "charts"
    )

    chart_data_path = (
        output_dir
        / "evaluation_chart_data.csv"
    )

    summary_output_path = (
        output_dir
        / "evaluation_chart_summary.json"
    )

    comparison_summary = load_json(
        comparison_summary_path
    )
    case_study_summary = load_json(
        case_study_summary_path
    )
    expert_summary = load_json(
        expert_summary_path
    )
    m2_results = load_jsonl(
        m2_results_path
    )
    m3_results = load_jsonl(
        m3_results_path
    )

    chart_data_rows = build_chart_data(
        comparison_summary=comparison_summary,
        case_study_summary=case_study_summary,
        expert_summary=expert_summary,
        m2_results=m2_results,
        m3_results=m3_results,
    )

    summary = build_summary(
        chart_data_rows=chart_data_rows,
        comparison_summary=comparison_summary,
        expert_summary=expert_summary,
    )
    summary.update({
        "comparison_summary_path": str(
            comparison_summary_path
        ),
        "case_study_summary_path": str(
            case_study_summary_path
        ),
        "expert_summary_path": str(
            expert_summary_path
        ),
        "chart_data_path": str(
            chart_data_path
        ),
    })

    write_csv(
        chart_data_path,
        chart_data_rows,
    )
    write_json(
        summary_output_path,
        summary,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()