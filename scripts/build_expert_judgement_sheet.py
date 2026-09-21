from __future__ import annotations

import csv
import json
from pathlib import Path
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


def recommendation_summary(
    recommendation: dict[str, Any] | None,
    system_name: str,
) -> dict[str, Any]:
    if recommendation is None:
        return {
            f"{system_name}_staff_id": "",
            f"{system_name}_full_name": "",
            f"{system_name}_position": "",
            f"{system_name}_department_name": "",
            f"{system_name}_college_name": "",
            f"{system_name}_semantic_score": "",
            f"{system_name}_final_score": "",
            f"{system_name}_shared_topic_count": "",
            f"{system_name}_top_shared_topics": "",
            f"{system_name}_graph_support_score": "",
            f"{system_name}_explanation_text": "",
        }

    graph_features = recommendation.get(
        "graph_features",
        {},
    )
    graph_components = recommendation.get(
        "m3_kg_components",
        {},
    )
    graph_explanation = recommendation.get(
        "graph_explanation",
        {},
    )

    strongest_shared_topics = (
        graph_features.get(
            "strongest_shared_topics",
            [],
        )
    )

    top_shared_topics = ", ".join(
        item.get("topic", "")
        for item in strongest_shared_topics[:3]
        if item.get("topic")
    )

    semantic_score = recommendation.get(
        "semantic_score",
        "",
    )

    final_score = recommendation.get(
        "m3_kg_score",
        semantic_score,
    )

    shared_topic_count = graph_features.get(
        "shared_topic_count",
        "",
    )

    graph_support_score = graph_components.get(
        "graph_support_score",
        "",
    )

    explanation_text = graph_explanation.get(
        "explanation_text",
        "",
    )

    return {
        f"{system_name}_staff_id": recommendation.get(
            "staff_id",
            "",
        ),
        f"{system_name}_full_name": recommendation.get(
            "full_name",
            "",
        ),
        f"{system_name}_position": recommendation.get(
            "position",
            "",
        ),
        f"{system_name}_department_name": recommendation.get(
            "department_name",
            "",
        ),
        f"{system_name}_college_name": recommendation.get(
            "college_name",
            "",
        ),
        f"{system_name}_semantic_score": semantic_score,
        f"{system_name}_final_score": final_score,
        f"{system_name}_shared_topic_count": (
            shared_topic_count
        ),
        f"{system_name}_top_shared_topics": (
            top_shared_topics
        ),
        f"{system_name}_graph_support_score": (
            graph_support_score
        ),
        f"{system_name}_explanation_text": (
            explanation_text
        ),
    }


def select_review_subset(
    case_rows: list[dict[str, Any]],
    max_changed_top: int = 10,
    max_strong_graph: int = 5,
    max_stable_explained: int = 5,
) -> list[dict[str, Any]]:
    changed_top = [
        row
        for row in case_rows
        if row.get("case_type")
        == "changed_top_rank"
    ]

    strong_graph = [
        row
        for row in case_rows
        if row.get("case_type")
        == "strong_graph_evidence"
    ]

    stable_explained = [
        row
        for row in case_rows
        if row.get("case_type")
        == "stable_but_explained"
    ]

    changed_top = sorted(
        changed_top,
        key=lambda item: (
            -int(
                item.get(
                    "shared_topic_count",
                    0,
                )
            ),
            -float(
                item.get(
                    "graph_support_score",
                    0.0,
                )
            ),
            item.get("module_title", ""),
        ),
    )[:max_changed_top]

    strong_graph = sorted(
        strong_graph,
        key=lambda item: (
            -int(
                item.get(
                    "shared_topic_count",
                    0,
                )
            ),
            -float(
                item.get(
                    "graph_support_score",
                    0.0,
                )
            ),
            item.get("module_title", ""),
        ),
    )[:max_strong_graph]

    stable_explained = sorted(
        stable_explained,
        key=lambda item: (
            -int(
                item.get(
                    "shared_topic_count",
                    0,
                )
            ),
            -float(
                item.get(
                    "graph_support_score",
                    0.0,
                )
            ),
            item.get("module_title", ""),
        ),
    )[:max_stable_explained]

    selected = (
        changed_top
        + strong_graph
        + stable_explained
    )

    selected = sorted(
        selected,
        key=lambda item: (
            item.get("case_type", ""),
            item.get("module_title", ""),
        ),
    )

    return selected


def build_sheet_rows(
    selected_cases: list[dict[str, Any]],
    m2_results: list[dict[str, Any]],
    m3_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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

    rows: list[dict[str, Any]] = []

    for case in selected_cases:
        module_id = str(case["module_id"])

        m2_result = m2_by_module_id.get(module_id, {})
        m3_result = m3_by_module_id.get(module_id, {})

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

        row = {
            "review_case_id": (
                f"CASE-{len(rows) + 1:02d}"
            ),
            "case_type": case.get(
                "case_type",
                "",
            ),
            "module_id": case.get(
                "module_id",
                "",
            ),
            "module_code": case.get(
                "module_code",
                "",
            ),
            "module_title": case.get(
                "module_title",
                "",
            ),
            "top_rank_changed": case.get(
                "top_rank_changed",
                "",
            ),
            "module_departments": ", ".join(
                m2_result.get(
                    "module_departments",
                    [],
                )
            ),
            "module_colleges": ", ".join(
                m2_result.get(
                    "module_colleges",
                    [],
                )
            ),
            "review_instruction": (
                "Please compare the M2-H and "
                "M3-KG top recommendations for "
                "primary teaching suitability. "
                "Rate relevance and explanation "
                "quality. Focus on subject fit "
                "within the module hierarchy."
            ),
        }

        row.update(
            recommendation_summary(
                m2_top,
                "m2_h",
            )
        )
        row.update(
            recommendation_summary(
                m3_top,
                "m3_kg",
            )
        )

        row.update({
            "m2_h_relevance_rating_1_to_5": "",
            "m3_kg_relevance_rating_1_to_5": "",
            "m3_kg_explanation_quality_1_to_5": "",
            "preferred_system": "",
            "preferred_system_options": (
                "M2-H, M3-KG, Tie, Unsure"
            ),
            "reviewer_comments": "",
        })

        rows.append(row)

    return rows


def build_summary(
    selected_cases: list[dict[str, Any]],
    sheet_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    case_type_counts: dict[str, int] = {}

    for case in selected_cases:
        case_type = str(
            case.get("case_type", "unknown")
        )
        case_type_counts[case_type] = (
            case_type_counts.get(
                case_type,
                0,
            )
            + 1
        )

    return {
        "selected_review_cases": len(
            selected_cases
        ),
        "sheet_rows": len(sheet_rows),
        "case_type_counts": (
            case_type_counts
        ),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    case_studies_path = (
        project_root
        / "data"
        / "outputs"
        / "evaluation"
        / "case_studies"
        / "m3_case_studies.json"
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
        / "expert_judgement"
    )

    sheet_output_path = (
        output_dir
        / "expert_judgement_sheet.csv"
    )

    selected_cases_output_path = (
        output_dir
        / "expert_judgement_selected_cases.json"
    )

    summary_output_path = (
        output_dir
        / "expert_judgement_summary.json"
    )

    case_rows = load_json(
        case_studies_path
    )
    m2_results = load_jsonl(
        m2_results_path
    )
    m3_results = load_jsonl(
        m3_results_path
    )

    selected_cases = select_review_subset(
        case_rows=case_rows,
        max_changed_top=10,
        max_strong_graph=5,
        max_stable_explained=5,
    )

    sheet_rows = build_sheet_rows(
        selected_cases=selected_cases,
        m2_results=m2_results,
        m3_results=m3_results,
    )

    summary = build_summary(
        selected_cases=selected_cases,
        sheet_rows=sheet_rows,
    )
    summary.update({
        "case_studies_path": str(
            case_studies_path
        ),
        "m2_results_path": str(
            m2_results_path
        ),
        "m3_results_path": str(
            m3_results_path
        ),
        "sheet_output_path": str(
            sheet_output_path
        ),
        "selected_cases_output_path": str(
            selected_cases_output_path
        ),
    })

    write_csv(
        sheet_output_path,
        sheet_rows,
    )
    write_json(
        selected_cases_output_path,
        selected_cases,
    )
    write_json(
        summary_output_path,
        summary,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()