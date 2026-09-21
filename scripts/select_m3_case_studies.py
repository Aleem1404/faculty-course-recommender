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


def top_candidate(
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


def select_case_study_rows(
    module_comparisons: list[dict[str, Any]],
    m3_results: list[dict[str, Any]],
    max_changed_top: int = 15,
    max_strong_graph: int = 15,
    max_no_change_but_explained: int = 10,
) -> list[dict[str, Any]]:
    m3_by_module_id = {
        str(record["module_id"]): record
        for record in m3_results
        if record.get("module_id")
    }

    changed_top_cases: list[dict[str, Any]] = []
    strong_graph_cases: list[dict[str, Any]] = []
    stable_but_explained_cases: list[dict[str, Any]] = []

    for comparison in module_comparisons:
        module_id = str(
            comparison.get("module_id")
        )
        m3_result = m3_by_module_id.get(module_id)

        if m3_result is None:
            continue

        recommendations = m3_result.get(
            "recommendations",
            [],
        )
        top_m3 = top_candidate(
            recommendations
        )

        if top_m3 is None:
            continue

        graph_features = top_m3.get(
            "graph_features",
            {},
        )
        shared_topic_count = int(
            graph_features.get(
                "shared_topic_count",
                0,
            )
        )
        strongest_shared_topics = (
            graph_features.get(
                "strongest_shared_topics",
                [],
            )
        )
        graph_explanation = (
            top_m3.get(
                "graph_explanation",
                {},
            )
        )
        graph_support_score = float(
            top_m3.get(
                "m3_kg_components",
                {},
            ).get(
                "graph_support_score",
                0.0,
            )
        )

        row = {
            "module_id": comparison.get(
                "module_id"
            ),
            "module_code": comparison.get(
                "module_code"
            ),
            "module_title": comparison.get(
                "module_title"
            ),
            "top_rank_changed": comparison.get(
                "top_rank_changed",
                False,
            ),
            "m2_top_staff_id": comparison.get(
                "m2_top_staff_id"
            ),
            "m3_top_staff_id": comparison.get(
                "m3_top_staff_id"
            ),
            "m3_top_staff_name": top_m3.get(
                "full_name",
                "",
            ),
            "m3_top_staff_position": top_m3.get(
                "position",
                "",
            ),
            "m3_top_staff_department": top_m3.get(
                "department_name",
                "",
            ),
            "semantic_score": round(
                float(
                    top_m3.get(
                        "semantic_score",
                        0.0,
                    )
                ),
                6,
            ),
            "m3_kg_score": round(
                float(
                    top_m3.get(
                        "m3_kg_score",
                        0.0,
                    )
                ),
                6,
            ),
            "graph_support_score": round(
                graph_support_score,
                6,
            ),
            "shared_topic_count": (
                shared_topic_count
            ),
            "top_shared_topics": ", ".join(
                item.get("topic", "")
                for item in strongest_shared_topics[:3]
                if item.get("topic")
            ),
            "same_department": bool(
                graph_features.get(
                    "same_department",
                    False,
                )
            ),
            "same_college": bool(
                graph_features.get(
                    "same_college",
                    False,
                )
            ),
            "graph_explanation": graph_explanation.get(
                "explanation_text",
                "",
            ),
        }

        if comparison.get(
            "top_rank_changed",
            False
        ):
            changed_top_cases.append(row)

        if (
            shared_topic_count >= 2
            or graph_support_score >= 0.7
        ):
            strong_graph_cases.append(row)

        if (
            not comparison.get(
                "top_rank_changed",
                False
            )
            and shared_topic_count >= 1
        ):
            stable_but_explained_cases.append(
                row
            )

    changed_top_cases.sort(
        key=lambda item: (
            -item["shared_topic_count"],
            -item["graph_support_score"],
            -item["m3_kg_score"],
            item["module_title"],
        )
    )

    strong_graph_cases.sort(
        key=lambda item: (
            -item["shared_topic_count"],
            -item["graph_support_score"],
            -item["m3_kg_score"],
            item["module_title"],
        )
    )

    stable_but_explained_cases.sort(
        key=lambda item: (
            -item["shared_topic_count"],
            -item["graph_support_score"],
            -item["m3_kg_score"],
            item["module_title"],
        )
    )

    selected: list[dict[str, Any]] = []
    seen_module_ids: set[str] = set()

    def add_cases(
        source_rows: list[dict[str, Any]],
        label: str,
        max_items: int,
    ) -> None:
        count = 0

        for row in source_rows:
            module_id = str(
                row["module_id"]
            )

            if module_id in seen_module_ids:
                continue

            enriched = dict(row)
            enriched["case_type"] = label

            selected.append(enriched)
            seen_module_ids.add(module_id)
            count += 1

            if count >= max_items:
                break

    add_cases(
        changed_top_cases,
        label="changed_top_rank",
        max_items=max_changed_top,
    )
    add_cases(
        strong_graph_cases,
        label="strong_graph_evidence",
        max_items=max_strong_graph,
    )
    add_cases(
        stable_but_explained_cases,
        label="stable_but_explained",
        max_items=max_no_change_but_explained,
    )

    selected.sort(
        key=lambda item: (
            item["case_type"],
            -item["shared_topic_count"],
            -item["graph_support_score"],
            item["module_title"],
        )
    )

    return selected


def build_summary(
    selected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    case_type_counts: dict[str, int] = {}

    for row in selected_rows:
        case_type = str(
            row.get("case_type", "unknown")
        )
        case_type_counts[case_type] = (
            case_type_counts.get(
                case_type,
                0,
            )
            + 1
        )

    return {
        "selected_case_count": len(
            selected_rows
        ),
        "selected_case_studies": len(
            selected_rows
        ),
        "case_type_counts": (
            case_type_counts
        ),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    module_comparisons_path = (
        project_root
        / "data"
        / "outputs"
        / "evaluation"
        / "m2_vs_m3"
        / "m2_h_vs_m3_kg_module_comparisons.json"
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
        / "case_studies"
    )

    json_output_path = (
        output_dir
        / "m3_case_studies.json"
    )

    csv_output_path = (
        output_dir
        / "m3_case_studies.csv"
    )

    summary_output_path = (
        output_dir
        / "m3_case_studies_summary.json"
    )

    module_comparisons = load_json(
        module_comparisons_path
    )

    m3_results: list[dict[str, Any]] = []
    with m3_results_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            m3_results.append(
                json.loads(line)
            )

    selected_rows = select_case_study_rows(
        module_comparisons=module_comparisons,
        m3_results=m3_results,
        max_changed_top=15,
        max_strong_graph=15,
        max_no_change_but_explained=10,
    )

    summary = build_summary(
        selected_rows
    )
    summary.update({
        "module_comparisons_path": str(
            module_comparisons_path
        ),
        "m3_results_path": str(
            m3_results_path
        ),
        "json_output_path": str(
            json_output_path
        ),
        "csv_output_path": str(
            csv_output_path
        ),
    })

    write_json(
        json_output_path,
        selected_rows,
    )
    write_csv(
        csv_output_path,
        selected_rows,
    )
    write_json(
        summary_output_path,
        summary,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()