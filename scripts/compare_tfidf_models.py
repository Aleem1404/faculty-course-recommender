from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def find_module(
    records: list[dict[str, Any]],
    module_code: str,
) -> dict[str, Any]:
    matches = [
        record
        for record in records
        if str(record.get("module_code", "")).upper()
        == module_code.upper()
    ]

    if not matches:
        raise ValueError(
            f"No recommendation found for {module_code}"
        )

    return matches[0]


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    m0_path = (
        project_root
        / "data"
        / "outputs"
        / "recommendations"
        / "tfidf"
        / "tfidf_module_recommendations.jsonl"
    )

    m1_path = (
        project_root
        / "data"
        / "outputs"
        / "recommendations"
        / "enriched_tfidf"
        / "enriched_tfidf_recommendations.jsonl"
    )

    output_directory = (
        project_root
        / "data"
        / "outputs"
        / "comparisons"
    )

    module_code = "CS5707"

    m0_module = find_module(
        load_jsonl(m0_path),
        module_code,
    )

    m1_module = find_module(
        load_jsonl(m1_path),
        module_code,
    )

    m0_recommendations = {
        recommendation["staff_id"]: recommendation
        for recommendation in m0_module["recommendations"]
    }

    m1_recommendations = {
        recommendation["staff_id"]: recommendation
        for recommendation in m1_module["recommendations"]
    }

    all_staff_ids = (
        set(m0_recommendations)
        | set(m1_recommendations)
    )

    rows = []

    for staff_id in all_staff_ids:
        m0 = m0_recommendations.get(staff_id)
        m1 = m1_recommendations.get(staff_id)

        source = m1 or m0 or {}

        rows.append({
            "staff_id": staff_id,
            "full_name": source.get("full_name"),
            "department_name": source.get(
                "department_name"
            ),
            "m0_rank": (
                m0.get("rank")
                if m0
                else None
            ),
            "m0_score": (
                m0.get("tfidf_score")
                if m0
                else None
            ),
            "m1_rank": (
                m1.get("rank")
                if m1
                else None
            ),
            "m1_score": (
                m1.get("enriched_tfidf_score")
                if m1
                else None
            ),
            "rank_change": (
                m0["rank"] - m1["rank"]
                if m0 and m1
                else None
            ),
            "m1_field_scores": (
                m1.get("field_scores", {})
                if m1
                else {}
            ),
            "m1_field_contributions": (
                m1.get(
                    "field_contributions",
                    {},
                )
                if m1
                else {}
            ),
        })

    rows.sort(
        key=lambda record: (
            record["m1_rank"] is None,
            record["m1_rank"]
            if record["m1_rank"] is not None
            else 999,
            record["m0_rank"]
            if record["m0_rank"] is not None
            else 999,
        )
    )

    comparison = {
        "module_id": m1_module.get("module_id"),
        "module_code": module_code,
        "module_title": m1_module.get(
            "module_title"
        ),
        "m0_top_five": [
            {
                "rank": recommendation["rank"],
                "staff_id": recommendation["staff_id"],
                "full_name": recommendation[
                    "full_name"
                ],
                "department_name": recommendation[
                    "department_name"
                ],
                "score": recommendation[
                    "tfidf_score"
                ],
            }
            for recommendation in m0_module[
                "recommendations"
            ]
        ],
        "m1_top_five": [
            {
                "rank": recommendation["rank"],
                "staff_id": recommendation["staff_id"],
                "full_name": recommendation[
                    "full_name"
                ],
                "department_name": recommendation[
                    "department_name"
                ],
                "score": recommendation[
                    "enriched_tfidf_score"
                ],
                "field_scores": recommendation[
                    "field_scores"
                ],
                "field_contributions": recommendation[
                    "field_contributions"
                ],
            }
            for recommendation in m1_module[
                "recommendations"
            ]
        ],
        "staff_comparison": rows,
        "top_five_overlap": len(
            set(m0_recommendations)
            & set(m1_recommendations)
        ),
        "entered_m1_top_five": sorted(
            set(m1_recommendations)
            - set(m0_recommendations)
        ),
        "left_m1_top_five": sorted(
            set(m0_recommendations)
            - set(m1_recommendations)
        ),
    }

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "cs5707_m0_m1_comparison.json"
    )

    output_path.write_text(
        json.dumps(
            comparison,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()