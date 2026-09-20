from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object in {path} at line {line_number}"
                )

            records.append(record)

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


def index_by_staff(
    recommendations: list[dict[str, Any]],
    score_key: str,
) -> dict[str, Any]:
    return {
        record["staff_id"]: {
            "rank": record.get("rank"),
            "score": record.get(score_key),
            "full_name": record.get("full_name"),
            "department_name": record.get("department_name"),
        }
        for record in recommendations
    }


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

    m2_path = (
        project_root
        / "data"
        / "outputs"
        / "recommendations"
        / "semantic"
        / "semantic_module_recommendations.jsonl"
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

    m2_module = find_module(
        load_jsonl(m2_path),
        module_code,
    )

    m0_index = index_by_staff(
        m0_module["recommendations"],
        "tfidf_score",
    )

    m1_index = index_by_staff(
        m1_module["recommendations"],
        "enriched_tfidf_score",
    )

    m2_index = index_by_staff(
        m2_module["recommendations"],
        "semantic_score",
    )

    all_staff_ids = (
        set(m0_index)
        | set(m1_index)
        | set(m2_index)
    )

    rows = []

    for staff_id in all_staff_ids:
        m0 = m0_index.get(staff_id)
        m1 = m1_index.get(staff_id)
        m2 = m2_index.get(staff_id)

        source = m2 or m1 or m0 or {}

        rows.append({
            "staff_id": staff_id,
            "full_name": source.get("full_name"),
            "department_name": source.get(
                "department_name"
            ),
            "m0_rank": m0["rank"] if m0 else None,
            "m0_score": m0["score"] if m0 else None,
            "m1_rank": m1["rank"] if m1 else None,
            "m1_score": m1["score"] if m1 else None,
            "m2_rank": m2["rank"] if m2 else None,
            "m2_score": m2["score"] if m2 else None,
        })

    rows.sort(
        key=lambda record: (
            record["m2_rank"] is None,
            record["m2_rank"]
            if record["m2_rank"] is not None
            else 999,
            record["m1_rank"]
            if record["m1_rank"] is not None
            else 999,
            record["m0_rank"]
            if record["m0_rank"] is not None
            else 999,
        )
    )

    comparison = {
        "module_id": m2_module.get("module_id"),
        "module_code": module_code,
        "module_title": m2_module.get("module_title"),
        "departments": m2_module.get("departments", []),
        "colleges": m2_module.get("colleges", []),
        "m0_top_five": [
            {
                "rank": rec["rank"],
                "staff_id": rec["staff_id"],
                "full_name": rec["full_name"],
                "department_name": rec["department_name"],
                "score": rec["tfidf_score"],
            }
            for rec in m0_module["recommendations"]
        ],
        "m1_top_five": [
            {
                "rank": rec["rank"],
                "staff_id": rec["staff_id"],
                "full_name": rec["full_name"],
                "department_name": rec["department_name"],
                "score": rec["enriched_tfidf_score"],
                "field_scores": rec["field_scores"],
            }
            for rec in m1_module["recommendations"]
        ],
        "m2_top_five": [
            {
                "rank": rec["rank"],
                "staff_id": rec["staff_id"],
                "full_name": rec["full_name"],
                "department_name": rec["department_name"],
                "score": rec["semantic_score"],
                "field_scores": rec["field_scores"],
            }
            for rec in m2_module["recommendations"]
        ],
        "staff_comparison": rows,
        "overlap_m0_m1": len(set(m0_index) & set(m1_index)),
        "overlap_m1_m2": len(set(m1_index) & set(m2_index)),
        "overlap_m0_m2": len(set(m0_index) & set(m2_index)),
        "entered_m2_top_five": sorted(
            set(m2_index) - set(m1_index)
        ),
        "left_m2_top_five": sorted(
            set(m1_index) - set(m2_index)
        ),
    }

    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_directory
        / "cs5707_m0_m1_m2_comparison.json"
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