from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


TARGET_STATUS_COUNTS = {
    "hierarchy_internal_ranked": 15,
    "limited_hierarchy_internal_pool": 5,
    "no_hierarchy_internal_staff": 5,
    "all_internal_candidates_failed_gate": 5,
}

RANDOM_SEED = 42


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def sample_by_status(
    results: list[dict[str, Any]],
    target_status_counts: dict[str, int],
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for result in results:
        status = result.get("decision_status", "unknown")
        grouped[status].append(result)

    rng = random.Random(seed)
    sampled_results: list[dict[str, Any]] = []
    sampling_report: dict[str, Any] = {}

    for status, target_count in target_status_counts.items():
        candidates = list(grouped.get(status, []))
        available_count = len(candidates)

        rng.shuffle(candidates)

        selected = candidates[: min(target_count, available_count)]
        sampled_results.extend(selected)

        sampling_report[status] = {
            "target_count": target_count,
            "available_count": available_count,
            "selected_count": len(selected),
        }

    sampled_results.sort(
        key=lambda record: (
            record.get("decision_status", ""),
            record.get("module_code", ""),
            record.get("module_id", ""),
        )
    )

    return sampled_results, sampling_report


def build_reviewer_rows(
    sampled_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for result in sampled_results:
        module_id = result.get("module_id")
        module_code = result.get("module_code")
        module_title = result.get("module_title")
        module_departments = "; ".join(
            result.get("module_departments", [])
        )
        module_colleges = "; ".join(
            result.get("module_colleges", [])
        )
        decision_status = result.get("decision_status")
        review_required = result.get("review_required")
        module_hierarchy_paths = "; ".join(
            result.get("module_hierarchy_paths", [])
        )

        recommendations = result.get("recommendations", [])

        if recommendations:
            for recommendation in recommendations:
                rows.append({
                    "module_id": module_id,
                    "module_code": module_code,
                    "module_title": module_title,
                    "module_departments": module_departments,
                    "module_colleges": module_colleges,
                    "module_hierarchy_paths": module_hierarchy_paths,
                    "decision_status": decision_status,
                    "review_required": review_required,
                    "rank": recommendation.get("rank"),
                    "staff_id": recommendation.get("staff_id"),
                    "full_name": recommendation.get("full_name"),
                    "position": recommendation.get("position"),
                    "staff_department_name": recommendation.get("department_name"),
                    "staff_college_name": recommendation.get("college_name"),
                    "profile_url": recommendation.get("profile_url"),
                    "semantic_score": recommendation.get("semantic_score"),
                    "top_evidence_field": recommendation.get("top_evidence_field"),
                    "publication_count": recommendation.get("publication_count"),
                    "enriched_publication_count": recommendation.get(
                        "enriched_publication_count"
                    ),
                    "gate_passed": recommendation.get("gate", {}).get("passed"),
                    "gate_reason": recommendation.get("gate", {}).get("reason"),
                    "human_relevance_rating": "",
                    "reviewer_notes": "",
                })
        else:
            rows.append({
                "module_id": module_id,
                "module_code": module_code,
                "module_title": module_title,
                "module_departments": module_departments,
                "module_colleges": module_colleges,
                "module_hierarchy_paths": module_hierarchy_paths,
                "decision_status": decision_status,
                "review_required": review_required,
                "rank": "",
                "staff_id": "",
                "full_name": "",
                "position": "",
                "staff_department_name": "",
                "staff_college_name": "",
                "profile_url": "",
                "semantic_score": "",
                "top_evidence_field": "",
                "publication_count": "",
                "enriched_publication_count": "",
                "gate_passed": "",
                "gate_reason": "",
                "human_relevance_rating": "",
                "reviewer_notes": "No returned recommendation for this sampled module.",
            })

    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    input_path = (
        project_root
        / "data"
        / "outputs"
        / "recommendations"
        / "m2_hierarchy_primary"
        / "m2_hierarchy_primary_recommendations.jsonl"
    )

    output_dir = (
        project_root
        / "data"
        / "outputs"
        / "evaluation"
        / "m2_hierarchy_primary"
    )

    results = load_jsonl(input_path)

    sampled_results, sampling_report = sample_by_status(
        results=results,
        target_status_counts=TARGET_STATUS_COUNTS,
        seed=RANDOM_SEED,
    )

    reviewer_rows = build_reviewer_rows(sampled_results)

    module_sample_path = output_dir / "evaluation_module_sample.json"
    reviewer_csv_path = output_dir / "evaluation_reviewer_template.csv"
    summary_path = output_dir / "evaluation_summary.json"

    write_json(module_sample_path, sampled_results)
    write_csv(reviewer_csv_path, reviewer_rows)

    summary = {
        "input_path": str(input_path),
        "module_sample_path": str(module_sample_path),
        "reviewer_csv_path": str(reviewer_csv_path),
        "random_seed": RANDOM_SEED,
        "target_status_counts": TARGET_STATUS_COUNTS,
        "sampling_report": sampling_report,
        "sampled_module_count": len(sampled_results),
        "reviewer_row_count": len(reviewer_rows),
        "rating_scale": {
            "2": "Relevant",
            "1": "Partly relevant",
            "0": "Not relevant",
        },
    }

    write_json(summary_path, summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()