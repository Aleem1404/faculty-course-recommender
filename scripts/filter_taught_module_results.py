from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ADMINISTRATIVE_MODULE_PATTERNS = [
    "major group project",
    "design and branding research project",
    "capstone project",
    "practice placement",
    "dissertation",
    "dissertation project",
    "student selected component",
    "performance",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def normalise_text(value: Any) -> str:
    if value is None:
        return ""

    return " ".join(str(value).casefold().split())


def get_exclusion_reason(
    module_title: str,
) -> str | None:
    title = normalise_text(module_title)

    for pattern in ADMINISTRATIVE_MODULE_PATTERNS:
        if pattern in title:
            return pattern

    return None


def summarise_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts = Counter(
        result.get("decision_status", "unknown")
        for result in results
    )

    gate_reason_counts = Counter()
    passed_recommendations = 0
    failed_candidates = 0

    for result in results:
        gate_summary = result.get("gate_summary", {})

        for reason, count in gate_summary.get(
            "gate_reason_counts",
            {}
        ).items():
            gate_reason_counts[reason] += count

        passed_recommendations += gate_summary.get(
            "passed_count",
            0,
        )
        failed_candidates += gate_summary.get(
            "failed_count",
            0,
        )

    return {
        "module_results": len(results),
        "status_counts": dict(status_counts),
        "fully_ranked_modules": sum(
            result.get("decision_status")
            == "hierarchy_internal_ranked"
            for result in results
        ),
        "limited_pool_modules": sum(
            result.get("decision_status")
            == "limited_hierarchy_internal_pool"
            for result in results
        ),
        "gate_failed_modules": sum(
            result.get("decision_status")
            == "all_internal_candidates_failed_gate"
            for result in results
        ),
        "review_required_modules": sum(
            bool(result.get("review_required"))
            for result in results
        ),
        "total_recommendations": sum(
            len(result.get("recommendations", []))
            for result in results
        ),
        "gate": {
            "passed_recommendations": passed_recommendations,
            "failed_candidates": failed_candidates,
            "reason_counts": dict(gate_reason_counts),
        },
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    input_results_path = (
        project_root
        / "data"
        / "outputs"
        / "recommendations"
        / "m2_hierarchy_primary"
        / "m2_hierarchy_primary_recommendations.jsonl"
    )

    input_rejections_path = (
        project_root
        / "data"
        / "outputs"
        / "recommendations"
        / "m2_hierarchy_primary"
        / "m2_hierarchy_primary_gate_rejections.jsonl"
    )

    output_dir = (
        project_root
        / "data"
        / "outputs"
        / "recommendations"
        / "m2_hierarchy_primary_filtered"
    )

    filtered_jsonl_path = (
        output_dir
        / "m2_hierarchy_primary_taught_modules.jsonl"
    )

    filtered_summary_path = (
        output_dir
        / "m2_hierarchy_primary_taught_modules_summary.json"
    )

    excluded_modules_path = (
        output_dir
        / "excluded_modules.json"
    )

    filtered_rejections_path = (
        output_dir
        / "m2_hierarchy_primary_taught_modules_gate_rejections.jsonl"
    )

    results = load_jsonl(input_results_path)
    rejection_rows = load_jsonl(input_rejections_path)

    kept_results: list[dict[str, Any]] = []
    excluded_modules: list[dict[str, Any]] = []
    kept_module_ids: set[str] = set()

    for result in results:
        module_title = result.get("module_title", "")
        exclusion_reason = get_exclusion_reason(
            module_title
        )

        if exclusion_reason is None:
            kept_results.append(result)
            kept_module_ids.add(
                str(result.get("module_id"))
            )
        else:
            excluded_modules.append({
                "module_id": result.get("module_id"),
                "module_code": result.get("module_code"),
                "module_title": result.get("module_title"),
                "decision_status": result.get("decision_status"),
                "review_required": result.get("review_required"),
                "exclusion_reason": exclusion_reason,
            })

    filtered_rejections = [
        row
        for row in rejection_rows
        if str(row.get("module_id")) in kept_module_ids
    ]

    filtered_summary = summarise_results(
        kept_results
    )

    filtered_summary.update({
        "source_input_path": str(input_results_path),
        "source_rejections_path": str(input_rejections_path),
        "filtered_output_path": str(filtered_jsonl_path),
        "filtered_rejections_path": str(
            filtered_rejections_path
        ),
        "excluded_modules_path": str(excluded_modules_path),
        "excluded_module_count": len(excluded_modules),
        "kept_module_count": len(kept_results),
        "filtered_rejection_count": len(
            filtered_rejections
        ),
        "administrative_module_patterns": (
            ADMINISTRATIVE_MODULE_PATTERNS
        ),
    })

    write_jsonl(filtered_jsonl_path, kept_results)
    write_jsonl(
        filtered_rejections_path,
        filtered_rejections,
    )
    write_json(filtered_summary_path, filtered_summary)
    write_json(excluded_modules_path, excluded_modules)

    print(json.dumps(filtered_summary, indent=2))


if __name__ == "__main__":
    main()