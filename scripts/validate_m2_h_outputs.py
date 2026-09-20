from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


TOP_K = 5


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def expected_ranks(count: int) -> list[int]:
    return list(range(1, count + 1))


def validate_result_record(
    result: dict[str, Any],
) -> dict[str, Any]:
    module_id = result.get("module_id")
    module_code = result.get("module_code")
    module_title = result.get("module_title")
    decision_status = result.get("decision_status")
    review_required = bool(
        result.get("review_required")
    )

    module_paths = result.get(
        "module_hierarchy_paths",
        [],
    )
    recommendations = result.get(
        "recommendations",
        [],
    )
    rejected_candidates = result.get(
        "rejected_candidates",
        [],
    )

    issues: list[str] = []
    warnings: list[str] = []

    if not module_paths and decision_status not in {
        "module_hierarchy_unresolved",
        "module_missing_semantic_embedding",
    }:
        warnings.append(
            "module_hierarchy_paths_missing"
        )

    if len(recommendations) > TOP_K:
        issues.append(
            f"recommendation_count_exceeds_top_k:"
            f"{len(recommendations)}"
        )

    actual_ranks = [
        recommendation.get("rank")
        for recommendation in recommendations
    ]

    if actual_ranks != expected_ranks(
        len(recommendations)
    ):
        issues.append(
            f"invalid_rank_sequence:{actual_ranks}"
        )

    staff_ids = [
        recommendation.get("staff_id")
        for recommendation in recommendations
    ]

    if len(staff_ids) != len(set(staff_ids)):
        issues.append(
            "duplicate_staff_in_recommendations"
        )

    gate_reason_counts = Counter()

    for recommendation in recommendations:
        matched_paths = recommendation.get(
            "matched_internal_paths",
            [],
        )
        staff_paths = recommendation.get(
            "staff_hierarchy_paths",
            [],
        )
        gate = recommendation.get("gate", {})
        rank = recommendation.get("rank")
        staff_id = recommendation.get("staff_id")

        if not matched_paths:
            issues.append(
                f"missing_matched_internal_paths:"
                f"rank={rank}:staff_id={staff_id}"
            )

        if not staff_paths:
            warnings.append(
                f"missing_staff_hierarchy_paths:"
                f"rank={rank}:staff_id={staff_id}"
            )

        if not gate.get("passed", False):
            issues.append(
                f"returned_recommendation_failed_gate:"
                f"rank={rank}:staff_id={staff_id}"
            )

        gate_reason = gate.get("reason")
        if gate_reason:
            gate_reason_counts[gate_reason] += 1

    rejected_gate_reason_counts = Counter()

    for rejected in rejected_candidates:
        gate = rejected.get("gate", {})
        gate_reason = gate.get("reason")

        if gate.get("passed", False):
            issues.append(
                "rejected_candidate_marked_passed"
            )

        if gate_reason:
            rejected_gate_reason_counts[
                gate_reason
            ] += 1

    gate_summary = result.get("gate_summary", {})
    reported_passed = gate_summary.get(
        "passed_count",
        0,
    )
    reported_failed = gate_summary.get(
        "failed_count",
        0,
    )
    reported_reason_counts = Counter(
        gate_summary.get(
            "gate_reason_counts",
            {},
        )
    )

    recomputed_reason_counts = (
        gate_reason_counts
        + rejected_gate_reason_counts
    )

    if reported_passed != len(recommendations):
        issues.append(
            f"gate_summary_passed_mismatch:"
            f"reported={reported_passed}:"
            f"actual={len(recommendations)}"
        )

    if reported_failed != len(rejected_candidates):
        issues.append(
            f"gate_summary_failed_mismatch:"
            f"reported={reported_failed}:"
            f"actual={len(rejected_candidates)}"
        )

    if dict(reported_reason_counts) != dict(
        recomputed_reason_counts
    ):
        issues.append(
            "gate_reason_counts_mismatch"
        )

    if decision_status == "hierarchy_internal_ranked":
        if len(recommendations) != TOP_K:
            issues.append(
                "hierarchy_internal_ranked_without_top_k"
            )
        if review_required:
            issues.append(
                "hierarchy_internal_ranked_marked_review_required"
            )

    if decision_status == "limited_hierarchy_internal_pool":
        if len(recommendations) >= TOP_K:
            issues.append(
                "limited_pool_has_top_k_or_more"
            )

    if decision_status == "all_internal_candidates_failed_gate":
        if len(recommendations) != 0:
            issues.append(
                "gate_failed_module_returned_recommendations"
            )
        if not review_required:
            issues.append(
                "gate_failed_module_not_marked_review_required"
            )

    if decision_status == "no_hierarchy_internal_staff":
        if len(recommendations) != 0:
            issues.append(
                "no_hierarchy_internal_staff_has_recommendations"
            )
        if not review_required:
            issues.append(
                "no_hierarchy_internal_staff_not_marked_review_required"
            )

    return {
        "module_id": module_id,
        "module_code": module_code,
        "module_title": module_title,
        "decision_status": decision_status,
        "review_required": review_required,
        "recommendation_count": len(
            recommendations
        ),
        "rejected_candidate_count": len(
            rejected_candidates
        ),
        "issues": issues,
        "warnings": warnings,
    }


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
            "passed_recommendations": (
                passed_recommendations
            ),
            "failed_candidates": (
                failed_candidates
            ),
            "reason_counts": dict(
                gate_reason_counts
            ),
        },
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    filtered_results_path = (
        project_root
        / "data"
        / "outputs"
        / "recommendations"
        / "m2_hierarchy_primary_filtered"
        / "m2_hierarchy_primary_taught_modules.jsonl"
    )

    filtered_summary_path = (
        project_root
        / "data"
        / "outputs"
        / "recommendations"
        / "m2_hierarchy_primary_filtered"
        / "m2_hierarchy_primary_taught_modules_summary.json"
    )

    rejection_path = (
        project_root
        / "data"
        / "outputs"
        / "recommendations"
        / "m2_hierarchy_primary_filtered"
        / "m2_hierarchy_primary_taught_modules_gate_rejections.jsonl"
    )

    output_dir = (
        project_root
        / "data"
        / "outputs"
        / "validation"
        / "m2_hierarchy_primary_filtered"
    )

    validation_report_path = (
        output_dir
        / "structural_validation_report.json"
    )

    issue_records_path = (
        output_dir
        / "structural_validation_issues.json"
    )

    results = load_jsonl(filtered_results_path)
    stored_summary = load_json(filtered_summary_path)
    rejection_rows = load_jsonl(rejection_path)

    record_validations = [
        validate_result_record(result)
        for result in results
    ]

    records_with_issues = [
        record
        for record in record_validations
        if record["issues"]
    ]

    records_with_warnings = [
        record
        for record in record_validations
        if record["warnings"]
    ]

    recomputed_summary = summarise_results(
        results
    )

    summary_alignment_checks = {
        "module_results_match": (
            stored_summary.get("module_results")
            == recomputed_summary["module_results"]
        ),
        "status_counts_match": (
            stored_summary.get("status_counts")
            == recomputed_summary["status_counts"]
        ),
        "fully_ranked_modules_match": (
            stored_summary.get(
                "fully_ranked_modules"
            )
            == recomputed_summary[
                "fully_ranked_modules"
            ]
        ),
        "limited_pool_modules_match": (
            stored_summary.get(
                "limited_pool_modules"
            )
            == recomputed_summary[
                "limited_pool_modules"
            ]
        ),
        "gate_failed_modules_match": (
            stored_summary.get(
                "gate_failed_modules"
            )
            == recomputed_summary[
                "gate_failed_modules"
            ]
        ),
        "review_required_modules_match": (
            stored_summary.get(
                "review_required_modules"
            )
            == recomputed_summary[
                "review_required_modules"
            ]
        ),
        "total_recommendations_match": (
            stored_summary.get(
                "total_recommendations"
            )
            == recomputed_summary[
                "total_recommendations"
            ]
        ),
        "gate_passed_recommendations_match": (
            stored_summary.get("gate", {}).get(
                "passed_recommendations"
            )
            == recomputed_summary["gate"][
                "passed_recommendations"
            ]
        ),
        "gate_failed_candidates_match": (
            stored_summary.get("gate", {}).get(
                "failed_candidates"
            )
            == recomputed_summary["gate"][
                "failed_candidates"
            ]
        ),
        "gate_reason_counts_match": (
            stored_summary.get("gate", {}).get(
                "reason_counts"
            )
            == recomputed_summary["gate"][
                "reason_counts"
            ]
        ),
    }

    rejection_alignment_checks = {
        "rejection_file_count_matches_summary": (
            len(rejection_rows)
            == stored_summary.get("gate", {}).get(
                "failed_candidates"
            )
        ),
        "rejection_file_count": len(
            rejection_rows
        ),
        "summary_failed_candidates": (
            stored_summary.get("gate", {}).get(
                "failed_candidates"
            )
        ),
    }

    issue_counter = Counter()
    warning_counter = Counter()

    for record in record_validations:
        for issue in record["issues"]:
            issue_counter[issue.split(":")[0]] += 1

        for warning in record["warnings"]:
            warning_counter[
                warning.split(":")[0]
            ] += 1

    report = {
        "input_paths": {
            "filtered_results_path": str(
                filtered_results_path
            ),
            "filtered_summary_path": str(
                filtered_summary_path
            ),
            "rejection_path": str(
                rejection_path
            ),
        },
        "top_k": TOP_K,
        "record_count": len(results),
        "records_with_issues": len(
            records_with_issues
        ),
        "records_with_warnings": len(
            records_with_warnings
        ),
        "issue_type_counts": dict(
            issue_counter
        ),
        "warning_type_counts": dict(
            warning_counter
        ),
        "summary_alignment_checks": (
            summary_alignment_checks
        ),
        "rejection_alignment_checks": (
            rejection_alignment_checks
        ),
        "overall_pass": (
            len(records_with_issues) == 0
            and all(
                summary_alignment_checks.values()
            )
            and rejection_alignment_checks[
                "rejection_file_count_matches_summary"
            ]
        ),
    }

    write_json(
        validation_report_path,
        report,
    )
    write_json(
        issue_records_path,
        {
            "records_with_issues": (
                records_with_issues
            ),
            "records_with_warnings": (
                records_with_warnings
            ),
        },
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()