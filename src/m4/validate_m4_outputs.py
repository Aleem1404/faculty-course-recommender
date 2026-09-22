from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RECOMMENDATIONS_INPUT = Path(
    "data/outputs/m4/collaboration_recommendations_enriched.jsonl"
)

DEFAULT_DISPLAY_INPUT = Path(
    "data/outputs/m4/m4_display_recommendations.jsonl"
)

DEFAULT_CASE_STUDIES_INPUT = Path(
    "data/outputs/m4/m4_case_studies.jsonl"
)

DEFAULT_OUTPUT = Path(
    "data/outputs/m4/m4_structural_validation_report.json"
)

DEFAULT_FAILURES_OUTPUT = Path(
    "data/outputs/m4/m4_structural_validation_failures.csv"
)


VALID_LEVELS = {"high", "moderate", "exploratory"}

FAILURE_COLUMNS = [
    "validation_check",
    "module_id",
    "module_title",
    "staff_a_id",
    "staff_a_name",
    "staff_b_id",
    "staff_b_name",
    "rank",
    "recommendation_level",
    "collaboration_score",
    "combined_module_coverage",
    "best_individual_coverage",
    "complementarity_gain",
    "explanation",
]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object on line {line_number} in {path}"
                )

            yield record


def read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    return list(read_jsonl(path))


def write_failures_csv(
    path: Path,
    failures: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FAILURE_COLUMNS,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(failures)


def as_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return "; ".join(
            as_text(item)
            for item in value
            if as_text(item)
        )

    return str(value).strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def has_non_empty_list(
    record: dict[str, Any],
    key: str,
) -> bool:
    value = record.get(key)

    return isinstance(value, list) and len(value) > 0


def failure_row(
    record: dict[str, Any],
    validation_check: str,
) -> dict[str, Any]:
    return {
        "validation_check": validation_check,
        "module_id": as_text(record.get("module_id")),
        "module_title": as_text(record.get("module_title")),
        "staff_a_id": as_text(record.get("staff_a_id")),
        "staff_a_name": as_text(record.get("staff_a_name")),
        "staff_b_id": as_text(record.get("staff_b_id")),
        "staff_b_name": as_text(record.get("staff_b_name")),
        "rank": safe_int(record.get("rank")),
        "recommendation_level": as_text(
            record.get("recommendation_level")
        ),
        "collaboration_score": safe_float(
            record.get("collaboration_score")
        ),
        "combined_module_coverage": safe_float(
            record.get("combined_module_coverage")
        ),
        "best_individual_coverage": safe_float(
            record.get("best_individual_coverage")
        ),
        "complementarity_gain": safe_float(
            record.get("complementarity_gain")
        ),
        "explanation": as_text(record.get("explanation")),
    }


def numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "minimum": 0.0,
            "maximum": 0.0,
            "mean": 0.0,
            "median": 0.0,
        }

    return {
        "count": len(values),
        "minimum": round(min(values), 6),
        "maximum": round(max(values), 6),
        "mean": round(statistics.mean(values), 6),
        "median": round(statistics.median(values), 6),
    }


def distribution(
    values: list[float],
    boundaries: list[float],
) -> dict[str, int]:
    if not boundaries:
        return {}

    ordered = sorted(boundaries)
    counts: Counter[str] = Counter()

    for value in values:
        previous = 0.0

        for boundary in ordered:
            if value < boundary:
                label = f"{previous:.2f}-{boundary:.2f}"
                counts[label] += 1
                break

            previous = boundary
        else:
            label = f"{ordered[-1]:.2f}+"
            counts[label] += 1

    return dict(sorted(counts.items()))


def validate_record(
    record: dict[str, Any],
    tolerance: float,
) -> list[str]:
    failures: list[str] = []

    module_id = as_text(record.get("module_id"))
    staff_a_id = as_text(record.get("staff_a_id"))
    staff_b_id = as_text(record.get("staff_b_id"))

    module_title = as_text(record.get("module_title"))
    staff_a_name = as_text(record.get("staff_a_name"))
    staff_b_name = as_text(record.get("staff_b_name"))
    explanation = as_text(record.get("explanation"))

    level = as_text(
        record.get("recommendation_level")
    ).lower()

    rank = safe_int(record.get("rank"), default=0)

    collaboration_score = safe_float(
        record.get("collaboration_score")
    )

    combined_coverage = safe_float(
        record.get("combined_module_coverage")
    )

    best_individual_coverage = safe_float(
        record.get("best_individual_coverage")
    )

    complementarity_gain = safe_float(
        record.get("complementarity_gain")
    )

    if not module_id:
        failures.append("missing_module_id")

    if not staff_a_id:
        failures.append("missing_staff_a_id")

    if not staff_b_id:
        failures.append("missing_staff_b_id")

    if staff_a_id and staff_a_id == staff_b_id:
        failures.append("self_pair")

    if not module_title:
        failures.append("missing_module_title")

    if not staff_a_name:
        failures.append("missing_staff_a_name")

    if not staff_b_name:
        failures.append("missing_staff_b_name")

    if level not in VALID_LEVELS:
        failures.append("invalid_recommendation_level")

    if rank < 1:
        failures.append("invalid_rank")

    if collaboration_score < 0:
        failures.append("negative_collaboration_score")

    if combined_coverage < 0 or combined_coverage > 1 + tolerance:
        failures.append("invalid_combined_coverage_range")

    if (
        best_individual_coverage < 0
        or best_individual_coverage > 1 + tolerance
    ):
        failures.append("invalid_individual_coverage_range")

    if complementarity_gain <= 0:
        failures.append("non_positive_complementarity_gain")

    expected_gain = (
        combined_coverage - best_individual_coverage
    )

    if abs(complementarity_gain - expected_gain) > tolerance:
        failures.append("inconsistent_complementarity_gain")

    if combined_coverage + tolerance < best_individual_coverage:
        failures.append("pair_coverage_less_than_best_individual")

    has_topic_evidence = (
        has_non_empty_list(record, "shared_module_topics")
        or has_non_empty_list(
            record,
            "complementary_module_topics",
        )
    )

    if not has_topic_evidence:
        failures.append("missing_topic_evidence")

    if not explanation:
        failures.append("missing_explanation")
    else:
        explanation_lower = explanation.lower()

        if module_title.lower() not in explanation_lower:
            failures.append("module_title_missing_from_explanation")

        if staff_a_name.lower() not in explanation_lower:
            failures.append("staff_a_name_missing_from_explanation")

        if staff_b_name.lower() not in explanation_lower:
            failures.append("staff_b_name_missing_from_explanation")

    return failures


def validate_rank_consistency(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []

    by_module: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        module_id = as_text(record.get("module_id"))

        if module_id:
            by_module[module_id].append(record)

    for module_records in by_module.values():
        module_records.sort(
            key=lambda record: safe_int(
                record.get("rank"),
                default=999999,
            )
        )

        ranks = [
            safe_int(record.get("rank"), default=0)
            for record in module_records
        ]

        expected_ranks = list(range(1, len(ranks) + 1))

        if ranks != expected_ranks:
            for record in module_records:
                failures.append(
                    failure_row(
                        record=record,
                        validation_check="non_contiguous_or_duplicate_ranks",
                    )
                )

            continue

        scores = [
            safe_float(record.get("collaboration_score"))
            for record in module_records
        ]

        for earlier_score, later_score in zip(
            scores,
            scores[1:],
        ):
            if earlier_score + 1e-12 < later_score:
                for record in module_records:
                    failures.append(
                        failure_row(
                            record=record,
                            validation_check="rank_score_order_inconsistent",
                        )
                    )

                break

    return failures


def validate_duplicate_pairs(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}

    for record in records:
        module_id = as_text(record.get("module_id"))
        staff_a_id = as_text(record.get("staff_a_id"))
        staff_b_id = as_text(record.get("staff_b_id"))

        ordered_pair = tuple(sorted([staff_a_id, staff_b_id]))

        key = (
            module_id,
            ordered_pair[0],
            ordered_pair[1],
        )

        if key in seen:
            failures.append(
                failure_row(
                    record=record,
                    validation_check="duplicate_staff_pair_within_module",
                )
            )
        else:
            seen[key] = record

    return failures


def display_policy_validation(
    display_records: list[dict[str, Any]],
    minimum_coverage: float,
    minimum_gain: float,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []

    for record in display_records:
        level = as_text(
            record.get("recommendation_level")
        ).lower()

        rank = safe_int(record.get("rank"))

        coverage = safe_float(
            record.get("combined_module_coverage")
        )

        gain = safe_float(
            record.get("complementarity_gain")
        )

        if level != "high":
            failures.append(
                failure_row(
                    record,
                    "display_record_not_high_level",
                )
            )

        if rank != 1:
            failures.append(
                failure_row(
                    record,
                    "display_record_not_top_ranked",
                )
            )

        if coverage < minimum_coverage:
            failures.append(
                failure_row(
                    record,
                    "display_record_below_coverage_threshold",
                )
            )

        if gain < minimum_gain:
            failures.append(
                failure_row(
                    record,
                    "display_record_below_gain_threshold",
                )
            )

    return {
        "records_checked": len(display_records),
        "failures": failures,
        "failure_count": len(failures),
        "policy_compliant": len(failures) == 0,
    }


def case_study_validation(
    case_studies: list[dict[str, Any]],
) -> dict[str, Any]:
    case_ids = [
        as_text(record.get("case_id"))
        for record in case_studies
    ]

    duplicate_case_ids = [
        case_id
        for case_id, count in Counter(case_ids).items()
        if case_id and count > 1
    ]

    selection_group_counts = Counter(
        as_text(record.get("selection_group"))
        for record in case_studies
    )

    level_counts = Counter(
        as_text(record.get("recommendation_level")).lower()
        for record in case_studies
    )

    modules = [
        as_text(record.get("module_id"))
        for record in case_studies
    ]

    duplicate_modules = [
        module_id
        for module_id, count in Counter(modules).items()
        if module_id and count > 1
    ]

    return {
        "case_studies_checked": len(case_studies),
        "selection_group_counts": dict(
            sorted(selection_group_counts.items())
        ),
        "recommendation_level_counts": dict(
            sorted(level_counts.items())
        ),
        "duplicate_case_ids": duplicate_case_ids,
        "duplicate_case_id_count": len(duplicate_case_ids),
        "duplicate_module_ids": duplicate_modules,
        "duplicate_module_id_count": len(duplicate_modules),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Perform structural validation of M4 collaboration "
            "recommendations, display output, and case-study samples."
        )
    )

    parser.add_argument(
        "--recommendations-input",
        type=Path,
        default=DEFAULT_RECOMMENDATIONS_INPUT,
        help="Path to enriched collaboration recommendations JSONL.",
    )

    parser.add_argument(
        "--display-input",
        type=Path,
        default=DEFAULT_DISPLAY_INPUT,
        help="Path to display-ready recommendation JSONL.",
    )

    parser.add_argument(
        "--case-studies-input",
        type=Path,
        default=DEFAULT_CASE_STUDIES_INPUT,
        help="Path to M4 case-study JSONL.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for structural validation JSON report.",
    )

    parser.add_argument(
        "--failures-output",
        type=Path,
        default=DEFAULT_FAILURES_OUTPUT,
        help="Path for validation failures CSV.",
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.00001,
        help=(
            "Numeric tolerance for coverage and complementarity "
            "consistency checks. Default: 0.00001."
        ),
    )

    parser.add_argument(
        "--display-minimum-coverage",
        type=float,
        default=0.40,
        help=(
            "Expected minimum combined coverage for display records. "
            "Default: 0.40."
        ),
    )

    parser.add_argument(
        "--display-minimum-gain",
        type=float,
        default=0.10,
        help=(
            "Expected minimum complementarity gain for display "
            "records. Default: 0.10."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.recommendations_input.exists():
        raise FileNotFoundError(
            "Enriched recommendation input was not found: "
            f"{args.recommendations_input}"
        )

    recommendations = list(
        read_jsonl(args.recommendations_input)
    )

    display_records = read_optional_jsonl(args.display_input)
    case_studies = read_optional_jsonl(args.case_studies_input)

    record_failures: list[dict[str, Any]] = []

    failure_counts: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()

    scores: list[float] = []
    combined_coverages: list[float] = []
    best_individual_coverages: list[float] = []
    complementarity_gains: list[float] = []
    overlap_ratios: list[float] = []

    modules_with_recommendations: set[str] = set()
    modules_with_shared_foundation: set[str] = set()

    explanation_complete_count = 0
    topic_evidence_count = 0
    positive_gain_count = 0

    for record in recommendations:
        level = as_text(
            record.get("recommendation_level")
        ).lower()

        level_counts[level] += 1

        module_id = as_text(record.get("module_id"))

        if module_id:
            modules_with_recommendations.add(module_id)

        if (
            has_non_empty_list(record, "shared_module_topics")
            or has_non_empty_list(record, "shared_staff_topics")
        ):
            if module_id:
                modules_with_shared_foundation.add(module_id)

        score = safe_float(record.get("collaboration_score"))
        combined_coverage = safe_float(
            record.get("combined_module_coverage")
        )
        best_individual_coverage = safe_float(
            record.get("best_individual_coverage")
        )
        complementarity_gain = safe_float(
            record.get("complementarity_gain")
        )
        overlap_ratio = safe_float(
            record.get("staff_topic_overlap_ratio")
        )

        scores.append(score)
        combined_coverages.append(combined_coverage)
        best_individual_coverages.append(
            best_individual_coverage
        )
        complementarity_gains.append(complementarity_gain)
        overlap_ratios.append(overlap_ratio)

        if complementarity_gain > 0:
            positive_gain_count += 1

        if (
            has_non_empty_list(record, "shared_module_topics")
            or has_non_empty_list(
                record,
                "complementary_module_topics",
            )
        ):
            topic_evidence_count += 1

        if (
            as_text(record.get("module_title"))
            and as_text(record.get("staff_a_name"))
            and as_text(record.get("staff_b_name"))
            and as_text(record.get("explanation"))
        ):
            explanation_complete_count += 1

        checks_failed = validate_record(
            record=record,
            tolerance=args.tolerance,
        )

        for check in checks_failed:
            failure_counts[check] += 1
            record_failures.append(
                failure_row(
                    record=record,
                    validation_check=check,
                )
            )

    rank_failures = validate_rank_consistency(
        records=recommendations
    )

    duplicate_pair_failures = validate_duplicate_pairs(
        records=recommendations
    )

    for failure in rank_failures + duplicate_pair_failures:
        failure_counts[failure["validation_check"]] += 1
        record_failures.append(failure)

    display_validation = display_policy_validation(
        display_records=display_records,
        minimum_coverage=args.display_minimum_coverage,
        minimum_gain=args.display_minimum_gain,
    )

    for failure in display_validation["failures"]:
        failure_counts[failure["validation_check"]] += 1
        record_failures.append(failure)

    case_validation = case_study_validation(
        case_studies=case_studies
    )

    write_failures_csv(
        path=args.failures_output,
        failures=record_failures,
    )

    recommendation_count = len(recommendations)

    report = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "inputs": {
            "enriched_recommendations": str(
                args.recommendations_input
            ),
            "display_recommendations": str(
                args.display_input
            ),
            "case_studies": str(args.case_studies_input),
        },
        "parameters": {
            "numeric_tolerance": args.tolerance,
            "display_minimum_combined_coverage": (
                args.display_minimum_coverage
            ),
            "display_minimum_complementarity_gain": (
                args.display_minimum_gain
            ),
        },
        "counts": {
            "recommendations_checked": recommendation_count,
            "modules_with_recommendations": len(
                modules_with_recommendations
            ),
            "recommendations_by_level": dict(
                sorted(level_counts.items())
            ),
            "recommendations_with_positive_gain": (
                positive_gain_count
            ),
            "recommendations_with_topic_evidence": (
                topic_evidence_count
            ),
            "recommendations_with_complete_explanation": (
                explanation_complete_count
            ),
            "record_level_failure_count": len(
                record_failures
            ),
        },
        "rates": {
            "positive_complementarity_rate": round(
                positive_gain_count / recommendation_count,
                6,
            )
            if recommendation_count
            else 0.0,
            "topic_evidence_rate": round(
                topic_evidence_count / recommendation_count,
                6,
            )
            if recommendation_count
            else 0.0,
            "complete_explanation_rate": round(
                explanation_complete_count / recommendation_count,
                6,
            )
            if recommendation_count
            else 0.0,
            "modules_with_shared_foundation_rate": round(
                len(modules_with_shared_foundation)
                / len(modules_with_recommendations),
                6,
            )
            if modules_with_recommendations
            else 0.0,
        },
        "numeric_distributions": {
            "collaboration_score": numeric_summary(scores),
            "combined_module_coverage": numeric_summary(
                combined_coverages
            ),
            "best_individual_coverage": numeric_summary(
                best_individual_coverages
            ),
            "complementarity_gain": numeric_summary(
                complementarity_gains
            ),
            "staff_topic_overlap_ratio": numeric_summary(
                overlap_ratios
            ),
            "combined_module_coverage_bands": distribution(
                combined_coverages,
                boundaries=[0.20, 0.40, 0.60, 0.80],
            ),
            "complementarity_gain_bands": distribution(
                complementarity_gains,
                boundaries=[0.05, 0.10, 0.20, 0.30],
            ),
        },
        "display_policy_validation": {
            "records_checked": display_validation[
                "records_checked"
            ],
            "failure_count": display_validation[
                "failure_count"
            ],
            "policy_compliant": display_validation[
                "policy_compliant"
            ],
        },
        "case_study_validation": case_validation,
        "failure_summary": {
            "total_failure_rows": len(record_failures),
            "failure_counts": dict(
                sorted(failure_counts.items())
            ),
            "failure_csv": str(args.failures_output),
        },
        "interpretation": {
            "scope": (
                "This report validates structural and internal "
                "consistency properties of M4 outputs. It does not "
                "establish domain-level teaching suitability, which "
                "requires subject-expert judgement."
            ),
            "recommended_use": (
                "Use this report for technical evaluation, "
                "reproducibility, explanation traceability, "
                "and limitation discussion in the dissertation."
            ),
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()