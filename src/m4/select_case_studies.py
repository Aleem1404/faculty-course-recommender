from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path(
    "data/outputs/m4/collaboration_recommendations_enriched.jsonl"
)

DEFAULT_DISPLAY_OUTPUT = Path(
    "data/outputs/m4/m4_display_recommendations.jsonl"
)

DEFAULT_CASE_STUDIES_OUTPUT = Path(
    "data/outputs/m4/m4_case_studies.jsonl"
)

DEFAULT_SUMMARY_OUTPUT = Path(
    "data/outputs/m4/m4_case_studies_summary.json"
)


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


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def as_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return ", ".join(
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


def topic_labels(
    topics: list[dict[str, Any]],
    maximum: int = 5,
) -> list[str]:
    labels: list[str] = []

    for topic in topics[:maximum]:
        label = as_text(topic.get("topic"))

        if label:
            labels.append(label)

    return labels


def has_shared_foundation(record: dict[str, Any]) -> bool:
    shared_module_topics = record.get(
        "shared_module_topics",
        [],
    )

    shared_staff_topics = record.get(
        "shared_staff_topics",
        [],
    )

    return bool(shared_module_topics or shared_staff_topics)


def record_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -safe_float(record.get("collaboration_score")),
        -safe_float(record.get("combined_module_coverage")),
        -safe_float(record.get("complementarity_gain")),
        safe_int(record.get("rank"), default=999999),
        as_text(record.get("module_title")).lower(),
        as_text(record.get("staff_a_name")).lower(),
        as_text(record.get("staff_b_name")).lower(),
    )


def normalise_level(value: Any) -> str:
    level = as_text(value).lower()

    if level in {"high", "moderate", "exploratory"}:
        return level

    return "unknown"


def passes_display_policy(
    record: dict[str, Any],
    minimum_coverage: float,
    minimum_gain: float,
) -> bool:
    return (
        normalise_level(record.get("recommendation_level")) == "high"
        and safe_int(record.get("rank"), default=999999) == 1
        and safe_float(record.get("combined_module_coverage"))
        >= minimum_coverage
        and safe_float(record.get("complementarity_gain"))
        >= minimum_gain
    )


def top_record_per_module(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        module_id = as_text(record.get("module_id"))

        if module_id:
            grouped[module_id].append(record)

    selected: list[dict[str, Any]] = []

    for module_records in grouped.values():
        module_records.sort(key=record_sort_key)
        selected.append(module_records[0])

    selected.sort(key=record_sort_key)

    return selected


def select_diverse_records(
    records: list[dict[str, Any]],
    requested_count: int,
) -> list[dict[str, Any]]:
    if requested_count <= 0:
        return []

    candidates = top_record_per_module(records)

    if not candidates:
        return []

    with_shared = [
        record
        for record in candidates
        if has_shared_foundation(record)
    ]

    without_shared = [
        record
        for record in candidates
        if not has_shared_foundation(record)
    ]

    with_shared.sort(key=record_sort_key)
    without_shared.sort(key=record_sort_key)

    selected: list[dict[str, Any]] = []
    selected_module_ids: set[str] = set()

    preferred_groups = [with_shared, without_shared]

    group_index = 0

    while len(selected) < requested_count:
        current_group = preferred_groups[group_index % len(preferred_groups)]
        group_index += 1

        record_to_add = None

        for record in current_group:
            module_id = as_text(record.get("module_id"))

            if module_id not in selected_module_ids:
                record_to_add = record
                break

        if record_to_add is None:
            remaining = [
                record
                for record in candidates
                if as_text(record.get("module_id"))
                not in selected_module_ids
            ]

            if not remaining:
                break

            remaining.sort(key=record_sort_key)
            record_to_add = remaining[0]

        module_id = as_text(record_to_add.get("module_id"))

        selected.append(record_to_add)
        selected_module_ids.add(module_id)

        with_shared = [
            record
            for record in with_shared
            if as_text(record.get("module_id"))
            not in selected_module_ids
        ]

        without_shared = [
            record
            for record in without_shared
            if as_text(record.get("module_id"))
            not in selected_module_ids
        ]

        preferred_groups = [with_shared, without_shared]

        if not any(preferred_groups):
            break

    return selected


def build_case_study_record(
    recommendation: dict[str, Any],
    case_id: str,
    selection_group: str,
) -> dict[str, Any]:
    shared_topics = topic_labels(
        recommendation.get("shared_module_topics", []),
    )

    complementary_topics = recommendation.get(
        "complementary_module_topics",
        [],
    )

    staff_a_topics = topic_labels(
        [
            topic
            for topic in complementary_topics
            if topic.get("provided_by") == "staff_a"
        ]
    )

    staff_b_topics = topic_labels(
        [
            topic
            for topic in complementary_topics
            if topic.get("provided_by") == "staff_b"
        ]
    )

    return {
        "case_id": case_id,
        "selection_group": selection_group,
        "module_id": as_text(recommendation.get("module_id")),
        "module_title": as_text(recommendation.get("module_title")),
        "module_code": as_text(recommendation.get("module_code")),
        "module_level": as_text(recommendation.get("module_level")),
        "module_department": as_text(
            recommendation.get("module_department")
        ),
        "staff_a_id": as_text(recommendation.get("staff_a_id")),
        "staff_a_name": as_text(recommendation.get("staff_a_name")),
        "staff_a_job_title": as_text(
            recommendation.get("staff_a_job_title")
        ),
        "staff_a_department": as_text(
            recommendation.get("staff_a_department")
        ),
        "staff_b_id": as_text(recommendation.get("staff_b_id")),
        "staff_b_name": as_text(recommendation.get("staff_b_name")),
        "staff_b_job_title": as_text(
            recommendation.get("staff_b_job_title")
        ),
        "staff_b_department": as_text(
            recommendation.get("staff_b_department")
        ),
        "rank": safe_int(recommendation.get("rank")),
        "recommendation_level": normalise_level(
            recommendation.get("recommendation_level")
        ),
        "collaboration_score": round(
            safe_float(recommendation.get("collaboration_score")),
            6,
        ),
        "combined_module_coverage": round(
            safe_float(
                recommendation.get("combined_module_coverage")
            ),
            6,
        ),
        "best_individual_coverage": round(
            safe_float(
                recommendation.get("best_individual_coverage")
            ),
            6,
        ),
        "complementarity_gain": round(
            safe_float(
                recommendation.get("complementarity_gain")
            ),
            6,
        ),
        "staff_topic_overlap_ratio": round(
            safe_float(
                recommendation.get("staff_topic_overlap_ratio")
            ),
            6,
        ),
        "has_shared_foundation": has_shared_foundation(
            recommendation
        ),
        "shared_module_topics": shared_topics,
        "staff_a_complementary_topics": staff_a_topics,
        "staff_b_complementary_topics": staff_b_topics,
        "explanation": as_text(recommendation.get("explanation")),
        "review": {
            "staff_a_relevance": "",
            "staff_b_relevance": "",
            "complementary_expertise": "",
            "explanation_clarity": "",
            "practical_plausibility": "",
            "overall_recommendation_quality": "",
            "reviewer_comments": "",
        },
    }


def build_summary(
    all_records: list[dict[str, Any]],
    display_records: list[dict[str, Any]],
    case_studies: list[dict[str, Any]],
    requested_counts: dict[str, int],
    policy: dict[str, float],
) -> dict[str, Any]:
    input_level_counts = Counter(
        normalise_level(record.get("recommendation_level"))
        for record in all_records
    )

    display_level_counts = Counter(
        normalise_level(record.get("recommendation_level"))
        for record in display_records
    )

    case_level_counts = Counter(
        normalise_level(record.get("recommendation_level"))
        for record in case_studies
    )

    case_group_counts = Counter(
        as_text(record.get("selection_group"))
        for record in case_studies
    )

    case_foundation_counts = Counter(
        "with_shared_foundation"
        if record.get("has_shared_foundation")
        else "without_shared_foundation"
        for record in case_studies
    )

    return {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "selection_policy": {
            "display_rule": (
                "Top-ranked high-level recommendation per module "
                "with sufficient combined coverage and "
                "complementarity gain."
            ),
            "minimum_display_combined_coverage": policy[
                "minimum_display_coverage"
            ],
            "minimum_display_complementarity_gain": policy[
                "minimum_display_gain"
            ],
            "requested_case_studies": requested_counts,
        },
        "counts": {
            "recommendations_read": len(all_records),
            "recommendations_by_level": dict(
                sorted(input_level_counts.items())
            ),
            "display_recommendations_written": len(display_records),
            "display_recommendations_by_level": dict(
                sorted(display_level_counts.items())
            ),
            "case_studies_written": len(case_studies),
            "case_studies_by_level": dict(
                sorted(case_level_counts.items())
            ),
            "case_studies_by_selection_group": dict(
                sorted(case_group_counts.items())
            ),
            "case_studies_by_foundation_type": dict(
                sorted(case_foundation_counts.items())
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select display-ready M4 collaboration recommendations "
            "and a balanced case-study sample for evaluation."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to enriched M4 collaboration recommendations.",
    )

    parser.add_argument(
        "--display-output",
        type=Path,
        default=DEFAULT_DISPLAY_OUTPUT,
        help="Output path for high-confidence display recommendations.",
    )

    parser.add_argument(
        "--case-studies-output",
        type=Path,
        default=DEFAULT_CASE_STUDIES_OUTPUT,
        help="Output path for M4 case-study records.",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Output path for selection summary.",
    )

    parser.add_argument(
        "--minimum-display-coverage",
        type=float,
        default=0.40,
        help=(
            "Minimum combined module coverage for a display-ready "
            "recommendation. Default: 0.40."
        ),
    )

    parser.add_argument(
        "--minimum-display-gain",
        type=float,
        default=0.10,
        help=(
            "Minimum complementarity gain for a display-ready "
            "recommendation. Default: 0.10."
        ),
    )

    parser.add_argument(
        "--high-cases",
        type=int,
        default=10,
        help="Requested high-level case studies. Default: 10.",
    )

    parser.add_argument(
        "--moderate-cases",
        type=int,
        default=10,
        help="Requested moderate-level case studies. Default: 10.",
    )

    parser.add_argument(
        "--exploratory-cases",
        type=int,
        default=5,
        help="Requested exploratory case studies. Default: 5.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Enriched recommendation input not found: {args.input}"
        )

    if args.minimum_display_coverage < 0:
        raise ValueError(
            "--minimum-display-coverage cannot be negative."
        )

    if args.minimum_display_gain < 0:
        raise ValueError(
            "--minimum-display-gain cannot be negative."
        )

    requested_counts = {
        "high": max(args.high_cases, 0),
        "moderate": max(args.moderate_cases, 0),
        "exploratory": max(args.exploratory_cases, 0),
    }

    all_records = list(read_jsonl(args.input))

    display_candidates = [
        record
        for record in all_records
        if passes_display_policy(
            record=record,
            minimum_coverage=args.minimum_display_coverage,
            minimum_gain=args.minimum_display_gain,
        )
    ]

    display_records = top_record_per_module(display_candidates)

    display_records.sort(
        key=lambda record: (
            as_text(record.get("module_title")).lower(),
            record_sort_key(record),
        )
    )

    records_by_level: dict[str, list[dict[str, Any]]] = {
        "high": [],
        "moderate": [],
        "exploratory": [],
    }

    for record in all_records:
        level = normalise_level(record.get("recommendation_level"))

        if level in records_by_level:
            records_by_level[level].append(record)

    selected_by_group: dict[str, list[dict[str, Any]]] = {}

    for level, requested_count in requested_counts.items():
        selected_by_group[level] = select_diverse_records(
            records=records_by_level[level],
            requested_count=requested_count,
        )

    case_studies: list[dict[str, Any]] = []
    case_number = 1

    for level in ["high", "moderate", "exploratory"]:
        for recommendation in selected_by_group[level]:
            case_studies.append(
                build_case_study_record(
                    recommendation=recommendation,
                    case_id=f"M4-{case_number:03d}",
                    selection_group=level,
                )
            )

            case_number += 1

    write_jsonl(args.display_output, display_records)
    write_jsonl(args.case_studies_output, case_studies)

    summary = build_summary(
        all_records=all_records,
        display_records=display_records,
        case_studies=case_studies,
        requested_counts=requested_counts,
        policy={
            "minimum_display_coverage": (
                args.minimum_display_coverage
            ),
            "minimum_display_gain": args.minimum_display_gain,
        },
    )

    args.summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.summary_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()