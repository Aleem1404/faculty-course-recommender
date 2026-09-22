from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path(
    "data/outputs/m4/m4_case_studies.jsonl"
)

DEFAULT_OUTPUT = Path(
    "data/outputs/m4/m4_case_study_review_sheet.csv"
)

DEFAULT_SUMMARY_OUTPUT = Path(
    "data/outputs/m4/m4_case_study_review_sheet_summary.json"
)


CSV_COLUMNS = [
    "case_id",
    "selection_group",
    "module_title",
    "module_code",
    "module_level",
    "module_department",
    "staff_a_name",
    "staff_a_job_title",
    "staff_a_department",
    "staff_b_name",
    "staff_b_job_title",
    "staff_b_department",
    "recommendation_level",
    "rank",
    "collaboration_score",
    "combined_module_coverage",
    "best_individual_coverage",
    "complementarity_gain",
    "staff_topic_overlap_ratio",
    "has_shared_foundation",
    "shared_module_topics",
    "staff_a_complementary_topics",
    "staff_b_complementary_topics",
    "explanation",
    "staff_a_relevance",
    "staff_b_relevance",
    "complementary_expertise",
    "explanation_clarity",
    "practical_plausibility",
    "overall_recommendation_quality",
    "reviewer_comments",
]


REVIEW_GUIDANCE = {
    "staff_a_relevance": (
        "Allowed values: relevant | partly_relevant | not_relevant"
    ),
    "staff_b_relevance": (
        "Allowed values: relevant | partly_relevant | not_relevant"
    ),
    "complementary_expertise": (
        "Allowed values: clear | partial | not_clear"
    ),
    "explanation_clarity": (
        "Allowed values: clear | partly_clear | unclear"
    ),
    "practical_plausibility": (
        "Allowed values: plausible | partly_plausible | not_plausible"
    ),
    "overall_recommendation_quality": (
        "Allowed values: good | acceptable | weak"
    ),
    "reviewer_comments": (
        "Free-text comments, limitations, justification, or corrections"
    ),
}


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


def percentage(value: Any) -> str:
    return f"{safe_float(value) * 100:.1f}%"


def yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def review_value(
    record: dict[str, Any],
    field_name: str,
) -> str:
    review = record.get("review", {})

    if not isinstance(review, dict):
        return ""

    return as_text(review.get(field_name))


def case_study_to_row(record: dict[str, Any]) -> dict[str, str]:
    return {
        "case_id": as_text(record.get("case_id")),
        "selection_group": as_text(record.get("selection_group")),
        "module_title": as_text(record.get("module_title")),
        "module_code": as_text(record.get("module_code")),
        "module_level": as_text(record.get("module_level")),
        "module_department": as_text(
            record.get("module_department")
        ),
        "staff_a_name": as_text(record.get("staff_a_name")),
        "staff_a_job_title": as_text(
            record.get("staff_a_job_title")
        ),
        "staff_a_department": as_text(
            record.get("staff_a_department")
        ),
        "staff_b_name": as_text(record.get("staff_b_name")),
        "staff_b_job_title": as_text(
            record.get("staff_b_job_title")
        ),
        "staff_b_department": as_text(
            record.get("staff_b_department")
        ),
        "recommendation_level": as_text(
            record.get("recommendation_level")
        ),
        "rank": str(safe_int(record.get("rank"))),
        "collaboration_score": (
            f"{safe_float(record.get('collaboration_score')):.6f}"
        ),
        "combined_module_coverage": percentage(
            record.get("combined_module_coverage")
        ),
        "best_individual_coverage": percentage(
            record.get("best_individual_coverage")
        ),
        "complementarity_gain": percentage(
            record.get("complementarity_gain")
        ),
        "staff_topic_overlap_ratio": percentage(
            record.get("staff_topic_overlap_ratio")
        ),
        "has_shared_foundation": yes_no(
            record.get("has_shared_foundation")
        ),
        "shared_module_topics": as_text(
            record.get("shared_module_topics")
        ),
        "staff_a_complementary_topics": as_text(
            record.get("staff_a_complementary_topics")
        ),
        "staff_b_complementary_topics": as_text(
            record.get("staff_b_complementary_topics")
        ),
        "explanation": as_text(record.get("explanation")),
        "staff_a_relevance": review_value(
            record,
            "staff_a_relevance",
        ),
        "staff_b_relevance": review_value(
            record,
            "staff_b_relevance",
        ),
        "complementary_expertise": review_value(
            record,
            "complementary_expertise",
        ),
        "explanation_clarity": review_value(
            record,
            "explanation_clarity",
        ),
        "practical_plausibility": review_value(
            record,
            "practical_plausibility",
        ),
        "overall_recommendation_quality": review_value(
            record,
            "overall_recommendation_quality",
        ),
        "reviewer_comments": review_value(
            record,
            "reviewer_comments",
        ),
    }


def write_review_csv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    input_path: Path,
    output_path: Path,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    selection_group_counts = Counter(
        row["selection_group"]
        for row in rows
    )

    recommendation_level_counts = Counter(
        row["recommendation_level"]
        for row in rows
    )

    shared_foundation_counts = Counter(
        row["has_shared_foundation"]
        for row in rows
    )

    return {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "inputs": {
            "case_studies": str(input_path),
        },
        "outputs": {
            "review_sheet_csv": str(output_path),
        },
        "review_guidance": REVIEW_GUIDANCE,
        "counts": {
            "case_studies_exported": len(rows),
            "selection_groups": dict(
                sorted(selection_group_counts.items())
            ),
            "recommendation_levels": dict(
                sorted(recommendation_level_counts.items())
            ),
            "shared_foundation": dict(
                sorted(shared_foundation_counts.items())
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export selected M4 collaboration case studies to a "
            "review-friendly CSV file."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to m4_case_studies.jsonl.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for the review CSV.",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Output path for the CSV export summary JSON.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Case-study input file not found: {args.input}"
        )

    records = list(read_jsonl(args.input))

    rows = [
        case_study_to_row(record)
        for record in records
    ]

    rows.sort(
        key=lambda row: (
            row["case_id"],
            row["module_title"].lower(),
        )
    )

    write_review_csv(
        path=args.output,
        rows=rows,
    )

    summary = build_summary(
        input_path=args.input,
        output_path=args.output,
        rows=rows,
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