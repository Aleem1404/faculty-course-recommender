from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CANDIDATES_PATH = Path(
    "data/outputs/m4/collaboration_candidates.jsonl"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/outputs/m4/collaboration_recommendations.jsonl"
)

DEFAULT_SUMMARY_PATH = Path(
    "data/outputs/m4/collaboration_recommendations_summary.json"
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


def topic_labels(
    topics: list[dict[str, Any]],
    maximum: int = 3,
) -> list[str]:
    labels: list[str] = []

    for topic in topics[:maximum]:
        label = str(topic.get("topic", "")).strip()

        if label:
            labels.append(label)

    return labels


def human_list(values: list[str]) -> str:
    if not values:
        return ""

    if len(values) == 1:
        return values[0]

    if len(values) == 2:
        return f"{values[0]} and {values[1]}"

    return f"{', '.join(values[:-1])}, and {values[-1]}"


# def recommendation_level(score: float) -> str:
#     if score >= 0.60:
#         return "high"

#     if score >= 0.35:
#         return "moderate"

#     return "exploratory"
def recommendation_level(score: float) -> str:
    if score >= 0.35:
        return "high"

    if score >= 0.20:
        return "moderate"

    return "exploratory"


def build_explanation(candidate: dict[str, Any]) -> str:
    staff_a = str(candidate.get("staff_a_id", "Staff A"))
    staff_b = str(candidate.get("staff_b_id", "Staff B"))

    shared_topics = topic_labels(
        candidate.get("shared_module_topics", []),
    )

    complementary = candidate.get(
        "complementary_module_topics",
        [],
    )

    staff_a_topics = topic_labels(
        [
            topic
            for topic in complementary
            if topic.get("provided_by") == "staff_a"
        ]
    )

    staff_b_topics = topic_labels(
        [
            topic
            for topic in complementary
            if topic.get("provided_by") == "staff_b"
        ]
    )

    combined_coverage = float(
        candidate.get("combined_module_coverage", 0.0)
    )

    complementarity_gain = float(
        candidate.get("complementarity_gain", 0.0)
    )

    coverage_percentage = round(combined_coverage * 100, 1)
    gain_percentage = round(complementarity_gain * 100, 1)

    sentences: list[str] = []

    if shared_topics:
        sentences.append(
            f"{staff_a} and {staff_b} share relevant expertise in "
            f"{human_list(shared_topics)}."
        )
    else:
        sentences.append(
            f"{staff_a} and {staff_b} provide complementary expertise "
            f"for this module."
        )

    contribution_parts: list[str] = []

    if staff_a_topics:
        contribution_parts.append(
            f"{staff_a} contributes {human_list(staff_a_topics)}"
        )

    if staff_b_topics:
        contribution_parts.append(
            f"{staff_b} contributes {human_list(staff_b_topics)}"
        )

    if contribution_parts:
        sentences.append(
            f"{human_list(contribution_parts)}."
        )

    sentences.append(
        f"Together, the pair covers approximately "
        f"{coverage_percentage}% of the weighted module topics, "
        f"which is an additional {gain_percentage}% coverage "
        f"beyond the stronger individual candidate."
    )

    return " ".join(sentences)


def build_recommendation(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    score = float(candidate.get("collaboration_score", 0.0))

    return {
        **candidate,
        "recommendation_level": recommendation_level(score),
        "explanation": build_explanation(candidate),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate evidence-based natural-language explanations for "
            "M4 collaboration candidates."
        )
    )

    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_CANDIDATES_PATH,
        help="Path to collaboration candidate JSONL input.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for explained collaboration recommendations.",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path for explanation-generation summary.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.candidates.exists():
        raise FileNotFoundError(
            f"Candidate input file not found: {args.candidates}"
        )

    recommendations = [
        build_recommendation(candidate)
        for candidate in read_jsonl(args.candidates)
    ]

    write_jsonl(args.output, recommendations)

    level_counts = {
        "high": 0,
        "moderate": 0,
        "exploratory": 0,
    }

    for recommendation in recommendations:
        level = recommendation["recommendation_level"]
        level_counts[level] += 1

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "collaboration_candidates": str(args.candidates),
        },
        "outputs": {
            "collaboration_recommendations": str(args.output),
        },
        "counts": {
            "recommendations_written": len(recommendations),
            "recommendation_levels": level_counts,
        },
    }

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    with args.summary_output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()