from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_STAFF_INPUT = Path("data/outputs/m4/staff_topic_profiles.jsonl")
DEFAULT_MODULE_INPUT = Path("data/outputs/m4/module_topic_profiles.jsonl")

DEFAULT_STAFF_OUTPUT = Path(
    "data/outputs/m4/staff_topic_profiles_filtered.jsonl"
)
DEFAULT_MODULE_OUTPUT = Path(
    "data/outputs/m4/module_topic_profiles_filtered.jsonl"
)

DEFAULT_REJECTED_OUTPUT = Path(
    "data/outputs/m4/rejected_topics.jsonl"
)

DEFAULT_SUMMARY_OUTPUT = Path(
    "data/outputs/m4/topic_quality_filter_summary.json"
)


LOCATION_AND_INSTITUTION_TERMS = {
    "brunel",
    "university",
    "college",
    "school",
    "campus",
    "london",
    "uxbridge",
    "pakistan",
    "islamabad",
    "nust",
    "national",
    "department",
    "faculty",
    "institute",
    "institution",
}

LOW_VALUE_TERMS = {
    "research",
    "study",
    "studies",
    "analysis",
    "system",
    "systems",
    "technology",
    "technologies",
    "engineering",
    "engineer",
    "technical",
    "application",
    "applications",
    "development",
    "advanced",
    "introduction",
    "principles",
    "method",
    "methods",
    "approach",
    "approaches",
    "practice",
    "practices",
    "general",
    "fundamentals",
    "overview",
}

MALFORMED_TOKENS = {
    "engi",
    "technol",
    "comput",
    "analysi",
    "generat",
    "develop",
    "manag",
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
                    f"Expected object on line {line_number} in {path}"
                )

            yield record


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def tokenise(topic: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", topic.lower())


def topic_rejection_reason(topic: str) -> str | None:
    cleaned = topic.strip().lower()
    tokens = tokenise(cleaned)

    if not cleaned:
        return "empty_topic"

    if len(tokens) < 2:
        return "too_short_or_generic_single_token"

    if len(tokens) > 6:
        return "too_many_tokens"

    if any(token in LOCATION_AND_INSTITUTION_TERMS for token in tokens):
        return "location_or_institution_term"

    if any(token in MALFORMED_TOKENS for token in tokens):
        return "malformed_token"

    if all(token in LOW_VALUE_TERMS for token in tokens):
        return "generic_low_value_phrase"

    if tokens[0] in {"generation", "resource", "quantitative"}:
        if any(
            token in {"renewable", "energy", "technology"}
            for token in tokens
        ):
            return "likely_broken_phrase_order"

    return None


def filter_profile(
    profile: dict[str, Any],
    rejected_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], int, int]:
    accepted_topics: list[dict[str, Any]] = []
    accepted_count = 0
    rejected_count = 0

    for topic in profile.get("topics", []):
        topic_label = str(topic.get("topic", "")).strip()
        reason = topic_rejection_reason(topic_label)

        if reason is None:
            accepted_topics.append(topic)
            accepted_count += 1
            continue

        rejected_count += 1

        rejected_records.append(
            {
                "source_record_id": profile.get("source_record_id"),
                "source_type": profile.get("source_type"),
                "topic_id": topic.get("topic_id"),
                "topic": topic_label,
                "weight": topic.get("weight"),
                "evidence_score": topic.get("evidence_score"),
                "reason": reason,
            }
        )

    filtered_profile = dict(profile)
    filtered_profile["topics"] = accepted_topics
    filtered_profile["topic_count"] = len(accepted_topics)
    filtered_profile["total_topic_assignments"] = sum(
        int(topic.get("assignment_count", 0))
        for topic in accepted_topics
    )

    return filtered_profile, accepted_count, rejected_count


def filter_profiles(
    input_path: Path,
    output_path: Path,
    rejected_records: list[dict[str, Any]],
) -> dict[str, int]:
    input_profiles = 0
    output_profiles = 0
    empty_profiles = 0
    accepted_topics = 0
    rejected_topics = 0

    filtered_profiles: list[dict[str, Any]] = []

    for profile in read_jsonl(input_path):
        input_profiles += 1

        filtered_profile, accepted, rejected = filter_profile(
            profile=profile,
            rejected_records=rejected_records,
        )

        accepted_topics += accepted
        rejected_topics += rejected

        if filtered_profile["topic_count"] == 0:
            empty_profiles += 1
            continue

        output_profiles += 1
        filtered_profiles.append(filtered_profile)

    write_jsonl(output_path, filtered_profiles)

    return {
        "input_profiles": input_profiles,
        "output_profiles": output_profiles,
        "empty_profiles_removed": empty_profiles,
        "accepted_topics": accepted_topics,
        "rejected_topics": rejected_topics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter low-quality topic phrases from M4 staff and module "
            "topic profiles."
        )
    )

    parser.add_argument(
        "--staff-input",
        type=Path,
        default=DEFAULT_STAFF_INPUT,
    )
    parser.add_argument(
        "--module-input",
        type=Path,
        default=DEFAULT_MODULE_INPUT,
    )
    parser.add_argument(
        "--staff-output",
        type=Path,
        default=DEFAULT_STAFF_OUTPUT,
    )
    parser.add_argument(
        "--module-output",
        type=Path,
        default=DEFAULT_MODULE_OUTPUT,
    )
    parser.add_argument(
        "--rejected-output",
        type=Path,
        default=DEFAULT_REJECTED_OUTPUT,
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.staff_input.exists():
        raise FileNotFoundError(
            f"Staff profile input not found: {args.staff_input}"
        )

    if not args.module_input.exists():
        raise FileNotFoundError(
            f"Module profile input not found: {args.module_input}"
        )

    rejected_records: list[dict[str, Any]] = []

    staff_summary = filter_profiles(
        input_path=args.staff_input,
        output_path=args.staff_output,
        rejected_records=rejected_records,
    )

    module_summary = filter_profiles(
        input_path=args.module_input,
        output_path=args.module_output,
        rejected_records=rejected_records,
    )

    write_jsonl(args.rejected_output, rejected_records)

    rejection_reasons = Counter(
        record["reason"]
        for record in rejected_records
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "staff_profiles": str(args.staff_input),
            "module_profiles": str(args.module_input),
        },
        "outputs": {
            "staff_profiles_filtered": str(args.staff_output),
            "module_profiles_filtered": str(args.module_output),
            "rejected_topics": str(args.rejected_output),
        },
        "staff": staff_summary,
        "modules": module_summary,
        "rejected_topic_reasons": dict(
            sorted(
                rejection_reasons.items(),
                key=lambda item: item[0],
            )
        ),
    }

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    with args.summary_output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()