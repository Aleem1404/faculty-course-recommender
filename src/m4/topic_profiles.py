from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ASSIGNMENTS_PATH = Path("data/outputs/m4/topic_assignments.jsonl")
DEFAULT_STAFF_OUTPUT_PATH = Path("data/outputs/m4/staff_topic_profiles.jsonl")
DEFAULT_MODULE_OUTPUT_PATH = Path("data/outputs/m4/module_topic_profiles.jsonl")
DEFAULT_SUMMARY_PATH = Path("data/outputs/m4/topic_profiles_summary.json")

VALID_SOURCE_TYPES = {"staff", "module"}


@dataclass
class TopicEvidence:
    topic_id: str
    topic_label: str
    assignment_count: int = 0
    weighted_evidence: float = 0.0
    similarity_sum: float = 0.0
    source_fields: Counter[str] = field(default_factory=Counter)
    original_texts: Counter[str] = field(default_factory=Counter)

    def add(
        self,
        source_field: str,
        original_text: str,
        source_weight: float,
        similarity_to_canonical: float,
    ) -> None:
        self.assignment_count += 1
        self.weighted_evidence += source_weight * similarity_to_canonical
        self.similarity_sum += similarity_to_canonical

        if source_field:
            self.source_fields[source_field] += 1

        if original_text:
            self.original_texts[original_text] += 1

    @property
    def average_similarity(self) -> float:
        if self.assignment_count == 0:
            return 0.0

        return self.similarity_sum / self.assignment_count

    @property
    def source_diversity(self) -> int:
        return len(self.source_fields)

    def evidence_score(self) -> float:
        frequency_bonus = math.log1p(self.assignment_count)
        diversity_bonus = 0.25 * math.log1p(self.source_diversity)

        return self.weighted_evidence + frequency_bonus + diversity_bonus

    def to_dict(self, normalised_weight: float) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "topic": self.topic_label,
            "weight": round(normalised_weight, 6),
            "evidence_score": round(self.evidence_score(), 6),
            "weighted_evidence": round(self.weighted_evidence, 6),
            "assignment_count": self.assignment_count,
            "average_similarity_to_canonical": round(
                self.average_similarity,
                6,
            ),
            "source_diversity": self.source_diversity,
            "source_fields": [
                {
                    "field": field_name,
                    "count": count,
                }
                for field_name, count in self.source_fields.most_common()
            ],
            "original_topic_examples": [
                {
                    "text": text,
                    "count": count,
                }
                for text, count in self.original_texts.most_common(10)
            ],
        }


@dataclass
class TopicProfile:
    source_record_id: str
    source_type: str
    topics: dict[str, TopicEvidence] = field(default_factory=dict)

    def add_topic(
        self,
        topic_id: str,
        topic_label: str,
        source_field: str,
        original_text: str,
        source_weight: float,
        similarity_to_canonical: float,
    ) -> None:
        if topic_id not in self.topics:
            self.topics[topic_id] = TopicEvidence(
                topic_id=topic_id,
                topic_label=topic_label,
            )

        self.topics[topic_id].add(
            source_field=source_field,
            original_text=original_text,
            source_weight=source_weight,
            similarity_to_canonical=similarity_to_canonical,
        )

    def to_dict(self) -> dict[str, Any]:
        ranked_topics = sorted(
            self.topics.values(),
            key=lambda topic: (
                topic.evidence_score(),
                topic.weighted_evidence,
                topic.assignment_count,
                topic.topic_label.lower(),
            ),
            reverse=True,
        )

        maximum_score = max(
            (topic.evidence_score() for topic in ranked_topics),
            default=0.0,
        )

        topics = [
            topic.to_dict(
                normalised_weight=(
                    topic.evidence_score() / maximum_score
                    if maximum_score > 0
                    else 0.0
                )
            )
            for topic in ranked_topics
        ]

        return {
            "source_record_id": self.source_record_id,
            "source_type": self.source_type,
            "topic_count": len(topics),
            "total_topic_assignments": sum(
                topic.assignment_count for topic in ranked_topics
            ),
            "topics": topics,
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
                    f"Invalid JSON on line {line_number} of {path}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {path}"
                )

            yield record


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_required_string(record: dict[str, Any], key: str) -> str:
    value = record.get(key)

    if value is None:
        return ""

    return str(value).strip()


def build_topic_profiles(
    assignments_path: Path,
) -> tuple[
    dict[str, TopicProfile],
    dict[str, TopicProfile],
    dict[str, int],
]:
    staff_profiles: dict[str, TopicProfile] = {}
    module_profiles: dict[str, TopicProfile] = {}

    statistics = {
        "assignment_records_read": 0,
        "staff_assignments_used": 0,
        "module_assignments_used": 0,
        "unknown_entity_type_skipped": 0,
        "incomplete_assignment_skipped": 0,
    }

    for record in read_jsonl(assignments_path):
        statistics["assignment_records_read"] += 1

        source_record_id = get_required_string(record, "source_record_id")
        source_type = get_required_string(record, "source_type").lower()
        canonical_topic_id = get_required_string(
            record,
            "canonical_topic_id",
        )
        canonical_label = get_required_string(record, "canonical_label")

        if source_type not in VALID_SOURCE_TYPES:
            statistics["unknown_entity_type_skipped"] += 1
            continue

        if not source_record_id or not canonical_topic_id or not canonical_label:
            statistics["incomplete_assignment_skipped"] += 1
            continue

        source_field = get_required_string(record, "source_field")
        original_text = get_required_string(record, "original_text")
        source_weight = safe_float(record.get("source_weight"), default=1.0)
        similarity_to_canonical = safe_float(
            record.get("similarity_to_canonical"),
            default=1.0,
        )

        profile_store = (
            staff_profiles
            if source_type == "staff"
            else module_profiles
        )

        if source_record_id not in profile_store:
            profile_store[source_record_id] = TopicProfile(
                source_record_id=source_record_id,
                source_type=source_type,
            )

        profile_store[source_record_id].add_topic(
            topic_id=canonical_topic_id,
            topic_label=canonical_label,
            source_field=source_field,
            original_text=original_text,
            source_weight=source_weight,
            similarity_to_canonical=similarity_to_canonical,
        )

        if source_type == "staff":
            statistics["staff_assignments_used"] += 1
        else:
            statistics["module_assignments_used"] += 1

    return staff_profiles, module_profiles, statistics


def build_summary(
    assignments_path: Path,
    staff_profiles: dict[str, TopicProfile],
    module_profiles: dict[str, TopicProfile],
    statistics: dict[str, int],
    staff_output_path: Path,
    module_output_path: Path,
) -> dict[str, Any]:
    staff_topic_counts = [
        len(profile.topics)
        for profile in staff_profiles.values()
    ]

    module_topic_counts = [
        len(profile.topics)
        for profile in module_profiles.values()
    ]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "topic_assignments": str(assignments_path),
        },
        "counts": {
            **statistics,
            "staff_profiles": len(staff_profiles),
            "module_profiles": len(module_profiles),
            "staff_topics_total": sum(staff_topic_counts),
            "module_topics_total": sum(module_topic_counts),
            "average_topics_per_staff": round(
                sum(staff_topic_counts) / len(staff_topic_counts),
                4,
            )
            if staff_topic_counts
            else 0.0,
            "average_topics_per_module": round(
                sum(module_topic_counts) / len(module_topic_counts),
                4,
            )
            if module_topic_counts
            else 0.0,
        },
        "outputs": {
            "staff_topic_profiles": str(staff_output_path),
            "module_topic_profiles": str(module_output_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build staff and module topic profiles from M4 "
            "topic assignments."
        )
    )

    parser.add_argument(
        "--assignments",
        type=Path,
        default=DEFAULT_ASSIGNMENTS_PATH,
        help="Path to data/outputs/m4/topic_assignments.jsonl",
    )

    parser.add_argument(
        "--staff-output",
        type=Path,
        default=DEFAULT_STAFF_OUTPUT_PATH,
        help="Output path for staff topic profiles.",
    )

    parser.add_argument(
        "--module-output",
        type=Path,
        default=DEFAULT_MODULE_OUTPUT_PATH,
        help="Output path for module topic profiles.",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Output path for topic profile summary.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.assignments.exists():
        raise FileNotFoundError(
            f"Topic assignments file not found: {args.assignments}"
        )

    staff_profiles, module_profiles, statistics = build_topic_profiles(
        assignments_path=args.assignments,
    )

    staff_records = sorted(
        (
            profile.to_dict()
            for profile in staff_profiles.values()
        ),
        key=lambda record: record["source_record_id"],
    )

    module_records = sorted(
        (
            profile.to_dict()
            for profile in module_profiles.values()
        ),
        key=lambda record: record["source_record_id"],
    )

    write_jsonl(args.staff_output, staff_records)
    write_jsonl(args.module_output, module_records)

    summary = build_summary(
        assignments_path=args.assignments,
        staff_profiles=staff_profiles,
        module_profiles=module_profiles,
        statistics=statistics,
        staff_output_path=args.staff_output,
        module_output_path=args.module_output,
    )

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    with args.summary_output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()