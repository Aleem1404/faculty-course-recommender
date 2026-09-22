from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RECOMMENDATIONS_INPUT = Path(
    "data/outputs/m4/collaboration_recommendations.jsonl"
)

DEFAULT_METADATA_INPUT = Path(
    "data/outputs/m4/entity_metadata.json"
)

DEFAULT_OUTPUT = Path(
    "data/outputs/m4/collaboration_recommendations_enriched.jsonl"
)

DEFAULT_SUMMARY_OUTPUT = Path(
    "data/outputs/m4/collaboration_recommendations_enriched_summary.json"
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


def load_metadata(path: Path) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object in metadata file: {path}"
        )

    staff = payload.get("staff", {})
    modules = payload.get("modules", {})

    if not isinstance(staff, dict):
        raise ValueError(
            "The 'staff' field in entity metadata must be a JSON object."
        )

    if not isinstance(modules, dict):
        raise ValueError(
            "The 'modules' field in entity metadata must be a JSON object."
        )

    return staff, modules


def as_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        values = [
            as_text(item)
            for item in value
            if as_text(item)
        ]
        return ", ".join(values)

    return str(value).strip()


def metadata_value(
    metadata: dict[str, Any] | None,
    key: str,
    fallback: str = "",
) -> str:
    if not metadata:
        return fallback

    value = as_text(metadata.get(key))

    return value or fallback


def display_name(
    staff_id: str,
    staff_metadata: dict[str, Any] | None,
) -> str:
    return metadata_value(
        metadata=staff_metadata,
        key="display_name",
        fallback=staff_id.replace("-", " ").title(),
    )


def module_display_name(
    module_id: str,
    module_metadata: dict[str, Any] | None,
) -> str:
    return metadata_value(
        metadata=module_metadata,
        key="display_name",
        fallback=module_id,
    )


def human_list(values: list[str]) -> str:
    values = [value for value in values if value]

    if not values:
        return ""

    if len(values) == 1:
        return values[0]

    if len(values) == 2:
        return f"{values[0]} and {values[1]}"

    return f"{', '.join(values[:-1])}, and {values[-1]}"


def topic_labels(
    topics: list[dict[str, Any]],
    maximum: int = 3,
) -> list[str]:
    labels: list[str] = []

    for topic in topics[:maximum]:
        label = as_text(topic.get("topic"))

        if label:
            labels.append(label)

    return labels


def percentage(value: Any) -> float:
    try:
        return round(float(value) * 100, 1)
    except (TypeError, ValueError):
        return 0.0


def build_explanation(
    recommendation: dict[str, Any],
    staff_a_name: str,
    staff_b_name: str,
    module_title: str,
) -> str:
    shared_module_topics = recommendation.get(
        "shared_module_topics",
        [],
    )

    complementary_module_topics = recommendation.get(
        "complementary_module_topics",
        [],
    )

    shared_topics = topic_labels(shared_module_topics)

    staff_a_topics = topic_labels(
        [
            topic
            for topic in complementary_module_topics
            if topic.get("provided_by") == "staff_a"
        ]
    )

    staff_b_topics = topic_labels(
        [
            topic
            for topic in complementary_module_topics
            if topic.get("provided_by") == "staff_b"
        ]
    )

    coverage = percentage(
        recommendation.get("combined_module_coverage")
    )

    gain = percentage(
        recommendation.get("complementarity_gain")
    )

    sentences: list[str] = []

    if module_title:
        sentences.append(
            f"For the module '{module_title}', "
            f"{staff_a_name} and {staff_b_name} "
        )
    else:
        sentences.append(
            f"{staff_a_name} and {staff_b_name} "
        )

    opening = sentences.pop()

    if shared_topics:
        sentences.append(
            f"{opening}share relevant expertise in "
            f"{human_list(shared_topics)}."
        )
    else:
        sentences.append(
            f"{opening}provide complementary expertise."
        )

    contributions: list[str] = []

    if staff_a_topics:
        contributions.append(
            f"{staff_a_name} contributes "
            f"{human_list(staff_a_topics)}"
        )

    if staff_b_topics:
        contributions.append(
            f"{staff_b_name} contributes "
            f"{human_list(staff_b_topics)}"
        )

    if contributions:
        sentences.append(f"{human_list(contributions)}.")

    sentences.append(
        f"Together, the pair covers approximately {coverage}% "
        f"of the weighted module topics, adding {gain}% coverage "
        f"beyond the stronger individual candidate."
    )

    return " ".join(sentences)


def replace_id_references(
    text: str,
    staff_a_id: str,
    staff_a_name: str,
    staff_b_id: str,
    staff_b_name: str,
) -> str:
    if not text:
        return ""

    replacements = {
        staff_a_id: staff_a_name,
        staff_b_id: staff_b_name,
    }

    updated = text

    for old_value, new_value in replacements.items():
        if old_value:
            updated = re.sub(
                pattern=re.escape(old_value),
                repl=new_value,
                string=updated,
                flags=re.IGNORECASE,
            )

    return updated


def enrich_recommendation(
    recommendation: dict[str, Any],
    staff_metadata: dict[str, dict[str, Any]],
    module_metadata: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    module_id = as_text(recommendation.get("module_id"))
    staff_a_id = as_text(recommendation.get("staff_a_id"))
    staff_b_id = as_text(recommendation.get("staff_b_id"))

    module_info = module_metadata.get(module_id)
    staff_a_info = staff_metadata.get(staff_a_id)
    staff_b_info = staff_metadata.get(staff_b_id)

    staff_a_name = display_name(
        staff_id=staff_a_id,
        staff_metadata=staff_a_info,
    )

    staff_b_name = display_name(
        staff_id=staff_b_id,
        staff_metadata=staff_b_info,
    )

    module_title = module_display_name(
        module_id=module_id,
        module_metadata=module_info,
    )

    original_explanation = as_text(
        recommendation.get("explanation")
    )

    enriched_explanation = build_explanation(
        recommendation=recommendation,
        staff_a_name=staff_a_name,
        staff_b_name=staff_b_name,
        module_title=module_title,
    )

    enriched_record = {
        **recommendation,
        "module_title": module_title,
        "module_code": metadata_value(
            metadata=module_info,
            key="module_code",
        ),
        "module_level": metadata_value(
            metadata=module_info,
            key="level",
        ),
        "module_department": metadata_value(
            metadata=module_info,
            key="department",
        ),
        "module_url": metadata_value(
            metadata=module_info,
            key="url",
        ),
        "staff_a_name": staff_a_name,
        "staff_a_job_title": metadata_value(
            metadata=staff_a_info,
            key="job_title",
        ),
        "staff_a_school": metadata_value(
            metadata=staff_a_info,
            key="school",
        ),
        "staff_a_department": metadata_value(
            metadata=staff_a_info,
            key="department",
        ),
        "staff_a_profile_url": metadata_value(
            metadata=staff_a_info,
            key="profile_url",
        ),
        "staff_b_name": staff_b_name,
        "staff_b_job_title": metadata_value(
            metadata=staff_b_info,
            key="job_title",
        ),
        "staff_b_school": metadata_value(
            metadata=staff_b_info,
            key="school",
        ),
        "staff_b_department": metadata_value(
            metadata=staff_b_info,
            key="department",
        ),
        "staff_b_profile_url": metadata_value(
            metadata=staff_b_info,
            key="profile_url",
        ),
        "original_explanation": replace_id_references(
            text=original_explanation,
            staff_a_id=staff_a_id,
            staff_a_name=staff_a_name,
            staff_b_id=staff_b_id,
            staff_b_name=staff_b_name,
        ),
        "explanation": enriched_explanation,
    }

    match_status = {
        "module_matched": module_info is not None,
        "staff_a_matched": staff_a_info is not None,
        "staff_b_matched": staff_b_info is not None,
    }

    return enriched_record, match_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich M4 collaboration recommendations with readable "
            "staff names and module metadata."
        )
    )

    parser.add_argument(
        "--recommendations-input",
        type=Path,
        default=DEFAULT_RECOMMENDATIONS_INPUT,
        help="Path to collaboration_recommendations.jsonl.",
    )

    parser.add_argument(
        "--metadata-input",
        type=Path,
        default=DEFAULT_METADATA_INPUT,
        help="Path to entity_metadata.json.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for enriched recommendations JSONL.",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Path for enrichment summary JSON.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.recommendations_input.exists():
        raise FileNotFoundError(
            "Recommendation input file not found: "
            f"{args.recommendations_input}"
        )

    if not args.metadata_input.exists():
        raise FileNotFoundError(
            f"Metadata input file not found: {args.metadata_input}"
        )

    staff_metadata, module_metadata = load_metadata(
        args.metadata_input
    )

    enriched_records: list[dict[str, Any]] = []

    counters: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()

    for recommendation in read_jsonl(args.recommendations_input):
        counters["recommendations_read"] += 1

        enriched_record, match_status = enrich_recommendation(
            recommendation=recommendation,
            staff_metadata=staff_metadata,
            module_metadata=module_metadata,
        )

        enriched_records.append(enriched_record)

        if match_status["module_matched"]:
            counters["module_metadata_matched"] += 1
        else:
            counters["module_metadata_unmatched"] += 1

        if match_status["staff_a_matched"]:
            counters["staff_a_metadata_matched"] += 1
        else:
            counters["staff_a_metadata_unmatched"] += 1

        if match_status["staff_b_matched"]:
            counters["staff_b_metadata_matched"] += 1
        else:
            counters["staff_b_metadata_unmatched"] += 1

        level = as_text(
            enriched_record.get("recommendation_level")
        ) or "unknown"

        level_counts[level] += 1

    enriched_records.sort(
        key=lambda item: (
            as_text(item.get("module_title")).lower(),
            int(item.get("rank", 0)),
            as_text(item.get("staff_a_name")).lower(),
            as_text(item.get("staff_b_name")).lower(),
        )
    )

    write_jsonl(args.output, enriched_records)

    total = counters["recommendations_read"]

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "collaboration_recommendations": str(
                args.recommendations_input
            ),
            "entity_metadata": str(args.metadata_input),
        },
        "outputs": {
            "enriched_collaboration_recommendations": str(
                args.output
            ),
        },
        "counts": {
            "recommendations_read": total,
            "recommendations_written": len(enriched_records),
            "module_metadata_matched": counters[
                "module_metadata_matched"
            ],
            "module_metadata_unmatched": counters[
                "module_metadata_unmatched"
            ],
            "staff_a_metadata_matched": counters[
                "staff_a_metadata_matched"
            ],
            "staff_a_metadata_unmatched": counters[
                "staff_a_metadata_unmatched"
            ],
            "staff_b_metadata_matched": counters[
                "staff_b_metadata_matched"
            ],
            "staff_b_metadata_unmatched": counters[
                "staff_b_metadata_unmatched"
            ],
            "recommendation_levels": dict(
                sorted(level_counts.items())
            ),
        },
        "coverage": {
            "module_metadata_coverage": round(
                counters["module_metadata_matched"] / total,
                6,
            )
            if total
            else 0.0,
            "staff_a_metadata_coverage": round(
                counters["staff_a_metadata_matched"] / total,
                6,
            )
            if total
            else 0.0,
            "staff_b_metadata_coverage": round(
                counters["staff_b_metadata_matched"] / total,
                6,
            )
            if total
            else 0.0,
        },
    }

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