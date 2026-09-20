from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from faculty_recommender.graph.topic_extraction import (
    extract_module_topics,
    extract_staff_topics,
)


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


def build_module_topic_records(
    module_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    topic_records: list[dict[str, Any]] = []

    for module_record in module_records:
        topic_records.append(
            extract_module_topics(module_record)
        )

    return topic_records


def build_staff_topic_records(
    staff_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    topic_records: list[dict[str, Any]] = []

    for staff_record in staff_records:
        topic_records.append(
            extract_staff_topics(staff_record)
        )

    return topic_records


def summarise_topic_records(
    module_topic_records: list[dict[str, Any]],
    staff_topic_records: list[dict[str, Any]],
) -> dict[str, Any]:
    module_topic_count = sum(
        len(record.get("topic_names", []))
        for record in module_topic_records
    )

    staff_topic_count = sum(
        len(record.get("topic_names", []))
        for record in staff_topic_records
    )

    modules_with_topics = sum(
        len(record.get("topic_names", [])) > 0
        for record in module_topic_records
    )

    staff_with_topics = sum(
        len(record.get("topic_names", [])) > 0
        for record in staff_topic_records
    )

    return {
        "module_record_count": len(
            module_topic_records
        ),
        "staff_record_count": len(
            staff_topic_records
        ),
        "modules_with_topics": modules_with_topics,
        "staff_with_topics": staff_with_topics,
        "total_module_topics": module_topic_count,
        "total_staff_topics": staff_topic_count,
        "average_topics_per_module": round(
            module_topic_count
            / len(module_topic_records),
            4,
        )
        if module_topic_records
        else 0.0,
        "average_topics_per_staff": round(
            staff_topic_count
            / len(staff_topic_records),
            4,
        )
        if staff_topic_records
        else 0.0,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    module_input_path = (
        project_root
        / "data"
        / "processed"
        / "core"
        / "modules.jsonl"
    )

    staff_input_path = (
        project_root
        / "data"
        / "processed"
        / "enrichment"
        / "staff_enriched.jsonl"
    )

    output_dir = (
        project_root
        / "data"
        / "processed"
        / "graph"
    )

    module_output_path = (
        output_dir / "module_topics.jsonl"
    )

    staff_output_path = (
        output_dir / "staff_topics.jsonl"
    )

    summary_output_path = (
        output_dir / "topic_extraction_summary.json"
    )

    module_records = load_jsonl(
        module_input_path
    )
    staff_records = load_jsonl(
        staff_input_path
    )

    module_topic_records = (
        build_module_topic_records(
            module_records
        )
    )

    staff_topic_records = (
        build_staff_topic_records(
            staff_records
        )
    )

    write_jsonl(
        module_output_path,
        module_topic_records,
    )
    write_jsonl(
        staff_output_path,
        staff_topic_records,
    )

    summary = summarise_topic_records(
        module_topic_records=module_topic_records,
        staff_topic_records=staff_topic_records,
    )

    summary.update({
        "module_input_path": str(
            module_input_path
        ),
        "staff_input_path": str(
            staff_input_path
        ),
        "module_output_path": str(
            module_output_path
        ),
        "staff_output_path": str(
            staff_output_path
        ),
    })

    summary_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_output_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()