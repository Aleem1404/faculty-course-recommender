import json
from pathlib import Path

from faculty_recommender.enrichment.staff_aggregation import (
    aggregate_staff_publications,
    repair_mojibake,
)


def write_jsonl(
    path: Path,
    records: list[dict],
) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )


def test_repair_mojibake() -> None:
    assert repair_mojibake("Shankarâ€™s research") == (
        "Shankar’s research"
    )


def test_aggregate_staff_publications(
    tmp_path: Path,
) -> None:
    staff_path = tmp_path / "staff.jsonl"
    publication_path = tmp_path / "publications.jsonl"
    link_path = tmp_path / "links.jsonl"
    output_directory = tmp_path / "output"

    write_jsonl(
        staff_path,
        [
            {
                "staff_id": "staff-1",
                "full_name": "AI Researcher",
                "matching_text_without_teaching": (
                    "machine learning neural networks"
                ),
            },
            {
                "staff_id": "staff-2",
                "full_name": "Law Researcher",
                "matching_text_without_teaching": (
                    "contract law"
                ),
            },
        ],
    )

    write_jsonl(
        publication_path,
        [
            {
                "publication_id": "publication-1",
                "pipeline_status": "enriched",
                "title": "Deep Learning for Images",
                "abstract": (
                    "A computer vision neural network study."
                ),
                "publication_year": 2025,
                "topic_names": ["Computer Vision"],
                "keyword_names": ["Deep Learning"],
                "field_names": ["Computer Science"],
                "subfield_names": [
                    "Artificial Intelligence"
                ],
                "domain_names": [
                    "Physical Sciences"
                ],
                "expertise_text": (
                    "Deep learning computer vision"
                ),
            }
        ],
    )

    write_jsonl(
        link_path,
        [
            {
                "staff_id": "staff-1",
                "publication_id": "publication-1",
            }
        ],
    )

    summary = aggregate_staff_publications(
        staff_path=staff_path,
        publication_path=publication_path,
        link_path=link_path,
        output_directory=output_directory,
    )

    assert summary["output_staff_records"] == 2
    assert summary["evidence_records"] == 1
    assert summary["unresolved_publication_links"] == 0

    output_records = []

    with (
        output_directory / "staff_enriched.jsonl"
    ).open("r", encoding="utf-8") as file:
        for line in file:
            output_records.append(json.loads(line))

    first_staff = output_records[0]
    second_staff = output_records[1]

    assert first_staff["publication_count"] == 1
    assert "Computer Vision" in first_staff[
        "publication_topic_text"
    ]
    assert "Deep Learning" in first_staff[
        "publication_keyword_text"
    ]

    assert second_staff["publication_count"] == 0
    assert second_staff[
        "publication_abstract_text"
    ] == ""