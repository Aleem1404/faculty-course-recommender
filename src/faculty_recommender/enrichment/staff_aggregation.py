from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


MOJIBAKE_MARKERS = (
    "â€™",
    "â€œ",
    "â€",
    "Ã©",
    "Ã",
    "Â",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected an object in {path} at line "
                    f"{line_number}"
                )

            records.append(record)

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


def normalise_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return " ".join(
            normalise_text(item)
            for item in value
            if normalise_text(item)
        )

    text = " ".join(str(value).split()).strip()

    return repair_mojibake(text)


def repair_mojibake(value: str) -> str:
    if not value:
        return ""

    if not any(marker in value for marker in MOJIBAKE_MARKERS):
        return value

    try:
        repaired = value.encode("cp1252").decode("utf-8")

        if repaired:
            return repaired
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    return value


def unique_text_values(values: list[Any]) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = normalise_text(value)

        if not text:
            continue

        key = text.casefold()

        if key in seen:
            continue

        seen.add(key)
        results.append(text)

    return results


def join_text(values: list[Any]) -> str:
    return " ".join(unique_text_values(values)).strip()


def publication_sort_key(
    publication: dict[str, Any],
) -> tuple[int, str]:
    year = publication.get("publication_year")

    try:
        numeric_year = int(year)
    except (TypeError, ValueError):
        numeric_year = 0

    publication_id = normalise_text(
        publication.get("publication_id")
    )

    return (-numeric_year, publication_id)


def aggregate_staff_publications(
    staff_path: Path,
    publication_path: Path,
    link_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    staff_records = load_jsonl(staff_path)
    publication_records = load_jsonl(publication_path)
    link_records = load_jsonl(link_path)

    publication_by_id = {
        record["publication_id"]: record
        for record in publication_records
        if normalise_text(record.get("publication_id"))
    }

    links_by_staff: dict[str, list[str]] = defaultdict(list)
    seen_links: set[tuple[str, str]] = set()

    invalid_link_records = 0
    duplicate_links = 0

    for link in link_records:
        staff_id = normalise_text(link.get("staff_id"))
        publication_id = normalise_text(
            link.get("publication_id")
        )

        if not staff_id or not publication_id:
            invalid_link_records += 1
            continue

        link_key = (staff_id, publication_id)

        if link_key in seen_links:
            duplicate_links += 1
            continue

        seen_links.add(link_key)
        links_by_staff[staff_id].append(publication_id)

    staff_ids = {
        normalise_text(record.get("staff_id"))
        for record in staff_records
        if normalise_text(record.get("staff_id"))
    }

    enriched_staff_records: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []

    unresolved_publication_links = 0
    links_with_unknown_staff = 0
    suspicious_truncated_abstracts = 0
    original_mojibake_records = 0

    staff_with_publications = 0
    staff_with_enriched_publications = 0
    staff_with_publication_abstracts = 0
    staff_with_topics = 0
    staff_with_keywords = 0

    for linked_staff_id in links_by_staff:
        if linked_staff_id not in staff_ids:
            links_with_unknown_staff += len(
                links_by_staff[linked_staff_id]
            )

    for staff in staff_records:
        staff_id = normalise_text(staff.get("staff_id"))

        publication_ids = links_by_staff.get(staff_id, [])

        linked_publications: list[dict[str, Any]] = []

        for publication_id in publication_ids:
            publication = publication_by_id.get(
                publication_id
            )

            if publication is None:
                unresolved_publication_links += 1
                continue

            linked_publications.append(publication)

        linked_publications.sort(
            key=publication_sort_key
        )

        titles = unique_text_values([
            publication.get("title")
            or publication.get("original_title")
            for publication in linked_publications
        ])

        abstracts = unique_text_values([
            publication.get("abstract")
            for publication in linked_publications
        ])

        topic_names = unique_text_values([
            topic
            for publication in linked_publications
            for topic in publication.get("topic_names", [])
        ])

        keyword_names = unique_text_values([
            keyword
            for publication in linked_publications
            for keyword in publication.get(
                "keyword_names",
                [],
            )
        ])

        field_names = unique_text_values([
            field
            for publication in linked_publications
            for field in publication.get("field_names", [])
        ])

        subfield_names = unique_text_values([
            subfield
            for publication in linked_publications
            for subfield in publication.get(
                "subfield_names",
                [],
            )
        ])

        domain_names = unique_text_values([
            domain
            for publication in linked_publications
            for domain in publication.get(
                "domain_names",
                [],
            )
        ])

        expertise_values = unique_text_values([
            publication.get("expertise_text")
            for publication in linked_publications
        ])

        for publication in linked_publications:
            original_values = [
                publication.get("title"),
                publication.get("abstract"),
                publication.get("expertise_text"),
            ]

            if any(
                marker in normalise_text_before_repair(value)
                for value in original_values
                for marker in MOJIBAKE_MARKERS
            ):
                original_mojibake_records += 1

            abstract = normalise_text(
                publication.get("abstract")
            )

            if abstract.endswith(("...", "…")):
                suspicious_truncated_abstracts += 1

            evidence_records.append({
                "staff_id": staff_id,
                "publication_id": publication.get(
                    "publication_id"
                ),
                "doi": publication.get("doi", ""),
                "pipeline_status": publication.get(
                    "pipeline_status",
                    "",
                ),
                "title": normalise_text(
                    publication.get("title")
                    or publication.get("original_title")
                ),
                "abstract": abstract,
                "publication_year": publication.get(
                    "publication_year"
                ),
                "publication_type": normalise_text(
                    publication.get("publication_type")
                ),
                "publisher": normalise_text(
                    publication.get("publisher")
                ),
                "container_title": normalise_text(
                    publication.get("container_title")
                ),
                "topic_names": unique_text_values(
                    publication.get("topic_names", [])
                ),
                "keyword_names": unique_text_values(
                    publication.get("keyword_names", [])
                ),
                "field_names": unique_text_values(
                    publication.get("field_names", [])
                ),
                "subfield_names": unique_text_values(
                    publication.get("subfield_names", [])
                ),
                "expertise_text": normalise_text(
                    publication.get("expertise_text")
                ),
                "field_sources": publication.get(
                    "field_sources",
                    {},
                ),
                "evidence_sources": publication.get(
                    "evidence_sources",
                    [],
                ),
            })

        enriched_publication_count = sum(
            publication.get("pipeline_status") == "enriched"
            for publication in linked_publications
        )

        publications_with_abstract = sum(
            bool(normalise_text(publication.get("abstract")))
            for publication in linked_publications
        )

        publications_with_topics = sum(
            bool(publication.get("topic_names"))
            for publication in linked_publications
        )

        publications_with_keywords = sum(
            bool(publication.get("keyword_names"))
            for publication in linked_publications
        )

        if linked_publications:
            staff_with_publications += 1

        if enriched_publication_count:
            staff_with_enriched_publications += 1

        if publications_with_abstract:
            staff_with_publication_abstracts += 1

        if publications_with_topics:
            staff_with_topics += 1

        if publications_with_keywords:
            staff_with_keywords += 1

        core_profile_text = normalise_text(
            staff.get("matching_text_without_teaching")
        )

        publication_title_text = join_text(titles)
        publication_abstract_text = join_text(abstracts)
        publication_topic_text = join_text(
            topic_names
            + field_names
            + subfield_names
            + domain_names
        )
        publication_keyword_text = join_text(
            keyword_names
        )
        publication_expertise_text = join_text(
            expertise_values
        )

        matching_text_enriched = join_text([
            core_profile_text,
            publication_title_text,
            publication_topic_text,
            publication_keyword_text,
            publication_abstract_text,
        ])

        enriched_record = dict(staff)

        enriched_record.update({
            "core_profile_text": core_profile_text,
            "publication_title_text": (
                publication_title_text
            ),
            "publication_abstract_text": (
                publication_abstract_text
            ),
            "publication_topic_text": (
                publication_topic_text
            ),
            "publication_keyword_text": (
                publication_keyword_text
            ),
            "publication_expertise_text": (
                publication_expertise_text
            ),
            "matching_text_enriched": (
                matching_text_enriched
            ),
            "publication_ids": [
                publication.get("publication_id")
                for publication in linked_publications
            ],
            "publication_count": len(
                linked_publications
            ),
            "enriched_publication_count": (
                enriched_publication_count
            ),
            "publications_with_abstract": (
                publications_with_abstract
            ),
            "publications_with_topics": (
                publications_with_topics
            ),
            "publications_with_keywords": (
                publications_with_keywords
            ),
            "publication_topic_names": topic_names,
            "publication_keyword_names": keyword_names,
            "publication_field_names": field_names,
            "publication_subfield_names": (
                subfield_names
            ),
            "publication_domain_names": domain_names,
            "evidence_availability": {
                "has_core_profile": bool(
                    core_profile_text
                ),
                "has_publications": bool(
                    linked_publications
                ),
                "has_enriched_publications": bool(
                    enriched_publication_count
                ),
                "has_publication_abstracts": bool(
                    publications_with_abstract
                ),
                "has_publication_topics": bool(
                    publications_with_topics
                ),
                "has_publication_keywords": bool(
                    publications_with_keywords
                ),
            },
        })

        enriched_staff_records.append(enriched_record)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    staff_output_path = (
        output_directory / "staff_enriched.jsonl"
    )

    evidence_output_path = (
        output_directory
        / "staff_publication_evidence.jsonl"
    )

    write_jsonl(
        staff_output_path,
        enriched_staff_records,
    )

    write_jsonl(
        evidence_output_path,
        evidence_records,
    )

    summary = {
        "input_staff_records": len(staff_records),
        "output_staff_records": len(
            enriched_staff_records
        ),
        "input_publication_records": len(
            publication_records
        ),
        "input_link_records": len(link_records),
        "unique_link_records": len(seen_links),
        "evidence_records": len(evidence_records),
        "staff_with_publications": (
            staff_with_publications
        ),
        "staff_with_enriched_publications": (
            staff_with_enriched_publications
        ),
        "staff_with_publication_abstracts": (
            staff_with_publication_abstracts
        ),
        "staff_with_publication_topics": (
            staff_with_topics
        ),
        "staff_with_publication_keywords": (
            staff_with_keywords
        ),
        "invalid_link_records": invalid_link_records,
        "duplicate_links": duplicate_links,
        "links_with_unknown_staff": (
            links_with_unknown_staff
        ),
        "unresolved_publication_links": (
            unresolved_publication_links
        ),
        "original_mojibake_records": (
            original_mojibake_records
        ),
        "suspicious_truncated_abstracts": (
            suspicious_truncated_abstracts
        ),
        "staff_output_path": str(staff_output_path),
        "evidence_output_path": str(
            evidence_output_path
        ),
    }

    summary_path = (
        output_directory
        / "staff_enrichment_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return summary


def normalise_text_before_repair(value: Any) -> str:
    if value is None:
        return ""

    return " ".join(str(value).split()).strip()