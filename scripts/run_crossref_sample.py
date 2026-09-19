from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from faculty_recommender.enrichment.audit import (
    extract_doi,
)
from faculty_recommender.enrichment.crossref import (
    CrossrefClient,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    email = os.getenv("CROSSREF_MAILTO", "").strip()

    publication_path = (
        project_root
        / "data"
        / "processed"
        / "core"
        / "publications.jsonl"
    )

    cache_directory = (
        project_root
        / "data"
        / "cache"
        / "publications"
        / "crossref"
    )

    output_directory = (
        project_root
        / "data"
        / "processed"
        / "enrichment"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    publications = load_jsonl(publication_path)

    selected_publications = []
    seen_dois = set()

    for publication in publications:
        doi = extract_doi(publication)

        if not doi or doi in seen_dois:
            continue

        seen_dois.add(doi)

        selected_publications.append(
            {
                "publication": publication,
                "doi": doi,
            }
        )

        if len(selected_publications) == 10:
            break

    if not selected_publications:
        raise ValueError(
            "No publication DOI values were found."
        )

    output_records = []

    with CrossrefClient(
        email=email,
        cache_directory=cache_directory,
        timeout_seconds=30,
        request_delay_seconds=0.25,
        max_retries=4,
        refresh_cache=False,
    ) as client:
        for selected in selected_publications:
            publication = selected["publication"]
            doi = selected["doi"]

            result = client.fetch(doi)

            output_records.append(
                {
                    "publication_id": publication.get(
                        "publication_id"
                    ),
                    "original_title": publication.get(
                        "title",
                        publication.get(
                            "publication_title",
                            "",
                        ),
                    ),
                    "doi": doi,
                    "crossref": result,
                }
            )

            print(
                f"{len(output_records):02d}/"
                f"{len(selected_publications):02d} "
                f"{doi}: {result['status']}"
            )

    output_path = (
        output_directory
        / "crossref_sample.jsonl"
    )

    with output_path.open("w", encoding="utf-8") as file:
        for record in output_records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    status_counts = Counter(
        record["crossref"]["status"]
        for record in output_records
    )

    successful_records = [
        record
        for record in output_records
        if record["crossref"]["status"] == "success"
    ]

    with_abstract = sum(
        bool(
            record["crossref"]
            .get("metadata", {})
            .get("abstract")
        )
        for record in successful_records
    )

    with_subjects = sum(
        bool(
            record["crossref"]
            .get("metadata", {})
            .get("subjects")
        )
        for record in successful_records
    )

    summary = {
        "requested": len(output_records),
        "status_counts": dict(status_counts),
        "successful": len(successful_records),
        "successful_with_abstract": with_abstract,
        "successful_with_subjects": with_subjects,
        "cache_directory": str(cache_directory),
        "output_path": str(output_path),
    }

    summary_path = (
        output_directory
        / "crossref_sample_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()