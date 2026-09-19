from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from faculty_recommender.enrichment.openalex import (
    OpenAlexClient,
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

    api_key = os.getenv("OPENALEX_API_KEY", "").strip()

    input_path = (
        project_root
        / "data"
        / "processed"
        / "enrichment"
        / "crossref_sample.jsonl"
    )

    cache_directory = (
        project_root
        / "data"
        / "cache"
        / "publications"
        / "openalex"
    )

    output_directory = (
        project_root
        / "data"
        / "processed"
        / "enrichment"
    )

    records = load_jsonl(input_path)
    output_records = []

    with OpenAlexClient(
        api_key=api_key,
        cache_directory=cache_directory,
        timeout_seconds=30,
        request_delay_seconds=0.25,
        max_retries=4,
        refresh_cache=False,
    ) as client:
        for index, record in enumerate(records, start=1):
            doi = record["doi"]
            openalex_result = client.fetch(doi)

            crossref_abstract = (
                record.get("crossref", {})
                .get("metadata", {})
                .get("abstract", "")
            )

            openalex_abstract = (
                openalex_result.get("metadata", {})
                .get("abstract", "")
            )

            output_records.append(
                {
                    "publication_id": record.get(
                        "publication_id"
                    ),
                    "doi": doi,
                    "crossref_has_abstract": bool(
                        crossref_abstract
                    ),
                    "openalex_fills_missing_abstract": bool(
                        not crossref_abstract
                        and openalex_abstract
                    ),
                    "openalex": openalex_result,
                }
            )

            print(
                f"{index:02d}/{len(records):02d} "
                f"{doi}: {openalex_result['status']}"
            )

    output_path = (
        output_directory
        / "openalex_sample.jsonl"
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

    successful = [
        record
        for record in output_records
        if record["openalex"]["status"] == "success"
    ]

    summary = {
        "requested": len(output_records),
        "status_counts": dict(
            Counter(
                record["openalex"]["status"]
                for record in output_records
            )
        ),
        "successful": len(successful),
        "successful_with_abstract": sum(
            bool(
                record["openalex"]["metadata"].get(
                    "abstract"
                )
            )
            for record in successful
        ),
        "successful_with_topics": sum(
            bool(
                record["openalex"]["metadata"].get(
                    "topics"
                )
            )
            for record in successful
        ),
        "successful_with_keywords": sum(
            bool(
                record["openalex"]["metadata"].get(
                    "keywords"
                )
            )
            for record in successful
        ),
        "missing_crossref_abstracts_filled": sum(
            record["openalex_fills_missing_abstract"]
            for record in output_records
        ),
        "output_path": str(output_path),
        "cache_directory": str(cache_directory),
    }

    summary_path = (
        output_directory
        / "openalex_sample_summary.json"
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