from __future__ import annotations

import argparse
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
from faculty_recommender.enrichment.merge import (
    create_no_doi_record,
    merge_publication_metadata,
)
from faculty_recommender.enrichment.openalex import (
    OpenAlexClient,
)


FINAL_STATUSES = {
    "enriched",
    "no_doi",
    "metadata_not_found",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from error

            if isinstance(record, dict):
                records.append(record)

    return records


def write_jsonl_atomic(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    temporary_path.replace(path)


def ordered_records(
    publications: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    results = []

    for publication in publications:
        publication_id = str(
            publication.get("publication_id", "")
        ).strip()

        if publication_id in records_by_id:
            results.append(
                records_by_id[publication_id]
            )

    return results


def save_checkpoint(
    output_path: Path,
    publications: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
) -> None:
    write_jsonl_atomic(
        output_path,
        ordered_records(
            publications,
            records_by_id,
        ),
    )


def build_summary(
    publications: list[dict[str, Any]],
    records: list[dict[str, Any]],
    processed_this_run: int,
    output_path: Path,
) -> dict[str, Any]:
    pipeline_statuses = Counter(
        record.get("pipeline_status", "unknown")
        for record in records
    )

    crossref_statuses = Counter(
        record.get("provider_results", {})
        .get("crossref", {})
        .get("status", "unknown")
        for record in records
    )

    openalex_statuses = Counter(
        record.get("provider_results", {})
        .get("openalex", {})
        .get("status", "unknown")
        for record in records
    )

    return {
        "total_publication_records": len(publications),
        "records_in_output": len(records),
        "processed_this_run": processed_this_run,
        "remaining_records": (
            len(publications) - len(records)
        ),
        "pipeline_status_counts": dict(
            pipeline_statuses
        ),
        "crossref_status_counts": dict(
            crossref_statuses
        ),
        "openalex_status_counts": dict(
            openalex_statuses
        ),
        "records_with_abstract": sum(
            bool(record.get("abstract"))
            for record in records
        ),
        "records_with_topics": sum(
            bool(record.get("topic_names"))
            for record in records
        ),
        "records_with_keywords": sum(
            bool(record.get("keyword_names"))
            for record in records
        ),
        "records_with_expertise_text": sum(
            bool(record.get("expertise_text"))
            for record in records
        ),
        "crossref_cache_hits": sum(
            bool(
                record.get("provider_results", {})
                .get("crossref", {})
                .get("from_cache")
            )
            for record in records
        ),
        "openalex_cache_hits": sum(
            bool(
                record.get("provider_results", {})
                .get("openalex", {})
                .get("from_cache")
            )
            for record in records
        ),
        "output_path": str(output_path),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich publication metadata using "
            "Crossref and OpenAlex."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of unfinished records "
            "to process during this run."
        ),
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Save progress after this many records.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    crossref_email = os.getenv(
        "CROSSREF_MAILTO",
        "",
    ).strip()

    openalex_api_key = os.getenv(
        "OPENALEX_API_KEY",
        "",
    ).strip()

    if not crossref_email:
        raise ValueError(
            "CROSSREF_MAILTO is missing from .env."
        )

    if not openalex_api_key:
        raise ValueError(
            "OPENALEX_API_KEY is missing from .env."
        )

    publication_path = (
        project_root
        / "data"
        / "processed"
        / "core"
        / "publications.jsonl"
    )

    output_directory = (
        project_root
        / "data"
        / "processed"
        / "enrichment"
    )

    output_path = (
        output_directory
        / "publication_enrichment.jsonl"
    )

    summary_path = (
        output_directory
        / "publication_enrichment_summary.json"
    )

    crossref_cache = (
        project_root
        / "data"
        / "cache"
        / "publications"
        / "crossref"
    )

    openalex_cache = (
        project_root
        / "data"
        / "cache"
        / "publications"
        / "openalex"
    )

    publications = load_jsonl(publication_path)
    existing_records = load_jsonl(output_path)

    records_by_id = {
        str(record.get("publication_id")): record
        for record in existing_records
        if record.get("publication_id")
    }

    processed_this_run = 0

    with (
        CrossrefClient(
            email=crossref_email,
            cache_directory=crossref_cache,
            timeout_seconds=30,
            request_delay_seconds=0.25,
            max_retries=4,
            refresh_cache=False,
        ) as crossref_client,
        OpenAlexClient(
            api_key=openalex_api_key,
            cache_directory=openalex_cache,
            timeout_seconds=30,
            request_delay_seconds=0.25,
            max_retries=4,
            refresh_cache=False,
        ) as openalex_client,
    ):
        try:
            for publication in publications:
                publication_id = str(
                    publication.get(
                        "publication_id",
                        "",
                    )
                ).strip()

                if not publication_id:
                    continue

                existing = records_by_id.get(
                    publication_id
                )

                if (
                    existing
                    and existing.get("pipeline_status")
                    in FINAL_STATUSES
                ):
                    continue

                if (
                    arguments.limit is not None
                    and processed_this_run
                    >= arguments.limit
                ):
                    break

                doi = extract_doi(publication)

                if not doi:
                    enriched_record = (
                        create_no_doi_record(publication)
                    )
                else:
                    try:
                        crossref_result = (
                            crossref_client.fetch(doi)
                        )

                        openalex_result = (
                            openalex_client.fetch(doi)
                        )

                        enriched_record = (
                            merge_publication_metadata(
                                publication=publication,
                                doi=doi,
                                crossref_result=(
                                    crossref_result
                                ),
                                openalex_result=(
                                    openalex_result
                                ),
                            )
                        )

                    except Exception as error:
                        enriched_record = {
                            "publication_id": (
                                publication_id
                            ),
                            "doi": doi,
                            "pipeline_status": (
                                "retrieval_error"
                            ),
                            "error": str(error),
                        }

                records_by_id[publication_id] = (
                    enriched_record
                )

                processed_this_run += 1

                if (
                    processed_this_run
                    % arguments.checkpoint_every
                    == 0
                ):
                    save_checkpoint(
                        output_path=output_path,
                        publications=publications,
                        records_by_id=records_by_id,
                    )

                    print(
                        f"Processed this run: "
                        f"{processed_this_run}; "
                        f"saved records: "
                        f"{len(records_by_id)}"
                    )

        except KeyboardInterrupt:
            print(
                "\nInterrupted. Saving checkpoint..."
            )

        finally:
            save_checkpoint(
                output_path=output_path,
                publications=publications,
                records_by_id=records_by_id,
            )

    final_records = ordered_records(
        publications,
        records_by_id,
    )

    summary = build_summary(
        publications=publications,
        records=final_records,
        processed_this_run=processed_this_run,
        output_path=output_path,
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
    print()
    print(
        "Publication enrichment run completed."
    )


if __name__ == "__main__":
    main()