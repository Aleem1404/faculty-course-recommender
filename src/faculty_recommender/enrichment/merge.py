from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue

        if isinstance(value, str):
            if value.strip():
                return value.strip()
        elif value:
            return value

    return ""


def unique_strings(values: list[Any]) -> list[str]:
    results = []
    seen = set()

    for value in values:
        text = clean_text(value)

        if not text:
            continue

        key = text.casefold()

        if key in seen:
            continue

        seen.add(key)
        results.append(text)

    return results


def extract_original_title(
    publication: dict[str, Any],
) -> str:
    return clean_text(
        first_nonempty(
            publication.get("title"),
            publication.get("publication_title"),
            publication.get("name"),
        )
    )


def extract_original_year(
    publication: dict[str, Any],
) -> int | None:
    values = (
        publication.get("year"),
        publication.get("publication_year"),
        publication.get("published_year"),
    )

    for value in values:
        if value is None:
            continue

        match = re.search(r"\b(19|20)\d{2}\b", str(value))

        if match:
            return int(match.group(0))

    return None


def merge_publication_metadata(
    publication: dict[str, Any],
    doi: str,
    crossref_result: dict[str, Any],
    openalex_result: dict[str, Any],
) -> dict[str, Any]:
    crossref_metadata = (
        crossref_result.get("metadata", {})
        if crossref_result.get("status") == "success"
        else {}
    )

    openalex_metadata = (
        openalex_result.get("metadata", {})
        if openalex_result.get("status") == "success"
        else {}
    )

    original_title = extract_original_title(publication)

    title = clean_text(
        first_nonempty(
            original_title,
            crossref_metadata.get("title"),
            openalex_metadata.get("title"),
        )
    )

    crossref_abstract = clean_text(
        crossref_metadata.get("abstract")
    )

    openalex_abstract = clean_text(
        openalex_metadata.get("abstract")
    )

    abstract = first_nonempty(
        crossref_abstract,
        openalex_abstract,
    )

    if crossref_abstract:
        abstract_source = "crossref"
    elif openalex_abstract:
        abstract_source = "openalex"
    else:
        abstract_source = ""

    original_year = extract_original_year(publication)

    publication_year = first_nonempty(
        original_year,
        crossref_metadata.get("publication_year"),
        openalex_metadata.get("publication_year"),
    )

    crossref_subjects = unique_strings(
        crossref_metadata.get("subjects", [])
    )

    topic_records = [
        topic
        for topic in openalex_metadata.get("topics", [])
        if isinstance(topic, dict)
    ]

    topic_names = unique_strings(
        [
            topic.get("name")
            for topic in topic_records
        ]
    )

    field_names = unique_strings(
        [
            topic.get("field")
            for topic in topic_records
        ]
    )

    subfield_names = unique_strings(
        [
            topic.get("subfield")
            for topic in topic_records
        ]
    )

    domain_names = unique_strings(
        [
            topic.get("domain")
            for topic in topic_records
        ]
    )

    keyword_records = [
        keyword
        for keyword in openalex_metadata.get(
            "keywords",
            [],
        )
        if isinstance(keyword, dict)
    ]

    keyword_names = unique_strings(
        [
            keyword.get("name")
            for keyword in keyword_records
        ]
    )

    expertise_parts = unique_strings(
        [
            title,
            abstract,
            *crossref_subjects,
            *topic_names,
            *field_names,
            *subfield_names,
            *domain_names,
            *keyword_names,
        ]
    )

    crossref_status = crossref_result.get(
        "status",
        "not_requested",
    )

    openalex_status = openalex_result.get(
        "status",
        "not_requested",
    )

    if "success" in {
        crossref_status,
        openalex_status,
    }:
        pipeline_status = "enriched"
    elif (
        crossref_status == "not_found"
        and openalex_status == "not_found"
    ):
        pipeline_status = "metadata_not_found"
    else:
        pipeline_status = "retrieval_error"

    evidence_sources = []

    if crossref_status == "success":
        evidence_sources.append(
            {
                "provider": "crossref",
                "endpoint": crossref_result.get(
                    "source_endpoint"
                ),
                "retrieved_at": crossref_result.get(
                    "retrieved_at"
                ),
            }
        )

    if openalex_status == "success":
        evidence_sources.append(
            {
                "provider": "openalex",
                "endpoint": openalex_result.get(
                    "source_endpoint"
                ),
                "retrieved_at": openalex_result.get(
                    "retrieved_at"
                ),
            }
        )

    return {
        "publication_id": publication.get(
            "publication_id"
        ),
        "doi": doi,
        "pipeline_status": pipeline_status,
        "original_title": original_title,
        "title": title,
        "abstract": abstract,
        "abstract_source": abstract_source,
        "publication_year": publication_year,
        "publication_type": first_nonempty(
            publication.get("publication_type"),
            publication.get("type"),
            crossref_metadata.get("publication_type"),
            openalex_metadata.get("work_type"),
        ),
        "publisher": clean_text(
            crossref_metadata.get("publisher")
        ),
        "container_title": clean_text(
            crossref_metadata.get("container_title")
        ),
        "subjects": crossref_subjects,
        "topics": topic_records,
        "topic_names": topic_names,
        "field_names": field_names,
        "subfield_names": subfield_names,
        "domain_names": domain_names,
        "keywords": keyword_records,
        "keyword_names": keyword_names,
        "authors": crossref_metadata.get(
            "authors",
            [],
        ),
        "cited_by_count": first_nonempty(
            openalex_metadata.get("cited_by_count"),
            crossref_metadata.get(
                "is_referenced_by_count"
            ),
        ),
        "is_open_access": openalex_metadata.get(
            "is_open_access",
            False,
        ),
        "expertise_text": " ".join(expertise_parts),
        "field_sources": {
            "title": (
                "original"
                if original_title
                else (
                    "crossref"
                    if crossref_metadata.get("title")
                    else "openalex"
                )
            ),
            "abstract": abstract_source,
            "topics": (
                "openalex" if topic_names else ""
            ),
            "keywords": (
                "openalex" if keyword_names else ""
            ),
        },
        "provider_results": {
            "crossref": {
                "status": crossref_status,
                "from_cache": crossref_result.get(
                    "from_cache",
                    False,
                ),
                "http_status": crossref_result.get(
                    "http_status"
                ),
            },
            "openalex": {
                "status": openalex_status,
                "from_cache": openalex_result.get(
                    "from_cache",
                    False,
                ),
                "http_status": openalex_result.get(
                    "http_status"
                ),
            },
        },
        "evidence_sources": evidence_sources,
        "enriched_at": utc_timestamp(),
    }


def create_no_doi_record(
    publication: dict[str, Any],
) -> dict[str, Any]:
    title = extract_original_title(publication)

    return {
        "publication_id": publication.get(
            "publication_id"
        ),
        "doi": "",
        "pipeline_status": "no_doi",
        "original_title": title,
        "title": title,
        "abstract": "",
        "abstract_source": "",
        "publication_year": extract_original_year(
            publication
        ),
        "publication_type": first_nonempty(
            publication.get("publication_type"),
            publication.get("type"),
        ),
        "publisher": "",
        "container_title": "",
        "subjects": [],
        "topics": [],
        "topic_names": [],
        "field_names": [],
        "subfield_names": [],
        "domain_names": [],
        "keywords": [],
        "keyword_names": [],
        "authors": [],
        "cited_by_count": None,
        "is_open_access": False,
        "expertise_text": title,
        "field_sources": {
            "title": "original" if title else "",
            "abstract": "",
            "topics": "",
            "keywords": "",
        },
        "provider_results": {
            "crossref": {
                "status": "not_requested",
                "from_cache": False,
                "http_status": None,
            },
            "openalex": {
                "status": "not_requested",
                "from_cache": False,
                "http_status": None,
            },
        },
        "evidence_sources": [],
        "enriched_at": utc_timestamp(),
    }