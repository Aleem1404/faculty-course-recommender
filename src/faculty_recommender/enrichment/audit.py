from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DOI_PATTERN = re.compile(
    r"10\.\d{4,9}/[-._;()/:A-Z0-9]+",
    flags=re.IGNORECASE,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def normalise_text(value: Any) -> str:
    if value is None:
        return ""

    return " ".join(str(value).split()).strip()


def walk_values(
    value: Any,
    path: str = "record",
) -> list[tuple[str, Any]]:
    results = []

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            results.append((child_path, child))
            results.extend(walk_values(child, child_path))

    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            results.extend(walk_values(child, child_path))

    return results


def extract_doi(record: dict[str, Any]) -> str:
    preferred_keys = {
        "doi",
        "doi_url",
        "doiurl",
        "digital_object_identifier",
    }

    for path, value in walk_values(record):
        key = path.rsplit(".", maxsplit=1)[-1].lower()

        if key in preferred_keys:
            match = DOI_PATTERN.search(normalise_text(value))

            if match:
                return clean_doi(match.group(0))

    for _, value in walk_values(record):
        if not isinstance(value, str):
            continue

        match = DOI_PATTERN.search(value)

        if match:
            return clean_doi(match.group(0))

    return ""


def clean_doi(value: str) -> str:
    doi = normalise_text(value).lower()

    prefixes = (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
        "doi:",
    )

    for prefix in prefixes:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]

    return doi.rstrip(".,;)]}")


def extract_urls(record: dict[str, Any]) -> list[str]:
    urls = []

    for _, value in walk_values(record):
        if not isinstance(value, str):
            continue

        text = value.strip()

        if text.startswith(("http://", "https://")):
            urls.append(text)

    return list(dict.fromkeys(urls))


def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except ValueError:
        return ""


def find_first(
    record: dict[str, Any],
    candidates: tuple[str, ...],
) -> Any:
    candidate_keys = {
        candidate.lower()
        for candidate in candidates
    }

    for path, value in walk_values(record):
        key = path.rsplit(".", maxsplit=1)[-1].lower()

        if key in candidate_keys and normalise_text(value):
            return value

    return None


def audit_enrichment_readiness(
    core_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    publications = load_jsonl(
        core_directory / "publications.jsonl"
    )

    links = load_jsonl(
        core_directory / "staff_publication_links.jsonl"
    )

    staff = load_jsonl(
        core_directory / "staff.jsonl"
    )

    dois = []
    publication_years: Counter[str] = Counter()
    publication_types: Counter[str] = Counter()
    url_domains: Counter[str] = Counter()
    publication_key_frequency: Counter[str] = Counter()

    publications_with_urls = 0
    publications_with_titles = 0

    for publication in publications:
        publication_key_frequency.update(publication.keys())

        doi = extract_doi(publication)

        if doi:
            dois.append(doi)

        title = find_first(
            publication,
            ("title", "publication_title"),
        )

        if normalise_text(title):
            publications_with_titles += 1

        year = find_first(
            publication,
            ("year", "publication_year", "published_year"),
        )

        if normalise_text(year):
            publication_years[normalise_text(year)] += 1

        publication_type = find_first(
            publication,
            ("type", "publication_type", "work_type"),
        )

        if normalise_text(publication_type):
            publication_types[
                normalise_text(publication_type)
            ] += 1

        urls = extract_urls(publication)

        if urls:
            publications_with_urls += 1

        for url in urls:
            domain = get_domain(url)

            if domain:
                url_domains[domain] += 1

    staff_with_profile_url = 0
    staff_with_any_external_url = 0
    staff_url_domains: Counter[str] = Counter()

    for record in staff:
        profile_url = normalise_text(
            record.get("profile_url")
        )

        if profile_url:
            staff_with_profile_url += 1

        urls = extract_urls(record)

        if urls:
            staff_with_any_external_url += 1

        for url in urls:
            domain = get_domain(url)

            if domain:
                staff_url_domains[domain] += 1

    doi_counter = Counter(dois)

    report = {
        "publication_records": len(publications),
        "staff_publication_links": len(links),
        "staff_records": len(staff),
        "publications_with_titles": publications_with_titles,
        "publications_with_doi": len(dois),
        "unique_dois": len(doi_counter),
        "duplicate_doi_occurrences": sum(
            count - 1
            for count in doi_counter.values()
            if count > 1
        ),
        "publications_without_doi": (
            len(publications) - len(dois)
        ),
        "publications_with_any_url": publications_with_urls,
        "staff_with_profile_url": staff_with_profile_url,
        "staff_with_any_external_url": (
            staff_with_any_external_url
        ),
        "publication_year_counts": dict(
            publication_years.most_common()
        ),
        "publication_type_counts": dict(
            publication_types.most_common()
        ),
        "publication_url_domains": dict(
            url_domains.most_common(30)
        ),
        "staff_url_domains": dict(
            staff_url_domains.most_common(30)
        ),
        "publication_key_frequency": dict(
            publication_key_frequency.most_common()
        ),
        "sample_dois": sorted(doi_counter.keys())[:20],
    }

    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_directory
        / "enrichment_readiness.json"
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return report