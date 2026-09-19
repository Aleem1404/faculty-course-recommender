from __future__ import annotations

import hashlib
import html
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup


RETRYABLE_STATUS_CODES = {
    429,
    500,
    502,
    503,
    504,
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalise_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        value = " ".join(
            str(item)
            for item in value
            if item is not None
        )

    text = html.unescape(str(value))
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def first_text(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            text = normalise_text(item)

            if text:
                return text

        return ""

    return normalise_text(value)


def extract_year(message: dict[str, Any]) -> int | None:
    date_fields = (
        "published-print",
        "published-online",
        "published",
        "issued",
        "created",
    )

    for field in date_fields:
        date_value = message.get(field)

        if not isinstance(date_value, dict):
            continue

        date_parts = date_value.get("date-parts")

        if not isinstance(date_parts, list) or not date_parts:
            continue

        first_part = date_parts[0]

        if (
            isinstance(first_part, list)
            and first_part
            and isinstance(first_part[0], int)
        ):
            return first_part[0]

    return None


def extract_authors(
    message: dict[str, Any],
) -> list[dict[str, Any]]:
    authors = []

    for author in message.get("author", []):
        if not isinstance(author, dict):
            continue

        given_name = normalise_text(author.get("given"))
        family_name = normalise_text(author.get("family"))

        full_name = " ".join(
            part
            for part in (given_name, family_name)
            if part
        )

        affiliations = []

        for affiliation in author.get("affiliation", []):
            if not isinstance(affiliation, dict):
                continue

            name = normalise_text(affiliation.get("name"))

            if name:
                affiliations.append(name)

        authors.append(
            {
                "given_name": given_name,
                "family_name": family_name,
                "full_name": full_name,
                "orcid": normalise_text(
                    author.get("ORCID")
                ),
                "affiliations": affiliations,
            }
        )

    return authors


def parse_crossref_message(
    message: dict[str, Any],
) -> dict[str, Any]:
    subjects = [
        normalise_text(subject)
        for subject in message.get("subject", [])
        if normalise_text(subject)
    ]

    licenses = []

    for licence in message.get("license", []):
        if not isinstance(licence, dict):
            continue

        licences_url = normalise_text(licence.get("URL"))

        if licences_url:
            licenses.append(licences_url)

    resource = message.get("resource", {})
    primary_url = ""

    if isinstance(resource, dict):
        primary = resource.get("primary", {})

        if isinstance(primary, dict):
            primary_url = normalise_text(
                primary.get("URL")
            )

    return {
        "doi": normalise_text(
            message.get("DOI")
        ).lower(),
        "title": first_text(message.get("title")),
        "subtitle": first_text(message.get("subtitle")),
        "abstract": normalise_text(
            message.get("abstract")
        ),
        "publication_year": extract_year(message),
        "publication_type": normalise_text(
            message.get("type")
        ),
        "publisher": normalise_text(
            message.get("publisher")
        ),
        "container_title": first_text(
            message.get("container-title")
        ),
        "subjects": list(dict.fromkeys(subjects)),
        "authors": extract_authors(message),
        "reference_count": message.get(
            "reference-count"
        ),
        "is_referenced_by_count": message.get(
            "is-referenced-by-count"
        ),
        "url": normalise_text(message.get("URL")),
        "primary_resource_url": primary_url,
        "licenses": list(dict.fromkeys(licenses)),
        "language": normalise_text(
            message.get("language")
        ),
    }


class CrossrefClient:
    def __init__(
        self,
        email: str,
        cache_directory: Path,
        timeout_seconds: float = 30.0,
        request_delay_seconds: float = 0.25,
        max_retries: int = 4,
        refresh_cache: bool = False,
    ) -> None:
        if not email.strip():
            raise ValueError(
                "CROSSREF_MAILTO is missing from the .env file."
            )

        self.email = email.strip()
        self.cache_directory = cache_directory
        self.timeout_seconds = timeout_seconds
        self.request_delay_seconds = (
            request_delay_seconds
        )
        self.max_retries = max_retries
        self.refresh_cache = refresh_cache

        self.cache_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = httpx.Client(
            base_url="https://api.crossref.org",
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers={
                "Accept": "application/json",
                "User-Agent": (
                    "faculty-course-recommender/0.1 "
                    f"(mailto:{self.email})"
                ),
            },
            params={
                "mailto": self.email,
            },
        )

    def __enter__(self) -> CrossrefClient:
        return self

    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        self.close()

    def close(self) -> None:
        self.client.close()

    def cache_path(self, doi: str) -> Path:
        digest = hashlib.sha256(
            doi.lower().encode("utf-8")
        ).hexdigest()

        return self.cache_directory / f"{digest}.json"

    def read_cache(
        self,
        doi: str,
    ) -> dict[str, Any] | None:
        path = self.cache_path(doi)

        if self.refresh_cache or not path.exists():
            return None

        try:
            return json.loads(
                path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            return None

    def write_cache(
        self,
        doi: str,
        result: dict[str, Any],
    ) -> None:
        path = self.cache_path(doi)
        temporary_path = path.with_suffix(".tmp")

        temporary_path.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(path)

    def fetch(
        self,
        doi: str,
    ) -> dict[str, Any]:
        clean_doi = doi.strip().lower()

        cached_result = self.read_cache(clean_doi)

        if cached_result is not None:
            result = dict(cached_result)
            result["from_cache"] = True
            return result

        encoded_doi = quote(clean_doi, safe="/")
        endpoint = f"/works/{encoded_doi}"

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.get(endpoint)

                if response.status_code == 200:
                    payload = response.json()
                    message = payload.get("message", {})

                    if not isinstance(message, dict):
                        raise ValueError(
                            "Crossref response has no "
                            "valid message object."
                        )

                    metadata = parse_crossref_message(
                        message
                    )

                    result = {
                        "doi": clean_doi,
                        "status": "success",
                        "source": "crossref",
                        "source_endpoint": endpoint,
                        "retrieved_at": utc_timestamp(),
                        "from_cache": False,
                        "http_status": 200,
                        "metadata": metadata,
                    }

                    self.write_cache(clean_doi, result)

                    time.sleep(
                        self.request_delay_seconds
                    )

                    return result

                if response.status_code == 404:
                    result = {
                        "doi": clean_doi,
                        "status": "not_found",
                        "source": "crossref",
                        "source_endpoint": endpoint,
                        "retrieved_at": utc_timestamp(),
                        "from_cache": False,
                        "http_status": 404,
                        "metadata": {},
                    }

                    self.write_cache(clean_doi, result)

                    return result

                if (
                    response.status_code
                    in RETRYABLE_STATUS_CODES
                    and attempt < self.max_retries
                ):
                    retry_after = response.headers.get(
                        "Retry-After"
                    )

                    if (
                        retry_after
                        and retry_after.isdigit()
                    ):
                        wait_seconds = float(retry_after)
                    else:
                        wait_seconds = min(
                            2 ** attempt,
                            30,
                        )

                    time.sleep(wait_seconds)
                    continue

                return {
                    "doi": clean_doi,
                    "status": "http_error",
                    "source": "crossref",
                    "source_endpoint": endpoint,
                    "retrieved_at": utc_timestamp(),
                    "from_cache": False,
                    "http_status": response.status_code,
                    "error": response.text[:500],
                    "metadata": {},
                }

            except (
                httpx.TimeoutException,
                httpx.NetworkError,
            ) as error:
                if attempt < self.max_retries:
                    time.sleep(
                        min(2 ** attempt, 30)
                    )
                    continue

                return {
                    "doi": clean_doi,
                    "status": "network_error",
                    "source": "crossref",
                    "source_endpoint": endpoint,
                    "retrieved_at": utc_timestamp(),
                    "from_cache": False,
                    "http_status": None,
                    "error": str(error),
                    "metadata": {},
                }

            except (
                ValueError,
                json.JSONDecodeError,
            ) as error:
                return {
                    "doi": clean_doi,
                    "status": "invalid_response",
                    "source": "crossref",
                    "source_endpoint": endpoint,
                    "retrieved_at": utc_timestamp(),
                    "from_cache": False,
                    "http_status": 200,
                    "error": str(error),
                    "metadata": {},
                }

        raise RuntimeError(
            f"Unexpected Crossref state for DOI: {clean_doi}"
        )