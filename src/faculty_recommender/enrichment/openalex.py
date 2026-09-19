from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx


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

    return " ".join(str(value).split()).strip()


def reconstruct_abstract(
    inverted_index: dict[str, list[int]] | None,
) -> str:
    if not inverted_index:
        return ""

    positions = [
        position
        for values in inverted_index.values()
        for position in values
        if isinstance(position, int)
    ]

    if not positions:
        return ""

    words = [""] * (max(positions) + 1)

    for word, word_positions in inverted_index.items():
        for position in word_positions:
            if isinstance(position, int):
                words[position] = word

    return " ".join(word for word in words if word).strip()


def parse_topic(topic: dict[str, Any]) -> dict[str, Any]:
    subfield = topic.get("subfield") or {}
    field = topic.get("field") or {}
    domain = topic.get("domain") or {}

    return {
        "id": normalise_text(topic.get("id")),
        "name": normalise_text(
            topic.get("display_name")
        ),
        "score": topic.get("score"),
        "subfield": normalise_text(
            subfield.get("display_name")
        ),
        "field": normalise_text(
            field.get("display_name")
        ),
        "domain": normalise_text(
            domain.get("display_name")
        ),
    }


def parse_openalex_work(
    work: dict[str, Any],
) -> dict[str, Any]:
    topics = [
        parse_topic(topic)
        for topic in work.get("topics", [])
        if isinstance(topic, dict)
    ]

    keywords = []

    for keyword in work.get("keywords", []):
        if not isinstance(keyword, dict):
            continue

        name = normalise_text(
            keyword.get("display_name")
        )

        if name:
            keywords.append(
                {
                    "name": name,
                    "score": keyword.get("score"),
                }
            )

    primary_topic_raw = work.get("primary_topic")
    primary_topic = {}

    if isinstance(primary_topic_raw, dict):
        primary_topic = parse_topic(primary_topic_raw)

    doi = normalise_text(work.get("doi")).lower()

    if doi.startswith("https://doi.org/"):
        doi = doi.removeprefix("https://doi.org/")

    return {
        "openalex_id": normalise_text(work.get("id")),
        "doi": doi,
        "title": normalise_text(
            work.get("display_name") or work.get("title")
        ),
        "publication_year": work.get(
            "publication_year"
        ),
        "publication_date": normalise_text(
            work.get("publication_date")
        ),
        "work_type": normalise_text(
            work.get("type")
        ),
        "abstract": reconstruct_abstract(
            work.get("abstract_inverted_index")
        ),
        "primary_topic": primary_topic,
        "topics": topics,
        "keywords": keywords,
        "cited_by_count": work.get(
            "cited_by_count"
        ),
        "is_open_access": bool(
            (work.get("open_access") or {}).get("is_oa")
        ),
    }


class OpenAlexClient:
    def __init__(
        self,
        api_key: str,
        cache_directory: Path,
        timeout_seconds: float = 30.0,
        request_delay_seconds: float = 0.25,
        max_retries: int = 4,
        refresh_cache: bool = False,
    ) -> None:
        if not api_key.strip():
            raise ValueError(
                "OPENALEX_API_KEY is missing from .env."
            )

        self.api_key = api_key.strip()
        self.cache_directory = cache_directory
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
            base_url="https://api.openalex.org",
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={
                "Accept": "application/json",
                "User-Agent": (
                    "faculty-course-recommender/0.1"
                ),
            },
            params={
                "api_key": self.api_key,
            },
        )

    def __enter__(self) -> OpenAlexClient:
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

    def fetch(self, doi: str) -> dict[str, Any]:
        clean_doi = doi.strip().lower()

        cached_result = self.read_cache(clean_doi)

        if cached_result is not None:
            result = dict(cached_result)
            result["from_cache"] = True
            return result

        encoded_doi = quote(clean_doi, safe="")
        endpoint = f"/works/doi:{encoded_doi}"

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.get(endpoint)

                if response.status_code == 200:
                    work = response.json()
                    metadata = parse_openalex_work(work)

                    result = {
                        "doi": clean_doi,
                        "status": "success",
                        "source": "openalex",
                        "source_endpoint": endpoint,
                        "retrieved_at": utc_timestamp(),
                        "from_cache": False,
                        "http_status": 200,
                        "metadata": metadata,
                    }

                    self.write_cache(clean_doi, result)
                    time.sleep(self.request_delay_seconds)

                    return result

                if response.status_code == 404:
                    result = {
                        "doi": clean_doi,
                        "status": "not_found",
                        "source": "openalex",
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

                    if retry_after and retry_after.isdigit():
                        wait_seconds = float(retry_after)
                    else:
                        wait_seconds = min(2 ** attempt, 30)

                    time.sleep(wait_seconds)
                    continue

                return {
                    "doi": clean_doi,
                    "status": "http_error",
                    "source": "openalex",
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
                    time.sleep(min(2 ** attempt, 30))
                    continue

                return {
                    "doi": clean_doi,
                    "status": "network_error",
                    "source": "openalex",
                    "source_endpoint": endpoint,
                    "retrieved_at": utc_timestamp(),
                    "from_cache": False,
                    "http_status": None,
                    "error": str(error),
                    "metadata": {},
                }

            except (ValueError, json.JSONDecodeError) as error:
                return {
                    "doi": clean_doi,
                    "status": "invalid_response",
                    "source": "openalex",
                    "source_endpoint": endpoint,
                    "retrieved_at": utc_timestamp(),
                    "from_cache": False,
                    "http_status": 200,
                    "error": str(error),
                    "metadata": {},
                }

        raise RuntimeError(
            f"Unexpected OpenAlex state for DOI: {clean_doi}"
        )