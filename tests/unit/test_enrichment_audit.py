from faculty_recommender.enrichment.audit import (
    clean_doi,
    extract_doi,
)


def test_clean_doi_url() -> None:
    value = "https://doi.org/10.1000/Example.123"

    assert clean_doi(value) == "10.1000/example.123"


def test_extract_doi_from_nested_record() -> None:
    record = {
        "publication_id": "publication-1",
        "metadata": {
            "doi_url": (
                "https://doi.org/10.1016/j.example.2025.100001"
            )
        },
    }

    assert extract_doi(record) == (
        "10.1016/j.example.2025.100001"
    )


def test_extract_doi_returns_empty_string() -> None:
    record = {
        "title": "Publication without a DOI",
    }

    assert extract_doi(record) == ""