from faculty_recommender.enrichment.merge import (
    create_no_doi_record,
    merge_publication_metadata,
)


def test_merge_uses_crossref_abstract() -> None:
    publication = {
        "publication_id": "publication-1",
        "title": "Original publication title",
        "year": "2024",
    }

    crossref_result = {
        "status": "success",
        "source_endpoint": "/works/example",
        "retrieved_at": "2026-09-19T00:00:00Z",
        "from_cache": False,
        "http_status": 200,
        "metadata": {
            "title": "Crossref title",
            "abstract": "Crossref abstract",
            "subjects": [],
        },
    }

    openalex_result = {
        "status": "success",
        "source_endpoint": "/works/doi:example",
        "retrieved_at": "2026-09-19T00:00:00Z",
        "from_cache": False,
        "http_status": 200,
        "metadata": {
            "title": "OpenAlex title",
            "abstract": "OpenAlex abstract",
            "topics": [
                {
                    "name": "Machine Learning",
                    "field": "Computer Science",
                    "subfield": "Artificial Intelligence",
                    "domain": "Physical Sciences",
                }
            ],
            "keywords": [
                {
                    "name": "Neural networks",
                    "score": 0.9,
                }
            ],
        },
    }

    result = merge_publication_metadata(
        publication=publication,
        doi="10.1000/example",
        crossref_result=crossref_result,
        openalex_result=openalex_result,
    )

    assert result["title"] == (
        "Original publication title"
    )

    assert result["abstract"] == (
        "Crossref abstract"
    )

    assert result["abstract_source"] == (
        "crossref"
    )

    assert result["topic_names"] == [
        "Machine Learning"
    ]

    assert "Neural networks" in result[
        "expertise_text"
    ]


def test_openalex_fills_missing_abstract() -> None:
    publication = {
        "publication_id": "publication-2",
        "title": "Example publication",
    }

    crossref_result = {
        "status": "success",
        "metadata": {
            "abstract": "",
            "subjects": [],
        },
    }

    openalex_result = {
        "status": "success",
        "metadata": {
            "abstract": "OpenAlex abstract",
            "topics": [],
            "keywords": [],
        },
    }

    result = merge_publication_metadata(
        publication=publication,
        doi="10.1000/example-2",
        crossref_result=crossref_result,
        openalex_result=openalex_result,
    )

    assert result["abstract"] == (
        "OpenAlex abstract"
    )

    assert result["abstract_source"] == (
        "openalex"
    )


def test_create_no_doi_record() -> None:
    publication = {
        "publication_id": "publication-3",
        "title": "Publication without DOI",
        "year": 2023,
    }

    result = create_no_doi_record(publication)

    assert result["pipeline_status"] == "no_doi"
    assert result["doi"] == ""
    assert result["publication_year"] == 2023
    assert result["expertise_text"] == (
        "Publication without DOI"
    )