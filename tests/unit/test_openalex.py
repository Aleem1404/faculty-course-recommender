from faculty_recommender.enrichment.openalex import (
    parse_openalex_work,
    reconstruct_abstract,
)


def test_reconstruct_abstract() -> None:
    inverted_index = {
        "Machine": [0],
        "learning": [1],
        "supports": [2],
        "prediction": [3],
    }

    assert reconstruct_abstract(inverted_index) == (
        "Machine learning supports prediction"
    )


def test_empty_abstract() -> None:
    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({}) == ""


def test_parse_openalex_work() -> None:
    work = {
        "id": "https://openalex.org/W123",
        "doi": "https://doi.org/10.1000/example",
        "display_name": "Artificial Intelligence Study",
        "publication_year": 2025,
        "type": "article",
        "abstract_inverted_index": {
            "Artificial": [0],
            "intelligence": [1],
            "research": [2],
        },
        "topics": [
            {
                "id": "https://openalex.org/T1",
                "display_name": "Artificial Intelligence",
                "score": 0.95,
                "subfield": {
                    "display_name": "Artificial Intelligence"
                },
                "field": {
                    "display_name": "Computer Science"
                },
                "domain": {
                    "display_name": "Physical Sciences"
                },
            }
        ],
        "keywords": [
            {
                "display_name": "Machine learning",
                "score": 0.8,
            }
        ],
        "open_access": {
            "is_oa": True,
        },
    }

    metadata = parse_openalex_work(work)

    assert metadata["doi"] == "10.1000/example"
    assert metadata["publication_year"] == 2025
    assert metadata["abstract"] == (
        "Artificial intelligence research"
    )
    assert metadata["topics"][0]["name"] == (
        "Artificial Intelligence"
    )
    assert metadata["keywords"][0]["name"] == (
        "Machine learning"
    )