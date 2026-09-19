from faculty_recommender.enrichment.crossref import (
    extract_authors,
    extract_year,
    normalise_text,
    parse_crossref_message,
)


def test_normalise_crossref_abstract() -> None:
    value = (
        "<jats:p>Machine learning &amp; "
        "computer vision.</jats:p>"
    )

    assert normalise_text(value) == (
        "Machine learning & computer vision."
    )


def test_extract_year() -> None:
    message = {
        "published-online": {
            "date-parts": [[2025, 4, 10]]
        }
    }

    assert extract_year(message) == 2025


def test_extract_authors() -> None:
    message = {
        "author": [
            {
                "given": "Ada",
                "family": "Lovelace",
                "ORCID": (
                    "https://orcid.org/"
                    "0000-0000-0000-0000"
                ),
                "affiliation": [
                    {"name": "Example University"}
                ],
            }
        ]
    }

    authors = extract_authors(message)

    assert authors[0]["full_name"] == "Ada Lovelace"
    assert authors[0]["affiliations"] == [
        "Example University"
    ]


def test_parse_crossref_message() -> None:
    message = {
        "DOI": "10.1000/EXAMPLE",
        "title": ["Artificial Intelligence Research"],
        "abstract": "<p>AI methods and systems.</p>",
        "published": {
            "date-parts": [[2024]]
        },
        "type": "journal-article",
        "subject": [
            "Artificial Intelligence",
            "Computer Science",
        ],
    }

    metadata = parse_crossref_message(message)

    assert metadata["doi"] == "10.1000/example"
    assert metadata["publication_year"] == 2024
    assert metadata["title"] == (
        "Artificial Intelligence Research"
    )
    assert metadata["abstract"] == (
        "AI methods and systems."
    )