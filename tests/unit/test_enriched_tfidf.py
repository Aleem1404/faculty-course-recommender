from faculty_recommender.retrieval.enriched_tfidf import (
    EnrichedTfidfFacultyRanker,
)


def test_enriched_tfidf_uses_publication_evidence() -> None:
    staff = [
        {
            "staff_id": "technical-ai",
            "full_name": "Technical AI Researcher",
            "position": "Lecturer",
            "department_name": "Computer Science",
            "college_name": "CEDPS",
            "profile_url": "",
            "core_profile_text": "computer science",
            "publication_title_text": (
                "Deep neural networks"
            ),
            "publication_abstract_text": (
                "Machine learning neural network "
                "classification and intelligent systems"
            ),
            "publication_topic_text": (
                "Artificial Intelligence "
                "Machine Learning"
            ),
            "publication_keyword_text": (
                "Deep Learning"
            ),
            "publication_count": 1,
            "enriched_publication_count": 1,
        },
        {
            "staff_id": "law",
            "full_name": "AI Law Researcher",
            "position": "Lecturer",
            "department_name": "Law",
            "college_name": "CALS",
            "profile_url": "",
            "core_profile_text": (
                "artificial intelligence law regulation"
            ),
            "publication_title_text": (
                "Legal governance of automated systems"
            ),
            "publication_abstract_text": (
                "Law policy regulation and governance"
            ),
            "publication_topic_text": (
                "Law Political Science"
            ),
            "publication_keyword_text": (
                "Regulation Governance"
            ),
            "publication_count": 1,
            "enriched_publication_count": 1,
        },
    ]

    modules = [
        {
            "module_id": "module-ai",
            "module_code": "CS0001",
            "module_title": "Artificial Intelligence",
            "departments": ["Computer Science"],
            "colleges": ["CEDPS"],
            "matching_text": (
                "artificial intelligence machine learning "
                "neural networks intelligent systems"
            ),
        }
    ]

    ranker = EnrichedTfidfFacultyRanker(
        top_k=2,
    )

    ranker.fit(
        staff_records=staff,
        module_records=modules,
    )

    result = ranker.rank_module(0)

    assert result["recommendations"][0][
        "staff_id"
    ] == "technical-ai"

    assert result["recommendations"][0][
        "enriched_tfidf_score"
    ] > result["recommendations"][1][
        "enriched_tfidf_score"
    ]