from faculty_recommender.retrieval.tfidf import (
    TfidfFacultyRanker,
)


def test_tfidf_ranks_relevant_staff_first() -> None:
    staff = [
        {
            "staff_id": "staff-1",
            "full_name": "AI Researcher",
            "position": "Lecturer",
            "department_name": "Computer Science",
            "college_name": "CEDPS",
            "profile_url": "https://example.com/staff-1",
            "matching_text_without_teaching": (
                "machine learning artificial intelligence "
                "neural networks predictive modelling"
            ),
        },
        {
            "staff_id": "staff-2",
            "full_name": "Law Researcher",
            "position": "Lecturer",
            "department_name": "Law",
            "college_name": "CBASS",
            "profile_url": "https://example.com/staff-2",
            "matching_text_without_teaching": (
                "contract law criminal justice legal policy"
            ),
        },
    ]

    modules = [
        {
            "module_id": "module-1",
            "module_code": "CS0001",
            "module_title": "Machine Learning",
            "departments": ["Computer Science"],
            "colleges": ["CEDPS"],
            "description": (
                "Machine learning and neural network methods"
            ),
            "matching_text": (
                "machine learning neural networks "
                "artificial intelligence"
            ),
        }
    ]

    ranker = TfidfFacultyRanker(
        top_k=2,
        shared_terms_count=5,
    )

    ranker.fit(
        staff_records=staff,
        module_records=modules,
    )

    result = ranker.rank_module(0)

    assert result["module_id"] == "module-1"
    assert len(result["recommendations"]) == 2

    assert (
        result["recommendations"][0]["staff_id"]
        == "staff-1"
    )

    assert (
        result["recommendations"][0]["tfidf_score"]
        >
        result["recommendations"][1]["tfidf_score"]
    )

    assert result["recommendations"][0]["shared_terms"]