from faculty_recommender.ranking.gate import (
    gate_primary_recommendation,
)


def test_gate_passes_relevant_technical_candidate() -> None:
    module = {
        "module_title": "Artificial Intelligence",
        "departments": ["Computer Science"],
    }

    staff = {
        "core_profile_text": (
            "machine learning and intelligent systems"
        ),
        "publication_expertise_text": (
            "deep learning neural networks"
        ),
        "publication_topic_names": [
            "Artificial Intelligence",
            "Machine Learning",
        ],
        "publication_keyword_names": [
            "Neural Networks",
        ],
    }

    recommendation = {
        "semantic_score": 0.42,
        "field_scores": {
            "core": 0.15,
            "publication_titles": 0.11,
            "publication_abstracts": 0.25,
            "publication_topics": 0.20,
            "publication_keywords": 0.17,
        },
    }

    decision = gate_primary_recommendation(
        module_record=module,
        staff_record=staff,
        recommendation=recommendation,
    )

    assert decision.passed is True
    assert decision.reason == "passed"


def test_gate_passes_relevant_law_candidate() -> None:
    module = {
        "module_title": "Contract Law",
        "departments": ["Brunel Law School"],
    }

    staff = {
        "core_profile_text": (
            "commercial law and contract law"
        ),
        "publication_expertise_text": (
            "contractual obligations and remedies"
        ),
        "publication_topic_names": [
            "Contract law",
            "Commercial law",
        ],
        "publication_keyword_names": [
            "Legal remedies",
        ],
    }

    recommendation = {
        "semantic_score": 0.31,
        "field_scores": {
            "core": 0.18,
            "publication_titles": 0.10,
            "publication_abstracts": 0.19,
            "publication_topics": 0.15,
            "publication_keywords": 0.11,
        },
    }

    decision = gate_primary_recommendation(
        module_record=module,
        staff_record=staff,
        recommendation=recommendation,
    )

    assert decision.passed is True
    assert decision.reason == "passed"


def test_gate_rejects_very_low_score() -> None:
    module = {
        "module_title": "Artificial Intelligence",
        "departments": ["Computer Science"],
    }

    staff = {
        "core_profile_text": "research",
        "publication_expertise_text": "",
        "publication_topic_names": [],
        "publication_keyword_names": [],
    }

    recommendation = {
        "semantic_score": 0.04,
        "field_scores": {
            "core": 0.04,
            "publication_titles": 0.00,
            "publication_abstracts": 0.00,
            "publication_topics": 0.00,
            "publication_keywords": 0.00,
        },
    }

    decision = gate_primary_recommendation(
        module_record=module,
        staff_record=staff,
        recommendation=recommendation,
    )

    assert decision.passed is False
    assert decision.reason == "low_total_score"