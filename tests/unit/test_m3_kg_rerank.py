from __future__ import annotations

from faculty_recommender.ranking.m3_kg_rerank import (
    M3KnowledgeGraphReranker,
)


def test_shared_topics_populated_from_graph_data() -> None:
    module_topics = [
        {
            "module_id": "mod-1",
            "module_code": "CS101",
            "module_title": "Intro to AI",
            "departments": ["Computer Science"],
            "colleges": ["Engineering"],
            "topics": [
                {"topic": "machine learning", "score": 5},
                {"topic": "neural networks", "score": 4},
                {"topic": "robotics", "score": 2},
            ],
            "topic_names": [
                "machine learning",
                "neural networks",
                "robotics",
            ],
        }
    ]

    staff_topics = [
        {
            "staff_id": "staff-ml",
            "full_name": "Dr ML",
            "department_name": "Computer Science",
            "college_name": "Engineering",
            "topics": [
                {"topic": "machine learning", "score": 10},
                {"topic": "neural networks", "score": 8},
                {"topic": "data mining", "score": 6},
            ],
            "topic_names": [
                "machine learning",
                "neural networks",
                "data mining",
            ],
        },
        {
            "staff_id": "staff-other",
            "full_name": "Dr Other",
            "department_name": "Computer Science",
            "college_name": "Engineering",
            "topics": [
                {"topic": "quantum computing", "score": 10},
            ],
            "topic_names": [
                "quantum computing",
            ],
        },
    ]

    reranker = M3KnowledgeGraphReranker(
        module_topic_records=module_topics,
        staff_topic_records=staff_topics,
    )

    module_result = {
        "module_id": "mod-1",
        "decision_status": "hierarchy_internal_ranked",
        "recommendations": [
            {
                "staff_id": "staff-ml",
                "semantic_score": 0.5,
            },
            {
                "staff_id": "staff-other",
                "semantic_score": 0.4,
            },
        ],
    }

    reranked = reranker.rerank_module_result(module_result)
    recs = reranked["recommendations"]

    rec_ml = next(r for r in recs if r["staff_id"] == "staff-ml")
    rec_other = next(r for r in recs if r["staff_id"] == "staff-other")

    # Overlap between mod-1 and staff-ml: machine learning, neural networks
    assert "shared_topics" in rec_ml
    assert rec_ml["shared_topics"] == [
        "machine learning",
        "neural networks",
    ]
    assert rec_ml["explanation"] != ""

    # staff-other has no overlap with mod-1
    assert "shared_topics" in rec_other
    assert rec_other["shared_topics"] == []
