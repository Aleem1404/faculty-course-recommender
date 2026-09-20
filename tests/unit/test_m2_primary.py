import numpy as np

from faculty_recommender.ranking.m2_primary import (
    M2PrimaryDeliveryRanker,
    SemanticScoreBundle,
)


def create_bundle() -> SemanticScoreBundle:
    scores = np.asarray(
        [
            [0.60, 0.95, 0.40],
            [0.30, 0.80, 0.70],
        ],
        dtype=np.float32,
    )

    return SemanticScoreBundle(
        staff_ids=[
            "cs-staff",
            "engineering-staff",
            "law-staff",
        ],
        module_ids=[
            "module-ai",
            "module-maths",
        ],
        final_scores=scores,
        field_scores={
            "core": scores,
        },
        field_weights={
            "core": 1.0,
        },
        model_name="test-model",
    )


def test_home_department_is_prioritised() -> None:
    staff = [
        {
            "staff_id": "cs-staff",
            "full_name": "Computer Scientist",
            "position": "Senior Lecturer",
            "department_name": (
                "Computer Science"
            ),
        },
        {
            "staff_id": "engineering-staff",
            "full_name": "Engineer",
            "position": "Professor",
            "department_name": (
                "Electronic and Electrical "
                "Engineering"
            ),
        },
        {
            "staff_id": "law-staff",
            "full_name": "Law Academic",
            "position": "Lecturer",
            "department_name": (
                "Brunel Law School"
            ),
        },
    ]

    modules = [
        {
            "module_id": "module-ai",
            "module_code": "CS0001",
            "module_title": (
                "Artificial Intelligence"
            ),
            "departments": [
                "Department of Computer Science"
            ],
        },
        {
            "module_id": "module-maths",
            "module_code": "MA0001",
            "module_title": "Mathematics",
            "departments": [
                "Department of Mathematics"
            ],
        },
    ]

    ranker = M2PrimaryDeliveryRanker(
        staff_records=staff,
        module_records=modules,
        score_bundle=create_bundle(),
        top_k=5,
    )

    result = ranker.rank_module(
        "module-ai"
    )

    assert result["fallback_used"] is False
    assert len(result["recommendations"]) == 1

    assert result["recommendations"][0][
        "staff_id"
    ] == "cs-staff"

    assert result["recommendations"][0][
        "semantic_score"
    ] == 0.6


def test_external_fallback_when_no_internal_staff() -> None:
    staff = [
        {
            "staff_id": "cs-staff",
            "full_name": "Computer Scientist",
            "position": "Senior Lecturer",
            "department_name": (
                "Computer Science"
            ),
        },
        {
            "staff_id": "engineering-staff",
            "full_name": "Engineer",
            "position": "Professor",
            "department_name": (
                "Electronic and Electrical "
                "Engineering"
            ),
        },
        {
            "staff_id": "law-staff",
            "full_name": "Law Academic",
            "position": "Lecturer",
            "department_name": (
                "Brunel Law School"
            ),
        },
    ]

    modules = [
        {
            "module_id": "module-ai",
            "module_code": "CS0001",
            "module_title": (
                "Artificial Intelligence"
            ),
            "departments": [
                "Department of Computer Science"
            ],
        },
        {
            "module_id": "module-maths",
            "module_code": "MA0001",
            "module_title": "Mathematics",
            "departments": [
                "Department of Mathematics"
            ],
        },
    ]

    ranker = M2PrimaryDeliveryRanker(
        staff_records=staff,
        module_records=modules,
        score_bundle=create_bundle(),
        top_k=2,
    )

    result = ranker.rank_module(
        "module-maths"
    )

    assert result["fallback_used"] is True

    assert result["decision_status"] == (
        "university_fallback_ranked"
    )

    assert result["recommendations"][0][
        "staff_id"
    ] == "engineering-staff"