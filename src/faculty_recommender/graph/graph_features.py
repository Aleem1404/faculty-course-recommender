from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def normalise_text(value: Any) -> str:
    if value is None:
        return ""

    return " ".join(str(value).strip().split()).casefold()


def topic_score_map(
    topic_record: dict[str, Any],
) -> dict[str, int]:
    return {
        normalise_text(item.get("topic")): int(
            item.get("score", 0)
        )
        for item in topic_record.get("topics", [])
        if normalise_text(item.get("topic"))
    }


def normalised_string_set(
    values: list[Any] | None,
) -> set[str]:
    if not values:
        return set()

    return {
        normalise_text(value)
        for value in values
        if normalise_text(value)
    }


@dataclass(frozen=True)
class GraphFeatureBundle:
    shared_topic_count: int
    shared_topics: list[str]
    strongest_shared_topics: list[dict[str, Any]]
    same_department: bool
    same_college: bool
    graph_support_score: float
    explanation: dict[str, Any]


def strongest_shared_topics(
    module_topic_record: dict[str, Any],
    staff_topic_record: dict[str, Any],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    module_scores = topic_score_map(
        module_topic_record
    )
    staff_scores = topic_score_map(
        staff_topic_record
    )

    shared = []

    for topic in sorted(
        set(module_scores).intersection(
            set(staff_scores)
        )
    ):
        shared.append({
            "topic": topic,
            "module_score": module_scores[topic],
            "staff_score": staff_scores[topic],
            "combined_score": (
                module_scores[topic]
                + staff_scores[topic]
            ),
        })

    shared.sort(
        key=lambda item: (
            -item["combined_score"],
            item["topic"],
        )
    )

    return shared[:top_n]


def compute_graph_support_score(
    shared_topic_count: int,
    strongest_topics: list[dict[str, Any]],
    same_department: bool,
    same_college: bool,
) -> float:
    shared_topic_component = min(
        shared_topic_count / 5.0,
        1.0,
    ) * 0.5

    strongest_topic_component = min(
        sum(
            item["combined_score"]
            for item in strongest_topics
        )
        / 30.0,
        1.0,
    ) * 0.3

    department_component = (
        0.15 if same_department else 0.0
    )

    college_component = (
        0.05 if same_college else 0.0
    )

    total = (
        shared_topic_component
        + strongest_topic_component
        + department_component
        + college_component
    )

    return round(min(total, 1.0), 6)


def build_explanation(
    module_topic_record: dict[str, Any],
    staff_topic_record: dict[str, Any],
    strongest_topics: list[dict[str, Any]],
    same_department: bool,
    same_college: bool,
) -> dict[str, Any]:
    explanation_parts = []

    if strongest_topics:
        topic_labels = [
            item["topic"]
            for item in strongest_topics[:3]
        ]
        explanation_parts.append(
            "Shared topic evidence: "
            + ", ".join(topic_labels)
        )

    if same_department:
        explanation_parts.append(
            "Same department match"
        )
    elif same_college:
        explanation_parts.append(
            "Same college match"
        )

    return {
        "module_id": module_topic_record.get(
            "module_id"
        ),
        "staff_id": staff_topic_record.get(
            "staff_id"
        ),
        "top_shared_topics": [
            item["topic"]
            for item in strongest_topics[:3]
        ],
        "same_department": same_department,
        "same_college": same_college,
        "explanation_text": "; ".join(
            explanation_parts
        ),
    }


def compute_graph_features(
    module_topic_record: dict[str, Any],
    staff_topic_record: dict[str, Any],
    top_n_topics: int = 5,
) -> GraphFeatureBundle:
    module_topic_names = set(
        topic_score_map(module_topic_record)
    )

    staff_topic_names = set(
        topic_score_map(staff_topic_record)
    )

    shared_topics = sorted(
        module_topic_names.intersection(
            staff_topic_names
        )
    )

    strongest_topics = strongest_shared_topics(
        module_topic_record=module_topic_record,
        staff_topic_record=staff_topic_record,
        top_n=top_n_topics,
    )

    module_departments = normalised_string_set(
        module_topic_record.get("departments", [])
    )

    staff_department = normalise_text(
        staff_topic_record.get(
            "department_name"
        )
    )

    same_department = (
        staff_department in module_departments
        if staff_department
        else False
    )

    module_colleges = normalised_string_set(
        module_topic_record.get("colleges", [])
    )

    staff_college = normalise_text(
        staff_topic_record.get(
            "college_name"
        )
    )

    same_college = (
        staff_college in module_colleges
        if staff_college
        else False
    )

    graph_support_score = (
        compute_graph_support_score(
            shared_topic_count=len(
                shared_topics
            ),
            strongest_topics=strongest_topics,
            same_department=same_department,
            same_college=same_college,
        )
    )

    explanation = build_explanation(
        module_topic_record=module_topic_record,
        staff_topic_record=staff_topic_record,
        strongest_topics=strongest_topics,
        same_department=same_department,
        same_college=same_college,
    )

    return GraphFeatureBundle(
        shared_topic_count=len(shared_topics),
        shared_topics=shared_topics,
        strongest_shared_topics=strongest_topics,
        same_department=same_department,
        same_college=same_college,
        graph_support_score=graph_support_score,
        explanation=explanation,
    )