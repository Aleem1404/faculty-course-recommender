from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from faculty_recommender.graph.graph_features import (
    compute_graph_features,
)


SEMANTIC_WEIGHT = 0.8
GRAPH_WEIGHT = 0.2


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


class M3KnowledgeGraphReranker:
    def __init__(
        self,
        module_topic_records: list[dict[str, Any]],
        staff_topic_records: list[dict[str, Any]],
        semantic_weight: float = SEMANTIC_WEIGHT,
        graph_weight: float = GRAPH_WEIGHT,
    ) -> None:
        self.module_topics_by_id = {
            str(record["module_id"]): record
            for record in module_topic_records
            if record.get("module_id")
        }

        self.staff_topics_by_id = {
            str(record["staff_id"]): record
            for record in staff_topic_records
            if record.get("staff_id")
        }

        self.semantic_weight = semantic_weight
        self.graph_weight = graph_weight

    def rerank_candidate(
        self,
        module_result: dict[str, Any],
        recommendation: dict[str, Any],
    ) -> dict[str, Any]:
        module_id = str(
            module_result.get("module_id")
        )
        staff_id = str(
            recommendation.get("staff_id")
        )

        module_topic_record = (
            self.module_topics_by_id.get(module_id)
        )
        staff_topic_record = (
            self.staff_topics_by_id.get(staff_id)
        )

        if (
            module_topic_record is None
            or staff_topic_record is None
        ):
            graph_features = None
            graph_support_score = 0.0
            explanation = {
                "module_id": module_id,
                "staff_id": staff_id,
                "top_shared_topics": [],
                "same_department": False,
                "same_college": False,
                "explanation_text": (
                    "Graph topic record unavailable"
                ),
            }
        else:
            graph_features = compute_graph_features(
                module_topic_record=module_topic_record,
                staff_topic_record=staff_topic_record,
            )
            graph_support_score = (
                graph_features.graph_support_score
            )
            explanation = (
                graph_features.explanation
            )

        semantic_score = float(
            recommendation.get("semantic_score", 0.0)
        )

        hybrid_score = round(
            (
                self.semantic_weight
                * semantic_score
            )
            + (
                self.graph_weight
                * graph_support_score
            ),
            6,
        )

        enriched = dict(recommendation)
        enriched["m3_kg_score"] = hybrid_score
        enriched["m3_kg_components"] = {
            "semantic_score": round(
                semantic_score,
                6,
            ),
            "graph_support_score": round(
                graph_support_score,
                6,
            ),
            "semantic_weight": self.semantic_weight,
            "graph_weight": self.graph_weight,
        }

        if graph_features is None:
            enriched["graph_features"] = {
                "shared_topic_count": 0,
                "shared_topics": [],
                "strongest_shared_topics": [],
                "same_department": False,
                "same_college": False,
            }
        else:
            enriched["graph_features"] = {
                "shared_topic_count": (
                    graph_features.shared_topic_count
                ),
                "shared_topics": (
                    graph_features.shared_topics
                ),
                "strongest_shared_topics": (
                    graph_features
                    .strongest_shared_topics
                ),
                "same_department": (
                    graph_features.same_department
                ),
                "same_college": (
                    graph_features.same_college
                ),
            }

        enriched["graph_explanation"] = explanation

        return enriched

    def rerank_module_result(
        self,
        module_result: dict[str, Any],
    ) -> dict[str, Any]:
        recommendations = module_result.get(
            "recommendations",
            [],
        )

        reranked = [
            self.rerank_candidate(
                module_result=module_result,
                recommendation=recommendation,
            )
            for recommendation in recommendations
        ]

        reranked.sort(
            key=lambda item: (
                -float(item["m3_kg_score"]),
                -float(item.get("semantic_score", 0.0)),
                item.get("staff_id", ""),
            )
        )

        for rank, recommendation in enumerate(
            reranked,
            start=1,
        ):
            recommendation["rank"] = rank

        result = dict(module_result)
        result["model"] = "M3-KG"
        result["base_model"] = "M2-H"
        result["kg_reranking"] = {
            "semantic_weight": (
                self.semantic_weight
            ),
            "graph_weight": (
                self.graph_weight
            ),
        }
        result["recommendations"] = reranked

        return result

    def rerank_all(
        self,
        module_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            self.rerank_module_result(
                module_result
            )
            for module_result in module_results
        ]

    @staticmethod
    def summarise(
        reranked_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        status_counts = Counter(
            result.get(
                "decision_status",
                "unknown",
            )
            for result in reranked_results
        )

        recommendation_count = sum(
            len(result.get("recommendations", []))
            for result in reranked_results
        )

        modules_with_graph_overlap = 0
        total_candidates_with_shared_topics = 0

        for result in reranked_results:
            any_overlap = False

            for recommendation in result.get(
                "recommendations",
                [],
            ):
                shared_topic_count = (
                    recommendation
                    .get("graph_features", {})
                    .get("shared_topic_count", 0)
                )

                if shared_topic_count > 0:
                    total_candidates_with_shared_topics += 1
                    any_overlap = True

            if any_overlap:
                modules_with_graph_overlap += 1

        return {
            "module_results": len(
                reranked_results
            ),
            "status_counts": dict(
                status_counts
            ),
            "total_recommendations": (
                recommendation_count
            ),
            "modules_with_graph_overlap": (
                modules_with_graph_overlap
            ),
            "candidates_with_shared_topics": (
                total_candidates_with_shared_topics
            ),
        }