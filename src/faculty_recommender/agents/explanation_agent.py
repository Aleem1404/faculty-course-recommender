from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from faculty_recommender.agents.base import (
    AgentContext,
    AgentResult,
    PipelineAgent,
)


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                records.append(
                    json.loads(line)
                )

    return records


class ExplanationAgent(PipelineAgent):
    agent_name = "ExplanationAgent"

    def run(
        self,
        context: AgentContext,
    ) -> AgentResult:
        m3_results_path = context.resolve(
            "data",
            "outputs",
            "recommendations",
            "m3_kg",
            "m3_kg_recommendations.jsonl",
        )

        if not m3_results_path.exists():
            raise FileNotFoundError(
                f"Missing M3-KG output: "
                f"{m3_results_path}"
            )

        results = load_jsonl(
            m3_results_path
        )

        validations: list[str] = [
            f"Found {m3_results_path.name}"
        ]
        metrics: dict[str, Any] = {}

        recommendation_count = 0
        explanation_count = 0
        shared_topic_count = 0
        graph_support_positive = 0

        for module_result in results:
            for recommendation in module_result.get(
                "recommendations",
                [],
            ):
                recommendation_count += 1

                explanation = recommendation.get(
                    "explanation",
                    ""
                )

                if explanation:
                    explanation_count += 1

                shared_topics = recommendation.get(
                    "shared_topics",
                    []
                )

                if shared_topics:
                    shared_topic_count += 1

                graph_support = float(
                    recommendation.get(
                        "m3_kg_components",
                        {},
                    ).get(
                        "graph_support_score",
                        0.0,
                    )
                )

                if graph_support > 0:
                    graph_support_positive += 1

        metrics[
            "recommendation_count"
        ] = recommendation_count
        metrics[
            "recommendations_with_explanation"
        ] = explanation_count
        metrics[
            "recommendations_with_shared_topics"
        ] = shared_topic_count
        metrics[
            "recommendations_with_graph_support"
        ] = graph_support_positive

        validations.append(
            "Explanation fields inspected"
        )

        context.set_value(
            "explanation_agent_summary",
            metrics,
        )

        return AgentResult(
            agent_name=self.agent_name,
            status="success",
            started_at="",
            completed_at="",
            message=(
                "M3-KG explanation coverage validated"
            ),
            inputs=[str(m3_results_path)],
            outputs=[],
            validations=validations,
            metrics=metrics,
        )