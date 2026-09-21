from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from faculty_recommender.agents.base import (
    AgentContext,
    AgentResult,
    PipelineAgent,
)


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


class GraphAgent(PipelineAgent):
    agent_name = "GraphAgent"

    def run(
        self,
        context: AgentContext,
    ) -> AgentResult:
        module_topics_path = context.resolve(
            "data",
            "processed",
            "graph",
            "module_topics.jsonl",
        )
        staff_topics_path = context.resolve(
            "data",
            "processed",
            "graph",
            "staff_topics.jsonl",
        )
        kg_summary_path = context.resolve(
            "data",
            "outputs",
            "recommendations",
            "m3_kg",
            "m3_kg_summary.json",
        )

        inputs = [
            module_topics_path,
            staff_topics_path,
            kg_summary_path,
        ]

        validations: list[str] = []
        metrics: dict[str, Any] = {}

        for path in inputs:
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing graph resource: {path}"
                )
            validations.append(
                f"Found {path.name}"
            )

        module_topics = load_jsonl(
            module_topics_path
        )
        staff_topics = load_jsonl(
            staff_topics_path
        )
        kg_summary = load_json(
            kg_summary_path
        )

        metrics[
            "module_topic_records"
        ] = len(module_topics)
        metrics[
            "staff_topic_records"
        ] = len(staff_topics)
        metrics[
            "kg_summary_keys"
        ] = sorted(
            list(kg_summary.keys())
        )[:20]

        non_empty_module_topics = sum(
            1
            for record in module_topics
            if record.get("topic_names")
        )

        non_empty_staff_topics = sum(
            1
            for record in staff_topics
            if record.get("topic_names")
        )

        metrics[
            "non_empty_module_topics"
        ] = non_empty_module_topics
        metrics[
            "non_empty_staff_topics"
        ] = non_empty_staff_topics

        context.set_value(
            "graph_agent_summary",
            metrics,
        )

        return AgentResult(
            agent_name=self.agent_name,
            status="success",
            started_at="",
            completed_at="",
            message=(
                "Knowledge graph resources validated"
            ),
            inputs=[str(path) for path in inputs],
            outputs=[],
            validations=validations,
            metrics=metrics,
        )