from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from faculty_recommender.agents.base import (
    AgentContext,
    AgentResult,
    PipelineAgent,
)


def load_jsonl_count(path: Path) -> int:
    count = 0

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                count += 1

    return count


class DataAgent(PipelineAgent):
    agent_name = "DataAgent"

    def run(
        self,
        context: AgentContext,
    ) -> AgentResult:
        inputs = [
            context.resolve(
                "data",
                "processed",
                "core",
                "modules.jsonl",
            ),
            context.resolve(
                "data",
                "processed",
                "enrichment",
                "staff_enriched.jsonl",
            ),
            context.resolve(
                "data",
                "processed",
                "audit",
                "academic_structure",
                "staff_unit_mappings.jsonl",
            ),
            context.resolve(
                "data",
                "processed",
                "audit",
                "academic_structure",
                "module_unit_mappings.jsonl",
            ),
            context.resolve(
                "config",
                "academic_structure.json",
            ),
        ]

        validations: list[str] = []
        metrics: dict[str, Any] = {}

        for path in inputs:
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing required file: {path}"
                )

            validations.append(
                f"Found {path.name}"
            )

        modules_path = inputs[0]
        staff_path = inputs[1]
        staff_mapping_path = inputs[2]
        module_mapping_path = inputs[3]
        hierarchy_path = inputs[4]

        metrics["module_count"] = load_jsonl_count(
            modules_path
        )
        metrics["staff_count"] = load_jsonl_count(
            staff_path
        )
        metrics[
            "staff_mapping_count"
        ] = load_jsonl_count(
            staff_mapping_path
        )
        metrics[
            "module_mapping_count"
        ] = load_jsonl_count(
            module_mapping_path
        )

        hierarchy = json.loads(
            hierarchy_path.read_text(
                encoding="utf-8"
            )
        )

        metrics["hierarchy_root_keys"] = len(
            hierarchy.keys()
        )

        context.set_value(
            "data_agent_summary",
            metrics,
        )

        return AgentResult(
            agent_name=self.agent_name,
            status="success",
            started_at="",
            completed_at="",
            message=(
                "Core processed data and hierarchy "
                "files validated"
            ),
            inputs=[str(path) for path in inputs],
            outputs=[],
            validations=validations,
            metrics=metrics,
        )