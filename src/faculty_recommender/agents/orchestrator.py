from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from faculty_recommender.agents.base import (
    AgentContext,
    AgentResult,
    PipelineAgent,
)


class PipelineOrchestrator:
    def __init__(
        self,
        agents: list[PipelineAgent],
    ) -> None:
        self.agents = agents

    def run(
        self,
        context: AgentContext,
    ) -> dict[str, Any]:
        agent_results: list[AgentResult] = []
        pipeline_status = "success"

        for agent in self.agents:
            result = agent.execute(context)
            agent_results.append(result)

            if not result.succeeded:
                pipeline_status = "failed"
                break

        return {
            "pipeline_status": pipeline_status,
            "dry_run": context.dry_run,
            "agent_results": [
                asdict(result)
                for result in agent_results
            ],
            "context_metadata": context.metadata,
        }

    @staticmethod
    def write_report(
        report: dict[str, Any],
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )