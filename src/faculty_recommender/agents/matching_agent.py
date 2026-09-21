from __future__ import annotations

import subprocess
import sys
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


def run_python_script(
    project_root: Path,
    script_path: Path,
) -> None:
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(project_root),
        check=True,
    )


class MatchingAgent(PipelineAgent):
    agent_name = "MatchingAgent"

    def run(
        self,
        context: AgentContext,
    ) -> AgentResult:
        script_m2 = context.resolve(
            "scripts",
            "run_m2_hierarchy_primary.py",
        )
        script_m3 = context.resolve(
            "scripts",
            "run_m3_kg_rerank.py",
        )

        output_m2 = context.resolve(
            "data",
            "outputs",
            "recommendations",
            "m2_hierarchy_primary",
            "m2_hierarchy_primary_recommendations.jsonl",
        )

        output_m3 = context.resolve(
            "data",
            "outputs",
            "recommendations",
            "m3_kg",
            "m3_kg_recommendations.jsonl",
        )

        inputs = [
            script_m2,
            script_m3,
        ]

        validations: list[str] = []
        metrics: dict[str, Any] = {}

        for path in inputs:
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing script: {path}"
                )
            validations.append(
                f"Found {path.name}"
            )

        if context.dry_run:
            message = (
                "Dry run only. Matching scripts "
                "were validated but not executed"
            )
        else:
            run_python_script(
                context.project_root,
                script_m2,
            )
            run_python_script(
                context.project_root,
                script_m3,
            )
            message = (
                "M2-H and M3-KG recommendation "
                "pipelines executed"
            )

        outputs: list[str] = []

        for output_path in [output_m2, output_m3]:
            if output_path.exists():
                outputs.append(str(output_path))
                validations.append(
                    f"Produced {output_path.name}"
                )

        if output_m2.exists():
            metrics[
                "m2_module_results"
            ] = load_jsonl_count(output_m2)

        if output_m3.exists():
            metrics[
                "m3_module_results"
            ] = load_jsonl_count(output_m3)

        context.set_value(
            "matching_agent_summary",
            metrics,
        )

        return AgentResult(
            agent_name=self.agent_name,
            status="success",
            started_at="",
            completed_at="",
            message=message,
            inputs=[str(path) for path in inputs],
            outputs=outputs,
            validations=validations,
            metrics=metrics,
        )