from __future__ import annotations
 
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from faculty_recommender.agents.base import (
    AgentContext,
    AgentResult,
    PipelineAgent,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

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

        m2_module_results = 0
        m3_module_results = 0

        if output_m2.exists():
            m2_module_results = load_jsonl_count(output_m2)
            metrics[
                "m2_module_results"
            ] = m2_module_results

        if output_m3.exists():
            m3_module_results = load_jsonl_count(output_m3)
            metrics[
                "m3_module_results"
            ] = m3_module_results

        if m2_module_results > 0:
            metrics["m3_coverage_rate"] = round(
                m3_module_results / m2_module_results,
                4,
            )
            metrics["m3_excluded_modules"] = (
                m2_module_results - m3_module_results
            )

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

        module_topics = load_jsonl(module_topics_path)
        staff_topics = load_jsonl(staff_topics_path)

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

        if module_topics:
            metrics["module_topic_coverage_rate"] = round(
                non_empty_module_topics / len(module_topics),
                4,
            )

        if staff_topics:
            metrics["staff_topic_coverage_rate"] = round(
                non_empty_staff_topics / len(staff_topics),
                4,
            )

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