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


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def run_python_script(
    project_root: Path,
    script_path: Path,
) -> None:
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(project_root),
        check=True,
    )


class EvaluationAgent(PipelineAgent):
    agent_name = "EvaluationAgent"

    def run(
        self,
        context: AgentContext,
    ) -> AgentResult:
        comparison_script = context.resolve(
            "scripts",
            "compare_m2_h_vs_m3_kg.py",
        )
        case_script = context.resolve(
            "scripts",
            "select_m3_case_studies.py",
        )
        expert_script = context.resolve(
            "scripts",
            "build_expert_judgement_sheet.py",
        )
        charts_script = context.resolve(
            "scripts",
            "make_evaluation_charts.py",
        )

        scripts = [
            comparison_script,
            case_script,
            expert_script,
            charts_script,
        ]

        validations: list[str] = []
        metrics: dict[str, Any] = {}

        for script in scripts:
            if not script.exists():
                raise FileNotFoundError(
                    f"Missing evaluation script: "
                    f"{script}"
                )
            validations.append(
                f"Found {script.name}"
            )

        if not context.dry_run:
            for script in scripts:
                run_python_script(
                    context.project_root,
                    script,
                )

        comparison_summary_path = context.resolve(
            "data",
            "outputs",
            "evaluation",
            "m2_vs_m3",
            "m2_h_vs_m3_kg_summary.json",
        )

        case_summary_path = context.resolve(
            "data",
            "outputs",
            "evaluation",
            "case_studies",
            "m3_case_studies_summary.json",
        )

        expert_summary_path = context.resolve(
            "data",
            "outputs",
            "evaluation",
            "expert_judgement",
            "expert_judgement_summary.json",
        )

        chart_summary_path = context.resolve(
            "data",
            "outputs",
            "evaluation",
            "charts",
            "evaluation_chart_summary.json",
        )

        outputs = []

        for path in [
            comparison_summary_path,
            case_summary_path,
            expert_summary_path,
            chart_summary_path,
        ]:
            if path.exists():
                outputs.append(str(path))
                validations.append(
                    f"Produced {path.name}"
                )

        if comparison_summary_path.exists():
            comparison_summary = load_json(
                comparison_summary_path
            )
            metrics[
                "compared_modules"
            ] = comparison_summary.get(
                "compared_modules",
                0,
            )
            metrics[
                "top_rank_changed_modules"
            ] = comparison_summary.get(
                "top_rank_changed_modules",
                0,
            )

        if case_summary_path.exists():
            case_summary = load_json(
                case_summary_path
            )
            metrics[
                "case_studies_selected"
            ] = case_summary.get(
                "selected_case_studies",
                0,
            )

        if expert_summary_path.exists():
            expert_summary = load_json(
                expert_summary_path
            )
            metrics[
                "expert_review_cases"
            ] = expert_summary.get(
                "selected_review_cases",
                0,
            )

        if chart_summary_path.exists():
            chart_summary = load_json(
                chart_summary_path
            )
            metrics[
                "chart_data_rows"
            ] = chart_summary.get(
                "chart_data_rows",
                0,
            )

        context.set_value(
            "evaluation_agent_summary",
            metrics,
        )

        return AgentResult(
            agent_name=self.agent_name,
            status="success",
            started_at="",
            completed_at="",
            message=(
                "Evaluation outputs validated"
                if context.dry_run
                else "Evaluation scripts executed and "
                "outputs validated"
            ),
            inputs=[str(path) for path in scripts],
            outputs=outputs,
            validations=validations,
            metrics=metrics,
        )