from __future__ import annotations

import json
from pathlib import Path

from faculty_recommender.agents import (
    AgentContext,
    DataAgent,
    EvaluationAgent,
    ExplanationAgent,
    GraphAgent,
    MatchingAgent,
    PipelineOrchestrator,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    context = AgentContext(
        project_root=project_root,
        dry_run=False,
    )

    orchestrator = PipelineOrchestrator(
        agents=[
            DataAgent(),
            GraphAgent(),
            MatchingAgent(),
            ExplanationAgent(),
            EvaluationAgent(),
        ]
    )

    report = orchestrator.run(context)

    output_path = (
        project_root
        / "data"
        / "outputs"
        / "agent_pipeline"
        / "agent_pipeline_report.json"
    )

    orchestrator.write_report(
        report=report,
        output_path=output_path,
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()