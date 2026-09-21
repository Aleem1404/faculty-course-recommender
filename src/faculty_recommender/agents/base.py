from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


@dataclass
class AgentContext:
    project_root: Path
    dry_run: bool = False
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def resolve(
        self,
        *parts: str,
    ) -> Path:
        return self.project_root.joinpath(*parts)

    def set_value(
        self,
        key: str,
        value: Any,
    ) -> None:
        self.metadata[key] = value

    def get_value(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        return self.metadata.get(key, default)


@dataclass
class AgentResult:
    agent_name: str
    status: str
    started_at: str
    completed_at: str
    message: str
    inputs: list[str] = field(
        default_factory=list
    )
    outputs: list[str] = field(
        default_factory=list
    )
    validations: list[str] = field(
        default_factory=list
    )
    metrics: dict[str, Any] = field(
        default_factory=dict
    )
    errors: list[str] = field(
        default_factory=list
    )

    @property
    def succeeded(self) -> bool:
        return self.status == "success"


class PipelineAgent(ABC):
    agent_name = "PipelineAgent"

    def execute(
        self,
        context: AgentContext,
    ) -> AgentResult:
        started_at = utc_now_iso()

        try:
            result = self.run(context)
            result.started_at = started_at
            result.completed_at = utc_now_iso()

            if not result.agent_name:
                result.agent_name = self.agent_name

            return result

        except Exception as exc:
            return AgentResult(
                agent_name=self.agent_name,
                status="failed",
                started_at=started_at,
                completed_at=utc_now_iso(),
                message=(
                    f"{self.agent_name} failed"
                ),
                errors=[str(exc)],
            )

    @abstractmethod
    def run(
        self,
        context: AgentContext,
    ) -> AgentResult:
        raise NotImplementedError