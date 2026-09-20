from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from faculty_recommender.policy.departments import (
    load_department_aliases,
)

from faculty_recommender.ranking.m2_primary import (
    M2PrimaryDeliveryRanker,
    load_semantic_score_bundle,
)


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at "
                    f"line {line_number}"
                ) from error

    return records


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    staff_path = (
        project_root
        / "data"
        / "processed"
        / "enrichment"
        / "staff_enriched.jsonl"
    )

    module_path = (
        project_root
        / "data"
        / "processed"
        / "core"
        / "modules.jsonl"
    )

    embedding_path = (
        project_root
        / "data"
        / "cache"
        / "embeddings"
        / "semantic"
        / "semantic_embeddings.npz"
    )

    index_path = (
        project_root
        / "data"
        / "cache"
        / "embeddings"
        / "semantic"
        / "semantic_embedding_index.json"
    )

    alias_path = (
        project_root
        / "config"
        / "department_aliases.yaml"
    )

    output_directory = (
        project_root
        / "data"
        / "outputs"
        / "recommendations"
        / "m2_primary"
    )

    staff_records = load_jsonl(staff_path)
    module_records = load_jsonl(module_path)

    aliases = load_department_aliases(
        alias_path
    )

    score_bundle = (
        load_semantic_score_bundle(
            embedding_path=embedding_path,
            index_path=index_path,
        )
    )

    ranker = M2PrimaryDeliveryRanker(
        staff_records=staff_records,
        module_records=module_records,
        score_bundle=score_bundle,
        department_aliases=aliases,
        top_k=5,
        # allow_external_fallback=True,
        allow_external_fallback=False,
    )

    results = ranker.rank_all_modules()
    summary = ranker.summarise(results)

    output_path = (
        output_directory
        / "m2_primary_recommendations.jsonl"
    )

    summary_path = (
        output_directory
        / "m2_primary_summary.json"
    )

    write_jsonl(
        output_path,
        results,
    )

    summary.update({
        "model": "M2-P",
        "semantic_model": (
            score_bundle.model_name
        ),
        "policy": "home_department_first",
        "fallback_policy": (
            "only_when_zero_eligible_internal_staff"
        ),
        "top_k": 5,
        "output_path": str(output_path),
    })

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()