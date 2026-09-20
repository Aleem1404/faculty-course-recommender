from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from faculty_recommender.ranking.m3_kg_rerank import (
    M3KnowledgeGraphReranker,
    load_jsonl,
)


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


def write_json(
    path: Path,
    data: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    m2_h_results_path = (
        project_root
        / "data"
        / "outputs"
        / "recommendations"
        / "m2_hierarchy_primary_filtered"
        / "m2_hierarchy_primary_taught_modules.jsonl"
    )

    module_topics_path = (
        project_root
        / "data"
        / "processed"
        / "graph"
        / "module_topics.jsonl"
    )

    staff_topics_path = (
        project_root
        / "data"
        / "processed"
        / "graph"
        / "staff_topics.jsonl"
    )

    output_dir = (
        project_root
        / "data"
        / "outputs"
        / "recommendations"
        / "m3_kg"
    )

    recommendations_output_path = (
        output_dir
        / "m3_kg_recommendations.jsonl"
    )

    summary_output_path = (
        output_dir
        / "m3_kg_summary.json"
    )

    m2_h_results = load_jsonl(
        m2_h_results_path
    )
    module_topic_records = load_jsonl(
        module_topics_path
    )
    staff_topic_records = load_jsonl(
        staff_topics_path
    )

    reranker = M3KnowledgeGraphReranker(
        module_topic_records=module_topic_records,
        staff_topic_records=staff_topic_records,
        semantic_weight=0.8,
        graph_weight=0.2,
    )

    reranked_results = reranker.rerank_all(
        m2_h_results
    )

    summary = reranker.summarise(
        reranked_results
    )

    summary.update({
        "model": "M3-KG",
        "base_model": "M2-H",
        "semantic_weight": 0.8,
        "graph_weight": 0.2,
        "m2_h_results_path": str(
            m2_h_results_path
        ),
        "module_topics_path": str(
            module_topics_path
        ),
        "staff_topics_path": str(
            staff_topics_path
        ),
        "recommendations_output_path": str(
            recommendations_output_path
        ),
    })

    write_jsonl(
        recommendations_output_path,
        reranked_results,
    )
    write_json(
        summary_output_path,
        summary,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()