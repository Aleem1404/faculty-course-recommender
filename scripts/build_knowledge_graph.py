from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from faculty_recommender.graph.build_graph import (
    FacultyCourseGraphBuilder,
    export_graph_json,
    load_jsonl,
)


def write_json(
    path: Path,
    data: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        / "processed"
        / "graph"
    )

    graphml_output_path = (
        output_dir
        / "faculty_course_graph.graphml"
    )

    json_output_path = (
        output_dir
        / "faculty_course_graph.json"
    )

    summary_output_path = (
        output_dir
        / "faculty_course_graph_summary.json"
    )

    module_topic_records = load_jsonl(
        module_topics_path
    )
    staff_topic_records = load_jsonl(
        staff_topics_path
    )

    builder = FacultyCourseGraphBuilder()
    graph = builder.build_from_topic_files(
        module_topic_records=module_topic_records,
        staff_topic_records=staff_topic_records,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    nx.write_graphml(
        graph,
        graphml_output_path,
    )

    graph_json = export_graph_json(graph)
    write_json(
        json_output_path,
        graph_json,
    )

    summary = builder.graph_summary()
    summary.update({
        "module_topic_input_path": str(
            module_topics_path
        ),
        "staff_topic_input_path": str(
            staff_topics_path
        ),
        "graphml_output_path": str(
            graphml_output_path
        ),
        "json_output_path": str(
            json_output_path
        ),
        "module_topic_record_count": len(
            module_topic_records
        ),
        "staff_topic_record_count": len(
            staff_topic_records
        ),
    })

    write_json(
        summary_output_path,
        summary,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()