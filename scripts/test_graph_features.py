from __future__ import annotations

import json
from pathlib import Path

from faculty_recommender.graph.graph_features import (
    compute_graph_features,
)


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


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

    module_records = load_jsonl(
        module_topics_path
    )
    staff_records = load_jsonl(
        staff_topics_path
    )

    module_record = module_records[0]
    staff_record = staff_records[0]

    features = compute_graph_features(
        module_topic_record=module_record,
        staff_topic_record=staff_record,
    )

    print(features)


if __name__ == "__main__":
    main()