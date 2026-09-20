from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx


def normalise_text(value: Any) -> str:
    if value is None:
        return ""

    return " ".join(str(value).strip().split())


def make_node_id(
    node_type: str,
    value: str,
) -> str:
    safe = normalise_text(value).casefold()
    safe = safe.replace(" ", "_")
    safe = safe.replace("/", "_")
    safe = safe.replace("\\", "_")
    safe = safe.replace("&", "and")

    return f"{node_type}:{safe}"


def add_node_if_missing(
    graph: nx.MultiDiGraph,
    node_id: str,
    **attributes: Any,
) -> None:
    if graph.has_node(node_id):
        existing = graph.nodes[node_id]

        for key, value in attributes.items():
            current_value = existing.get(key)

            if key not in existing or current_value is None:
                existing[key] = value
                continue

            if current_value == "":
                existing[key] = value
                continue

            if isinstance(current_value, list) and not current_value:
                existing[key] = value
                continue

        return

    graph.add_node(node_id, **attributes)

def add_edge(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
    edge_type: str,
    **attributes: Any,
) -> None:
    graph.add_edge(
        source,
        target,
        key=edge_type,
        edge_type=edge_type,
        **attributes,
    )


class FacultyCourseGraphBuilder:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_college(
        self,
        college_name: str,
    ) -> str | None:
        college_name = normalise_text(college_name)

        if not college_name:
            return None

        node_id = make_node_id(
            "college",
            college_name,
        )

        add_node_if_missing(
            self.graph,
            node_id,
            node_type="college",
            college_name=college_name,
            label=college_name,
        )

        return node_id

    def add_department(
        self,
        department_name: str,
        colleges: list[str] | None = None,
    ) -> str | None:
        department_name = normalise_text(
            department_name
        )

        if not department_name:
            return None

        node_id = make_node_id(
            "department",
            department_name,
        )

        add_node_if_missing(
            self.graph,
            node_id,
            node_type="department",
            department_name=department_name,
            label=department_name,
        )

        for college_name in colleges or []:
            college_id = self.add_college(
                college_name
            )

            if college_id is None:
                continue

            add_edge(
                self.graph,
                node_id,
                college_id,
                "department_in_college",
            )

        return node_id

    def add_topic(
        self,
        topic_name: str,
    ) -> str | None:
        topic_name = normalise_text(topic_name)

        if not topic_name:
            return None

        node_id = make_node_id(
            "topic",
            topic_name,
        )

        add_node_if_missing(
            self.graph,
            node_id,
            node_type="topic",
            topic_name=topic_name,
            label=topic_name,
        )

        return node_id

    def add_module(
        self,
        module_record: dict[str, Any],
    ) -> str | None:
        module_id = normalise_text(
            module_record.get("module_id")
        )

        if not module_id:
            return None

        node_id = make_node_id(
            "module",
            module_id,
        )

        add_node_if_missing(
            self.graph,
            node_id,
            node_type="module",
            module_id=module_id,
            module_code=module_record.get(
                "module_code"
            ),
            module_title=module_record.get(
                "module_title"
            ),
            label=module_record.get(
                "module_title",
                module_id,
            ),
        )

        for department_name in module_record.get(
            "departments",
            [],
        ):
            department_id = self.add_department(
                department_name,
                colleges=module_record.get(
                    "colleges",
                    [],
                ),
            )

            if department_id is None:
                continue

            add_edge(
                self.graph,
                node_id,
                department_id,
                "module_in_department",
            )

        for topic_record in module_record.get(
            "topics",
            [],
        ):
            topic_name = topic_record.get("topic")
            topic_score = topic_record.get(
                "score",
                0,
            )

            topic_id = self.add_topic(topic_name)

            if topic_id is None:
                continue

            add_edge(
                self.graph,
                node_id,
                topic_id,
                "module_has_topic",
                topic_score=int(topic_score),
            )

        return node_id

    def add_staff(
        self,
        staff_record: dict[str, Any],
    ) -> str | None:
        staff_id = normalise_text(
            staff_record.get("staff_id")
        )

        if not staff_id:
            return None

        node_id = make_node_id(
            "staff",
            staff_id,
        )

        add_node_if_missing(
            self.graph,
            node_id,
            node_type="staff",
            staff_id=staff_id,
            full_name=staff_record.get(
                "full_name"
            ),
            position=staff_record.get(
                "position"
            ),
            department_name=staff_record.get(
                "department_name"
            ),
            college_name=staff_record.get(
                "college_name"
            ),
            profile_url=staff_record.get(
                "profile_url"
            ),
            label=staff_record.get(
                "full_name",
                staff_id,
            ),
        )

        department_name = staff_record.get(
            "department_name"
        )
        college_name = staff_record.get(
            "college_name"
        )

        department_id = self.add_department(
            department_name,
            colleges=[college_name]
            if college_name
            else [],
        )

        if department_id is not None:
            add_edge(
                self.graph,
                node_id,
                department_id,
                "staff_in_department",
            )

        for topic_record in staff_record.get(
            "topics",
            [],
        ):
            topic_name = topic_record.get("topic")
            topic_score = topic_record.get(
                "score",
                0,
            )

            topic_id = self.add_topic(topic_name)

            if topic_id is None:
                continue

            add_edge(
                self.graph,
                node_id,
                topic_id,
                "staff_has_topic",
                topic_score=int(topic_score),
            )

        return node_id

    def build_from_topic_files(
        self,
        module_topic_records: list[dict[str, Any]],
        staff_topic_records: list[dict[str, Any]],
    ) -> nx.MultiDiGraph:
        for module_record in module_topic_records:
            self.add_module(module_record)

        for staff_record in staff_topic_records:
            self.add_staff(staff_record)

        return self.graph

    def graph_summary(self) -> dict[str, Any]:
        node_type_counts: dict[str, int] = {}
        edge_type_counts: dict[str, int] = {}

        for _, attributes in self.graph.nodes(
            data=True
        ):
            node_type = attributes.get(
                "node_type",
                "unknown",
            )
            node_type_counts[node_type] = (
                node_type_counts.get(node_type, 0)
                + 1
            )

        for _, _, _, attributes in self.graph.edges(
            keys=True,
            data=True,
        ):
            edge_type = attributes.get(
                "edge_type",
                "unknown",
            )
            edge_type_counts[edge_type] = (
                edge_type_counts.get(edge_type, 0)
                + 1
            )

        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "node_type_counts": node_type_counts,
            "edge_type_counts": edge_type_counts,
        }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    return records


def export_graph_json(
    graph: nx.MultiDiGraph,
) -> dict[str, Any]:
    nodes = []
    edges = []

    for node_id, attributes in graph.nodes(
        data=True
    ):
        nodes.append({
            "id": node_id,
            **attributes,
        })

    for source, target, key, attributes in graph.edges(
        keys=True,
        data=True,
    ):
        edges.append({
            "source": source,
            "target": target,
            "key": key,
            **attributes,
        })

    return {
        "nodes": nodes,
        "edges": edges,
    }