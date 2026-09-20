from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from faculty_recommender.policy.hierarchy import (
    AcademicHierarchy,
)

from faculty_recommender.ranking.m2_hierarchy_primary import (
    M2HierarchyPrimaryRanker,
    load_jsonl,
)

from faculty_recommender.ranking.m2_primary import (
    load_semantic_score_bundle,
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


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    hierarchy_path = (
        project_root
        / "config"
        / "academic_structure.json"
    )

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

    mapping_directory = (
        project_root
        / "data"
        / "processed"
        / "audit"
        / "academic_structure"
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

    output_directory = (
        project_root
        / "data"
        / "outputs"
        / "recommendations"
        / "m2_hierarchy_primary"
    )

    hierarchy = AcademicHierarchy.from_file(
        hierarchy_path
    )

    if hierarchy.validation_errors:
        raise ValueError(
            "Hierarchy validation errors found: "
            f"{hierarchy.validation_errors}"
        )

    staff_records = load_jsonl(staff_path)
    module_records = load_jsonl(module_path)

    staff_mappings = load_jsonl(
        mapping_directory
        / "staff_unit_mappings.jsonl"
    )

    module_mappings = load_jsonl(
        mapping_directory
        / "module_unit_mappings.jsonl"
    )

    score_bundle = load_semantic_score_bundle(
        embedding_path=embedding_path,
        index_path=index_path,
    )

    ranker = M2HierarchyPrimaryRanker(
        staff_records=staff_records,
        module_records=module_records,
        staff_mappings=staff_mappings,
        module_mappings=module_mappings,
        hierarchy=hierarchy,
        score_bundle=score_bundle,
        top_k=5,
    )

    results = ranker.rank_all_modules()
    summary = ranker.summarise(results)

    output_path = (
        output_directory
        / "m2_hierarchy_primary_recommendations.jsonl"
    )

    summary_path = (
        output_directory
        / "m2_hierarchy_primary_summary.json"
    )

    gate_rejections_path = (
        output_directory
        / "m2_hierarchy_primary_gate_rejections.jsonl"
    )

    write_jsonl(output_path, results)

    gate_rejections = []

    for result in results:
        for rejected in result.get(
            "rejected_candidates",
            [],
        ):
            gate_rejections.append({
                "module_id": result.get("module_id"),
                "module_code": result.get("module_code"),
                "module_title": result.get("module_title"),
                "module_departments": result.get(
                    "module_departments",
                    [],
                ),
                "module_colleges": result.get(
                    "module_colleges",
                    [],
                ),
                **rejected,
            })

    write_jsonl(
        gate_rejections_path,
        gate_rejections,
    )

    summary.update({
        "model": "M2-H",
        "policy": "hierarchy_internal_only",
        "semantic_model": score_bundle.model_name,
        "top_k": 5,
        "output_path": str(output_path),
        "gate_rejections_path": str(gate_rejections_path),
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