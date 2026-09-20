from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from faculty_recommender.policy.hierarchy import (
    AcademicHierarchy,
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
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at "
                    f"line {line_number}"
                ) from error

            records.append(record)

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


def status_counts(
    mappings: list[dict[str, Any]],
) -> dict[str, int]:
    return dict(
        Counter(
            mapping["mapping"]["status"]
            for mapping in mappings
        ).most_common()
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

    output_directory = (
        project_root
        / "data"
        / "processed"
        / "audit"
        / "academic_structure"
    )

    hierarchy = AcademicHierarchy.from_file(
        hierarchy_path
    )

    staff_records = load_jsonl(staff_path)
    module_records = load_jsonl(module_path)

    staff_mappings = []

    for staff in staff_records:
#         mapping = hierarchy.resolve_many(
#     labels=[
#         staff.get(
#             "department_name",
#             "",
#         )
#     ],
#     college_labels=[
#         staff.get(
#             "college_name",
#             "",
#         )
#     ],
# )
        mapping = hierarchy.resolve_many(
            labels=[
                staff.get(
                    "department_name",
                    "",
                )
            ],
            college_labels=[
                staff.get(
                    "college_name",
                    "",
                )
            ],
            source_type="staff_departments",
        )

        staff_mappings.append({
            "staff_id": staff.get("staff_id"),
            "full_name": staff.get(
                "full_name"
            ),
            "position": staff.get("position"),
            "source_college_name": (
                staff.get("college_name")
            ),
            "source_department_name": (
                staff.get("department_name")
            ),
            "mapping": mapping,
        })

    module_mappings = []

    for module in module_records:
        departments = module.get(
            "departments",
            [],
        )

        colleges = module.get(
            "colleges",
            [],
        )

        mapping_source = "departments"

        if not departments:
            departments = colleges
            mapping_source = "college_fallback"

        mapping = hierarchy.resolve_many(
            labels=departments,
            college_labels=colleges,
            source_type="module_departments",
        )

        module_mappings.append({
            "module_id": module.get("module_id"),
            "module_code": module.get(
                "module_code"
            ),
            "module_title": module.get(
                "module_title"
            ),
            "source_colleges": colleges,
            "source_departments": (
                module.get(
                    "departments",
                    [],
                )
            ),
            "mapping_source": (
                mapping_source
            ),
            "mapping": mapping,
        })

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    units_path = (
        output_directory
        / "academic_units.jsonl"
    )

    staff_mapping_path = (
        output_directory
        / "staff_unit_mappings.jsonl"
    )

    module_mapping_path = (
        output_directory
        / "module_unit_mappings.jsonl"
    )

    write_jsonl(
        units_path,
        hierarchy.export_units(),
    )

    write_jsonl(
        staff_mapping_path,
        staff_mappings,
    )

    write_jsonl(
        module_mapping_path,
        module_mappings,
    )

    unresolved_staff_labels = Counter(
        label
        for record in staff_mappings
        for label in record["mapping"][
            "unresolved_labels"
        ]
    )

    unresolved_module_labels = Counter(
        label
        for record in module_mappings
        for label in record["mapping"][
            "unresolved_labels"
        ]
    )

    level_counts = Counter(
        unit.level
        for unit in hierarchy.units.values()
    )

    summary = {
        "hierarchy": {
            "unit_count": len(
                hierarchy.units
            ),
            "level_counts": dict(
                level_counts
            ),
            "validation_errors": (
                hierarchy.validation_errors
            ),
            "alias_collision_count": len(
                hierarchy.alias_collisions()
            ),
            "alias_collisions": (
                hierarchy.alias_collisions()
            ),
        },
        "staff": {
            "record_count": len(
                staff_records
            ),
            "mapping_status_counts": (
                status_counts(
                    staff_mappings
                )
            ),
            "unresolved_label_counts": dict(
                unresolved_staff_labels
                .most_common()
            ),
        },
        "modules": {
            "record_count": len(
                module_records
            ),
            "mapping_status_counts": (
                status_counts(
                    module_mappings
                )
            ),
            "unresolved_label_counts": dict(
                unresolved_module_labels
                .most_common()
            ),
        },
        "outputs": {
            "academic_units": str(
                units_path
            ),
            "staff_mappings": str(
                staff_mapping_path
            ),
            "module_mappings": str(
                module_mapping_path
            ),
        },
    }

    summary_path = (
        output_directory
        / "hierarchy_mapping_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()