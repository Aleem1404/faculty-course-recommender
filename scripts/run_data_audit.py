from __future__ import annotations

import json
from pathlib import Path

from faculty_recommender.data.audit import run_audit


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    staff_path = (
        project_root
        / "data"
        / "raw"
        / "staff"
        / "university_faculty_profiles.json"
    )

    course_path = (
        project_root
        / "data"
        / "raw"
        / "courses"
        / "brunel_courses_nested.json"
    )

    output_directory = (
        project_root
        / "data"
        / "processed"
        / "audit"
    )

    report = run_audit(
        staff_path=staff_path,
        course_path=course_path,
        output_directory=output_directory,
    )

    terminal_summary = {
        "staff": {
            "total": report["staff"]["total_staff_records"],
            "unique_ids": report["staff"]["unique_staff_ids"],
            "duplicate_ids": report["staff"][
                "duplicate_staff_id_count"
            ],
            "with_publications": report["staff"][
                "staff_with_publications"
            ],
           "with_teaching_page": report["staff"][
                "staff_with_teaching_page"
            ],
            "with_teaching_evidence": report["staff"][
                "staff_with_teaching_evidence"
            ],
            "with_module_codes_in_teaching": report["staff"][
                "staff_with_module_codes_in_teaching"
            ],
        },
        "courses": {
            "total": report["courses"]["total_course_records"],
            "module_occurrences": report["courses"][
                "total_module_occurrences"
            ],
            "unique_modules": report["courses"][
                "unique_module_entities"
            ],
            "missing_descriptions": report["courses"][
                "missing_module_description"
            ],
            "missing_or_tbc_codes": report["courses"][
                "modules_with_tbc_or_missing_code"
            ],
        },
    }

    print(json.dumps(terminal_summary, indent=2))
    print()
    print("Audit completed.")
    print(
        "Detailed JSON: "
        "data/processed/audit/data_audit.json"
    )
    print(
        "Readable report: "
        "data/processed/audit/data_audit.md"
    )


if __name__ == "__main__":
    main()