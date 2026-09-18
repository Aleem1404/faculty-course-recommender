from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object in {path} at line {line_number}"
                )

            records.append(record)

    return records


def blank(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        return not value.strip()

    if isinstance(value, (list, dict)):
        return len(value) == 0

    return False


def duplicate_values(
    records: list[dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    counter = Counter(
        str(record.get(field, "")).strip()
        for record in records
        if not blank(record.get(field))
    )

    return [
        {"value": value, "count": count}
        for value, count in counter.items()
        if count > 1
    ]


def missing_count(
    records: list[dict[str, Any]],
    field: str,
) -> int:
    return sum(
        blank(record.get(field))
        for record in records
    )


def validate_processed_data(
    processed_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    staff = load_jsonl(processed_directory / "staff.jsonl")
    courses = load_jsonl(processed_directory / "courses.jsonl")
    modules = load_jsonl(processed_directory / "modules.jsonl")

    course_module_links = load_jsonl(
        processed_directory / "course_module_links.jsonl"
    )

    publications = load_jsonl(
        processed_directory / "publications.jsonl"
    )

    staff_publication_links = load_jsonl(
        processed_directory / "staff_publication_links.jsonl"
    )

    teaching_evidence = load_jsonl(
        processed_directory / "teaching_evidence.jsonl"
    )

    staff_ids = {
        record["staff_id"]
        for record in staff
        if not blank(record.get("staff_id"))
    }

    course_ids = {
        record["course_id"]
        for record in courses
        if not blank(record.get("course_id"))
    }

    module_ids = {
        record["module_id"]
        for record in modules
        if not blank(record.get("module_id"))
    }

    publication_ids = {
        record["publication_id"]
        for record in publications
        if not blank(record.get("publication_id"))
    }

    module_codes = {
        str(record.get("module_code", "")).strip().upper()
        for record in modules
        if not blank(record.get("module_code"))
    }

    teaching_codes = {
        str(code).strip().upper()
        for record in teaching_evidence
        for code in record.get("module_codes", [])
        if str(code).strip()
    }

    invalid_course_module_course_ids = sorted({
        record.get("course_id", "")
        for record in course_module_links
        if record.get("course_id") not in course_ids
    })

    invalid_course_module_module_ids = sorted({
        record.get("module_id", "")
        for record in course_module_links
        if record.get("module_id") not in module_ids
    })

    invalid_publication_staff_ids = sorted({
        record.get("staff_id", "")
        for record in staff_publication_links
        if record.get("staff_id") not in staff_ids
    })

    invalid_publication_ids = sorted({
        record.get("publication_id", "")
        for record in staff_publication_links
        if record.get("publication_id") not in publication_ids
    })

    invalid_teaching_staff_ids = sorted({
        record.get("staff_id", "")
        for record in teaching_evidence
        if record.get("staff_id") not in staff_ids
    })

    matched_teaching_codes = teaching_codes.intersection(module_codes)
    unmatched_teaching_codes = teaching_codes.difference(module_codes)

    modules_without_course_links = sorted(
        module_ids.difference({
            record.get("module_id")
            for record in course_module_links
        })
    )

    courses_without_module_links = sorted(
        course_ids.difference({
            record.get("course_id")
            for record in course_module_links
        })
    )

    checks = {
        "staff_ids_unique": (
            len(duplicate_values(staff, "staff_id")) == 0
        ),
        "course_ids_unique": (
            len(duplicate_values(courses, "course_id")) == 0
        ),
        "module_ids_unique": (
            len(duplicate_values(modules, "module_id")) == 0
        ),
        "publication_ids_unique": (
            len(
                duplicate_values(
                    publications,
                    "publication_id",
                )
            )
            == 0
        ),
        "course_module_references_valid": (
            not invalid_course_module_course_ids
            and not invalid_course_module_module_ids
        ),
        "staff_publication_references_valid": (
            not invalid_publication_staff_ids
            and not invalid_publication_ids
        ),
        "teaching_staff_references_valid": (
            not invalid_teaching_staff_ids
        ),
        "all_modules_linked_to_courses": (
            not modules_without_course_links
        ),
    }

    report = {
        "counts": {
            "staff": len(staff),
            "courses": len(courses),
            "modules": len(modules),
            "course_module_links": len(course_module_links),
            "publications": len(publications),
            "staff_publication_links": len(
                staff_publication_links
            ),
            "teaching_evidence": len(teaching_evidence),
        },
        "checks": checks,
        "staff_quality": {
            "missing_full_name": missing_count(
                staff,
                "full_name",
            ),
            "missing_department": missing_count(
                staff,
                "department_name",
            ),
            "missing_college": missing_count(
                staff,
                "college_name",
            ),
            "missing_matching_text": missing_count(
                staff,
                "matching_text_without_teaching",
            ),
            "missing_profile_url": missing_count(
                staff,
                "profile_url",
            ),
        },
        "module_quality": {
            "missing_title": missing_count(
                modules,
                "module_title",
            ),
            "missing_description": missing_count(
                modules,
                "description",
            ),
            "missing_matching_text": missing_count(
                modules,
                "matching_text",
            ),
            "missing_department": missing_count(
                modules,
                "departments",
            ),
            "missing_module_code": missing_count(
                modules,
                "module_code",
            ),
            "modules_without_course_links": len(
                modules_without_course_links
            ),
        },
        "relationship_errors": {
            "invalid_course_ids": (
                invalid_course_module_course_ids
            ),
            "invalid_module_ids": (
                invalid_course_module_module_ids
            ),
            "invalid_publication_staff_ids": (
                invalid_publication_staff_ids
            ),
            "invalid_publication_ids": (
                invalid_publication_ids
            ),
            "invalid_teaching_staff_ids": (
                invalid_teaching_staff_ids
            ),
            "courses_without_module_links": (
                courses_without_module_links
            ),
        },
        "teaching_code_analysis": {
            "unique_teaching_codes": len(teaching_codes),
            "unique_current_module_codes": len(module_codes),
            "matched_teaching_codes": len(
                matched_teaching_codes
            ),
            "unmatched_teaching_codes": len(
                unmatched_teaching_codes
            ),
            "matched_code_values": sorted(
                matched_teaching_codes
            ),
            "unmatched_code_values": sorted(
                unmatched_teaching_codes
            ),
        },
        "validation_passed": all(checks.values()),
    }

    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_directory
        / "processed_data_validation.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return report