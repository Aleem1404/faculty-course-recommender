from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def load_json_records(path: Path, possible_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = []

        for key in possible_keys:
            candidate = payload.get(key)

            if isinstance(candidate, list):
                records = candidate
                break
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")

    if not records:
        raise ValueError(f"No records found in {path}")

    invalid_records = [
        index for index, record in enumerate(records)
        if not isinstance(record, dict)
    ]

    if invalid_records:
        raise ValueError(
            f"{path} contains non-object records at indexes: "
            f"{invalid_records[:10]}"
        )

    return records


def normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def normalise_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return " ".join(value.split()).strip()

    return str(value).strip()


def is_blank(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        return not value.strip()

    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0

    return False


def direct_value(data: Any, *aliases: str) -> Any:
    if not isinstance(data, dict):
        return None

    alias_keys = {normalise_key(alias) for alias in aliases}

    for key, value in data.items():
        if normalise_key(key) in alias_keys:
            return value

    return None


def profile_value(record: dict[str, Any], *aliases: str) -> Any:
    value = direct_value(record, *aliases)

    if not is_blank(value):
        return value

    profile = direct_value(record, "profile")

    if isinstance(profile, dict):
        value = direct_value(profile, *aliases)

        if not is_blank(value):
            return value

    return None


def deep_find_first(data: Any, *aliases: str) -> Any:
    alias_keys = {normalise_key(alias) for alias in aliases}

    if isinstance(data, dict):
        for key, value in data.items():
            if normalise_key(key) in alias_keys and not is_blank(value):
                return value

        for value in data.values():
            result = deep_find_first(value, *aliases)

            if not is_blank(result):
                return result

    elif isinstance(data, list):
        for item in data:
            result = deep_find_first(item, *aliases)

            if not is_blank(result):
                return result

    return None


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value

    if is_blank(value):
        return []

    return [value]


def string_values(value: Any) -> list[str]:
    values = []

    for item in as_list(value):
        text = normalise_text(item)

        if text:
            values.append(text)

    return values


def duplicate_count(values: list[str]) -> int:
    counter = Counter(value for value in values if value)
    return sum(count - 1 for count in counter.values() if count > 1)


def duplicate_values(values: list[str]) -> list[dict[str, Any]]:
    counter = Counter(value for value in values if value)

    return [
        {"value": value, "count": count}
        for value, count in counter.most_common()
        if count > 1
    ]


def coverage(
    records: list[dict[str, Any]],
    field_aliases: dict[str, tuple[str, ...]],
) -> dict[str, dict[str, Any]]:
    result = {}
    total = len(records)

    for output_name, aliases in field_aliases.items():
        available = sum(
            1
            for record in records
            if not is_blank(profile_value(record, *aliases))
        )

        result[output_name] = {
            "available": available,
            "missing": total - available,
            "coverage_percentage": round((available / total) * 100, 2),
        }

    return result


def category_counts(
    records: list[dict[str, Any]],
    *aliases: str,
) -> dict[str, int]:
    counter: Counter[str] = Counter()

    for record in records:
        value = profile_value(record, *aliases)

        for item in string_values(value):
            counter[item] += 1

    return dict(counter.most_common())


def get_publications(record: dict[str, Any]) -> list[dict[str, Any]]:
    publications = profile_value(
        record,
        "selected_publications",
        "selectedpublications",
        "publications",
    )

    if not isinstance(publications, list):
        publications = deep_find_first(
            record,
            "selected_publications",
            "selectedpublications",
        )

    if not isinstance(publications, list):
        return []

    return [
        publication
        for publication in publications
        if isinstance(publication, dict)
    ]


def get_evidence_pages(record: dict[str, Any]) -> list[dict[str, Any]]:
    pages = profile_value(
        record,
        "evidence_pages",
        "evidencepages",
    )

    if not isinstance(pages, (list, dict)):
        pages = deep_find_first(
            record,
            "evidence_pages",
            "evidencepages",
        )

    if isinstance(pages, list):
        return [
            page
            for page in pages
            if isinstance(page, dict)
        ]

    if isinstance(pages, dict):
        results = []

        for evidence_key, page in pages.items():
            if not isinstance(page, dict):
                continue

            normalised_page = dict(page)

            if is_blank(
                direct_value(
                    normalised_page,
                    "evidence_type",
                    "evidencetype",
                )
            ):
                normalised_page["evidence_type"] = evidence_key

            results.append(normalised_page)

        return results

    return []

def has_teaching_evidence(record: dict[str, Any]) -> bool:
    for page in get_evidence_pages(record):
        evidence_type = normalise_text(
            direct_value(
                page,
                "evidence_type",
                "evidencetype",
            )
        ).lower()

        label = normalise_text(
            direct_value(page, "label")
        ).lower()

        text = normalise_text(
            direct_value(page, "text", "content")
        )

        sections = direct_value(page, "sections")

        has_section_text = False

        if isinstance(sections, list):
            has_section_text = any(
                isinstance(section, dict)
                and not is_blank(
                    direct_value(section, "text", "content")
                )
                for section in sections
            )

        is_teaching_page = (
            evidence_type == "teaching"
            or "teaching" in label
        )

        if is_teaching_page and (text or has_section_text):
            return True

    return False

MODULE_CODE_PATTERN = re.compile(
    r"\b[A-Z]{2,5}\s?-?\d{3,5}\b",
    re.IGNORECASE,
)


def get_teaching_text(record: dict[str, Any]) -> str:
    text_parts = []

    for page in get_evidence_pages(record):
        evidence_type = normalise_text(
            direct_value(
                page,
                "evidence_type",
                "evidencetype",
            )
        ).lower()

        label = normalise_text(
            direct_value(page, "label")
        ).lower()

        if evidence_type != "teaching" and "teaching" not in label:
            continue

        page_text = normalise_text(
            direct_value(page, "text", "content")
        )

        if page_text:
            text_parts.append(page_text)

        sections = direct_value(page, "sections")

        if isinstance(sections, list):
            for section in sections:
                if not isinstance(section, dict):
                    continue

                section_text = normalise_text(
                    direct_value(section, "text", "content")
                )

                if section_text:
                    text_parts.append(section_text)

    return " ".join(dict.fromkeys(text_parts))


def has_teaching_page(record: dict[str, Any]) -> bool:
    for page in get_evidence_pages(record):
        evidence_type = normalise_text(
            direct_value(
                page,
                "evidence_type",
                "evidencetype",
            )
        ).lower()

        label = normalise_text(
            direct_value(page, "label")
        ).lower()

        if evidence_type == "teaching" or "teaching" in label:
            return True

    return False


def has_module_code_in_teaching(record: dict[str, Any]) -> bool:
    teaching_text = get_teaching_text(record)
    return bool(MODULE_CODE_PATTERN.search(teaching_text))

def module_identity(module: dict[str, Any]) -> str:
    identity_fields = (
        ("cms", ("module_cms_id", "modulecmsid")),
        ("id", ("module_id", "moduleid")),
        ("code", ("module_code", "modulecode")),
        ("slug", ("module_slug", "moduleslug")),
    )

    invalid_values = {"", "none", "null", "n/a", "na", "tbc"}

    for prefix, aliases in identity_fields:
        value = normalise_text(direct_value(module, *aliases))
        normalised = value.lower()

        if normalised not in invalid_values:
            return f"{prefix}:{normalised}"

    title = normalise_text(
        direct_value(module, "module_title", "moduletitle", "title")
    ).lower()

    description = normalise_text(
        direct_value(module, "description", "matching_text")
    ).lower()

    return f"title:{title}|description:{description[:150]}"


def audit_staff(staff_records: list[dict[str, Any]]) -> dict[str, Any]:
    staff_ids = [
        normalise_text(profile_value(record, "staff_id", "staffid"))
        for record in staff_records
    ]

    profile_urls = [
        normalise_text(
            profile_value(
                record,
                "profile_url",
                "profileurl",
                "canonical_url",
                "canonicalurl",
            )
        )
        for record in staff_records
    ]

    statuses = [
        normalise_text(
            profile_value(record, "scrape_status", "scrapestatus")
        ).lower()
        for record in staff_records
    ]

    top_level_keys = Counter(
        key
        for record in staff_records
        for key in record.keys()
    )

    total_publications = 0
    publications_with_doi = 0
    staff_with_publications = 0

    for record in staff_records:
        publications = get_publications(record)

        if publications:
            staff_with_publications += 1

        total_publications += len(publications)

        for publication in publications:
            doi = direct_value(
                publication,
                "doi",
                "doi_url",
                "doiurl",
            )

            if not is_blank(doi):
                publications_with_doi += 1

    staff_fields = {
        "staff_id": ("staff_id", "staffid"),
        "full_name": ("full_name", "fullname", "name"),
        "position": ("position", "job_title", "jobtitle"),
        "profile_url": (
            "profile_url",
            "profileurl",
            "canonical_url",
            "canonicalurl",
        ),
        "college_name": ("college_name", "collegename", "college"),
        "department_name": (
            "department_name",
            "departmentname",
            "department",
        ),
        "summary": (
            "summary",
            "short_summary",
            "shortsummary",
            "meta_description",
            "metadescription",
        ),
        "research_keywords": (
            "research_keywords",
            "researchkeywords",
        ),
        "qualifications": ("qualifications",),
    }

    return {
        "total_staff_records": len(staff_records),
        "unique_staff_ids": len({value for value in staff_ids if value}),
        "duplicate_staff_id_count": duplicate_count(staff_ids),
        "duplicate_staff_ids": duplicate_values(staff_ids),
        "unique_profile_urls": len(
            {value for value in profile_urls if value}
        ),
        "duplicate_profile_url_count": duplicate_count(profile_urls),
        "duplicate_profile_urls": duplicate_values(profile_urls),
        "scrape_status_counts": dict(Counter(statuses).most_common()),
        "staff_with_publications": staff_with_publications,
        "total_selected_publications": total_publications,
        "publications_with_doi": publications_with_doi,
        "staff_with_teaching_page": sum(
            has_teaching_page(record)
            for record in staff_records
        ),
        "staff_with_teaching_evidence": sum(
            has_teaching_evidence(record)
            for record in staff_records
        ),
        "staff_with_module_codes_in_teaching": sum(
            has_module_code_in_teaching(record)
            for record in staff_records
        ),
        "field_coverage": coverage(staff_records, staff_fields),
        "college_counts": category_counts(
            staff_records,
            "college_name",
            "collegename",
            "college",
        ),
        "department_counts": category_counts(
            staff_records,
            "department_name",
            "departmentname",
            "department",
        ),
        "top_level_key_frequency": dict(
            top_level_keys.most_common()
        ),
    }


def audit_courses(course_records: list[dict[str, Any]]) -> dict[str, Any]:
    course_ids = []
    course_urls = []
    module_records = []

    course_departments: Counter[str] = Counter()
    course_colleges: Counter[str] = Counter()

    for course in course_records:
        course_ids.append(
            normalise_text(
                direct_value(course, "course_id", "courseid")
            )
        )

        course_urls.append(
            normalise_text(
                direct_value(course, "source_url", "sourceurl")
            )
        )

        for department in string_values(
            direct_value(course, "department", "departments")
        ):
            course_departments[department] += 1

        for college in string_values(
            direct_value(course, "college", "colleges")
        ):
            course_colleges[college] += 1

        modules = direct_value(course, "modules")

        if isinstance(modules, list):
            for module in modules:
                if isinstance(module, dict):
                    module_records.append(module)

    module_keys = [module_identity(module) for module in module_records]

    module_ids = [
        normalise_text(
            direct_value(module, "module_id", "moduleid")
        )
        for module in module_records
    ]

    module_codes = [
        normalise_text(
            direct_value(module, "module_code", "modulecode")
        )
        for module in module_records
    ]

    def module_missing(*aliases: str) -> int:
        return sum(
            is_blank(direct_value(module, *aliases))
            for module in module_records
        )

    invalid_codes = {"", "none", "null", "n/a", "na", "tbc"}

    return {
        "total_course_records": len(course_records),
        "unique_course_ids": len(
            {value for value in course_ids if value}
        ),
        "duplicate_course_id_count": duplicate_count(course_ids),
        "duplicate_course_ids": duplicate_values(course_ids),
        "duplicate_course_url_count": duplicate_count(course_urls),
        "total_module_occurrences": len(module_records),
        "unique_module_entities": len(set(module_keys)),
        "repeated_module_occurrences": (
            len(module_records) - len(set(module_keys))
        ),
        "unique_module_ids": len(
            {value for value in module_ids if value}
        ),
        "unique_valid_module_codes": len(
            {
                value.upper()
                for value in module_codes
                if value.lower() not in invalid_codes
            }
        ),
        "modules_with_tbc_or_missing_code": sum(
            value.lower() in invalid_codes
            for value in module_codes
        ),
        "missing_module_title": module_missing(
            "module_title",
            "moduletitle",
            "title",
        ),
        "missing_module_description": module_missing(
            "description",
        ),
        "missing_module_matching_text": module_missing(
            "matching_text",
            "matchingtext",
        ),
        "missing_module_credits": module_missing("credits"),
        "department_counts": dict(
            course_departments.most_common()
        ),
        "college_counts": dict(course_colleges.most_common()),
    }


def create_markdown_report(report: dict[str, Any]) -> str:
    staff = report["staff"]
    courses = report["courses"]

    lines = [
        "# Dataset Audit Report",
        "",
        "## Staff dataset",
        "",
        f"- Total staff records: {staff['total_staff_records']}",
        f"- Unique staff IDs: {staff['unique_staff_ids']}",
        f"- Duplicate staff IDs: {staff['duplicate_staff_id_count']}",
        f"- Unique profile URLs: {staff['unique_profile_urls']}",
        f"- Staff with publications: {staff['staff_with_publications']}",
        f"- Selected publications: {staff['total_selected_publications']}",
        f"- Publications with DOI: {staff['publications_with_doi']}",
        (
            "- Staff with teaching evidence: "
            f"{staff['staff_with_teaching_evidence']}"
        ),
                (
            "- Staff with a teaching page: "
            f"{staff['staff_with_teaching_page']}"
        ),
        (
            "- Staff with usable teaching evidence: "
            f"{staff['staff_with_teaching_evidence']}"
        ),
        (
            "- Staff with module codes in teaching evidence: "
            f"{staff['staff_with_module_codes_in_teaching']}"
        ),
        "",
        "## Course dataset",
        "",
        f"- Total courses: {courses['total_course_records']}",
        f"- Unique course IDs: {courses['unique_course_ids']}",
        (
            "- Total module occurrences: "
            f"{courses['total_module_occurrences']}"
        ),
        (
            "- Unique module entities: "
            f"{courses['unique_module_entities']}"
        ),
        (
            "- Repeated module occurrences: "
            f"{courses['repeated_module_occurrences']}"
        ),
        (
            "- Modules with missing descriptions: "
            f"{courses['missing_module_description']}"
        ),
        (
            "- Modules with TBC or missing codes: "
            f"{courses['modules_with_tbc_or_missing_code']}"
        ),
        (
            "- Modules with missing credits: "
            f"{courses['missing_module_credits']}"
        ),
        "",
        "## Staff field coverage",
        "",
        "| Field | Available | Missing | Coverage |",
        "|---|---:|---:|---:|",
    ]

    for field, statistics in staff["field_coverage"].items():
        lines.append(
            f"| {field} | {statistics['available']} | "
            f"{statistics['missing']} | "
            f"{statistics['coverage_percentage']}% |"
        )

    return "\n".join(lines) + "\n"


def run_audit(
    staff_path: Path,
    course_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    staff_records = load_json_records(
        staff_path,
        ("staff", "profiles", "records", "data"),
    )

    course_records = load_json_records(
        course_path,
        ("courses", "records", "data"),
    )

    report = {
        "staff_source": str(staff_path),
        "course_source": str(course_path),
        "staff": audit_staff(staff_records),
        "courses": audit_courses(course_records),
    }

    output_directory.mkdir(parents=True, exist_ok=True)

    json_output = output_directory / "data_audit.json"
    markdown_output = output_directory / "data_audit.md"

    with json_output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    markdown_output.write_text(
        create_markdown_report(report),
        encoding="utf-8",
    )

    return report