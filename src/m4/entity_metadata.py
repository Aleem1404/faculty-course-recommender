from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_STAFF_INPUT = Path("data/processed/faculty_profiles.jsonl")
DEFAULT_MODULE_INPUT = Path("data/processed/modules.jsonl")

DEFAULT_STAFF_FALLBACK_INPUT = Path(
    "data/raw/staff/university_faculty_profiles.json"
)

DEFAULT_MODULE_FALLBACK_INPUT = Path(
    "data/raw/courses/brunel_courses_nested.json"
)

DEFAULT_OUTPUT = Path("data/outputs/m4/entity_metadata.json")
DEFAULT_SUMMARY_OUTPUT = Path(
    "data/outputs/m4/entity_metadata_summary.json"
)

DEFAULT_STAFF_PROFILES = Path(
    "data/outputs/m4/staff_topic_profiles_filtered.jsonl"
)

DEFAULT_MODULE_PROFILES = Path(
    "data/outputs/m4/module_topic_profiles_filtered.jsonl"
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} in {path}"
                )

            yield record


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return list(read_jsonl(path))

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    return flatten_json_records(payload)


def flatten_json_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records: list[dict[str, Any]] = []

        for item in payload:
            records.extend(flatten_json_records(item))

        return records

    if isinstance(payload, dict):
        if looks_like_record(payload):
            return [payload]

        records = []

        for value in payload.values():
            records.extend(flatten_json_records(value))

        return records

    return []


def looks_like_record(value: dict[str, Any]) -> bool:
    likely_record_keys = {
        "id",
        "name",
        "title",
        "full_name",
        "staff_id",
        "faculty_id",
        "module_id",
        "course_id",
        "module_code",
        "course_code",
        "profile_url",
        "url",
        "slug",
    }

    return bool(set(value) & likely_record_keys)


def normalise_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)

    return text.strip("-")


def first_value(record: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = record.get(key)

        if value is None:
            continue

        if isinstance(value, str) and not value.strip():
            continue

        return value

    return None


def as_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return " ".join(as_text(item) for item in value if as_text(item))

    if isinstance(value, dict):
        return " ".join(
            as_text(item)
            for item in value.values()
            if as_text(item)
        )

    return str(value).strip()


def stable_module_id(record: dict[str, Any]) -> str:
    explicit_id = first_value(
        record,
        [
            "source_record_id",
            "module_id",
            "course_id",
            "id",
            "uuid",
            "module_code",
            "course_code",
            "code",
        ],
    )

    if explicit_id:
        text = str(explicit_id).strip()

        if text.startswith("module_"):
            return text

        return f"module_{normalise_text(text)}"

    title = as_text(
        first_value(
            record,
            [
                "module_title",
                "course_title",
                "title",
                "name",
            ],
        )
    )

    if not title:
        return ""

    digest = hashlib.sha1(
        title.encode("utf-8")
    ).hexdigest()[:16]

    return f"module_{digest}"


def stable_staff_id(record: dict[str, Any]) -> str:
    explicit_id = first_value(
        record,
        [
            "source_record_id",
            "staff_id",
            "faculty_id",
            "id",
            "uuid",
            "slug",
            "username",
        ],
    )

    if explicit_id:
        text = str(explicit_id).strip()

        if text.startswith("staff_"):
            return text

        return normalise_text(text)

    name = as_text(
        first_value(
            record,
            [
                "full_name",
                "staff_name",
                "faculty_name",
                "name",
                "title",
            ],
        )
    )

    return normalise_text(name)


def extract_staff_metadata(record: dict[str, Any]) -> dict[str, Any] | None:
    staff_id = stable_staff_id(record)

    if not staff_id:
        return None

    display_name = as_text(
        first_value(
            record,
            [
                "full_name",
                "staff_name",
                "faculty_name",
                "name",
                "title",
            ],
        )
    )

    if not display_name:
        return None

    return {
        "display_name": display_name,
        "job_title": as_text(
            first_value(
                record,
                [
                    "job_title",
                    "position",
                    "role",
                    "academic_title",
                ],
            )
        ),
        "school": as_text(
            first_value(
                record,
                [
                    "school",
                    "college",
                    "faculty",
                    "department",
                    "organisation",
                    "organization",
                ],
            )
        ),
        "department": as_text(
            first_value(
                record,
                [
                    "department",
                    "subject_area",
                    "discipline",
                ],
            )
        ),
        "profile_url": as_text(
            first_value(
                record,
                [
                    "profile_url",
                    "url",
                    "website",
                    "staff_url",
                ],
            )
        ),
    }


def extract_module_metadata(record: dict[str, Any]) -> dict[str, Any] | None:
    module_id = stable_module_id(record)

    if not module_id:
        return None

    title = as_text(
        first_value(
            record,
            [
                "module_title",
                "course_title",
                "title",
                "name",
            ],
        )
    )

    if not title:
        return None

    return {
        "display_name": title,
        "module_code": as_text(
            first_value(
                record,
                [
                    "module_code",
                    "course_code",
                    "code",
                ],
            )
        ),
        "level": as_text(
            first_value(
                record,
                [
                    "level",
                    "study_level",
                    "academic_level",
                ],
            )
        ),
        "department": as_text(
            first_value(
                record,
                [
                    "department",
                    "school",
                    "subject_area",
                    "faculty",
                ],
            )
        ),
        "url": as_text(
            first_value(
                record,
                [
                    "url",
                    "module_url",
                    "course_url",
                    "link",
                ],
            )
        ),
    }


def existing_profile_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    ids: set[str] = set()

    for record in read_jsonl(path):
        record_id = str(
            record.get("source_record_id", "")
        ).strip()

        if record_id:
            ids.add(record_id)

    return ids


def resolve_input(
    preferred_path: Path,
    fallback_path: Path,
) -> Path:
    if preferred_path.exists():
        return preferred_path

    if fallback_path.exists():
        return fallback_path

    raise FileNotFoundError(
        "No metadata source was found. Checked: "
        f"{preferred_path} and {fallback_path}"
    )


def index_staff_metadata(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}

    for record in records:
        item = extract_staff_metadata(record)

        if item is None:
            continue

        staff_id = stable_staff_id(record)

        if staff_id not in metadata:
            metadata[staff_id] = item

    return metadata


def index_module_metadata(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}

    for record in records:
        item = extract_module_metadata(record)

        if item is None:
            continue

        module_id = stable_module_id(record)

        if module_id not in metadata:
            metadata[module_id] = item

    return metadata


def profile_coverage(
    profile_ids: set[str],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    matched_ids = profile_ids & set(metadata)
    missing_ids = sorted(profile_ids - set(metadata))

    return {
        "profile_ids": len(profile_ids),
        "metadata_ids": len(metadata),
        "matched_profile_ids": len(matched_ids),
        "unmatched_profile_ids": len(missing_ids),
        "coverage_rate": round(
            len(matched_ids) / len(profile_ids),
            6,
        )
        if profile_ids
        else 0.0,
        "unmatched_id_examples": missing_ids[:20],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build display metadata for M4 staff and module "
            "collaboration recommendations."
        )
    )

    parser.add_argument(
        "--staff-input",
        type=Path,
        default=DEFAULT_STAFF_INPUT,
        help=(
            "Preferred processed staff input. If unavailable, "
            "the raw staff fallback path is used."
        ),
    )

    parser.add_argument(
        "--module-input",
        type=Path,
        default=DEFAULT_MODULE_INPUT,
        help=(
            "Preferred processed module input. If unavailable, "
            "the raw module fallback path is used."
        ),
    )

    parser.add_argument(
        "--staff-fallback-input",
        type=Path,
        default=DEFAULT_STAFF_FALLBACK_INPUT,
        help="Fallback raw staff JSON input.",
    )

    parser.add_argument(
        "--module-fallback-input",
        type=Path,
        default=DEFAULT_MODULE_FALLBACK_INPUT,
        help="Fallback raw module JSON input.",
    )

    parser.add_argument(
        "--staff-profiles",
        type=Path,
        default=DEFAULT_STAFF_PROFILES,
        help="Filtered staff profiles used by M4.",
    )

    parser.add_argument(
        "--module-profiles",
        type=Path,
        default=DEFAULT_MODULE_PROFILES,
        help="Filtered module profiles used by M4.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Metadata output JSON path.",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Metadata summary JSON path.",
    )

    return parser.parse_args()

 
def main() -> None:
    args = parse_args()

    staff_source = resolve_input(
        preferred_path=args.staff_input,
        fallback_path=args.staff_fallback_input,
    )

    module_source = resolve_input(
        preferred_path=args.module_input,
        fallback_path=args.module_fallback_input,
    )

    staff_records = read_json_or_jsonl(staff_source)
    module_records = extract_nested_modules(module_source)

    staff_metadata = index_staff_metadata(staff_records)
    module_metadata = index_module_metadata(module_records)

    staff_profile_ids = existing_profile_ids(args.staff_profiles)
    module_profile_ids = existing_profile_ids(args.module_profiles)

    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "staff": str(staff_source),
            "modules": str(module_source),
        },
        "staff": staff_metadata,
        "modules": module_metadata,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "staff_source": str(staff_source),
            "module_source": str(module_source),
            "staff_profiles": str(args.staff_profiles),
            "module_profiles": str(args.module_profiles),
        },
        "outputs": {
            "entity_metadata": str(args.output),
        },
        "counts": {
            "staff_records_read": len(staff_records),
            "module_records_read": len(module_records),
            "staff_metadata_entries": len(staff_metadata),
            "module_metadata_entries": len(module_metadata),
        },
        "m4_profile_id_coverage": {
            "staff": profile_coverage(
                profile_ids=staff_profile_ids,
                metadata=staff_metadata,
            ),
            "modules": profile_coverage(
                profile_ids=module_profile_ids,
                metadata=module_metadata,
            ),
        },
    }

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    with args.summary_output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

def extract_nested_modules(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        courses = json.load(file)

    if not isinstance(courses, list):
        raise ValueError(
            f"Expected a list of course records in {path}, "
            f"but received {type(courses).__name__}."
        )

    modules: list[dict[str, Any]] = []

    for course in courses:
        if not isinstance(course, dict):
            continue

        course_modules = course.get("modules", [])

        if not isinstance(course_modules, list):
            continue

        for module in course_modules:
            if not isinstance(module, dict):
                continue

            module_copy = dict(module)

            module_copy.setdefault(
                "course_id",
                course.get("course_id", ""),
            )
            module_copy.setdefault(
                "course_title",
                course.get("course_title", ""),
            )
            module_copy.setdefault(
                "college",
                course.get("college", []),
            )
            module_copy.setdefault(
                "department",
                course.get("department", []),
            )
            module_copy.setdefault(
                "subject_areas",
                course.get("subject_areas", []),
            )
            module_copy.setdefault(
                "level",
                course.get("level", []),
            )
            module_copy.setdefault(
                "source_url",
                course.get("source_url", ""),
            )

            modules.append(module_copy)

    return modules
if __name__ == "__main__":
    main()