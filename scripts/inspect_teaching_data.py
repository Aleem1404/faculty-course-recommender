from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("staff", "profiles", "records", "data"):
            value = payload.get(key)

            if isinstance(value, list):
                return value

    raise ValueError("Could not find the staff record list.")


def find_teaching_paths(
    value: Any,
    current_path: str = "record",
) -> list[tuple[str, str]]:
    results = []

    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{current_path}.{key}"

            if "teach" in key.lower():
                preview = str(child).replace("\n", " ")[:500]
                results.append((path, preview))

            results.extend(find_teaching_paths(child, path))

    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{current_path}[{index}]"
            results.extend(find_teaching_paths(child, path))

    return results


def find_teaching_values(
    value: Any,
    current_path: str = "record",
) -> list[tuple[str, str]]:
    results = []

    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{current_path}.{key}"

            if isinstance(child, str) and "teaching" in child.lower():
                preview = child.replace("\n", " ")[:500]
                results.append((path, preview))

            results.extend(find_teaching_values(child, path))

    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{current_path}[{index}]"
            results.extend(find_teaching_values(child, path))

    return results


def staff_name(record: dict[str, Any]) -> str:
    for key in ("full_name", "fullname", "name"):
        value = record.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    profile = record.get("profile")

    if isinstance(profile, dict):
        for key in ("full_name", "fullname", "name"):
            value = profile.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

    return "Unknown staff member"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    staff_path = (
        project_root
        / "data"
        / "raw"
        / "staff"
        / "university_faculty_profiles.json"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "audit"
        / "teaching_structure_samples.txt"
    )

    records = load_records(staff_path)

    key_path_counter: Counter[str] = Counter()
    value_path_counter: Counter[str] = Counter()
    samples = []
    matching_records = 0

    for record_index, record in enumerate(records):
        key_matches = find_teaching_paths(record)
        value_matches = find_teaching_values(record)

        if not key_matches and not value_matches:
            continue

        matching_records += 1

        for path, _ in key_matches:
            cleaned_path = path.replace(
                f"record[{record_index}]",
                "record",
            )
            key_path_counter[cleaned_path] += 1

        for path, _ in value_matches:
            cleaned_path = path.replace(
                f"record[{record_index}]",
                "record",
            )
            value_path_counter[cleaned_path] += 1

        if len(samples) < 10:
            samples.append(
                {
                    "record_index": record_index,
                    "staff_name": staff_name(record),
                    "key_matches": key_matches[:10],
                    "value_matches": value_matches[:10],
                }
            )

    lines = [
        f"Total staff records: {len(records)}",
        f"Records containing teaching evidence: {matching_records}",
        "",
        "Most common teaching key paths:",
    ]

    for path, count in key_path_counter.most_common(30):
        lines.append(f"{count}: {path}")

    lines.extend(
        [
            "",
            "Most common teaching value paths:",
        ]
    )

    for path, count in value_path_counter.most_common(30):
        lines.append(f"{count}: {path}")

    lines.extend(
        [
            "",
            "Sample records:",
            "",
        ]
    )

    for sample in samples:
        lines.append(json.dumps(sample, indent=2, ensure_ascii=False))
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Total staff records: {len(records)}")
    print(f"Records containing teaching evidence: {matching_records}")
    print()
    print("Most common teaching key paths:")

    for path, count in key_path_counter.most_common(10):
        print(f"{count}: {path}")

    print()
    print(f"Detailed output: {output_path}")


if __name__ == "__main__":
    main()