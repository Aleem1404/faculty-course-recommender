from __future__ import annotations

import json
from pathlib import Path

from faculty_recommender.data.preprocess import (
    run_preprocessing,
)


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
        / "core"
    )

    manifest = run_preprocessing(
        staff_path=staff_path,
        course_path=course_path,
        output_directory=output_directory,
    )

    print(json.dumps(manifest, indent=2))
    print()
    print("Preprocessing completed.")
    print(f"Output directory: {output_directory}")


if __name__ == "__main__":
    main()