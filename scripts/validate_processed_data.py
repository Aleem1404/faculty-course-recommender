from __future__ import annotations

import json
from pathlib import Path

from faculty_recommender.data.validate import (
    validate_processed_data,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    processed_directory = (
        project_root
        / "data"
        / "processed"
        / "core"
    )

    output_directory = (
        project_root
        / "data"
        / "processed"
        / "validation"
    )

    report = validate_processed_data(
        processed_directory=processed_directory,
        output_directory=output_directory,
    )

    summary = {
        "counts": report["counts"],
        "checks": report["checks"],
        "staff_quality": report["staff_quality"],
        "module_quality": report["module_quality"],
        "teaching_code_analysis": {
            "unique_teaching_codes": report[
                "teaching_code_analysis"
            ]["unique_teaching_codes"],
            "unique_current_module_codes": report[
                "teaching_code_analysis"
            ]["unique_current_module_codes"],
            "matched_teaching_codes": report[
                "teaching_code_analysis"
            ]["matched_teaching_codes"],
            "unmatched_teaching_codes": report[
                "teaching_code_analysis"
            ]["unmatched_teaching_codes"],
        },
        "validation_passed": report[
            "validation_passed"
        ],
    }

    print(json.dumps(summary, indent=2))
    print()
    print(
        "Detailed report: "
        "data/processed/validation/"
        "processed_data_validation.json"
    )


if __name__ == "__main__":
    main()