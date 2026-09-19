from __future__ import annotations

import json
from pathlib import Path

from faculty_recommender.enrichment.staff_aggregation import (
    aggregate_staff_publications,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    core_directory = (
        project_root
        / "data"
        / "processed"
        / "core"
    )

    enrichment_directory = (
        project_root
        / "data"
        / "processed"
        / "enrichment"
    )

    summary = aggregate_staff_publications(
        staff_path=core_directory / "staff.jsonl",
        publication_path=(
            enrichment_directory
            / "publication_enrichment.jsonl"
        ),
        link_path=(
            core_directory
            / "staff_publication_links.jsonl"
        ),
        output_directory=enrichment_directory,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()