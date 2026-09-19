from __future__ import annotations

import json
from pathlib import Path

from faculty_recommender.enrichment.audit import (
    audit_enrichment_readiness,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    core_directory = (
        project_root
        / "data"
        / "processed"
        / "core"
    )

    output_directory = (
        project_root
        / "data"
        / "processed"
        / "audit"
    )

    report = audit_enrichment_readiness(
        core_directory=core_directory,
        output_directory=output_directory,
    )

    summary = {
        "publication_records": report[
            "publication_records"
        ],
        "staff_publication_links": report[
            "staff_publication_links"
        ],
        "staff_records": report["staff_records"],
        "publications_with_titles": report[
            "publications_with_titles"
        ],
        "publications_with_doi": report[
            "publications_with_doi"
        ],
        "unique_dois": report["unique_dois"],
        "publications_without_doi": report[
            "publications_without_doi"
        ],
        "publications_with_any_url": report[
            "publications_with_any_url"
        ],
        "staff_with_profile_url": report[
            "staff_with_profile_url"
        ],
        "staff_with_any_external_url": report[
            "staff_with_any_external_url"
        ],
    }

    print(json.dumps(summary, indent=2))
    print()
    print(
        "Detailed report: "
        "data/processed/audit/"
        "enrichment_readiness.json"
    )


if __name__ == "__main__":
    main()