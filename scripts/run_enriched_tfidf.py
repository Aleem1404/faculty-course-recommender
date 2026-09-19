from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

import joblib

from faculty_recommender.retrieval.enriched_tfidf import (
    EnrichedTfidfFacultyRanker,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    staff_path = (
        project_root
        / "data"
        / "processed"
        / "enrichment"
        / "staff_enriched.jsonl"
    )

    modules_path = (
        project_root
        / "data"
        / "processed"
        / "core"
        / "modules.jsonl"
    )

    output_directory = (
        project_root
        / "data"
        / "outputs"
        / "recommendations"
        / "enriched_tfidf"
    )

    cache_directory = (
        project_root
        / "data"
        / "cache"
        / "embeddings"
    )

    staff_records = load_jsonl(staff_path)
    module_records = load_jsonl(modules_path)

    ranker = EnrichedTfidfFacultyRanker(
        top_k=5,
        shared_terms_count=8,
    )

    ranker.fit(
        staff_records=staff_records,
        module_records=module_records,
    )

    results = ranker.rank_all_modules()

    output_path = (
        output_directory
        / "enriched_tfidf_recommendations.jsonl"
    )

    write_jsonl(output_path, results)

    cache_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = (
        cache_directory
        / "enriched_tfidf.joblib"
    )

    joblib.dump(ranker, cache_path)

    all_scores = [
        recommendation["enriched_tfidf_score"]
        for result in results
        for recommendation in result[
            "recommendations"
        ]
    ]

    summary = {
        "model": "field_weighted_enriched_tfidf",
        "input_staff": len(staff_records),
        "fitted_staff": len(
            ranker.fitted_staff_indices
        ),
        "excluded_staff": (
            len(staff_records)
            - len(ranker.fitted_staff_indices)
        ),
        "input_modules": len(module_records),
        "fitted_modules": len(
            ranker.fitted_module_indices
        ),
        "excluded_modules": (
            len(module_records)
            - len(ranker.fitted_module_indices)
        ),
        "output_module_results": len(results),
        "output_recommendations": sum(
            len(result["recommendations"])
            for result in results
        ),
        "mean_recommendation_score": round(
            mean(all_scores),
            6,
        ),
        "maximum_recommendation_score": round(
            max(all_scores),
            6,
        ),
        "field_weights": {
            field.name: field.weight
            for field in ranker.fields
        },
        "output_path": str(output_path),
        "cache_path": str(cache_path),
    }

    summary_path = (
        output_directory
        / "enriched_tfidf_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()