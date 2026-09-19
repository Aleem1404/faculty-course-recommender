from __future__ import annotations

import json
from pathlib import Path

from faculty_recommender.retrieval.tfidf import (
    run_tfidf_baseline,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    processed_directory = (
        project_root
        / "data"
        / "processed"
        / "core"
    )

    cache_directory = (
        project_root
        / "data"
        / "cache"
        / "embeddings"
    )

    output_directory = (
        project_root
        / "data"
        / "outputs"
        / "recommendations"
        / "tfidf"
    )

    summary = run_tfidf_baseline(
        processed_directory=processed_directory,
        cache_directory=cache_directory,
        output_directory=output_directory,
        top_k=5,
    )

    print(json.dumps(summary, indent=2))
    print()
    print("TF-IDF baseline completed.")
    print(
        "Recommendations: "
        "data/outputs/recommendations/tfidf/"
        "tfidf_module_recommendations.jsonl"
    )


if __name__ == "__main__":
    main()