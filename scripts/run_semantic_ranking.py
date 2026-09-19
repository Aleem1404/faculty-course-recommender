from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from faculty_recommender.retrieval.semantic import (
    SemanticFacultyRanker,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at "
                    f"line {line_number}"
                ) from error

            records.append(record)

    return records


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    module_path = (
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
        / "semantic"
    )

    embedding_cache_directory = (
        project_root
        / "data"
        / "cache"
        / "embeddings"
        / "semantic"
    )

    model_cache_directory = (
        project_root
        / "data"
        / "cache"
        / "models"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedding_cache_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    staff_records = load_jsonl(staff_path)
    module_records = load_jsonl(module_path)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model_name = (
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    )

    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Staff records: {len(staff_records)}")
    print(f"Module records: {len(module_records)}")
    print()

    started_at = time.perf_counter()

    ranker = SemanticFacultyRanker(
        model_name=model_name,
        top_k=5,
        batch_size=32,
        chunk_size_words=180,
        overlap_words=30,
        max_chunks_per_field=16,
        device=device,
        model_cache_directory=(
            model_cache_directory
        ),
        show_progress_bar=True,
    )

    ranker.fit(
        staff_records=staff_records,
        module_records=module_records,
    )

    results = ranker.rank_all_modules()

    elapsed_seconds = (
        time.perf_counter() - started_at
    )

    output_path = (
        output_directory
        / "semantic_module_recommendations.jsonl"
    )

    write_jsonl(
        output_path,
        results,
    )

    embedding_path = (
        embedding_cache_directory
        / "semantic_embeddings.npz"
    )

    ranker.save_embeddings(
        embedding_path
    )

    index_path = (
        embedding_cache_directory
        / "semantic_embedding_index.json"
    )

    index_data = {
        "model_name": model_name,
        "embedding_dimension": (
            ranker.embedding_dimension
        ),
        "field_names": [
            field.name
            for field in ranker.fields
        ],
        "staff_ids": [
            staff_records[index].get("staff_id")
            for index in (
                ranker.fitted_staff_indices
            )
        ],
        "module_ids": [
            module_records[index].get(
                "module_id"
            )
            for index in (
                ranker.fitted_module_indices
            )
        ],
    }

    index_path.write_text(
        json.dumps(
            index_data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    all_scores = [
        recommendation["semantic_score"]
        for result in results
        for recommendation in result[
            "recommendations"
        ]
    ]

    zero_score_modules = sum(
        max(
            recommendation["semantic_score"]
            for recommendation in result[
                "recommendations"
            ]
        )
        <= 0
        for result in results
    )

    summary = {
        "model": "field_weighted_semantic",
        "embedding_model": model_name,
        "device": device,
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
        "embedding_dimension": (
            ranker.embedding_dimension
        ),
        "chunk_counts": ranker.chunk_counts,
        "field_weights": {
            field.name: field.weight
            for field in ranker.fields
        },
        "top_k": ranker.top_k,
        "output_module_results": len(results),
        "output_recommendations": sum(
            len(result["recommendations"])
            for result in results
        ),
        "zero_score_modules": (
            zero_score_modules
        ),
        "mean_recommendation_score": round(
            mean(all_scores),
            6,
        ),
        "maximum_recommendation_score": round(
            max(all_scores),
            6,
        ),
        "elapsed_seconds": round(
            elapsed_seconds,
            2,
        ),
        "output_path": str(output_path),
        "embedding_path": str(
            embedding_path
        ),
        "index_path": str(index_path),
    }

    summary_path = (
        output_directory
        / "semantic_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()