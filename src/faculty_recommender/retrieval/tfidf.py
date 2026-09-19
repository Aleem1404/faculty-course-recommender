from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected an object in {path} at line {line_number}"
                )

            records.append(record)

    return records


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    return " ".join(str(value).split()).strip()


def normalise_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [
            clean_text(item)
            for item in value
            if clean_text(item)
        ]

    text = clean_text(value)

    return [text] if text else []


class TfidfFacultyRanker:
    def __init__(
        self,
        top_k: int = 5,
        shared_terms_count: int = 8,
    ) -> None:
        self.top_k = top_k
        self.shared_terms_count = shared_terms_count

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            max_features=75_000,
            sublinear_tf=True,
            norm="l2",
        )

        self.staff_records: list[dict[str, Any]] = []
        self.module_records: list[dict[str, Any]] = []

        self.staff_matrix: csr_matrix | None = None
        self.module_matrix: csr_matrix | None = None

    def fit(
        self,
        staff_records: list[dict[str, Any]],
        module_records: list[dict[str, Any]],
    ) -> dict[str, int]:
        self.staff_records = [
            record
            for record in staff_records
            if clean_text(
                record.get("matching_text_without_teaching")
            )
        ]

        self.module_records = [
            record
            for record in module_records
            if clean_text(record.get("matching_text"))
        ]

        if not self.staff_records:
            raise ValueError("No staff records have usable matching text.")

        if not self.module_records:
            raise ValueError("No modules have usable matching text.")

        staff_texts = [
            clean_text(
                record["matching_text_without_teaching"]
            )
            for record in self.staff_records
        ]

        module_texts = [
            clean_text(record["matching_text"])
            for record in self.module_records
        ]

        # combined_texts = staff_texts + module_texts
        # combined_matrix = self.vectorizer.fit_transform(
        #     combined_texts
        # )

        # staff_count = len(staff_texts)

        # self.staff_matrix = combined_matrix[:staff_count].tocsr()
        # self.module_matrix = combined_matrix[staff_count:].tocsr()
        self.staff_matrix = self.vectorizer.fit_transform(
            staff_texts
        ).tocsr()

        self.module_matrix = self.vectorizer.transform(
            module_texts
        ).tocsr()

        return {
            "input_staff": len(staff_records),
            "fitted_staff": len(self.staff_records),
            "excluded_staff": (
                len(staff_records) - len(self.staff_records)
            ),
            "input_modules": len(module_records),
            "fitted_modules": len(self.module_records),
            "excluded_modules": (
                len(module_records) - len(self.module_records)
            ),
            "vocabulary_size": len(
                self.vectorizer.vocabulary_
            ),
        }

    def _check_fitted(self) -> None:
        if self.staff_matrix is None:
            raise RuntimeError("The TF-IDF ranker has not been fitted.")

        if self.module_matrix is None:
            raise RuntimeError("The TF-IDF ranker has not been fitted.")

    def _shared_terms(
        self,
        module_index: int,
        staff_index: int,
    ) -> list[dict[str, Any]]:
        self._check_fitted()

        feature_names = self.vectorizer.get_feature_names_out()

        module_vector = self.module_matrix[module_index]
        staff_vector = self.staff_matrix[staff_index]

        common_vector = module_vector.multiply(
            staff_vector
        ).tocsr()

        if common_vector.nnz == 0:
            return []

        order = np.argsort(common_vector.data)[::-1]
        order = order[: self.shared_terms_count]

        shared_terms = []

        for position in order:
            feature_index = common_vector.indices[position]
            weight = common_vector.data[position]

            shared_terms.append(
                {
                    "term": str(feature_names[feature_index]),
                    "weight": round(float(weight), 6),
                }
            )

        return shared_terms

    def rank_module(
        self,
        module_index: int,
    ) -> dict[str, Any]:
        self._check_fitted()

        module = self.module_records[module_index]
        module_vector = self.module_matrix[module_index]

        scores = linear_kernel(
            module_vector,
            self.staff_matrix,
        ).ravel()

        result_count = min(self.top_k, len(scores))

        if result_count == 0:
            ranked_indices = np.array([], dtype=int)
        elif result_count == len(scores):
            ranked_indices = np.argsort(scores)[::-1]
        else:
            candidate_indices = np.argpartition(
                scores,
                -result_count,
            )[-result_count:]

            ranked_indices = candidate_indices[
                np.argsort(scores[candidate_indices])[::-1]
            ]

        recommendations = []

        for rank, staff_index in enumerate(
            ranked_indices,
            start=1,
        ):
            staff = self.staff_records[int(staff_index)]

            recommendations.append(
                {
                    "rank": rank,
                    "staff_id": staff.get("staff_id"),
                    "full_name": staff.get("full_name"),
                    "position": staff.get("position"),
                    "department_name": staff.get(
                        "department_name"
                    ),
                    "college_name": staff.get("college_name"),
                    "profile_url": staff.get("profile_url"),
                    "tfidf_score": round(
                        float(scores[staff_index]),
                        6,
                    ),
                    "shared_terms": self._shared_terms(
                        module_index=module_index,
                        staff_index=int(staff_index),
                    ),
                }
            )

        return {
            "module_id": module.get("module_id"),
            "module_code": module.get("module_code"),
            "module_title": module.get("module_title"),
            "departments": normalise_list(
                module.get("departments")
            ),
            "colleges": normalise_list(
                module.get("colleges")
            ),
            "has_description": bool(
                clean_text(module.get("description"))
            ),
            "model": "tfidf_cosine_baseline",
            "top_k": self.top_k,
            "recommendations": recommendations,
        }

    def rank_all_modules(self) -> list[dict[str, Any]]:
        self._check_fitted()

        return [
            self.rank_module(module_index)
            for module_index in range(len(self.module_records))
        ]

    def save_cache(
        self,
        path: Path,
    ) -> None:
        self._check_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)

        cache = {
            "vectorizer": self.vectorizer,
            "staff_matrix": self.staff_matrix,
            "module_matrix": self.module_matrix,
            "staff_ids": [
                record.get("staff_id")
                for record in self.staff_records
            ],
            "module_ids": [
                record.get("module_id")
                for record in self.module_records
            ],
        }

        joblib.dump(cache, path)


def write_jsonl(
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )


def run_tfidf_baseline(
    processed_directory: Path,
    cache_directory: Path,
    output_directory: Path,
    top_k: int = 5,
) -> dict[str, Any]:
    staff_records = load_jsonl(
        processed_directory / "staff.jsonl"
    )

    module_records = load_jsonl(
        processed_directory / "modules.jsonl"
    )

    ranker = TfidfFacultyRanker(top_k=top_k)

    fit_summary = ranker.fit(
        staff_records=staff_records,
        module_records=module_records,
    )

    recommendations = ranker.rank_all_modules()

    output_path = (
        output_directory
        / "tfidf_module_recommendations.jsonl"
    )

    cache_path = (
        cache_directory
        / "tfidf_baseline.joblib"
    )

    write_jsonl(recommendations, output_path)
    ranker.save_cache(cache_path)

    scores = [
        recommendation["tfidf_score"]
        for module_result in recommendations
        for recommendation in module_result["recommendations"]
    ]

    zero_score_modules = sum(
        bool(result["recommendations"])
        and result["recommendations"][0]["tfidf_score"] == 0
        for result in recommendations
    )

    summary = {
        **fit_summary,
        "top_k": top_k,
        "output_module_results": len(recommendations),
        "output_recommendations": sum(
            len(result["recommendations"])
            for result in recommendations
        ),
        "zero_score_modules": zero_score_modules,
        "mean_recommendation_score": round(
            float(np.mean(scores)) if scores else 0.0,
            6,
        ),
        "maximum_recommendation_score": round(
            float(np.max(scores)) if scores else 0.0,
            6,
        ),
        "output_path": str(output_path),
        "cache_path": str(cache_path),
    }

    summary_path = (
        output_directory
        / "tfidf_baseline_summary.json"
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return summary