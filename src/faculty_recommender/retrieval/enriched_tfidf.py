from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


@dataclass(frozen=True)
class FieldConfiguration:
    name: str
    staff_field: str
    weight: float


DEFAULT_FIELDS = (
    FieldConfiguration(
        name="core",
        staff_field="core_profile_text",
        weight=0.35,
    ),
    FieldConfiguration(
        name="publication_titles",
        staff_field="publication_title_text",
        weight=0.15,
    ),
    FieldConfiguration(
        name="publication_abstracts",
        staff_field="publication_abstract_text",
        weight=0.25,
    ),
    FieldConfiguration(
        name="publication_topics",
        staff_field="publication_topic_text",
        weight=0.15,
    ),
    FieldConfiguration(
        name="publication_keywords",
        staff_field="publication_keyword_text",
        weight=0.10,
    ),
)


def normalise_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return " ".join(
            normalise_text(item)
            for item in value
            if normalise_text(item)
        )

    return " ".join(str(value).split()).strip()


class EnrichedTfidfFacultyRanker:
    def __init__(
        self,
        top_k: int = 5,
        shared_terms_count: int = 8,
        fields: tuple[
            FieldConfiguration,
            ...,
        ] = DEFAULT_FIELDS,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        weight_total = sum(field.weight for field in fields)

        if not np.isclose(weight_total, 1.0):
            raise ValueError(
                "Field weights must add up to 1.0."
            )

        self.top_k = top_k
        self.shared_terms_count = shared_terms_count
        self.fields = fields

        self.staff_records: list[dict[str, Any]] = []
        self.module_records: list[dict[str, Any]] = []

        self.vectorizers: dict[
            str,
            TfidfVectorizer,
        ] = {}

        self.staff_matrices: dict[str, csr_matrix] = {}
        self.module_matrices: dict[str, csr_matrix] = {}

        self.field_scores: dict[str, np.ndarray] = {}
        self.final_scores: np.ndarray | None = None

        self.fitted_staff_indices: list[int] = []
        self.fitted_module_indices: list[int] = []

    def _new_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
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

    def fit(
        self,
        staff_records: list[dict[str, Any]],
        module_records: list[dict[str, Any]],
    ) -> EnrichedTfidfFacultyRanker:
        self.staff_records = staff_records
        self.module_records = module_records

        self.fitted_staff_indices = [
            index
            for index, record in enumerate(staff_records)
            if any(
                normalise_text(
                    record.get(field.staff_field)
                )
                for field in self.fields
            )
        ]

        self.fitted_module_indices = [
            index
            for index, record in enumerate(module_records)
            if normalise_text(
                record.get("matching_text")
            )
        ]

        if not self.fitted_staff_indices:
            raise ValueError(
                "No staff records have usable enriched text."
            )

        if not self.fitted_module_indices:
            raise ValueError(
                "No modules have usable matching text."
            )

        fitted_staff = [
            staff_records[index]
            for index in self.fitted_staff_indices
        ]

        fitted_modules = [
            module_records[index]
            for index in self.fitted_module_indices
        ]

        module_texts = [
            normalise_text(record.get("matching_text"))
            for record in fitted_modules
        ]

        score_shape = (
            len(fitted_modules),
            len(fitted_staff),
        )

        combined_scores = np.zeros(
            score_shape,
            dtype=np.float64,
        )

        for field in self.fields:
            staff_texts = [
                normalise_text(
                    record.get(field.staff_field)
                )
                for record in fitted_staff
            ]

            non_empty_count = sum(
                bool(text)
                for text in staff_texts
            )

            if non_empty_count == 0:
                self.field_scores[field.name] = (
                    np.zeros(
                        score_shape,
                        dtype=np.float64,
                    )
                )
                continue

            vectorizer = self._new_vectorizer()

            try:
                staff_matrix = (
                    vectorizer.fit_transform(
                        staff_texts
                    ).tocsr()
                )
            except ValueError:
                self.field_scores[field.name] = (
                    np.zeros(
                        score_shape,
                        dtype=np.float64,
                    )
                )
                continue

            module_matrix = vectorizer.transform(
                module_texts
            ).tocsr()

            scores = linear_kernel(
                module_matrix,
                staff_matrix,
            )

            self.vectorizers[field.name] = vectorizer
            self.staff_matrices[field.name] = (
                staff_matrix
            )
            self.module_matrices[field.name] = (
                module_matrix
            )
            self.field_scores[field.name] = scores

            combined_scores += field.weight * scores

        self.final_scores = combined_scores
        return self

    def _shared_terms(
        self,
        field_name: str,
        module_position: int,
        staff_position: int,
    ) -> list[dict[str, Any]]:
        vectorizer = self.vectorizers.get(field_name)
        module_matrix = self.module_matrices.get(
            field_name
        )
        staff_matrix = self.staff_matrices.get(
            field_name
        )

        if (
            vectorizer is None
            or module_matrix is None
            or staff_matrix is None
        ):
            return []

        module_vector = module_matrix.getrow(
            module_position
        )

        staff_vector = staff_matrix.getrow(
            staff_position
        )

        product = module_vector.multiply(staff_vector)

        if product.nnz == 0:
            return []

        feature_names = (
            vectorizer.get_feature_names_out()
        )

        weighted_terms = sorted(
            zip(
                product.indices,
                product.data,
                strict=False,
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            {
                "term": feature_names[index],
                "weight": round(float(weight), 6),
            }
            for index, weight in weighted_terms[
                : self.shared_terms_count
            ]
        ]

    def rank_module(
        self,
        module_position: int,
    ) -> dict[str, Any]:
        if self.final_scores is None:
            raise RuntimeError(
                "The ranker must be fitted first."
            )

        module_source_index = (
            self.fitted_module_indices[module_position]
        )

        module = self.module_records[
            module_source_index
        ]

        scores = self.final_scores[module_position]

        candidate_order = np.argsort(
            -scores,
            kind="stable",
        )[: self.top_k]

        recommendations = []

        for rank, staff_position in enumerate(
            candidate_order,
            start=1,
        ):
            staff_source_index = (
                self.fitted_staff_indices[
                    int(staff_position)
                ]
            )

            staff = self.staff_records[
                staff_source_index
            ]

            field_score_values = {
                field.name: round(
                    float(
                        self.field_scores[field.name][
                            module_position,
                            staff_position,
                        ]
                    ),
                    6,
                )
                for field in self.fields
            }

            field_contributions = {
                field.name: round(
                    field_score_values[field.name]
                    * field.weight,
                    6,
                )
                for field in self.fields
            }

            shared_terms = {
                field.name: self._shared_terms(
                    field_name=field.name,
                    module_position=module_position,
                    staff_position=int(
                        staff_position
                    ),
                )
                for field in self.fields
            }

            recommendations.append({
                "rank": rank,
                "staff_id": staff.get(
                    "staff_id"
                ),
                "full_name": staff.get(
                    "full_name"
                ),
                "position": staff.get(
                    "position"
                ),
                "department_name": staff.get(
                    "department_name"
                ),
                "college_name": staff.get(
                    "college_name"
                ),
                "profile_url": staff.get(
                    "profile_url"
                ),
                "publication_count": staff.get(
                    "publication_count",
                    0,
                ),
                "enriched_publication_count": (
                    staff.get(
                        "enriched_publication_count",
                        0,
                    )
                ),
                "enriched_tfidf_score": round(
                    float(scores[staff_position]),
                    6,
                ),
                "field_scores": field_score_values,
                "field_contributions": (
                    field_contributions
                ),
                "shared_terms": shared_terms,
            })

        return {
            "module_id": module.get("module_id"),
            "module_code": module.get(
                "module_code"
            ),
            "module_title": module.get(
                "module_title"
            ),
            "departments": module.get(
                "departments",
                [],
            ),
            "colleges": module.get(
                "colleges",
                [],
            ),
            "model": (
                "field_weighted_enriched_tfidf"
            ),
            "top_k": self.top_k,
            "field_weights": {
                field.name: field.weight
                for field in self.fields
            },
            "recommendations": recommendations,
        }

    def rank_all_modules(
        self,
    ) -> list[dict[str, Any]]:
        return [
            self.rank_module(module_position)
            for module_position in range(
                len(self.fitted_module_indices)
            )
        ]