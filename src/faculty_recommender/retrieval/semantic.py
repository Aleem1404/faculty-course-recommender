from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SemanticFieldConfiguration:
    name: str
    staff_field: str
    weight: float


DEFAULT_SEMANTIC_FIELDS = (
    SemanticFieldConfiguration(
        name="core",
        staff_field="core_profile_text",
        weight=0.35,
    ),
    SemanticFieldConfiguration(
        name="publication_titles",
        staff_field="publication_title_text",
        weight=0.15,
    ),
    SemanticFieldConfiguration(
        name="publication_abstracts",
        staff_field="publication_abstract_text",
        weight=0.25,
    ),
    SemanticFieldConfiguration(
        name="publication_topics",
        staff_field="publication_topic_text",
        weight=0.15,
    ),
    SemanticFieldConfiguration(
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
            text
            for item in value
            if (text := normalise_text(item))
        )

    return " ".join(str(value).split()).strip()


def chunk_text(
    text: str,
    chunk_size_words: int = 180,
    overlap_words: int = 30,
    max_chunks: int = 16,
) -> list[str]:
    text = normalise_text(text)

    if not text:
        return []

    if chunk_size_words < 1:
        raise ValueError(
            "chunk_size_words must be at least 1."
        )

    if overlap_words < 0:
        raise ValueError(
            "overlap_words cannot be negative."
        )

    if overlap_words >= chunk_size_words:
        raise ValueError(
            "overlap_words must be smaller than "
            "chunk_size_words."
        )

    if max_chunks < 1:
        raise ValueError(
            "max_chunks must be at least 1."
        )

    words = text.split()

    if len(words) <= chunk_size_words:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(words) and len(chunks) < max_chunks:
        end = min(start + chunk_size_words, len(words))

        chunks.append(
            " ".join(words[start:end])
        )

        if end >= len(words):
            break

        start = end - overlap_words

    return chunks


def normalise_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    norms[norms == 0] = 1.0

    return embeddings / norms


class SemanticFacultyRanker:
    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        top_k: int = 5,
        batch_size: int = 32,
        chunk_size_words: int = 180,
        overlap_words: int = 30,
        max_chunks_per_field: int = 16,
        fields: tuple[
            SemanticFieldConfiguration,
            ...,
        ] = DEFAULT_SEMANTIC_FIELDS,
        device: str | None = None,
        model_cache_directory: Path | None = None,
        encoder: Any | None = None,
        show_progress_bar: bool = True,
    ) -> None:
        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        weight_total = sum(
            field.weight for field in fields
        )

        if not np.isclose(weight_total, 1.0):
            raise ValueError(
                "Field weights must add up to 1.0."
            )

        self.model_name = model_name
        self.top_k = top_k
        self.batch_size = batch_size
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words
        self.max_chunks_per_field = (
            max_chunks_per_field
        )
        self.fields = fields
        self.device = device
        self.model_cache_directory = (
            model_cache_directory
        )
        self.encoder = encoder
        self.show_progress_bar = show_progress_bar

        self.staff_records: list[
            dict[str, Any]
        ] = []

        self.module_records: list[
            dict[str, Any]
        ] = []

        self.fitted_staff_indices: list[int] = []
        self.fitted_module_indices: list[int] = []

        self.module_embeddings: (
            np.ndarray | None
        ) = None

        self.staff_embeddings: dict[
            str,
            np.ndarray,
        ] = {}

        self.field_scores: dict[
            str,
            np.ndarray,
        ] = {}

        self.final_scores: np.ndarray | None = None

        self.field_availability: dict[
            str,
            np.ndarray,
        ] = {}

        self.chunk_counts: dict[str, int] = {}

        self.embedding_dimension: int | None = None

    def _get_encoder(self) -> Any:
        if self.encoder is not None:
            return self.encoder

        try:
            from sentence_transformers import (
                SentenceTransformer,
            )
        except ImportError as error:
            raise ImportError(
                "Install the semantic dependencies using "
                'python -m pip install -e ".[semantic]"'
            ) from error

        cache_folder = None

        if self.model_cache_directory is not None:
            self.model_cache_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            cache_folder = str(
                self.model_cache_directory
            )

        self.encoder = SentenceTransformer(
            self.model_name,
            device=self.device,
            cache_folder=cache_folder,
        )

        return self.encoder

    def _encode_documents(
        self,
        texts: list[str],
        max_chunks: int,
        expected_dimension: int | None = None,
    ) -> tuple[np.ndarray, int]:
        encoder = self._get_encoder()

        all_chunks: list[str] = []
        chunk_owners: list[int] = []

        for document_index, text in enumerate(texts):
            chunks = chunk_text(
                text=text,
                chunk_size_words=self.chunk_size_words,
                overlap_words=self.overlap_words,
                max_chunks=max_chunks,
            )

            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_owners.append(document_index)

        if not all_chunks:
            if expected_dimension is None:
                raise ValueError(
                    "Cannot determine embedding dimension "
                    "from empty documents."
                )

            return (
                np.zeros(
                    (
                        len(texts),
                        expected_dimension,
                    ),
                    dtype=np.float32,
                ),
                0,
            )

        chunk_embeddings = encoder.encode(
            all_chunks,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=(
                self.show_progress_bar
            ),
        )

        chunk_embeddings = np.asarray(
            chunk_embeddings,
            dtype=np.float32,
        )

        if chunk_embeddings.ndim == 1:
            chunk_embeddings = (
                chunk_embeddings.reshape(1, -1)
            )

        dimension = chunk_embeddings.shape[1]

        document_embeddings = np.zeros(
            (len(texts), dimension),
            dtype=np.float32,
        )

        document_chunk_counts = np.zeros(
            len(texts),
            dtype=np.int32,
        )

        for owner, embedding in zip(
            chunk_owners,
            chunk_embeddings,
            strict=False,
        ):
            document_embeddings[owner] += embedding
            document_chunk_counts[owner] += 1

        available_indices = (
            document_chunk_counts > 0
        )

        document_embeddings[
            available_indices
        ] /= document_chunk_counts[
            available_indices,
            None,
        ]

        document_embeddings = normalise_embeddings(
            document_embeddings
        )

        return document_embeddings, len(all_chunks)

    def fit(
        self,
        staff_records: list[dict[str, Any]],
        module_records: list[dict[str, Any]],
    ) -> SemanticFacultyRanker:
        self.staff_records = staff_records
        self.module_records = module_records

        self.fitted_staff_indices = [
            index
            for index, record in enumerate(
                staff_records
            )
            if any(
                normalise_text(
                    record.get(field.staff_field)
                )
                for field in self.fields
            )
        ]

        self.fitted_module_indices = [
            index
            for index, record in enumerate(
                module_records
            )
            if normalise_text(
                record.get("matching_text")
            )
        ]

        if not self.fitted_staff_indices:
            raise ValueError(
                "No staff records have usable text."
            )

        if not self.fitted_module_indices:
            raise ValueError(
                "No modules have usable text."
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
            normalise_text(
                record.get("matching_text")
            )
            for record in fitted_modules
        ]

        (
            self.module_embeddings,
            module_chunk_count,
        ) = self._encode_documents(
            texts=module_texts,
            max_chunks=4,
        )

        self.chunk_counts["modules"] = (
            module_chunk_count
        )

        self.embedding_dimension = (
            self.module_embeddings.shape[1]
        )

        score_shape = (
            len(fitted_modules),
            len(fitted_staff),
        )

        final_scores = np.zeros(
            score_shape,
            dtype=np.float32,
        )

        for field in self.fields:
            field_texts = [
                normalise_text(
                    record.get(field.staff_field)
                )
                for record in fitted_staff
            ]

            availability = np.asarray(
                [
                    bool(text)
                    for text in field_texts
                ],
                dtype=bool,
            )

            (
                field_embeddings,
                field_chunk_count,
            ) = self._encode_documents(
                texts=field_texts,
                max_chunks=(
                    self.max_chunks_per_field
                ),
                expected_dimension=(
                    self.embedding_dimension
                ),
            )

            if (
                field_embeddings.shape[1]
                != self.embedding_dimension
            ):
                raise ValueError(
                    f"Embedding dimension mismatch for "
                    f"field {field.name}."
                )

            similarities = (
                self.module_embeddings
                @ field_embeddings.T
            )

            similarities = np.clip(
                similarities,
                0.0,
                1.0,
            )

            similarities[
                :,
                ~availability,
            ] = 0.0

            self.staff_embeddings[
                field.name
            ] = field_embeddings

            self.field_scores[
                field.name
            ] = similarities

            self.field_availability[
                field.name
            ] = availability

            self.chunk_counts[
                field.name
            ] = field_chunk_count

            final_scores += (
                field.weight * similarities
            )

        self.final_scores = final_scores

        return self

    def rank_module(
        self,
        module_position: int,
    ) -> dict[str, Any]:
        if self.final_scores is None:
            raise RuntimeError(
                "The semantic ranker must be fitted first."
            )

        module_source_index = (
            self.fitted_module_indices[
                module_position
            ]
        )

        module = self.module_records[
            module_source_index
        ]

        scores = self.final_scores[
            module_position
        ]

        candidate_order = np.argsort(
            -scores,
            kind="stable",
        )[: self.top_k]

        recommendations = []

        for rank, staff_position_value in enumerate(
            candidate_order,
            start=1,
        ):
            staff_position = int(
                staff_position_value
            )

            staff_source_index = (
                self.fitted_staff_indices[
                    staff_position
                ]
            )

            staff = self.staff_records[
                staff_source_index
            ]

            field_score_values = {
                field.name: round(
                    float(
                        self.field_scores[
                            field.name
                        ][
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
                    field_score_values[
                        field.name
                    ]
                    * field.weight,
                    6,
                )
                for field in self.fields
            }

            top_semantic_field = max(
                field_contributions,
                key=field_contributions.get,
            )

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
                "semantic_score": round(
                    float(scores[staff_position]),
                    6,
                ),
                "field_scores": (
                    field_score_values
                ),
                "field_contributions": (
                    field_contributions
                ),
                "top_semantic_field": (
                    top_semantic_field
                ),
                "evidence_availability": {
                    field.name: bool(
                        self.field_availability[
                            field.name
                        ][staff_position]
                    )
                    for field in self.fields
                },
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
            "has_description": bool(
                normalise_text(
                    module.get("description")
                )
            ),
            "model": "field_weighted_semantic",
            "embedding_model": self.model_name,
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

    def save_embeddings(
        self,
        output_path: Path,
    ) -> None:
        if self.module_embeddings is None:
            raise RuntimeError(
                "Fit the ranker before saving embeddings."
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        arrays = {
            "module_embeddings": (
                self.module_embeddings
            ),
        }

        for field_name, embeddings in (
            self.staff_embeddings.items()
        ):
            arrays[
                f"staff_{field_name}_embeddings"
            ] = embeddings

        np.savez_compressed(
            output_path,
            **arrays,
        )