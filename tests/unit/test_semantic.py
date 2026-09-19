from __future__ import annotations

from typing import Any

import numpy as np

from faculty_recommender.retrieval.semantic import (
    SemanticFacultyRanker,
    chunk_text,
)


class FakeEncoder:
    vocabulary = (
        "artificial",
        "intelligence",
        "machine",
        "learning",
        "neural",
        "network",
        "systems",
        "law",
        "regulation",
        "governance",
    )

    def encode(
        self,
        sentences: list[str],
        **_: Any,
    ) -> np.ndarray:
        vectors = []

        for sentence in sentences:
            lower_text = sentence.lower()

            vector = np.asarray(
                [
                    lower_text.count(term)
                    for term in self.vocabulary
                ],
                dtype=np.float32,
            )

            norm = np.linalg.norm(vector)

            if norm > 0:
                vector = vector / norm

            vectors.append(vector)

        return np.asarray(
            vectors,
            dtype=np.float32,
        )


def test_chunk_text_has_overlap() -> None:
    text = "one two three four five six"

    chunks = chunk_text(
        text=text,
        chunk_size_words=4,
        overlap_words=2,
        max_chunks=3,
    )

    assert chunks == [
        "one two three four",
        "three four five six",
    ]


def test_semantic_ranker_uses_publications() -> None:
    staff = [
        {
            "staff_id": "technical-ai",
            "full_name": "Technical AI Researcher",
            "position": "Lecturer",
            "department_name": "Computer Science",
            "college_name": "CEDPS",
            "profile_url": "",
            "core_profile_text": (
                "computer science intelligent systems"
            ),
            "publication_title_text": (
                "Deep neural networks"
            ),
            "publication_abstract_text": (
                "Machine learning neural network "
                "classification"
            ),
            "publication_topic_text": (
                "Artificial intelligence machine learning"
            ),
            "publication_keyword_text": (
                "Neural network"
            ),
            "publication_count": 1,
            "enriched_publication_count": 1,
        },
        {
            "staff_id": "ai-law",
            "full_name": "AI Law Researcher",
            "position": "Lecturer",
            "department_name": "Law",
            "college_name": "CALS",
            "profile_url": "",
            "core_profile_text": (
                "artificial intelligence law regulation"
            ),
            "publication_title_text": (
                "Legal governance"
            ),
            "publication_abstract_text": (
                "Law regulation governance"
            ),
            "publication_topic_text": (
                "Law"
            ),
            "publication_keyword_text": (
                "Regulation"
            ),
            "publication_count": 1,
            "enriched_publication_count": 1,
        },
    ]

    modules = [
        {
            "module_id": "module-ai",
            "module_code": "CS0001",
            "module_title": "Artificial Intelligence",
            "departments": ["Computer Science"],
            "colleges": ["CEDPS"],
            "description": (
                "Artificial intelligence methods"
            ),
            "matching_text": (
                "artificial intelligence machine learning "
                "neural networks intelligent systems"
            ),
        }
    ]

    ranker = SemanticFacultyRanker(
        top_k=2,
        encoder=FakeEncoder(),
        show_progress_bar=False,
    )

    ranker.fit(
        staff_records=staff,
        module_records=modules,
    )

    result = ranker.rank_module(0)

    recommendations = result[
        "recommendations"
    ]

    assert recommendations[0][
        "staff_id"
    ] == "technical-ai"

    assert recommendations[0][
        "semantic_score"
    ] > recommendations[1][
        "semantic_score"
    ]