from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from faculty_recommender.policy.departments import (
    canonicalise_department,
    canonicalise_departments,
)

from faculty_recommender.policy.eligibility import (
    primary_delivery_eligibility,
)


DEFAULT_FIELD_WEIGHTS = {
    "core": 0.35,
    "publication_titles": 0.15,
    "publication_abstracts": 0.25,
    "publication_topics": 0.15,
    "publication_keywords": 0.10,
}


@dataclass
class SemanticScoreBundle:
    staff_ids: list[str]
    module_ids: list[str]
    final_scores: np.ndarray
    field_scores: dict[str, np.ndarray]
    field_weights: dict[str, float]
    model_name: str


def load_semantic_score_bundle(
    embedding_path: Path,
    index_path: Path,
    field_weights: dict[str, float] | None = None,
) -> SemanticScoreBundle:
    field_weights = (
        field_weights or DEFAULT_FIELD_WEIGHTS
    )

    weight_total = sum(field_weights.values())

    if not np.isclose(weight_total, 1.0):
        raise ValueError(
            "Semantic field weights must add to 1.0."
        )

    index_data = json.loads(
        index_path.read_text(encoding="utf-8")
    )

    staff_ids = index_data["staff_ids"]
    module_ids = index_data["module_ids"]

    with np.load(embedding_path) as archive:
        module_embeddings = np.asarray(
            archive["module_embeddings"],
            dtype=np.float32,
        )

        field_scores: dict[str, np.ndarray] = {}

        final_scores = np.zeros(
            (
                len(module_ids),
                len(staff_ids),
            ),
            dtype=np.float32,
        )

        for field_name, weight in (
            field_weights.items()
        ):
            array_name = (
                f"staff_{field_name}_embeddings"
            )

            if array_name not in archive:
                raise ValueError(
                    f"Missing semantic embedding array: "
                    f"{array_name}"
                )

            staff_embeddings = np.asarray(
                archive[array_name],
                dtype=np.float32,
            )

            if (
                module_embeddings.shape[1]
                != staff_embeddings.shape[1]
            ):
                raise ValueError(
                    f"Embedding dimension mismatch for "
                    f"{field_name}."
                )

            similarities = (
                module_embeddings
                @ staff_embeddings.T
            )

            similarities = np.clip(
                similarities,
                0.0,
                1.0,
            )

            field_scores[field_name] = similarities
            final_scores += weight * similarities

    expected_shape = (
        len(module_ids),
        len(staff_ids),
    )

    if final_scores.shape != expected_shape:
        raise ValueError(
            "Semantic score dimensions do not match "
            "the embedding index."
        )

    return SemanticScoreBundle(
        staff_ids=staff_ids,
        module_ids=module_ids,
        final_scores=final_scores,
        field_scores=field_scores,
        field_weights=field_weights,
        model_name=index_data.get(
            "model_name",
            "unknown",
        ),
    )


class M2PrimaryDeliveryRanker:
    def __init__(
        self,
        staff_records: list[dict[str, Any]],
        module_records: list[dict[str, Any]],
        score_bundle: SemanticScoreBundle,
        department_aliases: (
            dict[str, str] | None
        ) = None,
        top_k: int = 5,
        allow_external_fallback: bool = True,
    ) -> None:
        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        self.staff_records = staff_records
        self.module_records = module_records
        self.score_bundle = score_bundle
        self.department_aliases = (
            department_aliases or {}
        )
        self.top_k = top_k
        self.allow_external_fallback = (
            allow_external_fallback
        )

        self.staff_by_id = {
            str(record.get("staff_id")): record
            for record in staff_records
            if record.get("staff_id")
        }

        self.module_by_id = {
            str(record.get("module_id")): record
            for record in module_records
            if record.get("module_id")
        }

        self.staff_position_by_id = {
            staff_id: position
            for position, staff_id in enumerate(
                score_bundle.staff_ids
            )
        }

        self.module_position_by_id = {
            module_id: position
            for position, module_id in enumerate(
                score_bundle.module_ids
            )
        }

    def _candidate_details(
        self,
        module_position: int,
        staff_position: int,
        rank: int,
        recommendation_role: str,
        eligibility_reason: str,
    ) -> dict[str, Any]:
        staff_id = self.score_bundle.staff_ids[
            staff_position
        ]

        staff = self.staff_by_id[staff_id]

        field_scores = {
            field_name: round(
                float(
                    score_matrix[
                        module_position,
                        staff_position,
                    ]
                ),
                6,
            )
            for field_name, score_matrix in (
                self.score_bundle
                .field_scores
                .items()
            )
        }

        field_contributions = {
            field_name: round(
                score
                * self.score_bundle.field_weights[
                    field_name
                ],
                6,
            )
            for field_name, score in (
                field_scores.items()
            )
        }

        top_evidence_field = max(
            field_contributions,
            key=field_contributions.get,
        )

        return {
            "rank": rank,
            "staff_id": staff_id,
            "full_name": staff.get("full_name"),
            "position": staff.get("position"),
            "department_name": staff.get(
                "department_name"
            ),
            "canonical_department": (
                canonicalise_department(
                    staff.get("department_name"),
                    self.department_aliases,
                )
            ),
            "college_name": staff.get(
                "college_name"
            ),
            "profile_url": staff.get(
                "profile_url"
            ),
            "recommendation_role": (
                recommendation_role
            ),
            "eligibility_reason": (
                eligibility_reason
            ),
            "semantic_score": round(
                float(
                    self.score_bundle.final_scores[
                        module_position,
                        staff_position,
                    ]
                ),
                6,
            ),
            "field_scores": field_scores,
            "field_contributions": (
                field_contributions
            ),
            "top_evidence_field": (
                top_evidence_field
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
        }

    def _eligible_candidates(
        self,
        home_departments: set[str],
        internal: bool,
    ) -> list[tuple[int, str]]:
        candidates: list[tuple[int, str]] = []

        for staff_position, staff_id in enumerate(
            self.score_bundle.staff_ids
        ):
            staff = self.staff_by_id.get(staff_id)

            if staff is None:
                continue

            decision = (
                primary_delivery_eligibility(staff)
            )

            if not decision.eligible:
                continue

            staff_department = (
                canonicalise_department(
                    staff.get("department_name"),
                    self.department_aliases,
                )
            )

            department_matches = (
                staff_department in home_departments
            )

            if internal and department_matches:
                candidates.append(
                    (
                        staff_position,
                        decision.reason,
                    )
                )

            if (
                not internal
                and not department_matches
            ):
                candidates.append(
                    (
                        staff_position,
                        decision.reason,
                    )
                )

        return candidates

    def rank_module(
        self,
        module_id: str,
    ) -> dict[str, Any]:
        module = self.module_by_id.get(module_id)

        if module is None:
            raise ValueError(
                f"Unknown module ID: {module_id}"
            )

        if module_id not in (
            self.module_position_by_id
        ):
            raise ValueError(
                f"Module {module_id} was not fitted "
                "by M2."
            )

        module_position = (
            self.module_position_by_id[module_id]
        )

        raw_departments = module.get(
            "departments",
            [],
        )

        home_departments = (
            canonicalise_departments(
                raw_departments,
                self.department_aliases,
            )
        )

        if not home_departments:
            return {
                "module_id": module_id,
                "module_code": module.get(
                    "module_code"
                ),
                "module_title": module.get(
                    "module_title"
                ),
                "owning_departments": (
                    raw_departments
                ),
                "canonical_departments": [],
                "policy": (
                    "home_department_first"
                ),
                "decision_status": (
                    "owning_department_missing"
                ),
                "fallback_used": False,
                "review_required": True,
                "recommendations": [],
            }

        internal_candidates = (
            self._eligible_candidates(
                home_departments=(
                    home_departments
                ),
                internal=True,
            )
        )

        if internal_candidates:
            selected_pool = internal_candidates
            fallback_used = False
            recommendation_role = (
                "primary_internal"
            )

            if len(internal_candidates) >= self.top_k:
                decision_status = (
                    "home_department_ranked"
                )
            else:
                decision_status = (
                    "limited_internal_pool"
                )

            fallback_reason = ""
        else:
            if not self.allow_external_fallback:
                return {
                    "module_id": module_id,
                    "module_code": module.get(
                        "module_code"
                    ),
                    "module_title": module.get(
                        "module_title"
                    ),
                    "owning_departments": (
                        raw_departments
                    ),
                    "canonical_departments": sorted(
                        home_departments
                    ),
                    "policy": (
                        "home_department_first"
                    ),
                    "decision_status": (
                        "no_eligible_internal_staff"
                    ),
                    "fallback_used": False,
                    "review_required": True,
                    "recommendations": [],
                }

            selected_pool = (
                self._eligible_candidates(
                    home_departments=(
                        home_departments
                    ),
                    internal=False,
                )
            )

            fallback_used = True
            recommendation_role = (
                "external_primary_fallback"
            )
            decision_status = (
                "university_fallback_ranked"
            )
            fallback_reason = (
                "No eligible internal staff with "
                "usable M2 evidence were available."
            )

        selected_pool.sort(
            key=lambda item: (
                -float(
                    self.score_bundle.final_scores[
                        module_position,
                        item[0],
                    ]
                ),
                self.score_bundle.staff_ids[
                    item[0]
                ],
            )
        )

        selected_pool = selected_pool[
            : self.top_k
        ]

        recommendations = [
            self._candidate_details(
                module_position=module_position,
                staff_position=staff_position,
                rank=rank,
                recommendation_role=(
                    recommendation_role
                ),
                eligibility_reason=(
                    eligibility_reason
                ),
            )
            for rank, (
                staff_position,
                eligibility_reason,
            ) in enumerate(
                selected_pool,
                start=1,
            )
        ]

        return {
            "module_id": module_id,
            "module_code": module.get(
                "module_code"
            ),
            "module_title": module.get(
                "module_title"
            ),
            "owning_departments": (
                raw_departments
            ),
            "canonical_departments": sorted(
                home_departments
            ),
            "policy": "home_department_first",
            "semantic_model": (
                self.score_bundle.model_name
            ),
            "decision_status": decision_status,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "internal_candidate_count": len(
                internal_candidates
            ),
            "returned_candidate_count": len(
                recommendations
            ),
            "review_required": (
                fallback_used
                or len(recommendations)
                < self.top_k
            ),
            "recommendations": recommendations,
        }

    def rank_all_modules(
        self,
    ) -> list[dict[str, Any]]:
        return [
            self.rank_module(module_id)
            for module_id in (
                self.score_bundle.module_ids
            )
        ]

    @staticmethod
    def summarise(
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        status_counts = Counter(
            result["decision_status"]
            for result in results
        )

        return {
            "module_results": len(results),
            "status_counts": dict(
                status_counts
            ),
            "home_department_results": sum(
                result["decision_status"]
                in {
                    "home_department_ranked",
                    "limited_internal_pool",
                }
                for result in results
            ),
            "fallback_results": sum(
                result.get(
                    "fallback_used",
                    False,
                )
                for result in results
            ),
            "review_required": sum(
                result.get(
                    "review_required",
                    False,
                )
                for result in results
            ),
            "total_recommendations": sum(
                len(
                    result.get(
                        "recommendations",
                        [],
                    )
                )
                for result in results
            ),
        }