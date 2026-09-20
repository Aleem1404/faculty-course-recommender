from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from faculty_recommender.policy.eligibility import (
    primary_delivery_eligibility,
)

from faculty_recommender.policy.hierarchy import (
    AcademicHierarchy,
)

from faculty_recommender.ranking.m2_primary import (
    SemanticScoreBundle,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


class M2HierarchyPrimaryRanker:
    def __init__(
        self,
        staff_records: list[dict[str, Any]],
        module_records: list[dict[str, Any]],
        staff_mappings: list[dict[str, Any]],
        module_mappings: list[dict[str, Any]],
        hierarchy: AcademicHierarchy,
        score_bundle: SemanticScoreBundle,
        top_k: int = 5,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        self.staff_records = staff_records
        self.module_records = module_records
        self.hierarchy = hierarchy
        self.score_bundle = score_bundle
        self.top_k = top_k

        self.staff_by_id = {
            str(record["staff_id"]): record
            for record in staff_records
            if record.get("staff_id")
        }

        self.module_by_id = {
            str(record["module_id"]): record
            for record in module_records
            if record.get("module_id")
        }

        self.staff_mapping_by_id = {
            str(record["staff_id"]): record["mapping"]
            for record in staff_mappings
            if record.get("staff_id")
        }

        self.module_mapping_by_id = {
            str(record["module_id"]): record["mapping"]
            for record in module_mappings
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

    def _internal_candidate_paths(
        self,
        module_paths: list[str],
    ) -> dict[str, list[str]]:
        internal: dict[str, list[str]] = {}

        for staff_id, staff_mapping in (
            self.staff_mapping_by_id.items()
        ):
            staff_paths = staff_mapping.get(
                "resolved_paths",
                [],
            )

            matching_paths = [
                staff_path
                for staff_path in staff_paths
                if any(
                    self.hierarchy.is_same_or_descendant(
                        staff_path,
                        module_path,
                    )
                    for module_path in module_paths
                )
            ]

            if matching_paths:
                internal[staff_id] = matching_paths

        return internal

    def _candidate_record(
        self,
        module_position: int,
        staff_position: int,
        rank: int,
        staff_internal_paths: list[str],
    ) -> dict[str, Any]:
        staff_id = self.score_bundle.staff_ids[
            staff_position
        ]

        staff = self.staff_by_id[staff_id]

        field_scores = {
            field_name: round(
                float(
                    matrix[
                        module_position,
                        staff_position,
                    ]
                ),
                6,
            )
            for field_name, matrix in (
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
            "college_name": staff.get(
                "college_name"
            ),
            "profile_url": staff.get(
                "profile_url"
            ),
            "recommendation_role": "primary_internal",
            "staff_hierarchy_paths": (
                self.staff_mapping_by_id[staff_id]
                .get("resolved_paths", [])
            ),
            "matched_internal_paths": (
                staff_internal_paths
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

    def rank_module(
        self,
        module_id: str,
    ) -> dict[str, Any]:
        module = self.module_by_id.get(module_id)

        if module is None:
            raise ValueError(
                f"Unknown module ID: {module_id}"
            )

        module_mapping = (
            self.module_mapping_by_id.get(module_id)
        )

        if module_mapping is None:
            raise ValueError(
                f"Missing hierarchy mapping for "
                f"{module_id}"
            )

        module_paths = module_mapping.get(
            "resolved_paths",
            [],
        )

        if not module_paths:
            return {
                "module_id": module_id,
                "module_code": module.get(
                    "module_code"
                ),
                "module_title": module.get(
                    "module_title"
                ),
                "module_hierarchy_paths": [],
                "policy": "hierarchy_internal_only",
                "decision_status": (
                    "module_hierarchy_unresolved"
                ),
                "review_required": True,
                "recommendations": [],
            }

        if module_id not in (
            self.module_position_by_id
        ):
            return {
                "module_id": module_id,
                "module_code": module.get(
                    "module_code"
                ),
                "module_title": module.get(
                    "module_title"
                ),
                "module_hierarchy_paths": (
                    module_paths
                ),
                "policy": "hierarchy_internal_only",
                "decision_status": (
                    "module_missing_semantic_embedding"
                ),
                "review_required": True,
                "recommendations": [],
            }

        module_position = (
            self.module_position_by_id[module_id]
        )

        internal_path_matches = (
            self._internal_candidate_paths(
                module_paths
            )
        )

        eligible_candidates: list[
            tuple[int, list[str], str]
        ] = []

        raw_internal_count = len(
            internal_path_matches
        )

        for staff_id, matching_paths in (
            internal_path_matches.items()
        ):
            staff = self.staff_by_id.get(staff_id)

            if staff is None:
                continue

            eligibility = (
                primary_delivery_eligibility(staff)
            )

            if not eligibility.eligible:
                continue

            staff_position = (
                self.staff_position_by_id.get(staff_id)
            )

            if staff_position is None:
                continue

            eligible_candidates.append(
                (
                    staff_position,
                    matching_paths,
                    eligibility.reason,
                )
            )

        eligible_before_embedding = sum(
            primary_delivery_eligibility(
                self.staff_by_id[staff_id]
            ).eligible
            for staff_id in internal_path_matches
            if staff_id in self.staff_by_id
        )

        eligible_candidates.sort(
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

        recommendations = [
            self._candidate_record(
                module_position=module_position,
                staff_position=staff_position,
                rank=rank,
                staff_internal_paths=matching_paths,
            )
            for rank, (
                staff_position,
                matching_paths,
                _eligibility_reason,
            ) in enumerate(
                eligible_candidates[:self.top_k],
                start=1,
            )
        ]

        if recommendations:
            if len(recommendations) >= self.top_k:
                status = "hierarchy_internal_ranked"
            else:
                status = "limited_hierarchy_internal_pool"

            review_required = (
                len(recommendations) < self.top_k
            )
        elif raw_internal_count == 0:
            status = "no_hierarchy_internal_staff"
            review_required = True
        elif eligible_before_embedding == 0:
            status = "no_eligible_hierarchy_staff"
            review_required = True
        else:
            status = "no_hierarchy_staff_with_m2_evidence"
            review_required = True

        return {
            "module_id": module_id,
            "module_code": module.get(
                "module_code"
            ),
            "module_title": module.get(
                "module_title"
            ),
            "module_departments": module.get(
                "departments",
                [],
            ),
            "module_colleges": module.get(
                "colleges",
                [],
            ),
            "module_hierarchy_paths": module_paths,
            "policy": "hierarchy_internal_only",
            "semantic_model": (
                self.score_bundle.model_name
            ),
            "decision_status": status,
            "review_required": review_required,
            "diagnostics": {
                "raw_internal_staff_count": (
                    raw_internal_count
                ),
                "eligible_internal_staff_count": (
                    eligible_before_embedding
                ),
                "fitted_internal_staff_count": len(
                    eligible_candidates
                ),
                "returned_candidate_count": len(
                    recommendations
                ),
            },
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
            "fully_ranked_modules": sum(
                result["decision_status"]
                == "hierarchy_internal_ranked"
                for result in results
            ),
            "limited_pool_modules": sum(
                result["decision_status"]
                == "limited_hierarchy_internal_pool"
                for result in results
            ),
            "review_required_modules": sum(
                result["review_required"]
                for result in results
            ),
            "total_recommendations": sum(
                len(result["recommendations"])
                for result in results
            ),
        }