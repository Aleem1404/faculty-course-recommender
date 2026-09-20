from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from faculty_recommender.policy.departments import (
    canonicalise_department,
    canonicalise_departments,
    load_department_aliases,
    normalise_department_label,
)

from faculty_recommender.policy.eligibility import (
    primary_delivery_eligibility,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line "
                    f"{line_number}"
                ) from error

            if isinstance(record, dict):
                records.append(record)

    return records


def normalise_text(value: Any) -> str:
    if value is None:
        return ""

    return " ".join(str(value).split()).strip()


def normalised_values(value: Any) -> set[str]:
    if value is None:
        return set()

    if not isinstance(value, list):
        value = [value]

    return {
        normalise_department_label(item)
        for item in value
        if normalise_department_label(item)
    }


def staff_preview(
    staff: dict[str, Any],
    eligibility_reason: str,
    m2_fitted: bool,
    match_reason: str,
    department_similarity: float | None = None,
) -> dict[str, Any]:
    return {
        "staff_id": staff.get("staff_id"),
        "full_name": staff.get("full_name"),
        "position": staff.get("position"),
        "department_name": staff.get(
            "department_name"
        ),
        "college_name": staff.get(
            "college_name"
        ),
        "profile_url": staff.get("profile_url"),
        "primary_eligible": (
            primary_delivery_eligibility(
                staff
            ).eligible
        ),
        "eligibility_reason": eligibility_reason,
        "m2_fitted": m2_fitted,
        "match_reason": match_reason,
        "department_similarity": (
            round(department_similarity, 2)
            if department_similarity is not None
            else None
        ),
        "has_core_profile": bool(
            staff.get("core_profile_text")
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
        "has_publication_abstracts": bool(
            staff.get(
                "publication_abstract_text"
            )
        ),
        "publication_topic_names": staff.get(
            "publication_topic_names",
            [],
        )[:20],
        "publication_keyword_names": staff.get(
            "publication_keyword_names",
            [],
        )[:20],
    }


def determine_cause(
    exact_department_count: int,
    canonical_department_count: int,
    same_college_count: int,
    eligible_internal_count: int,
    fitted_internal_count: int,
) -> str:
    if exact_department_count == 0:
        if same_college_count > 0:
            return (
                "probable_organisational_mapping_gap"
            )

        return "no_staff_found_for_owning_unit"

    if canonical_department_count == 0:
        return "department_alias_mapping_failure"

    if eligible_internal_count == 0:
        return "position_eligibility_failure"

    if fitted_internal_count == 0:
        return "missing_semantic_evidence"

    return "unexpected_policy_or_index_failure"


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

    result_path = (
        project_root
        / "data"
        / "outputs"
        / "recommendations"
        / "m2_primary"
        / "m2_primary_recommendations.jsonl"
    )

    semantic_index_path = (
        project_root
        / "data"
        / "cache"
        / "embeddings"
        / "semantic"
        / "semantic_embedding_index.json"
    )

    aliases_path = (
        project_root
        / "config"
        / "department_aliases.yaml"
    )

    output_directory = (
        project_root
        / "data"
        / "processed"
        / "audit"
        / "m2_primary"
    )

    staff_records = load_jsonl(staff_path)
    module_records = load_jsonl(module_path)
    recommendation_results = load_jsonl(
        result_path
    )

    semantic_index = json.loads(
        semantic_index_path.read_text(
            encoding="utf-8"
        )
    )

    fitted_staff_ids = set(
        semantic_index.get("staff_ids", [])
    )

    aliases = load_department_aliases(
        aliases_path
    )

    modules_by_id = {
        record["module_id"]: record
        for record in module_records
        if record.get("module_id")
    }

    # fallback_results = [
    #     result
    #     for result in recommendation_results
    #     if result.get("decision_status")
    #     == "university_fallback_ranked"
    #     or result.get("fallback_used") is True
    # ]

    REVIEW_STATUSES = {
    "university_fallback_ranked",
    "no_eligible_internal_staff",
    "owning_department_missing",
    "internal_evidence_insufficient",
    "manual_review_required",
    }

    fallback_results = [
        result
        for result in recommendation_results
        if (
            result.get("decision_status")
            in REVIEW_STATUSES
            or result.get("fallback_used") is True
        )
    ]

    audit_records = []
    cause_counter: Counter[str] = Counter()
    department_counter: Counter[str] = Counter()
    external_staff_counter: Counter[str] = Counter()

    for fallback_result in fallback_results:
        module_id = fallback_result.get(
            "module_id"
        )

        module = modules_by_id.get(
            module_id,
            fallback_result,
        )

        module_departments = module.get(
            "departments",
            fallback_result.get(
                "owning_departments",
                [],
            ),
        )

        module_colleges = module.get(
            "colleges",
            [],
        )

        raw_module_departments = (
            normalised_values(
                module_departments
            )
        )

        raw_module_colleges = normalised_values(
            module_colleges
        )

        canonical_module_departments = (
            canonicalise_departments(
                module_departments,
                aliases,
            )
        )

        exact_matches = []
        canonical_matches = []
        same_college_profiles = []
        possible_department_matches = []

        for staff in staff_records:
            staff_id = normalise_text(
                staff.get("staff_id")
            )

            staff_department_raw = (
                normalise_department_label(
                    staff.get("department_name")
                )
            )

            staff_college_raw = (
                normalise_department_label(
                    staff.get("college_name")
                )
            )

            staff_department_canonical = (
                canonicalise_department(
                    staff.get("department_name"),
                    aliases,
                )
            )

            eligibility = (
                primary_delivery_eligibility(
                    staff
                )
            )

            is_m2_fitted = (
                staff_id in fitted_staff_ids
            )

            if (
                staff_department_raw
                in raw_module_departments
            ):
                exact_matches.append(
                    staff_preview(
                        staff=staff,
                        eligibility_reason=(
                            eligibility.reason
                        ),
                        m2_fitted=is_m2_fitted,
                        match_reason=(
                            "exact_department_label"
                        ),
                    )
                )

            if (
                staff_department_canonical
                in canonical_module_departments
            ):
                canonical_matches.append(
                    staff_preview(
                        staff=staff,
                        eligibility_reason=(
                            eligibility.reason
                        ),
                        m2_fitted=is_m2_fitted,
                        match_reason=(
                            "canonical_department"
                        ),
                    )
                )

            if (
                staff_college_raw
                and staff_college_raw
                in raw_module_colleges
            ):
                same_college_profiles.append(
                    staff_preview(
                        staff=staff,
                        eligibility_reason=(
                            eligibility.reason
                        ),
                        m2_fitted=is_m2_fitted,
                        match_reason="same_college",
                    )
                )

            if raw_module_departments:
                similarity = max(
                    fuzz.token_set_ratio(
                        staff_department_raw,
                        module_department,
                    )
                    for module_department
                    in raw_module_departments
                )
            else:
                similarity = 0

            if similarity >= 55:
                possible_department_matches.append(
                    staff_preview(
                        staff=staff,
                        eligibility_reason=(
                            eligibility.reason
                        ),
                        m2_fitted=is_m2_fitted,
                        match_reason=(
                            "similar_department_label"
                        ),
                        department_similarity=(
                            float(similarity)
                        ),
                    )
                )

        eligible_internal = [
            profile
            for profile in canonical_matches
            if profile["primary_eligible"]
        ]

        fitted_internal = [
            profile
            for profile in eligible_internal
            if profile["m2_fitted"]
        ]

        possible_department_matches.sort(
            key=lambda profile: (
                -(
                    profile[
                        "department_similarity"
                    ]
                    or 0
                ),
                profile["department_name"]
                or "",
                profile["full_name"] or "",
            )
        )

        cause = determine_cause(
            exact_department_count=len(
                exact_matches
            ),
            canonical_department_count=len(
                canonical_matches
            ),
            same_college_count=len(
                same_college_profiles
            ),
            eligible_internal_count=len(
                eligible_internal
            ),
            fitted_internal_count=len(
                fitted_internal
            ),
        )

        cause_counter[cause] += 1

        for department in module_departments:
            department_counter[
                normalise_text(department)
            ] += 1

        current_external_recommendations = []

        for recommendation in fallback_result.get(
            "recommendations",
            [],
        ):
            staff_id = recommendation.get(
                "staff_id"
            )

            external_staff_counter[
                staff_id
            ] += 1

            current_external_recommendations.append({
                "rank": recommendation.get(
                    "rank"
                ),
                "staff_id": staff_id,
                "full_name": recommendation.get(
                    "full_name"
                ),
                "position": recommendation.get(
                    "position"
                ),
                "department_name": (
                    recommendation.get(
                        "department_name"
                    )
                ),
                "college_name": recommendation.get(
                    "college_name"
                ),
                "semantic_score": (
                    recommendation.get(
                        "semantic_score"
                    )
                ),
                "profile_url": recommendation.get(
                    "profile_url"
                ),
            })

        audit_records.append({
            "module_id": module_id,
            "module_code": module.get(
                "module_code"
            ),
            "module_title": module.get(
                "module_title"
            ),
            "module_colleges": module_colleges,
            "module_departments": (
                module_departments
            ),
            "current_canonical_departments": (
                sorted(
                    canonical_module_departments
                )
            ),
            "diagnosis": {
                "staff_with_exact_department_label": (
                    len(exact_matches)
                ),
                "staff_with_canonical_department": (
                    len(canonical_matches)
                ),
                "staff_in_same_college": len(
                    same_college_profiles
                ),
                "eligible_internal_staff": len(
                    eligible_internal
                ),
                "internal_staff_with_m2_evidence": (
                    len(fitted_internal)
                ),
                "probable_cause": cause,
            },
            "exact_department_profiles": (
                exact_matches
            ),
            "canonical_department_profiles": (
                canonical_matches
            ),
            "possible_department_profiles": (
                possible_department_matches[
                    :30
                ]
            ),
            "same_college_profiles": (
                same_college_profiles[:50]
            ),
            "current_external_recommendations": (
                current_external_recommendations
            ),
            "manual_review": {
                "reviewed": False,
                "confirmed_cause": "",
                "expected_college_id": "",
                "expected_department_id": "",
                "expected_division_id": "",
                "notes": "",
            },
        })

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    detailed_output = (
        output_directory
        / "fallback_review.json"
    )

    detailed_output.write_text(
        json.dumps(
            {
                "summary": {
                    "fallback_module_count": len(
                        audit_records
                    ),
                    "probable_cause_counts": dict(
                        cause_counter.most_common()
                    ),
                    "fallback_department_counts": (
                        dict(
                            department_counter
                            .most_common()
                        )
                    ),
                },
                "fallback_modules": audit_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    unique_external_profiles = []

    for staff_id, count in (
        external_staff_counter.most_common()
    ):
        staff = next(
            (
                record
                for record in staff_records
                if record.get("staff_id")
                == staff_id
            ),
            None,
        )

        if staff is None:
            continue

        eligibility = (
            primary_delivery_eligibility(staff)
        )

        profile = staff_preview(
            staff=staff,
            eligibility_reason=(
                eligibility.reason
            ),
            m2_fitted=(
                staff_id in fitted_staff_ids
            ),
            match_reason=(
                "current_fallback_recommendation"
            ),
        )

        profile[
            "fallback_recommendation_count"
        ] = count

        unique_external_profiles.append(profile)

    profiles_output = (
        output_directory
        / "fallback_recommended_profiles.json"
    )

    profiles_output.write_text(
        json.dumps(
            unique_external_profiles,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {
        # "fallback_module_count": len(
        #     audit_records
        # ),
        "review_module_count": len(
            audit_records
        ),
        "unique_external_staff_recommended": (
            len(unique_external_profiles)
        ),
        "probable_cause_counts": dict(
            cause_counter.most_common()
        ),
        "fallback_department_counts": dict(
            department_counter.most_common()
        ),
        "detailed_output": str(
            detailed_output
        ),
        "profiles_output": str(
            profiles_output
        ),
    }

    summary_output = (
        output_directory
        / "fallback_audit_summary.json"
    )

    summary_output.write_text(
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