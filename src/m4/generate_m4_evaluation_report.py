from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_TOPIC_PROFILES_SUMMARY = Path(
    "data/outputs/m4/topic_profiles_summary.json"
)

DEFAULT_QUALITY_FILTER_SUMMARY = Path(
    "data/outputs/m4/topic_quality_filter_summary.json"
)

DEFAULT_CANDIDATES_SUMMARY = Path(
    "data/outputs/m4/collaboration_candidates_summary.json"
)

DEFAULT_RECOMMENDATIONS_SUMMARY = Path(
    "data/outputs/m4/collaboration_recommendations_summary.json"
)

DEFAULT_METADATA_SUMMARY = Path(
    "data/outputs/m4/entity_metadata_summary.json"
)

DEFAULT_CASE_STUDIES_SUMMARY = Path(
    "data/outputs/m4/m4_case_studies_summary.json"
)

DEFAULT_STRUCTURAL_VALIDATION_REPORT = Path(
    "data/outputs/m4/m4_structural_validation_report.json"
)

DEFAULT_SEMANTIC_FOUNDATION_SUMMARY = Path(
    "data/outputs/m4/semantic_shared_foundation_summary.json"
)

DEFAULT_OUTPUT_JSON = Path(
    "data/outputs/m4/m4_evaluation_report.json"
)

DEFAULT_OUTPUT_MARKDOWN = Path(
    "data/outputs/m4/m4_evaluation_report.md"
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required M4 summary file was not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected JSON object in {path}, "
            f"but found {type(payload).__name__}."
        )

    return payload


def optional_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "_available": False,
            "_missing_path": str(path),
        }

    payload = read_json(path)
    payload["_available"] = True

    return payload


def as_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback

    text = str(value).strip()

    return text or fallback


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percent(value: Any, decimal_places: int = 1) -> str:
    return f"{safe_float(value) * 100:.{decimal_places}f}%"


def number(value: Any, decimal_places: int = 3) -> str:
    return f"{safe_float(value):.{decimal_places}f}"


def get_nested(
    payload: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    current: Any = payload

    for key in keys:
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def markdown_table(
    headers: list[str],
    rows: list[list[str]],
) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(
        "---" for _ in headers
    ) + " |"

    body_rows = [
        "| " + " | ".join(row) + " |"
        for row in rows
    ]

    return "\n".join(
        [header_row, separator_row, *body_rows]
    )


def build_report(
    topic_profiles: dict[str, Any],
    quality_filter: dict[str, Any],
    candidates: dict[str, Any],
    recommendations: dict[str, Any],
    metadata: dict[str, Any],
    case_studies: dict[str, Any],
    structural_validation: dict[str, Any],
    semantic_foundation: dict[str, Any],
    input_paths: dict[str, Path],
) -> dict[str, Any]:
    profile_counts = get_nested(
        topic_profiles,
        "counts",
        default={},
    )

    quality_staff = get_nested(
        quality_filter,
        "staff",
        default={},
    )

    quality_modules = get_nested(
        quality_filter,
        "modules",
        default={},
    )

    candidate_counts = get_nested(
        candidates,
        "counts",
        default={},
    )

    recommendation_counts = get_nested(
        recommendations,
        "counts",
        default={},
    )

    metadata_coverage = get_nested(
        metadata,
        "m4_profile_id_coverage",
        default={},
    )

    structural_counts = get_nested(
        structural_validation,
        "counts",
        default={},
    )

    structural_rates = get_nested(
        structural_validation,
        "rates",
        default={},
    )

    distributions = get_nested(
        structural_validation,
        "numeric_distributions",
        default={},
    )

    display_validation = get_nested(
        structural_validation,
        "display_policy_validation",
        default={},
    )

    case_validation = get_nested(
        structural_validation,
        "case_study_validation",
        default={},
    )

    failure_summary = get_nested(
        structural_validation,
        "failure_summary",
        default={},
    )

    semantic_module_comparison = get_nested(
        semantic_foundation,
        "top_ranked_module_comparison",
        default={},
    )

    semantic_counts = get_nested(
        semantic_foundation,
        "recommendation_foundation_counts",
        default={},
    )

    semantic_parameters = get_nested(
        semantic_foundation,
        "parameters",
        default={},
    )

    parameter_summary = {
        "topic_normalisation_embedding_model": (
            get_nested(
                topic_profiles,
                "embedding_model",
                default="all-MiniLM-L6-v2",
            )
        ),
        "topic_normalisation_similarity_threshold": (
            get_nested(
                topic_profiles,
                "similarity_threshold",
                default=0.84,
            )
        ),
        "minimum_staff_relevance": get_nested(
            candidates,
            "parameters",
            "min_staff_relevance",
            default=0.10,
        ),
        "maximum_individual_candidates": get_nested(
            candidates,
            "parameters",
            "max_individual_candidates",
            default=20,
        ),
        "minimum_complementarity_gain": get_nested(
            candidates,
            "parameters",
            "min_complementarity",
            default=0.05,
        ),
        "maximum_pairs_per_module": get_nested(
            candidates,
            "parameters",
            "max_pairs_per_module",
            default=50,
        ),
        "display_minimum_combined_coverage": get_nested(
            structural_validation,
            "parameters",
            "display_minimum_combined_coverage",
            default=0.40,
        ),
        "display_minimum_complementarity_gain": get_nested(
            structural_validation,
            "parameters",
            "display_minimum_complementarity_gain",
            default=0.10,
        ),
        "semantic_shared_foundation_threshold": get_nested(
            semantic_parameters,
            "semantic_similarity_threshold",
            default=0.80,
        ),
    }

    report = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "report_scope": {
            "title": (
                "M4 Collaboration Recommendation Evaluation Report"
            ),
            "purpose": (
                "This report consolidates technical, structural, "
                "coverage, explainability and semantic-foundation "
                "evaluation evidence for the M4 collaboration "
                "recommendation component."
            ),
            "evaluation_boundary": (
                "The results validate structural correctness and "
                "traceable recommendation evidence. They do not "
                "establish domain-level teaching suitability across "
                "all subject areas, which requires relevant "
                "academic or course-leader judgement."
            ),
        },
        "source_files": {
            key: str(path)
            for key, path in input_paths.items()
        },
        "parameters": parameter_summary,
        "data_and_profile_coverage": {
            "topic_assignments_processed": safe_int(
                profile_counts.get("assignment_records_read")
            ),
            "staff_topic_profiles": safe_int(
                profile_counts.get("staff_profiles")
            ),
            "module_topic_profiles": safe_int(
                profile_counts.get("module_profiles")
            ),
            "average_topics_per_staff": safe_float(
                profile_counts.get("average_topics_per_staff")
            ),
            "average_topics_per_module": safe_float(
                profile_counts.get("average_topics_per_module")
            ),
            "staff_topics_retained_after_filtering": safe_int(
                quality_staff.get("accepted_topics")
            ),
            "staff_topics_rejected_by_filtering": safe_int(
                quality_staff.get("rejected_topics")
            ),
            "module_topics_retained_after_filtering": safe_int(
                quality_modules.get("accepted_topics")
            ),
            "module_topics_rejected_by_filtering": safe_int(
                quality_modules.get("rejected_topics")
            ),
            "staff_metadata_coverage_rate": safe_float(
                get_nested(
                    metadata_coverage,
                    "staff",
                    "coverage_rate",
                    default=0.0,
                )
            ),
            "module_metadata_coverage_rate": safe_float(
                get_nested(
                    metadata_coverage,
                    "modules",
                    "coverage_rate",
                    default=0.0,
                )
            ),
        },
        "candidate_generation": {
            "staff_profiles_loaded": safe_int(
                candidate_counts.get("staff_profiles_loaded")
            ),
            "module_profiles_loaded": safe_int(
                candidate_counts.get("module_profiles_loaded")
            ),
            "modules_with_collaboration_candidates": safe_int(
                candidate_counts.get(
                    "modules_with_collaboration_candidates"
                )
            ),
            "modules_without_collaboration_candidates": safe_int(
                candidate_counts.get(
                    "modules_without_collaboration_candidates"
                )
            ),
            "pair_candidates_before_limit": safe_int(
                candidate_counts.get(
                    "pair_candidates_before_per_module_limit"
                )
            ),
            "collaboration_candidates_written": safe_int(
                candidate_counts.get("pair_candidates_written")
            ),
        },
        "recommendation_output": {
            "recommendations_written": safe_int(
                recommendation_counts.get("recommendations_written")
            ),
            "recommendation_levels": get_nested(
                recommendation_counts,
                "recommendation_levels",
                default={},
            ),
            "strict_display_recommendations": safe_int(
                case_studies.get(
                    "counts",
                    {},
                ).get(
                    "display_recommendations_written",
                    0,
                )
            ),
            "case_studies_selected": safe_int(
                case_studies.get(
                    "counts",
                    {},
                ).get(
                    "case_studies_written",
                    0,
                )
            ),
        },
        "structural_validation": {
            "recommendations_checked": safe_int(
                structural_counts.get("recommendations_checked")
            ),
            "modules_with_recommendations": safe_int(
                structural_counts.get(
                    "modules_with_recommendations"
                )
            ),
            "positive_complementarity_rate": safe_float(
                structural_rates.get(
                    "positive_complementarity_rate"
                )
            ),
            "topic_evidence_rate": safe_float(
                structural_rates.get("topic_evidence_rate")
            ),
            "complete_explanation_rate": safe_float(
                structural_rates.get(
                    "complete_explanation_rate"
                )
            ),
            "modules_with_shared_foundation_rate": safe_float(
                structural_rates.get(
                    "modules_with_shared_foundation_rate"
                )
            ),
            "record_level_failure_count": safe_int(
                structural_counts.get(
                    "record_level_failure_count"
                )
            ),
            "display_policy_compliant": bool(
                display_validation.get(
                    "policy_compliant",
                    False,
                )
            ),
            "display_policy_failure_count": safe_int(
                display_validation.get("failure_count")
            ),
            "total_failure_rows": safe_int(
                failure_summary.get("total_failure_rows")
            ),
            "failure_counts": failure_summary.get(
                "failure_counts",
                {},
            ),
        },
        "numeric_results": {
            "collaboration_score": distributions.get(
                "collaboration_score",
                {},
            ),
            "combined_module_coverage": distributions.get(
                "combined_module_coverage",
                {},
            ),
            "best_individual_coverage": distributions.get(
                "best_individual_coverage",
                {},
            ),
            "complementarity_gain": distributions.get(
                "complementarity_gain",
                {},
            ),
            "staff_topic_overlap_ratio": distributions.get(
                "staff_topic_overlap_ratio",
                {},
            ),
            "combined_module_coverage_bands": distributions.get(
                "combined_module_coverage_bands",
                {},
            ),
            "complementarity_gain_bands": distributions.get(
                "complementarity_gain_bands",
                {},
            ),
        },
        "case_study_evaluation_set": {
            "case_studies_checked": safe_int(
                case_validation.get("case_studies_checked")
            ),
            "selection_group_counts": case_validation.get(
                "selection_group_counts",
                {},
            ),
            "recommendation_level_counts": case_validation.get(
                "recommendation_level_counts",
                {},
            ),
            "duplicate_case_id_count": safe_int(
                case_validation.get("duplicate_case_id_count")
            ),
            "duplicate_module_id_count": safe_int(
                case_validation.get("duplicate_module_id_count")
            ),
            "duplicate_module_ids": case_validation.get(
                "duplicate_module_ids",
                [],
            ),
        },
        "semantic_shared_foundation_analysis": {
            "available": semantic_foundation.get(
                "_available",
                False,
            ),
            "threshold": safe_float(
                semantic_parameters.get(
                    "semantic_similarity_threshold"
                )
            ),
            "recommendations_processed": safe_int(
                semantic_counts.get("recommendations_processed")
            ),
            "exact_foundation_recommendations": safe_int(
                semantic_counts.get(
                    "exact_foundation_recommendations"
                )
            ),
            "semantic_only_foundation_recommendations": safe_int(
                semantic_counts.get(
                    "semantic_only_foundation_recommendations"
                )
            ),
            "no_foundation_recommendations": safe_int(
                semantic_counts.get(
                    "no_foundation_recommendations"
                )
            ),
            "top_ranked_modules": safe_int(
                semantic_module_comparison.get(
                    "modules_with_top_ranked_recommendation"
                )
            ),
            "exact_only_foundation_rate": safe_float(
                semantic_module_comparison.get(
                    "exact_only_foundation_rate"
                )
            ),
            "exact_plus_semantic_foundation_rate": safe_float(
                semantic_module_comparison.get(
                    "exact_plus_semantic_foundation_rate"
                )
            ),
            "additional_modules_supported_by_semantic_evidence": (
                safe_int(
                    semantic_module_comparison.get(
                        "additional_modules_supported_by_semantic_evidence"
                    )
                )
            ),
            "interpretation": (
                "Semantic shared-foundation analysis is supplementary. "
                "It adds explainable evidence but does not alter the "
                "original M4 collaboration score, rank, topic coverage "
                "or complementarity-gain calculation."
            ),
        },
        "key_findings": [],
        "limitations": [
            (
                "The recommendations are based on publicly available "
                "course and staff-profile information, which may be "
                "incomplete or outdated."
            ),
            (
                "Topic extraction and canonicalisation may retain "
                "awkward, overly broad or imperfectly ordered phrases."
            ),
            (
                "The framework does not include staff availability, "
                "workload, timetabling, contractual responsibilities, "
                "teaching preferences or institutional allocation rules."
            ),
            (
                "Structural validation confirms internal consistency "
                "and explanation traceability, not domain-level "
                "teaching suitability."
            ),
            (
                "Domain-level assessment should be performed by "
                "module leaders, subject specialists or academic "
                "managers in future work."
            ),
        ],
    }

    structural = report["structural_validation"]
    numeric = report["numeric_results"]
    semantic = report["semantic_shared_foundation_analysis"]

    report["key_findings"] = [
        (
            f"M4 produced {report['recommendation_output']['recommendations_written']:,} "
            f"collaboration recommendations across "
            f"{structural['modules_with_recommendations']:,} modules."
        ),
        (
            f"All retained recommendations had positive "
            f"complementarity gain "
            f"({percent(structural['positive_complementarity_rate'])}) "
            f"and traceable topic evidence "
            f"({percent(structural['topic_evidence_rate'])})."
        ),
        (
            f"All recommendations contained complete readable "
            f"explanations "
            f"({percent(structural['complete_explanation_rate'])})."
        ),
        (
            f"The mean combined module coverage was "
            f"{number(get_nested(numeric, 'combined_module_coverage', 'mean'))}, "
            f"while mean best individual coverage was "
            f"{number(get_nested(numeric, 'best_individual_coverage', 'mean'))}."
        ),
        (
            f"The mean complementarity gain was "
            f"{number(get_nested(numeric, 'complementarity_gain', 'mean'))}."
        ),
        (
            f"Structural validation recorded "
            f"{structural['record_level_failure_count']} record-level "
            f"failures and the display policy was "
            f"{'compliant' if structural['display_policy_compliant'] else 'not compliant'}."
        ),
        (
            f"At semantic threshold "
            f"{number(semantic['threshold'], 2)}, "
            f"{semantic['semantic_only_foundation_recommendations']:,} "
            f"recommendations gained semantic-only "
            f"shared-foundation evidence."
        ),
    ]

    return report


def build_markdown_report(report: dict[str, Any]) -> str:
    coverage = report["data_and_profile_coverage"]
    candidates = report["candidate_generation"]
    output = report["recommendation_output"]
    structural = report["structural_validation"]
    numeric = report["numeric_results"]
    cases = report["case_study_evaluation_set"]
    semantic = report["semantic_shared_foundation_analysis"]
    parameters = report["parameters"]

    markdown: list[str] = []

    markdown.append("# M4 Collaboration Evaluation Report")
    markdown.append("")

    markdown.append(
        "This report consolidates the technical and structural "
        "evaluation of the M4 faculty-collaboration recommendation "
        "extension. It validates internal correctness, coverage, "
        "traceability, explainability and complementarity evidence. "
        "It does not claim domain-level teaching suitability across "
        "all academic disciplines."
    )
    markdown.append("")

    markdown.append("## Configuration")
    markdown.append("")

    markdown.append(
        markdown_table(
            ["Parameter", "Value"],
            [
                [
                    "Topic embedding model",
                    as_text(
                        parameters.get(
                            "topic_normalisation_embedding_model"
                        )
                    ),
                ],
                [
                    "Topic normalisation similarity threshold",
                    number(
                        parameters.get(
                            "topic_normalisation_similarity_threshold"
                        ),
                        2,
                    ),
                ],
                [
                    "Minimum individual staff relevance",
                    number(
                        parameters.get(
                            "minimum_staff_relevance"
                        ),
                        2,
                    ),
                ],
                [
                    "Maximum individual candidates per module",
                    str(
                        safe_int(
                            parameters.get(
                                "maximum_individual_candidates"
                            )
                        )
                    ),
                ],
                [
                    "Minimum complementarity gain",
                    number(
                        parameters.get(
                            "minimum_complementarity_gain"
                        ),
                        2,
                    ),
                ],
                [
                    "Display minimum combined coverage",
                    number(
                        parameters.get(
                            "display_minimum_combined_coverage"
                        ),
                        2,
                    ),
                ],
                [
                    "Display minimum complementarity gain",
                    number(
                        parameters.get(
                            "display_minimum_complementarity_gain"
                        ),
                        2,
                    ),
                ],
                [
                    "Semantic shared-foundation threshold",
                    number(
                        parameters.get(
                            "semantic_shared_foundation_threshold"
                        ),
                        2,
                    ),
                ],
            ],
        )
    )
    markdown.append("")

    markdown.append("## Data Preparation")
    markdown.append("")

    markdown.append(
        markdown_table(
            ["Measure", "Result"],
            [
                [
                    "Topic assignments processed",
                    f"{safe_int(coverage.get('topic_assignments_processed')):,}",
                ],
                [
                    "Staff topic profiles",
                    f"{safe_int(coverage.get('staff_topic_profiles')):,}",
                ],
                [
                    "Module topic profiles",
                    f"{safe_int(coverage.get('module_topic_profiles')):,}",
                ],
                [
                    "Average topics per staff profile",
                    number(
                        coverage.get(
                            "average_topics_per_staff"
                        )
                    ),
                ],
                [
                    "Average topics per module profile",
                    number(
                        coverage.get(
                            "average_topics_per_module"
                        )
                    ),
                ],
                [
                    "Staff metadata coverage",
                    percent(
                        coverage.get(
                            "staff_metadata_coverage_rate"
                        )
                    ),
                ],
                [
                    "Module metadata coverage",
                    percent(
                        coverage.get(
                            "module_metadata_coverage_rate"
                        )
                    ),
                ],
            ],
        )
    )
    markdown.append("")

    markdown.append("## Candidate Generation")
    markdown.append("")

    markdown.append(
        markdown_table(
            ["Measure", "Result"],
            [
                [
                    "Staff profiles loaded",
                    f"{safe_int(candidates.get('staff_profiles_loaded')):,}",
                ],
                [
                    "Module profiles loaded",
                    f"{safe_int(candidates.get('module_profiles_loaded')):,}",
                ],
                [
                    "Modules with collaboration candidates",
                    f"{safe_int(candidates.get('modules_with_collaboration_candidates')):,}",
                ],
                [
                    "Modules without collaboration candidates",
                    f"{safe_int(candidates.get('modules_without_collaboration_candidates')):,}",
                ],
                [
                    "Pair candidates before output limit",
                    f"{safe_int(candidates.get('pair_candidates_before_limit')):,}",
                ],
                [
                    "Collaboration recommendations written",
                    f"{safe_int(candidates.get('collaboration_candidates_written')):,}",
                ],
            ],
        )
    )
    markdown.append("")

    markdown.append("## Structural Validation")
    markdown.append("")

    markdown.append(
        markdown_table(
            ["Validation measure", "Result"],
            [
                [
                    "Recommendations checked",
                    f"{safe_int(structural.get('recommendations_checked')):,}",
                ],
                [
                    "Modules with recommendations",
                    f"{safe_int(structural.get('modules_with_recommendations')):,}",
                ],
                [
                    "Positive complementarity rate",
                    percent(
                        structural.get(
                            "positive_complementarity_rate"
                        )
                    ),
                ],
                [
                    "Topic-evidence rate",
                    percent(
                        structural.get("topic_evidence_rate")
                    ),
                ],
                [
                    "Complete-explanation rate",
                    percent(
                        structural.get(
                            "complete_explanation_rate"
                        )
                    ),
                ],
                [
                    "Record-level structural failures",
                    str(
                        safe_int(
                            structural.get(
                                "record_level_failure_count"
                            )
                        )
                    ),
                ],
                [
                    "Display-policy compliant",
                    "Yes"
                    if structural.get(
                        "display_policy_compliant"
                    )
                    else "No",
                ],
                [
                    "Display-policy failures",
                    str(
                        safe_int(
                            structural.get(
                                "display_policy_failure_count"
                            )
                        )
                    ),
                ],
            ],
        )
    )
    markdown.append("")

    markdown.append("## Ranking Results")
    markdown.append("")

    markdown.append(
        markdown_table(
            ["Metric", "Minimum", "Mean", "Median", "Maximum"],
            [
                [
                    "Collaboration score",
                    number(
                        get_nested(
                            numeric,
                            "collaboration_score",
                            "minimum",
                        )
                    ),
                    number(
                        get_nested(
                            numeric,
                            "collaboration_score",
                            "mean",
                        )
                    ),
                    number(
                        get_nested(
                            numeric,
                            "collaboration_score",
                            "median",
                        )
                    ),
                    number(
                        get_nested(
                            numeric,
                            "collaboration_score",
                            "maximum",
                        )
                    ),
                ],
                [
                    "Combined module coverage",
                    number(
                        get_nested(
                            numeric,
                            "combined_module_coverage",
                            "minimum",
                        )
                    ),
                    number(
                        get_nested(
                            numeric,
                            "combined_module_coverage",
                            "mean",
                        )
                    ),
                    number(
                        get_nested(
                            numeric,
                            "combined_module_coverage",
                            "median",
                        )
                    ),
                    number(
                        get_nested(
                            numeric,
                            "combined_module_coverage",
                            "maximum",
                        )
                    ),
                ],
                [
                    "Best individual coverage",
                    number(
                        get_nested(
                            numeric,
                            "best_individual_coverage",
                            "minimum",
                        )
                    ),
                    number(
                        get_nested(
                            numeric,
                            "best_individual_coverage",
                            "mean",
                        )
                    ),
                    number(
                        get_nested(
                            numeric,
                            "best_individual_coverage",
                            "median",
                        )
                    ),
                    number(
                        get_nested(
                            numeric,
                            "best_individual_coverage",
                            "maximum",
                        )
                    ),
                ],
                [
                    "Complementarity gain",
                    number(
                        get_nested(
                            numeric,
                            "complementarity_gain",
                            "minimum",
                        )
                    ),
                    number(
                        get_nested(
                            numeric,
                            "complementarity_gain",
                            "mean",
                        )
                    ),
                    number(
                        get_nested(
                            numeric,
                            "complementarity_gain",
                            "median",
                        )
                    ),
                    number(
                        get_nested(
                            numeric,
                            "complementarity_gain",
                            "maximum",
                        )
                    ),
                ],
                [
                    "Staff-topic overlap ratio",
                    number(
                        get_nested(
                            numeric,
                            "staff_topic_overlap_ratio",
                            "minimum",
                        )
                    ),
                    number(
                        get_nested(
                            numeric,
                            "staff_topic_overlap_ratio",
                            "mean",
                        )
                    ),
                    number(
                        get_nested(
                            numeric,
                            "staff_topic_overlap_ratio",
                            "median",
                        )
                    ),
                    number(
                        get_nested(
                            numeric,
                            "staff_topic_overlap_ratio",
                            "maximum",
                        )
                    ),
                ],
            ],
        )
    )
    markdown.append("")

    markdown.append(
        "The complementarity gain is calculated as:"
    )
    markdown.append("")

    markdown.append(
        r"\["
        "\text{ComplementarityGain}(a,b,m) = "
        "\text{Coverage}(a \cup b,m) - "
        "\max(\text{Coverage}(a,m), \text{Coverage}(b,m))"
        r"\]"
    )
    markdown.append("")

    markdown.append(
        "A positive value means that the second staff member adds "
        "weighted module-topic coverage beyond the stronger "
        "individual candidate."
    )
    markdown.append("")

    markdown.append("## Recommendation Outputs")
    markdown.append("")

    levels = output.get("recommendation_levels", {})

    markdown.append(
        markdown_table(
            ["Output measure", "Result"],
            [
                [
                    "Recommendations written",
                    f"{safe_int(output.get('recommendations_written')):,}",
                ],
                [
                    "High recommendations",
                    f"{safe_int(levels.get('high')):,}",
                ],
                [
                    "Moderate recommendations",
                    f"{safe_int(levels.get('moderate')):,}",
                ],
                [
                    "Exploratory recommendations",
                    f"{safe_int(levels.get('exploratory')):,}",
                ],
                [
                    "Strict display recommendations",
                    f"{safe_int(output.get('strict_display_recommendations')):,}",
                ],
                [
                    "Case studies selected",
                    f"{safe_int(output.get('case_studies_selected')):,}",
                ],
            ],
        )
    )
    markdown.append("")

    markdown.append("## Semantic Foundation Analysis")
    markdown.append("")

    markdown.append(
        markdown_table(
            ["Measure", "Result"],
            [
                [
                    "Semantic similarity threshold",
                    number(semantic.get("threshold"), 2),
                ],
                [
                    "Exact shared-foundation recommendations",
                    f"{safe_int(semantic.get('exact_foundation_recommendations')):,}",
                ],
                [
                    "Semantic-only foundation recommendations",
                    f"{safe_int(semantic.get('semantic_only_foundation_recommendations')):,}",
                ],
                [
                    "Top-ranked exact foundation rate",
                    percent(
                        semantic.get(
                            "exact_only_foundation_rate"
                        )
                    ),
                ],
                [
                    "Top-ranked exact plus semantic foundation rate",
                    percent(
                        semantic.get(
                            "exact_plus_semantic_foundation_rate"
                        )
                    ),
                ],
                [
                    "Additional top-ranked modules supported by semantic evidence",
                    str(
                        safe_int(
                            semantic.get(
                                "additional_modules_supported_by_semantic_evidence"
                            )
                        )
                    ),
                ],
            ],
        )
    )
    markdown.append("")

    markdown.append(
        "Semantic shared foundation was treated as supplementary "
        "explanation evidence. It did not alter the original "
        "collaboration ranking, coverage scores or complementarity "
        "gain."
    )
    markdown.append("")

    markdown.append("## Case-Study Set")
    markdown.append("")

    markdown.append(
        markdown_table(
            ["Measure", "Result"],
            [
                [
                    "Case studies checked",
                    str(
                        safe_int(
                            cases.get("case_studies_checked")
                        )
                    ),
                ],
                [
                    "Duplicate case IDs",
                    str(
                        safe_int(
                            cases.get(
                                "duplicate_case_id_count"
                            )
                        )
                    ),
                ],
                [
                    "Repeated module IDs across groups",
                    str(
                        safe_int(
                            cases.get(
                                "duplicate_module_id_count"
                            )
                        )
                    ),
                ],
            ],
        )
    )
    markdown.append("")

    markdown.append(
        "The case-study set was designed as an audit and "
        "expert-review instrument. It should not be treated as "
        "researcher-labelled ground truth for all academic "
        "disciplines."
    )
    markdown.append("")

    markdown.append("## Key Findings")
    markdown.append("")

    for finding in report["key_findings"]:
        markdown.append(f"- {finding}")

    markdown.append("")
    markdown.append("## Limitations")
    markdown.append("")

    for limitation in report["limitations"]:
        markdown.append(f"- {limitation}")

    markdown.append("")
    markdown.append("## Evaluation Boundary")
    markdown.append("")

    markdown.append(
        "The M4 evaluation demonstrates internal consistency, "
        "reproducible processing, positive complementarity, "
        "topic-evidence traceability and explanation completeness. "
        "It does not replace institutional decision-making or "
        "subject-specialist validation of teaching allocations."
    )
    markdown.append("")

    return "\n".join(markdown)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a consolidated JSON and Markdown evaluation "
            "report for the M4 collaboration recommender."
        )
    )

    parser.add_argument(
        "--topic-profiles-summary",
        type=Path,
        default=DEFAULT_TOPIC_PROFILES_SUMMARY,
    )

    parser.add_argument(
        "--quality-filter-summary",
        type=Path,
        default=DEFAULT_QUALITY_FILTER_SUMMARY,
    )

    parser.add_argument(
        "--candidates-summary",
        type=Path,
        default=DEFAULT_CANDIDATES_SUMMARY,
    )

    parser.add_argument(
        "--recommendations-summary",
        type=Path,
        default=DEFAULT_RECOMMENDATIONS_SUMMARY,
    )

    parser.add_argument(
        "--metadata-summary",
        type=Path,
        default=DEFAULT_METADATA_SUMMARY,
    )

    parser.add_argument(
        "--case-studies-summary",
        type=Path,
        default=DEFAULT_CASE_STUDIES_SUMMARY,
    )

    parser.add_argument(
        "--structural-validation-report",
        type=Path,
        default=DEFAULT_STRUCTURAL_VALIDATION_REPORT,
    )

    parser.add_argument(
        "--semantic-foundation-summary",
        type=Path,
        default=DEFAULT_SEMANTIC_FOUNDATION_SUMMARY,
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
    )

    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=DEFAULT_OUTPUT_MARKDOWN,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_paths = {
        "topic_profiles_summary": args.topic_profiles_summary,
        "topic_quality_filter_summary": (
            args.quality_filter_summary
        ),
        "collaboration_candidates_summary": (
            args.candidates_summary
        ),
        "collaboration_recommendations_summary": (
            args.recommendations_summary
        ),
        "entity_metadata_summary": args.metadata_summary,
        "case_studies_summary": args.case_studies_summary,
        "structural_validation_report": (
            args.structural_validation_report
        ),
        "semantic_shared_foundation_summary": (
            args.semantic_foundation_summary
        ),
    }

    topic_profiles = read_json(
        args.topic_profiles_summary
    )

    quality_filter = read_json(
        args.quality_filter_summary
    )

    candidates = read_json(
        args.candidates_summary
    )

    recommendations = read_json(
        args.recommendations_summary
    )

    metadata = read_json(
        args.metadata_summary
    )

    case_studies = read_json(
        args.case_studies_summary
    )

    structural_validation = read_json(
        args.structural_validation_report
    )

    semantic_foundation = optional_read_json(
        args.semantic_foundation_summary
    )

    report = build_report(
        topic_profiles=topic_profiles,
        quality_filter=quality_filter,
        candidates=candidates,
        recommendations=recommendations,
        metadata=metadata,
        case_studies=case_studies,
        structural_validation=structural_validation,
        semantic_foundation=semantic_foundation,
        input_paths=input_paths,
    )

    markdown_report = build_markdown_report(report)

    args.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output_json.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    args.output_markdown.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output_markdown.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(markdown_report)

    print(
        json.dumps(
            {
                "generated_at_utc": report[
                    "generated_at_utc"
                ],
                "output_json": str(args.output_json),
                "output_markdown": str(args.output_markdown),
                "recommendations_written": (
                    report["recommendation_output"][
                        "recommendations_written"
                    ]
                ),
                "record_level_failure_count": (
                    report["structural_validation"][
                        "record_level_failure_count"
                    ]
                ),
                "semantic_only_foundation_recommendations": (
                    report[
                        "semantic_shared_foundation_analysis"
                    ][
                        "semantic_only_foundation_recommendations"
                    ]
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()