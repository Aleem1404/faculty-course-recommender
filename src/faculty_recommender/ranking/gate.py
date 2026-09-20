from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MIN_TOTAL_SCORE = 0.12
MIN_STRONG_FIELD_SCORE = 0.10
MIN_PUBLICATION_SIGNAL = 0.06

GENERIC_TERMS = {
    "research",
    "education",
    "technology",
    "innovation",
    "science",
    "engineering",
    "business",
    "health",
    "social sciences",
    "academic",
    "university",
    "teaching",
}


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reason: str
    diagnostics: dict[str, Any]


def normalise_text(value: Any) -> str:
    if value is None:
        return ""

    return " ".join(str(value).casefold().split())


def unique_normalised_strings(values: Any) -> set[str]:
    if values is None:
        return set()

    if not isinstance(values, list):
        values = [values]

    results = set()

    for value in values:
        text = normalise_text(value)

        if text:
            results.add(text)

    return results


def staff_evidence_terms(
    staff_record: dict[str, Any],
) -> set[str]:
    terms = set()

    terms.update(
        unique_normalised_strings(
            staff_record.get(
                "publication_topic_names",
                [],
            )
        )
    )

    terms.update(
        unique_normalised_strings(
            staff_record.get(
                "publication_keyword_names",
                [],
            )
        )
    )

    return terms


def has_specific_evidence(
    terms: set[str],
) -> bool:
    return any(
        term not in GENERIC_TERMS
        and len(term) >= 4
        for term in terms
    )


def gate_primary_recommendation(
    module_record: dict[str, Any],
    staff_record: dict[str, Any],
    recommendation: dict[str, Any],
) -> GateDecision:
    total_score = float(
        recommendation.get("semantic_score", 0.0)
    )

    field_scores = recommendation.get(
        "field_scores",
        {},
    )

    core_score = float(
        field_scores.get("core", 0.0)
    )

    title_score = float(
        field_scores.get(
            "publication_titles",
            0.0,
        )
    )

    abstract_score = float(
        field_scores.get(
            "publication_abstracts",
            0.0,
        )
    )

    topic_score = float(
        field_scores.get(
            "publication_topics",
            0.0,
        )
    )

    keyword_score = float(
        field_scores.get(
            "publication_keywords",
            0.0,
        )
    )

    max_field_score = max(
        core_score,
        title_score,
        abstract_score,
        topic_score,
        keyword_score,
    )

    publication_signal = max(
        title_score,
        abstract_score,
        topic_score,
        keyword_score,
    )

    evidence_terms = staff_evidence_terms(
        staff_record
    )

    specific_evidence = has_specific_evidence(
        evidence_terms
    )

    has_core_profile = bool(
        normalise_text(
            staff_record.get(
                "core_profile_text"
            )
        )
    )

    has_publication_evidence = bool(
        normalise_text(
            staff_record.get(
                "publication_expertise_text"
            )
        )
    )

    diagnostics = {
        "semantic_score": round(total_score, 6),
        "core_score": round(core_score, 6),
        "max_field_score": round(
            max_field_score,
            6,
        ),
        "publication_signal": round(
            publication_signal,
            6,
        ),
        "has_core_profile": has_core_profile,
        "has_publication_evidence": (
            has_publication_evidence
        ),
        "has_specific_evidence": (
            specific_evidence
        ),
        "evidence_terms_preview": sorted(
            evidence_terms
        )[:15],
    }

    if total_score < MIN_TOTAL_SCORE:
        return GateDecision(
            passed=False,
            reason="low_total_score",
            diagnostics=diagnostics,
        )

    if max_field_score < MIN_STRONG_FIELD_SCORE:
        return GateDecision(
            passed=False,
            reason="no_strong_field_signal",
            diagnostics=diagnostics,
        )

    if (
        publication_signal < MIN_PUBLICATION_SIGNAL
        and core_score < MIN_PUBLICATION_SIGNAL
    ):
        return GateDecision(
            passed=False,
            reason="insufficient_evidence_signal",
            diagnostics=diagnostics,
        )

    if not has_core_profile and not has_publication_evidence:
        return GateDecision(
            passed=False,
            reason="profile_evidence_missing",
            diagnostics=diagnostics,
        )

    if not specific_evidence and publication_signal > 0:
        return GateDecision(
            passed=False,
            reason="generic_evidence_only",
            diagnostics=diagnostics,
        )

    return GateDecision(
        passed=True,
        reason="passed",
        diagnostics=diagnostics,
    )