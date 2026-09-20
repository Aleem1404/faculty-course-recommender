from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PRIMARY_ALLOWED_TERMS = (
    "lecturer",
    "professor",
    "reader",
    "teaching fellow",
    "academic",
)

PRIMARY_EXCLUDED_TERMS = (
    "emeritus",
    "honorary",
    "visiting",
    "research fellow",
    "research assistant",
    "postdoctoral",
    "post-doctoral",
    "doctoral researcher",
    "phd student",
    "student",
    "administrator",
    "administrative",
    "technician",
    "technical officer",
)


@dataclass(frozen=True)
class EligibilityDecision:
    eligible: bool
    reason: str


def normalise_position(value: Any) -> str:
    if value is None:
        return ""

    return " ".join(
        str(value).casefold().split()
    )


def primary_delivery_eligibility(
    staff: dict[str, Any],
) -> EligibilityDecision:
    position = normalise_position(
        staff.get("position")
    )

    if not position:
        return EligibilityDecision(
            eligible=False,
            reason="missing_position",
        )

    excluded_term = next(
        (
            term
            for term in PRIMARY_EXCLUDED_TERMS
            if term in position
        ),
        None,
    )

    if excluded_term:
        return EligibilityDecision(
            eligible=False,
            reason=f"excluded_role:{excluded_term}",
        )

    allowed_term = next(
        (
            term
            for term in PRIMARY_ALLOWED_TERMS
            if term in position
        ),
        None,
    )

    if allowed_term:
        return EligibilityDecision(
            eligible=True,
            reason=f"eligible_role:{allowed_term}",
        )

    return EligibilityDecision(
        eligible=False,
        reason="unclassified_position",
    )