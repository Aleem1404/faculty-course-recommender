from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class FacultyRoleQuota:
    """Defines teaching capacity quotas and research buy-out limits per academic role."""
    role_pattern: str
    max_primary_modules: int
    max_guest_lectures: int
    description: str


DEFAULT_ROLE_QUOTAS = [
    FacultyRoleQuota(
        role_pattern="professor",
        max_primary_modules=2,
        max_guest_lectures=4,
        description="Professor / Chair: High research & doctoral supervision commitments",
    ),
    FacultyRoleQuota(
        role_pattern="head of department",
        max_primary_modules=1,
        max_guest_lectures=2,
        description="Head of Department: High administrative load",
    ),
    FacultyRoleQuota(
        role_pattern="reader",
        max_primary_modules=3,
        max_guest_lectures=4,
        description="Reader: Balanced research and primary teaching",
    ),
    FacultyRoleQuota(
        role_pattern="senior lecturer",
        max_primary_modules=3,
        max_guest_lectures=4,
        description="Senior Lecturer: Core curriculum and module leadership",
    ),
    FacultyRoleQuota(
        role_pattern="lecturer",
        max_primary_modules=4,
        max_guest_lectures=4,
        description="Lecturer: Core teaching delivery & developing research profile",
    ),
]


@dataclass
class CapacityPolicy:
    """Configurable institutional policy governing workload balancing and qualification thresholds."""
    default_max_primary_modules: int = 3
    default_max_guest_lectures: int = 3
    min_competency_threshold: float = 0.15
    role_quotas: list[FacultyRoleQuota] = field(default_factory=lambda: list(DEFAULT_ROLE_QUOTAS))
    faculty_capacity_overrides: dict[str, int] = field(default_factory=dict)

    def get_max_primary_capacity(self, staff_id: str, position: Optional[str] = None) -> int:
        """Determines the maximum allowed primary module allocations for a given faculty member."""
        if staff_id in self.faculty_capacity_overrides:
            return self.faculty_capacity_overrides[staff_id]

        if not position:
            return self.default_max_primary_modules

        pos_lower = position.lower()
        for quota in self.role_quotas:
            if quota.role_pattern in pos_lower:
                return quota.max_primary_modules

        return self.default_max_primary_modules

    def is_competent_for_module(self, relevance_score: float) -> bool:
        """Verifies if the candidate meets the baseline academic qualification threshold."""
        return relevance_score >= self.min_competency_threshold
