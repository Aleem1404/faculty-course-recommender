from __future__ import annotations

from typing import Any, Dict, List, Optional
from m5.load_balancer import BalancedAllocationResult


class M5ExplanationGenerator:
    """Generates policy-aware, auditable natural language rationales for balanced teaching allocations."""

    @staticmethod
    def generate_explanation(result: BalancedAllocationResult) -> dict[str, Any]:
        """
        Produces a rich explanation payload detailing the decision logic behind the allocation.
        """
        if result.module_classification == "independent_project_placement":
            return {
                "allocation_category": "Project Supervision",
                "summary": f"Exempt unit: '{result.assigned_staff_name}' assigned as academic supervisor / dissertation coordinator.",
                "policy_flag": "ExemptFromLoadBalance",
                "displaced_count": 0,
                "allocation_rationale": "Non-taught independent research / placement module; requires individual student supervision rather than cohort lecture load balancing.",
            }

        if not result.is_displaced_due_to_capacity:
            return {
                "allocation_category": "Direct Top-Rank Match",
                "summary": f"Optimal primary allocation: {result.assigned_staff_name} assigned as Rank #1 subject expert (Relevance score: {result.assigned_score:.4f}).",
                "policy_flag": "UnconstrainedOptimal",
                "displaced_count": 0,
                "allocation_rationale": (
                    f"Selected as top-ranked subject expert from {result.assigned_department} without capacity conflicts. "
                    f"Maintains maximum pedagogical alignment."
                ),
            }

        # Case where capacity policy displaced higher-ranked candidate(s)
        displaced_names = [f"{d['full_name']} (Rank #{d['rank']})" for d in result.displaced_candidates]
        displaced_str = ", ".join(displaced_names)

        score_diff = result.original_greedy_score - result.assigned_score

        return {
            "allocation_category": "Capacity-Balanced Allocation",
            "summary": (
                f"Assigned {result.assigned_staff_name} (Rank #{result.rank_shift + 1}, Relevance: {result.assigned_score:.4f}) "
                f"under Institutional Workload Policy."
            ),
            "policy_flag": "CapacityQuotaEnforced",
            "displaced_count": len(result.displaced_candidates),
            "displaced_candidates": displaced_names,
            "relevance_tradeoff_delta": round(score_diff, 4),
            "allocation_rationale": (
                f"Higher-ranked candidate(s) [{displaced_str}] were capped at their maximum institutional teaching quota. "
                f"{result.assigned_staff_name} was allocated as the best available qualified candidate from {result.assigned_department} "
                f"with a strong domain relevance score of {result.assigned_score:.4f} (trade-off delta: {score_diff:.4f})."
            ),
        }
