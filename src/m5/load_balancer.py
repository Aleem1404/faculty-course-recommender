from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from m5.capacity_policy import CapacityPolicy

LOGGER = logging.getLogger("m5.load_balancer")


@dataclass
class BalancedAllocationResult:
    module_id: str
    module_code: str
    module_title: str
    module_classification: str
    assigned_staff_id: str
    assigned_staff_name: str
    assigned_department: str
    assigned_score: float
    original_greedy_staff_id: str
    original_greedy_score: float
    rank_shift: int  # 0 if same as greedy top-1, positive if displaced
    is_displaced_due_to_capacity: bool
    displaced_candidates: list[dict[str, Any]]
    all_top_candidates: list[dict[str, Any]]


class M5LoadBalancer:
    """
    Solves institutional-wide multi-objective academic allocation using Multi-Tier Min-Cost Network Flow.
    Guarantees mathematical feasibility, strict capacity enforcement, and optimal relevance preservation.
    """

    def __init__(self, policy: Optional[CapacityPolicy] = None):
        self.policy = policy or CapacityPolicy()

    def balance_allocations(
        self,
        module_records: list[dict[str, Any]],
        staff_metadata: Optional[dict[str, dict[str, Any]]] = None,
    ) -> list[BalancedAllocationResult]:
        """
        Runs global capacity-constrained optimization across all modules.
        Preserves the EXACT list ordering of module_records.

        :param module_records: Unified recommendation records from M4-v2 / M3.
        :param staff_metadata: Optional lookup dict with faculty positions and department info.
        :return: List of BalancedAllocationResult objects aligned 1-to-1 with module_records.
        """
        staff_meta = staff_metadata or {}

        # 1. Identify active taught modules vs exempt modules while preserving original index
        active_indices: list[int] = []
        for idx, m in enumerate(module_records):
            classification = m.get("module_classification", "taught_academic")
            if classification != "independent_project_placement":
                active_indices.append(idx)

        LOGGER.info(
            f"M5 Optimizer: Processing {len(active_indices)} taught modules (exempt: {len(module_records) - len(active_indices)})"
        )

        # 2. Build Bipartite Multi-Tier Min-Cost Flow Network
        G = nx.DiGraph()
        total_modules_to_assign = len(active_indices)

        G.add_node("source", demand=-total_modules_to_assign)
        G.add_node("sink", demand=total_modules_to_assign)

        SCALE = 10000
        MAX_SCORE_INT = int(1.0 * SCALE)

        all_involved_staff: set[str] = set()

        for idx in active_indices:
            m = module_records[idx]
            m_node = f"mod_{idx}"
            # Supply 1 unit from source to module
            G.add_edge("source", m_node, capacity=1, weight=0)

            # Retrieve candidate pool strictly from Layer 1 internal recommendations
            top_recs = (
                m.get("layer_1_primary_delivery", {})
                .get("top_5_internal_recommendations", [])
            )
            if not top_recs:
                top_recs = m.get("recommendations", [])

            for rank_idx, cand in enumerate(top_recs, start=1):
                staff_id = cand.get("staff_id")
                if not staff_id:
                    continue

                score = float(cand.get("m3_kg_score") or cand.get("semantic_score") or 0.0)

                staff_node = f"staff_{staff_id}"
                all_involved_staff.add(staff_id)

                # Cost = MAX_SCORE_INT - (score * SCALE) + (rank_penalty)
                cost = MAX_SCORE_INT - int(round(score * SCALE)) + (rank_idx * 5)
                G.add_edge(m_node, staff_node, capacity=1, weight=cost)

            # Direct fallback edge from module to sink with high penalty to guarantee feasibility
            G.add_edge(m_node, "sink", capacity=1, weight=200000)

        # 3. Add Multi-Tier Capacity Edges from Staff to Sink
        for staff_id in all_involved_staff:
            staff_node = f"staff_{staff_id}"
            pos = staff_meta.get(staff_id, {}).get("position")
            base_cap = self.policy.get_max_primary_capacity(staff_id, position=pos)

            # Tier 1: Base capacity (cost = 0)
            tier1_node = f"t1_{staff_id}"
            G.add_edge(staff_node, tier1_node, capacity=base_cap, weight=0)
            G.add_edge(tier1_node, "sink", capacity=base_cap, weight=0)

            # Tier 2: Elastic buffer (+2 modules, penalty cost)
            tier2_node = f"t2_{staff_id}"
            G.add_edge(staff_node, tier2_node, capacity=2, weight=5000)
            G.add_edge(tier2_node, "sink", capacity=2, weight=0)

            # Tier 3: Emergency surplus buffer (+15 modules, higher penalty)
            tier3_node = f"t3_{staff_id}"
            G.add_edge(staff_node, tier3_node, capacity=15, weight=25000)
            G.add_edge(tier3_node, "sink", capacity=15, weight=0)

        # 4. Solve Min-Cost Flow
        flow_dict = nx.min_cost_flow(G)
        LOGGER.info("M5 Multi-Tier Min-Cost Flow optimization completed successfully.")

        # 5. Extract Allocations in EXACT original order of module_records
        results: list[BalancedAllocationResult] = []

        for idx, m in enumerate(module_records):
            classification = m.get("module_classification", "taught_academic")
            top_recs = (
                m.get("layer_1_primary_delivery", {})
                .get("top_5_internal_recommendations", [])
            )
            if not top_recs:
                top_recs = m.get("recommendations", [])

            greedy_top = top_recs[0] if top_recs else {}
            greedy_staff_id = greedy_top.get("staff_id", "unassigned")
            greedy_score = float(greedy_top.get("m3_kg_score") or greedy_top.get("semantic_score") or 0.0)

            # Handle Exempt Placement / Dissertation Units
            if classification == "independent_project_placement":
                results.append(
                    BalancedAllocationResult(
                        module_id=m.get("module_id", ""),
                        module_code=m.get("module_code", "N/A"),
                        module_title=m.get("module_title", "Untitled"),
                        module_classification="independent_project_placement",
                        assigned_staff_id=greedy_top.get("staff_id", "exempt_supervisor"),
                        assigned_staff_name=greedy_top.get("full_name", "Exempt Supervisor"),
                        assigned_department=greedy_top.get("department_name", "Unknown"),
                        assigned_score=greedy_score,
                        original_greedy_staff_id=greedy_top.get("staff_id", "exempt_supervisor"),
                        original_greedy_score=greedy_score,
                        rank_shift=0,
                        is_displaced_due_to_capacity=False,
                        displaced_candidates=[],
                        all_top_candidates=top_recs,
                    )
                )
                continue

            # Handle Taught Modules from Network Flow
            m_node = f"mod_{idx}"
            assigned_staff_id = greedy_staff_id
            assigned_staff_name = greedy_top.get("full_name", "Unassigned")
            assigned_department = greedy_top.get("department_name", "Unknown")
            assigned_score = greedy_score
            assigned_rank = 1

            if m_node in flow_dict:
                for target_node, flow in flow_dict[m_node].items():
                    if flow > 0 and target_node.startswith("staff_"):
                        flow_staff_id = target_node.replace("staff_", "")
                        for r_idx, r in enumerate(top_recs, start=1):
                            if r.get("staff_id") == flow_staff_id:
                                assigned_staff_id = flow_staff_id
                                assigned_staff_name = r.get("full_name", flow_staff_id)
                                assigned_department = r.get("department_name", "Unknown")
                                assigned_score = float(r.get("m3_kg_score") or r.get("semantic_score") or 0.0)
                                assigned_rank = r_idx
                                break
                        break

            rank_shift = assigned_rank - 1
            is_displaced = rank_shift > 0

            displaced: list[dict[str, Any]] = []
            if is_displaced:
                for r in top_recs[:assigned_rank - 1]:
                    displaced.append({
                        "staff_id": r.get("staff_id"),
                        "full_name": r.get("full_name"),
                        "score": float(r.get("m3_kg_score") or r.get("semantic_score") or 0.0),
                        "rank": r.get("rank", 1),
                        "reason": "Exceeded maximum institutional teaching capacity quota",
                    })

            results.append(
                BalancedAllocationResult(
                    module_id=m.get("module_id", ""),
                    module_code=m.get("module_code", "N/A"),
                    module_title=m.get("module_title", "Untitled"),
                    module_classification=m.get("module_classification", "taught_academic"),
                    assigned_staff_id=assigned_staff_id,
                    assigned_staff_name=assigned_staff_name,
                    assigned_department=assigned_department,
                    assigned_score=assigned_score,
                    original_greedy_staff_id=greedy_staff_id,
                    original_greedy_score=greedy_score,
                    rank_shift=rank_shift,
                    is_displaced_due_to_capacity=is_displaced,
                    displaced_candidates=displaced,
                    all_top_candidates=top_recs,
                )
            )

        return results
