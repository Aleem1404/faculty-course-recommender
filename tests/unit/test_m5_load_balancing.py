from __future__ import annotations

import pytest
import numpy as np

from m5.capacity_policy import CapacityPolicy, FacultyRoleQuota
from m5.fairness_metrics import FairnessMetricsCalculator
from m5.load_balancer import M5LoadBalancer, BalancedAllocationResult
from m5.explanation_generator import M5ExplanationGenerator


def test_gini_calculation():
    # Case 1: Perfect equality (everyone has 2 modules)
    equal_workloads = [2, 2, 2, 2, 2]
    gini_equal = FairnessMetricsCalculator.calculate_gini(equal_workloads)
    assert abs(gini_equal - 0.0) < 1e-4

    # Case 2: Extreme inequality (one person has 10, everyone else has 0)
    extreme_workloads = [10, 0, 0, 0, 0]
    gini_extreme = FairnessMetricsCalculator.calculate_gini(extreme_workloads)
    # Gini of [10, 0, 0, 0, 0] is (5-1)/5 = 0.8
    assert abs(gini_extreme - 0.8) < 1e-3

    # Case 3: Empty / zero workload
    assert FairnessMetricsCalculator.calculate_gini([]) == 0.0
    assert FairnessMetricsCalculator.calculate_gini([0, 0, 0]) == 0.0


def test_capacity_policy():
    policy = CapacityPolicy(default_max_primary_modules=3)
    
    # Professor role quota
    assert policy.get_max_primary_capacity("prof-1", position="Professor of AI") == 2
    # Senior Lecturer role quota
    assert policy.get_max_primary_capacity("sl-1", position="Senior Lecturer in Computer Science") == 3
    # Standard fallback
    assert policy.get_max_primary_capacity("misc-1", position="Research Fellow") == 3
    # Override
    policy.faculty_capacity_overrides["special-1"] = 5
    assert policy.get_max_primary_capacity("special-1") == 5


def test_m5_load_balancer_strictly_enforces_capacity():
    # Construct synthetic modules where Prof A is #1 for all 10 modules
    synthetic_modules = []
    for i in range(10):
        synthetic_modules.append({
            "module_id": f"mod_{i}",
            "module_code": f"CS500{i}",
            "module_title": f"Machine Learning {i}",
            "module_classification": "taught_academic",
            "layer_1_primary_delivery": {
                "top_5_internal_recommendations": [
                    {"staff_id": "star-prof", "full_name": "Star Professor", "m3_kg_score": 0.95, "department_name": "CS"},
                    {"staff_id": f"lecturer_{i}", "full_name": f"Lecturer {i}", "m3_kg_score": 0.80, "department_name": "CS"},
                ]
            }
        })

    # Policy: max 3 modules per staff
    policy = CapacityPolicy(default_max_primary_modules=3)
    balancer = M5LoadBalancer(policy)
    
    results = balancer.balance_allocations(synthetic_modules)
    
    # Verify exactly 10 allocations
    assert len(results) == 10
    
    # Tally allocations
    tally = {}
    for r in results:
        tally[r.assigned_staff_id] = tally.get(r.assigned_staff_id, 0) + 1
        
    # Star prof should have EXACTLY 3 allocations (capped), not 10!
    assert tally.get("star-prof", 0) <= 3
    
    # Other modules should be assigned to alternative lecturers
    displaced_count = sum(1 for r in results if r.is_displaced_due_to_capacity)
    assert displaced_count == 7  # 10 - 3 = 7 displaced


def test_m5_explanation_generator():
    # Direct match result
    res_direct = BalancedAllocationResult(
        module_id="m1",
        module_code="CS101",
        module_title="Intro to CS",
        module_classification="taught_academic",
        assigned_staff_id="staff-a",
        assigned_staff_name="Dr A",
        assigned_department="Computer Science",
        assigned_score=0.88,
        original_greedy_staff_id="staff-a",
        original_greedy_score=0.88,
        rank_shift=0,
        is_displaced_due_to_capacity=False,
        displaced_candidates=[],
        all_top_candidates=[],
    )
    
    exp_direct = M5ExplanationGenerator.generate_explanation(res_direct)
    assert exp_direct["policy_flag"] == "UnconstrainedOptimal"
    assert "Direct Top-Rank Match" in exp_direct["allocation_category"]
    
    # Displaced match result
    res_displaced = BalancedAllocationResult(
        module_id="m2",
        module_code="CS102",
        module_title="Algorithms",
        module_classification="taught_academic",
        assigned_staff_id="staff-b",
        assigned_staff_name="Dr B",
        assigned_department="Computer Science",
        assigned_score=0.82,
        original_greedy_staff_id="staff-a",
        original_greedy_score=0.88,
        rank_shift=1,
        is_displaced_due_to_capacity=True,
        displaced_candidates=[{"staff_id": "staff-a", "full_name": "Dr A", "score": 0.88, "rank": 1}],
        all_top_candidates=[],
    )
    
    exp_displaced = M5ExplanationGenerator.generate_explanation(res_displaced)
    assert exp_displaced["policy_flag"] == "CapacityQuotaEnforced"
    assert exp_displaced["displaced_count"] == 1
    assert "Dr A (Rank #1)" in exp_displaced["allocation_rationale"]
