"""M5: Multi-Objective Load-Balanced and Fair Academic Allocation (M5-FairAlloc)."""

from m5.capacity_policy import CapacityPolicy, FacultyRoleQuota
from m5.explanation_generator import M5ExplanationGenerator
from m5.fairness_metrics import AllocationFairnessReport, FairnessMetricsCalculator
from m5.load_balancer import M5LoadBalancer
from m5.pipeline import M5Pipeline

__all__ = [
    "CapacityPolicy",
    "FacultyRoleQuota",
    "M5LoadBalancer",
    "FairnessMetricsCalculator",
    "AllocationFairnessReport",
    "M5ExplanationGenerator",
    "M5Pipeline",
]
