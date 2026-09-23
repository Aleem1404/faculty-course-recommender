from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple
import numpy as np


@dataclass
class AllocationFairnessReport:
    """Comprehensive fairness, inequality, and workload distribution metrics report."""
    total_modules: int
    total_faculty_in_pool: int
    allocated_faculty_count: int
    faculty_utilization_pct: float
    gini_coefficient: float
    max_workload: int
    min_workload_active: int
    mean_workload_active: float
    workload_standard_deviation: float
    top_10_faculty_concentration_pct: float
    top_5_percent_concentration_pct: float
    average_relevance_score: float
    workload_distribution_histogram: dict[int, int]
    lorenz_curve_points: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FairnessMetricsCalculator:
    """Calculates formal econometric and recommender system fairness metrics for academic workloads."""

    @staticmethod
    def calculate_gini(workloads: list[int] | np.ndarray) -> float:
        """
        Calculates the Gini coefficient of inequality for an array of workloads.
        Gini = 0 represents perfect equality, Gini = 1 represents maximal inequality.
        """
        w = np.array(workloads, dtype=np.float64)
        if len(w) == 0 or np.sum(w) == 0:
            return 0.0

        # Ensure all workloads are non-negative
        w = np.clip(w, a_min=0, a_max=None)
        n = len(w)
        w_sorted = np.sort(w)
        index = np.arange(1, n + 1)
        return float((2.0 * np.sum(index * w_sorted)) / (n * np.sum(w_sorted)) - (n + 1.0) / n)

    @staticmethod
    def compute_lorenz_curve(workloads: list[int] | np.ndarray, num_points: int = 10) -> list[dict[str, float]]:
        """Computes points along the empirical Lorenz curve for plotting."""
        w = np.array(workloads, dtype=np.float64)
        if len(w) == 0 or np.sum(w) == 0:
            return [{"faculty_share": p, "workload_share": p} for p in np.linspace(0, 1.0, num_points)]

        w_sorted = np.sort(w)
        cum_workload = np.cumsum(w_sorted) / np.sum(w_sorted)
        cum_faculty = np.linspace(0.0, 1.0, len(w_sorted))

        # Sample evenly spaced points
        indices = np.linspace(0, len(w_sorted) - 1, num_points, dtype=int)
        points = []
        for idx in indices:
            points.append({
                "faculty_share": round(float(cum_faculty[idx]), 4),
                "workload_share": round(float(cum_workload[idx]), 4),
            })
        return points

    @classmethod
    def evaluate_allocations(
        cls,
        allocations: list[dict[str, Any]],
        all_faculty_pool: list[str],
        relevance_key: str = "m3_kg_score",
    ) -> AllocationFairnessReport:
        """
        Evaluates a complete university allocation mapping.
        
        :param allocations: List of module allocation records containing assigned 'staff_id' and score.
        :param all_faculty_pool: Full list of eligible faculty member IDs.
        :param relevance_key: The score field name used for average relevance calculation.
        """
        total_modules = len(allocations)
        assigned_staff = [a["staff_id"] for a in allocations if a.get("staff_id")]
        scores = [float(a.get(relevance_key, 0.0)) for a in allocations if a.get("staff_id")]
        
        counts = Counter(assigned_staff)
        
        # Build workload vector across the entire available faculty pool
        full_pool_set = set(all_faculty_pool) | set(assigned_staff)
        workload_vector = [counts.get(fid, 0) for fid in full_pool_set]
        active_workloads = list(counts.values()) if counts else [0]
        
        total_faculty = len(full_pool_set)
        allocated_faculty_count = len(counts)
        faculty_utilization_pct = round((allocated_faculty_count / total_faculty * 100.0) if total_faculty else 0.0, 2)
        
        gini = cls.calculate_gini(workload_vector)
        lorenz = cls.compute_lorenz_curve(workload_vector)
        
        # Calculate concentration metrics
        sorted_counts = sorted(counts.values(), reverse=True)
        top_10_sum = sum(sorted_counts[:10])
        top_10_pct = round((top_10_sum / total_modules * 100.0) if total_modules else 0.0, 2)
        
        top_5_pct_count = max(1, int(math.ceil(0.05 * total_faculty)))
        top_5_pct_sum = sum(sorted_counts[:top_5_pct_count])
        top_5_pct_conc = round((top_5_pct_sum / total_modules * 100.0) if total_modules else 0.0, 2)
        
        # Histogram of active workloads
        hist_counter = Counter(counts.values())
        hist_dict = {k: hist_counter[k] for k in sorted(hist_counter.keys())}
        
        return AllocationFairnessReport(
            total_modules=total_modules,
            total_faculty_in_pool=total_faculty,
            allocated_faculty_count=allocated_faculty_count,
            faculty_utilization_pct=faculty_utilization_pct,
            gini_coefficient=round(gini, 4),
            max_workload=max(active_workloads) if active_workloads else 0,
            min_workload_active=min(active_workloads) if active_workloads else 0,
            mean_workload_active=round(float(np.mean(active_workloads)), 3) if active_workloads else 0.0,
            workload_standard_deviation=round(float(np.std(active_workloads)), 3) if active_workloads else 0.0,
            top_10_faculty_concentration_pct=top_10_pct,
            top_5_percent_concentration_pct=top_5_pct_conc,
            average_relevance_score=round(float(np.mean(scores)), 4) if scores else 0.0,
            workload_distribution_histogram=hist_dict,
            lorenz_curve_points=lorenz,
        )
