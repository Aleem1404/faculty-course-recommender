from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import numpy as np

from m4_v2.aspect_definitions import InterdisciplinaryAspect
from m4_v2.curriculum_gap_analyzer import AspectGap


@dataclass
class AspectCollaborationRecommendation:
    rank: int
    staff_id: str
    full_name: str
    position: str
    department_name: str
    college_name: str
    profile_url: str
    aspect_id: str
    aspect_title: str
    collaboration_role: str
    aspect_affinity_score: float
    module_semantic_score: float
    collaboration_suitability_score: float
    rationale: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "staff_id": self.staff_id,
            "full_name": self.full_name,
            "position": self.position,
            "department_name": self.department_name,
            "college_name": self.college_name,
            "profile_url": self.profile_url,
            "aspect_id": self.aspect_id,
            "aspect_title": self.aspect_title,
            "collaboration_role": self.collaboration_role,
            "aspect_affinity_score": round(self.aspect_affinity_score, 4),
            "module_semantic_score": round(self.module_semantic_score, 4),
            "collaboration_suitability_score": round(self.collaboration_suitability_score, 4),
            "rationale": self.rationale,
            "explanation": self.explanation,
        }


class SoftSemanticMatcher:
    def __init__(
        self,
        staff_metadata: Dict[str, Dict[str, Any]],
        staff_embeddings: np.ndarray,
        staff_id_to_idx: Dict[str, int],
        aspect_vectors: np.ndarray,
        aspects: List[InterdisciplinaryAspect],
        min_aspect_affinity: float = 0.32,
        min_collaboration_score: float = 0.30,
    ) -> None:
        self.staff_metadata = staff_metadata
        self.staff_embeddings = staff_embeddings
        self.staff_id_to_idx = staff_id_to_idx
        self.aspect_vectors = aspect_vectors
        self.aspects = aspects
        self.aspect_id_to_idx = {a.aspect_id: i for i, a in enumerate(aspects)}
        self.min_aspect_affinity = min_aspect_affinity
        self.min_collaboration_score = min_collaboration_score

    def find_collaborators_for_gaps(
        self,
        module_id: str,
        module_title: str,
        module_departments: List[str],
        module_vector: np.ndarray,
        primary_lead_name: str,
        primary_lead_dept: str,
        active_gaps: List[AspectGap],
        max_collaborators_per_module: int = 3,
    ) -> List[AspectCollaborationRecommendation]:
        """Finds top cross-department faculty to cover active curriculum gaps."""
        mod_v = module_vector / (np.linalg.norm(module_vector) + 1e-9)
        all_staff_module_sims = np.dot(self.staff_embeddings, mod_v)

        # Normalize home departments for cross-department filtering
        home_depts_norm = {d.strip().lower() for d in module_departments if d}

        recommendations: List[AspectCollaborationRecommendation] = []
        selected_staff_ids: Set[str] = set()

        for gap in active_gaps:
            if gap.need_level == "Covered By Primary Lead":
                continue

            aspect = gap.aspect
            aspect_idx = self.aspect_id_to_idx.get(aspect.aspect_id)
            if aspect_idx is None:
                continue

            aspect_vec = self.aspect_vectors[aspect_idx]
            # Aspect similarity across all staff
            staff_aspect_sims = np.dot(self.staff_embeddings, aspect_vec)

            # Candidate pool of external staff
            candidate_pool: List[Dict[str, Any]] = []

            for staff_id, s_idx in self.staff_id_to_idx.items():
                if staff_id in selected_staff_ids:
                    continue

                meta = self.staff_metadata.get(staff_id, {})
                s_dept = meta.get("department_name", meta.get("department", "Unknown Department")).strip()
                s_dept_norm = s_dept.lower()

                # MUST BE CROSS-DEPARTMENTAL (not from home department)
                is_internal = any(
                    hd in s_dept_norm or s_dept_norm in hd
                    for hd in home_depts_norm
                    if hd and len(hd) > 3
                )
                if is_internal:
                    continue

                aspect_aff = float(staff_aspect_sims[s_idx])
                mod_sim = float(all_staff_module_sims[s_idx])

                if aspect_aff < self.min_aspect_affinity:
                    continue

                # Collaboration score: balances aspect specialty (70%) with module general relevance (30%)
                collab_score = 0.70 * aspect_aff + 0.30 * max(0.0, mod_sim)

                if collab_score < self.min_collaboration_score:
                    continue

                candidate_pool.append({
                    "staff_id": staff_id,
                    "full_name": meta.get("full_name", meta.get("display_name", staff_id)),
                    "position": meta.get("position", meta.get("job_title", "Faculty Member")),
                    "department_name": s_dept,
                    "college_name": meta.get("college_name", meta.get("school", "")),
                    "profile_url": meta.get("profile_url", f"https://www.brunel.ac.uk/people/{staff_id}"),
                    "aspect_affinity": aspect_aff,
                    "module_semantic": mod_sim,
                    "collab_score": collab_score,
                })

            if not candidate_pool:
                continue

            # Sort candidate pool for this aspect
            candidate_pool.sort(key=lambda x: x["collab_score"], reverse=True)
            best_candidate = candidate_pool[0]

            selected_staff_ids.add(best_candidate["staff_id"])
            current_rank = len(recommendations) + 1

            rationale = (
                f"Enriches '{module_title}' by delivering specialist guest sessions on "
                f"'{aspect.title}'. Complements {primary_lead_name}'s ({primary_lead_dept}) "
                f"core delivery with external domain perspectives from {best_candidate['department_name']}."
            )

            explanation = (
                f"Cross-department co-delivery match for {aspect.title} "
                f"(Aspect affinity: {best_candidate['aspect_affinity']:.2f}, "
                f"Module semantic match: {best_candidate['module_semantic']:.2f})."
            )

            rec = AspectCollaborationRecommendation(
                rank=current_rank,
                staff_id=best_candidate["staff_id"],
                full_name=best_candidate["full_name"],
                position=best_candidate["position"],
                department_name=best_candidate["department_name"],
                college_name=best_candidate["college_name"],
                profile_url=best_candidate["profile_url"],
                aspect_id=aspect.aspect_id,
                aspect_title=aspect.title,
                collaboration_role="Interdisciplinary Co-Lecturer",
                aspect_affinity_score=best_candidate["aspect_affinity"],
                module_semantic_score=best_candidate["module_semantic"],
                collaboration_suitability_score=best_candidate["collab_score"],
                rationale=rationale,
                explanation=explanation,
            )
            recommendations.append(rec)

            if len(recommendations) >= max_collaborators_per_module:
                break

        return recommendations
