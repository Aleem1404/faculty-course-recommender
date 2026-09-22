from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from m4_v2.aspect_definitions import INTERDISCIPLINARY_ASPECTS, InterdisciplinaryAspect


@dataclass
class AspectGap:
    aspect: InterdisciplinaryAspect
    module_affinity: float
    primary_lead_affinity: float
    gap_score: float
    need_level: str  # "High", "Moderate", "Covered By Lead"


class CurriculumGapAnalyzer:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        aspects: Optional[List[InterdisciplinaryAspect]] = None,
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.aspects = aspects or INTERDISCIPLINARY_ASPECTS
        self.aspect_vectors = self._encode_aspects()

    def _encode_aspects(self) -> np.ndarray:
        anchor_texts = [aspect.anchor_text for aspect in self.aspects]
        vectors = self.model.encode(anchor_texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    def analyze_module_gaps(
        self,
        module_vector: np.ndarray,
        primary_lead_vector: Optional[np.ndarray],
        min_module_affinity: float = 0.30,
    ) -> List[AspectGap]:
        """Calculates which interdisciplinary curriculum aspects are active gaps for co-delivery."""
        # Ensure vectors are 1D and normalized
        mod_v = module_vector / (np.linalg.norm(module_vector) + 1e-9)
        
        # Module affinity to each aspect: Cosine Similarity
        module_affinities = np.dot(self.aspect_vectors, mod_v)
        
        if primary_lead_vector is not None:
            lead_v = primary_lead_vector / (np.linalg.norm(primary_lead_vector) + 1e-9)
            lead_affinities = np.dot(self.aspect_vectors, lead_v)
        else:
            lead_affinities = np.zeros_like(module_affinities)

        gaps: List[AspectGap] = []

        for i, aspect in enumerate(self.aspects):
            m_aff = float(module_affinities[i])
            l_aff = float(lead_affinities[i])

            if m_aff < min_module_affinity:
                continue

            # Gap score: high when module needs the aspect, but lead does not already dominate it
            # Redundancy suppression factor: if lead affinity >= 0.65, gap is heavily discounted
            lead_coverage_factor = max(0.0, 1.0 - (l_aff / 0.65))
            gap_score = m_aff * lead_coverage_factor

            if l_aff >= 0.55:
                need_level = "Covered By Primary Lead"
            elif gap_score >= 0.35:
                need_level = "High Co-Delivery Need"
            else:
                need_level = "Moderate Co-Delivery Need"

            gaps.append(
                AspectGap(
                    aspect=aspect,
                    module_affinity=round(m_aff, 4),
                    primary_lead_affinity=round(l_aff, 4),
                    gap_score=round(gap_score, 4),
                    need_level=need_level,
                )
            )

        # Sort gaps by gap_score descending
        gaps.sort(key=lambda g: g.gap_score, reverse=True)
        return gaps
