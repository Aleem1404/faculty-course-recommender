from __future__ import annotations

import csv
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from m4_v2.aspect_definitions import INTERDISCIPLINARY_ASPECTS
from m4_v2.curriculum_gap_analyzer import CurriculumGapAnalyzer
from m4_v2.soft_semantic_matcher import SoftSemanticMatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("m4_v2.pipeline")


NON_TAUGHT_PATTERNS: List[str] = [
    "placement",
    "practice placement",
    "clinical placement",
    "work placement",
    "internship",
    "year in industry",
    "work experience",
    "dissertation",
    "final year project",
    "individual project",
    "major project",
    "capstone project",
    "independent study",
    "student selected component",
    "study abroad",
    "student exchange",
]


def classify_module(title: str) -> tuple[str, bool]:
    """Classifies whether a module is a taught academic module or non-taught placement/project unit."""
    t_lower = title.strip().lower()
    for p in NON_TAUGHT_PATTERNS:
        if p in t_lower:
            return "independent_project_placement", False
    return "taught_academic", True


def sanitize_module_code_and_title(
    code: str | None,
    title: str | None,
    slug: str | None = None,
) -> tuple[str, str]:
    """Sanitizes messy/missing module codes and cleans titles with duplicate prefix codes."""
    raw_code = (code or "").strip()
    raw_title = (title or "").strip()
    raw_slug = (slug or "").strip()

    clean_code = ""
    clean_title = raw_title

    # 1. Clean redundant code prefix from title like 'CS5704 - Research...' or 'ED3702 - ED3702 - ...'
    prefix_match = re.match(
        r"^([A-Za-z]{2,4}[0-9]{1,4}[A-Za-z0-9]{0,3})\s*[-:]\s*(.*)$",
        clean_title,
    )
    extracted_from_title = ""
    if prefix_match:
        extracted_from_title = prefix_match.group(1).upper()
        rest = prefix_match.group(2).strip()
        double_match = re.match(
            r"^([A-Za-z]{2,4}[0-9]{1,4}[A-Za-z0-9]{0,3})\s*[-:]\s*(.*)$",
            rest,
        )
        if double_match:
            rest = double_match.group(2).strip()
        clean_title = rest or clean_title

    # 2. Check if raw_code is already a clean valid module code (e.g. CS5704, BE1607, AF1604)
    if raw_code and re.match(r"^[A-Za-z]{2,4}[0-9]{1,4}[A-Za-z0-9]{0,3}$", raw_code):
        if (
            raw_code.upper() not in ["TB", "TBC", "TBA"]
            and raw_code.upper() != clean_title.replace(" ", "").upper()
        ):
            clean_code = raw_code.upper()

    # 3. If raw_code wasn't valid, check if we extracted one from the title prefix
    if not clean_code and extracted_from_title:
        clean_code = extracted_from_title

    # 4. If still not found, check slug prefix
    if not clean_code and raw_slug:
        slug_match = re.match(
            r"^([a-z]{2,4}[0-9]{1,4}[a-z0-9]{0,3})-(.*)$",
            raw_slug,
        )
        if slug_match:
            clean_code = slug_match.group(1).upper()

    # 5. Check if raw_code was mashed like BB1722FUNDAMENTALBIOCHEMISTRY
    if not clean_code and raw_code:
        mashed_match = re.match(
            r"^([A-Za-z]{2,4}[0-9]{2,4}[A-Za-z0-9]{0,2})[A-Za-z]+$",
            raw_code,
        )
        if mashed_match:
            clean_code = mashed_match.group(1).upper()

    if (
        not clean_code
        or clean_code in ["TB", "TBC", "TBA"]
        or clean_code.upper() == clean_title.replace(" ", "").upper()
    ):
        clean_code = "N/A"

    return clean_code, clean_title


def run_unified_pipeline(
    project_root: Path,
    output_dir: Path,
    min_module_affinity: float = 0.20,
    min_aspect_affinity: float = 0.28,
    min_collab_score: float = 0.26,
    max_collaborators_per_module: int = 3,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load M3-KG Primary Recommendations
    m3_kg_path = (
        project_root
        / "data"
        / "outputs"
        / "recommendations"
        / "m3_kg"
        / "m3_kg_recommendations.jsonl"
    )
    LOGGER.info(f"Loading M3-KG recommendations from: {m3_kg_path}")

    m3_recs_by_module: Dict[str, Dict[str, Any]] = {}
    with m3_kg_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            m3_recs_by_module[record["module_id"]] = record

    LOGGER.info(f"Loaded {len(m3_recs_by_module)} M3-KG module recommendations.")

    # 2. Load Core Module Metadata (slugs, titles, descriptions)
    modules_core_path = project_root / "data" / "processed" / "core" / "modules.jsonl"
    core_modules_by_id: Dict[str, Dict[str, Any]] = {}
    if modules_core_path.exists():
        with modules_core_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                m = json.loads(line)
                core_modules_by_id[m["module_id"]] = m

    # 3. Load Core Staff Metadata
    staff_path = project_root / "data" / "processed" / "core" / "staff.jsonl"
    staff_metadata: Dict[str, Dict[str, Any]] = {}
    with staff_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            staff_metadata[s["staff_id"]] = s

    # 4. Load Cached Embeddings
    emb_idx_path = (
        project_root
        / "data"
        / "cache"
        / "embeddings"
        / "semantic"
        / "semantic_embedding_index.json"
    )
    emb_npz_path = (
        project_root
        / "data"
        / "cache"
        / "embeddings"
        / "semantic"
        / "semantic_embeddings.npz"
    )

    with emb_idx_path.open("r", encoding="utf-8") as f:
        emb_index = json.load(f)

    npz = np.load(emb_npz_path)
    module_embeddings = npz["module_embeddings"]
    module_ids = emb_index["module_ids"]
    mod_id_to_idx = {mid: i for i, mid in enumerate(module_ids)}

    staff_core = npz["staff_core_embeddings"]
    staff_pub_topics = npz["staff_publication_topics_embeddings"]
    staff_pub_abstracts = npz["staff_publication_abstracts_embeddings"]
    staff_ids = emb_index["staff_ids"]
    staff_id_to_idx = {sid: i for i, sid in enumerate(staff_ids)}

    # Blend staff embeddings: 50% core bio + 25% publication topics + 25% abstracts
    staff_combined = (
        0.50 * staff_core
        + 0.25 * staff_pub_topics
        + 0.25 * staff_pub_abstracts
    )
    staff_norms = np.linalg.norm(staff_combined, axis=1, keepdims=True) + 1e-9
    staff_embeddings_normed = staff_combined / staff_norms

    # 4. Initialize Analyzer & Matcher
    gap_analyzer = CurriculumGapAnalyzer(aspects=INTERDISCIPLINARY_ASPECTS)
    matcher = SoftSemanticMatcher(
        staff_metadata=staff_metadata,
        staff_embeddings=staff_embeddings_normed,
        staff_id_to_idx=staff_id_to_idx,
        aspect_vectors=gap_analyzer.aspect_vectors,
        aspects=INTERDISCIPLINARY_ASPECTS,
        min_aspect_affinity=min_aspect_affinity,
        min_collaboration_score=min_collab_score,
    )

    # 5. Process all modules
    unified_records: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    total_taught_modules = 0
    total_exempt_modules = 0
    taught_modules_with_collaborations = 0
    total_collaborations = 0
    aspect_counts: Dict[str, int] = {a.title: 0 for a in INTERDISCIPLINARY_ASPECTS}

    for mid, m3_rec in m3_recs_by_module.items():
        raw_title = m3_rec.get("module_title", "")
        raw_code = m3_rec.get("module_code", "")
        mod_depts = m3_rec.get("module_departments", [])
        mod_colleges = m3_rec.get("module_colleges", [])
        m3_primary_recs = m3_rec.get("recommendations", [])[:5]

        # Fetch core metadata if available
        core_m = core_modules_by_id.get(mid, {})
        core_slug = core_m.get("module_slug", "")
        core_title = core_m.get("module_title") or core_m.get("title") or raw_title

        # Sanitize module code and title
        mod_code, mod_title = sanitize_module_code_and_title(
            code=raw_code,
            title=core_title,
            slug=core_slug,
        )

        # Extract Rank 1 Primary Lead
        lead_id = m3_primary_recs[0]["staff_id"] if m3_primary_recs else None
        lead_name = m3_primary_recs[0]["full_name"] if m3_primary_recs else "Unassigned Lead"
        lead_dept = m3_primary_recs[0]["department_name"] if m3_primary_recs else (mod_depts[0] if mod_depts else "")

        # Classify module
        mod_type, collab_eligible = classify_module(mod_title)

        if collab_eligible:
            total_taught_modules += 1
        else:
            total_exempt_modules += 1

        # Get module vector
        m_idx = mod_id_to_idx.get(mid)
        if m_idx is None:
            continue
        m_vec = module_embeddings[m_idx]

        # Get primary lead vector
        l_idx = staff_id_to_idx.get(lead_id) if lead_id else None
        l_vec = staff_embeddings_normed[l_idx] if l_idx is not None else None

        if collab_eligible:
            # Analyze curriculum gaps (Primary Pass: threshold 0.20)
            gaps = gap_analyzer.analyze_module_gaps(
                module_vector=m_vec,
                primary_lead_vector=l_vec,
                min_module_affinity=min_module_affinity,
            )

            # Adaptive Fallback: If 0 gaps found, apply lenience window (0.18)
            if not gaps:
                gaps = gap_analyzer.analyze_module_gaps(
                    module_vector=m_vec,
                    primary_lead_vector=l_vec,
                    min_module_affinity=0.18,
                )

            # Find intelligent cross-department collaborators
            collab_recs = matcher.find_collaborators_for_gaps(
                module_id=mid,
                module_title=mod_title,
                module_departments=mod_depts,
                module_vector=m_vec,
                primary_lead_name=lead_name,
                primary_lead_dept=lead_dept,
                active_gaps=gaps,
                max_collaborators_per_module=max_collaborators_per_module,
            )

            if collab_recs:
                taught_modules_with_collaborations += 1
                total_collaborations += len(collab_recs)
                for c in collab_recs:
                    aspect_counts[c.aspect_title] = aspect_counts.get(c.aspect_title, 0) + 1

            layer_2_payload = {
                "status": "Active Taught Co-Delivery" if collab_recs else "No Cross-Department Gap Identified",
                "collaboration_eligible": True,
                "curriculum_gaps_identified": [
                    {
                        "aspect_id": g.aspect.aspect_id,
                        "aspect_title": g.aspect.title,
                        "module_affinity": g.module_affinity,
                        "primary_lead_affinity": g.primary_lead_affinity,
                        "gap_score": g.gap_score,
                        "need_level": g.need_level,
                    }
                    for g in gaps
                ],
                "cross_department_collaborations": [c.to_dict() for c in collab_recs],
            }
        else:
            # Exempt placement / dissertation module
            collab_recs = []
            layer_2_payload = {
                "status": "Exempt from Co-Delivery",
                "collaboration_eligible": False,
                "rationale": "Independent student research project / placement unit; individual academic supervision only.",
                "curriculum_gaps_identified": [],
                "cross_department_collaborations": [],
            }

        # Build unified record
        record: Dict[str, Any] = {
            "module_id": mid,
            "module_code": mod_code,
            "module_title": mod_title,
            "module_classification": mod_type,
            "collaboration_eligible": collab_eligible,
            "module_departments": mod_depts,
            "module_colleges": mod_colleges,
            "layer_1_primary_delivery": {
                "lead_staff_id": lead_id,
                "lead_staff_name": lead_name,
                "lead_department": lead_dept,
                "supervision_role": "Primary Module Leader" if collab_eligible else "Project Supervisor / Coordinator",
                "top_5_internal_recommendations": [
                    {
                        "rank": r.get("rank"),
                        "staff_id": r.get("staff_id"),
                        "full_name": r.get("full_name"),
                        "position": r.get("position"),
                        "department_name": r.get("department_name"),
                        "college_name": r.get("college_name"),
                        "profile_url": r.get("profile_url"),
                        "recommendation_role": r.get("recommendation_role", "primary_internal"),
                        "m3_kg_score": r.get("m3_kg_score"),
                        "shared_topics": r.get("shared_topics", []),
                        "explanation": r.get("explanation", ""),
                    }
                    for r in m3_primary_recs
                ],
            },
            "layer_2_intelligent_collaboration": layer_2_payload,
        }
        unified_records.append(record)

        # Build CSV row for case study / review sheet
        if collab_eligible and collab_recs:
            for rank_idx, c in enumerate(collab_recs, start=1):
                csv_rows.append({
                    "module_id": mid,
                    "module_code": mod_code,
                    "module_title": mod_title,
                    "module_type": mod_type,
                    "host_department": lead_dept,
                    "primary_lead_name": lead_name,
                    "primary_lead_m3_score": m3_primary_recs[0].get("m3_kg_score") if m3_primary_recs else 0.0,
                    "collaborator_rank": rank_idx,
                    "collaborator_name": c.full_name,
                    "collaborator_position": c.position,
                    "collaborator_department": c.department_name,
                    "collaboration_aspect": c.aspect_title,
                    "aspect_affinity_score": c.aspect_affinity_score,
                    "collaboration_suitability_score": c.collaboration_suitability_score,
                    "rationale": c.rationale,
                })
        elif not collab_eligible:
            csv_rows.append({
                "module_id": mid,
                "module_code": mod_code,
                "module_title": mod_title,
                "module_type": mod_type,
                "host_department": lead_dept,
                "primary_lead_name": lead_name,
                "primary_lead_m3_score": m3_primary_recs[0].get("m3_kg_score") if m3_primary_recs else 0.0,
                "collaborator_rank": 0,
                "collaborator_name": "N/A (Exempt Unit)",
                "collaborator_position": "N/A",
                "collaborator_department": "N/A",
                "collaboration_aspect": "Independent Supervision Only",
                "aspect_affinity_score": 0.0,
                "collaboration_suitability_score": 0.0,
                "rationale": "Exempt from co-delivery: Independent student research project / placement unit.",
            })

    # 6. Write JSONL Output
    unified_jsonl_path = output_dir / "unified_recommendations.jsonl"
    with unified_jsonl_path.open("w", encoding="utf-8") as f:
        for r in unified_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    LOGGER.info(f"Wrote {len(unified_records)} unified records to: {unified_jsonl_path}")

    # 7. Write CSV Review Sheet
    csv_path = output_dir / "m4_v2_case_study_review_sheet.csv"
    if csv_rows:
        try:
            with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)
            LOGGER.info(f"Wrote {len(csv_rows)} case study rows to: {csv_path}")
        except PermissionError:
            LOGGER.warning(f"Could not overwrite {csv_path} (file may be open). Saved JSONL successfully.")

    # 8. Write Summary Report JSON
    summary_path = output_dir / "m4_v2_evaluation_summary.json"
    summary_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_catalog_modules_evaluated": len(m3_recs_by_module),
        "taught_lecture_modules_count": total_taught_modules,
        "non_taught_exempt_modules_count": total_exempt_modules,
        "taught_curriculum_with_collaborations": taught_modules_with_collaborations,
        "taught_curriculum_coverage_pct": round(
            (taught_modules_with_collaborations / max(1, total_taught_modules)) * 100, 2
        ),
        "overall_catalog_collaboration_pct": round(
            (taught_modules_with_collaborations / max(1, len(m3_recs_by_module))) * 100, 2
        ),
        "total_cross_department_collaborations": total_collaborations,
        "average_collaborations_per_taught_module": round(
            total_collaborations / max(1, total_taught_modules), 3
        ),
        "aspect_distribution": aspect_counts,
        "outputs": {
            "unified_recommendations_jsonl": str(unified_jsonl_path),
            "case_study_review_csv": str(csv_path),
            "summary_json": str(summary_path),
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    LOGGER.info(f"Pipeline summary written to: {summary_path}")
    return summary_data


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    out = root / "data" / "outputs" / "m4_v2"
    summary = run_unified_pipeline(project_root=root, output_dir=out)
    print("\n" + json.dumps(summary, indent=2))
