from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("m4_v2.uncovered_modules")


def export_uncovered_modules(
    project_root: Path,
    unified_jsonl_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load core module descriptions for diagnostics
    modules_core_path = project_root / "data" / "processed" / "core" / "modules.jsonl"
    core_modules_by_id: Dict[str, Dict[str, Any]] = {}
    if modules_core_path.exists():
        with modules_core_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                m = json.loads(line)
                core_modules_by_id[m["module_id"]] = m

    uncovered_records: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    department_breakdown: Dict[str, int] = {}
    reason_counts: Dict[str, int] = {}

    total_modules = 0
    total_exempt = 0
    total_taught = 0

    with unified_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_modules += 1
            rec = json.loads(line)

            mod_type = rec.get("module_classification", "taught_academic")
            collab_eligible = rec.get("collaboration_eligible", True)
            if collab_eligible:
                total_taught += 1
            else:
                total_exempt += 1

            collabs = (
                rec.get("layer_2_intelligent_collaboration", {})
                .get("cross_department_collaborations", [])
            )

            # Only process modules with 0 collaborations
            if not collabs:
                mid = rec.get("module_id", "")
                m_code = rec.get("module_code", "")
                m_title = rec.get("module_title", "")
                m_depts = rec.get("module_departments", [])
                m_colleges = rec.get("module_colleges", [])

                l1 = rec.get("layer_1_primary_delivery", {})
                lead_name = l1.get("lead_staff_name", "Unassigned")
                lead_dept = l1.get("lead_department", (m_depts[0] if m_depts else "Unknown"))

                gaps = (
                    rec.get("layer_2_intelligent_collaboration", {})
                    .get("curriculum_gaps_identified", [])
                )

                # Get description text
                core_m = core_modules_by_id.get(mid, {})
                desc = core_m.get("description", "")
                desc_snippet = (desc[:160] + "...") if len(desc) > 160 else desc

                # Determine primary reason for lack of collaboration
                if not collab_eligible:
                    reason = "Non-Taught Independent Project / Placement (Formally Exempt)"
                elif not desc or len(desc.strip()) < 30:
                    reason = "Sparse / Missing Syllabus Description"
                elif gaps and all(g.get("need_level") == "Covered By Primary Lead" for g in gaps):
                    reason = "Saturated / Fully Covered by Primary Lead"
                elif not gaps:
                    reason = "Narrow Domain (Below Interdisciplinary Aspect Thresholds)"
                else:
                    reason = "Internal Specialism (No External Match Passing Affinity Gate)"

                reason_counts[reason] = reason_counts.get(reason, 0) + 1

                # Track department
                dept_label = m_depts[0] if m_depts else "Unassigned Department"
                department_breakdown[dept_label] = department_breakdown.get(dept_label, 0) + 1

                uncovered_record = {
                    "module_id": mid,
                    "module_code": m_code,
                    "module_title": m_title,
                    "module_classification": mod_type,
                    "collaboration_eligible": collab_eligible,
                    "module_departments": m_depts,
                    "module_colleges": m_colleges,
                    "primary_lead_name": lead_name,
                    "primary_lead_department": lead_dept,
                    "syllabus_description": desc,
                    "omission_reason": reason,
                    "identified_gaps": gaps,
                }
                uncovered_records.append(uncovered_record)

                csv_rows.append({
                    "module_id": mid,
                    "module_code": m_code,
                    "module_title": m_title,
                    "module_classification": mod_type,
                    "collaboration_eligible": collab_eligible,
                    "department": dept_label,
                    "primary_lead_name": lead_name,
                    "omission_reason": reason,
                    "description_preview": desc_snippet,
                    "gap_aspect_count": len(gaps),
                })

    # 1. Write JSONL file
    output_jsonl = output_dir / "modules_without_collaborations.jsonl"
    with output_jsonl.open("w", encoding="utf-8") as f:
        for r in uncovered_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    LOGGER.info(f"Wrote {len(uncovered_records)} uncovered modules to: {output_jsonl}")

    # 2. Write CSV file
    output_csv = output_dir / "modules_without_collaborations.csv"
    if csv_rows:
        try:
            with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)
            LOGGER.info(f"Wrote {len(csv_rows)} rows to CSV: {output_csv}")
        except PermissionError:
            LOGGER.warning(f"Could not overwrite {output_csv} (file may be open in Excel). Saved JSONL successfully.")

    # 3. Write Summary JSON
    output_summary = output_dir / "modules_without_collaborations_summary.json"
    taught_uncovered = len([r for r in uncovered_records if r["collaboration_eligible"]])
    summary_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_catalog_modules_evaluated": total_modules,
        "total_taught_modules": total_taught,
        "total_exempt_non_taught_modules": total_exempt,
        "total_without_collaboration": len(uncovered_records),
        "taught_modules_without_collaboration": taught_uncovered,
        "taught_uncovered_percentage": round((taught_uncovered / max(1, total_taught)) * 100, 2),
        "reasons_breakdown": reason_counts,
        "department_breakdown": department_breakdown,
        "output_files": {
            "jsonl": str(output_jsonl),
            "csv": str(output_csv),
        },
    }

    with output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    LOGGER.info(f"Summary written to: {output_summary}")
    return summary_data


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    jsonl_in = root / "data" / "outputs" / "m4_v2" / "unified_recommendations.jsonl"
    out = root / "data" / "outputs" / "m4_v2"

    summary = export_uncovered_modules(
        project_root=root,
        unified_jsonl_path=jsonl_in,
        output_dir=out,
    )
    print("\n" + json.dumps(summary, indent=2))
