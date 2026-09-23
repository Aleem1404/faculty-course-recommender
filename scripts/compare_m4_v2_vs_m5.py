from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compare_models(module_code_filter: Optional[str] = None) -> None:
    m4_path = PROJECT_ROOT / "data" / "outputs" / "m4_v2" / "unified_recommendations.jsonl"
    m5_path = PROJECT_ROOT / "data" / "outputs" / "m5" / "m5_balanced_recommendations.jsonl"
    output_dir = PROJECT_ROOT / "data" / "outputs" / "evaluation" / "m4_vs_m5"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not m4_path.exists() or not m5_path.exists():
        print(f"Error: Missing input files ({m4_path} or {m5_path})")
        return

    m4_records = load_jsonl(m4_path)
    m5_records = load_jsonl(m5_path)

    m4_by_id = {r["module_id"]: r for r in m4_records}
    m5_by_id = {r["module_id"]: r for r in m5_records}

    common_ids = [mid for mid in m4_by_id if mid in m5_by_id]

    comparisons = []
    case_rows = []

    lead_changed_count = 0
    stable_lead_count = 0
    exempt_count = 0

    total_m4_scores = []
    total_m5_scores = []

    for mid in common_ids:
        m4 = m4_by_id[mid]
        m5 = m5_by_id[mid]

        mod_code = m4.get("module_code", "N/A")
        mod_title = m4.get("module_title", "Untitled")
        classification = m4.get("module_classification", "taught_academic")
        dept = (m4.get("module_departments") or ["Unknown"])[0]

        l1_m4 = m4.get("layer_1_primary_delivery", {})
        l1_m5 = m5.get("m5_balanced_primary_delivery", {})

        m4_lead_id = l1_m4.get("lead_staff_id")
        m4_lead_name = l1_m4.get("lead_staff_name")
        m4_lead_dept = l1_m4.get("lead_department")
        top_recs_m4 = l1_m4.get("top_5_internal_recommendations", [])
        m4_lead_score = float(top_recs_m4[0].get("m3_kg_score") or 0.0) if top_recs_m4 else 0.0

        m5_lead_id = l1_m5.get("lead_staff_id")
        m5_lead_name = l1_m5.get("lead_staff_name")
        m5_lead_dept = l1_m5.get("lead_department")
        m5_lead_score = float(l1_m5.get("assigned_relevance_score") or 0.0)
        rank_shift = int(l1_m5.get("rank_shift_from_greedy") or 0)
        is_displaced = bool(l1_m5.get("is_capacity_constrained"))

        exp = l1_m5.get("m5_workload_policy_explanation", {})
        policy_flag = exp.get("policy_flag", "UnconstrainedOptimal")
        rationale = exp.get("allocation_rationale", "")

        if classification == "independent_project_placement":
            exempt_count += 1
            lead_changed = False
        else:
            total_m4_scores.append(m4_lead_score)
            total_m5_scores.append(m5_lead_score)
            lead_changed = (m4_lead_id != m5_lead_id)
            if lead_changed:
                lead_changed_count += 1
            else:
                stable_lead_count += 1

        comp_record = {
            "module_id": mid,
            "module_code": mod_code,
            "module_title": mod_title,
            "department": dept,
            "classification": classification,
            "m4_greedy_lead": {
                "staff_id": m4_lead_id,
                "full_name": m4_lead_name,
                "department": m4_lead_dept,
                "score": round(m4_lead_score, 4),
            },
            "m5_balanced_lead": {
                "staff_id": m5_lead_id,
                "full_name": m5_lead_name,
                "department": m5_lead_dept,
                "score": round(m5_lead_score, 4),
                "rank_shift": rank_shift,
                "is_capacity_displaced": is_displaced,
                "policy_flag": policy_flag,
            },
            "lead_changed_due_to_capacity": lead_changed,
            "relevance_tradeoff_delta": round(m4_lead_score - m5_lead_score, 4),
            "allocation_rationale": rationale,
        }
        comparisons.append(comp_record)

        case_rows.append({
            "module_code": mod_code,
            "module_title": mod_title,
            "department": dept,
            "classification": classification,
            "m4_lead_name": m4_lead_name,
            "m4_lead_score": round(m4_lead_score, 4),
            "m5_lead_name": m5_lead_name,
            "m5_lead_score": round(m5_lead_score, 4),
            "lead_status": "DISPLACED_BY_QUOTA" if is_displaced else ("EXEMPT" if classification == "independent_project_placement" else "RETAINED_OPTIMAL"),
            "rank_shift": rank_shift,
            "tradeoff_delta": round(m4_lead_score - m5_lead_score, 4),
            "policy_summary": exp.get("summary", ""),
        })

    # Summary Statistics
    total_taught = len(total_m4_scores)
    avg_m4 = sum(total_m4_scores) / total_taught if total_taught else 0.0
    avg_m5 = sum(total_m5_scores) / total_taught if total_taught else 0.0

    summary = {
        "comparison_title": "Ablation Comparison: M4-v2 (Greedy Matching) vs M5 (Load-Balanced Allocation)",
        "total_catalog_modules": len(common_ids),
        "total_taught_modules": total_taught,
        "exempt_modules": exempt_count,
        "reallocated_modules_count": lead_changed_count,
        "reallocated_modules_pct": round((lead_changed_count / total_taught * 100.0) if total_taught else 0.0, 2),
        "stable_modules_count": stable_lead_count,
        "stable_modules_pct": round((stable_lead_count / total_taught * 100.0) if total_taught else 0.0, 2),
        "average_m4_greedy_score": round(avg_m4, 4),
        "average_m5_balanced_score": round(avg_m5, 4),
        "score_tradeoff_delta": round(avg_m4 - avg_m5, 4),
        "score_retention_pct": round((avg_m5 / avg_m4 * 100.0) if avg_m4 else 100.0, 2),
        "outputs": {
            "detailed_comparisons_json": str(output_dir / "m4_v2_vs_m5_module_comparisons.json"),
            "comparison_table_csv": str(output_dir / "m4_v2_vs_m5_case_rows.csv"),
        }
    }

    # Save files
    write_json(output_dir / "m4_v2_vs_m5_module_comparisons.json", comparisons)
    write_json(output_dir / "m4_v2_vs_m5_summary.json", summary)
    write_csv(output_dir / "m4_v2_vs_m5_case_rows.csv", case_rows)

    # Print Report
    print("\n" + "=" * 76)
    print("         ABLATION COMPARISON: M4-v2 (GREEDY) vs M5 (LOAD-BALANCED)")
    print("=" * 76)
    print(f"Total Catalog Modules Compared : {summary['total_catalog_modules']}")
    print(f"Taught Academic Modules        : {summary['total_taught_modules']}")
    print(f"Exempt Supervision/Placements  : {summary['exempt_modules']}")
    print("-" * 76)
    print(f"Reallocated under Policy Quota : {summary['reallocated_modules_count']} modules ({summary['reallocated_modules_pct']}%)")
    print(f"Retained Direct Rank #1 Fit    : {summary['stable_modules_count']} modules ({summary['stable_modules_pct']}%)")
    print(f"Average Relevance Retention    : {summary['average_m4_greedy_score']:.4f} (M4) -> {summary['average_m5_balanced_score']:.4f} (M5) [{summary['score_retention_pct']}% Retention]")
    print(f"Mean Score Trade-off Delta     : -{summary['score_tradeoff_delta']:.4f}")
    print("=" * 76)

    # If single module requested or sample spotlight
    spotlight_code = (module_code_filter or "CS5707").upper()
    spotlight_matches = [c for c in comparisons if c["module_code"].upper() == spotlight_code]

    if spotlight_matches:
        print(f"\n[SPOTLIGHT MODULE CASE: {spotlight_code}]")
        for match in spotlight_matches:
            print(f"  Title      : {match['module_title']}")
            print(f"  Department : {match['department']}")
            print(f"  M4 Lead    : {match['m4_greedy_lead']['full_name']} ({match['m4_greedy_lead']['department']}) - Score: {match['m4_greedy_lead']['score']}")
            print(f"  M5 Lead    : {match['m5_balanced_lead']['full_name']} ({match['m5_balanced_lead']['department']}) - Score: {match['m5_balanced_lead']['score']}")
            print(f"  Status     : {match['m5_balanced_lead']['policy_flag']} (Rank Shift: {match['m5_balanced_lead']['rank_shift']})")
            print(f"  Rationale  : {match['allocation_rationale']}")
        print("-" * 76)

    # Also show a sample module where capacity displacement occurred
    displaced_sample = next((c for c in comparisons if c["lead_changed_due_to_capacity"]), None)
    if displaced_sample:
        print(f"\n[SAMPLE CAPACITY-BALANCED CASE: {displaced_sample['module_code']} - {displaced_sample['module_title']}]")
        print(f"  Department : {displaced_sample['department']}")
        print(f"  M4 Greedy  : {displaced_sample['m4_greedy_lead']['full_name']} (Rank #1, Score: {displaced_sample['m4_greedy_lead']['score']})")
        print(f"  M5 Balanced: {displaced_sample['m5_balanced_lead']['full_name']} (Rank #{displaced_sample['m5_balanced_lead']['rank_shift'] + 1}, Score: {displaced_sample['m5_balanced_lead']['score']})")
        print(f"  Rationale  : {displaced_sample['allocation_rationale']}")
        print("=" * 76 + "\n")

    print(f"Detailed JSON comparison exported to: {summary['outputs']['detailed_comparisons_json']}")
    print(f"Full CSV comparison table exported to: {summary['outputs']['comparison_table_csv']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare M4-v2 (Greedy) vs M5 (Load-Balanced).")
    parser.add_argument("--module-code", type=str, default=None, help="Module code to spotlight (e.g. CS5707).")
    args = parser.parse_args()
    compare_models(module_code_filter=args.module_code)


if __name__ == "__main__":
    main()
