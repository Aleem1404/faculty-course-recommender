from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def print_module_card(rec: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    code_display = rec.get("module_code") or "N/A"
    title_display = rec.get("module_title", "")
    print(f"MODULE: [{code_display}] {title_display}")
    print(f"Module ID: {rec.get('module_id')}")
    print(f"Classification: {rec.get('module_classification', 'taught_academic').replace('_', ' ').title()}")
    print(f"Collaboration Eligible: {rec.get('collaboration_eligible', True)}")
    print(f"Offering Departments: {rec.get('module_departments')}")
    print("=" * 80)

    # Layer 1
    l1 = rec.get("layer_1_primary_delivery", {})
    role_label = l1.get("supervision_role", "Primary Module Leader")
    print("\n" + "-" * 30 + " LAYER 1: PRIMARY FACULTY DELIVERY " + "-" * 30)
    print(f"Assigned Role: {role_label}")
    print(f"Assigned Lead/Supervisor: {l1.get('lead_staff_name')} ({l1.get('lead_department')})")
    print("\nTop 5 Departmental Recommendations:")
    for r in l1.get("top_5_internal_recommendations", []):
        rank = r.get("rank")
        name = r.get("full_name")
        dept = r.get("department_name")
        score = r.get("m3_kg_score", 0.0)
        exp = r.get("explanation", "")
        topics = r.get("shared_topics", [])
        print(f"  [Rank {rank}] {name} ({dept}) | M3-KG Score: {score:.4f}")
        if topics:
            print(f"         Shared Topics: {topics}")
        if exp:
            print(f"         Evidence: {exp}")

    # Layer 2
    l2 = rec.get("layer_2_intelligent_collaboration", {})
    print("\n" + "-" * 28 + " LAYER 2: INTELLIGENT COLLABORATION " + "-" * 28)
    status = l2.get("status", "N/A")
    print(f"Layer 2 Status: {status}")

    if not rec.get("collaboration_eligible", True):
        print(f"Exemption Rationale: {l2.get('rationale')}")
        print("Note: Placements and dissertations are student-led/independent units requiring supervision rather than guest lectures.")
    else:
        gaps = l2.get("curriculum_gaps_identified", [])
        print(f"\nCurriculum Gaps / Aspect Opportunities Identified ({len(gaps)}):")
        if not gaps:
            print("  * No gaps above aspect affinity threshold.")
        for g in gaps:
            print(f"  * {g.get('aspect_title'):<48} | Need: {g.get('need_level')} (Gap Score: {g.get('gap_score', 0.0):.3f})")

        collabs = l2.get("cross_department_collaborations", [])
        print(f"\nRecommended Cross-Department Collaborators ({len(collabs)}):")
        if not collabs:
            print("  No external collaboration required or identified.")
        for c in collabs:
            rank = c.get("rank")
            name = c.get("full_name")
            dept = c.get("department_name")
            aspect = c.get("aspect_title")
            suit_score = c.get("collaboration_suitability_score", 0.0)
            rat = c.get("rationale")
            print(f"\n  [Collab Rank {rank}] {name} ({dept})")
            print(f"         Target Aspect: {aspect} (Suitability: {suit_score:.4f})")
            print(f"         Rationale: {rat}")


def inspect_module(module_query: str = "CS5707", max_full_cards: int = 3) -> None:
    root = Path(__file__).resolve().parents[1]
    unified_path = root / "data" / "outputs" / "m4_v2" / "unified_recommendations.jsonl"

    if not unified_path.exists():
        print(f"File not found: {unified_path}")
        return

    q = module_query.strip().lower()
    exact_matches: List[Dict[str, Any]] = []
    partial_matches: List[Dict[str, Any]] = []

    with unified_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            m_code = str(rec.get("module_code", "")).strip().lower()
            m_id = str(rec.get("module_id", "")).strip().lower()
            m_title = str(rec.get("module_title", "")).strip().lower()

            # Check exact code or exact ID
            if q == m_code or q == m_id or q == m_title:
                exact_matches.append(rec)
            elif q in m_code or q in m_id or q in m_title:
                partial_matches.append(rec)

    # 1. If exact matches exist, prioritize them
    matches_to_show = exact_matches if exact_matches else partial_matches

    if not matches_to_show:
        print(f"\n[!] No module found matching query: '{module_query}'")
        print("Tip: Try searching by partial title (e.g. 'Sport', 'Artificial Intelligence', 'Neuroscience') or module code ('CS5707', 'BB27AA').")
        return

    if len(matches_to_show) == 1:
        print_module_card(matches_to_show[0])
    elif len(matches_to_show) <= max_full_cards:
        print(f"\nFound {len(matches_to_show)} matching modules for '{module_query}':")
        for rec in matches_to_show:
            print_module_card(rec)
    else:
        print(f"\nFound {len(matches_to_show)} matching modules for '{module_query}'.")
        print("Showing search index (use exact module code or ID for full report):\n")
        print(f"{'Code':<12} | {'Classification':<22} | {'Offering Department':<38} | {'Title'}")
        print("-" * 115)
        for rec in matches_to_show[:20]:
            code = (rec.get("module_code") or "N/A")[:10]
            mtype = rec.get("module_classification", "taught_academic")[:20]
            depts = rec.get("module_departments", ["Unknown"])
            dept_str = (depts[0] if depts else "Unknown")[:36]
            title = rec.get("module_title", "")[:40]
            print(f"{code:<12} | {mtype:<22} | {dept_str:<38} | {title}")
        if len(matches_to_show) > 20:
            print(f"... and {len(matches_to_show) - 20} more modules.")
        print("\nTo view full recommendation for a specific module above, run:")
        first_code = matches_to_show[0].get("module_code")
        first_query = first_code if first_code and first_code != "N/A" else matches_to_show[0].get("module_id")
        print(f"  .\\.venv\\Scripts\\python scripts/check_m4_v2_recommendations.py --query \"{first_query}\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect M4-v2 Dual-Layer Recommendations for any module.")
    parser.add_argument("--query", "-q", type=str, default="CS5707", help="Module code (e.g. CS5704, BB27AA), ID, or keyword in title")
    args = parser.parse_args()
    inspect_module(args.query)
