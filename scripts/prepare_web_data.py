from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def prepare_web_data() -> None:
    root = Path(__file__).resolve().parents[1]
    m5_dir = root / "data" / "outputs" / "m5"
    m4_v2_dir = root / "data" / "outputs" / "m4_v2"
    core_dir = root / "data" / "processed" / "core"
    web_data_dir = root / "web" / "data"
    web_data_dir.mkdir(parents=True, exist_ok=True)

    # Prefer M5 balanced recommendations if present, fallback to M4-v2
    m5_jsonl = m5_dir / "m5_balanced_recommendations.jsonl"
    m4_jsonl = m4_v2_dir / "unified_recommendations.jsonl"
    jsonl_in = m5_jsonl if m5_jsonl.exists() else m4_jsonl

    m5_summary_in = m5_dir / "m5_fairness_evaluation_summary.json"
    m4_summary_in = m4_v2_dir / "m4_v2_evaluation_summary.json"

    courses_in = core_dir / "courses.jsonl"
    links_in = core_dir / "course_module_links.jsonl"

    print(f"[*] Sourcing recommendations from: {jsonl_in.name}")

    # 1. Load Courses
    courses_list = []
    courses_by_id = {}
    if courses_in.exists():
        with courses_in.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                c = json.loads(line)
                c_data = {
                    "course_id": c["course_id"],
                    "course_title": c.get("course_title", ""),
                    "departments": c.get("departments", []),
                    "colleges": c.get("colleges", []),
                    "level": c.get("level", ""),
                }
                courses_list.append(c_data)
                courses_by_id[c["course_id"]] = c_data

    # Sort courses alphabetically
    courses_list.sort(key=lambda x: x["course_title"].lower())

    courses_out = web_data_dir / "courses.json"
    with courses_out.open("w", encoding="utf-8") as f:
        json.dump(courses_list, f, ensure_ascii=False, indent=2)
    print(f"Exported {len(courses_list)} courses to {courses_out.name}")

    # 2. Load Course-Module Links
    mod_courses = defaultdict(list)
    if links_in.exists():
        with links_in.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                link = json.loads(line)
                mod_courses[link["module_id"]].append({
                    "course_id": link["course_id"],
                    "course_title": link.get("course_title", ""),
                    "year_of_study": link.get("year_of_study", ""),
                    "module_type": link.get("module_type", ""),
                    "departments": link.get("course_departments", []),
                })

    # 3. Convert JSONL to JSON array with embedded course info
    records = []
    if jsonl_in.exists():
        with jsonl_in.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                mid = rec.get("module_id", "")
                rec["courses"] = mod_courses.get(mid, [])
                records.append(rec)

    json_out = web_data_dir / "recommendations.json"
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"Exported {len(records)} records with course links to {json_out.name} ({json_out.stat().st_size / 1024:.1f} KB)")

    # 4. Copy and merge summary JSON
    summary_data = {}
    if m4_summary_in.exists():
        with m4_summary_in.open("r", encoding="utf-8") as f:
            summary_data = json.load(f)

    if m5_summary_in.exists():
        with m5_summary_in.open("r", encoding="utf-8") as f:
            m5_summary = json.load(f)
            summary_data["m5_fairness_summary"] = m5_summary

    m6_dir = root / "data" / "outputs" / "m6"
    m6_json = m6_dir / "consolidated_ir_benchmark.json"
    if m6_json.exists():
        with m6_json.open("r", encoding="utf-8") as f:
            m6_data = json.load(f)
            summary_data["m6_benchmark_summary"] = m6_data
        
        # Also copy m6_benchmark.json directly to web/data/
        m6_out = web_data_dir / "m6_benchmark.json"
        with m6_out.open("w", encoding="utf-8") as f:
            json.dump(m6_data, f, indent=2, ensure_ascii=False)
        print(f"Exported M6 benchmark to {m6_out.name}")

    summary_out = web_data_dir / "summary.json"
    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"Exported unified summary data to {summary_out.name}")

    # 5. Copy M6 Charts to web/data/charts/
    m6_charts_dir = m6_dir / "charts"
    web_charts_dir = web_data_dir / "charts"
    if m6_charts_dir.exists():
        web_charts_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for chart_file in m6_charts_dir.glob("*.png"):
            shutil.copy2(chart_file, web_charts_dir / chart_file.name)
            print(f"Copied chart: {chart_file.name} to web/data/charts/")


if __name__ == "__main__":
    prepare_web_data()

