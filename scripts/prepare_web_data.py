from __future__ import annotations

import json
from pathlib import Path


def prepare_web_data() -> None:
    root = Path(__file__).resolve().parents[1]
    m4_v2_dir = root / "data" / "outputs" / "m4_v2"
    web_data_dir = root / "web" / "data"
    web_data_dir.mkdir(parents=True, exist_ok=True)

    jsonl_in = m4_v2_dir / "unified_recommendations.jsonl"
    summary_in = m4_v2_dir / "m4_v2_evaluation_summary.json"

    # 1. Convert JSONL to JSON array
    records = []
    if jsonl_in.exists():
        with jsonl_in.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))

    json_out = web_data_dir / "recommendations.json"
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"Exported {len(records)} records to {json_out} ({json_out.stat().st_size / 1024:.1f} KB)")

    # 2. Copy summary JSON
    if summary_in.exists():
        with summary_in.open("r", encoding="utf-8") as f:
            summary_data = json.load(f)
        summary_out = web_data_dir / "summary.json"
        with summary_out.open("w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"Exported summary data to {summary_out}")


if __name__ == "__main__":
    prepare_web_data()
