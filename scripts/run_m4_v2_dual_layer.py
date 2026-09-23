from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is in sys.path
project_root = Path(__file__).resolve().parents[1]
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from m4_v2.unified_pipeline import run_unified_pipeline


def main() -> None:
    output_dir = project_root / "data" / "outputs" / "m4_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(" Executing M4-v2: Dual-Layer Interdisciplinary Recommender Pipeline")
    print("=" * 70)
    summary = run_unified_pipeline(project_root=project_root, output_dir=output_dir)
    print("\n[✓] M4-v2 Dual-Layer Pipeline completed successfully.")
    print(f"    Outputs written to: {output_dir / 'unified_recommendations.jsonl'}")


if __name__ == "__main__":
    main()
