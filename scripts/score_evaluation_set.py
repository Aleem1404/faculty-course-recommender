from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_RATINGS = {0, 1, 2}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_rating(value: str) -> int | None:
    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    try:
        rating = int(text)
    except ValueError:
        return None

    if rating not in VALID_RATINGS:
        return None

    return rating


def dcg_at_k(relevances: list[int], k: int = 5) -> float:
    score = 0.0

    for index, rel in enumerate(relevances[:k], start=1):
        gain = (2**rel) - 1
        discount = math.log2(index + 1)
        score += gain / discount

    return score


def ndcg_at_k(relevances: list[int], k: int = 5) -> float:
    actual = dcg_at_k(relevances, k=k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k=k)

    if ideal == 0:
        return 0.0

    return actual / ideal


def precision_at_k(
    relevances: list[int],
    k: int = 5,
    relevant_threshold: int = 1,
) -> float:
    top_k = relevances[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(
        1 for rel in top_k
        if rel >= relevant_threshold
    )

    return relevant_count / len(top_k)


def strict_precision_at_k(
    relevances: list[int],
    k: int = 5,
) -> float:
    top_k = relevances[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(
        1 for rel in top_k
        if rel >= 2
    )

    return relevant_count / len(top_k)


def mean_relevance(relevances: list[int]) -> float:
    if not relevances:
        return 0.0

    return sum(relevances) / len(relevances)


def group_rows_by_module(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        module_id = str(row.get("module_id", "")).strip()

        if not module_id:
            continue

        grouped[module_id].append(row)

    for module_rows in grouped.values():
        module_rows.sort(
            key=lambda row: (
                9999 if str(row.get("rank", "")).strip() == ""
                else int(row["rank"])
            )
        )

    return grouped


def build_module_metric(
    module_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    first = module_rows[0]

    scored_rows = []
    missing_rating_rows = []
    invalid_rating_rows = []

    for row in module_rows:
        raw_rating = row.get(
            "human_relevance_rating",
            "",
        )

        parsed = parse_rating(raw_rating)

        rank_text = str(row.get("rank", "")).strip()

        if rank_text == "":
            continue

        if str(raw_rating).strip() == "":
            missing_rating_rows.append(row)
            continue

        if parsed is None:
            invalid_rating_rows.append(row)
            continue

        scored_rows.append((row, parsed))

    relevances = [rating for _, rating in scored_rows]

    valid_recommendation_count = len(scored_rows)
    unscored_recommendation_count = len(
        missing_rating_rows
    )
    invalid_recommendation_count = len(
        invalid_rating_rows
    )

    rating_counts = Counter(relevances)

    return {
        "module_id": first.get("module_id"),
        "module_code": first.get("module_code"),
        "module_title": first.get("module_title"),
        "module_departments": first.get(
            "module_departments"
        ),
        "module_colleges": first.get(
            "module_colleges"
        ),
        "module_hierarchy_paths": first.get(
            "module_hierarchy_paths"
        ),
        "decision_status": first.get(
            "decision_status"
        ),
        "review_required": (
            str(first.get("review_required", "")).strip().lower()
            == "true"
        ),
        "recommendation_count": sum(
            1
            for row in module_rows
            if str(row.get("rank", "")).strip() != ""
        ),
        "scored_recommendation_count": (
            valid_recommendation_count
        ),
        "missing_rating_count": (
            unscored_recommendation_count
        ),
        "invalid_rating_count": (
            invalid_recommendation_count
        ),
        "rating_counts": {
            "0": rating_counts.get(0, 0),
            "1": rating_counts.get(1, 0),
            "2": rating_counts.get(2, 0),
        },
        "mean_relevance": round(
            mean_relevance(relevances),
            6,
        ),
        "precision_at_5_relaxed": round(
            precision_at_k(
                relevances,
                k=5,
                relevant_threshold=1,
            ),
            6,
        ),
        "precision_at_5_strict": round(
            strict_precision_at_k(
                relevances,
                k=5,
            ),
            6,
        ),
        "ndcg_at_5": round(
            ndcg_at_k(
                relevances,
                k=5,
            ),
            6,
        ),
    }


def summarise_modules(
    module_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    complete_modules = [
        metric
        for metric in module_metrics
        if metric["missing_rating_count"] == 0
        and metric["invalid_rating_count"] == 0
        and metric["scored_recommendation_count"] > 0
    ]

    all_scored_modules = [
        metric
        for metric in module_metrics
        if metric["scored_recommendation_count"] > 0
    ]

    def average(
        metrics: list[dict[str, Any]],
        field: str,
    ) -> float:
        if not metrics:
            return 0.0

        return round(
            sum(metric[field] for metric in metrics)
            / len(metrics),
            6,
        )

    overall_rating_counts = Counter()

    for metric in module_metrics:
        for rating, count in metric[
            "rating_counts"
        ].items():
            overall_rating_counts[rating] += count

    return {
        "module_count": len(module_metrics),
        "modules_with_any_scores": len(
            all_scored_modules
        ),
        "fully_scored_modules": len(
            complete_modules
        ),
        "average_mean_relevance_all_scored": average(
            all_scored_modules,
            "mean_relevance",
        ),
        "average_precision_at_5_relaxed_all_scored": average(
            all_scored_modules,
            "precision_at_5_relaxed",
        ),
        "average_precision_at_5_strict_all_scored": average(
            all_scored_modules,
            "precision_at_5_strict",
        ),
        "average_ndcg_at_5_all_scored": average(
            all_scored_modules,
            "ndcg_at_5",
        ),
        "average_mean_relevance_fully_scored": average(
            complete_modules,
            "mean_relevance",
        ),
        "average_precision_at_5_relaxed_fully_scored": average(
            complete_modules,
            "precision_at_5_relaxed",
        ),
        "average_precision_at_5_strict_fully_scored": average(
            complete_modules,
            "precision_at_5_strict",
        ),
        "average_ndcg_at_5_fully_scored": average(
            complete_modules,
            "ndcg_at_5",
        ),
        "rating_distribution": {
            "0": overall_rating_counts.get("0", 0),
            "1": overall_rating_counts.get("1", 0),
            "2": overall_rating_counts.get("2", 0),
        },
    }


def summarise_by_status(
    module_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for metric in module_metrics:
        grouped[metric["decision_status"]].append(
            metric
        )

    status_summary: dict[str, Any] = {}

    for status, metrics in grouped.items():
        status_summary[status] = summarise_modules(
            metrics
        )

    return status_summary


def collect_validation_issues(
    module_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    modules_with_missing = [
        {
            "module_id": metric["module_id"],
            "module_code": metric["module_code"],
            "module_title": metric["module_title"],
            "missing_rating_count": metric[
                "missing_rating_count"
            ],
        }
        for metric in module_metrics
        if metric["missing_rating_count"] > 0
    ]

    modules_with_invalid = [
        {
            "module_id": metric["module_id"],
            "module_code": metric["module_code"],
            "module_title": metric["module_title"],
            "invalid_rating_count": metric[
                "invalid_rating_count"
            ],
        }
        for metric in module_metrics
        if metric["invalid_rating_count"] > 0
    ]

    return {
        "modules_with_missing_ratings": (
            modules_with_missing
        ),
        "modules_with_invalid_ratings": (
            modules_with_invalid
        ),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    input_path = (
        project_root
        / "data"
        / "outputs"
        / "evaluation"
        / "m2_hierarchy_primary"
        / "evaluation_reviewer_template.csv"
    )

    output_dir = (
        project_root
        / "data"
        / "outputs"
        / "evaluation"
        / "m2_hierarchy_primary"
    )

    module_metrics_path = (
        output_dir
        / "evaluation_module_metrics.json"
    )

    summary_path = (
        output_dir
        / "evaluation_scores_summary.json"
    )

    rows = read_csv(input_path)
    grouped = group_rows_by_module(rows)

    module_metrics = [
        build_module_metric(module_rows)
        for module_rows in grouped.values()
    ]

    module_metrics.sort(
        key=lambda metric: (
            metric["decision_status"],
            metric["module_code"] or "",
            metric["module_id"] or "",
        )
    )

    summary = {
        "input_path": str(input_path),
        "module_metrics_path": str(
            module_metrics_path
        ),
        "scoring_scale": {
            "2": "Relevant",
            "1": "Partly relevant",
            "0": "Not relevant",
        },
        "notes": {
            "precision_at_5_relaxed": (
                "Counts ratings 1 and 2 as relevant."
            ),
            "precision_at_5_strict": (
                "Counts only rating 2 as relevant."
            ),
            "ndcg_at_5": (
                "Uses graded relevance with higher weight "
                "for better-ranked relevant recommendations."
            ),
        },
        "overall": summarise_modules(
            module_metrics
        ),
        "by_decision_status": summarise_by_status(
            module_metrics
        ),
        "validation": collect_validation_issues(
            module_metrics
        ),
    }

    write_json(module_metrics_path, module_metrics)
    write_json(summary_path, summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()