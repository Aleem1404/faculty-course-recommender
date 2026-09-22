from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


DEFAULT_STAFF_PROFILES = Path(
    "data/outputs/m4/staff_topic_profiles_filtered.jsonl"
)

DEFAULT_MODULE_PROFILES = Path(
    "data/outputs/m4/module_topic_profiles_filtered.jsonl"
)

DEFAULT_OUTPUT = Path(
    "data/outputs/m4/collaboration_candidates.jsonl"
)

DEFAULT_SUMMARY_OUTPUT = Path(
    "data/outputs/m4/collaboration_candidates_summary.json"
)


@dataclass(frozen=True)
class Topic:
    topic_id: str
    label: str
    weight: float


@dataclass
class Profile:
    record_id: str
    record_type: str
    topics: dict[str, Topic]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {path}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object on line {line_number} in {path}"
                )

            yield record


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_profiles(path: Path, expected_type: str) -> dict[str, Profile]:
    profiles: dict[str, Profile] = {}

    for record in read_jsonl(path):
        record_id = str(record.get("source_record_id", "")).strip()
        record_type = str(record.get("source_type", "")).strip().lower()

        if not record_id or record_type != expected_type:
            continue

        topics: dict[str, Topic] = {}

        for item in record.get("topics", []):
            topic_id = str(item.get("topic_id", "")).strip()
            label = str(item.get("topic", "")).strip()

            if not topic_id or not label:
                continue

            weight = safe_float(item.get("weight"), default=0.0)

            if weight <= 0:
                continue

            topics[topic_id] = Topic(
                topic_id=topic_id,
                label=label,
                weight=weight,
            )

        if topics:
            profiles[record_id] = Profile(
                record_id=record_id,
                record_type=record_type,
                topics=topics,
            )

    return profiles


def build_topic_to_staff_index(
    staff_profiles: dict[str, Profile],
) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)

    for staff_id, profile in staff_profiles.items():
        for topic_id in profile.topics:
            index[topic_id].add(staff_id)

    return index


def weighted_module_relevance(
    staff: Profile,
    module: Profile,
) -> tuple[float, list[dict[str, Any]]]:
    module_topic_weight_total = sum(
        topic.weight for topic in module.topics.values()
    )

    if module_topic_weight_total <= 0:
        return 0.0, []

    matching_topics: list[dict[str, Any]] = []
    covered_weight = 0.0

    for topic_id, module_topic in module.topics.items():
        staff_topic = staff.topics.get(topic_id)

        if staff_topic is None:
            continue

        contribution = min(module_topic.weight, staff_topic.weight)
        covered_weight += contribution

        matching_topics.append(
            {
                "topic_id": topic_id,
                "topic": module_topic.label,
                "module_weight": round(module_topic.weight, 6),
                "staff_weight": round(staff_topic.weight, 6),
                "contribution": round(contribution, 6),
            }
        )

    relevance = covered_weight / module_topic_weight_total

    matching_topics.sort(
        key=lambda item: (
            item["contribution"],
            item["topic"].lower(),
        ),
        reverse=True,
    )

    return relevance, matching_topics


def candidate_staff_for_module(
    module: Profile,
    staff_profiles: dict[str, Profile],
    topic_to_staff: dict[str, set[str]],
    min_staff_relevance: float,
    max_individual_candidates: int,
) -> list[dict[str, Any]]:
    possible_staff_ids: set[str] = set()

    for topic_id in module.topics:
        possible_staff_ids.update(topic_to_staff.get(topic_id, set()))

    candidates: list[dict[str, Any]] = []

    for staff_id in possible_staff_ids:
        staff = staff_profiles[staff_id]

        relevance, matched_topics = weighted_module_relevance(
            staff=staff,
            module=module,
        )

        if relevance < min_staff_relevance:
            continue

        candidates.append(
            {
                "staff_id": staff_id,
                "module_relevance": round(relevance, 6),
                "matched_topic_count": len(matched_topics),
                "matched_topics": matched_topics,
            }
        )

    candidates.sort(
        key=lambda item: (
            item["module_relevance"],
            item["matched_topic_count"],
            item["staff_id"],
        ),
        reverse=True,
    )

    return candidates[:max_individual_candidates]


def combined_coverage(
    staff_a: Profile,
    staff_b: Profile,
    module: Profile,
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    module_weight_total = sum(
        topic.weight for topic in module.topics.values()
    )

    if module_weight_total <= 0:
        return 0.0, [], []

    covered_weight = 0.0
    shared_contributions: list[dict[str, Any]] = []
    complementary_contributions: list[dict[str, Any]] = []

    for topic_id, module_topic in module.topics.items():
        a_topic = staff_a.topics.get(topic_id)
        b_topic = staff_b.topics.get(topic_id)

        if a_topic is None and b_topic is None:
            continue

        a_weight = a_topic.weight if a_topic else 0.0
        b_weight = b_topic.weight if b_topic else 0.0

        contribution = min(
            module_topic.weight,
            max(a_weight, b_weight),
        )

        covered_weight += contribution

        evidence = {
            "topic_id": topic_id,
            "topic": module_topic.label,
            "module_weight": round(module_topic.weight, 6),
            "staff_a_weight": round(a_weight, 6),
            "staff_b_weight": round(b_weight, 6),
            "contribution": round(contribution, 6),
        }

        if a_topic is not None and b_topic is not None:
            shared_contributions.append(evidence)
        else:
            provider = "staff_a" if a_topic is not None else "staff_b"

            complementary_contributions.append(
                {
                    **evidence,
                    "provided_by": provider,
                }
            )

    coverage = covered_weight / module_weight_total

    shared_contributions.sort(
        key=lambda item: (
            item["contribution"],
            item["topic"].lower(),
        ),
        reverse=True,
    )

    complementary_contributions.sort(
        key=lambda item: (
            item["contribution"],
            item["topic"].lower(),
        ),
        reverse=True,
    )

    return coverage, shared_contributions, complementary_contributions


def overlap_ratio(
    staff_a: Profile,
    staff_b: Profile,
) -> tuple[float, list[dict[str, str]]]:
    shared_topic_ids = set(staff_a.topics) & set(staff_b.topics)
    union_topic_ids = set(staff_a.topics) | set(staff_b.topics)

    if not union_topic_ids:
        return 0.0, []

    shared_topics = [
        {
            "topic_id": topic_id,
            "topic": staff_a.topics[topic_id].label,
        }
        for topic_id in shared_topic_ids
    ]

    shared_topics.sort(key=lambda item: item["topic"].lower())

    return len(shared_topic_ids) / len(union_topic_ids), shared_topics


def pair_candidate(
    module: Profile,
    staff_a: Profile,
    staff_b: Profile,
    staff_a_relevance: float,
    staff_b_relevance: float,
    minimum_complementarity: float,
) -> dict[str, Any] | None:
    combined_score, shared_module_topics, complementary_topics = (
        combined_coverage(
            staff_a=staff_a,
            staff_b=staff_b,
            module=module,
        )
    )

    best_individual_score = max(
        staff_a_relevance,
        staff_b_relevance,
    )

    complementarity_gain = max(
        0.0,
        combined_score - best_individual_score,
    )

    if complementarity_gain < minimum_complementarity:
        return None

    staff_overlap, shared_staff_topics = overlap_ratio(
        staff_a=staff_a,
        staff_b=staff_b,
    )

    shared_foundation_score = min(staff_overlap, 0.20) / 0.20

    collaboration_score = (
        0.50 * combined_score
        + 0.35 * complementarity_gain
        + 0.10 * best_individual_score
        + 0.05 * shared_foundation_score
    )

    return {
        "module_id": module.record_id,
        "staff_a_id": staff_a.record_id,
        "staff_b_id": staff_b.record_id,
        "staff_a_module_relevance": round(staff_a_relevance, 6),
        "staff_b_module_relevance": round(staff_b_relevance, 6),
        "combined_module_coverage": round(combined_score, 6),
        "best_individual_coverage": round(best_individual_score, 6),
        "complementarity_gain": round(complementarity_gain, 6),
        "staff_topic_overlap_ratio": round(staff_overlap, 6),
        "collaboration_score": round(collaboration_score, 6),
        "shared_module_topics": shared_module_topics,
        "complementary_module_topics": complementary_topics,
        "shared_staff_topics": shared_staff_topics,
    }


def build_candidates(
    staff_profiles: dict[str, Profile],
    module_profiles: dict[str, Profile],
    min_staff_relevance: float,
    max_individual_candidates: int,
    min_complementarity: float,
    max_pairs_per_module: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    topic_to_staff = build_topic_to_staff_index(staff_profiles)

    all_candidates: list[dict[str, Any]] = []

    modules_with_candidates = 0
    modules_without_candidates = 0
    candidate_staff_total = 0
    pair_candidates_before_limit = 0

    for module_id, module in module_profiles.items():
        individual_candidates = candidate_staff_for_module(
            module=module,
            staff_profiles=staff_profiles,
            topic_to_staff=topic_to_staff,
            min_staff_relevance=min_staff_relevance,
            max_individual_candidates=max_individual_candidates,
        )

        if len(individual_candidates) < 2:
            modules_without_candidates += 1
            continue

        candidate_staff_total += len(individual_candidates)

        relevance_by_staff_id = {
            candidate["staff_id"]: candidate["module_relevance"]
            for candidate in individual_candidates
        }

        module_pairs: list[dict[str, Any]] = []

        for staff_a_id, staff_b_id in combinations(
            sorted(relevance_by_staff_id),
            2,
        ):
            candidate = pair_candidate(
                module=module,
                staff_a=staff_profiles[staff_a_id],
                staff_b=staff_profiles[staff_b_id],
                staff_a_relevance=relevance_by_staff_id[staff_a_id],
                staff_b_relevance=relevance_by_staff_id[staff_b_id],
                minimum_complementarity=min_complementarity,
            )

            if candidate is not None:
                module_pairs.append(candidate)

        if not module_pairs:
            modules_without_candidates += 1
            continue

        modules_with_candidates += 1
        pair_candidates_before_limit += len(module_pairs)

        module_pairs.sort(
            key=lambda item: (
                item["collaboration_score"],
                item["complementarity_gain"],
                item["combined_module_coverage"],
                item["staff_a_id"],
                item["staff_b_id"],
            ),
            reverse=True,
        )

        for rank, candidate in enumerate(
            module_pairs[:max_pairs_per_module],
            start=1,
        ):
            candidate["rank"] = rank
            candidate["individual_candidate_pool_size"] = len(
                individual_candidates
            )
            all_candidates.append(candidate)

    summary = {
        "staff_profiles_loaded": len(staff_profiles),
        "module_profiles_loaded": len(module_profiles),
        "modules_with_collaboration_candidates": modules_with_candidates,
        "modules_without_collaboration_candidates": modules_without_candidates,
        "candidate_staff_total_across_modules": candidate_staff_total,
        "pair_candidates_before_per_module_limit": pair_candidates_before_limit,
        "pair_candidates_written": len(all_candidates),
        "average_individual_candidates_per_eligible_module": round(
            candidate_staff_total / modules_with_candidates,
            4,
        )
        if modules_with_candidates
        else 0.0,
    }

    return all_candidates, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate M4 faculty collaboration candidates from filtered "
            "staff and module topic profiles."
        )
    )

    parser.add_argument(
        "--staff-profiles",
        type=Path,
        default=DEFAULT_STAFF_PROFILES,
        help="Path to filtered staff topic profiles JSONL.",
    )

    parser.add_argument(
        "--module-profiles",
        type=Path,
        default=DEFAULT_MODULE_PROFILES,
        help="Path to filtered module topic profiles JSONL.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for collaboration candidate JSONL output.",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Path for candidate-generation summary JSON.",
    )

    parser.add_argument(
        "--min-staff-relevance",
        type=float,
        default=0.10,
        help=(
            "Minimum weighted module-topic coverage required for an "
            "individual staff candidate. Default: 0.10."
        ),
    )

    parser.add_argument(
        "--max-individual-candidates",
        type=int,
        default=20,
        help=(
            "Maximum individual staff candidates retained per module "
            "before creating staff pairs. Default: 20."
        ),
    )

    parser.add_argument(
        "--min-complementarity",
        type=float,
        default=0.05,
        help=(
            "Minimum extra module coverage supplied by the pair beyond "
            "the stronger individual candidate. Default: 0.05."
        ),
    )

    parser.add_argument(
        "--max-pairs-per-module",
        type=int,
        default=50,
        help=(
            "Maximum ranked collaboration pairs written per module. "
            "Default: 50."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.staff_profiles.exists():
        raise FileNotFoundError(
            f"Staff profile file not found: {args.staff_profiles}"
        )

    if not args.module_profiles.exists():
        raise FileNotFoundError(
            f"Module profile file not found: {args.module_profiles}"
        )

    if args.max_individual_candidates < 2:
        raise ValueError(
            "--max-individual-candidates must be at least 2."
        )

    if args.max_pairs_per_module < 1:
        raise ValueError(
            "--max-pairs-per-module must be at least 1."
        )

    staff_profiles = load_profiles(
        path=args.staff_profiles,
        expected_type="staff",
    )

    module_profiles = load_profiles(
        path=args.module_profiles,
        expected_type="module",
    )

    candidates, summary = build_candidates(
        staff_profiles=staff_profiles,
        module_profiles=module_profiles,
        min_staff_relevance=args.min_staff_relevance,
        max_individual_candidates=args.max_individual_candidates,
        min_complementarity=args.min_complementarity,
        max_pairs_per_module=args.max_pairs_per_module,
    )

    candidates.sort(
        key=lambda item: (
            item["module_id"],
            item["rank"],
            item["staff_a_id"],
            item["staff_b_id"],
        )
    )

    write_jsonl(args.output, candidates)

    final_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "staff_profiles": str(args.staff_profiles),
            "module_profiles": str(args.module_profiles),
        },
        "parameters": {
            "min_staff_relevance": args.min_staff_relevance,
            "max_individual_candidates": args.max_individual_candidates,
            "min_complementarity": args.min_complementarity,
            "max_pairs_per_module": args.max_pairs_per_module,
        },
        "outputs": {
            "collaboration_candidates": str(args.output),
        },
        "counts": summary,
    }

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    with args.summary_output.open("w", encoding="utf-8") as file:
        json.dump(final_summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(final_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()