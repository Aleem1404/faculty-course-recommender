from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RECOMMENDATIONS_INPUT = Path(
    "data/outputs/m4/collaboration_recommendations_enriched.jsonl"
)

DEFAULT_STAFF_PROFILES_INPUT = Path(
    "data/outputs/m4/staff_topic_profiles_filtered.jsonl"
)

DEFAULT_TOPIC_VECTORS_INPUT = Path(
    "data/outputs/m4/topic_vectors.jsonl"
)

DEFAULT_OUTPUT = Path(
    "data/outputs/m4/"
    "collaboration_recommendations_semantic_foundation.jsonl"
)

DEFAULT_SUMMARY_OUTPUT = Path(
    "data/outputs/m4/"
    "semantic_shared_foundation_summary.json"
)

DEFAULT_FAILURES_OUTPUT = Path(
    "data/outputs/m4/"
    "semantic_shared_foundation_missing_vectors.jsonl"
)


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
                    f"Expected a JSON object on line {line_number} in {path}"
                )

            yield record


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def as_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_vector(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None

    vector: list[float] = []

    for item in value:
        try:
            vector.append(float(item))
        except (TypeError, ValueError):
            return None

    if not vector:
        return None

    return vector


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    if len(vector_a) != len(vector_b):
        return 0.0

    dot_product = sum(
        item_a * item_b
        for item_a, item_b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(item * item for item in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(item * item for item in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# def extract_topic_id(record: dict[str, Any]) -> str:
#     for key in [
#         "canonical_topic_id",
#         "topic_id",
#         "id",
#     ]:
#         value = as_text(record.get(key))

#         if value:
#             return value

#     return ""


# def extract_topic_label(record: dict[str, Any]) -> str:
#     for key in [
#         "canonical_label",
#         "topic",
#         "topic_label",
#         "label",
#         "normalised_text",
#         "normalized_text",
#     ]:
#         value = as_text(record.get(key))

#         if value:
#             return value

#     return ""


def extract_vector(record: dict[str, Any]) -> list[float] | None:
    for key in [
        "embedding",
        "vector",
        "topic_vector",
        "values",
    ]:
        vector = safe_vector(record.get(key))

        if vector is not None:
            return vector

    return None

def normalise_topic_key(value: Any) -> str:
    text = as_text(value).lower()

    if not text:
        return ""

    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = " ".join(text.split())

    cleaned = []

    for token in text.split():
        token = "".join(
            character
            for character in token
            if character.isalnum()
        )

        if token:
            cleaned.append(token)

    return " ".join(cleaned)

def load_topic_vectors(
    path: Path,
) -> tuple[
    dict[str, list[float]],
    dict[str, int],
]:
    vectors: dict[str, list[float]] = {}

    statistics = {
        "records_read": 0,
        "vectors_loaded": 0,
        "records_skipped_missing_normalised_text": 0,
        "records_skipped_invalid_vector": 0,
        "duplicate_normalised_text_keys": 0,
    }

    for record in read_jsonl(path):
        statistics["records_read"] += 1

        topic_key = normalise_topic_key(
            record.get("normalised_text")
        )

        vector = safe_vector(record.get("vector"))

        if not topic_key:
            statistics[
                "records_skipped_missing_normalised_text"
            ] += 1
            continue

        if vector is None:
            statistics["records_skipped_invalid_vector"] += 1
            continue

        if topic_key in vectors:
            statistics["duplicate_normalised_text_keys"] += 1
            continue

        vectors[topic_key] = vector
        statistics["vectors_loaded"] += 1

    return vectors, statistics

def load_staff_topics(
    path: Path,
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, int],
]:
    staff_topics: dict[str, dict[str, dict[str, Any]]] = {}

    statistics = {
        "profiles_read": 0,
        "profiles_loaded": 0,
        "topics_loaded": 0,
        "topics_skipped": 0,
    }

    for profile in read_jsonl(path):
        statistics["profiles_read"] += 1

        staff_id = as_text(profile.get("source_record_id"))
        source_type = as_text(profile.get("source_type")).lower()

        if not staff_id or source_type != "staff":
            continue

        topics_by_id: dict[str, dict[str, Any]] = {}

        for topic in profile.get("topics", []):
            if not isinstance(topic, dict):
                statistics["topics_skipped"] += 1
                continue

            topic_id = as_text(topic.get("topic_id"))
            label = as_text(topic.get("topic"))
            weight = safe_float(topic.get("weight"))

            if not topic_id or not label or weight <= 0:
                statistics["topics_skipped"] += 1
                continue

            topics_by_id[topic_id] = {
                "topic_id": topic_id,
                "topic": label,
                "weight": weight,
            }

            statistics["topics_loaded"] += 1

        if topics_by_id:
            staff_topics[staff_id] = topics_by_id
            statistics["profiles_loaded"] += 1

    return staff_topics, statistics


def module_relevant_topic_ids(
    recommendation: dict[str, Any],
) -> set[str]:
    relevant_ids: set[str] = set()

    for key in [
        "shared_module_topics",
        "complementary_module_topics",
    ]:
        topics = recommendation.get(key, [])

        if not isinstance(topics, list):
            continue

        for topic in topics:
            if not isinstance(topic, dict):
                continue

            topic_id = as_text(topic.get("topic_id"))

            if topic_id:
                relevant_ids.add(topic_id)

    return relevant_ids


def exact_shared_foundation(
    recommendation: dict[str, Any],
) -> list[dict[str, str]]:
    shared_topics = recommendation.get(
        "shared_staff_topics",
        [],
    )

    if not isinstance(shared_topics, list):
        shared_topics = []

    evidence: list[dict[str, str]] = []

    for topic in shared_topics:
        if not isinstance(topic, dict):
            continue

        topic_id = as_text(topic.get("topic_id"))
        label = as_text(topic.get("topic"))

        if topic_id or label:
            evidence.append(
                {
                    "topic_id": topic_id,
                    "topic": label,
                }
            )

    if evidence:
        return evidence

    shared_module_topics = recommendation.get(
        "shared_module_topics",
        [],
    )

    if not isinstance(shared_module_topics, list):
        return []

    for topic in shared_module_topics:
        if not isinstance(topic, dict):
            continue

        topic_id = as_text(topic.get("topic_id"))
        label = as_text(topic.get("topic"))

        if topic_id or label:
            evidence.append(
                {
                    "topic_id": topic_id,
                    "topic": label,
                }
            )

    return evidence


def select_relevant_staff_topics(
    staff_topics: dict[str, dict[str, Any]],
    relevant_module_topic_ids: set[str],
    maximum_topics: int,
) -> list[dict[str, Any]]:
    if not staff_topics:
        return []

    exact_relevant = [
        topic
        for topic_id, topic in staff_topics.items()
        if topic_id in relevant_module_topic_ids
    ]

    if exact_relevant:
        selected = exact_relevant
    else:
        selected = list(staff_topics.values())

    selected.sort(
        key=lambda topic: (
            safe_float(topic.get("weight")),
            as_text(topic.get("topic")).lower(),
        ),
        reverse=True,
    )

    return selected[:maximum_topics]


def best_semantic_foundation(
    staff_a_topics: dict[str, dict[str, Any]],
    staff_b_topics: dict[str, dict[str, Any]],
    topic_vectors: dict[str, list[float]],
    relevant_module_topic_ids: set[str],
    threshold: float,
    maximum_topics_per_staff: int,
) -> tuple[
    list[dict[str, Any]],
    float,
    list[dict[str, Any]],
]:
    candidate_topics_a = select_relevant_staff_topics(
        staff_topics=staff_a_topics,
        relevant_module_topic_ids=relevant_module_topic_ids,
        maximum_topics=maximum_topics_per_staff,
    )

    candidate_topics_b = select_relevant_staff_topics(
        staff_topics=staff_b_topics,
        relevant_module_topic_ids=relevant_module_topic_ids,
        maximum_topics=maximum_topics_per_staff,
    )

    best_similarity = 0.0
    semantic_evidence: list[dict[str, Any]] = []
    missing_vector_records: list[dict[str, Any]] = []

    for topic_a in candidate_topics_a:
        topic_a_id = as_text(topic_a.get("topic_id"))
        topic_a_label = as_text(topic_a.get("topic"))

        topic_a_key = normalise_topic_key(topic_a_label)
        vector_a = topic_vectors.get(topic_a_key)

        if vector_a is None:
            missing_vector_records.append(
                {
                    "staff_side": "staff_a",
                    "topic_id": topic_a_id,
                    "topic": topic_a_label,
                    "normalised_lookup_key": topic_a_key,
                }
            )
            continue

        for topic_b in candidate_topics_b:
            topic_b_id = as_text(topic_b.get("topic_id"))
            topic_b_label = as_text(topic_b.get("topic"))

            if topic_a_id == topic_b_id:
                continue

            topic_b_key = normalise_topic_key(topic_b_label)
            vector_b = topic_vectors.get(topic_b_key)

            if vector_b is None:
                missing_vector_records.append(
                    {
                        "staff_side": "staff_b",
                        "topic_id": topic_b_id,
                        "topic": topic_b_label,
                        "normalised_lookup_key": topic_b_key,
                    }
                )
                continue

            similarity = cosine_similarity(
                vector_a=vector_a,
                vector_b=vector_b,
            )

            best_similarity = max(best_similarity, similarity)

            if similarity >= threshold:
                semantic_evidence.append(
                    {
                        "staff_a_topic_id": topic_a_id,
                        "staff_a_topic": topic_a_label,
                        "staff_a_topic_weight": round(
                            safe_float(topic_a.get("weight")),
                            6,
                        ),
                        "staff_b_topic_id": topic_b_id,
                        "staff_b_topic": topic_b_label,
                        "staff_b_topic_weight": round(
                            safe_float(topic_b.get("weight")),
                            6,
                        ),
                        "semantic_similarity": round(
                            similarity,
                            6,
                        ),
                        "topic_a_is_module_relevant": (
                            topic_a_id
                            in relevant_module_topic_ids
                        ),
                        "topic_b_is_module_relevant": (
                            topic_b_id
                            in relevant_module_topic_ids
                        ),
                    }
                )

    semantic_evidence.sort(
        key=lambda item: (
            item["semantic_similarity"],
            item["staff_a_topic_weight"],
            item["staff_b_topic_weight"],
        ),
        reverse=True,
    )

    return (
        semantic_evidence,
        round(best_similarity, 6),
        missing_vector_records,
    )

def enrich_foundation_record(
    recommendation: dict[str, Any],
    staff_topics: dict[str, dict[str, dict[str, Any]]],
    topic_vectors: dict[str, list[float]],
    threshold: float,
    maximum_topics_per_staff: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    staff_a_id = as_text(recommendation.get("staff_a_id"))
    staff_b_id = as_text(recommendation.get("staff_b_id"))

    staff_a_topics = staff_topics.get(staff_a_id, {})
    staff_b_topics = staff_topics.get(staff_b_id, {})

    exact_evidence = exact_shared_foundation(recommendation)

    relevant_module_topic_ids = module_relevant_topic_ids(
        recommendation
    )

    missing_vector_records: list[dict[str, Any]] = []

    if exact_evidence:
        enriched = {
            **recommendation,
            "foundation_type": "exact",
            "has_shared_foundation_exact": True,
            "has_shared_foundation_semantic": True,
            "maximum_foundation_similarity": 1.0,
            "exact_foundation_topics": exact_evidence,
            "semantic_foundation_topics": [],
            "semantic_similarity_threshold": threshold,
        }

        return enriched, missing_vector_records

    semantic_evidence, maximum_similarity, missing_vectors = (
        best_semantic_foundation(
            staff_a_topics=staff_a_topics,
            staff_b_topics=staff_b_topics,
            topic_vectors=topic_vectors,
            relevant_module_topic_ids=relevant_module_topic_ids,
            threshold=threshold,
            maximum_topics_per_staff=maximum_topics_per_staff,
        )
    )

    missing_vector_records.extend(missing_vectors)

    if semantic_evidence:
        foundation_type = "semantic"
        has_semantic_foundation = True
    else:
        foundation_type = "none"
        has_semantic_foundation = False

    enriched = {
        **recommendation,
        "foundation_type": foundation_type,
        "has_shared_foundation_exact": False,
        "has_shared_foundation_semantic": (
            has_semantic_foundation
        ),
        "maximum_foundation_similarity": maximum_similarity,
        "exact_foundation_topics": [],
        "semantic_foundation_topics": semantic_evidence[:5],
        "semantic_similarity_threshold": threshold,
    }

    return enriched, missing_vector_records


def first_recommendation_per_module(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}

    for record in records:
        module_id = as_text(record.get("module_id"))

        if not module_id:
            continue

        current = selected.get(module_id)

        if current is None:
            selected[module_id] = record
            continue

        current_rank = int(current.get("rank", 999999))
        incoming_rank = int(record.get("rank", 999999))

        if incoming_rank < current_rank:
            selected[module_id] = record

    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add semantic shared-foundation evidence to M4 "
            "collaboration recommendations without altering their "
            "original collaboration score."
        )
    )

    parser.add_argument(
        "--recommendations-input",
        type=Path,
        default=DEFAULT_RECOMMENDATIONS_INPUT,
        help="Path to enriched M4 recommendations JSONL.",
    )

    parser.add_argument(
        "--staff-profiles-input",
        type=Path,
        default=DEFAULT_STAFF_PROFILES_INPUT,
        help="Path to filtered staff topic profiles JSONL.",
    )

    parser.add_argument(
        "--topic-vectors-input",
        type=Path,
        default=DEFAULT_TOPIC_VECTORS_INPUT,
        help="Path to canonical topic vectors JSONL.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for enriched semantic-foundation JSONL.",
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Path for semantic-foundation summary JSON.",
    )

    parser.add_argument(
        "--missing-vectors-output",
        type=Path,
        default=DEFAULT_FAILURES_OUTPUT,
        help="Path for missing-vector audit JSONL.",
    )

    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.80,
        help=(
            "Minimum cosine similarity for semantic shared "
            "foundation evidence. Default: 0.80."
        ),
    )

    parser.add_argument(
        "--maximum-topics-per-staff",
        type=int,
        default=10,
        help=(
            "Maximum staff topics considered for semantic matching. "
            "Default: 10."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    required_paths = [
        args.recommendations_input,
        args.staff_profiles_input,
        args.topic_vectors_input,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input file was not found: {path}"
            )

    if not 0.0 < args.similarity_threshold <= 1.0:
        raise ValueError(
            "--similarity-threshold must be greater than 0 "
            "and less than or equal to 1."
        )

    if args.maximum_topics_per_staff < 1:
        raise ValueError(
            "--maximum-topics-per-staff must be at least 1."
        )

    topic_vectors, vector_statistics = load_topic_vectors(
    args.topic_vectors_input
)

    staff_topics, staff_statistics = load_staff_topics(
        args.staff_profiles_input
    )

    recommendations = list(
        read_jsonl(args.recommendations_input)
    )

    enriched_records: list[dict[str, Any]] = []
    missing_vector_records: list[dict[str, Any]] = []

    foundation_counts: Counter[str] = Counter()
    exact_foundation_count = 0
    semantic_foundation_count = 0
    no_foundation_count = 0

    for recommendation in recommendations:
        enriched, missing_vectors = enrich_foundation_record(
            recommendation=recommendation,
            staff_topics=staff_topics,
            topic_vectors=topic_vectors,
            threshold=args.similarity_threshold,
            maximum_topics_per_staff=(
                args.maximum_topics_per_staff
            ),
        )

        enriched_records.append(enriched)

        for missing_vector in missing_vectors:
            missing_vector_records.append(
                {
                    "module_id": as_text(
                        recommendation.get("module_id")
                    ),
                    "module_title": as_text(
                        recommendation.get("module_title")
                    ),
                    "staff_a_id": as_text(
                        recommendation.get("staff_a_id")
                    ),
                    "staff_b_id": as_text(
                        recommendation.get("staff_b_id")
                    ),
                    **missing_vector,
                }
            )

        foundation_type = as_text(
            enriched.get("foundation_type")
        )

        foundation_counts[foundation_type] += 1

        if foundation_type == "exact":
            exact_foundation_count += 1
        elif foundation_type == "semantic":
            semantic_foundation_count += 1
        else:
            no_foundation_count += 1

    enriched_records.sort(
        key=lambda record: (
            as_text(record.get("module_title")).lower(),
            int(record.get("rank", 999999)),
            as_text(record.get("staff_a_name")).lower(),
            as_text(record.get("staff_b_name")).lower(),
        )
    )

    write_jsonl(args.output, enriched_records)
    write_jsonl(
        args.missing_vectors_output,
        missing_vector_records,
    )

    top_recommendations = first_recommendation_per_module(
        enriched_records
    )

    exact_modules = {
        module_id
        for module_id, record in top_recommendations.items()
        if as_text(record.get("foundation_type")) == "exact"
    }

    semantic_modules = {
        module_id
        for module_id, record in top_recommendations.items()
        if as_text(record.get("foundation_type"))
        in {"exact", "semantic"}
    }

    no_foundation_modules = {
        module_id
        for module_id, record in top_recommendations.items()
        if as_text(record.get("foundation_type")) == "none"
    }

    total_modules = len(top_recommendations)
    total_recommendations = len(enriched_records)

    summary = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "inputs": {
            "recommendations": str(
                args.recommendations_input
            ),
            "staff_topic_profiles": str(
                args.staff_profiles_input
            ),
            "topic_vectors": str(args.topic_vectors_input),
        },
        "parameters": {
            "semantic_similarity_threshold": (
                args.similarity_threshold
            ),
            "maximum_topics_per_staff": (
                args.maximum_topics_per_staff
            ),
            "embedding_source": (
                "topic_vectors.jsonl generated by "
                "topic_normalisation.py"
            ),
        },
        "outputs": {
            "semantic_foundation_recommendations": str(
                args.output
            ),
            "missing_vector_audit": str(
                args.missing_vectors_output
            ),
        },
        "topic_vector_loading": vector_statistics,
        "staff_topic_loading": staff_statistics,
        "recommendation_foundation_counts": {
            "recommendations_processed": total_recommendations,
            "by_foundation_type": dict(
                sorted(foundation_counts.items())
            ),
            "exact_foundation_recommendations": (
                exact_foundation_count
            ),
            "semantic_only_foundation_recommendations": (
                semantic_foundation_count
            ),
            "no_foundation_recommendations": (
                no_foundation_count
            ),
        },
        "top_ranked_module_comparison": {
            "modules_with_top_ranked_recommendation": total_modules,
            "exact_only_foundation_modules": len(exact_modules),
            "exact_only_foundation_rate": round(
                len(exact_modules) / total_modules,
                6,
            )
            if total_modules
            else 0.0,
            "exact_plus_semantic_foundation_modules": len(
                semantic_modules
            ),
            "exact_plus_semantic_foundation_rate": round(
                len(semantic_modules) / total_modules,
                6,
            )
            if total_modules
            else 0.0,
            "modules_without_foundation": len(
                no_foundation_modules
            ),
            "modules_without_foundation_rate": round(
                len(no_foundation_modules) / total_modules,
                6,
            )
            if total_modules
            else 0.0,
            "additional_modules_supported_by_semantic_evidence": (
                len(semantic_modules) - len(exact_modules)
            ),
        },
        "missing_vector_audit": {
            "missing_vector_records_written": len(
                missing_vector_records
            ),
        },
        "interpretation": {
            "exact_foundation": (
                "The staff pair shares at least one identical "
                "canonical topic."
            ),
            "semantic_foundation": (
                "The staff pair does not share an identical "
                "canonical topic, but has at least one topic pair "
                "whose cosine similarity meets the specified "
                "threshold."
            ),
            "none": (
                "No exact or semantic shared-foundation evidence "
                "was found at the selected threshold."
            ),
            "important_note": (
                "This sensitivity analysis does not alter the "
                "original collaboration score, rank, coverage, or "
                "complementarity gain. It only adds explainable "
                "shared-foundation evidence."
            ),
        },
    }

    args.summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.summary_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()