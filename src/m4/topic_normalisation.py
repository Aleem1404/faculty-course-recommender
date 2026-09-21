from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import spacy
except ImportError:
    spacy = None


LOGGER = logging.getLogger("m4.topic_normalisation")


DEFAULT_GENERIC_TOPICS = {
    "research",
    "teaching",
    "education",
    "university",
    "academic",
    "module",
    "course",
    "student",
    "students",
    "learning",
    "lecture",
    "lectures",
    "seminar",
    "seminars",
    "topic",
    "topics",
    "knowledge",
    "skills",
    "study",
    "studies",
    "introduction",
    "advanced topics",
    "fundamentals",
    "principles",
    "applications",
    "methods",
    "methodologies",
    "systems",
    "technology",
    "technologies",
}

DEFAULT_STOPWORDS = {
    "and",
    "or",
    "of",
    "for",
    "in",
    "on",
    "to",
    "with",
    "the",
    "a",
    "an",
    "by",
    "from",
    "at",
    "into",
    "about",
    "using",
    "use",
    "based",
}

DEFAULT_FIELD_CANDIDATES = [
    "topics",
    "topic_keywords",
    "keywords",
    "extracted_topics",
    "staff_topics",
    "expertise_topics",
    "research_interests_topics",
]

DEFAULT_TEXT_FIELD_CANDIDATES = [
    "description",
    "summary",
    "overview",
    "content",
    "module_description",
    "learning_outcomes",
    "research_interests",
    "expertise",
    "specialisms",
    "profile_text",
    "biography",
    "teaching_interests",
]

DEFAULT_RECORD_ID_FIELDS = [
    "module_code",
    "staff_id",
    "id",
    "code",
    "slug",
    "url",
    "name",
    "title",
]


@dataclass
class TopicCandidate:
    source_record_id: str
    source_type: str
    original_text: str
    cleaned_text: str
    normalised_text: str
    source_field: str
    source_weight: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalTopic:
    topic_id: str
    canonical_label: str
    member_topics: List[str]
    member_count: int
    source_types: List[str]
    source_count: int
    centroid_vector: Optional[List[float]] = None


@dataclass
class TopicAssignment:
    source_record_id: str
    source_type: str
    source_field: str
    original_text: str
    normalised_text: str
    canonical_topic_id: str
    canonical_label: str
    source_weight: float
    similarity_to_canonical: Optional[float] = None


class TopicNormaliser:
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.84,
        min_topic_words: int = 1,
        max_topic_words: int = 8,
        generic_topics: Optional[set[str]] = None,
        stopwords: Optional[set[str]] = None,
        use_lemma: bool = True,
        spacy_model: str = "en_core_web_sm",
        batch_size: int = 128,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.similarity_threshold = similarity_threshold
        self.min_topic_words = min_topic_words
        self.max_topic_words = max_topic_words
        self.generic_topics = generic_topics or set(DEFAULT_GENERIC_TOPICS)
        self.stopwords = stopwords or set(DEFAULT_STOPWORDS)
        self.use_lemma = use_lemma
        self.spacy_model = spacy_model
        self.batch_size = batch_size

        self._embedding_model = None
        self._nlp = None

    def load_nlp(self):
        if not self.use_lemma:
            return None
        if self._nlp is not None:
            return self._nlp
        if spacy is None:
            LOGGER.warning("spaCy not installed. Falling back to token cleaning without lemmatisation.")
            self.use_lemma = False
            return None
        try:
            self._nlp = spacy.load(self.spacy_model, disable=["ner", "parser"])
        except Exception as exc:
            LOGGER.warning("Could not load spaCy model '%s': %s. Falling back without lemmatisation.", self.spacy_model, exc)
            self.use_lemma = False
            self._nlp = None
        return self._nlp

    def load_embedding_model(self):
        if self._embedding_model is not None:
            return self._embedding_model
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for topic embedding. "
                "Install it with: pip install sentence-transformers"
            )
        self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    @staticmethod
    def load_json_records(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        if path.suffix.lower() == ".jsonl":
            records = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
            return records

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("records", "items", "data", "modules", "staff", "results"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]

        raise ValueError(f"Unsupported JSON structure in {path}")

    @staticmethod
    def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def write_json(path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denom == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / denom)

    @staticmethod
    def stable_topic_id(label: str) -> str:
        digest = hashlib.md5(label.encode("utf-8")).hexdigest()[:12]
        safe = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        safe = safe[:40] if safe else "topic"
        return f"topic_{safe}_{digest}"

    @staticmethod
    def find_record_id(record: Dict[str, Any]) -> str:
        for key in DEFAULT_RECORD_ID_FIELDS:
            if key in record and record[key] not in (None, ""):
                return str(record[key])
        return hashlib.md5(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def flatten_topic_values(value: Any) -> List[Tuple[str, float]]:
        items: List[Tuple[str, float]] = []

        if value is None:
            return items

        if isinstance(value, str):
            text = value.strip()
            if text:
                items.append((text, 1.0))
            return items

        if isinstance(value, (int, float)):
            return items

        if isinstance(value, list):
            for v in value:
                items.extend(TopicNormaliser.flatten_topic_values(v))
            return items

        if isinstance(value, dict):
            if "topic" in value:
                topic_text = str(value["topic"]).strip()
                weight = float(value.get("weight", 1.0))
                if topic_text:
                    items.append((topic_text, weight))
                return items
            if "label" in value:
                topic_text = str(value["label"]).strip()
                weight = float(value.get("weight", 1.0))
                if topic_text:
                    items.append((topic_text, weight))
                return items
            for k, v in value.items():
                if isinstance(v, (int, float)):
                    items.append((str(k).strip(), float(v)))
                elif isinstance(v, str) and v.strip():
                    items.append((v.strip(), 1.0))
            return items

        return items

    def normalise_whitespace(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def basic_clean(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = text.lower()
        text = text.replace("&", " and ")
        text = re.sub(r"[/|]", " ", text)
        text = re.sub(r"[_\-]+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = self.normalise_whitespace(text)
        return text

    def tokenise_and_normalise(self, text: str) -> str:
        cleaned = self.basic_clean(text)
        if not cleaned:
            return ""

        if self.use_lemma:
            nlp = self.load_nlp()
        else:
            nlp = None

        if nlp is not None:
            doc = nlp(cleaned)
            tokens = []
            for token in doc:
                t = token.lemma_.strip().lower()
                if not t or t == "-pron-":
                    continue
                if t in self.stopwords:
                    continue
                if token.is_space or token.is_punct:
                    continue
                if re.fullmatch(r"\d+", t):
                    continue
                tokens.append(t)
        else:
            tokens = []
            for token in cleaned.split():
                t = token.strip().lower()
                if not t or t in self.stopwords:
                    continue
                if re.fullmatch(r"\d+", t):
                    continue
                tokens.append(t)

        text_out = " ".join(tokens)
        text_out = self.normalise_whitespace(text_out)
        return text_out

    def is_valid_topic(self, topic: str) -> bool:
        if not topic:
            return False

        topic = self.normalise_whitespace(topic)
        if topic in self.generic_topics:
            return False

        words = topic.split()
        if len(words) < self.min_topic_words:
            return False
        if len(words) > self.max_topic_words:
            return False

        if len(topic) < 3:
            return False

        if all(w in self.stopwords for w in words):
            return False

        if re.fullmatch(r"[a-z]$", topic):
            return False

        return True

    def extract_candidates_from_record(
        self,
        record: Dict[str, Any],
        source_type: str,
        topic_fields: Sequence[str],
    ) -> List[TopicCandidate]:
        record_id = self.find_record_id(record)
        candidates: List[TopicCandidate] = []

        for field_name in topic_fields:
            if field_name not in record:
                continue

            for text, weight in self.flatten_topic_values(record[field_name]):
                cleaned = self.basic_clean(text)
                normalised = self.tokenise_and_normalise(text)
                if not self.is_valid_topic(normalised):
                    continue

                candidates.append(
                    TopicCandidate(
                        source_record_id=record_id,
                        source_type=source_type,
                        original_text=text,
                        cleaned_text=cleaned,
                        normalised_text=normalised,
                        source_field=field_name,
                        source_weight=weight,
                    )
                )

        return candidates

    def extract_candidates(
        self,
        records: Sequence[Dict[str, Any]],
        source_type: str,
        topic_fields: Optional[Sequence[str]] = None,
    ) -> List[TopicCandidate]:
        topic_fields = topic_fields or DEFAULT_FIELD_CANDIDATES
        all_candidates: List[TopicCandidate] = []

        for record in records:
            all_candidates.extend(
                self.extract_candidates_from_record(
                    record=record,
                    source_type=source_type,
                    topic_fields=topic_fields,
                )
            )

        return all_candidates

    def embed_topics(self, topics: Sequence[str]) -> np.ndarray:
        model = self.load_embedding_model()
        embeddings = model.encode(
            list(topics),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def cluster_topics(
        self,
        unique_topics: Sequence[str],
        embeddings: np.ndarray,
    ) -> Tuple[List[CanonicalTopic], Dict[str, str], Dict[str, float]]:
        canonical_topics: List[CanonicalTopic] = []
        topic_to_canonical_id: Dict[str, str] = {}
        topic_to_similarity: Dict[str, float] = {}

        centroids: List[np.ndarray] = []
        clusters: List[List[str]] = []
        cluster_source_types: List[set[str]] = []
        cluster_counts: List[int] = []

        for idx, topic in enumerate(unique_topics):
            vec = embeddings[idx]

            if not centroids:
                cluster_id = self.stable_topic_id(topic)
                centroids.append(vec.copy())
                clusters.append([topic])
                cluster_source_types.append(set())
                cluster_counts.append(1)
                topic_to_canonical_id[topic] = cluster_id
                topic_to_similarity[topic] = 1.0
                continue

            best_cluster_idx = -1
            best_score = -1.0

            for c_idx, centroid in enumerate(centroids):
                score = self.cosine_similarity(vec, centroid)
                if score > best_score:
                    best_score = score
                    best_cluster_idx = c_idx

            if best_score >= self.similarity_threshold:
                clusters[best_cluster_idx].append(topic)
                cluster_counts[best_cluster_idx] += 1
                centroids[best_cluster_idx] = (
                    (centroids[best_cluster_idx] * (cluster_counts[best_cluster_idx] - 1) + vec)
                    / cluster_counts[best_cluster_idx]
                )
                canonical_label = self.choose_canonical_label(clusters[best_cluster_idx])
                cluster_id = self.stable_topic_id(canonical_label)
                topic_to_canonical_id[topic] = cluster_id
                topic_to_similarity[topic] = best_score
            else:
                cluster_id = self.stable_topic_id(topic)
                centroids.append(vec.copy())
                clusters.append([topic])
                cluster_source_types.append(set())
                cluster_counts.append(1)
                topic_to_canonical_id[topic] = cluster_id
                topic_to_similarity[topic] = 1.0

        for c_idx, members in enumerate(clusters):
            canonical_label = self.choose_canonical_label(members)
            cluster_id = self.stable_topic_id(canonical_label)
            centroid = centroids[c_idx]
            canonical_topics.append(
                CanonicalTopic(
                    topic_id=cluster_id,
                    canonical_label=canonical_label,
                    member_topics=sorted(members),
                    member_count=len(members),
                    source_types=[],
                    source_count=0,
                    centroid_vector=centroid.tolist(),
                )
            )

            for member in members:
                topic_to_canonical_id[member] = cluster_id

        canonical_topics.sort(key=lambda x: (-x.member_count, x.canonical_label))
        return canonical_topics, topic_to_canonical_id, topic_to_similarity

    @staticmethod
    def choose_canonical_label(members: Sequence[str]) -> str:
        members = list(members)
        members.sort(key=lambda x: (len(x.split()), len(x), x))
        return members[0]

    def build_outputs(
        self,
        candidates: Sequence[TopicCandidate],
    ) -> Tuple[List[CanonicalTopic], List[TopicAssignment], Dict[str, np.ndarray]]:
        unique_topics = sorted({c.normalised_text for c in candidates})
        LOGGER.info("Unique normalised topics: %d", len(unique_topics))

        embeddings = self.embed_topics(unique_topics)
        topic_vectors = {topic: embeddings[i] for i, topic in enumerate(unique_topics)}

        canonical_topics, topic_to_canonical_id, topic_to_similarity = self.cluster_topics(unique_topics, embeddings)

        canonical_by_id = {c.topic_id: c for c in canonical_topics}

        source_types_by_canonical: Dict[str, set[str]] = {}
        assignments: List[TopicAssignment] = []

        for c in candidates:
            canonical_id = topic_to_canonical_id[c.normalised_text]
            canonical = canonical_by_id[canonical_id]
            source_types_by_canonical.setdefault(canonical_id, set()).add(c.source_type)

            assignments.append(
                TopicAssignment(
                    source_record_id=c.source_record_id,
                    source_type=c.source_type,
                    source_field=c.source_field,
                    original_text=c.original_text,
                    normalised_text=c.normalised_text,
                    canonical_topic_id=canonical_id,
                    canonical_label=canonical.canonical_label,
                    source_weight=c.source_weight,
                    similarity_to_canonical=topic_to_similarity.get(c.normalised_text, None),
                )
            )

        assignment_counts: Dict[str, int] = {}
        for assignment in assignments:
            assignment_counts[assignment.canonical_topic_id] = assignment_counts.get(assignment.canonical_topic_id, 0) + 1

        for c in canonical_topics:
            c.source_types = sorted(source_types_by_canonical.get(c.topic_id, set()))
            c.source_count = assignment_counts.get(c.topic_id, 0)

        return canonical_topics, assignments, topic_vectors

    def run(
        self,
        module_records: Sequence[Dict[str, Any]],
        staff_records: Sequence[Dict[str, Any]],
        module_topic_fields: Optional[Sequence[str]] = None,
        staff_topic_fields: Optional[Sequence[str]] = None,
    ) -> Tuple[List[CanonicalTopic], List[TopicAssignment], Dict[str, np.ndarray]]:
        module_candidates = self.extract_candidates(
            records=module_records,
            source_type="module",
            topic_fields=module_topic_fields or DEFAULT_FIELD_CANDIDATES,
        )
        staff_candidates = self.extract_candidates(
            records=staff_records,
            source_type="staff",
            topic_fields=staff_topic_fields or DEFAULT_FIELD_CANDIDATES,
        )

        LOGGER.info("Module topic candidates kept: %d", len(module_candidates))
        LOGGER.info("Staff topic candidates kept: %d", len(staff_candidates))

        all_candidates = module_candidates + staff_candidates
        canonical_topics, assignments, vectors = self.build_outputs(all_candidates)
        return canonical_topics, assignments, vectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalise module and staff topics for M4 collaboration.")

    parser.add_argument("--modules", type=str, required=True, help="Path to module JSON or JSONL file.")
    parser.add_argument("--staff", type=str, required=True, help="Path to staff JSON or JSONL file.")
    parser.add_argument(
        "--module-topic-fields",
        type=str,
        default=",".join(DEFAULT_FIELD_CANDIDATES),
        help="Comma-separated topic fields for module records.",
    )
    parser.add_argument(
        "--staff-topic-fields",
        type=str,
        default=",".join(DEFAULT_FIELD_CANDIDATES),
        help="Comma-separated topic fields for staff records.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/outputs/m4",
        help="Output directory.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.84,
        help="Cosine similarity threshold for topic clustering.",
    )
    parser.add_argument(
        "--min-topic-words",
        type=int,
        default=1,
        help="Minimum number of words in a valid topic.",
    )
    parser.add_argument(
        "--max-topic-words",
        type=int,
        default=8,
        help="Maximum number of words in a valid topic.",
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model for lemmatisation.",
    )
    parser.add_argument(
        "--disable-lemma",
        action="store_true",
        help="Disable lemmatisation even if spaCy is installed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    module_path = Path(args.modules)
    staff_path = Path(args.staff)
    out_dir = Path(args.out_dir)

    module_records = TopicNormaliser.load_json_records(module_path)
    staff_records = TopicNormaliser.load_json_records(staff_path)

    module_topic_fields = [x.strip() for x in args.module_topic_fields.split(",") if x.strip()]
    staff_topic_fields = [x.strip() for x in args.staff_topic_fields.split(",") if x.strip()]

    normaliser = TopicNormaliser(
        embedding_model_name=args.embedding_model,
        similarity_threshold=args.similarity_threshold,
        min_topic_words=args.min_topic_words,
        max_topic_words=args.max_topic_words,
        use_lemma=not args.disable_lemma,
        spacy_model=args.spacy_model,
        batch_size=args.batch_size,
    )

    canonical_topics, assignments, topic_vectors = normaliser.run(
        module_records=module_records,
        staff_records=staff_records,
        module_topic_fields=module_topic_fields,
        staff_topic_fields=staff_topic_fields,
    )

    canonical_path = out_dir / "canonical_topics.jsonl"
    assignments_path = out_dir / "topic_assignments.jsonl"
    summary_path = out_dir / "topic_normalisation_summary.json"
    vectors_path = out_dir / "topic_vectors.jsonl"

    TopicNormaliser.write_jsonl(
        canonical_path,
        (asdict(item) for item in canonical_topics),
    )
    TopicNormaliser.write_jsonl(
        assignments_path,
        (asdict(item) for item in assignments),
    )
    TopicNormaliser.write_jsonl(
        vectors_path,
        (
            {
                "normalised_text": topic,
                "vector": vec.tolist(),
            }
            for topic, vec in topic_vectors.items()
        ),
    )

    summary = {
        "module_input_records": len(module_records),
        "staff_input_records": len(staff_records),
        "canonical_topic_count": len(canonical_topics),
        "assignment_count": len(assignments),
        "embedding_model": args.embedding_model,
        "similarity_threshold": args.similarity_threshold,
        "module_topic_fields": module_topic_fields,
        "staff_topic_fields": staff_topic_fields,
        "outputs": {
            "canonical_topics": str(canonical_path),
            "topic_assignments": str(assignments_path),
            "topic_vectors": str(vectors_path),
        },
    }
    TopicNormaliser.write_json(summary_path, summary)

    LOGGER.info("Wrote canonical topics to %s", canonical_path)
    LOGGER.info("Wrote topic assignments to %s", assignments_path)
    LOGGER.info("Wrote topic vectors to %s", vectors_path)
    LOGGER.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()