from __future__ import annotations

import re
from collections import Counter
from typing import Any


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
    "within",
    "using",
    "used",
    "use",
    "based",
    "study",
    "studies",
    "student",
    "students",
    "module",
    "course",
    "courses",
    "teaching",
    "learning",
    "introduction",
    "advanced",
    "principles",
    "practice",
    "theory",
    "method",
    "methods",
    "approach",
    "approaches",
    "analysis",
    "project",
    "projects",
    "research",
    "dissertation",
    "capstone",
    "placement",
    "component",
    "selected",
    "performance",
}

NOISE_TERMS = {
    "about",
    "across",
    "academic",
    "academy",
    "admission",
    "aim",
    "aims",
    "also",
    "allows",
    "area",
    "article",
    "articles",
    "assistant",
    "associate",
    "award",
    "awards",
    "based",
    "best",
    "brings",
    "brunel",
    "businesses",
    "case",
    "cases",
    "certificate",
    "college",
    "communication",
    "communications",
    "conference",
    "consider",
    "coordinator",
    "course",
    "courses",
    "data",
    "dean",
    "demonstrate",
    "department",
    "develop",
    "different",
    "diploma",
    "during",
    "education",
    "enable",
    "enquiry",
    "evidence",
    "examined",
    "expertise",
    "explore",
    "faculty",
    "field",
    "fellow",
    "findings",
    "framework",
    "general",
    "gained",
    "grant",
    "group",
    "healthcare",
    "here",
    "higher",
    "ict",
    "identification",
    "implementing",
    "including",
    "independent",
    "information",
    "introduction",
    "journal",
    "knowledge",
    "lecturer",
    "learner",
    "learning",
    "london",
    "main",
    "member",
    "metadata",
    "methodologies",
    "module",
    "nothing",
    "number",
    "paper",
    "papers",
    "piece",
    "position",
    "poster",
    "postgraduate",
    "practice",
    "presentation",
    "proceedings",
    "professor",
    "programme",
    "provide",
    "provides",
    "quality",
    "reader",
    "related",
    "report",
    "reports",
    "results",
    "review",
    "reviews",
    "school",
    "science",
    "sciences",
    "senior",
    "session",
    "sessions",
    "singapore",
    "skills",
    "solid",
    "specific",
    "staff",
    "strategy",
    "student",
    "students",
    "study",
    "such",
    "supervision",
    "support",
    "team",
    "term",
    "terms",
    "these",
    "through",
    "together",
    "under",
    "understanding",
    "university",
    "vice",
    "volume",
    "week",
    "weeks",
    "will",
    "winter",
    "work",
    "world",
    "year",
    "years",
    "you",
    "your",
}

GENERIC_TOPIC_TERMS = {
    "computer",
    "computer science",
    "education",
    "engineering",
    "health",
    "innovation",
    "management",
    "research",
    "science",
    "social",
    "social sciences",
    "system",
    "systems",
    "technology",
    "technologies",
}

ALLOWED_SINGLE_WORD_TOPICS = {
    "algorithms",
    "anthropology",
    "architecture",
    "automation",
    "biology",
    "brain",
    "citizenship",
    "cloud",
    "cognition",
    "cybersecurity",
    "economics",
    "energy",
    "ethics",
    "finance",
    "genetics",
    "geospatial",
    "graphics",
    "haptic",
    "identity",
    "linguistics",
    "mathematics",
    "migration",
    "molecular",
    "neuroscience",
    "pandemic",
    "pharmacology",
    "policy",
    "psychology",
    "robotics",
    "simulation",
    "software",
    "sustainability",
    "tourism",
}

MIN_TOPIC_TOKEN_LENGTH = 3
MIN_TOPIC_CHARACTER_LENGTH = 5
MAX_NGRAM = 3
MIN_MODULE_TOPIC_SCORE = 2
MIN_STAFF_TOPIC_SCORE = 2
MIN_TRIGRAM_SCORE = 3


def normalise_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).casefold()
    text = text.replace("&", " and ")
    text = text.replace("covid-19", "covid19")
    text = re.sub(r"[/|_]", " ", text)
    text = re.sub(r"[^a-z0-9\-\+\.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_sentences(text: str) -> list[str]:
    if not text:
        return []

    parts = re.split(r"[.;:\n\r]+", text)

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def is_year_or_code(token: str) -> bool:
    if re.fullmatch(r"(19|20)\d{2}", token):
        return True

    if re.fullmatch(r"\d+", token):
        return True

    if re.fullmatch(
        r"[a-z]{1,6}\d{1,8}[a-z0-9\-]*",
        token,
    ):
        return True

    return False


def is_noise_token(token: str) -> bool:
    return (
        token in STOPWORDS
        or token in NOISE_TERMS
        or is_year_or_code(token)
    )


def tokenise(text: str) -> list[str]:
    if not text:
        return []

    tokens = re.findall(
        r"[a-z0-9][a-z0-9\-\+\.]*",
        text,
    )

    cleaned: list[str] = []

    for token in tokens:
        token = token.strip("-+.")

        if not token:
            continue

        if len(token) < MIN_TOPIC_TOKEN_LENGTH:
            continue

        if is_noise_token(token):
            continue

        cleaned.append(token)

    return cleaned


def build_name_terms(full_name: Any) -> set[str]:
    name = normalise_text(full_name)

    if not name:
        return set()

    return {
        token
        for token in tokenise(name)
        if token not in {"prof", "dr"}
    }


def contains_blocked_token(
    tokens: list[str],
    blocked_terms: set[str],
) -> bool:
    return any(
        token in blocked_terms
        for token in tokens
    )


def phrase_has_substantive_signal(
    tokens: list[str],
) -> bool:
    if not tokens:
        return False

    if len(tokens) == 1:
        return tokens[0] in ALLOWED_SINGLE_WORD_TOPICS

    return all(
        token not in NOISE_TERMS
        and token not in STOPWORDS
        for token in tokens
    )


def clean_topic_phrase(
    text: str,
    blocked_terms: set[str] | None = None,
) -> str:
    if blocked_terms is None:
        blocked_terms = set()

    text = normalise_text(text)

    if not text:
        return ""

    tokens = tokenise(text)

    if not tokens:
        return ""

    if contains_blocked_token(
        tokens,
        blocked_terms,
    ):
        return ""

    if not phrase_has_substantive_signal(tokens):
        return ""

    phrase = " ".join(tokens).strip()

    if len(phrase) < MIN_TOPIC_CHARACTER_LENGTH:
        return ""

    if phrase in GENERIC_TOPIC_TERMS:
        return ""

    return phrase


def collect_text_values(
    values: list[Any],
) -> list[str]:
    texts: list[str] = []

    for value in values:
        if value is None:
            continue

        if isinstance(value, list):
            for item in value:
                if item is None:
                    continue

                text = str(item).strip()

                if text:
                    texts.append(text)
        else:
            text = str(value).strip()

            if text:
                texts.append(text)

    return texts


def generate_ngrams(
    tokens: list[str],
    max_n: int = MAX_NGRAM,
) -> list[str]:
    phrases: list[str] = []

    for n in range(1, max_n + 1):
        for start in range(
            len(tokens) - n + 1
        ):
            gram_tokens = tokens[
                start:start + n
            ]

            if not gram_tokens:
                continue

            phrase = " ".join(gram_tokens).strip()

            if not phrase:
                continue

            phrases.append(phrase)

    return phrases


def extract_topic_candidates_from_text(
    text: str,
    blocked_terms: set[str] | None = None,
    max_ngram: int = MAX_NGRAM,
) -> Counter[str]:
    counter: Counter[str] = Counter()

    if blocked_terms is None:
        blocked_terms = set()

    normalised = normalise_text(text)

    if not normalised:
        return counter

    for sentence in split_sentences(normalised):
        tokens = tokenise(sentence)

        if not tokens:
            continue

        for phrase in generate_ngrams(
            tokens,
            max_n=max_ngram,
        ):
            cleaned = clean_topic_phrase(
                phrase,
                blocked_terms=blocked_terms,
            )

            if cleaned:
                counter[cleaned] += 1

    return counter


def is_valid_topic(
    topic: str,
    score: int,
    blocked_terms: set[str],
    min_score: int,
) -> bool:
    if not topic:
        return False

    tokens = topic.split()

    if not tokens:
        return False

    if topic in GENERIC_TOPIC_TERMS:
        return False

    if contains_blocked_token(
        tokens,
        blocked_terms,
    ):
        return False

    if any(is_noise_token(token) for token in tokens):
        return False

    if not phrase_has_substantive_signal(tokens):
        return False

    if len(tokens) >= 3 and score < MIN_TRIGRAM_SCORE:
        return False

    if score < min_score:
        return False

    return True


def extract_ranked_topics(
    raw_texts: list[str],
    explicit_topics: list[str] | None = None,
    blocked_terms: set[str] | None = None,
    max_topics: int = 10,
    min_score: int = 2,
) -> list[dict[str, Any]]:
    topic_counter: Counter[str] = Counter()

    if blocked_terms is None:
        blocked_terms = set()

    for text in raw_texts:
        topic_counter.update(
            extract_topic_candidates_from_text(
                text,
                blocked_terms=blocked_terms,
            )
        )

    if explicit_topics:
        for topic in explicit_topics:
            cleaned = clean_topic_phrase(
                topic,
                blocked_terms=blocked_terms,
            )

            if cleaned:
                topic_counter[cleaned] += 12

    ranked = sorted(
        topic_counter.items(),
        key=lambda item: (
            -item[1],
            -len(item[0].split()),
            item[0],
        ),
    )

    results: list[dict[str, Any]] = []

    for topic, score in ranked:
        if not is_valid_topic(
            topic=topic,
            score=score,
            blocked_terms=blocked_terms,
            min_score=min_score,
        ):
            continue

        results.append({
            "topic": topic,
            "score": int(score),
        })

        if len(results) >= max_topics:
            break

    return results


def extract_module_topics(
    module_record: dict[str, Any],
    max_topics: int = 8,
) -> dict[str, Any]:
    raw_texts = collect_text_values([
        module_record.get("module_title"),
        module_record.get("module_name"),
        module_record.get("description"),
        module_record.get("module_description"),
        module_record.get("overview"),
        module_record.get("summary"),
        module_record.get("aims"),
        module_record.get("learning_outcomes"),
        module_record.get("content"),
        module_record.get("syllabus"),
        module_record.get("matching_text"),
    ])

    explicit_topics = collect_text_values([
        module_record.get("topics"),
        module_record.get("keywords"),
    ])

    ranked_topics = extract_ranked_topics(
        raw_texts=raw_texts,
        explicit_topics=explicit_topics,
        blocked_terms=set(),
        max_topics=max_topics,
        min_score=MIN_MODULE_TOPIC_SCORE,
    )

    return {
        "module_id": module_record.get("module_id"),
        "module_code": module_record.get("module_code"),
        "module_title": module_record.get("module_title"),
        "departments": module_record.get(
            "departments",
            [],
        ),
        "colleges": module_record.get(
            "colleges",
            [],
        ),
        "topics": ranked_topics,
        "topic_names": [
            item["topic"]
            for item in ranked_topics
        ],
    }


def extract_staff_topics(
    staff_record: dict[str, Any],
    max_topics: int = 10,
) -> dict[str, Any]:
    blocked_terms = build_name_terms(
        staff_record.get("full_name")
    )

    raw_texts = collect_text_values([
        staff_record.get("summary"),
        staff_record.get("profile_summary"),
        staff_record.get("core_profile_text"),
        staff_record.get(
            "publication_expertise_text"
        ),
        staff_record.get("research_interests"),
        staff_record.get("specialisms"),
        staff_record.get("teaching_interests"),
        staff_record.get("publication_titles"),
        staff_record.get("publication_abstracts"),
    ])

    explicit_topics = collect_text_values([
        staff_record.get("research_interest_names"),
        staff_record.get("publication_topic_names"),
        staff_record.get(
            "publication_keyword_names"
        ),
        staff_record.get("keywords"),
        staff_record.get("topics"),
    ])

    ranked_topics = extract_ranked_topics(
        raw_texts=raw_texts,
        explicit_topics=explicit_topics,
        blocked_terms=blocked_terms,
        max_topics=max_topics,
        min_score=MIN_STAFF_TOPIC_SCORE,
    )

    return {
        "staff_id": staff_record.get("staff_id"),
        "full_name": staff_record.get("full_name"),
        "position": staff_record.get("position"),
        "department_name": staff_record.get(
            "department_name"
        ),
        "college_name": staff_record.get(
            "college_name"
        ),
        "profile_url": staff_record.get(
            "profile_url"
        ),
        "topics": ranked_topics,
        "topic_names": [
            item["topic"]
            for item in ranked_topics
        ],
    }


def shared_topics(
    module_topics: dict[str, Any],
    staff_topics: dict[str, Any],
) -> list[str]:
    module_topic_names = set(
        module_topics.get("topic_names", [])
    )
    staff_topic_names = set(
        staff_topics.get("topic_names", [])
    )

    return sorted(
        module_topic_names.intersection(
            staff_topic_names
        )
    )


def strongest_shared_topics(
    module_topics: dict[str, Any],
    staff_topics: dict[str, Any],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    module_scores = {
        item["topic"]: item["score"]
        for item in module_topics.get(
            "topics",
            []
        )
    }

    staff_scores = {
        item["topic"]: item["score"]
        for item in staff_topics.get(
            "topics",
            []
        )
    }

    shared: list[dict[str, Any]] = []

    for topic in sorted(
        set(module_scores).intersection(
            set(staff_scores)
        )
    ):
        shared.append({
            "topic": topic,
            "module_score": module_scores[topic],
            "staff_score": staff_scores[topic],
            "combined_score": (
                module_scores[topic]
                + staff_scores[topic]
            ),
        })

    shared.sort(
        key=lambda item: (
            -item["combined_score"],
            item["topic"],
        )
    )

    return shared[:top_n]