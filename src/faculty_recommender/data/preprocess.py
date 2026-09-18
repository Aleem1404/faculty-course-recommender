from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from faculty_recommender.data.audit import (
    direct_value,
    get_evidence_pages,
    get_publications,
    is_blank,
    load_json_records,
    normalise_text,
    profile_value,
)


MODULE_CODE_PATTERN = re.compile(
    r"\b[A-Z]{2,5}\s?-?\d{3,5}\b",
    re.IGNORECASE,
)

DOI_PATTERN = re.compile(
    r"(10\.\d{4,9}/[^\s\"<>]+)",
    re.IGNORECASE,
)


def stable_hash(value: str, length: int = 20) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:length]


def unique_text_parts(values: list[Any]) -> list[str]:
    results = []
    seen = set()

    for value in values:
        text = normalise_text(value)

        if not text:
            continue

        comparable = text.casefold()

        if comparable in seen:
            continue

        seen.add(comparable)
        results.append(text)

    return results


def combine_text(*values: Any) -> str:
    return " ".join(unique_text_parts(list(values)))


def as_text_list(value: Any) -> list[str]:
    if is_blank(value):
        return []

    if isinstance(value, list):
        values = value
    else:
        values = [value]

    results = []

    for item in values:
        if isinstance(item, dict):
            candidate = direct_value(
                item,
                "name",
                "title",
                "label",
                "value",
            )
        else:
            candidate = item

        text = normalise_text(candidate)

        if text:
            results.append(text)

    return unique_text_parts(results)


def research_keywords(record: dict[str, Any]) -> list[str]:
    value = profile_value(
        record,
        "research_keywords",
        "researchkeywords",
    )

    if isinstance(value, list):
        return as_text_list(value)

    text = normalise_text(value)

    if not text:
        return []

    return unique_text_parts(
        part.strip()
        for part in re.split(r"[,;|]", text)
        if part.strip()
    )


def get_evidence_text(
    record: dict[str, Any],
    evidence_types: set[str],
) -> tuple[str, str]:
    text_parts = []
    source_urls = []

    for page in get_evidence_pages(record):
        evidence_type = normalise_text(
            direct_value(
                page,
                "evidence_type",
                "evidencetype",
            )
        ).lower()

        if evidence_type not in evidence_types:
            continue

        source_url = normalise_text(
            direct_value(
                page,
                "source_url",
                "sourceurl",
                "url",
            )
        )

        if source_url:
            source_urls.append(source_url)

        sections = direct_value(page, "sections")
        section_texts = []

        if isinstance(sections, list):
            for section in sections:
                if not isinstance(section, dict):
                    continue

                section_text = normalise_text(
                    direct_value(
                        section,
                        "text",
                        "content",
                    )
                )

                if section_text:
                    section_texts.append(section_text)

        if section_texts:
            text_parts.extend(section_texts)
        else:
            page_text = normalise_text(
                direct_value(page, "text", "content")
            )

            if page_text:
                text_parts.append(page_text)

    return (
        combine_text(*text_parts),
        combine_text(*source_urls),
    )


def normalise_doi(value: Any) -> str:
    text = normalise_text(value)

    if not text:
        return ""

    text = (
        text.replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .replace("https://dx.doi.org/", "")
        .replace("http://dx.doi.org/", "")
        .replace("doi:", "")
        .strip()
    )

    match = DOI_PATTERN.search(text)

    if not match:
        return ""

    return match.group(1).rstrip(".,;)").lower()


def normalise_module_code(value: Any) -> str:
    text = normalise_text(value).upper()
    text = re.sub(r"[\s-]+", "", text)

    if text.lower() in {
        "",
        "none",
        "null",
        "n/a",
        "na",
        "tbc",
    }:
        return ""

    return text


def extract_module_codes(text: str) -> list[str]:
    codes = {
        normalise_module_code(match.group(0))
        for match in MODULE_CODE_PATTERN.finditer(text)
    }

    return sorted(code for code in codes if code)


def create_module_id(module: dict[str, Any]) -> str:
    module_id = normalise_text(
        direct_value(module, "module_id", "moduleid")
    )

    if module_id:
        return module_id

    cms_id = normalise_text(
        direct_value(
            module,
            "module_cms_id",
            "modulecmsid",
        )
    )

    if cms_id:
        return f"module-cms-{cms_id}"

    title = normalise_text(
        direct_value(
            module,
            "module_title",
            "moduletitle",
            "title",
        )
    )

    description = normalise_text(
        direct_value(module, "description")
    )

    identity_text = f"{title.casefold()}|{description.casefold()}"
    return f"module-generated-{stable_hash(identity_text)}"


def publication_identity(
    publication: dict[str, Any],
) -> tuple[str, str]:
    doi = normalise_doi(
        direct_value(
            publication,
            "doi",
            "doi_url",
            "doiurl",
        )
    )

    if doi:
        return f"publication-doi-{stable_hash(doi)}", doi

    title = normalise_text(
        direct_value(publication, "title")
    )

    year = normalise_text(
        direct_value(publication, "year")
    )

    citation = normalise_text(
        direct_value(publication, "citation")
    )

    identity_text = combine_text(
        title.casefold(),
        year,
        citation.casefold(),
    )

    return (
        f"publication-generated-{stable_hash(identity_text)}",
        "",
    )


def flatten_staff(
    staff_records: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    staff_rows = []
    publication_map: dict[str, dict[str, Any]] = {}
    publication_links = []
    teaching_rows = []

    seen_publication_links = set()

    for record in staff_records:
        staff_id = normalise_text(
            profile_value(record, "staff_id", "staffid")
        )

        if not staff_id:
            continue

        full_name = normalise_text(
            profile_value(
                record,
                "full_name",
                "fullname",
                "name",
            )
        )

        position = normalise_text(
            profile_value(
                record,
                "position",
                "job_title",
                "jobtitle",
            )
        )

        profile_url = normalise_text(
            profile_value(
                record,
                "profile_url",
                "profileurl",
                "canonical_url",
                "canonicalurl",
            )
        )

        college_code = normalise_text(
            profile_value(
                record,
                "college_code",
                "collegecode",
            )
        )

        college_name = normalise_text(
            profile_value(
                record,
                "college_name",
                "collegename",
                "college",
            )
        )

        department_code = normalise_text(
            profile_value(
                record,
                "department_code",
                "departmentcode",
            )
        )

        department_name = normalise_text(
            profile_value(
                record,
                "department_name",
                "departmentname",
                "department",
            )
        )

        summary = normalise_text(
            profile_value(
                record,
                "summary",
                "short_summary",
                "shortsummary",
                "meta_description",
                "metadescription",
            )
        )

        qualifications = normalise_text(
            profile_value(record, "qualifications")
        )

        responsibility = normalise_text(
            profile_value(record, "responsibility")
        )

        keywords = research_keywords(record)

        research_text, research_url = get_evidence_text(
            record,
            {"research"},
        )

        phd_text, phd_url = get_evidence_text(
            record,
            {"phd-topics", "phd_topics"},
        )

        teaching_text, teaching_url = get_evidence_text(
            record,
            {"teaching"},
        )

        publication_text_parts = []

        for publication in get_publications(record):
            publication_id, doi = publication_identity(
                publication
            )

            title = normalise_text(
                direct_value(publication, "title")
            )

            year = normalise_text(
                direct_value(publication, "year")
            )

            publication_type = normalise_text(
                direct_value(
                    publication,
                    "publication_type",
                    "publicationtype",
                    "type",
                )
            )

            citation = normalise_text(
                direct_value(publication, "citation")
            )

            doi_url = (
                f"https://doi.org/{doi}"
                if doi
                else ""
            )

            publication_row = {
                "publication_id": publication_id,
                "title": title,
                "year": year,
                "publication_type": publication_type,
                "citation": citation,
                "doi": doi,
                "doi_url": doi_url,
                "matching_text": combine_text(
                    title,
                    citation,
                ),
            }

            current = publication_map.get(publication_id)

            if current is None:
                publication_map[publication_id] = (
                    publication_row
                )
            elif (
                len(publication_row["matching_text"])
                > len(current["matching_text"])
            ):
                publication_map[publication_id] = (
                    publication_row
                )

            link_key = (staff_id, publication_id)

            if link_key not in seen_publication_links:
                seen_publication_links.add(link_key)
                publication_links.append(
                    {
                        "staff_id": staff_id,
                        "publication_id": publication_id,
                    }
                )

            if title:
                publication_text_parts.append(title)

        profile_matching_text = combine_text(
            summary,
            " ".join(keywords),
            qualifications,
            responsibility,
            research_text,
            phd_text,
        )

        publication_matching_text = combine_text(
            *publication_text_parts
        )

        staff_rows.append(
            {
                "staff_id": staff_id,
                "full_name": full_name,
                "position": position,
                "profile_url": profile_url,
                "college_code": college_code,
                "college_name": college_name,
                "department_code": department_code,
                "department_name": department_name,
                "research_keywords": keywords,
                "summary": summary,
                "qualifications": qualifications,
                "responsibility": responsibility,
                "research_text": research_text,
                "research_url": research_url,
                "phd_text": phd_text,
                "phd_url": phd_url,
                "profile_matching_text": (
                    profile_matching_text
                ),
                "publication_matching_text": (
                    publication_matching_text
                ),
                "matching_text_without_teaching": (
                    combine_text(
                        profile_matching_text,
                        publication_matching_text,
                    )
                ),
                "has_publications": bool(
                    publication_text_parts
                ),
                "has_teaching_evidence": bool(
                    teaching_text
                ),
            }
        )

        if teaching_text:
            teaching_rows.append(
                {
                    "teaching_evidence_id": (
                        f"teaching-{staff_id}"
                    ),
                    "staff_id": staff_id,
                    "source_url": teaching_url,
                    "text": teaching_text,
                    "module_codes": extract_module_codes(
                        teaching_text
                    ),
                }
            )

    publications = sorted(
        publication_map.values(),
        key=lambda row: row["publication_id"],
    )

    return (
        staff_rows,
        publications,
        publication_links,
        teaching_rows,
    )


def flatten_courses(
    course_records: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    course_rows = []
    module_map: dict[str, dict[str, Any]] = {}
    module_contexts: dict[str, dict[str, set[str]]] = (
        defaultdict(
            lambda: {
                "course_ids": set(),
                "departments": set(),
                "colleges": set(),
            }
        )
    )

    course_module_links = []
    seen_links = set()

    for course in course_records:
        course_id = normalise_text(
            direct_value(course, "course_id", "courseid")
        )

        if not course_id:
            continue

        course_title = normalise_text(
            direct_value(
                course,
                "course_title",
                "coursetitle",
                "title",
            )
        )

        departments = as_text_list(
            direct_value(
                course,
                "department",
                "departments",
            )
        )

        colleges = as_text_list(
            direct_value(
                course,
                "college",
                "colleges",
            )
        )

        modules = direct_value(course, "modules")

        if not isinstance(modules, list):
            modules = []

        course_rows.append(
            {
                "course_id": course_id,
                "course_cms_id": normalise_text(
                    direct_value(
                        course,
                        "course_cms_id",
                        "coursecmsid",
                    )
                ),
                "course_slug": normalise_text(
                    direct_value(
                        course,
                        "course_slug",
                        "courseslug",
                    )
                ),
                "course_title": course_title,
                "source_url": normalise_text(
                    direct_value(
                        course,
                        "source_url",
                        "sourceurl",
                    )
                ),
                "academic_year": normalise_text(
                    direct_value(
                        course,
                        "academic_year",
                        "academicyear",
                    )
                ),
                "level": normalise_text(
                    direct_value(course, "level")
                ),
                "colleges": colleges,
                "departments": departments,
                "subject_areas": as_text_list(
                    direct_value(
                        course,
                        "subject_areas",
                        "subjectareas",
                    )
                ),
                "subjects": as_text_list(
                    direct_value(course, "subjects")
                ),
                "study_modes": as_text_list(
                    direct_value(
                        course,
                        "study_modes",
                        "studymodes",
                    )
                ),
                "start_dates": as_text_list(
                    direct_value(
                        course,
                        "start_dates",
                        "startdates",
                    )
                ),
                "overview": normalise_text(
                    direct_value(course, "overview")
                ),
                "matching_text": normalise_text(
                    direct_value(
                        course,
                        "matching_text",
                        "matchingtext",
                    )
                ),
                "module_count": sum(
                    isinstance(module, dict)
                    for module in modules
                ),
            }
        )

        for occurrence_index, module in enumerate(modules):
            if not isinstance(module, dict):
                continue

            module_id = create_module_id(module)

            title = normalise_text(
                direct_value(
                    module,
                    "module_title",
                    "moduletitle",
                    "title",
                )
            )

            description = normalise_text(
                direct_value(module, "description")
            )

            existing_matching_text = normalise_text(
                direct_value(
                    module,
                    "matching_text",
                    "matchingtext",
                )
            )

            module_matching_text = (
                existing_matching_text
                or combine_text(title, description)
            )

            candidate_row = {
                "module_id": module_id,
                "module_cms_id": normalise_text(
                    direct_value(
                        module,
                        "module_cms_id",
                        "modulecmsid",
                    )
                ),
                "module_slug": normalise_text(
                    direct_value(
                        module,
                        "module_slug",
                        "moduleslug",
                    )
                ),
                "module_code": normalise_module_code(
                    direct_value(
                        module,
                        "module_code",
                        "modulecode",
                    )
                ),
                "module_title": title,
                "description": description,
                "what_you_will_study": normalise_text(
                    direct_value(
                        module,
                        "what_you_will_study",
                        "whatyoullstudy",
                    )
                ),
                "how_you_will_learn": normalise_text(
                    direct_value(
                        module,
                        "how_you_will_learn",
                        "howyoulllearn",
                    )
                ),
                "assessment": normalise_text(
                    direct_value(module, "assessment")
                ),
                "credits": normalise_text(
                    direct_value(module, "credits")
                ),
                "matching_text": module_matching_text,
            }

            current = module_map.get(module_id)

            if current is None:
                module_map[module_id] = candidate_row
            elif (
                len(candidate_row["matching_text"])
                > len(current["matching_text"])
            ):
                module_map[module_id] = candidate_row

            context = module_contexts[module_id]
            context["course_ids"].add(course_id)
            context["departments"].update(departments)
            context["colleges"].update(colleges)

            link_key = (
                course_id,
                module_id,
                normalise_text(
                    direct_value(
                        module,
                        "year_of_study",
                        "yearofstudy",
                    )
                ),
                normalise_text(
                    direct_value(
                        module,
                        "module_type",
                        "moduletype",
                    )
                ),
            )

            if link_key in seen_links:
                continue

            seen_links.add(link_key)

            course_module_links.append(
                {
                    "course_id": course_id,
                    "module_id": module_id,
                    "year_of_study": link_key[2],
                    "module_type": link_key[3],
                    "position_in_group": normalise_text(
                        direct_value(
                            module,
                            "position_in_group",
                            "positioningroup",
                        )
                    ),
                    "occurrence_index": occurrence_index,
                    "course_title": course_title,
                    "course_departments": departments,
                    "course_colleges": colleges,
                }
            )

    module_rows = []

    for module_id, module in module_map.items():
        context = module_contexts[module_id]

        module_rows.append(
            {
                **module,
                "course_ids": sorted(
                    context["course_ids"]
                ),
                "departments": sorted(
                    context["departments"]
                ),
                "colleges": sorted(
                    context["colleges"]
                ),
                "course_count": len(
                    context["course_ids"]
                ),
            }
        )

    module_rows.sort(key=lambda row: row["module_id"])

    return (
        course_rows,
        module_rows,
        course_module_links,
    )


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            file.write("\n")


def run_preprocessing(
    staff_path: Path,
    course_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    staff_records = load_json_records(
        staff_path,
        ("staff", "profiles", "records", "data"),
    )

    course_records = load_json_records(
        course_path,
        ("courses", "records", "data"),
    )

    (
        staff_rows,
        publications,
        publication_links,
        teaching_rows,
    ) = flatten_staff(staff_records)

    (
        course_rows,
        module_rows,
        course_module_links,
    ) = flatten_courses(course_records)

    datasets = {
        "staff": staff_rows,
        "courses": course_rows,
        "modules": module_rows,
        "course_module_links": course_module_links,
        "publications": publications,
        "staff_publication_links": publication_links,
        "teaching_evidence": teaching_rows,
    }

    for name, records in datasets.items():
        write_jsonl(
            output_directory / f"{name}.jsonl",
            records,
        )

    manifest = {
        "input_staff_records": len(staff_records),
        "input_course_records": len(course_records),
        "output_counts": {
            name: len(records)
            for name, records in datasets.items()
        },
    }

    with (
        output_directory / "preprocessing_manifest.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return manifest