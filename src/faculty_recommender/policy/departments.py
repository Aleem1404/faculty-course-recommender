from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml


def normalise_department_label(value: Any) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize(
        "NFKD",
        str(value),
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    text = text.casefold()
    text = text.replace("&", " and ")

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    text = " ".join(text.split())

    prefixes = (
        "department of ",
        "school of ",
        "division of ",
        "brunel university of london ",
        "brunel university ",
        "brunel ",
    )

    changed = True

    while changed:
        changed = False

        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                changed = True

    suffixes = (
        " department",
        " school",
        " division",
    )

    for suffix in suffixes:
        if text.endswith(suffix):
            text = text[:-len(suffix)].strip()

    return text


def department_slug(value: Any) -> str:
    normalised = normalise_department_label(value)

    return normalised.replace(" ", "_")


def load_department_aliases(
    path: Path,
) -> dict[str, str]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    departments = payload.get("departments", {})

    if not isinstance(departments, dict):
        raise ValueError(
            "The department aliases file must contain "
            "a 'departments' mapping."
        )

    aliases: dict[str, str] = {}

    for canonical_id, values in departments.items():
        canonical_key = department_slug(canonical_id)

        aliases[
            normalise_department_label(canonical_id)
        ] = canonical_key

        if not isinstance(values, list):
            continue

        for value in values:
            normalised_value = (
                normalise_department_label(value)
            )

            if normalised_value:
                aliases[normalised_value] = canonical_key

    return aliases


def canonicalise_department(
    value: Any,
    aliases: dict[str, str] | None = None,
) -> str:
    normalised = normalise_department_label(value)

    if not normalised:
        return ""

    aliases = aliases or {}

    if normalised in aliases:
        return aliases[normalised]

    return department_slug(normalised)


def canonicalise_departments(
    values: Any,
    aliases: dict[str, str] | None = None,
) -> set[str]:
    if values is None:
        return set()

    if not isinstance(values, list):
        values = [values]

    canonical_values = {
        canonicalise_department(
            value,
            aliases,
        )
        for value in values
    }

    return {
        value
        for value in canonical_values
        if value
    }