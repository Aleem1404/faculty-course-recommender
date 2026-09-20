# from __future__ import annotations

# import json
# from collections import defaultdict
# from dataclasses import asdict, dataclass
# from pathlib import Path
# from typing import Any

# from faculty_recommender.policy.departments import (
#     normalise_department_label,
# )


# @dataclass(frozen=True)
# class AcademicUnit:
#     path: str
#     unit_id: str
#     name: str
#     level: str
#     parent_path: str | None
#     college_path: str
#     department_path: str | None
#     aliases: tuple[str, ...]
#     staff_url: str
#     active: bool


# class AcademicHierarchy:
#     def __init__(
#         self,
#         units: dict[str, AcademicUnit],
#         validation_errors: list[str],
#     ) -> None:
#         self.units = units
#         self.validation_errors = validation_errors

#         self.label_index: dict[
#             str,
#             list[str],
#         ] = defaultdict(list)

#         for path, unit in units.items():
#             labels = [
#                 unit.unit_id,
#                 unit.name,
#                 *unit.aliases,
#             ]

#             for label in labels:
#                 normalised = normalise_department_label(
#                     label
#                 )

#                 if (
#                     normalised
#                     and path
#                     not in self.label_index[normalised]
#                 ):
#                     self.label_index[
#                         normalised
#                     ].append(path)

#     @classmethod
#     def from_file(
#         cls,
#         path: Path,
#     ) -> AcademicHierarchy:
#         payload = json.loads(
#             path.read_text(encoding="utf-8")
#         )

#         return cls.from_dict(payload)

#     @classmethod
#     def from_dict(
#         cls,
#         payload: dict[str, Any],
#     ) -> AcademicHierarchy:
#         units: dict[str, AcademicUnit] = {}
#         errors: list[str] = []

#         colleges = payload.get("colleges")

#         if not isinstance(colleges, dict):
#             colleges = {
#                 key: value
#                 for key, value in payload.items()
#                 if isinstance(value, dict)
#                 and "departments" in value
#             }

#         if not colleges:
#             errors.append(
#                 "No college objects were found."
#             )

#         def add_unit(unit: AcademicUnit) -> None:
#             if unit.path in units:
#                 errors.append(
#                     f"Duplicate unit path: {unit.path}"
#                 )
#                 return

#             if not unit.name:
#                 errors.append(
#                     f"Missing unit name: {unit.path}"
#                 )

#             units[unit.path] = unit

#         for college_id, college in colleges.items():
#             if not isinstance(college, dict):
#                 errors.append(
#                     f"Invalid college: {college_id}"
#                 )
#                 continue

#             college_path = str(college_id)
#             college_name = str(
#                 college.get("name", "")
#             ).strip()

#             add_unit(
#                 AcademicUnit(
#                     path=college_path,
#                     unit_id=str(college_id),
#                     name=college_name,
#                     level="college",
#                     parent_path=None,
#                     college_path=college_path,
#                     department_path=None,
#                     aliases=tuple(
#                         college.get("aliases", [])
#                         or []
#                     ),
#                     staff_url=str(
#                         college.get("staff_url", "")
#                     ).strip(),
#                     active=bool(
#                         college.get("active", True)
#                     ),
#                 )
#             )

#             departments = college.get(
#                 "departments",
#                 {},
#             )

#             if not isinstance(departments, dict):
#                 errors.append(
#                     f"Invalid departments mapping: "
#                     f"{college_path}"
#                 )
#                 continue

#             for department_id, department in (
#                 departments.items()
#             ):
#                 if not isinstance(department, dict):
#                     errors.append(
#                         f"Invalid department: "
#                         f"{college_path}/"
#                         f"{department_id}"
#                     )
#                     continue

#                 department_path = (
#                     f"{college_path}/{department_id}"
#                 )

#                 add_unit(
#                     AcademicUnit(
#                         path=department_path,
#                         unit_id=str(department_id),
#                         name=str(
#                             department.get(
#                                 "name",
#                                 "",
#                             )
#                         ).strip(),
#                         level="department",
#                         parent_path=college_path,
#                         college_path=college_path,
#                         department_path=(
#                             department_path
#                         ),
#                         aliases=tuple(
#                             department.get(
#                                 "aliases",
#                                 [],
#                             )
#                             or []
#                         ),
#                         staff_url=str(
#                             department.get(
#                                 "staff_url",
#                                 "",
#                             )
#                         ).strip(),
#                         active=bool(
#                             department.get(
#                                 "active",
#                                 True,
#                             )
#                         ),
#                     )
#                 )

#                 divisions = department.get(
#                     "divisions",
#                     {},
#                 )

#                 if divisions is None:
#                     divisions = {}

#                 if not isinstance(divisions, dict):
#                     errors.append(
#                         f"Invalid divisions mapping: "
#                         f"{department_path}"
#                     )
#                     continue

#                 for division_id, division in (
#                     divisions.items()
#                 ):
#                     if not isinstance(
#                         division,
#                         dict,
#                     ):
#                         errors.append(
#                             f"Invalid division: "
#                             f"{department_path}/"
#                             f"{division_id}"
#                         )
#                         continue

#                     division_path = (
#                         f"{department_path}/"
#                         f"{division_id}"
#                     )

#                     add_unit(
#                         AcademicUnit(
#                             path=division_path,
#                             unit_id=str(
#                                 division_id
#                             ),
#                             name=str(
#                                 division.get(
#                                     "name",
#                                     "",
#                                 )
#                             ).strip(),
#                             level="division",
#                             parent_path=(
#                                 department_path
#                             ),
#                             college_path=(
#                                 college_path
#                             ),
#                             department_path=(
#                                 department_path
#                             ),
#                             aliases=tuple(
#                                 division.get(
#                                     "aliases",
#                                     [],
#                                 )
#                                 or []
#                             ),
#                             staff_url=str(
#                                 division.get(
#                                     "staff_url",
#                                     "",
#                                 )
#                             ).strip(),
#                             active=bool(
#                                 division.get(
#                                     "active",
#                                     True,
#                                 )
#                             ),
#                         )
#                     )

#         return cls(
#             units=units,
#             validation_errors=errors,
#         )

#     def resolve_one(
#         self,
#         label: Any,
#         allowed_college_paths: (
#             set[str] | None
#         ) = None,
#     ) -> dict[str, Any]:
#         original_label = str(
#             label or ""
#         ).strip()

#         normalised = normalise_department_label(
#             original_label
#         )

#         if not normalised:
#             return {
#                 "label": original_label,
#                 "normalised_label": "",
#                 "status": "missing",
#                 "paths": [],
#             }

#         candidates = list(
#             self.label_index.get(
#                 normalised,
#                 [],
#             )
#         )

#         if allowed_college_paths:
#             candidates = [
#                 path
#                 for path in candidates
#                 if self.units[
#                     path
#                 ].college_path
#                 in allowed_college_paths
#             ]

#         if len(candidates) == 1:
#             status = "resolved"
#         elif len(candidates) > 1:
#             status = "ambiguous"
#         else:
#             status = "unresolved"

#         return {
#             "label": original_label,
#             "normalised_label": normalised,
#             "status": status,
#             "paths": candidates,
#         }

#     def resolve_many(
#         self,
#         labels: Any,
#         college_labels: Any = None,
#     ) -> dict[str, Any]:
#         if labels is None:
#             labels = []

#         if not isinstance(labels, list):
#             labels = [labels]

#         if college_labels is None:
#             college_labels = []

#         if not isinstance(
#             college_labels,
#             list,
#         ):
#             college_labels = [
#                 college_labels
#             ]

#         college_results = [
#             self.resolve_one(label)
#             for label in college_labels
#         ]

#         college_paths = {
#             path
#             for result in college_results
#             for path in result["paths"]
#             if self.units[path].level
#             == "college"
#         }

#         label_results = [
#             self.resolve_one(
#                 label,
#                 allowed_college_paths=(
#                     college_paths or None
#                 ),
#             )
#             for label in labels
#         ]

#         resolved_paths = sorted({
#             result["paths"][0]
#             for result in label_results
#             if result["status"] == "resolved"
#         })

#         ambiguous = [
#             result
#             for result in label_results
#             if result["status"] == "ambiguous"
#         ]

#         unresolved = [
#             result["label"]
#             for result in label_results
#             if result["status"] == "unresolved"
#         ]

#         missing = [
#             result["label"]
#             for result in label_results
#             if result["status"] == "missing"
#         ]

#         if resolved_paths and not (
#             ambiguous or unresolved
#         ):
#             status = "resolved"
#         elif resolved_paths:
#             status = "partially_resolved"
#         elif ambiguous:
#             status = "ambiguous"
#         elif unresolved:
#             status = "unresolved"
#         else:
#             status = "missing"

#         return {
#             "status": status,
#             "resolved_paths": resolved_paths,
#             "unresolved_labels": unresolved,
#             "ambiguous_labels": ambiguous,
#             "missing_labels": missing,
#             "college_paths": sorted(
#                 college_paths
#             ),
#             "college_results": (
#                 college_results
#             ),
#             "label_results": label_results,
#         }

#     def is_same_or_descendant(
#         self,
#         possible_child: str,
#         possible_parent: str,
#     ) -> bool:
#         return (
#             possible_child == possible_parent
#             or possible_child.startswith(
#                 f"{possible_parent}/"
#             )
#         )

#     def alias_collisions(
#         self,
#     ) -> dict[str, list[str]]:
#         return {
#             label: paths
#             for label, paths
#             in self.label_index.items()
#             if len(paths) > 1
#         }

#     def export_units(
#         self,
#     ) -> list[dict[str, Any]]:
#         return [
#             asdict(unit)
#             for unit in self.units.values()
#         ]
from __future__ import annotations
from huggingface_hub.inference._generated.types import zero_shot_image_classification
from huggingface_hub.inference._generated.types import zero_shot_image_classification
from huggingface_hub.inference._generated.types import zero_shot_image_classification
from huggingface_hub.inference._generated.types import zero_shot_image_classification
from huggingface_hub.inference._generated.types import zero_shot_image_classification
from huggingface_hub.inference._generated.types import zero_shot_image_classification

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from faculty_recommender.policy.departments import (
    normalise_department_label,
)


@dataclass(frozen=True)
class AcademicUnit:
    path: str
    unit_id: str
    name: str
    level: str
    parent_path: str | None
    college_path: str
    department_path: str | None
    aliases: tuple[str, ...]
    staff_url: str
    active: bool


class AcademicHierarchy:
    def __init__(
        self,
        units: dict[str, AcademicUnit],
        validation_errors: list[str],
        source_mappings: dict[str, dict[str, list[str]]],
    ) -> None:
        self.units = units
        self.validation_errors = validation_errors
        self.source_mappings = source_mappings

        self.label_index: dict[str, list[str]] = defaultdict(list)

        for path, unit in units.items():
            labels = [
                unit.unit_id,
                unit.name,
                *unit.aliases,
            ]

            for label in labels:
                normalised = normalise_department_label(label)

                if (
                    normalised
                    and path not in self.label_index[normalised]
                ):
                    self.label_index[normalised].append(path)

    @classmethod
    def from_file(
        cls,
        path: Path,
    ) -> AcademicHierarchy:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> AcademicHierarchy:
        units: dict[str, AcademicUnit] = {}
        errors: list[str] = []

        raw_source_mappings = payload.get("source_mappings", {})
        source_mappings = {
            "staff_departments": {},
            "module_departments": {},
        }

        for mapping_group in ("staff_departments", "module_departments"):
            group_values = raw_source_mappings.get(mapping_group, {})

            if isinstance(group_values, dict):
                for raw_label, raw_paths in group_values.items():
                    normalised_label = normalise_department_label(raw_label)

                    if not normalised_label:
                        continue

                    if not isinstance(raw_paths, list):
                        errors.append(
                            f"Invalid source mapping list for "
                            f"{mapping_group}:{raw_label}"
                        )
                        continue

                    clean_paths = [
                        str(path).strip()
                        for path in raw_paths
                        if str(path).strip()
                    ]

                    source_mappings[mapping_group][normalised_label] = clean_paths

        colleges = payload.get("colleges")

        if not isinstance(colleges, dict):
            colleges = {
                key: value
                for key, value in payload.items()
                if isinstance(value, dict) and "departments" in value
            }

        if not colleges:
            errors.append("No college objects were found.")

        def add_unit(unit: AcademicUnit) -> None:
            if unit.path in units:
                errors.append(f"Duplicate unit path: {unit.path}")
                return

            if not unit.name:
                errors.append(f"Missing unit name: {unit.path}")

            units[unit.path] = unit

        for college_id, college in colleges.items():
            if not isinstance(college, dict):
                errors.append(f"Invalid college: {college_id}")
                continue

            college_path = str(college_id)
            college_name = str(college.get("name", "")).strip()

            add_unit(
                AcademicUnit(
                    path=college_path,
                    unit_id=str(college_id),
                    name=college_name,
                    level="college",
                    parent_path=None,
                    college_path=college_path,
                    department_path=None,
                    aliases=tuple(college.get("aliases", []) or []),
                    staff_url=str(college.get("staff_url", "")).strip(),
                    active=bool(college.get("active", True)),
                )
            )

            departments = college.get("departments", {})

            if not isinstance(departments, dict):
                errors.append(f"Invalid departments mapping: {college_path}")
                continue

            for department_id, department in departments.items():
                if not isinstance(department, dict):
                    errors.append(
                        f"Invalid department: {college_path}/{department_id}"
                    )
                    continue

                department_path = f"{college_path}/{department_id}"

                add_unit(
                    AcademicUnit(
                        path=department_path,
                        unit_id=str(department_id),
                        name=str(department.get("name", "")).strip(),
                        level="department",
                        parent_path=college_path,
                        college_path=college_path,
                        department_path=department_path,
                        aliases=tuple(department.get("aliases", []) or []),
                        staff_url=str(department.get("staff_url", "")).strip(),
                        active=bool(department.get("active", True)),
                    )
                )

                divisions = department.get("divisions", {})

                if divisions is None:
                    divisions = {}

                if not isinstance(divisions, dict):
                    errors.append(f"Invalid divisions mapping: {department_path}")
                    continue

                for division_id, division in divisions.items():
                    if not isinstance(division, dict):
                        errors.append(
                            f"Invalid division: {department_path}/{division_id}"
                        )
                        continue

                    division_path = f"{department_path}/{division_id}"

                    add_unit(
                        AcademicUnit(
                            path=division_path,
                            unit_id=str(division_id),
                            name=str(division.get("name", "")).strip(),
                            level="division",
                            parent_path=department_path,
                            college_path=college_path,
                            department_path=department_path,
                            aliases=tuple(division.get("aliases", []) or []),
                            staff_url=str(division.get("staff_url", "")).strip(),
                            active=bool(division.get("active", True)),
                        )
                    )

        for mapping_group, mappings in source_mappings.items():
            for source_label, mapped_paths in mappings.items():
                for mapped_path in mapped_paths:
                    if mapped_path not in units:
                        errors.append(
                            f"Unknown mapped path in {mapping_group}: "
                            f"{source_label} -> {mapped_path}"
                        )

        return cls(
            units=units,
            validation_errors=errors,
            source_mappings=source_mappings,
        )

    def resolve_one(
        self,
        label: Any,
        allowed_college_paths: set[str] | None = None,
        source_type: str | None = None,
    ) -> dict[str, Any]:
        original_label = str(label or "").strip()
        normalised = normalise_department_label(original_label)

        if not normalised:
            return {
                "label": original_label,
                "normalised_label": "",
                "status": "missing",
                "paths": [],
                "resolution_source": "missing",
            }

        # if source_type in self.source_mappings:
        #     mapped_paths = list(
        #         self.source_mappings[source_type].get(normalised, [])
        #     )

        #     if allowed_college_paths:
        #         mapped_paths = [
        #             path
        #             for path in mapped_paths
        #             if self.units[path].college_path in allowed_college_paths
        #         ]

        #     if mapped_paths:
        #         return {
        #             "label": original_label,
        #             "normalised_label": normalised,
        #             "status": "resolved",
        #             "paths": mapped_paths,
        #             "resolution_source": "source_mapping",
        #         }
        if source_type in self.source_mappings:
            mapped_paths = list(
                self.source_mappings[source_type].get(normalised, [])
            )

            if mapped_paths:
                return {
                    "label": original_label,
                    "normalised_label": normalised,
                    "status": "resolved",
                    "paths": mapped_paths,
                    "resolution_source": "source_mapping",
                }

        candidates = list(self.label_index.get(normalised, []))

        if allowed_college_paths:
            candidates = [
                path
                for path in candidates
                if self.units[path].college_path in allowed_college_paths
            ]

        if len(candidates) == 1:
            status = "resolved"
        elif len(candidates) > 1:
            status = "ambiguous"
        else:
            status = "unresolved"

        return {
            "label": original_label,
            "normalised_label": normalised,
            "status": status,
            "paths": candidates,
            "resolution_source": "label_index",
        }

    def resolve_many(
        self,
        labels: Any,
        college_labels: Any = None,
        source_type: str | None = None,
    ) -> dict[str, Any]:
        if labels is None:
            labels = []

        if not isinstance(labels, list):
            labels = [labels]

        if college_labels is None:
            college_labels = []

        if not isinstance(college_labels, list):
            college_labels = [college_labels]

        college_results = [
            self.resolve_one(label)
            for label in college_labels
        ]

        college_paths = {
            path
            for result in college_results
            for path in result["paths"]
            if self.units[path].level == "college"
        }

        label_results = [
            self.resolve_one(
                label,
                allowed_college_paths=(college_paths or None),
                source_type=source_type,
            )
            for label in labels
        ]

        resolved_paths = sorted({
            path
            for result in label_results
            if result["status"] == "resolved"
            for path in result["paths"]
        })

        ambiguous = [
            result
            for result in label_results
            if result["status"] == "ambiguous"
        ]

        unresolved = [
            result["label"]
            for result in label_results
            if result["status"] == "unresolved"
        ]

        missing = [
            result["label"]
            for result in label_results
            if result["status"] == "missing"
        ]

        if resolved_paths and not (ambiguous or unresolved):
            status = "resolved"
        elif resolved_paths:
            status = "partially_resolved"
        elif ambiguous:
            status = "ambiguous"
        elif unresolved:
            status = "unresolved"
        else:
            status = "missing"

        return {
            "status": status,
            "resolved_paths": resolved_paths,
            "unresolved_labels": unresolved,
            "ambiguous_labels": ambiguous,
            "missing_labels": missing,
            "college_paths": sorted(college_paths),
            "college_results": college_results,
            "label_results": label_results,
        }

    def is_same_or_descendant(
        self,
        possible_child: str,
        possible_parent: str,
    ) -> bool:
        return (
            possible_child == possible_parent
            or possible_child.startswith(f"{possible_parent}/")
        )

    def alias_collisions(self) -> dict[str, list[str]]:
        return {
            label: paths
            for label, paths in self.label_index.items()
            if len(paths) > 1
        }

    def export_units(self) -> list[dict[str, Any]]:
        return [
            asdict(unit)
            for unit in self.units.values()
        ]