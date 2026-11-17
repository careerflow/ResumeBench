"""Section specifications for resume extraction scoring."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SectionKind(str, Enum):
    """Type of section structure in the resume schema."""

    SINGLETON = "singleton"
    FLAT_LIST = "flat_list"
    ENTITY_LIST = "entity_list"


@dataclass(frozen=True)
class SectionSpec:
    """Specification for a resume section."""

    name: str
    kind: SectionKind
    key_fields: tuple[str, ...]
    score_description: bool = False


SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec(
        "basics",
        SectionKind.SINGLETON,
        (
            "prefix",
            "fname",
            "lname",
            "suffix",
            "email",
            "phone",
            "city",
            "state",
            "country",
            "hasPersonalPhoto",
            "linkedinUrl",
        ),
    ),
    SectionSpec("experience", SectionKind.ENTITY_LIST, ("company", "position"), score_description=True),
    SectionSpec("education", SectionKind.ENTITY_LIST, ("institution", "area"), score_description=True),
    SectionSpec("projects", SectionKind.ENTITY_LIST, ("name",), score_description=True),
    SectionSpec("personalSummary", SectionKind.ENTITY_LIST, ("text",)),
    SectionSpec("certifications", SectionKind.ENTITY_LIST, ("name", "issuer")),
    SectionSpec("awards", SectionKind.ENTITY_LIST, ("title",)),
    SectionSpec("volunteering", SectionKind.ENTITY_LIST, ("organization", "position"), score_description=True),
    SectionSpec("skills", SectionKind.FLAT_LIST, ()),
    SectionSpec("publications", SectionKind.ENTITY_LIST, ("name",)),
    SectionSpec("languages", SectionKind.ENTITY_LIST, ("name",)),
    SectionSpec("interests", SectionKind.ENTITY_LIST, ("name",)),
    SectionSpec("profiles", SectionKind.ENTITY_LIST, ("network",)),
    SectionSpec("customSections", SectionKind.ENTITY_LIST, ("sectionTitle",), score_description=True),
)
