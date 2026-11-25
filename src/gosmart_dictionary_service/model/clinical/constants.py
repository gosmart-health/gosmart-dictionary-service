"""Canonical FHIR CodeSystem URIs, ingestion profiles, and OHDSI Athena relationship routing helpers."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Dict, Optional

# Canonical FHIR CodeSystem URIs keyed by OHDSI Athena vocabulary_id.
# Pre-populated with planned clinical/imaging vocabularies so adding future
# Athena exports (ICD10, RxNorm, CPT4, HCPCS, ICD10PCS) or supplements (ICD11, DCM)
# requires zero schema or code changes.
CANONICAL_CODE_SYSTEM_URIS: Dict[str, str] = {
    "SNOMED": "http://snomed.info/sct",
    "LOINC": "http://loinc.org",
    "ICD10CM": "http://hl7.org/fhir/sid/icd-10-cm",
    "ICD10": "http://hl7.org/fhir/sid/icd-10",
    "ICD11": "http://id.who.int/icd/release/11/mms",
    "UCUM": "http://unitsofmeasure.org",
    "RxNorm": "http://www.nlm.nih.gov/research/umls/rxnorm",
    "RxNorm Extension": "http://gosmart.health/terminology/omop/rxnorm-extension",
    "NDC": "http://hl7.org/fhir/sid/ndc",
    "ATC": "http://www.whocc.no/atc",
    "CPT4": "http://www.ama-assn.org/go/cpt",
    "HCPCS": "http://www.cms.gov/Medicare/Coding/HCPCSReleaseCodeSets",
    "ICD10PCS": "http://www.cms.gov/Medicare/Coding/ICD10",
    "DCM": "http://dicom.nema.org/resources/ontology/DCM",
}

DEFAULT_CLINICAL_VOCABULARIES: tuple[str, ...] = (
    "SNOMED",
    "LOINC",
    "ICD10CM",
    "UCUM",
)

DEMO_CORE_DOMAINS: tuple[str, ...] = (
    "Condition",
    "Measurement",
    "Procedure",
    "Spec Anatomic Site",
    "Unit",
)

# Standard OHDSI/SNOMED concept_id for English language in CONCEPT_SYNONYM.csv
ENGLISH_LANGUAGE_CONCEPT_ID: int = 4180186
DEFAULT_SYNONYM_LANGUAGE_IDS: tuple[int, ...] = (ENGLISH_LANGUAGE_CONCEPT_ID,)

# Canonical forward relationship routing sets (exact inverses are omitted losslessly
# because Neo4j traverses relationships bidirectionally in O(1) pointer hops).
IS_A_RELATIONSHIP_IDS: frozenset[str] = frozenset({"Is a"})
MAPS_TO_RELATIONSHIP_IDS: frozenset[str] = frozenset({"Maps to", "Maps to value"})
REPLACED_BY_RELATIONSHIP_IDS: frozenset[str] = frozenset(
    {
        "Concept replaced by",
        "Concept same_as to",
        "Concept poss_eq to",
        "Concept was_a to",
        "Concept alt_to to",
    }
)
RF2_METADATA_RELATIONSHIP_IDS: frozenset[str] = frozenset(
    {
        "Has status",
        "Has Module",
    }
)

# Explicit inverse relationship_ids that are always dropped in favor of their canonical forward edge
_EXPLICIT_REVERSE_RELATIONSHIP_IDS: frozenset[str] = frozenset(
    {
        "Subsumes",
        "Mapped from",
        "Value mapped from",
        "Concept replaces",
        "Concept same_as from",
        "Concept poss_eq from",
        "Concept was_a from",
        "Concept alt_to from",
        "Status of",
        "Module of",
        "Contained in panel",
    }
)

_EXPLICIT_FORWARD_ATTRIBUTE_IDS: frozenset[str] = frozenset(
    {
        "Panel contains",
    }
)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_COMPACT_DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")


class ClinicalIngestionProfile(StrEnum):
    """Built-in scope presets for full self-hosted vs. 8 GB Linux demo deployments."""

    FULL = "full"
    VALID = "valid"
    DEMO_CORE = "demo-core"


def resolve_code_system_uri(vocabulary_id: str) -> str:
    """Resolve the canonical FHIR CodeSystem URI for an OHDSI Athena vocabulary_id."""
    cleaned = vocabulary_id.strip() if isinstance(vocabulary_id, str) else ""
    if not cleaned:
        raise ValueError("vocabulary_id must be a non-empty string.")
    canonical = CANONICAL_CODE_SYSTEM_URIS.get(cleaned)
    if canonical is not None:
        return canonical
    slug = _NON_ALNUM_RE.sub("-", cleaned.lower()).strip("-") or "unknown"
    return f"http://gosmart.health/terminology/omop/{slug}"


def parse_athena_date(raw: Optional[str]) -> str:
    """Normalize an Athena ``YYYYMMDD`` date string into ISO ``YYYY-MM-DD``."""
    if raw is None:
        return ""
    cleaned = str(raw).strip()
    if not cleaned:
        return ""
    match = _COMPACT_DATE_RE.match(cleaned)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return cleaned


def is_canonical_forward_relationship(
    relationship_id: str,
    reverse_relationship_id: Optional[str] = None,
) -> bool:
    """Return True if ``relationship_id`` is the canonical forward edge of a symmetric pair.

    Athena ``CONCEPT_RELATIONSHIP.csv`` materializes every relationship in both directions
    (e.g., ``Is a`` / ``Subsumes``, ``Maps to`` / ``Mapped from``, ``Has finding site`` /
    ``Finding site of``). Storing only the canonical forward direction cuts edge storage
    in half with zero information loss in Neo4j.
    """
    rid = relationship_id.strip()
    if not rid:
        return False
    if rid in _EXPLICIT_REVERSE_RELATIONSHIP_IDS:
        return False
    if (
        rid in IS_A_RELATIONSHIP_IDS
        or rid in MAPS_TO_RELATIONSHIP_IDS
        or rid in REPLACED_BY_RELATIONSHIP_IDS
        or rid in RF2_METADATA_RELATIONSHIP_IDS
        or rid in _EXPLICIT_FORWARD_ATTRIBUTE_IDS
    ):
        return True

    rev = (reverse_relationship_id or "").strip()
    if not rev or rev == rid:
        return True

    if rev in _EXPLICIT_FORWARD_ATTRIBUTE_IDS:
        return False

    # Prefer "Has ..." / "... to" over "... of" / "... from"
    rid_lower = rid.lower()
    rev_lower = rev.lower()
    rid_is_has = rid_lower.startswith("has ")
    rev_is_has = rev_lower.startswith("has ")
    if rid_is_has != rev_is_has:
        return rid_is_has

    rid_is_of = rid_lower.endswith(" of")
    rev_is_of = rev_lower.endswith(" of")
    if rid_is_of != rev_is_of:
        return not rid_is_of

    rid_is_to = rid_lower.endswith(" to")
    rev_is_from = rev_lower.endswith(" from")
    if rid_is_to and rev_is_from:
        return True
    if rid_lower.endswith(" from") and rev_lower.endswith(" to"):
        return False

    # Deterministic fallback tie-breaker for any remaining symmetric pairs
    return rid <= rev


def classify_clinical_relationship(
    relationship_id: str,
    reverse_relationship_id: Optional[str] = None,
    *,
    include_rf2_metadata_edges: bool = False,
    include_attribute_edges: bool = True,
) -> Optional[str]:
    """Map an Athena ``relationship_id`` to a canonical Neo4j relationship type or ``None`` if skipped.

    Returns one of ``"IS_A"``, ``"MAPS_TO"``, ``"REPLACED_BY"``, ``"HAS_ATTRIBUTE"``, or ``None``.
    """
    rid = relationship_id.strip()
    if not is_canonical_forward_relationship(rid, reverse_relationship_id):
        return None
    if rid in IS_A_RELATIONSHIP_IDS:
        return "IS_A"
    if rid in MAPS_TO_RELATIONSHIP_IDS:
        return "MAPS_TO"
    if rid in REPLACED_BY_RELATIONSHIP_IDS:
        return "REPLACED_BY"
    if rid in RF2_METADATA_RELATIONSHIP_IDS:
        return "HAS_ATTRIBUTE" if (include_rf2_metadata_edges and include_attribute_edges) else None
    return "HAS_ATTRIBUTE" if include_attribute_edges else None


__all__ = [
    "CANONICAL_CODE_SYSTEM_URIS",
    "DEFAULT_CLINICAL_VOCABULARIES",
    "DEMO_CORE_DOMAINS",
    "ENGLISH_LANGUAGE_CONCEPT_ID",
    "DEFAULT_SYNONYM_LANGUAGE_IDS",
    "IS_A_RELATIONSHIP_IDS",
    "MAPS_TO_RELATIONSHIP_IDS",
    "REPLACED_BY_RELATIONSHIP_IDS",
    "RF2_METADATA_RELATIONSHIP_IDS",
    "ClinicalIngestionProfile",
    "resolve_code_system_uri",
    "parse_athena_date",
    "is_canonical_forward_relationship",
    "classify_clinical_relationship",
]
