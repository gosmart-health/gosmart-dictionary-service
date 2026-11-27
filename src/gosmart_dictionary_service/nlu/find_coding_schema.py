"""``find_coding_schema`` intent: parse coding schema and code/tag lookups."""

from __future__ import annotations

import re
from collections.abc import Collection
from datetime import datetime

from gosmart_clinical_nlu.intents import EntityBinding, IntentSpec
from pydantic import BaseModel

# Canonical vocabulary / coding schema normalization map.
# Captures both currently supported schemas (DICOM, LOINC, ICD-10/ICD-10-CM, SNOMED)
# and additional clinical schemas so downstream services can dynamically check
# active database availability and alert users when a requested schema is not loaded.
_SCHEMA_ALIASES: dict[str, str] = {
    "snomed": "SNOMED",
    "snomedct": "SNOMED",
    "sct": "SNOMED",
    "loinc": "LOINC",
    "icd": "ICD10",
    "icd10": "ICD10",
    "icd10cm": "ICD10CM",
    "icd10pcs": "ICD10PCS",
    "icd11": "ICD11",
    "dicom": "DICOM",
    "dicomtag": "DICOM",
    "dicomelement": "DICOM",
    "dicomattribute": "DICOM",
    "dicomuid": "DICOM",
    "dicomvr": "DICOM",
    "dcm": "DCM",
    "ucum": "UCUM",
    "rxnorm": "RxNorm",
    "cpt": "CPT4",
    "cpt4": "CPT4",
    "hcpcs": "HCPCS",
    "ndc": "NDC",
    "atc": "ATC",
    "mesh": "MeSH",
    "radlex": "RadLex",
}

_SCHEMA_COMPACT_RE = re.compile(r"[\s_-]+")
_HYPHEN_SPACE_RE = re.compile(r"\s*-\s*")
_DICOM_TAG_STRIP_RE = re.compile(r"[\s(),]+")


def normalize_coding_schema(raw: str, ref_date: datetime | None = None) -> str:
    """Normalize a matched coding schema mention to its canonical identifier."""
    del ref_date
    compact = _SCHEMA_COMPACT_RE.sub("", raw.strip().lower())
    return _SCHEMA_ALIASES.get(compact, raw.strip().upper())


def normalize_coding_code(raw: str, ref_date: datetime | None = None) -> str:
    """Normalize a matched clinical terminology code."""
    del ref_date
    cleaned = _HYPHEN_SPACE_RE.sub("-", raw.strip())
    return cleaned.upper()


def normalize_dicom_tag(raw: str, ref_date: datetime | None = None) -> str:
    """Normalize a DICOM tag (``(0008, 0010)``, ``0018,0050``, ``00100020``) to 8-char uppercase hex."""
    del ref_date
    return _DICOM_TAG_STRIP_RE.sub("", raw.strip()).upper()


def normalize_dicom_uid(raw: str, ref_date: datetime | None = None) -> str:
    """Normalize a DICOM UID string."""
    del ref_date
    return raw.strip()


class CodingSchemaQueryIntent(BaseModel):
    """Slots extracted for a clinical code or DICOM tag lookup utterance."""

    coding_schema: str | None = None
    code: str | None = None
    dicom_tag: str | None = None
    dicom_uid: str | None = None

    @property
    def effective_schema(self) -> str | None:
        """Return explicit schema, or infer ``DICOM`` when a DICOM tag/UID was extracted."""
        if self.coding_schema:
            return self.coding_schema
        if self.dicom_tag or self.dicom_uid:
            return "DICOM"
        return None

    @property
    def identifier(self) -> str | None:
        """Return the extracted code, DICOM tag, or DICOM UID value if present."""
        return self.dicom_uid or self.dicom_tag or self.code

    def is_schema_available(self, available_schemas: Collection[str]) -> bool:
        """Check whether the requested schema exists in the active database's loaded schemas."""
        schema = self.effective_schema
        if schema is None:
            return True
        normalized_available = {item.strip().upper() for item in available_schemas}
        return schema.strip().upper() in normalized_available


ENTITY_MAP: dict[str, EntityBinding] = {
    "CODING_SCHEMA": EntityBinding(field="coding_schema", normalize=normalize_coding_schema),
    "CODING_CODE": EntityBinding(field="code", normalize=normalize_coding_code),
    "DICOM_TAG": EntityBinding(field="dicom_tag", normalize=normalize_dicom_tag),
    "DICOM_UID": EntityBinding(field="dicom_uid", normalize=normalize_dicom_uid),
}

FIND_CODING_SCHEMA_SPEC = IntentSpec(
    name="find_coding_schema",
    label="FIND_CODING_SCHEMA",
    model_cls=CodingSchemaQueryIntent,
    entity_map=ENTITY_MAP,
    description="Look up a specific clinical terminology code or DICOM tag by coding schema and identifier.",
    min_confidence=0.6,
    required_fields=(),
)
