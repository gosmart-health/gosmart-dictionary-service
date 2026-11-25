"""Auditable Neo4j graph node models for the Clinical Terminology & Ontology subgraph."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import Field, field_validator, model_validator

from gosmart_dictionary_service.model.auditable import AuditableNode, generate_deterministic_id
from gosmart_dictionary_service.model.clinical.constants import (
    parse_athena_date,
    resolve_code_system_uri,
)
from gosmart_dictionary_service.model.clinical.seed import (
    AthenaConceptSeed,
    AthenaVocabularySeed,
)


def _serialize_neo4j_value(value: Any) -> Any:
    """Convert Python model field values into Neo4j-compatible primitives."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, list):
        return [_serialize_neo4j_value(item) for item in value]
    return value


def compute_code_system_id(system_uri: str, *, test_id: Optional[str] = None) -> str:
    """Compute the deterministic UUIDv5 ID for a ``:CodeSystem`` node."""
    subset: Dict[str, Any] = {
        "entity": "CodeSystem",
        "system": system_uri.strip(),
    }
    if test_id and test_id.strip():
        subset["test_id"] = test_id.strip()
    return generate_deterministic_id(subset)


def compute_concept_id(system_uri: str, code: str, *, test_id: Optional[str] = None) -> str:
    """Compute the deterministic UUIDv5 ID for a ``:Concept`` node from ``{system, code}``."""
    subset: Dict[str, Any] = {
        "system": system_uri.strip(),
        "code": code.strip(),
    }
    if test_id and test_id.strip():
        subset["test_id"] = test_id.strip()
    return generate_deterministic_id(subset)


class ClinicalAuditableNode(AuditableNode):
    """Shared base for clinical terminology graph nodes with Neo4j property serialization."""

    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Serialize node fields into a flat Neo4j-safe property dictionary."""
        return {k: _serialize_neo4j_value(v) for k, v in self.model_dump().items()}


class CodeSystemNode(ClinicalAuditableNode):
    """Auditable graph node representing a FHIR-aligned terminology ``:CodeSystem``."""

    __label__: ClassVar[str] = "CodeSystem"

    vocabulary_id: str = Field(..., min_length=1, description="OHDSI vocabulary_id (e.g. 'SNOMED', 'LOINC')")
    system_uri: str = Field(default="", description="Canonical FHIR CodeSystem URI (e.g. 'http://snomed.info/sct')")
    name: str = Field(..., min_length=1, description="Human-readable terminology name")
    version: str = Field(default="", description="Vocabulary release version string")
    reference: str = Field(default="", description="Authoritative reference URL or citation")
    athena_concept_id: Optional[int] = Field(
        default=None,
        description="Optional OHDSI Athena vocabulary_concept_id",
    )

    @field_validator("vocabulary_id", "name", mode="before")
    @classmethod
    def _validate_required_str(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Field must be a non-empty string.")
        return v.strip()

    @field_validator("system_uri", "version", "reference", mode="before")
    @classmethod
    def _strip_optional_str(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()

    @model_validator(mode="before")
    @classmethod
    def _populate_system_uri(cls, data: Any) -> Any:
        if isinstance(data, dict):
            uri = data.get("system_uri")
            vocab = data.get("vocabulary_id")
            if (not uri or not str(uri).strip()) and isinstance(vocab, str) and vocab.strip():
                data = dict(data)
                data["system_uri"] = resolve_code_system_uri(vocab)
        return data

    def get_identity_subset(self) -> Dict[str, Any]:
        subset: Dict[str, Any] = {
            "entity": self.__label__,
            "system": self.system_uri,
        }
        if self.test_id:
            subset["test_id"] = self.test_id
        return subset

    @classmethod
    def from_seed(
        cls,
        seed: AthenaVocabularySeed,
        *,
        test_id: Optional[str] = None,
        **audit_kwargs: Any,
    ) -> "CodeSystemNode":
        """Build a ``CodeSystemNode`` from a validated ``AthenaVocabularySeed``."""
        return cls(
            vocabulary_id=seed.vocabulary_id,
            system_uri=resolve_code_system_uri(seed.vocabulary_id),
            name=seed.vocabulary_name,
            version=seed.vocabulary_version,
            reference=seed.vocabulary_reference,
            athena_concept_id=seed.vocabulary_concept_id,
            test_id=test_id,
            **audit_kwargs,
        )


class ConceptNode(ClinicalAuditableNode):
    """Auditable graph node representing a clinical terminology ``:Concept``."""

    __label__: ClassVar[str] = "Concept"

    athena_concept_id: Optional[int] = Field(
        default=None,
        description="OHDSI Athena numeric concept_id",
    )
    system_uri: str = Field(default="", description="Canonical FHIR CodeSystem URI")
    vocabulary_id: str = Field(..., min_length=1, description="Vocabulary identifier (e.g. 'SNOMED', 'LOINC')")
    code: str = Field(..., min_length=1, description="Canonical code within the vocabulary")
    display: str = Field(..., min_length=1, description="Preferred concept display term")
    synonyms: List[str] = Field(default_factory=list, description="Deduplicated concept synonyms")
    definition: str = Field(default="", description="Optional long-form narrative definition")
    domain_id: str = Field(..., min_length=1, description="Clinical domain (Condition, Measurement, Procedure, etc.)")
    concept_class_id: str = Field(default="", description="Vocabulary-specific concept class")
    standard_concept: Optional[str] = Field(
        default=None,
        description="'S' (Standard), 'C' (Classification), or None",
    )
    is_deprecated: bool = Field(default=False, description="True if concept carries invalid_reason ('D' or 'U')")
    invalid_reason: Optional[str] = Field(default=None, description="Athena invalid_reason ('D', 'U', or None)")
    valid_start_date: str = Field(default="", description="ISO validity start date (YYYY-MM-DD)")
    valid_end_date: str = Field(default="", description="ISO validity end date (YYYY-MM-DD)")
    concept_category: Optional[str] = Field(
        default=None,
        description="Optional Athena CONCEPT_METADATA category flag ('A', 'M', 'V')",
    )
    coding_key: str = Field(
        default="",
        description="Precomputed canonical coding key: '<system_uri>|<code>'",
    )
    coding_macro: str = Field(
        default="",
        description="Precomputed gosmart-graphite Coding Macro: '<system_uri>|<code>|<display>'",
    )

    @field_validator("vocabulary_id", "code", "display", "domain_id", mode="before")
    @classmethod
    def _validate_required_str(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Field must be a non-empty string.")
        return v.strip()

    @field_validator("system_uri", "definition", "concept_class_id", mode="before")
    @classmethod
    def _strip_optional_str(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()

    @field_validator("standard_concept", "invalid_reason", "concept_category", mode="before")
    @classmethod
    def _normalize_nullable_str(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        cleaned = str(v).strip()
        return cleaned if cleaned else None

    @field_validator("valid_start_date", "valid_end_date", mode="before")
    @classmethod
    def _normalize_date(cls, v: Any) -> str:
        return parse_athena_date(v)

    @field_validator("synonyms", mode="before")
    @classmethod
    def _normalize_synonyms(cls, v: Any) -> List[str]:
        if v is None:
            return []
        seen: set[str] = set()
        result: List[str] = []
        for item in v:
            cleaned = str(item).strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
        return result

    @model_validator(mode="before")
    @classmethod
    def _populate_system_uri(cls, data: Any) -> Any:
        if isinstance(data, dict):
            uri = data.get("system_uri")
            vocab = data.get("vocabulary_id")
            if (not uri or not str(uri).strip()) and isinstance(vocab, str) and vocab.strip():
                data = dict(data)
                data["system_uri"] = resolve_code_system_uri(vocab)
        return data

    @model_validator(mode="after")
    def _populate_computed_fields(self) -> "ConceptNode":
        if self.invalid_reason and not self.is_deprecated:
            object.__setattr__(self, "is_deprecated", True)
        if not self.coding_key:
            object.__setattr__(self, "coding_key", f"{self.system_uri}|{self.code}")
        if not self.coding_macro:
            object.__setattr__(self, "coding_macro", f"{self.system_uri}|{self.code}|{self.display}")
        return self

    def get_identity_subset(self) -> Dict[str, Any]:
        subset: Dict[str, Any] = {
            "system": self.system_uri,
            "code": self.code,
        }
        if self.test_id:
            subset["test_id"] = self.test_id
        return subset

    @classmethod
    def from_seed(
        cls,
        seed: AthenaConceptSeed,
        *,
        synonyms: Optional[List[str]] = None,
        concept_category: Optional[str] = None,
        test_id: Optional[str] = None,
        **audit_kwargs: Any,
    ) -> "ConceptNode":
        """Build a ``ConceptNode`` from a validated ``AthenaConceptSeed``."""
        code = seed.concept_code.strip()
        # Guard against OMOP metadata vocabularies that reuse "OMOP generated" for hundreds of rows
        if code == "OMOP generated":
            code = f"OMOP:{seed.concept_id}"
        return cls(
            athena_concept_id=seed.concept_id,
            system_uri=resolve_code_system_uri(seed.vocabulary_id),
            vocabulary_id=seed.vocabulary_id,
            code=code,
            display=seed.concept_name,
            synonyms=synonyms or [],
            domain_id=seed.domain_id,
            concept_class_id=seed.concept_class_id,
            standard_concept=seed.standard_concept,
            is_deprecated=bool(seed.invalid_reason),
            invalid_reason=seed.invalid_reason,
            valid_start_date=seed.valid_start_date,
            valid_end_date=seed.valid_end_date,
            concept_category=concept_category,
            test_id=test_id,
            **audit_kwargs,
        )


__all__ = [
    "compute_code_system_id",
    "compute_concept_id",
    "ClinicalAuditableNode",
    "CodeSystemNode",
    "ConceptNode",
]
