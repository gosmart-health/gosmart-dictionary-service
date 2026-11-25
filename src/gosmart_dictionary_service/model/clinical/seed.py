"""1:1 Pydantic v2 seed DTOs mirroring OHDSI Athena OMOP CDM vocabulary CSV files."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _AthenaSeedBase(BaseModel):
    """Base configuration shared by all OHDSI Athena tab-delimited CSV seed DTOs."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class AthenaVocabularySeed(_AthenaSeedBase):
    """Seed DTO for a single row in ``VOCABULARY.csv``."""

    vocabulary_id: str = Field(..., min_length=1, description="OHDSI vocabulary identifier (e.g. 'SNOMED', 'LOINC')")
    vocabulary_name: str = Field(..., min_length=1, description="Human-readable vocabulary name")
    vocabulary_reference: str = Field(default="", description="Authoritative reference URL or citation")
    vocabulary_version: str = Field(default="", description="Release version string")
    vocabulary_concept_id: int = Field(..., description="Athena concept_id representing this vocabulary")


class AthenaDomainSeed(_AthenaSeedBase):
    """Seed DTO for a single row in ``DOMAIN.csv``."""

    domain_id: str = Field(..., min_length=1, description="OHDSI domain identifier (e.g. 'Condition', 'Measurement')")
    domain_name: str = Field(..., min_length=1, description="Human-readable domain name")
    domain_concept_id: int = Field(..., description="Athena concept_id representing this domain")


class AthenaConceptClassSeed(_AthenaSeedBase):
    """Seed DTO for a single row in ``CONCEPT_CLASS.csv``."""

    concept_class_id: str = Field(..., min_length=1, description="OHDSI concept class identifier")
    concept_class_name: str = Field(..., min_length=1, description="Human-readable concept class name")
    concept_class_concept_id: int = Field(..., description="Athena concept_id representing this class")


class AthenaRelationshipSeed(_AthenaSeedBase):
    """Seed DTO for a single row in ``RELATIONSHIP.csv``."""

    relationship_id: str = Field(
        ...,
        min_length=1,
        description="OHDSI relationship identifier (e.g. 'Is a', 'Maps to')",
    )
    relationship_name: str = Field(..., min_length=1, description="Human-readable relationship name")
    is_hierarchical: str = Field(default="0", description="'1' if relationship defines a hierarchy, else '0'")
    defines_ancestry: str = Field(default="0", description="'1' if relationship contributes to CONCEPT_ANCESTOR")
    reverse_relationship_id: str = Field(default="", description="Exact inverse relationship_id")
    relationship_concept_id: int = Field(..., description="Athena concept_id representing this relationship type")


class AthenaConceptSeed(_AthenaSeedBase):
    """Seed DTO for a single row in ``CONCEPT.csv``."""

    concept_id: int = Field(..., description="OHDSI Athena numeric concept_id")
    concept_name: str = Field(..., min_length=1, description="Preferred concept display name")
    domain_id: str = Field(..., min_length=1, description="Clinical domain (Condition, Measurement, Procedure, etc.)")
    vocabulary_id: str = Field(..., min_length=1, description="Vocabulary identifier (SNOMED, LOINC, ICD10CM, etc.)")
    concept_class_id: str = Field(..., min_length=1, description="Vocabulary-specific concept class")
    standard_concept: Optional[str] = Field(
        default=None,
        description="'S' (Standard), 'C' (Classification), or None (Non-standard)",
    )
    concept_code: str = Field(..., min_length=1, description="Native code within the source vocabulary")
    valid_start_date: str = Field(default="", description="Validity start date (YYYYMMDD or YYYY-MM-DD)")
    valid_end_date: str = Field(default="", description="Validity end date (YYYYMMDD or YYYY-MM-DD)")
    invalid_reason: Optional[str] = Field(
        default=None,
        description="None if active/valid, 'D' if deleted/deprecated, 'U' if upgraded/replaced",
    )

    @field_validator("standard_concept", "invalid_reason", mode="before")
    @classmethod
    def _empty_to_none(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        cleaned = str(v).strip()
        return cleaned if cleaned else None


class AthenaConceptSynonymSeed(_AthenaSeedBase):
    """Seed DTO for a single row in ``CONCEPT_SYNONYM.csv``."""

    concept_id: int = Field(..., description="Target Athena concept_id")
    concept_synonym_name: str = Field(..., min_length=1, description="Synonym text")
    language_concept_id: int = Field(..., description="Athena language concept_id (e.g. 4180186 for English)")


class AthenaConceptRelationshipSeed(_AthenaSeedBase):
    """Seed DTO for a single row in ``CONCEPT_RELATIONSHIP.csv``."""

    concept_id_1: int = Field(..., description="Source Athena concept_id")
    concept_id_2: int = Field(..., description="Target Athena concept_id")
    relationship_id: str = Field(..., min_length=1, description="Athena relationship_id")
    valid_start_date: str = Field(default="", description="Relationship validity start date")
    valid_end_date: str = Field(default="", description="Relationship validity end date")
    invalid_reason: Optional[str] = Field(default=None, description="None if active, 'D'/'U' if deprecated")

    @field_validator("invalid_reason", mode="before")
    @classmethod
    def _empty_to_none(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        cleaned = str(v).strip()
        return cleaned if cleaned else None


class AthenaConceptMetadataSeed(_AthenaSeedBase):
    """Seed DTO for a single row in ``CONCEPT_METADATA.csv``."""

    concept_id: int = Field(..., description="Target Athena concept_id")
    concept_category: str = Field(default="", description="Athena concept category flag (e.g. 'A', 'M', 'V')")
    reuse_status: str = Field(default="", description="Optional reuse status")


class AthenaConceptRelationshipMetadataSeed(_AthenaSeedBase):
    """Seed DTO for a single row in ``CONCEPT_RELATIONSHIP_METADATA.csv``."""

    concept_id_1: int = Field(..., description="Source Athena concept_id")
    concept_id_2: int = Field(..., description="Target Athena concept_id")
    relationship_id: str = Field(..., min_length=1, description="Relationship identifier (e.g. 'Maps to')")
    relationship_predicate_id: Optional[str] = Field(
        default=None,
        description="SKOS mapping predicate (e.g. 'exactMatch', 'broadMatch', 'narrowMatch')",
    )
    relationship_group: Optional[str] = Field(default=None, description="Optional relationship group")
    mapping_source: Optional[str] = Field(default=None, description="Mapping provenance source (e.g. 'OHDSI')")
    confidence: Optional[float] = Field(default=None, description="Mapping confidence score in [0, 1]")

    @field_validator("relationship_predicate_id", "relationship_group", "mapping_source", mode="before")
    @classmethod
    def _empty_to_none(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        cleaned = str(v).strip()
        return cleaned if cleaned else None

    @field_validator("confidence", mode="before")
    @classmethod
    def _parse_confidence(cls, v: object) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, str):
            cleaned = v.strip()
            if not cleaned:
                return None
            return float(cleaned)
        return float(v)  # type: ignore[arg-type]


__all__ = [
    "AthenaVocabularySeed",
    "AthenaDomainSeed",
    "AthenaConceptClassSeed",
    "AthenaRelationshipSeed",
    "AthenaConceptSeed",
    "AthenaConceptSynonymSeed",
    "AthenaConceptRelationshipSeed",
    "AthenaConceptMetadataSeed",
    "AthenaConceptRelationshipMetadataSeed",
]
