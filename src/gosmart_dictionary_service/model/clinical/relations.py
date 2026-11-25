"""Auditable Neo4j graph relationship models for the Clinical Terminology & Ontology subgraph."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar, Dict, Optional

from pydantic import Field, field_validator

from gosmart_dictionary_service.model.auditable import AuditableRelation, generate_deterministic_id
from gosmart_dictionary_service.model.clinical.constants import parse_athena_date


def _serialize_neo4j_value(value: Any) -> Any:
    """Convert Python model field values into Neo4j-compatible primitives."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, list):
        return [_serialize_neo4j_value(item) for item in value]
    return value


def compute_clinical_relation_id(
    start_node_id: str,
    rel_type: str,
    end_node_id: str,
    *,
    relationship_id: Optional[str] = None,
    test_id: Optional[str] = None,
) -> str:
    """Compute a deterministic UUIDv5 ID matching the clinical ``AuditableRelation`` subclasses."""
    subset: Dict[str, Any] = {
        "start_node_id": start_node_id,
        "type": rel_type,
        "end_node_id": end_node_id,
    }
    if relationship_id:
        subset["relationship_id"] = relationship_id
    if test_id and test_id.strip():
        subset["test_id"] = test_id.strip()
    return generate_deterministic_id(subset)


class ClinicalAuditableRelation(AuditableRelation):
    """Shared base for clinical ontology graph relationships with Neo4j property serialization."""

    def to_neo4j_properties(self, *, include_endpoints: bool = False) -> Dict[str, Any]:
        """Serialize relationship fields into a Neo4j-safe property dictionary."""
        exclude = None if include_endpoints else {"start_node_id", "end_node_id"}
        data = self.model_dump(exclude=exclude)
        return {k: _serialize_neo4j_value(v) for k, v in data.items()}


class BelongsToSystemRelation(ClinicalAuditableRelation):
    """Relationship ``(:Concept)-[:BELONGS_TO_SYSTEM]->(:CodeSystem)``."""

    __type__: ClassVar[str] = "BELONGS_TO_SYSTEM"


class IsARelation(ClinicalAuditableRelation):
    """Taxonomic subsumption relationship ``(:Concept)-[:IS_A]->(:Concept)`` (child -> parent)."""

    __type__: ClassVar[str] = "IS_A"

    valid_start_date: str = Field(default="", description="ISO validity start date")
    valid_end_date: str = Field(default="", description="ISO validity end date")

    @field_validator("valid_start_date", "valid_end_date", mode="before")
    @classmethod
    def _normalize_date(cls, v: Any) -> str:
        return parse_athena_date(v)


class MapsToRelation(ClinicalAuditableRelation):
    """Cross-vocabulary or standardization mapping ``(:Concept)-[:MAPS_TO]->(:Concept)``."""

    __type__: ClassVar[str] = "MAPS_TO"

    relationship_id: str = Field(default="Maps to", description="Athena relationship_id ('Maps to' or 'Maps to value')")
    equivalence: Optional[str] = Field(
        default=None,
        description=(
            "Optional SKOS predicate from CONCEPT_RELATIONSHIP_METADATA ('exactMatch', 'broadMatch', 'narrowMatch')"
        ),
    )
    confidence: Optional[float] = Field(default=None, description="Optional mapping confidence score")
    mapping_source: Optional[str] = Field(default=None, description="Optional mapping provenance source")
    valid_start_date: str = Field(default="", description="ISO validity start date")
    valid_end_date: str = Field(default="", description="ISO validity end date")

    @field_validator("valid_start_date", "valid_end_date", mode="before")
    @classmethod
    def _normalize_date(cls, v: Any) -> str:
        return parse_athena_date(v)

    @field_validator("equivalence", "mapping_source", mode="before")
    @classmethod
    def _normalize_nullable_str(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        cleaned = str(v).strip()
        return cleaned if cleaned else None

    def get_identity_subset(self) -> Dict[str, Any]:
        subset = super().get_identity_subset()
        subset["relationship_id"] = self.relationship_id
        return subset


class HasAttributeRelation(ClinicalAuditableRelation):
    """Clinical defining attribute relationship ``(:Concept)-[:HAS_ATTRIBUTE]->(:Concept)``."""

    __type__: ClassVar[str] = "HAS_ATTRIBUTE"

    relationship_id: str = Field(
        ...,
        min_length=1,
        description="Athena attribute relationship_id (e.g. 'Has finding site')",
    )
    relationship_name: str = Field(default="", description="Human-readable relationship name from RELATIONSHIP.csv")
    valid_start_date: str = Field(default="", description="ISO validity start date")
    valid_end_date: str = Field(default="", description="ISO validity end date")

    @field_validator("valid_start_date", "valid_end_date", mode="before")
    @classmethod
    def _normalize_date(cls, v: Any) -> str:
        return parse_athena_date(v)

    def get_identity_subset(self) -> Dict[str, Any]:
        subset = super().get_identity_subset()
        subset["relationship_id"] = self.relationship_id
        return subset


class ReplacedByRelation(ClinicalAuditableRelation):
    """Historical concept lifecycle link ``(:Concept)-[:REPLACED_BY]->(:Concept)``."""

    __type__: ClassVar[str] = "REPLACED_BY"

    relationship_id: str = Field(
        ...,
        min_length=1,
        description="Athena historical replacement relationship_id (e.g. 'Concept replaced by', 'Concept same_as to')",
    )
    valid_start_date: str = Field(default="", description="ISO validity start date")
    valid_end_date: str = Field(default="", description="ISO validity end date")

    @field_validator("valid_start_date", "valid_end_date", mode="before")
    @classmethod
    def _normalize_date(cls, v: Any) -> str:
        return parse_athena_date(v)

    def get_identity_subset(self) -> Dict[str, Any]:
        subset = super().get_identity_subset()
        subset["relationship_id"] = self.relationship_id
        return subset


__all__ = [
    "compute_clinical_relation_id",
    "ClinicalAuditableRelation",
    "BelongsToSystemRelation",
    "IsARelation",
    "MapsToRelation",
    "HasAttributeRelation",
    "ReplacedByRelation",
]
