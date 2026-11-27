"""Auditable Neo4j graph relationship models for the DICOM Dictionary subgraph."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar, Dict, Optional

from pydantic import Field, field_validator

from gosmart_dictionary_service.model.auditable import AuditableRelation
from gosmart_dictionary_service.model.dicom.constants import (
    DicomAttributeRequirementType,
    DicomFuncGroupModuleType,
    DicomModuleUsage,
    strip_html_to_text,
)
from gosmart_dictionary_service.model.dicom.nodes import (
    DicomCIODNode,
    DicomConfidentialityProfileNode,
    DicomElementNode,
    DicomSOPClassNode,
    DicomStandardReferenceNode,
)
from gosmart_dictionary_service.model.dicom.seed import (
    DicomCiodToFuncGroupMacroSeed,
    DicomCiodToModuleSeed,
    DicomExternalReferenceSeed,
    DicomMacroToAttributeSeed,
    DicomModuleToAttributeSeed,
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


class DicomAuditableRelation(AuditableRelation):
    """Shared base for DICOM dictionary graph relationships with Neo4j property serialization."""

    def to_neo4j_properties(self, *, include_endpoints: bool = False) -> Dict[str, Any]:
        """Serialize relationship fields into a Neo4j-safe property dictionary.

        By default excludes ``start_node_id`` and ``end_node_id`` because Neo4j
        stores endpoints structurally on the relationship rather than as properties.
        """
        exclude = None if include_endpoints else {"start_node_id", "end_node_id"}
        data = self.model_dump(exclude=exclude)
        return {k: _serialize_neo4j_value(v) for k, v in data.items()}


class ImplementsCiodRelation(DicomAuditableRelation):
    """Relationship ``(:DicomSOPClass)-[:IMPLEMENTS_CIOD]->(:DicomCIOD)``."""

    __type__: ClassVar[str] = "IMPLEMENTS_CIOD"

    @classmethod
    def from_nodes(
        cls,
        sop_class: DicomSOPClassNode,
        ciod: DicomCIODNode,
        *,
        test_id: Optional[str] = None,
        **audit_kwargs: Any,
    ) -> "ImplementsCiodRelation":
        """Create an ``IMPLEMENTS_CIOD`` edge linking a SOP Class node to its CIOD node."""
        return cls(
            start_node_id=sop_class.id,
            end_node_id=ciod.id,
            test_id=test_id,
            **audit_kwargs,
        )


class HasModuleRelation(DicomAuditableRelation):
    """Relationship ``(:DicomCIOD)-[:HAS_MODULE]->(:DicomModule)``."""

    __type__: ClassVar[str] = "HAS_MODULE"

    usage: DicomModuleUsage = Field(..., description="Module usage within the CIOD: 'M', 'C', or 'U'")
    conditional_statement: Optional[str] = Field(
        default=None,
        description="Optional conditional inclusion statement when usage is 'C'",
    )
    information_entity: str = Field(
        ...,
        description="Information Entity category (e.g. 'Patient', 'Study', 'Series', 'Image')",
    )

    @field_validator("information_entity", mode="before")
    @classmethod
    def _validate_ie(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("information_entity must be a non-empty string.")
        return v.strip()

    @field_validator("conditional_statement", mode="before")
    @classmethod
    def _clean_conditional_statement(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        cleaned = strip_html_to_text(v)
        return cleaned if cleaned else None

    @classmethod
    def from_seed(
        cls,
        seed: DicomCiodToModuleSeed,
        *,
        start_node_id: str,
        end_node_id: str,
        test_id: Optional[str] = None,
        **audit_kwargs: Any,
    ) -> "HasModuleRelation":
        """Build a ``HasModuleRelation`` from a validated ``DicomCiodToModuleSeed``."""
        return cls(
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            usage=seed.usage,
            conditional_statement=seed.conditional_statement,
            information_entity=seed.information_entity,
            test_id=test_id,
            **audit_kwargs,
        )


class HasFuncGroupMacroRelation(DicomAuditableRelation):
    """Relationship ``(:DicomCIOD)-[:HAS_FUNC_GROUP_MACRO]->(:DicomMacro)``."""

    __type__: ClassVar[str] = "HAS_FUNC_GROUP_MACRO"

    usage: DicomModuleUsage = Field(..., description="Macro usage within the CIOD: 'M', 'C', or 'U'")
    conditional_statement: Optional[str] = Field(
        default=None,
        description="Optional conditional inclusion statement when usage is 'C'",
    )
    module_type: DicomFuncGroupModuleType = Field(
        ...,
        description="Functional Group module type ('Multi-frame' or 'Current Frame')",
    )

    @field_validator("conditional_statement", mode="before")
    @classmethod
    def _clean_conditional_statement(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        cleaned = strip_html_to_text(v)
        return cleaned if cleaned else None

    @classmethod
    def from_seed(
        cls,
        seed: DicomCiodToFuncGroupMacroSeed,
        *,
        start_node_id: str,
        end_node_id: str,
        test_id: Optional[str] = None,
        **audit_kwargs: Any,
    ) -> "HasFuncGroupMacroRelation":
        """Build a ``HasFuncGroupMacroRelation`` from a validated ``DicomCiodToFuncGroupMacroSeed``."""
        return cls(
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            usage=seed.usage,
            conditional_statement=seed.conditional_statement,
            module_type=seed.module_type,
            test_id=test_id,
            **audit_kwargs,
        )


class UsesElementRelation(DicomAuditableRelation):
    """Relationship ``(:DicomModule | :DicomMacro)-[:USES_ELEMENT]->(:DicomElement)``.

    Because the same DICOM tag can appear multiple times inside a single Module or Macro
    at different sequence nesting levels (e.g. ``acquisition-context:00400555:00081199``),
    ``path`` is included in ``get_identity_subset()`` to guarantee deterministic uniqueness.
    """

    __type__: ClassVar[str] = "USES_ELEMENT"

    path: str = Field(..., description="Unique colon-delimited attribute path within the Module or Macro")
    tag: str = Field(..., description="Formatted DICOM tag (e.g. '(0040,0555)')")
    requirement_type: DicomAttributeRequirementType = Field(
        ...,
        description="Requirement type within the Module/Macro ('1', '1C', '2', '2C', '3', or 'None')",
    )
    description: str = Field(
        default="",
        description="Contextual plain-text attribute description (HTML stripped)",
    )
    link_to_standard: str = Field(default="", description="URL to DICOM PS3.3 attribute table")
    external_reference_urls: list[str] = Field(
        default_factory=list,
        description="List of referenced DICOM standard URLs from externalReferences",
    )

    @field_validator("path", "tag", mode="before")
    @classmethod
    def _validate_required_str(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Field must be a non-empty string.")
        return v.strip()

    @field_validator("description", mode="before")
    @classmethod
    def _clean_description(cls, v: Any) -> str:
        return strip_html_to_text(v)

    @field_validator("link_to_standard", mode="before")
    @classmethod
    def _clean_link(cls, v: Any) -> str:
        return "" if v is None else str(v).strip()

    def get_identity_subset(self) -> Dict[str, Any]:
        subset = super().get_identity_subset()
        subset["path"] = self.path
        return subset

    @classmethod
    def from_seed(
        cls,
        seed: DicomModuleToAttributeSeed | DicomMacroToAttributeSeed,
        *,
        start_node_id: str,
        end_node_id: str,
        test_id: Optional[str] = None,
        **audit_kwargs: Any,
    ) -> "UsesElementRelation":
        """Build a ``UsesElementRelation`` from a module- or macro-attribute seed DTO."""
        ref_urls = [ref.source_url for ref in seed.external_references if ref.source_url]
        return cls(
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            path=seed.path,
            tag=seed.tag,
            requirement_type=seed.type,
            description=seed.description,
            link_to_standard=seed.link_to_standard,
            external_reference_urls=ref_urls,
            test_id=test_id,
            **audit_kwargs,
        )


class HasConfidentialityProfileRelation(DicomAuditableRelation):
    """Relationship ``(:DicomElement)-[:HAS_CONFIDENTIALITY_PROFILE]->(:DicomConfidentialityProfile)``."""

    __type__: ClassVar[str] = "HAS_CONFIDENTIALITY_PROFILE"

    @classmethod
    def from_nodes(
        cls,
        element: DicomElementNode,
        profile: DicomConfidentialityProfileNode,
        *,
        test_id: Optional[str] = None,
        **audit_kwargs: Any,
    ) -> "HasConfidentialityProfileRelation":
        """Create a ``HAS_CONFIDENTIALITY_PROFILE`` edge from a Data Element to its PS3.15 profile."""
        return cls(
            start_node_id=element.id,
            end_node_id=profile.id,
            test_id=test_id,
            **audit_kwargs,
        )


class ReferencesStandardRelation(DicomAuditableRelation):
    """Relationship ``[:REFERENCES_STANDARD]`` linking a DICOM node to ``:DicomStandardReference``."""

    __type__: ClassVar[str] = "REFERENCES_STANDARD"

    source_url: str = Field(..., description="Referenced DICOM standard URL")
    title: str = Field(default="", description="Human-readable reference title (plain text)")
    path: Optional[str] = Field(
        default=None,
        description="Optional attribute path where the reference originated",
    )

    @field_validator("source_url", mode="before")
    @classmethod
    def _validate_source_url(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("source_url must be a non-empty string.")
        return v.strip()

    @field_validator("title", mode="before")
    @classmethod
    def _clean_title(cls, v: Any) -> str:
        return strip_html_to_text(v)

    @field_validator("path", mode="before")
    @classmethod
    def _normalize_path(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        stripped = str(v).strip()
        return stripped if stripped else None

    def get_identity_subset(self) -> Dict[str, Any]:
        subset = super().get_identity_subset()
        if self.path:
            subset["path"] = self.path
        return subset

    @classmethod
    def from_seed(
        cls,
        seed: DicomExternalReferenceSeed,
        *,
        start_node_id: str,
        reference: DicomStandardReferenceNode,
        path: Optional[str] = None,
        test_id: Optional[str] = None,
        **audit_kwargs: Any,
    ) -> "ReferencesStandardRelation":
        """Build a ``ReferencesStandardRelation`` from an external reference seed and target node."""
        return cls(
            start_node_id=start_node_id,
            end_node_id=reference.id,
            source_url=seed.source_url,
            title=seed.title,
            path=path,
            test_id=test_id,
            **audit_kwargs,
        )


__all__ = [
    "DicomAuditableRelation",
    "ImplementsCiodRelation",
    "HasModuleRelation",
    "HasFuncGroupMacroRelation",
    "UsesElementRelation",
    "HasConfidentialityProfileRelation",
    "ReferencesStandardRelation",
]
