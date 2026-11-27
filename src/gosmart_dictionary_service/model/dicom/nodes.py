"""Auditable Neo4j graph node models for the DICOM Dictionary subgraph."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar, Dict, Optional

from pydantic import Field, field_validator, model_validator

from gosmart_dictionary_service.model.auditable import AuditableNode
from gosmart_dictionary_service.model.dicom.constants import (
    DICOM_CIOD_SYSTEM_URI,
    DICOM_CONFIDENTIALITY_SYSTEM_URI,
    DICOM_MACRO_SYSTEM_URI,
    DICOM_MODULE_SYSTEM_URI,
    DICOM_REFERENCE_SYSTEM_URI,
    DICOM_SOP_SYSTEM_URI,
    DICOM_TAG_SYSTEM_URI,
    normalize_confidentiality_attribute_id,
    normalize_dicom_tag_hex,
    parse_yn_flag,
    strip_html_to_text,
)
from gosmart_dictionary_service.model.dicom.seed import (
    DicomAttributeSeed,
    DicomCiodSeed,
    DicomConfidentialityProfileAttributeSeed,
    DicomMacroSeed,
    DicomModuleSeed,
    DicomSopClassSeed,
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


class DicomAuditableNode(AuditableNode):
    """Shared base for DICOM dictionary graph nodes with Neo4j property serialization."""

    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Serialize node fields into a flat Neo4j-safe property dictionary."""
        return {k: _serialize_neo4j_value(v) for k, v in self.model_dump().items()}


class DicomElementNode(DicomAuditableNode):
    """Auditable graph node representing a DICOM PS3.6 Data Element (``:DicomElement``)."""

    __label__: ClassVar[str] = "DicomElement"

    system_uri: str = Field(
        default=DICOM_TAG_SYSTEM_URI,
        description="Canonical FHIR-style CodeSystem URI for DICOM Data Elements",
    )
    tag: str = Field(..., description="Formatted DICOM tag (e.g. '(0018,0050)')")
    tag_hex: str = Field(
        ...,
        description="Canonical 8-char uppercase hex code (e.g. '00180050', '002031XX')",
    )
    keyword: str = Field(default="", description="DICOM Data Element keyword (e.g. 'SliceThickness')")
    name: str = Field(default="", description="DICOM Data Element display name (e.g. 'Slice Thickness')")
    vr: str = Field(default="", description="Value Representation (e.g. 'DS', 'OB or OW', or empty if retired)")
    vm: str = Field(default="", description="Value Multiplicity (e.g. '1', '1-n')")
    is_retired: bool = Field(default=False, description="True if the Data Element is retired in DICOM PS3.6")
    description: str = Field(
        default="",
        description="Optional plain-text normative summary (HTML stripped)",
    )
    coding_key: str = Field(
        default="",
        description="Precomputed canonical coding key: '<system_uri>|<tag_hex>'",
    )
    coding_macro: str = Field(
        default="",
        description="Precomputed gosmart-graphite Coding Macro: '<system_uri>|<tag_hex>|<display>'",
    )

    @field_validator("tag_hex", mode="before")
    @classmethod
    def _normalize_tag_hex(cls, v: Any) -> str:
        return normalize_dicom_tag_hex(v)

    @field_validator("tag", "keyword", "name", "vr", "vm", mode="before")
    @classmethod
    def _strip_strings(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()

    @field_validator("is_retired", mode="before")
    @classmethod
    def _parse_is_retired(cls, v: Any) -> bool:
        return parse_yn_flag(v)

    @field_validator("description", mode="before")
    @classmethod
    def _clean_description(cls, v: Any) -> str:
        return strip_html_to_text(v)

    @model_validator(mode="after")
    def _populate_coding_fields(self) -> "DicomElementNode":
        if not self.coding_key:
            object.__setattr__(self, "coding_key", f"{self.system_uri}|{self.tag_hex}")
        if not self.coding_macro:
            display = self.name or self.keyword or self.tag
            object.__setattr__(self, "coding_macro", f"{self.system_uri}|{self.tag_hex}|{display}")
        return self

    def get_identity_subset(self) -> Dict[str, Any]:
        subset: Dict[str, Any] = {
            "system": self.system_uri,
            "code": self.tag_hex,
        }
        if self.test_id:
            subset["test_id"] = self.test_id
        return subset

    @classmethod
    def from_seed(
        cls,
        seed: DicomAttributeSeed,
        *,
        description: str = "",
        test_id: Optional[str] = None,
        **audit_kwargs: Any,
    ) -> "DicomElementNode":
        """Build a ``DicomElementNode`` from a validated ``DicomAttributeSeed``."""
        return cls(
            tag=seed.tag,
            tag_hex=seed.id,
            keyword=seed.keyword,
            name=seed.name,
            vr=seed.value_representation,
            vm=seed.value_multiplicity,
            is_retired=seed.retired,
            description=description,
            test_id=test_id,
            **audit_kwargs,
        )


class DicomCIODNode(DicomAuditableNode):
    """Auditable graph node representing a DICOM PS3.3 Composite IOD (``:DicomCIOD``)."""

    __label__: ClassVar[str] = "DicomCIOD"

    system_uri: str = Field(
        default=DICOM_CIOD_SYSTEM_URI,
        description="Canonical CodeSystem URI for DICOM CIOD definitions",
    )
    ciod_id: str = Field(..., description="Innolitics slug identifier (e.g. 'ct-image')")
    name: str = Field(..., description="Composite IOD display name (e.g. 'CT Image')")
    description: str = Field(default="", description="Plain-text CIOD description (HTML stripped)")
    link_to_standard: str = Field(default="", description="URL to DICOM PS3.3 CIOD table")

    @field_validator("ciod_id", "name", mode="before")
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
        subset: Dict[str, Any] = {
            "system": self.system_uri,
            "code": self.ciod_id,
        }
        if self.test_id:
            subset["test_id"] = self.test_id
        return subset

    @classmethod
    def from_seed(
        cls,
        seed: DicomCiodSeed,
        *,
        test_id: Optional[str] = None,
        **audit_kwargs: Any,
    ) -> "DicomCIODNode":
        """Build a ``DicomCIODNode`` from a validated ``DicomCiodSeed``."""
        return cls(
            ciod_id=seed.id,
            name=seed.name,
            description=seed.description,
            link_to_standard=seed.link_to_standard,
            test_id=test_id,
            **audit_kwargs,
        )


class DicomModuleNode(DicomAuditableNode):
    """Auditable graph node representing a DICOM PS3.3 Module (``:DicomModule``)."""

    __label__: ClassVar[str] = "DicomModule"

    system_uri: str = Field(
        default=DICOM_MODULE_SYSTEM_URI,
        description="Canonical CodeSystem URI for DICOM Module definitions",
    )
    module_id: str = Field(..., description="Innolitics slug identifier (e.g. 'acquisition-context')")
    name: str = Field(..., description="Module display name (e.g. 'Acquisition Context')")
    description: str = Field(default="", description="Plain-text Module description (HTML stripped)")
    link_to_standard: str = Field(default="", description="URL to DICOM PS3.3 Module table")

    @field_validator("module_id", "name", mode="before")
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
        subset: Dict[str, Any] = {
            "system": self.system_uri,
            "code": self.module_id,
        }
        if self.test_id:
            subset["test_id"] = self.test_id
        return subset

    @classmethod
    def from_seed(
        cls,
        seed: DicomModuleSeed,
        *,
        test_id: Optional[str] = None,
        **audit_kwargs: Any,
    ) -> "DicomModuleNode":
        """Build a ``DicomModuleNode`` from a validated ``DicomModuleSeed``."""
        return cls(
            module_id=seed.id,
            name=seed.name,
            description=seed.description,
            link_to_standard=seed.link_to_standard,
            test_id=test_id,
            **audit_kwargs,
        )


class DicomMacroNode(DicomAuditableNode):
    """Auditable graph node representing a DICOM PS3.3 Macro (``:DicomMacro``)."""

    __label__: ClassVar[str] = "DicomMacro"

    system_uri: str = Field(
        default=DICOM_MACRO_SYSTEM_URI,
        description="Canonical CodeSystem URI for DICOM Macro definitions",
    )
    macro_id: str = Field(..., description="Innolitics slug identifier (e.g. 'basic-code-sequence')")
    name: str = Field(..., description="Macro display name (e.g. 'Basic Code Sequence')")
    description: str = Field(default="", description="Plain-text Macro description (HTML stripped)")
    link_to_standard: str = Field(default="", description="URL to DICOM PS3.3 Macro table")

    @field_validator("macro_id", "name", mode="before")
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
        subset: Dict[str, Any] = {
            "system": self.system_uri,
            "code": self.macro_id,
        }
        if self.test_id:
            subset["test_id"] = self.test_id
        return subset

    @classmethod
    def from_seed(
        cls,
        seed: DicomMacroSeed,
        *,
        test_id: Optional[str] = None,
        **audit_kwargs: Any,
    ) -> "DicomMacroNode":
        """Build a ``DicomMacroNode`` from a validated ``DicomMacroSeed``."""
        return cls(
            macro_id=seed.id,
            name=seed.name,
            description=seed.description,
            link_to_standard=seed.link_to_standard,
            test_id=test_id,
            **audit_kwargs,
        )


class DicomSOPClassNode(DicomAuditableNode):
    """Auditable graph node representing a DICOM SOP Class (``:DicomSOPClass``)."""

    __label__: ClassVar[str] = "DicomSOPClass"

    system_uri: str = Field(
        default=DICOM_SOP_SYSTEM_URI,
        description="Canonical CodeSystem URI for DICOM SOP Classes",
    )
    sop_class_uid: str = Field(..., description="DICOM SOP Class UID (e.g. '1.2.840.10008.5.1.4.1.1.1')")
    name: str = Field(..., description="SOP Class display name")
    ciod_name: str = Field(..., description="Name of the Composite IOD implemented by this SOP Class")
    modality: Optional[str] = Field(
        default=None,
        description="Optional modality code enrichment (e.g. 'CT', 'MR', 'CR')",
    )

    @field_validator("sop_class_uid", "name", "ciod_name", mode="before")
    @classmethod
    def _validate_required_str(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Field must be a non-empty string.")
        return v.strip()

    @field_validator("modality", mode="before")
    @classmethod
    def _normalize_modality(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        stripped = str(v).strip()
        return stripped if stripped else None

    def get_identity_subset(self) -> Dict[str, Any]:
        subset: Dict[str, Any] = {
            "system": self.system_uri,
            "code": self.sop_class_uid,
        }
        if self.test_id:
            subset["test_id"] = self.test_id
        return subset

    @classmethod
    def from_seed(
        cls,
        seed: DicomSopClassSeed,
        *,
        modality: Optional[str] = None,
        test_id: Optional[str] = None,
        **audit_kwargs: Any,
    ) -> "DicomSOPClassNode":
        """Build a ``DicomSOPClassNode`` from a validated ``DicomSopClassSeed``."""
        return cls(
            sop_class_uid=seed.id,
            name=seed.name,
            ciod_name=seed.ciod,
            modality=modality,
            test_id=test_id,
            **audit_kwargs,
        )


class DicomConfidentialityProfileNode(DicomAuditableNode):
    """Auditable graph node for a DICOM PS3.15 Confidentiality Profile rule (``:DicomConfidentialityProfile``)."""

    __label__: ClassVar[str] = "DicomConfidentialityProfile"

    system_uri: str = Field(
        default=DICOM_CONFIDENTIALITY_SYSTEM_URI,
        description="Canonical CodeSystem URI for DICOM PS3.15 Confidentiality Profile entries",
    )
    attribute_id: str = Field(
        ...,
        description=(
            "Normalized uppercase 8-char hex tag ID (e.g. '00080050', '50XXXXXX') "
            "or descriptive wildcard identifier (e.g. 'ggggeeee-where-gggg-is-odd')"
        ),
    )
    tag: str = Field(..., description="Formatted DICOM tag or wildcard expression")
    name: str = Field(..., description="Attribute name in DICOM PS3.15 Table E.1-1")
    std_comp_iod: bool = Field(..., description="True when the attribute appears in a Standard Composite IOD")
    basic_profile: str = Field(
        ...,
        description="Basic Application Level Confidentiality Profile action code (e.g. 'X', 'Z', 'D', 'U', 'X/Z/U*')",
    )
    clean_desc_opt: Optional[str] = Field(default=None, description="Clean Descriptors Option action code")
    clean_graph_opt: Optional[str] = Field(default=None, description="Clean Graphics Option action code")
    clean_struct_cont_opt: Optional[str] = Field(
        default=None,
        description="Clean Structured Content Option action code",
    )
    rtn_dev_id_opt: Optional[str] = Field(default=None, description="Retain Device Identity Option action code")
    rtn_inst_id_opt: Optional[str] = Field(default=None, description="Retain Institution Identity Option action code")
    rtn_long_full_dates_opt: Optional[str] = Field(
        default=None,
        description="Retain Longitudinal Temporal Information Full Dates Option action code",
    )
    rtn_long_modif_dates_opt: Optional[str] = Field(
        default=None,
        description="Retain Longitudinal Temporal Information Modified Dates Option action code",
    )
    rtn_pat_chars_opt: Optional[str] = Field(
        default=None,
        description="Retain Patient Characteristics Option action code",
    )
    rtn_safe_priv_opt: Optional[str] = Field(default=None, description="Retain Safe Private Option action code")
    rtn_uids_opt: Optional[str] = Field(default=None, description="Retain UIDs Option action code")

    @field_validator("attribute_id", mode="before")
    @classmethod
    def _normalize_attribute_id(cls, v: Any) -> str:
        return normalize_confidentiality_attribute_id(v)

    @field_validator("tag", "name", "basic_profile", mode="before")
    @classmethod
    def _validate_required_str(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Field must be a non-empty string.")
        return v.strip()

    @field_validator("std_comp_iod", mode="before")
    @classmethod
    def _parse_std_comp_iod(cls, v: Any) -> bool:
        return parse_yn_flag(v)

    def get_identity_subset(self) -> Dict[str, Any]:
        subset: Dict[str, Any] = {
            "system": self.system_uri,
            "code": self.attribute_id,
        }
        if self.test_id:
            subset["test_id"] = self.test_id
        return subset

    @classmethod
    def from_seed(
        cls,
        seed: DicomConfidentialityProfileAttributeSeed,
        *,
        test_id: Optional[str] = None,
        **audit_kwargs: Any,
    ) -> "DicomConfidentialityProfileNode":
        """Build a ``DicomConfidentialityProfileNode`` from a validated seed DTO."""
        return cls(
            attribute_id=seed.id,
            tag=seed.tag,
            name=seed.name,
            std_comp_iod=seed.std_comp_iod,
            basic_profile=seed.basic_profile,
            clean_desc_opt=seed.clean_desc_opt,
            clean_graph_opt=seed.clean_graph_opt,
            clean_struct_cont_opt=seed.clean_struct_cont_opt,
            rtn_dev_id_opt=seed.rtn_dev_id_opt,
            rtn_inst_id_opt=seed.rtn_inst_id_opt,
            rtn_long_full_dates_opt=seed.rtn_long_full_dates_opt,
            rtn_long_modif_dates_opt=seed.rtn_long_modif_dates_opt,
            rtn_pat_chars_opt=seed.rtn_pat_chars_opt,
            rtn_safe_priv_opt=seed.rtn_safe_priv_opt,
            rtn_uids_opt=seed.rtn_uids_opt,
            test_id=test_id,
            **audit_kwargs,
        )


class DicomStandardReferenceNode(DicomAuditableNode):
    """Auditable graph node representing a DICOM standard reference entry (``:DicomStandardReference``)."""

    __label__: ClassVar[str] = "DicomStandardReference"

    system_uri: str = Field(
        default=DICOM_REFERENCE_SYSTEM_URI,
        description="Canonical CodeSystem URI for DICOM standard reference entries",
    )
    source_url: str = Field(..., description="Canonical DICOM standard section/table/bibliography URL")
    content: str = Field(default="", description="Plain-text reference content (HTML stripped)")

    @field_validator("source_url", mode="before")
    @classmethod
    def _validate_source_url(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("source_url must be a non-empty string.")
        return v.strip()

    @field_validator("content", mode="before")
    @classmethod
    def _clean_content(cls, v: Any) -> str:
        return strip_html_to_text(v)

    def get_identity_subset(self) -> Dict[str, Any]:
        subset: Dict[str, Any] = {
            "system": self.system_uri,
            "code": self.source_url,
        }
        if self.test_id:
            subset["test_id"] = self.test_id
        return subset

    @classmethod
    def from_seed(
        cls,
        source_url: str,
        raw_content: str,
        *,
        test_id: Optional[str] = None,
        **audit_kwargs: Any,
    ) -> "DicomStandardReferenceNode":
        """Build a ``DicomStandardReferenceNode`` from a ``references.json`` entry."""
        return cls(
            source_url=source_url,
            content=raw_content,
            test_id=test_id,
            **audit_kwargs,
        )


__all__ = [
    "DicomAuditableNode",
    "DicomElementNode",
    "DicomCIODNode",
    "DicomModuleNode",
    "DicomMacroNode",
    "DicomSOPClassNode",
    "DicomConfidentialityProfileNode",
    "DicomStandardReferenceNode",
]
