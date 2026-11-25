"""1:1 Pydantic v2 seed DTOs mirroring Innolitics dicom-standard JSON files."""

from __future__ import annotations

from typing import ItemsView, Iterator, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, RootModel

from gosmart_dictionary_service.model.dicom.constants import (
    DicomAttributeRequirementType,
    DicomFuncGroupModuleType,
    DicomModuleUsage,
)


class _DicomSeedBase(BaseModel):
    """Base configuration shared by all Innolitics DICOM JSON seed DTOs."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class DicomExternalReferenceSeed(_DicomSeedBase):
    """External reference entry embedded in module/macro attribute definitions."""

    source_url: str = Field(..., alias="sourceUrl", description="URL to the referenced DICOM standard section")
    title: str = Field(..., description="Human-readable reference title (e.g. 'Section 8.1')")


class DicomAttributeSeed(_DicomSeedBase):
    """Seed DTO for a single row in ``attributes.json`` (DICOM PS3.6 Data Elements)."""

    id: str = Field(..., description="8-char hex/repeating-group tag identifier (e.g. '00080005', '002031xx')")
    tag: str = Field(..., description="Formatted DICOM tag (e.g. '(0008,0005)')")
    name: str = Field(default="", description="Data Element Name (may be empty for some retired elements)")
    keyword: str = Field(default="", description="Data Element Keyword (may be empty for some retired elements)")
    value_representation: str = Field(
        default="",
        alias="valueRepresentation",
        description="Value Representation (e.g. 'DS', 'OB or OW', or empty for retired delimiters)",
    )
    value_multiplicity: str = Field(
        default="",
        alias="valueMultiplicity",
        description="Value Multiplicity (e.g. '1', '1-n', or empty for some retired elements)",
    )
    retired: Literal["Y", "N"] = Field(..., description="Whether the element is retired ('Y' or 'N')")


class DicomCiodSeed(_DicomSeedBase):
    """Seed DTO for a single row in ``ciods.json`` (Composite IOD definitions)."""

    id: str = Field(..., description="Slug identifier of the CIOD (e.g. 'ct-image')")
    name: str = Field(..., description="Display name of the CIOD (e.g. 'CT Image')")
    description: str = Field(..., description="Source description string (HTML in Innolitics JSON)")
    link_to_standard: str = Field(..., alias="linkToStandard", description="Link to the DICOM PS3.3 table/section")


class DicomModuleSeed(_DicomSeedBase):
    """Seed DTO for a single row in ``modules.json`` (DICOM PS3.3 Module definitions)."""

    id: str = Field(..., description="Slug identifier of the Module (e.g. 'acquisition-context')")
    name: str = Field(..., description="Display name of the Module (e.g. 'Acquisition Context')")
    description: str = Field(..., description="Source description string (HTML in Innolitics JSON)")
    link_to_standard: str = Field(..., alias="linkToStandard", description="Link to the DICOM PS3.3 table/section")


class DicomMacroSeed(_DicomSeedBase):
    """Seed DTO for a single row in ``macros.json`` (DICOM PS3.3 Macro definitions)."""

    id: str = Field(..., description="Slug identifier of the Macro (e.g. 'basic-code-sequence')")
    name: str = Field(..., description="Display name of the Macro (e.g. 'Basic Code Sequence')")
    description: str = Field(..., description="Source description string (HTML in Innolitics JSON)")
    link_to_standard: str = Field(..., alias="linkToStandard", description="Link to the DICOM PS3.3 table/section")


class DicomSopClassSeed(_DicomSeedBase):
    """Seed DTO for a single row in ``sops.json`` (SOP Class UID to CIOD mapping)."""

    id: str = Field(..., description="SOP Class UID (e.g. '1.2.840.10008.5.1.4.1.1.1')")
    name: str = Field(..., description="SOP Class Name (e.g. 'Computed Radiography Image Storage')")
    ciod: str = Field(..., description="Name of the corresponding CIOD in ciods.json")


class DicomCiodToModuleSeed(_DicomSeedBase):
    """Seed DTO for a single row in ``ciod_to_modules.json``."""

    ciod_id: str = Field(..., alias="ciodId", description="Slug ID of the CIOD")
    module_id: str = Field(..., alias="moduleId", description="Slug ID of the Module")
    usage: DicomModuleUsage = Field(..., description="Usage requirement: 'M', 'C', or 'U'")
    conditional_statement: Optional[str] = Field(
        default=None,
        alias="conditionalStatement",
        description="Optional conditional inclusion statement when usage is 'C'",
    )
    information_entity: str = Field(
        ...,
        alias="informationEntity",
        description="Information Entity category (e.g. 'Patient', 'Study', 'Series', 'Image')",
    )


class DicomCiodToFuncGroupMacroSeed(_DicomSeedBase):
    """Seed DTO for a single row in ``ciod_to_func_group_macros.json``."""

    ciod_id: str = Field(..., alias="ciodId", description="Slug ID of the CIOD")
    macro_id: str = Field(..., alias="macroId", description="Slug ID of the Functional Group Macro")
    usage: DicomModuleUsage = Field(..., description="Usage requirement: 'M', 'C', or 'U'")
    conditional_statement: Optional[str] = Field(
        default=None,
        alias="conditionalStatement",
        description="Optional conditional inclusion statement when usage is 'C'",
    )
    module_type: DicomFuncGroupModuleType = Field(
        ...,
        alias="moduleType",
        description="Functional group module type ('Multi-frame' or 'Current Frame')",
    )


class DicomModuleToAttributeSeed(_DicomSeedBase):
    """Seed DTO for a single row in ``module_to_attributes.json``."""

    module_id: str = Field(..., alias="moduleId", description="Slug ID of the parent Module")
    path: str = Field(..., description="Unique colon-delimited attribute path within the Module")
    tag: str = Field(..., description="Formatted DICOM tag (e.g. '(0040,0555)')")
    type: DicomAttributeRequirementType = Field(
        ...,
        description="Attribute requirement type ('1', '1C', '2', '2C', '3', or 'None')",
    )
    link_to_standard: str = Field(..., alias="linkToStandard", description="Link to the DICOM PS3.3 table/section")
    description: str = Field(..., description="Contextual attribute description (HTML in Innolitics JSON)")
    external_references: list[DicomExternalReferenceSeed] = Field(
        default_factory=list,
        alias="externalReferences",
        description="External DICOM standard sections referenced in this attribute description",
    )


class DicomMacroToAttributeSeed(_DicomSeedBase):
    """Seed DTO for a single row in ``macro_to_attributes.json``."""

    macro_id: str = Field(..., alias="macroId", description="Slug ID of the parent Macro")
    path: str = Field(..., description="Unique colon-delimited attribute path within the Macro")
    tag: str = Field(..., description="Formatted DICOM tag (e.g. '(0008,0100)')")
    type: DicomAttributeRequirementType = Field(
        ...,
        description="Attribute requirement type ('1', '1C', '2', '2C', '3', or 'None')",
    )
    link_to_standard: str = Field(..., alias="linkToStandard", description="Link to the DICOM PS3.3 table/section")
    description: str = Field(..., description="Contextual attribute description (HTML in Innolitics JSON)")
    external_references: list[DicomExternalReferenceSeed] = Field(
        default_factory=list,
        alias="externalReferences",
        description="External DICOM standard sections referenced in this attribute description",
    )


class DicomConfidentialityProfileAttributeSeed(_DicomSeedBase):
    """Seed DTO for a single row in ``confidentiality_profile_attributes.json`` (DICOM PS3.15)."""

    id: str = Field(
        ...,
        description="Hex tag ID or descriptive wildcard identifier (e.g. '00080050', 'ggggeeee-where-gggg-is-odd')",
    )
    tag: str = Field(..., description="Formatted DICOM tag or wildcard expression")
    name: str = Field(..., description="Attribute name in PS3.15 Table E.1-1")
    std_comp_iod: Literal["Y", "N"] = Field(
        ...,
        alias="stdCompIOD",
        description="Whether attribute is in a Standard Composite IOD ('Y' or 'N')",
    )
    basic_profile: str = Field(
        ...,
        alias="basicProfile",
        description="Basic Application Level Confidentiality Profile action code (e.g. 'X', 'Z', 'D', 'U', 'X/Z/U*')",
    )
    clean_desc_opt: Optional[str] = Field(default=None, alias="cleanDescOpt")
    clean_graph_opt: Optional[str] = Field(default=None, alias="cleanGraphOpt")
    clean_struct_cont_opt: Optional[str] = Field(default=None, alias="cleanStructContOpt")
    rtn_dev_id_opt: Optional[str] = Field(default=None, alias="rtnDevIdOpt")
    rtn_inst_id_opt: Optional[str] = Field(default=None, alias="rtnInstIdOpt")
    rtn_long_full_dates_opt: Optional[str] = Field(default=None, alias="rtnLongFullDatesOpt")
    rtn_long_modif_dates_opt: Optional[str] = Field(default=None, alias="rtnLongModifDatesOpt")
    rtn_pat_chars_opt: Optional[str] = Field(default=None, alias="rtnPatCharsOpt")
    rtn_safe_priv_opt: Optional[str] = Field(default=None, alias="rtnSafePrivOpt")
    rtn_uids_opt: Optional[str] = Field(default=None, alias="rtnUIDsOpt")


class DicomReferencesSeed(RootModel[dict[str, str]]):
    """Seed DTO for ``references.json`` mapping standard URLs to raw reference snippets."""

    def __len__(self) -> int:
        return len(self.root)

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        return iter(self.root)

    def __getitem__(self, url: str) -> str:
        return self.root[url]

    def items(self) -> ItemsView[str, str]:
        return self.root.items()


__all__ = [
    "DicomExternalReferenceSeed",
    "DicomAttributeSeed",
    "DicomCiodSeed",
    "DicomModuleSeed",
    "DicomMacroSeed",
    "DicomSopClassSeed",
    "DicomCiodToModuleSeed",
    "DicomCiodToFuncGroupMacroSeed",
    "DicomModuleToAttributeSeed",
    "DicomMacroToAttributeSeed",
    "DicomConfidentialityProfileAttributeSeed",
    "DicomReferencesSeed",
]
