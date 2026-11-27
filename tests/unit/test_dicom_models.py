"""Modular unit tests for DICOM seed DTOs, auditable graph nodes, and relations."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from gosmart_dictionary_service.model.auditable import (
    DICOM_GRAPH_NAMESPACE,
    DICTIONARY_GRAPH_NAMESPACE,
    generate_deterministic_id,
)
from gosmart_dictionary_service.model.dicom import (
    DICOM_CIOD_SYSTEM_URI,
    DICOM_CONFIDENTIALITY_SYSTEM_URI,
    DICOM_MACRO_SYSTEM_URI,
    DICOM_MODULE_SYSTEM_URI,
    DICOM_REFERENCE_SYSTEM_URI,
    DICOM_SOP_SYSTEM_URI,
    DICOM_TAG_SYSTEM_URI,
    DicomAttributeRequirementType,
    DicomAttributeSeed,
    DicomCIODNode,
    DicomCiodSeed,
    DicomCiodToFuncGroupMacroSeed,
    DicomCiodToModuleSeed,
    DicomConfidentialityProfileAttributeSeed,
    DicomConfidentialityProfileNode,
    DicomElementNode,
    DicomExternalReferenceSeed,
    DicomFuncGroupModuleType,
    DicomMacroNode,
    DicomMacroSeed,
    DicomMacroToAttributeSeed,
    DicomModuleNode,
    DicomModuleSeed,
    DicomModuleToAttributeSeed,
    DicomModuleUsage,
    DicomReferencesSeed,
    DicomSOPClassNode,
    DicomSopClassSeed,
    DicomStandardReferenceNode,
    HasConfidentialityProfileRelation,
    HasFuncGroupMacroRelation,
    HasModuleRelation,
    ImplementsCiodRelation,
    ReferencesStandardRelation,
    UsesElementRelation,
    normalize_confidentiality_attribute_id,
    normalize_dicom_tag_hex,
    parse_yn_flag,
    strip_html_to_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_SOURCE_DICOM_DIR = PROJECT_ROOT / "data-source" / "dicom"


# ---------------------------------------------------------------------------
# Namespace & normalization helpers
# ---------------------------------------------------------------------------


def test_dictionary_graph_namespace_matches_architecture() -> None:
    expected = uuid.uuid5(uuid.NAMESPACE_DNS, "gosmart.dictionary.graph")
    assert DICTIONARY_GRAPH_NAMESPACE == expected
    assert DICOM_GRAPH_NAMESPACE == DICTIONARY_GRAPH_NAMESPACE


def test_normalize_dicom_tag_hex_accepts_canonical_and_formatted_inputs() -> None:
    assert normalize_dicom_tag_hex("00180050") == "00180050"
    assert normalize_dicom_tag_hex("300a0782") == "300A0782"
    assert normalize_dicom_tag_hex("002031xx") == "002031XX"
    assert normalize_dicom_tag_hex("(0018,0050)") == "00180050"
    assert normalize_dicom_tag_hex(" 0010,0010 ") == "00100010"


@pytest.mark.parametrize("bad_value", ["", "   ", "1234", "0018ZZZZ", "ggggeeee-where-gggg-is-odd"])
def test_normalize_dicom_tag_hex_rejects_invalid_values(bad_value: str) -> None:
    with pytest.raises(ValueError):
        normalize_dicom_tag_hex(bad_value)


def test_normalize_confidentiality_attribute_id_handles_hex_and_wildcards() -> None:
    assert normalize_confidentiality_attribute_id("00080050") == "00080050"
    assert normalize_confidentiality_attribute_id("50xxxxxx") == "50XXXXXX"
    assert normalize_confidentiality_attribute_id("ggggeeee-where-gggg-is-odd") == "ggggeeee-where-gggg-is-odd"


def test_parse_yn_flag() -> None:
    assert parse_yn_flag("Y") is True
    assert parse_yn_flag("y") is True
    assert parse_yn_flag(True) is True
    assert parse_yn_flag("N") is False
    assert parse_yn_flag("n") is False
    assert parse_yn_flag(False) is False
    with pytest.raises(ValueError):
        parse_yn_flag("MAYBE")


def test_strip_html_to_text_removes_tags_unescapes_and_is_idempotent() -> None:
    raw_html = (
        "<p>\nThe identifier of the Coded Entry.</p>\n"
        '<p>\nSee <span href="">Section\xa08.1</span> &amp; <a href="http://example.org">Table 1</a>.</p>'
    )
    cleaned = strip_html_to_text(raw_html)
    assert cleaned == "The identifier of the Coded Entry.\nSee Section 8.1 & Table 1."
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert strip_html_to_text(cleaned) == cleaned
    assert strip_html_to_text("") == ""
    assert strip_html_to_text(None) == ""


# ---------------------------------------------------------------------------
# Seed DTO validation (1:1 Innolitics JSON)
# ---------------------------------------------------------------------------


def test_attribute_seed_handles_standard_multi_vr_and_retired_edge_cases() -> None:
    standard = DicomAttributeSeed.model_validate(
        {
            "tag": "(0018,0050)",
            "name": "Slice Thickness",
            "keyword": "SliceThickness",
            "valueRepresentation": "DS",
            "valueMultiplicity": "1",
            "retired": "N",
            "id": "00180050",
        }
    )
    assert standard.value_representation == "DS"
    assert standard.retired == "N"

    retired_empty = DicomAttributeSeed.model_validate(
        {
            "tag": "(0008,0202)",
            "name": "",
            "keyword": "",
            "valueRepresentation": "",
            "valueMultiplicity": "",
            "retired": "Y",
            "id": "00080202",
        }
    )
    assert retired_empty.keyword == ""
    assert retired_empty.value_representation == ""

    multi_vr = DicomAttributeSeed.model_validate(
        {
            "tag": "(7FE0,0010)",
            "name": "Pixel Data",
            "keyword": "PixelData",
            "valueRepresentation": "OB or OW",
            "valueMultiplicity": "1",
            "retired": "N",
            "id": "7fe00010",
        }
    )
    assert multi_vr.value_representation == "OB or OW"


def test_seed_dtos_reject_unexpected_extra_fields() -> None:
    with pytest.raises(ValidationError):
        DicomSopClassSeed.model_validate(
            {
                "id": "1.2.840.10008.5.1.4.1.1.2",
                "name": "CT Image Storage",
                "ciod": "CT Image",
                "unexpectedField": "boom",
            }
        )


def test_mapping_and_confidentiality_and_references_seed_dtos() -> None:
    c2m = DicomCiodToModuleSeed.model_validate(
        {
            "ciodId": "computed-radiography-image",
            "moduleId": "patient",
            "usage": "M",
            "conditionalStatement": None,
            "informationEntity": "Patient",
        }
    )
    assert c2m.usage == DicomModuleUsage.MANDATORY
    assert c2m.conditional_statement is None

    c2fg = DicomCiodToFuncGroupMacroSeed.model_validate(
        {
            "ciodId": "multi-frame-grayscale-byte-secondary-capture-image",
            "macroId": "pixel-measures",
            "usage": "C",
            "conditionalStatement": "Required if present",
            "moduleType": "Multi-frame",
        }
    )
    assert c2fg.module_type == DicomFuncGroupModuleType.MULTI_FRAME

    m2a = DicomModuleToAttributeSeed.model_validate(
        {
            "moduleId": "acquisition-context",
            "path": "acquisition-context:00400555:00081199",
            "tag": "(0008,1199)",
            "type": "None",
            "linkToStandard": "http://dicom.nema.org/example",
            "description": "<p>Sequence item</p>",
            "externalReferences": [{"sourceUrl": "http://dicom.nema.org/ref", "title": "Section 8.1"}],
        }
    )
    assert m2a.type == DicomAttributeRequirementType.NONE
    assert len(m2a.external_references) == 1

    conf = DicomConfidentialityProfileAttributeSeed.model_validate(
        {
            "id": "00184000",
            "tag": "(0018,4000)",
            "name": "Acquisition Comments",
            "stdCompIOD": "N",
            "basicProfile": "X",
            "cleanDescOpt": "C",
        }
    )
    assert conf.clean_desc_opt == "C"
    assert conf.rtn_uids_opt is None

    refs = DicomReferencesSeed.model_validate({"http://dicom.nema.org/ref": "<div><p>Reference text</p></div>"})
    assert len(refs) == 1
    assert refs["http://dicom.nema.org/ref"] == "<div><p>Reference text</p></div>"


# ---------------------------------------------------------------------------
# Auditable Neo4j graph node models
# ---------------------------------------------------------------------------


def test_dicom_element_node_from_seed_identity_coding_and_serialization() -> None:
    seed = DicomAttributeSeed.model_validate(
        {
            "tag": "(300A,0782)",
            "name": "Sample Element",
            "keyword": "SampleElement",
            "valueRepresentation": "US",
            "valueMultiplicity": "1",
            "retired": "Y",
            "id": "300a0782",
        }
    )
    created_at = datetime(2026, 11, 25, 12, 0, 0, tzinfo=timezone.utc)
    node_a = DicomElementNode.from_seed(seed, description="<p>Normative <b>summary</b></p>", CreatedAt=created_at)
    node_b = DicomElementNode.from_seed(seed, description="Normative summary")

    assert node_a.__label__ == "DicomElement"
    assert node_a.tag_hex == "300A0782"
    assert node_a.is_retired is True
    assert node_a.description == "Normative summary"
    assert node_a.coding_key == f"{DICOM_TAG_SYSTEM_URI}|300A0782"
    assert node_a.coding_macro == f"{DICOM_TAG_SYSTEM_URI}|300A0782|Sample Element"

    expected_id = generate_deterministic_id({"system": DICOM_TAG_SYSTEM_URI, "code": "300A0782"})
    assert node_a.id == expected_id
    assert node_a.id == node_b.id

    test_node = DicomElementNode.from_seed(seed, test_id="test_dicom_element")
    assert test_node.id != node_a.id

    props = node_a.to_neo4j_properties()
    assert props["id"] == expected_id
    assert props["CreatedAt"] == created_at.isoformat()
    assert "description_html" not in props


def test_dicom_element_node_empty_name_fallback_for_coding_macro() -> None:
    seed = DicomAttributeSeed.model_validate(
        {
            "tag": "(0008,0202)",
            "name": "",
            "keyword": "",
            "valueRepresentation": "",
            "valueMultiplicity": "",
            "retired": "Y",
            "id": "00080202",
        }
    )
    node = DicomElementNode.from_seed(seed)
    assert node.coding_macro == f"{DICOM_TAG_SYSTEM_URI}|00080202|(0008,0202)"


def test_ciod_module_macro_sop_and_reference_nodes_strip_html_and_use_deterministic_ids() -> None:
    ciod_seed = DicomCiodSeed.model_validate(
        {
            "id": "ct-image",
            "name": "CT Image",
            "description": "<p>\nThe <em>CT Image</em> IOD.</p>",
            "linkToStandard": "http://dicom.nema.org/ct",
        }
    )
    ciod = DicomCIODNode.from_seed(ciod_seed)
    assert ciod.__label__ == "DicomCIOD"
    assert ciod.description == "The CT Image IOD."
    assert ciod.id == generate_deterministic_id({"system": DICOM_CIOD_SYSTEM_URI, "code": "ct-image"})
    assert not hasattr(ciod, "description_html")

    mod_seed = DicomModuleSeed.model_validate(
        {
            "id": "acquisition-context",
            "name": "Acquisition Context",
            "description": '<p><a href="#t">This module</a> describes acquisition.</p>',
            "linkToStandard": "http://dicom.nema.org/mod",
        }
    )
    mod = DicomModuleNode.from_seed(mod_seed)
    assert mod.__label__ == "DicomModule"
    assert mod.description == "This module describes acquisition."
    assert mod.id == generate_deterministic_id({"system": DICOM_MODULE_SYSTEM_URI, "code": "acquisition-context"})

    macro_seed = DicomMacroSeed.model_validate(
        {
            "id": "basic-code-sequence",
            "name": "Basic Code Sequence",
            "description": "<p>Default set of Attributes.</p>",
            "linkToStandard": "http://dicom.nema.org/macro",
        }
    )
    macro = DicomMacroNode.from_seed(macro_seed)
    assert macro.__label__ == "DicomMacro"
    assert macro.description == "Default set of Attributes."
    assert macro.id == generate_deterministic_id({"system": DICOM_MACRO_SYSTEM_URI, "code": "basic-code-sequence"})

    sop_seed = DicomSopClassSeed.model_validate(
        {
            "id": "1.2.840.10008.5.1.4.1.1.2",
            "name": "CT Image Storage",
            "ciod": "CT Image",
        }
    )
    sop = DicomSOPClassNode.from_seed(sop_seed, modality="CT")
    assert sop.__label__ == "DicomSOPClass"
    assert sop.modality == "CT"
    assert sop.id == generate_deterministic_id({"system": DICOM_SOP_SYSTEM_URI, "code": "1.2.840.10008.5.1.4.1.1.2"})

    ref = DicomStandardReferenceNode.from_seed(
        "http://dicom.nema.org/biblio",
        "<div><p>[<abbr>EXIF 2.31</abbr>] <em>Exchangeable Image File Format</em>.</p></div>",
    )
    assert ref.__label__ == "DicomStandardReference"
    assert ref.content == "[EXIF 2.31] Exchangeable Image File Format."
    assert not hasattr(ref, "content_html")
    assert ref.id == generate_deterministic_id(
        {"system": DICOM_REFERENCE_SYSTEM_URI, "code": "http://dicom.nema.org/biblio"}
    )


def test_confidentiality_profile_node_handles_standard_and_wildcard_entries() -> None:
    standard_seed = DicomConfidentialityProfileAttributeSeed.model_validate(
        {
            "id": "00080050",
            "tag": "(0008,0050)",
            "name": "Accession Number",
            "stdCompIOD": "Y",
            "basicProfile": "Z",
        }
    )
    profile = DicomConfidentialityProfileNode.from_seed(standard_seed)
    assert profile.__label__ == "DicomConfidentialityProfile"
    assert profile.attribute_id == "00080050"
    assert profile.std_comp_iod is True
    assert profile.id == generate_deterministic_id({"system": DICOM_CONFIDENTIALITY_SYSTEM_URI, "code": "00080050"})

    wildcard_seed = DicomConfidentialityProfileAttributeSeed.model_validate(
        {
            "id": "ggggeeee-where-gggg-is-odd",
            "tag": "(gggg,eeee) where gggg is odd",
            "name": "Private attributes",
            "stdCompIOD": "N",
            "basicProfile": "X",
            "rtnSafePrivOpt": "C",
        }
    )
    wildcard_profile = DicomConfidentialityProfileNode.from_seed(wildcard_seed)
    assert wildcard_profile.attribute_id == "ggggeeee-where-gggg-is-odd"
    assert wildcard_profile.std_comp_iod is False
    assert wildcard_profile.rtn_safe_priv_opt == "C"


# ---------------------------------------------------------------------------
# Auditable Neo4j graph relationship models
# ---------------------------------------------------------------------------


def test_relations_deterministic_ids_negation_and_neo4j_properties() -> None:
    ciod = DicomCIODNode(ciod_id="ct-image", name="CT Image")
    sop = DicomSOPClassNode(sop_class_uid="1.2.840.10008.5.1.4.1.1.2", name="CT Image Storage", ciod_name="CT Image")
    module = DicomModuleNode(module_id="patient", name="Patient")
    macro = DicomMacroNode(macro_id="pixel-measures", name="Pixel Measures")
    elem = DicomElementNode(tag="(0008,0050)", tag_hex="00080050", name="Accession Number", keyword="AccessionNumber")
    profile = DicomConfidentialityProfileNode(
        attribute_id="00080050",
        tag="(0008,0050)",
        name="Accession Number",
        std_comp_iod="Y",
        basic_profile="Z",
    )

    impl_rel = ImplementsCiodRelation.from_nodes(sop, ciod)
    assert impl_rel.__type__ == "IMPLEMENTS_CIOD"
    assert impl_rel.to_negated_type() == "NOT_IMPLEMENTS_CIOD"
    assert impl_rel.id == ImplementsCiodRelation.from_nodes(sop, ciod).id

    has_mod = HasModuleRelation.from_seed(
        DicomCiodToModuleSeed.model_validate(
            {
                "ciodId": "ct-image",
                "moduleId": "patient",
                "usage": "M",
                "conditionalStatement": None,
                "informationEntity": "Patient",
            }
        ),
        start_node_id=ciod.id,
        end_node_id=module.id,
    )
    props_without_endpoints = has_mod.to_neo4j_properties()
    assert "start_node_id" not in props_without_endpoints
    assert "end_node_id" not in props_without_endpoints
    assert props_without_endpoints["usage"] == "M"
    assert type(props_without_endpoints["usage"]) is str

    props_with_endpoints = has_mod.to_neo4j_properties(include_endpoints=True)
    assert props_with_endpoints["start_node_id"] == ciod.id
    assert props_with_endpoints["end_node_id"] == module.id

    has_fg = HasFuncGroupMacroRelation.from_seed(
        DicomCiodToFuncGroupMacroSeed.model_validate(
            {
                "ciodId": "ct-image",
                "macroId": "pixel-measures",
                "usage": "C",
                "conditionalStatement": "<p>Required if multi-frame</p>",
                "moduleType": "Multi-frame",
            }
        ),
        start_node_id=ciod.id,
        end_node_id=macro.id,
    )
    assert has_fg.conditional_statement == "Required if multi-frame"
    assert has_fg.to_neo4j_properties()["module_type"] == "Multi-frame"

    conf_rel = HasConfidentialityProfileRelation.from_nodes(elem, profile)
    assert conf_rel.__type__ == "HAS_CONFIDENTIALITY_PROFILE"
    conf_rel_test = HasConfidentialityProfileRelation.from_nodes(elem, profile, test_id="test_case_1")
    assert conf_rel_test.id != conf_rel.id


def test_uses_element_relation_path_prevents_sequence_id_collisions() -> None:
    module = DicomModuleNode(module_id="acquisition-context", name="Acquisition Context")
    elem = DicomElementNode(tag="(0008,0100)", tag_hex="00080100", name="Code Value", keyword="CodeValue")

    seed_path_1 = DicomModuleToAttributeSeed.model_validate(
        {
            "moduleId": "acquisition-context",
            "path": "acquisition-context:00400555:0040a043:00080100",
            "tag": "(0008,0100)",
            "type": "1C",
            "linkToStandard": "http://dicom.nema.org/t1",
            "description": "<p>Code Value for Concept Name</p>",
            "externalReferences": [{"sourceUrl": "http://dicom.nema.org/s8.1", "title": "Section\xa08.1"}],
        }
    )
    seed_path_2 = DicomModuleToAttributeSeed.model_validate(
        {
            "moduleId": "acquisition-context",
            "path": "acquisition-context:00400555:0040a168:00080100",
            "tag": "(0008,0100)",
            "type": "1C",
            "linkToStandard": "http://dicom.nema.org/t1",
            "description": "<p>Code Value for Concept Code</p>",
            "externalReferences": [],
        }
    )

    rel_1 = UsesElementRelation.from_seed(seed_path_1, start_node_id=module.id, end_node_id=elem.id)
    rel_2 = UsesElementRelation.from_seed(seed_path_2, start_node_id=module.id, end_node_id=elem.id)

    assert rel_1.id != rel_2.id
    assert rel_1.id == UsesElementRelation.from_seed(seed_path_1, start_node_id=module.id, end_node_id=elem.id).id
    assert rel_1.description == "Code Value for Concept Name"
    assert rel_1.external_reference_urls == ["http://dicom.nema.org/s8.1"]
    assert not hasattr(rel_1, "description_html")


def test_references_standard_relation_path_scoping() -> None:
    module = DicomModuleNode(module_id="acquisition-context", name="Acquisition Context")
    ref_node = DicomStandardReferenceNode.from_seed("http://dicom.nema.org/s8.1", "<p>Section 8.1</p>")
    ext_seed = DicomExternalReferenceSeed.model_validate(
        {"sourceUrl": "http://dicom.nema.org/s8.1", "title": "Section\xa08.1"}
    )

    rel_a = ReferencesStandardRelation.from_seed(
        ext_seed,
        start_node_id=module.id,
        reference=ref_node,
        path="acquisition-context:00400555:00080100",
    )
    rel_b = ReferencesStandardRelation.from_seed(
        ext_seed,
        start_node_id=module.id,
        reference=ref_node,
        path="acquisition-context:00400555:00080102",
    )
    assert rel_a.title == "Section 8.1"
    assert rel_a.id != rel_b.id


# ---------------------------------------------------------------------------
# Real local seed file smoke validation (skipped if ./data-source/dicom absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not DATA_SOURCE_DICOM_DIR.exists(), reason="Local ./data-source/dicom seed files not present")
def test_real_dicom_seed_files_validate_and_convert_cleanly() -> None:
    # Validate 100% of records in the 9 compact/medium seed files
    attrs = [
        DicomAttributeSeed.model_validate(r)
        for r in json.loads((DATA_SOURCE_DICOM_DIR / "attributes.json").read_text())
    ]
    ciods = [DicomCiodSeed.model_validate(r) for r in json.loads((DATA_SOURCE_DICOM_DIR / "ciods.json").read_text())]
    modules = [
        DicomModuleSeed.model_validate(r) for r in json.loads((DATA_SOURCE_DICOM_DIR / "modules.json").read_text())
    ]
    macros = [DicomMacroSeed.model_validate(r) for r in json.loads((DATA_SOURCE_DICOM_DIR / "macros.json").read_text())]
    sops = [DicomSopClassSeed.model_validate(r) for r in json.loads((DATA_SOURCE_DICOM_DIR / "sops.json").read_text())]
    c2m = [
        DicomCiodToModuleSeed.model_validate(r)
        for r in json.loads((DATA_SOURCE_DICOM_DIR / "ciod_to_modules.json").read_text())
    ]
    c2fg = [
        DicomCiodToFuncGroupMacroSeed.model_validate(r)
        for r in json.loads((DATA_SOURCE_DICOM_DIR / "ciod_to_func_group_macros.json").read_text())
    ]
    conf = [
        DicomConfidentialityProfileAttributeSeed.model_validate(r)
        for r in json.loads((DATA_SOURCE_DICOM_DIR / "confidentiality_profile_attributes.json").read_text())
    ]
    refs = DicomReferencesSeed.model_validate(json.loads((DATA_SOURCE_DICOM_DIR / "references.json").read_text()))

    assert len(attrs) > 5000
    assert len(ciods) > 100
    assert len(modules) > 400
    assert len(macros) > 200
    assert len(sops) > 100
    assert len(c2m) > 3000
    assert len(c2fg) > 400
    assert len(conf) > 600
    assert len(refs) > 1000

    # Convert all core nodes and ensure deterministic IDs are unique across each dataset
    elem_nodes = [DicomElementNode.from_seed(s) for s in attrs]
    assert len({n.id for n in elem_nodes}) == len(elem_nodes)

    ciod_nodes = [DicomCIODNode.from_seed(s) for s in ciods]
    assert len({n.id for n in ciod_nodes}) == len(ciod_nodes)
    assert all("<p>" not in n.description for n in ciod_nodes)

    conf_nodes = [DicomConfidentialityProfileNode.from_seed(s) for s in conf]
    assert len({n.id for n in conf_nodes}) == len(conf_nodes)

    # Validate representative slices from the two multi-megabyte mapping files
    m2a_raw = json.loads((DATA_SOURCE_DICOM_DIR / "module_to_attributes.json").read_text())
    m2a_sample = [DicomModuleToAttributeSeed.model_validate(r) for r in m2a_raw[:200]]
    assert len(m2a_sample) == 200

    mac2a_raw = json.loads((DATA_SOURCE_DICOM_DIR / "macro_to_attributes.json").read_text())
    mac2a_sample = [DicomMacroToAttributeSeed.model_validate(r) for r in mac2a_raw[:200]]
    assert len(mac2a_sample) == 200
