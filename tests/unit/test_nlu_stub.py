"""Unit tests for StubNLUService and NLUParserProtocol."""

from __future__ import annotations

import pytest

from gosmart_dictionary_service.model.search import NLUIntent, SearchEntityType
from gosmart_dictionary_service.service.nlu_service import NLUParserProtocol, StubNLUService


@pytest.fixture
def nlu() -> StubNLUService:
    return StubNLUService()


def test_stub_implements_protocol(nlu: StubNLUService) -> None:
    assert isinstance(nlu, NLUParserProtocol)


def test_parse_exact_clinical_codes(nlu: StubNLUService) -> None:
    res_loinc = nlu.parse("LOINC 87279-6")
    assert res_loinc.intent == NLUIntent.EXACT_LOOKUP
    assert len(res_loinc.extracted_codes) == 1
    assert res_loinc.extracted_codes[0].vocabulary_id == "LOINC"
    assert res_loinc.extracted_codes[0].code == "87279-6"
    assert res_loinc.inferred_entity_types == [SearchEntityType.CONCEPT]

    res_snomed = nlu.parse("SNOMED:233604007")
    assert res_snomed.intent == NLUIntent.EXACT_LOOKUP
    assert res_snomed.extracted_codes[0].vocabulary_id == "SNOMED"
    assert res_snomed.extracted_codes[0].code == "233604007"

    res_bare_icd = nlu.parse("e11.9")
    assert res_bare_icd.intent == NLUIntent.EXACT_LOOKUP
    assert res_bare_icd.extracted_codes[0].vocabulary_id == "ICD10CM"
    assert res_bare_icd.extracted_codes[0].code == "E11.9"

    res_bare_loinc = nlu.parse("24627-2")
    assert res_bare_loinc.intent == NLUIntent.EXACT_LOOKUP
    assert res_bare_loinc.extracted_codes[0].vocabulary_id == "LOINC"


def test_parse_dicom_tags_and_uids(nlu: StubNLUService) -> None:
    res_tag = nlu.parse("(0018,0050)")
    assert res_tag.intent == NLUIntent.EXACT_LOOKUP
    assert res_tag.extracted_codes[0].vocabulary_id == "DICOM_TAG"
    assert res_tag.extracted_codes[0].code == "00180050"
    assert res_tag.inferred_entity_types == [SearchEntityType.DICOM_ELEMENT]

    res_uid = nlu.parse("1.2.840.10008.5.1.4.1.1.2")
    assert res_uid.intent == NLUIntent.EXACT_LOOKUP
    assert res_uid.extracted_codes[0].vocabulary_id == "DICOM_UID"
    assert res_uid.extracted_codes[0].code == "1.2.840.10008.5.1.4.1.1.2"
    assert res_uid.inferred_entity_types == [SearchEntityType.DICOM_SOP_CLASS]


def test_parse_related_lookup_with_and_without_modifier(nlu: StubNLUService) -> None:
    res = nlu.parse("Find something related to LOINC 87279-6")
    assert res.intent == NLUIntent.RELATED_LOOKUP
    assert len(res.extracted_codes) == 1
    assert res.extracted_codes[0].vocabulary_id == "LOINC"
    assert res.extracted_codes[0].code == "87279-6"
    assert res.modifier_text is None

    res_mod = nlu.parse("Find something related to LOINC 87279-6 with contrast")
    assert res_mod.intent == NLUIntent.RELATED_LOOKUP
    assert res_mod.extracted_codes[0].code == "87279-6"
    assert res_mod.modifier_text == "with contrast"


def test_parse_crosswalk_and_subtypes_intents(nlu: StubNLUService) -> None:
    res_cw = nlu.parse("Map ICD-10-CM E11.9 to SNOMED")
    assert res_cw.intent == NLUIntent.CROSSWALK
    assert res_cw.extracted_codes[0].vocabulary_id == "ICD10CM"
    assert res_cw.extracted_codes[0].code == "E11.9"
    assert res_cw.target_vocabularies == ["SNOMED"]

    res_sub = nlu.parse("Show subtypes of SNOMED 233604007")
    assert res_sub.intent == NLUIntent.EXPAND_SUBTYPES
    assert res_sub.extracted_codes[0].vocabulary_id == "SNOMED"
    assert res_sub.extracted_codes[0].code == "233604007"


def test_parse_free_text_and_hints(nlu: StubNLUService) -> None:
    res_plain = nlu.parse("acute pneumonia")
    assert res_plain.intent == NLUIntent.SEARCH
    assert res_plain.extracted_codes == []
    assert res_plain.inferred_entity_types == []
    assert res_plain.normalized_query == "acute pneumonia"

    res_dicom = nlu.parse("Find DICOM tag for slice thickness")
    assert res_dicom.intent == NLUIntent.SEARCH
    assert SearchEntityType.DICOM_ELEMENT in res_dicom.inferred_entity_types

    res_snomed = nlu.parse("Find SNOMED codes for heart attack")
    assert res_snomed.intent == NLUIntent.SEARCH
    assert res_snomed.extracted_codes == []
    assert res_snomed.inferred_vocabularies == ["SNOMED"]
    assert SearchEntityType.CONCEPT in res_snomed.inferred_entity_types
