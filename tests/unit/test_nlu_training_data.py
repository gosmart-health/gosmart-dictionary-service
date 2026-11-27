"""Unit tests for the ``find_coding_schema`` spaCy NLU training set and plugin."""

from __future__ import annotations

from pathlib import Path

import pytest
import spacy
from gosmart_clinical_nlu import (
    get_intent_spec,
    load_intent_models,
    load_training_dir,
    parse_utterance,
    save_bundle,
    train_intent_models,
)
from gosmart_clinical_nlu.training import to_ruler_pattern

from gosmart_dictionary_service.nlu.find_coding_schema import (
    FIND_CODING_SCHEMA_SPEC,
    CodingSchemaQueryIntent,
    normalize_coding_code,
    normalize_coding_schema,
    normalize_dicom_tag,
    normalize_dicom_uid,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = REPO_ROOT / "training"


@pytest.fixture(scope="module")
def corpus():
    return load_training_dir(TRAINING_DIR)


@pytest.fixture(scope="module")
def ruler_nlp(corpus):
    intent = next(item for item in corpus.intents if item.name == "find_coding_schema")
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([to_ruler_pattern(pattern) for pattern in intent.patterns])
    return nlp


@pytest.fixture(scope="module")
def trained_models(corpus, tmp_path_factory: pytest.TempPathFactory):
    models = train_intent_models(corpus)
    bundle_path = tmp_path_factory.mktemp("nlu_bundle") / "intent_models.zip"
    save_bundle(models, bundle_path)
    return load_intent_models(bundle_path)


def test_training_corpus_loads_and_validates(corpus) -> None:
    intent_names = [item.name for item in corpus.intents]
    assert "find_coding_schema" in intent_names

    spec = next(item for item in corpus.intents if item.name == "find_coding_schema")
    assert spec.label == "FIND_CODING_SCHEMA"
    assert len(spec.patterns) > 0
    assert len(spec.examples.positive) >= 35
    assert len(spec.examples.negative) >= 30


def test_intent_spec_registered_via_entry_point() -> None:
    discovered = get_intent_spec("find_coding_schema")
    assert discovered == FIND_CODING_SCHEMA_SPEC
    assert set(discovered.entity_map) == {"CODING_SCHEMA", "CODING_CODE", "DICOM_TAG", "DICOM_UID"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("LOINC", "LOINC"),
        ("loinc", "LOINC"),
        ("SNOMED", "SNOMED"),
        ("SNOMED CT", "SNOMED"),
        ("SNOMED-CT", "SNOMED"),
        ("SNOMED_CT", "SNOMED"),
        ("SCT", "SNOMED"),
        ("ICD", "ICD10"),
        ("ICD-10", "ICD10"),
        ("ICD-10-CM", "ICD10CM"),
        ("ICD10CM", "ICD10CM"),
        ("ICD-11", "ICD11"),
        ("DICOM", "DICOM"),
        ("DICOM tag", "DICOM"),
        ("DICOM element", "DICOM"),
        ("DICOM UID", "DICOM"),
        ("RxNorm", "RxNorm"),
        ("CPT-4", "CPT4"),
        ("HCPCS", "HCPCS"),
    ],
)
def test_normalize_coding_schema(raw: str, expected: str) -> None:
    assert normalize_coding_schema(raw) == expected


def test_code_and_dicom_normalizers() -> None:
    assert normalize_coding_code("87279-6") == "87279-6"
    assert normalize_coding_code(" 87279 - 6 ") == "87279-6"
    assert normalize_coding_code("e11.9") == "E11.9"
    assert normalize_dicom_tag("(0008, 0010)") == "00080010"
    assert normalize_dicom_tag("(0018,001a)") == "0018001A"
    assert normalize_dicom_tag("00100020") == "00100020"
    assert normalize_dicom_uid(" 1.2.840.10008.5.1.4.1.1.2 ") == "1.2.840.10008.5.1.4.1.1.2"


def test_dynamic_schema_availability_check() -> None:
    demo_loaded_schemas = {"DICOM", "LOINC"}
    full_loaded_schemas = {"DICOM", "LOINC", "ICD10", "ICD10CM", "SNOMED"}

    loinc_query = CodingSchemaQueryIntent(coding_schema="LOINC", code="87279-6")
    assert loinc_query.effective_schema == "LOINC"
    assert loinc_query.identifier == "87279-6"
    assert loinc_query.is_schema_available(demo_loaded_schemas) is True

    snomed_query = CodingSchemaQueryIntent(coding_schema="SNOMED", code="233604007")
    assert snomed_query.is_schema_available(demo_loaded_schemas) is False
    assert snomed_query.is_schema_available(full_loaded_schemas) is True

    bare_tag_query = CodingSchemaQueryIntent(dicom_tag="00180050")
    assert bare_tag_query.effective_schema == "DICOM"
    assert bare_tag_query.identifier == "00180050"
    assert bare_tag_query.is_schema_available(demo_loaded_schemas) is True

    unsupported_query = CodingSchemaQueryIntent(coding_schema="RxNorm", code="197361")
    assert unsupported_query.is_schema_available(full_loaded_schemas) is False


def test_all_positive_examples_extract_identifier_entities(corpus, ruler_nlp) -> None:
    spec = next(item for item in corpus.intents if item.name == "find_coding_schema")
    identifier_labels = {"CODING_CODE", "DICOM_TAG", "DICOM_UID"}

    for utterance in spec.examples.positive:
        doc = ruler_nlp(utterance)
        labels = {ent.label_ for ent in doc.ents}
        assert labels & identifier_labels, f"No code/tag/UID extracted from positive example: {utterance!r}"


def test_negative_examples_do_not_extract_identifier_entities(corpus, ruler_nlp) -> None:
    spec = next(item for item in corpus.intents if item.name == "find_coding_schema")
    identifier_labels = {"CODING_CODE", "DICOM_TAG", "DICOM_UID"}

    for utterance in spec.examples.negative:
        doc = ruler_nlp(utterance)
        extracted_ids = [(ent.label_, ent.text) for ent in doc.ents if ent.label_ in identifier_labels]
        assert not extracted_ids, f"False identifier extraction {extracted_ids} in negative example: {utterance!r}"


def test_entity_ruler_separates_snomed_digits_from_dicom_hex_tags(ruler_nlp) -> None:
    snomed_doc = ruler_nlp("SNOMED CT 59282003")
    assert [(e.label_, e.text) for e in snomed_doc.ents] == [
        ("CODING_SCHEMA", "SNOMED CT"),
        ("CODING_CODE", "59282003"),
    ]

    dicom_doc = ruler_nlp("DICOM 00100020")
    assert [(e.label_, e.text) for e in dicom_doc.ents] == [
        ("CODING_SCHEMA", "DICOM"),
        ("DICOM_TAG", "00100020"),
    ]

    pixel_doc = ruler_nlp("DICOM tag 7FE00010")
    assert [(e.label_, e.text) for e in pixel_doc.ents] == [
        ("CODING_SCHEMA", "DICOM tag"),
        ("DICOM_TAG", "7FE00010"),
    ]


@pytest.mark.parametrize(
    ("utterance", "expected_schema", "expected_code", "expected_tag", "expected_uid"),
    [
        ("LOINC 232456", "LOINC", "232456", None, None),
        ("Please find LOINC 12345-6", "LOINC", "12345-6", None, None),
        ("Can you look up SNOMED-CT 38341003?", "SNOMED", "38341003", None, None),
        ("SNOMED_CT 650981923", "SNOMED", "650981923", None, None),
        ("What is ICD-10-CM K21.9?", "ICD10CM", "K21.9", None, None),
        ("Find ICD-10 J18.9", "ICD10", "J18.9", None, None),
        ("Look up DICOM (0008, 0010)", "DICOM", None, "00080010", None),
        ("Tell me about DICOM tag (0028, 0011)", "DICOM", None, "00280011", None),
        ("DICOM tag 00080010", "DICOM", None, "00080010", None),
        ("Look up DICOM UID 1.2.840.10008.5.1.4.1.1.2", "DICOM", None, None, "1.2.840.10008.5.1.4.1.1.2"),
        ("(0018,0050)", None, None, "00180050", None),
        ("Look up RxNorm 197361", "RxNorm", "197361", None, None),
        ("What is ICD-11 8B11?", "ICD11", "8B11", None, None),
    ],
)
def test_trained_bundle_parses_positive_utterances(
    trained_models,
    utterance: str,
    expected_schema: str | None,
    expected_code: str | None,
    expected_tag: str | None,
    expected_uid: str | None,
) -> None:
    result = parse_utterance(utterance, trained_models)
    assert result.status == "matched", f"Expected matched for {utterance!r}, got {result}"
    assert result.intent == "find_coding_schema"
    assert isinstance(result.data, CodingSchemaQueryIntent)
    assert result.data.coding_schema == expected_schema
    assert result.data.code == expected_code
    assert result.data.dicom_tag == expected_tag
    assert result.data.dicom_uid == expected_uid


@pytest.mark.parametrize(
    "utterance",
    [
        "Hello",
        "What is the weather today?",
        "Find DICOM tag for slice thickness",
        "Find SNOMED codes for heart attack",
        "What is acute pneumonia?",
        "Look up LOINC tests for cholesterol",
        "Search ICD-10 for asthma",
        "Show me DICOM modules for CT Image",
    ],
)
def test_trained_bundle_rejects_negative_utterances(trained_models, utterance: str) -> None:
    result = parse_utterance(utterance, trained_models)
    assert result.status == "low_confidence", f"Expected low_confidence for {utterance!r}, got {result}"
