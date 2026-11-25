"""Unit tests for clinical terminology models, FHIR URI resolution, and Athena edge routing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gosmart_dictionary_service.model.actor_message import (
    ALLOWED_EMBEDDING_TARGET_LABELS,
    ClinicalTermsCleanupMessage,
    ClinicalTermsIngestMessage,
    EmbeddingItem,
)
from gosmart_dictionary_service.model.clinical import (
    AthenaConceptMetadataSeed,
    AthenaConceptRelationshipMetadataSeed,
    AthenaConceptSeed,
    AthenaVocabularySeed,
    BelongsToSystemRelation,
    ClinicalIngestionProfile,
    CodeSystemNode,
    ConceptNode,
    HasAttributeRelation,
    IsARelation,
    MapsToRelation,
    ReplacedByRelation,
    classify_clinical_relationship,
    compute_clinical_relation_id,
    compute_code_system_id,
    compute_concept_id,
    is_canonical_forward_relationship,
    parse_athena_date,
    resolve_code_system_uri,
)
from gosmart_dictionary_service.model.embedding import DictionaryEmbeddingFieldTag


def test_resolve_code_system_uri_canonical_and_fallback() -> None:
    assert resolve_code_system_uri("SNOMED") == "http://snomed.info/sct"
    assert resolve_code_system_uri("LOINC") == "http://loinc.org"
    assert resolve_code_system_uri("ICD10CM") == "http://hl7.org/fhir/sid/icd-10-cm"
    assert resolve_code_system_uri("ICD10") == "http://hl7.org/fhir/sid/icd-10"
    assert resolve_code_system_uri("ICD11") == "http://id.who.int/icd/release/11/mms"
    assert resolve_code_system_uri("UCUM") == "http://unitsofmeasure.org"
    assert resolve_code_system_uri("RxNorm") == "http://www.nlm.nih.gov/research/umls/rxnorm"
    assert resolve_code_system_uri("DCM") == "http://dicom.nema.org/resources/ontology/DCM"
    assert resolve_code_system_uri("UB04 Typ bill") == "http://gosmart.health/terminology/omop/ub04-typ-bill"

    with pytest.raises(ValueError):
        resolve_code_system_uri("   ")


def test_parse_athena_date_normalizes_yyyymmdd() -> None:
    assert parse_athena_date("20180131") == "2018-01-31"
    assert parse_athena_date("20991231") == "2099-12-31"
    assert parse_athena_date("2026-03-01") == "2026-03-01"
    assert parse_athena_date("") == ""
    assert parse_athena_date(None) == ""


def test_canonical_forward_relationship_deduplication_is_strictly_antisymmetric() -> None:
    pairs = [
        ("Is a", "Subsumes", "IS_A"),
        ("Maps to", "Mapped from", "MAPS_TO"),
        ("Maps to value", "Value mapped from", "MAPS_TO"),
        ("Concept replaced by", "Concept replaces", "REPLACED_BY"),
        ("Concept same_as to", "Concept same_as from", "REPLACED_BY"),
        ("Concept poss_eq to", "Concept poss_eq from", "REPLACED_BY"),
        ("Concept was_a to", "Concept was_a from", "REPLACED_BY"),
        ("Has finding site", "Finding site of", "HAS_ATTRIBUTE"),
        ("Has asso morph", "Asso morph of", "HAS_ATTRIBUTE"),
        ("Has component", "Component of", "HAS_ATTRIBUTE"),
        ("Panel contains", "Contained in panel", "HAS_ATTRIBUTE"),
    ]
    for fwd, rev, expected_type in pairs:
        assert is_canonical_forward_relationship(fwd, rev) is True
        assert is_canonical_forward_relationship(rev, fwd) is False
        assert classify_clinical_relationship(fwd, rev) == expected_type
        assert classify_clinical_relationship(rev, fwd) is None

    # RF2 housekeeping edges are skipped by default and opt-in via flag
    assert classify_clinical_relationship("Has status", "Status of") is None
    assert classify_clinical_relationship("Has Module", "Module of") is None
    assert classify_clinical_relationship("Has status", "Status of", include_rf2_metadata_edges=True) == "HAS_ATTRIBUTE"
    assert classify_clinical_relationship("Status of", "Has status", include_rf2_metadata_edges=True) is None

    # Disabling attribute edges still preserves IS_A / MAPS_TO / REPLACED_BY
    assert classify_clinical_relationship("Has finding site", "Finding site of", include_attribute_edges=False) is None
    assert classify_clinical_relationship("Is a", "Subsumes", include_attribute_edges=False) == "IS_A"


def test_code_system_and_concept_deterministic_ids_match_fast_helpers() -> None:
    vocab_seed = AthenaVocabularySeed(
        vocabulary_id="SNOMED",
        vocabulary_name="Systematic Nomenclature of Medicine - Clinical Terms",
        vocabulary_reference="http://www.snomed.org/",
        vocabulary_version="2026-02-01 SNOMED CT International Edition",
        vocabulary_concept_id=44819097,
    )
    cs = CodeSystemNode.from_seed(vocab_seed, test_id="t_models")
    assert cs.system_uri == "http://snomed.info/sct"
    assert cs.id == compute_code_system_id("http://snomed.info/sct", test_id="t_models")
    assert cs.id != compute_code_system_id("http://snomed.info/sct")

    concept_seed = AthenaConceptSeed(
        concept_id=44054006,
        concept_name="Pulmonary embolism",
        domain_id="Condition",
        vocabulary_id="SNOMED",
        concept_class_id="Disorder",
        standard_concept="S",
        concept_code="59282003",
        valid_start_date="20020131",
        valid_end_date="20991231",
        invalid_reason="",
    )
    concept = ConceptNode.from_seed(
        concept_seed,
        synonyms=["PE", "Pulmonary artery embolism", "PE"],
        concept_category="A",
        test_id="t_models",
    )
    assert concept.id == compute_concept_id("http://snomed.info/sct", "59282003", test_id="t_models")
    assert concept.coding_key == "http://snomed.info/sct|59282003"
    assert concept.coding_macro == "http://snomed.info/sct|59282003|Pulmonary embolism"
    assert concept.synonyms == ["PE", "Pulmonary artery embolism"]
    assert concept.is_deprecated is False
    assert concept.valid_start_date == "2002-01-31"
    assert concept.valid_end_date == "2099-12-31"
    assert concept.concept_category == "A"


def test_concept_node_guards_omop_generated_code_collisions() -> None:
    s1 = AthenaConceptSeed(
        concept_id=101,
        concept_name="Rel A",
        domain_id="Metadata",
        vocabulary_id="Relationship",
        concept_class_id="Relationship",
        standard_concept="",
        concept_code="OMOP generated",
        valid_start_date="19700101",
        valid_end_date="20991231",
        invalid_reason="",
    )
    s2 = s1.model_copy(update={"concept_id": 102, "concept_name": "Rel B"})
    n1 = ConceptNode.from_seed(s1)
    n2 = ConceptNode.from_seed(s2)
    assert n1.code == "OMOP:101"
    assert n2.code == "OMOP:102"
    assert n1.id != n2.id


def test_clinical_relations_deterministic_ids_match_fast_helper() -> None:
    start_id = compute_concept_id("http://hl7.org/fhir/sid/icd-10-cm", "I26.99", test_id="t_rel")
    end_id = compute_concept_id("http://snomed.info/sct", "59282003", test_id="t_rel")
    cs_id = compute_code_system_id("http://hl7.org/fhir/sid/icd-10-cm", test_id="t_rel")

    belongs = BelongsToSystemRelation(start_node_id=start_id, end_node_id=cs_id, test_id="t_rel")
    assert belongs.id == compute_clinical_relation_id(start_id, "BELONGS_TO_SYSTEM", cs_id, test_id="t_rel")

    is_a = IsARelation(
        start_node_id=start_id,
        end_node_id=end_id,
        valid_start_date="20200101",
        valid_end_date="20991231",
        test_id="t_rel",
    )
    assert is_a.id == compute_clinical_relation_id(start_id, "IS_A", end_id, test_id="t_rel")
    assert is_a.valid_start_date == "2020-01-01"

    maps_to = MapsToRelation(
        start_node_id=start_id,
        end_node_id=end_id,
        relationship_id="Maps to",
        equivalence="exactMatch",
        confidence=1.0,
        mapping_source="OHDSI",
        test_id="t_rel",
    )
    assert maps_to.id == compute_clinical_relation_id(
        start_id,
        "MAPS_TO",
        end_id,
        relationship_id="Maps to",
        test_id="t_rel",
    )
    maps_to_val = MapsToRelation(
        start_node_id=start_id,
        end_node_id=end_id,
        relationship_id="Maps to value",
        test_id="t_rel",
    )
    assert maps_to.id != maps_to_val.id

    has_attr = HasAttributeRelation(
        start_node_id=start_id,
        end_node_id=end_id,
        relationship_id="Has finding site",
        relationship_name="Has finding site (SNOMED)",
        test_id="t_rel",
    )
    assert has_attr.id == compute_clinical_relation_id(
        start_id,
        "HAS_ATTRIBUTE",
        end_id,
        relationship_id="Has finding site",
        test_id="t_rel",
    )

    replaced = ReplacedByRelation(
        start_node_id=start_id,
        end_node_id=end_id,
        relationship_id="Concept replaced by",
        test_id="t_rel",
    )
    assert replaced.id == compute_clinical_relation_id(
        start_id,
        "REPLACED_BY",
        end_id,
        relationship_id="Concept replaced by",
        test_id="t_rel",
    )


def test_metadata_seeds_and_actor_messages_validation() -> None:
    cm = AthenaConceptMetadataSeed(concept_id=100, concept_category="A", reuse_status="")
    assert cm.concept_category == "A"

    crm = AthenaConceptRelationshipMetadataSeed(
        concept_id_1=1,
        concept_id_2=2,
        relationship_id="Maps to",
        relationship_predicate_id="broadMatch",
        mapping_source="OHDSI",
        confidence="0.9",
    )
    assert crm.confidence == 0.9
    assert crm.relationship_predicate_id == "broadMatch"

    assert "Concept" in ALLOWED_EMBEDDING_TARGET_LABELS
    assert "CodeSystem" in ALLOWED_EMBEDDING_TARGET_LABELS
    emb_item = EmbeddingItem(
        target_node_id="concept-uuid",
        target_label="Concept",
        target_field_tag=DictionaryEmbeddingFieldTag.DISPLAY,
        text="SNOMED Condition (59282003): Pulmonary embolism",
    )
    assert emb_item.target_label == "Concept"
    assert emb_item.target_field_tag == "DISPLAY"

    msg = ClinicalTermsIngestMessage(profile=ClinicalIngestionProfile.DEMO_CORE, test_id=" t1 ")
    assert msg.profile == ClinicalIngestionProfile.DEMO_CORE
    assert msg.test_id == "t1"

    cleanup = ClinicalTermsCleanupMessage(test_id=" t1 ")
    assert cleanup.test_id == "t1"
    with pytest.raises(ValidationError):
        ClinicalTermsCleanupMessage(test_id="   ")
