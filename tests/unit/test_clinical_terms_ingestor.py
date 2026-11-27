"""Unit tests for streaming OHDSI Athena clinical terms ingestion and Pykka actor orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from gosmart_dictionary_service.__main__ import main as cli_main
from gosmart_dictionary_service.actor.clinical_terms_ingestion_actor import ClinicalTermsIngestionActor
from gosmart_dictionary_service.actor.embedding_actor import EmbeddingActor
from gosmart_dictionary_service.ingestor.clinical_terms_ingestor import (
    ClinicalTermsIngesterService,
    ClinicalTermsIngestionOptions,
    build_concept_embedding_items,
)
from gosmart_dictionary_service.model.actor_message import (
    ClinicalTermsCleanupMessage,
    ClinicalTermsIngestMessage,
)
from gosmart_dictionary_service.model.clinical import ClinicalIngestionProfile
from gosmart_dictionary_service.model.result import STATUS_BAD_REQUEST, STATUS_OK
from gosmart_dictionary_service.service.embedding_service import EmbeddingBatchReport


def _write_tsv(path: Path, headers: List[str], rows: List[List[str]]) -> None:
    lines = ["\t".join(headers)] + ["\t".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_synthetic_athena_dir(base: Path) -> Path:
    """Write a realistic, compact synthetic OHDSI Athena CSV bundle for fast deterministic tests."""
    base.mkdir(parents=True, exist_ok=True)

    _write_tsv(
        base / "VOCABULARY.csv",
        [
            "vocabulary_id",
            "vocabulary_name",
            "vocabulary_reference",
            "vocabulary_version",
            "vocabulary_concept_id",
        ],
        [
            ["SNOMED", "SNOMED CT", "http://www.snomed.org/", "2026-02-01", "44819097"],
            ["LOINC", "LOINC", "http://loinc.org", "LOINC 2.82", "44819102"],
            ["ICD10CM", "ICD-10-CM", "https://www.cdc.gov/nchs/icd/icd-10-cm.htm", "FY2027", "44819098"],
            ["UCUM", "UCUM", "http://unitsofmeasure.org", "1.8.2", "44819107"],
            ["OSM", "OpenStreetMap", "https://www.openstreetmap.org", "2019", "32541"],
        ],
    )

    _write_tsv(
        base / "RELATIONSHIP.csv",
        [
            "relationship_id",
            "relationship_name",
            "is_hierarchical",
            "defines_ancestry",
            "reverse_relationship_id",
            "relationship_concept_id",
        ],
        [
            ["Is a", "Is a", "0", "0", "Subsumes", "44818721"],
            ["Subsumes", "Subsumes", "1", "1", "Is a", "44818821"],
            ["Maps to", "Maps to", "0", "0", "Mapped from", "44818722"],
            ["Mapped from", "Mapped from", "0", "0", "Maps to", "44818822"],
            ["Has finding site", "Has finding site (SNOMED)", "0", "0", "Finding site of", "44818723"],
            ["Finding site of", "Finding site of (SNOMED)", "0", "0", "Has finding site", "44818823"],
            ["Has status", "Has status (SNOMED)", "0", "0", "Status of", "44818724"],
            ["Status of", "Status of (SNOMED)", "0", "0", "Has status", "44818824"],
            ["Concept replaced by", "Concept replaced by", "0", "0", "Concept replaces", "44818725"],
            ["Concept replaces", "Concept replaces", "0", "0", "Concept replaced by", "44818825"],
        ],
    )

    # 8 concepts:
    # 1001: SNOMED Condition (valid, standard) — Pulmonary embolism
    # 1002: SNOMED Condition (valid, standard) — Venous thromboembolism (parent of 1001)
    # 1003: SNOMED Spec Anatomic Site (valid, standard) — Pulmonary artery structure
    # 1004: SNOMED Condition (deprecated 'U') — Legacy pulmonary embolism
    # 2001: ICD10CM Condition (valid, non-standard) — I26.99 Other pulmonary embolism
    # 3001: LOINC Procedure (valid, standard) — CT Chest for screening
    # 4001: UCUM Unit (valid, standard) — mm
    # 5001: OSM Geography (noise vocab, should be excluded by default)
    _write_tsv(
        base / "CONCEPT.csv",
        [
            "concept_id",
            "concept_name",
            "domain_id",
            "vocabulary_id",
            "concept_class_id",
            "standard_concept",
            "concept_code",
            "valid_start_date",
            "valid_end_date",
            "invalid_reason",
        ],
        [
            [
                "1001",
                "Pulmonary embolism",
                "Condition",
                "SNOMED",
                "Disorder",
                "S",
                "59282003",
                "20020131",
                "20991231",
                "",
            ],
            [
                "1002",
                "Venous thromboembolism",
                "Condition",
                "SNOMED",
                "Disorder",
                "S",
                "429098002",
                "20020131",
                "20991231",
                "",
            ],
            [
                "1003",
                "Pulmonary artery structure",
                "Spec Anatomic Site",
                "SNOMED",
                "Body Structure",
                "S",
                "81040000",
                "20020131",
                "20991231",
                "",
            ],
            [
                "1004",
                "Legacy pulmonary embolism",
                "Condition",
                "SNOMED",
                "Disorder",
                "",
                "9990001",
                "20020131",
                "20180131",
                "U",
            ],
            [
                "2001",
                "Other pulmonary embolism without acute cor pulmonale",
                "Condition",
                "ICD10CM",
                "5-char billing code",
                "",
                "I26.99",
                "20151001",
                "20991231",
                "",
            ],
            [
                "3001",
                "CT Chest for screening",
                "Procedure",
                "LOINC",
                "Clinical Observation",
                "S",
                "87279-6",
                "20180101",
                "20991231",
                "",
            ],
            ["4001", "millimeter", "Unit", "UCUM", "Unit", "S", "mm", "19700101", "20991231", ""],
            ["5001", "San Francisco", "Geography", "OSM", "City", "S", "osm-sf", "20190221", "20991231", ""],
        ],
    )

    _write_tsv(
        base / "CONCEPT_SYNONYM.csv",
        ["concept_id", "concept_synonym_name", "language_concept_id"],
        [
            ["1001", "PE", "4180186"],  # English
            ["1001", "Pulmonary embolism", "4180186"],  # Identical to display -> omitted
            ["1001", "Blood clot in lung artery", "4180186"],  # English
            ["1001", "Embolia pulmonar", "4182511"],  # Spanish -> filtered by default
            ["5001", "SF", "4180186"],  # Unselected OSM concept
        ],
    )

    _write_tsv(
        base / "CONCEPT_METADATA.csv",
        ["concept_id", "concept_category", "reuse_status"],
        [
            ["1001", "A", ""],
            ["3001", "A", ""],
        ],
    )

    _write_tsv(
        base / "CONCEPT_RELATIONSHIP.csv",
        [
            "concept_id_1",
            "concept_id_2",
            "relationship_id",
            "valid_start_date",
            "valid_end_date",
            "invalid_reason",
        ],
        [
            # Self-loop (must be skipped)
            ["1001", "1001", "Maps to", "20020131", "20991231", ""],
            # IS_A + mirror Subsumes (only IS_A kept)
            ["1001", "1002", "Is a", "20020131", "20991231", ""],
            ["1002", "1001", "Subsumes", "20020131", "20991231", ""],
            # Cross-vocabulary MAPS_TO + mirror Mapped from (only MAPS_TO kept)
            ["2001", "1001", "Maps to", "20151001", "20991231", ""],
            ["1001", "2001", "Mapped from", "20151001", "20991231", ""],
            # Defining attribute HAS_ATTRIBUTE + mirror
            ["1001", "1003", "Has finding site", "20020131", "20991231", ""],
            ["1003", "1001", "Finding site of", "20020131", "20991231", ""],
            # SNOMED RF2 metadata edge (skipped by default)
            ["1001", "1002", "Has status", "20020131", "20991231", ""],
            ["1002", "1001", "Status of", "20020131", "20991231", ""],
            # Historical replacement edge + mirror
            ["1004", "1001", "Concept replaced by", "20180131", "20991231", ""],
            ["1001", "1004", "Concept replaces", "20180131", "20991231", ""],
            # Deprecated relationship row (must be skipped)
            ["3001", "1001", "Is a", "20100101", "20150101", "D"],
        ],
    )

    _write_tsv(
        base / "CONCEPT_RELATIONSHIP_METADATA.csv",
        [
            "concept_id_1",
            "concept_id_2",
            "relationship_id",
            "relationship_predicate_id",
            "relationship_group",
            "mapping_source",
            "confidence",
            "mapping_tool",
            "mapper",
            "reviewer",
        ],
        [
            ["2001", "1001", "Maps to", "exactMatch", "", "OHDSI", "1", "", "", ""],
        ],
    )

    return base


def test_dry_run_full_valid_and_demo_core_profiles(tmp_path: Path) -> None:
    seed_dir = write_synthetic_athena_dir(tmp_path / "athena")
    service = ClinicalTermsIngesterService(client=MagicMock())

    # 1. FULL profile: SNOMED, LOINC, ICD10CM, UCUM (excludes OSM); includes deprecated 1004
    full_report = service.ingest_sync(
        ClinicalTermsIngestionOptions(
            source_dir=seed_dir,
            profile=ClinicalIngestionProfile.FULL,
            dry_run=True,
        )
    )
    assert full_report.dry_run is True
    assert full_report.vocabularies_ingested == ["ICD10CM", "LOINC", "SNOMED", "UCUM"]
    assert full_report.node_counts == {"CodeSystem": 4, "Concept": 7}
    assert full_report.relation_counts == {
        "BELONGS_TO_SYSTEM": 7,
        "IS_A": 1,
        "MAPS_TO": 1,
        "HAS_ATTRIBUTE": 1,
        "REPLACED_BY": 1,
    }
    assert full_report.synonyms_attached == 2  # English only, excluding exact display match
    assert full_report.skipped_counts["CONCEPT_FILTERED_VOCABULARY"] == 1  # OSM
    assert full_report.skipped_counts["SYNONYM_FILTERED_LANGUAGE"] == 1  # Spanish
    assert full_report.skipped_counts["REL_SELF_LOOP"] == 1
    assert full_report.skipped_counts["REL_INVALID"] == 1
    # 5 valid Standard concepts (1001, 1002, 1003, 3001, 4001)
    assert full_report.embedding_candidates_count == 5

    # 2. DEMO_CORE profile: excludes deprecated 1004 -> 6 valid concepts across demo domains
    demo_report = service.ingest_sync(
        ClinicalTermsIngestionOptions(
            source_dir=seed_dir,
            profile=ClinicalIngestionProfile.DEMO_CORE,
            dry_run=True,
        )
    )
    assert demo_report.node_counts == {"CodeSystem": 4, "Concept": 6}
    assert demo_report.relation_counts["REPLACED_BY"] == 0
    assert demo_report.skipped_counts["CONCEPT_FILTERED_DEPRECATED"] == 1

    # 3. Opt-in RF2 metadata edges + all languages
    rf2_report = service.ingest_sync(
        ClinicalTermsIngestionOptions(
            source_dir=seed_dir,
            profile=ClinicalIngestionProfile.FULL,
            include_rf2_metadata_edges=True,
            all_synonym_languages=True,
            dry_run=True,
        )
    )
    assert rf2_report.relation_counts["HAS_ATTRIBUTE"] == 2  # Has finding site + Has status
    assert rf2_report.synonyms_attached == 3  # PE, Blood clot..., Embolia pulmonar


def test_ingester_writes_batched_cypher_and_skms_metadata_with_mock_client(tmp_path: Path) -> None:
    seed_dir = write_synthetic_athena_dir(tmp_path / "athena")
    executed_queries: List[tuple[str, List[Dict[str, Any]]]] = []

    class _FakeRecord(dict):
        pass

    class _FakeResult:
        def __init__(self, rec: Dict[str, Any]) -> None:
            self._rec = _FakeRecord(rec)

        def single(self) -> _FakeRecord:
            return self._rec

    class _FakeTx:
        def run(self, query: str, params: Dict[str, Any]) -> _FakeResult:
            rows = params.get("rows", [])
            executed_queries.append((query, rows))
            if "concepts_written" in query:
                return _FakeResult({"concepts_written": len(rows), "links_written": len(rows)})
            return _FakeResult({"written": len(rows)})

    mock_client = MagicMock()
    mock_client.execute_write_tx_sync.side_effect = lambda fn: fn(_FakeTx())

    service = ClinicalTermsIngesterService(client=mock_client)
    report = service.ingest_sync(
        ClinicalTermsIngestionOptions(
            source_dir=seed_dir,
            batch_size=3,
            ensure_schema=False,
            test_id="unit_clinical_mock",
            acted_by="pytest_unit",
        )
    )

    assert report.dry_run is False
    assert report.node_counts == {"CodeSystem": 4, "Concept": 7}
    assert report.relation_counts == {
        "BELONGS_TO_SYSTEM": 7,
        "IS_A": 1,
        "MAPS_TO": 1,
        "HAS_ATTRIBUTE": 1,
        "REPLACED_BY": 1,
    }

    # Inspect the MAPS_TO batch payload to verify SKOS metadata enrichment
    maps_to_batches = [rows for q, rows in executed_queries if "MERGE (a)-[r:MAPS_TO" in q]
    assert len(maps_to_batches) == 1
    assert len(maps_to_batches[0]) == 1
    edge_props = maps_to_batches[0][0]["props"]
    assert edge_props["relationship_id"] == "Maps to"
    assert edge_props["equivalence"] == "exactMatch"
    assert edge_props["confidence"] == 1.0
    assert edge_props["mapping_source"] == "OHDSI"


def test_clinical_terms_ingestion_actor_and_embedding_delegation(tmp_path: Path) -> None:
    seed_dir = write_synthetic_athena_dir(tmp_path / "athena")

    class _FakeTx:
        def run(self, query: str, params: Dict[str, Any]) -> Any:
            rows = params.get("rows", [])
            res = MagicMock()
            if "concepts_written" in query:
                res.single.return_value = {"concepts_written": len(rows), "links_written": len(rows)}
            else:
                res.single.return_value = {"written": len(rows)}
            return res

    mock_client = MagicMock()
    mock_client.execute_write_tx_sync.side_effect = lambda fn: fn(_FakeTx())

    mock_sys = MagicMock()
    mock_sys.cleanup_test_data_sync.return_value = {"deleted": 7, "message": "Cleaned test_id"}

    mock_emb_svc = MagicMock()
    mock_emb_svc.embed_items_sync.side_effect = lambda items, **kwargs: EmbeddingBatchReport(
        total_items=len(items),
        embedded_count=len(items),
        skipped_existing_count=0,
        persisted_count=len(items),
        model_tag="embedding-v1",
        provider="ollama",
        model="embeddinggemma-2:270m",
        dimensions=768,
        test_id=kwargs.get("test_id"),
    )

    ingester = ClinicalTermsIngesterService(
        client=mock_client,
        system_service=mock_sys,
        embedding_service=mock_emb_svc,
    )

    emb_actor = EmbeddingActor.start(service=mock_emb_svc)
    ingest_actor = ClinicalTermsIngestionActor.start(ingester=ingester, embedding_actor_ref=emb_actor)
    try:
        res = ingest_actor.ask(
            ClinicalTermsIngestMessage(
                source_dir=str(seed_dir),
                ensure_schema=False,
                compute_embeddings=True,
                embedding_dispatch_chunk_size=2,
                test_id="unit_actor_clinical",
            )
        )
        assert res.status == STATUS_OK, res.message
        assert res.value is not None
        assert res.value.node_counts["Concept"] == 7
        assert res.value.embedding_report is not None
        assert res.value.embedding_report.persisted_count == 5

        cleanup_res = ingest_actor.ask(ClinicalTermsCleanupMessage(test_id="unit_actor_clinical"))
        assert cleanup_res.status == STATUS_OK
        mock_sys.cleanup_test_data_sync.assert_called_once_with("unit_actor_clinical")

        bad_res = ingest_actor.ask("unsupported_payload")
        assert bad_res.status == STATUS_BAD_REQUEST
    finally:
        ingest_actor.stop()
        emb_actor.stop()


def test_build_concept_embedding_items_contextual_and_synonyms() -> None:
    items = build_concept_embedding_items(
        concept_id="cid-1",
        vocabulary_id="SNOMED",
        domain_id="Condition",
        code="59282003",
        display="Pulmonary embolism",
        synonyms=["PE", "Blood clot in lung artery"],
        contextual=True,
        embed_synonyms=True,
    )
    assert len(items) == 2
    assert items[0].target_field_tag == "DISPLAY"
    assert items[0].text == "SNOMED Condition (59282003): Pulmonary embolism"
    assert items[1].target_field_tag == "SYNONYM"
    assert "PE; Blood clot in lung artery" in items[1].text


def test_missing_source_directory_raises_file_not_found(tmp_path: Path) -> None:
    service = ClinicalTermsIngesterService(client=MagicMock())
    with pytest.raises(FileNotFoundError):
        service.ingest_sync(ClinicalTermsIngestionOptions(source_dir=tmp_path / "does_not_exist", dry_run=True))


def test_cli_ingest_clinical_terms_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    seed_dir = write_synthetic_athena_dir(tmp_path / "athena_cli")
    exit_code = cli_main(
        [
            "ingest-clinical-terms",
            "--source-dir",
            str(seed_dir),
            "--profile",
            "demo-core",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["profile"] == "demo-core"
    assert payload["node_counts"] == {"CodeSystem": 4, "Concept": 6}
    assert payload["relation_counts"]["IS_A"] == 1
    assert payload["relation_counts"]["MAPS_TO"] == 1
