"""Unit tests for DicomIngesterService, DicomIngestionActor, and CLI ingest-dicom."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from gosmart_dictionary_service.__main__ import main as cli_main
from gosmart_dictionary_service.actor.dicom_ingestion_actor import DicomIngestionActor
from gosmart_dictionary_service.actor.embedding_actor import EmbeddingActor
from gosmart_dictionary_service.ingestor.dicom_ingestor import (
    DicomIngesterService,
    DicomIngestionOptions,
    extract_dicom_embedding_items,
    load_dicom_seed_bundle,
)
from gosmart_dictionary_service.model.actor_message import DicomCleanupMessage, DicomIngestMessage
from gosmart_dictionary_service.model.llm_config import LLM_Config
from gosmart_dictionary_service.model.result import STATUS_BAD_REQUEST, STATUS_OK
from gosmart_dictionary_service.service.embedding_service import EmbeddingBatchReport, EmbeddingService
from tests.conftest import make_test_settings


def write_synthetic_dicom_seed_dir(target_dir: Path) -> Path:
    """Create a complete 11-file synthetic Innolitics DICOM JSON fixture in ``target_dir``."""
    target_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {
        "attributes.json": [
            {
                "id": "00100010",
                "tag": "(0010,0010)",
                "name": "Patient's Name",
                "keyword": "PatientName",
                "valueRepresentation": "PN",
                "valueMultiplicity": "1",
                "retired": "N",
            },
            {
                "id": "00180050",
                "tag": "(0018,0050)",
                "name": "Slice Thickness",
                "keyword": "SliceThickness",
                "valueRepresentation": "DS",
                "valueMultiplicity": "1",
                "retired": "N",
            },
            {
                "id": "002031xx",
                "tag": "(0020,31xx)",
                "name": "",
                "keyword": "",
                "valueRepresentation": "",
                "valueMultiplicity": "",
                "retired": "Y",
            },
        ],
        "ciods.json": [
            {
                "id": "ct-image",
                "name": "CT Image",
                "description": "<p>Computed Tomography <b>Image</b> IOD.</p>",
                "linkToStandard": "https://dicom.nema.org/ps3.3#sect_A.3",
            }
        ],
        "modules.json": [
            {
                "id": "patient",
                "name": "Patient",
                "description": "<p>Specifies Patient identifying attributes.</p>",
                "linkToStandard": "https://dicom.nema.org/ps3.3#sect_C.7.1.1",
            },
            {
                "id": "image-plane",
                "name": "Image Plane",
                "description": "<p>Specifies physical slice geometry.</p>",
                "linkToStandard": "https://dicom.nema.org/ps3.3#sect_C.7.6.2",
            },
        ],
        "macros.json": [
            {
                "id": "pixel-measures",
                "name": "Pixel Measures",
                "description": "<p>Defines physical spacing of pixels.</p>",
                "linkToStandard": "https://dicom.nema.org/ps3.3#sect_C.7.6.16.2.1",
            }
        ],
        "sops.json": [
            {
                "id": "1.2.840.10008.5.1.4.1.1.2",
                "name": "CT Image Storage",
                "ciod": "CT Image",
            }
        ],
        "confidentiality_profile_attributes.json": [
            {
                "id": "00100010",
                "tag": "(0010,0010)",
                "name": "Patient's Name",
                "stdCompIOD": "Y",
                "basicProfile": "Z",
                "cleanDescOpt": None,
                "cleanGraphOpt": None,
                "cleanStructContOpt": None,
                "rtnDevIdOpt": None,
                "rtnInstIdOpt": None,
                "rtnLongFullDatesOpt": None,
                "rtnLongModifDatesOpt": None,
                "rtnPatCharsOpt": None,
                "rtnSafePrivOpt": None,
                "rtnUIDsOpt": None,
            },
            {
                "id": "ggggeeee-where-gggg-is-odd",
                "tag": "(gggg,eeee) where gggg is odd",
                "name": "Private attributes",
                "stdCompIOD": "N",
                "basicProfile": "X",
                "rtnSafePrivOpt": "C",
            },
        ],
        "references.json": {
            "https://dicom.nema.org/ps3.3#sect_C.7.1.1": "<p>Patient Module Reference</p>",
            "https://dicom.nema.org/ps3.3#sect_C.7.6.2": "<p>Image Plane Reference</p>",
        },
        "ciod_to_modules.json": [
            {
                "ciodId": "ct-image",
                "moduleId": "patient",
                "usage": "M",
                "conditionalStatement": None,
                "informationEntity": "Patient",
            },
            {
                "ciodId": "ct-image",
                "moduleId": "image-plane",
                "usage": "M",
                "conditionalStatement": None,
                "informationEntity": "Image",
            },
        ],
        "ciod_to_func_group_macros.json": [
            {
                "ciodId": "ct-image",
                "macroId": "pixel-measures",
                "usage": "M",
                "conditionalStatement": None,
                "moduleType": "Multi-frame",
            }
        ],
        "module_to_attributes.json": [
            {
                "moduleId": "patient",
                "path": "patient:00100010",
                "tag": "(0010,0010)",
                "type": "2",
                "linkToStandard": "https://dicom.nema.org/ps3.3#table_C.7-1",
                "description": "<p>Patient's full legal name.</p>",
                "externalReferences": [
                    {
                        "sourceUrl": "https://dicom.nema.org/ps3.3#sect_C.7.1.1",
                        "title": "Section C.7.1.1",
                    }
                ],
            },
            {
                "moduleId": "patient",
                "path": "patient:002031xx",
                "tag": "(0020,31xx)",
                "type": "3",
                "linkToStandard": "https://dicom.nema.org/ps3.3#table_C.7-1",
                "description": "<p>Repeating group attribute.</p>",
                "externalReferences": [
                    {
                        "sourceUrl": "https://dicom.nema.org/ps3.3#sect_C.7.1.1",
                        "title": "Section C.7.1.1",
                    }
                ],
            },
            {
                "moduleId": "image-plane",
                "path": "image-plane:00180050",
                "tag": "(0018,0050)",
                "type": "2",
                "linkToStandard": "https://dicom.nema.org/ps3.3#table_C.7-10",
                "description": "<p>Nominal slice thickness, in mm.</p>",
                "externalReferences": [
                    {
                        "sourceUrl": "https://dicom.nema.org/ps3.3#sect_C.7.6.2",
                        "title": "Section C.7.6.2",
                    }
                ],
            },
        ],
        "macro_to_attributes.json": [
            {
                "macroId": "pixel-measures",
                "path": "pixel-measures:00180050",
                "tag": "(0018,0050)",
                "type": "1C",
                "linkToStandard": "https://dicom.nema.org/ps3.3#table_C.7.6.16-2",
                "description": "<p>Nominal reconstructed slice thickness, in mm.</p>",
                "externalReferences": [
                    {
                        "sourceUrl": "https://dicom.nema.org/ps3.3#sect_C.7.6.2",
                        "title": "Section C.7.6.2",
                    }
                ],
            }
        ],
    }

    for filename, payload in files.items():
        (target_dir / filename).write_text(json.dumps(payload), encoding="utf-8")
    return target_dir


def test_load_dicom_seed_bundle_and_embedding_extraction(tmp_path: Path) -> None:
    seed_dir = write_synthetic_dicom_seed_dir(tmp_path / "dicom_seed")

    bundle = load_dicom_seed_bundle(seed_dir, test_id="test_bundle", dedupe_reference_edges_per_owner=True)
    assert len(bundle.elements) == 3
    assert len(bundle.ciods) == 1
    assert len(bundle.modules) == 2
    assert len(bundle.macros) == 1
    assert len(bundle.sop_classes) == 1
    assert len(bundle.confidentiality_profiles) == 2
    assert len(bundle.standard_references) == 2

    assert len(bundle.implements_ciod) == 1
    assert len(bundle.has_module) == 2
    assert len(bundle.has_func_group_macro) == 1
    assert len(bundle.module_uses_element) == 3
    assert len(bundle.macro_uses_element) == 1
    assert len(bundle.has_confidentiality_profile) == 1
    assert bundle.skipped_edges["HAS_CONFIDENTIALITY_PROFILE_UNMATCHED_WILDCARD"] == 1
    # Deduped: patient referenced sect_C.7.1.1 twice -> 1 edge; image-plane -> 1 edge
    assert len(bundle.module_references_standard) == 2
    assert len(bundle.macro_references_standard) == 1

    # Without deduping, both patient attributes emit a path-scoped edge
    bundle_no_dedupe = load_dicom_seed_bundle(seed_dir, test_id="test_bundle", dedupe_reference_edges_per_owner=False)
    assert len(bundle_no_dedupe.module_references_standard) == 3

    # Contextual embedding items (Option A):
    # Elements: 2 non-empty names (retired empty name skipped) = 2
    # CIOD: 1 name + 1 desc = 2
    # Modules: 2 names + 2 descs = 4
    # Macros: 1 name + 1 desc = 2
    # SOP: 1 name = 1
    # Confidentiality profiles: 2 names = 2
    # Total = 13
    contextual_items = extract_dicom_embedding_items(bundle, contextual=True)
    assert len(contextual_items) == 13
    texts = [item.text for item in contextual_items]
    assert "DICOM Element (0010,0010) PatientName: Patient's Name" in texts
    assert "DICOM CIOD CT Image: Computed Tomography Image IOD." in texts
    assert "DICOM Module: Image Plane" in texts
    assert "DICOM SOP Class 1.2.840.10008.5.1.4.1.1.2 (CT Image): CT Image Storage" in texts

    raw_items = extract_dicom_embedding_items(bundle, contextual=False)
    assert len(raw_items) == 13
    assert "Patient's Name" in [item.text for item in raw_items]


def test_dicom_ingester_dry_run_and_batched_writes(tmp_path: Path) -> None:
    seed_dir = write_synthetic_dicom_seed_dir(tmp_path / "dicom_seed")
    mock_client = MagicMock()
    mock_sys = MagicMock()
    mock_emb = MagicMock(spec=EmbeddingService)

    ingester = DicomIngesterService(
        client=mock_client,
        system_service=mock_sys,
        embedding_service=mock_emb,
        settings=make_test_settings(),
    )

    dry_report = ingester.ingest_sync(DicomIngestionOptions(source_dir=seed_dir, dry_run=True, test_id="dry_1"))
    assert dry_report.dry_run is True
    assert dry_report.node_counts["DicomElement"] == 3
    assert dry_report.relation_counts["USES_ELEMENT"] == 4
    assert dry_report.embedding_candidates_count == 13
    assert dry_report.embedding_report is None
    mock_client.execute_write_tx_sync.assert_not_called()

    # Live-mocked batched write with batch_size=2 and compute_embeddings=True
    executed_queries: list[tuple[str, int]] = []

    def fake_write_tx(tx_fn):
        tx = MagicMock()

        def tx_run(query: str, params: dict[str, Any]):
            rows = params["rows"]
            executed_queries.append((query, len(rows)))
            res = MagicMock()
            res.single.return_value = {"written": len(rows)}
            return res

        tx.run.side_effect = tx_run
        return tx_fn(tx)

    mock_client.execute_write_tx_sync.side_effect = fake_write_tx
    mock_emb.embed_items_sync.return_value = EmbeddingBatchReport(
        total_items=13,
        embedded_count=13,
        skipped_existing_count=0,
        persisted_count=13,
        model_tag="embedding-v1",
        provider="ollama",
        model="embeddinggemma-2:270m",
        dimensions=768,
    )

    report = ingester.ingest_sync(
        DicomIngestionOptions(
            source_dir=seed_dir,
            batch_size=2,
            ensure_schema=False,
            bootstrap_system=True,
            compute_embeddings=True,
            test_id="batch_test",
        )
    )
    assert report.dry_run is False
    assert sum(report.node_counts.values()) == 12
    assert sum(report.relation_counts.values()) == 12
    assert report.embedding_report is not None
    assert report.embedding_report.persisted_count == 13
    mock_sys.bootstrap_system_sync.assert_called_once()
    mock_emb.embed_items_sync.assert_called_once()
    # Confirm batch chunking occurred (3 elements with batch_size=2 -> 2 batches of 2 and 1)
    element_batches = [size for q, size in executed_queries if "MERGE (n:DicomElement" in q]
    assert element_batches == [2, 1]


def test_dicom_ingestion_actor_orchestrates_embedding_actor(tmp_path: Path) -> None:
    seed_dir = write_synthetic_dicom_seed_dir(tmp_path / "dicom_seed")
    mock_client = MagicMock()
    mock_client.settings = make_test_settings()

    def fake_write_tx(tx_fn):
        tx = MagicMock()

        def tx_run(_query: str, params: dict[str, Any]):
            res = MagicMock()
            res.single.return_value = {"written": len(params["rows"])}
            return res

        tx.run.side_effect = tx_run
        return tx_fn(tx)

    mock_client.execute_write_tx_sync.side_effect = fake_write_tx
    mock_session = MagicMock()
    mock_session.run.return_value = []
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_client.get_sync_driver.return_value = mock_driver

    mock_llm_svc = MagicMock()
    mock_llm_svc.get_llm_config_by_tag_sync.return_value = LLM_Config(
        tag="embedding-v1",
        provider="ollama",
        model="embeddinggemma-2:270m",
        url="http://m5.local.:11434",
        dimensions=4,
    )
    emb_service = EmbeddingService(
        client=mock_client,
        llm_config_service=mock_llm_svc,
        settings=make_test_settings(),
        embed_fn=lambda texts, _cfg: [[1.0, 2.0, 3.0, 4.0] for _ in texts],
    )
    mock_sys = MagicMock()
    mock_sys.cleanup_test_data_sync.return_value = {"success": True, "total_deleted": 25, "message": "Cleaned 25"}

    ingester = DicomIngesterService(
        client=mock_client,
        system_service=mock_sys,
        embedding_service=emb_service,
        settings=make_test_settings(),
    )

    emb_actor = EmbeddingActor.start(service=emb_service)
    ingest_actor = DicomIngestionActor.start(ingester=ingester, embedding_actor_ref=emb_actor)
    try:
        res = ingest_actor.ask(
            DicomIngestMessage(
                source_dir=str(seed_dir),
                ensure_schema=False,
                compute_embeddings=True,
                embedding_batch_size=5,
                test_id="actor_flow",
            )
        )
        assert res.status == STATUS_OK
        assert res.value is not None
        assert res.value.embedding_report is not None
        assert res.value.embedding_report.persisted_count == 13

        cleanup_res = ingest_actor.ask(DicomCleanupMessage(test_id="actor_flow"))
        assert cleanup_res.status == STATUS_OK
        assert cleanup_res.value["total_deleted"] == 25

        bad_res = ingest_actor.ask({"type": "CLEANUP_TEST_DATA", "test_id": "   "})
        assert bad_res.status == STATUS_BAD_REQUEST
    finally:
        ingest_actor.stop()
        emb_actor.stop()


def test_cli_ingest_dicom_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    seed_dir = write_synthetic_dicom_seed_dir(tmp_path / "dicom_seed")
    exit_code = cli_main(["ingest-dicom", "--source-dir", str(seed_dir), "--dry-run"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["node_counts"]["DicomElement"] == 3
    assert payload["embedding_candidates_count"] == 13
