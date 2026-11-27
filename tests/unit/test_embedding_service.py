"""Unit tests for EmbeddingService and EmbeddingActor."""

from __future__ import annotations

import math
from typing import Any, Sequence
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr, ValidationError

from gosmart_dictionary_service.actor.embedding_actor import EmbeddingActor
from gosmart_dictionary_service.model.actor_message import EmbeddingBatchMessage, EmbeddingItem
from gosmart_dictionary_service.model.embedding import DictionaryEmbeddingFieldTag, Embedding
from gosmart_dictionary_service.model.llm_config import LLM_Config
from gosmart_dictionary_service.model.result import STATUS_BAD_REQUEST, STATUS_INTERNAL_ERROR, STATUS_OK
from gosmart_dictionary_service.service.embedding_service import (
    EmbeddingService,
    normalize_and_truncate_vector,
)
from tests.conftest import make_test_settings


def _deterministic_embed_fn(dims: int = 8):
    calls: list[list[str]] = []

    def _embed(texts: Sequence[str], _cfg: LLM_Config) -> list[list[float]]:
        batch = list(texts)
        calls.append(batch)
        return [[float((idx + i + 1) % 7 + 1) for i in range(dims)] for idx, _ in enumerate(batch)]

    return _embed, calls


def test_normalize_and_truncate_vector_validation() -> None:
    vec = normalize_and_truncate_vector([3.0, 4.0, 0.0, 0.0], dimensions_truncated=2, expected_dimensions=2)
    assert len(vec) == 2
    assert math.isclose(vec[0], 0.6, rel_tol=1e-6)
    assert math.isclose(vec[1], 0.8, rel_tol=1e-6)
    assert math.isclose(math.sqrt(sum(v * v for v in vec)), 1.0, rel_tol=1e-6)

    with pytest.raises(ValueError, match="empty"):
        normalize_and_truncate_vector([])

    with pytest.raises(ValueError, match="finite"):
        normalize_and_truncate_vector([1.0, float("nan")])

    with pytest.raises(ValueError, match="bool"):
        normalize_and_truncate_vector([True, 1.0])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="zero-magnitude"):
        normalize_and_truncate_vector([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="exceeds"):
        normalize_and_truncate_vector([1.0, 2.0], dimensions_truncated=4)

    with pytest.raises(ValueError, match="expected"):
        normalize_and_truncate_vector([1.0, 2.0], expected_dimensions=4)


def test_embedding_item_rejects_disallowed_target_label() -> None:
    item = EmbeddingItem(
        target_node_id="elem-1",
        target_label="DicomElement",
        target_field_tag=DictionaryEmbeddingFieldTag.NAME,
        text="DICOM Element (0018,0050) SliceThickness: Slice Thickness",
    )
    assert item.target_field_tag == "NAME"

    with pytest.raises(ValidationError):
        EmbeddingItem(
            target_node_id="x",
            target_label="ArbitraryLabel; DROP DATABASE",
            target_field_tag="NAME",
            text="bad",
        )


def test_embedding_service_genkit_client_builds_and_caches_providers() -> None:
    settings = make_test_settings(
        VERTEX_AI_API_KEY=SecretStr("fake-key"),
        VERTEX_AI_PROJECT_ID="test-proj",
        VERTEX_AI_LOCATION="us-central1",
    )
    svc = EmbeddingService(client=MagicMock(), llm_config_service=MagicMock(), settings=settings)

    ollama_cfg = LLM_Config(
        tag="embedding-v1",
        provider="ollama",
        model="embeddinggemma-2:270m",
        url="http://m5.local.:11434",
        dimensions=768,
    )
    ai1, ref1 = svc._get_or_create_genkit_client(ollama_cfg)
    ai2, ref2 = svc._get_or_create_genkit_client(ollama_cfg)
    assert ai1 is ai2
    assert ref1 == "ollama/embeddinggemma-2:270m"
    assert ref2 == ref1

    google_cfg = LLM_Config(
        tag="embedding-cloud",
        provider="googleai",
        model="gemini-embedding-001",
        api_key="test-google-key",
        dimensions=768,
    )
    ai_g, ref_g = svc._get_or_create_genkit_client(google_cfg)
    assert ai_g is not ai1
    assert ref_g.name == "googleai/gemini-embedding-001"

    vertex_cfg = LLM_Config(
        tag="embedding-vertex",
        provider="vertexai",
        model="gemini-embedding-001",
        api_key="test-vertex-key",
        dimensions=768,
        location="us-central1",
    )
    ai_v, ref_v = svc._get_or_create_genkit_client(vertex_cfg)
    assert ai_v is not ai_g
    assert ref_v.name == "vertexai/gemini-embedding-001"

    bad_cfg = LLM_Config(tag="bad", provider="unsupported_provider", model="m1")
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        svc._get_or_create_genkit_client(bad_cfg)


def test_embedding_service_batching_dedup_skip_existing_and_persistence() -> None:
    cfg = LLM_Config(
        tag="embedding-v1",
        provider="ollama",
        model="embeddinggemma-2:270m",
        url="http://m5.local.:11434",
        dimensions=8,
    )
    embed_fn, embed_calls = _deterministic_embed_fn(dims=8)

    existing_store: dict[str, str] = {}
    persisted_rows: list[dict[str, Any]] = []

    # Pre-seed item1 as already embedded with identical text
    pre_id = Embedding.compute_target_scoped_id("elem-1", "embedding-v1", "NAME", test_id="test_emb")
    existing_store[pre_id] = "DICOM Element (0010,0010) PatientName: Patient's Name"

    mock_client = MagicMock()
    mock_client.settings = make_test_settings()

    def fake_session_run(query: str, params: dict[str, Any]):
        ids = params.get("ids", [])
        return [{"id": eid, "text": existing_store[eid]} for eid in ids if eid in existing_store]

    mock_session = MagicMock()
    mock_session.run.side_effect = fake_session_run
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_client.get_sync_driver.return_value = mock_driver

    def fake_write_tx(tx_fn):
        tx = MagicMock()

        def tx_run(_query: str, params: dict[str, Any]):
            rows = params["rows"]
            persisted_rows.extend(rows)
            for row in rows:
                existing_store[row["embedding_id"]] = row["text"]
            res = MagicMock()
            res.single.return_value = {"written": len(rows)}
            return res

        tx.run.side_effect = tx_run
        return tx_fn(tx)

    mock_client.execute_write_tx_sync.side_effect = fake_write_tx

    mock_llm_svc = MagicMock()
    mock_llm_svc.get_llm_config_by_tag_sync.return_value = cfg

    svc = EmbeddingService(
        client=mock_client,
        llm_config_service=mock_llm_svc,
        settings=make_test_settings(),
        embed_fn=embed_fn,
    )

    items = [
        EmbeddingItem(
            target_node_id="elem-1",
            target_label="DicomElement",
            target_field_tag=DictionaryEmbeddingFieldTag.NAME,
            text="DICOM Element (0010,0010) PatientName: Patient's Name",
        ),
        # Duplicate of elem-1 NAME should be deduplicated before provider call
        EmbeddingItem(
            target_node_id="elem-1",
            target_label="DicomElement",
            target_field_tag=DictionaryEmbeddingFieldTag.NAME,
            text="DICOM Element (0010,0010) PatientName: Patient's Name",
        ),
        EmbeddingItem(
            target_node_id="ciod-1",
            target_label="DicomCIOD",
            target_field_tag=DictionaryEmbeddingFieldTag.NAME,
            text="DICOM CIOD: CT Image",
        ),
        EmbeddingItem(
            target_node_id="ciod-1",
            target_label="DicomCIOD",
            target_field_tag=DictionaryEmbeddingFieldTag.DESCRIPTION,
            text="DICOM CIOD CT Image: Computed tomography image IOD.",
        ),
        EmbeddingItem(
            target_node_id="mod-1",
            target_label="DicomModule",
            target_field_tag=DictionaryEmbeddingFieldTag.NAME,
            text="DICOM Module: Patient",
        ),
    ]

    report = svc.embed_items_sync(
        items,
        tag="embedding-v1",
        batch_size=2,
        skip_existing=True,
        test_id="test_emb",
        acted_by="pytest",
    )

    assert report.total_items == 4
    assert report.skipped_existing_count == 1
    assert report.embedded_count == 3
    assert report.persisted_count == 3
    assert report.dimensions == 8
    assert len(embed_calls) == 2  # 3 pending items split into batches of [2, 1]
    assert len(persisted_rows) == 3

    # Verify auto-scoped test group code lookup
    mock_llm_svc.get_llm_config_by_tag_sync.assert_called_once_with(
        "embedding-v1",
        group_id=None,
        group_code="GOSMART_HEALTH__TEST_EMB",
        group_name="GoSmart Health",
        test_id="test_emb",
    )

    # Re-running with skip_existing=True skips all 4 items with zero new embed calls
    prev_call_count = len(embed_calls)
    report2 = svc.embed_items_sync(items, tag="embedding-v1", batch_size=2, skip_existing=True, test_id="test_emb")
    assert report2.skipped_existing_count == 4
    assert report2.embedded_count == 0
    assert len(embed_calls) == prev_call_count


def test_embedding_actor_success_and_error_handling() -> None:
    mock_svc = MagicMock(spec=EmbeddingService)
    mock_svc.embed_items_sync.return_value = MagicMock(
        total_items=2,
        embedded_count=2,
        skipped_existing_count=0,
        persisted_count=2,
    )
    from gosmart_dictionary_service.service.embedding_service import EmbeddingBatchReport

    mock_svc.embed_items_sync.return_value = EmbeddingBatchReport(
        total_items=2,
        embedded_count=2,
        skipped_existing_count=0,
        persisted_count=2,
        model_tag="embedding-v1",
        provider="ollama",
        model="embeddinggemma-2:270m",
        dimensions=768,
    )

    actor = EmbeddingActor.start(service=mock_svc)
    try:
        msg = EmbeddingBatchMessage(
            items=[
                EmbeddingItem(
                    target_node_id="elem-1",
                    target_label="DicomElement",
                    target_field_tag=DictionaryEmbeddingFieldTag.NAME,
                    text="DICOM Element (0010,0010) PatientName: Patient's Name",
                )
            ]
        )
        res = actor.ask(msg)
        assert res.status == STATUS_OK
        assert res.value is not None
        assert res.value.persisted_count == 2

        # Invalid payload returns STATUS_BAD_REQUEST
        bad_res = actor.ask({"items": [{"target_node_id": "", "target_label": "Bad", "text": ""}]})
        assert bad_res.status == STATUS_BAD_REQUEST

        # Unsupported message type
        wrong_type = actor.ask("not-a-valid-message")
        assert wrong_type.status == STATUS_BAD_REQUEST

        # Runtime failure returns STATUS_INTERNAL_ERROR
        mock_svc.embed_items_sync.side_effect = RuntimeError("Ollama offline")
        err_res = actor.ask(msg)
        assert err_res.status == STATUS_INTERNAL_ERROR
    finally:
        actor.stop()
