"""Optional live Neo4j (and optional live Ollama) integration tests for Phase 2 DICOM ingestion & embeddings.

Follows AGENTS.md testing policy:
- Skips cleanly if local Neo4j is unreachable.
- Uses an isolated ``test_id`` so production data is never touched.
- Cleans up prior test artifacts **before** running the test (NOT after) so post-run graph
  state remains inspectable in Neo4j Browser.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from gosmart_dictionary_service.actor.dicom_ingestion_actor import DicomIngestionActor
from gosmart_dictionary_service.actor.embedding_actor import EmbeddingActor
from gosmart_dictionary_service.core.config import get_settings
from gosmart_dictionary_service.db.neo4j import Neo4jClient
from gosmart_dictionary_service.ingestor.dicom_ingestor import DicomIngesterService
from gosmart_dictionary_service.model.actor_message import DicomIngestMessage
from gosmart_dictionary_service.model.llm_config import LLM_Config
from gosmart_dictionary_service.model.result import STATUS_OK
from gosmart_dictionary_service.service.embedding_service import EmbeddingService
from gosmart_dictionary_service.service.llm_config_service import LLMConfigService
from gosmart_dictionary_service.service.system_service import SystemService
from tests.unit.test_dicom_ingestor import write_synthetic_dicom_seed_dir

PHASE2_INTEGRATION_TEST_ID = "test_phase2_dicom_pipeline"


def _get_live_client_or_skip() -> Neo4jClient:
    get_settings.cache_clear()
    settings = get_settings()
    client = Neo4jClient(settings=settings)
    try:
        driver = client.get_sync_driver()
        driver.verify_connectivity()
    except Exception as exc:
        if client._sync_driver is not None:
            client._sync_driver.close()
        pytest.skip(f"Live Neo4j at {settings.DB_CONNECTION_URI} is unavailable: {exc}")
    return client


def _deterministic_768_embedder(texts, config: LLM_Config):
    dims = config.dimensions_truncated or config.dimensions or 768
    vectors = []
    for text in texts:
        seed = sum(ord(ch) for ch in text)
        vec = [math.sin(seed + idx + 1) + 1.5 for idx in range(dims)]
        vectors.append(vec)
    return vectors


def test_phase2_dicom_ingestion_and_embedding_pipeline_live_neo4j(tmp_path: Path) -> None:
    client = _get_live_client_or_skip()
    try:
        settings = client.settings
        llm_service = LLMConfigService(client=client)
        sys_service = SystemService(client=client, llm_service=llm_service)
        emb_service = EmbeddingService(
            client=client,
            llm_config_service=llm_service,
            settings=settings,
            embed_fn=_deterministic_768_embedder,
        )
        ingester = DicomIngesterService(
            client=client,
            system_service=sys_service,
            embedding_service=emb_service,
            settings=settings,
        )

        # Clean up ONLY prior test data BEFORE running the test (leave post-test data for inspection)
        sys_service.cleanup_test_data_sync(PHASE2_INTEGRATION_TEST_ID)

        seed_dir = write_synthetic_dicom_seed_dir(tmp_path / "dicom_seed")

        emb_actor = EmbeddingActor.start(service=emb_service)
        ingest_actor = DicomIngestionActor.start(ingester=ingester, embedding_actor_ref=emb_actor)
        try:
            first_res = ingest_actor.ask(
                DicomIngestMessage(
                    source_dir=str(seed_dir),
                    batch_size=2,
                    acted_by="pytest_phase2",
                    test_id=PHASE2_INTEGRATION_TEST_ID,
                    ensure_schema=True,
                    bootstrap_system=True,
                    compute_embeddings=True,
                    embedding_batch_size=4,
                    skip_existing_embeddings=True,
                )
            )
            assert first_res.status == STATUS_OK, first_res.message
            first = first_res.value
            assert first is not None
            assert sum(first.node_counts.values()) == 12
            assert sum(first.relation_counts.values()) == 12
            assert first.embedding_report is not None
            assert first.embedding_report.embedded_count == 13
            assert first.embedding_report.persisted_count == 13
            assert first.embedding_report.skipped_existing_count == 0
            assert first.embedding_report.dimensions == settings.VECTOR_EMBED_DIMENSIONS

            # Second run verifies strict graph idempotency + incremental embedding skip
            second_res = ingest_actor.ask(
                DicomIngestMessage(
                    source_dir=str(seed_dir),
                    batch_size=2,
                    acted_by="pytest_phase2",
                    test_id=PHASE2_INTEGRATION_TEST_ID,
                    ensure_schema=False,
                    bootstrap_system=False,
                    compute_embeddings=True,
                    embedding_batch_size=4,
                    skip_existing_embeddings=True,
                )
            )
            assert second_res.status == STATUS_OK, second_res.message
            second = second_res.value
            assert second is not None
            assert second.node_counts == first.node_counts
            assert second.relation_counts == first.relation_counts
            assert second.embedding_report is not None
            assert second.embedding_report.embedded_count == 0
            assert second.embedding_report.skipped_existing_count == 13

            # Verify live Neo4j counts for this test_id (zero duplicates)
            driver = client.get_sync_driver()
            with driver.session(database=settings.DB_DATABASE) as session:
                emb_count = session.run(
                    "MATCH (e:Embedding {test_id: $tid})-[r:EMBEDS]->(t) RETURN count(r) AS cnt",
                    {"tid": PHASE2_INTEGRATION_TEST_ID},
                ).single()["cnt"]
                elem_count = session.run(
                    "MATCH (n:DicomElement {test_id: $tid}) RETURN count(n) AS cnt",
                    {"tid": PHASE2_INTEGRATION_TEST_ID},
                ).single()["cnt"]
                uses_count = session.run(
                    "MATCH ()-[r:USES_ELEMENT {test_id: $tid}]->(:DicomElement) RETURN count(r) AS cnt",
                    {"tid": PHASE2_INTEGRATION_TEST_ID},
                ).single()["cnt"]

            assert emb_count == 13
            assert elem_count == 3
            assert uses_count == 4
        finally:
            ingest_actor.stop()
            emb_actor.stop()
    finally:
        if client._sync_driver is not None:
            client._sync_driver.close()
        get_settings.cache_clear()
