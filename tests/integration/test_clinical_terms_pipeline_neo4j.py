"""Optional live Neo4j integration tests for OHDSI Athena clinical terms ingestion & embeddings.

Follows AGENTS.md testing policy:
- Skips cleanly if local/configured Neo4j is unreachable.
- Uses an isolated ``test_id`` so production data is never touched.
- Cleans up prior test artifacts **before** running the test (NOT after) so post-run graph
  state remains inspectable in Neo4j Browser.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from gosmart_dictionary_service.actor.clinical_terms_ingestion_actor import ClinicalTermsIngestionActor
from gosmart_dictionary_service.actor.embedding_actor import EmbeddingActor
from gosmart_dictionary_service.core.config import get_settings
from gosmart_dictionary_service.db.neo4j import Neo4jClient
from gosmart_dictionary_service.ingestor.clinical_terms_ingestor import ClinicalTermsIngesterService
from gosmart_dictionary_service.model.actor_message import ClinicalTermsIngestMessage
from gosmart_dictionary_service.model.llm_config import LLM_Config
from gosmart_dictionary_service.model.result import STATUS_OK
from gosmart_dictionary_service.service.embedding_service import EmbeddingService
from gosmart_dictionary_service.service.llm_config_service import LLMConfigService
from gosmart_dictionary_service.service.system_service import SystemService
from tests.unit.test_clinical_terms_ingestor import write_synthetic_athena_dir

CLINICAL_INTEGRATION_TEST_ID = "test_phase3_clinical_terms_pipeline"


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


def test_clinical_terms_ingestion_and_embedding_pipeline_live_neo4j(tmp_path: Path) -> None:
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
        ingester = ClinicalTermsIngesterService(
            client=client,
            system_service=sys_service,
            embedding_service=emb_service,
            settings=settings,
        )

        # Clean up ONLY prior test data BEFORE running (keep post-test graph for manual inspection)
        sys_service.cleanup_test_data_sync(CLINICAL_INTEGRATION_TEST_ID)

        seed_dir = write_synthetic_athena_dir(tmp_path / "athena_seed")

        emb_actor = EmbeddingActor.start(service=emb_service)
        ingest_actor = ClinicalTermsIngestionActor.start(ingester=ingester, embedding_actor_ref=emb_actor)
        try:
            first_res = ingest_actor.ask(
                ClinicalTermsIngestMessage(
                    source_dir=str(seed_dir),
                    batch_size=3,
                    acted_by="pytest_clinical",
                    test_id=CLINICAL_INTEGRATION_TEST_ID,
                    ensure_schema=True,
                    bootstrap_system=True,
                    compute_embeddings=True,
                    embedding_batch_size=4,
                    embedding_dispatch_chunk_size=3,
                    skip_existing_embeddings=True,
                )
            )
            assert first_res.status == STATUS_OK, first_res.message
            first = first_res.value
            assert first is not None
            assert first.node_counts == {"CodeSystem": 4, "Concept": 7}
            assert first.relation_counts == {
                "BELONGS_TO_SYSTEM": 7,
                "IS_A": 1,
                "MAPS_TO": 1,
                "HAS_ATTRIBUTE": 1,
                "REPLACED_BY": 1,
            }
            assert first.embedding_report is not None
            assert first.embedding_report.embedded_count == 5
            assert first.embedding_report.persisted_count == 5
            assert first.embedding_report.skipped_existing_count == 0

            # Second run verifies strict graph idempotency + incremental embedding skip
            second_res = ingest_actor.ask(
                ClinicalTermsIngestMessage(
                    source_dir=str(seed_dir),
                    batch_size=3,
                    acted_by="pytest_clinical",
                    test_id=CLINICAL_INTEGRATION_TEST_ID,
                    ensure_schema=False,
                    bootstrap_system=False,
                    compute_embeddings=True,
                    embedding_batch_size=4,
                    embedding_dispatch_chunk_size=3,
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
            assert second.embedding_report.skipped_existing_count == 5

            # Verify live Neo4j topology, crosswalk SKOS metadata, and zero duplicates
            driver = client.get_sync_driver()
            with driver.session(database=settings.DB_DATABASE) as session:
                concept_count = session.run(
                    "MATCH (c:Concept {test_id: $tid}) RETURN count(c) AS cnt",
                    {"tid": CLINICAL_INTEGRATION_TEST_ID},
                ).single()["cnt"]
                crosswalk = session.run(
                    """
                    MATCH (src:Concept {test_id: $tid, vocabulary_id: 'ICD10CM', code: 'I26.99'})
                          -[r:MAPS_TO]->(dst:Concept {test_id: $tid, vocabulary_id: 'SNOMED', code: '59282003'})
                    RETURN r.equivalence AS equivalence, r.confidence AS confidence, dst.display AS display
                    """,
                    {"tid": CLINICAL_INTEGRATION_TEST_ID},
                ).single()
                is_a_parent = session.run(
                    """
                    MATCH (c:Concept {test_id: $tid, code: '59282003'})-[:IS_A]->(p:Concept {test_id: $tid})
                    RETURN p.code AS parent_code
                    """,
                    {"tid": CLINICAL_INTEGRATION_TEST_ID},
                ).single()
                emb_count = session.run(
                    "MATCH (e:Embedding {test_id: $tid})-[r:EMBEDS]->(:Concept {test_id: $tid}) RETURN count(r) AS cnt",
                    {"tid": CLINICAL_INTEGRATION_TEST_ID},
                ).single()["cnt"]

            assert concept_count == 7
            assert crosswalk is not None
            assert crosswalk["equivalence"] == "exactMatch"
            assert crosswalk["confidence"] == 1.0
            assert crosswalk["display"] == "Pulmonary embolism"
            assert is_a_parent is not None
            assert is_a_parent["parent_code"] == "429098002"
            assert emb_count == 5
        finally:
            ingest_actor.stop()
            emb_actor.stop()
    finally:
        if client._sync_driver is not None:
            client._sync_driver.close()
            client._sync_driver = None
