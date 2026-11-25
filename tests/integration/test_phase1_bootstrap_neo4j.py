"""Optional live Neo4j integration test for Phase 1 System + GoSmart Health Group + LLM_Config bootstrap.

Follows AGENTS.md testing policy:
- Skips cleanly if local Neo4j is unreachable.
- Uses an isolated ``test_id`` so production data is never touched.
- Cleans up prior test artifacts **before** running the test (NOT after) so post-run graph
  state remains inspectable in Neo4j Browser.
"""

from __future__ import annotations

import pytest

from gosmart_dictionary_service.core.config import get_settings
from gosmart_dictionary_service.db.neo4j import Neo4jClient
from gosmart_dictionary_service.model.llm_config import (
    DEFAULT_EMBEDDING_CONFIG_TAG,
    DEFAULT_INFERENCE_CONFIG_TAG,
)
from gosmart_dictionary_service.service.llm_config_service import LLMConfigService
from gosmart_dictionary_service.service.system_service import (
    DEFAULT_GROUP_NAME,
    SystemService,
    resolve_root_group_code,
)

PHASE1_INTEGRATION_TEST_ID = "test_phase1_bootstrap_integration"


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


def test_phase1_bootstrap_idempotent_against_live_neo4j() -> None:
    client = _get_live_client_or_skip()
    try:
        settings = client.settings
        llm_service = LLMConfigService(client=client)
        sys_service = SystemService(client=client, llm_service=llm_service)

        # Clean up ONLY prior test data BEFORE running the test (leave post-test data for inspection)
        sys_service.cleanup_test_data_sync(PHASE1_INTEGRATION_TEST_ID)

        first = sys_service.bootstrap_system_sync(
            acted_by="pytest_phase1",
            test_id=PHASE1_INTEGRATION_TEST_ID,
            settings=settings,
            ensure_schema=True,
            seed_default_llm_configs=True,
        )
        second = sys_service.bootstrap_system_sync(
            acted_by="pytest_phase1",
            test_id=PHASE1_INTEGRATION_TEST_ID,
            settings=settings,
            ensure_schema=False,
            seed_default_llm_configs=True,
        )

        # Strict deterministic ID idempotency
        assert first["success"] is True
        assert second["success"] is True
        assert first["system"]["id"] == second["system"]["id"]
        assert first["root_group"]["id"] == second["root_group"]["id"]
        assert first["root_group"]["name"] == DEFAULT_GROUP_NAME
        expected_group_code = resolve_root_group_code(PHASE1_INTEGRATION_TEST_ID)
        assert first["root_group"]["code"] == expected_group_code

        first_cfg_ids = {cfg["tag"]: cfg["id"] for cfg in first["llm_configs"]}
        second_cfg_ids = {cfg["tag"]: cfg["id"] for cfg in second["llm_configs"]}
        assert first_cfg_ids == second_cfg_ids
        assert set(first_cfg_ids.keys()) == {
            settings.VECTOR_EMBED_TAG or DEFAULT_EMBEDDING_CONFIG_TAG,
            settings.INFERENCE_TAG or DEFAULT_INFERENCE_CONFIG_TAG,
        }

        # Verify synchronous actor-facing lookups
        embed_cfg = llm_service.get_llm_config_by_tag_sync(
            settings.VECTOR_EMBED_TAG or DEFAULT_EMBEDDING_CONFIG_TAG,
            group_code=expected_group_code,
            test_id=PHASE1_INTEGRATION_TEST_ID,
        )
        assert embed_cfg is not None
        assert embed_cfg.provider == settings.VECTOR_EMBED_PROVIDER.strip().lower()
        assert embed_cfg.model == settings.VECTOR_EMBED_MODEL
        assert embed_cfg.dimensions == settings.VECTOR_EMBED_DIMENSIONS

        infer_cfg = llm_service.get_llm_config_by_tag_sync(
            settings.INFERENCE_TAG or DEFAULT_INFERENCE_CONFIG_TAG,
            group_code=expected_group_code,
            test_id=PHASE1_INTEGRATION_TEST_ID,
        )
        assert infer_cfg is not None
        assert infer_cfg.provider == settings.INFERENCE_PROVIDER.strip().lower()
        assert infer_cfg.model == settings.INFERENCE_MODEL
    finally:
        if client._sync_driver is not None:
            client._sync_driver.close()
        get_settings.cache_clear()
