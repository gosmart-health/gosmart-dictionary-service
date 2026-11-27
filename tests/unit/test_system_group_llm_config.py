"""Unit tests for Phase 1 platform backbone: System, Group, LLM_Config, Embedding, schema loader, and services."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr, ValidationError

from gosmart_dictionary_service.core.config import get_settings
from gosmart_dictionary_service.core.security import (
    classify_llm_api_key,
    encrypt_llm_api_key,
    is_env_var_reference,
    mask_llm_api_key,
    resolve_llm_api_key,
)
from gosmart_dictionary_service.db.schema_loader import (
    SCHEMA_SCRIPT_FILES,
    SchemaInitializationError,
    apply_schema_async,
    apply_schema_sync,
    load_schema_statements,
)
from gosmart_dictionary_service.model.auth import BelongsToRelation, GroupNode
from gosmart_dictionary_service.model.embedding import (
    DEFAULT_EMBEDDING_VECTOR_DIMENSIONS,
    DictionaryEmbeddingFieldTag,
    Embedding,
    EmbedsRelation,
)
from gosmart_dictionary_service.model.llm_config import (
    DEFAULT_EMBEDDING_CONFIG_TAG,
    DEFAULT_INFERENCE_CONFIG_TAG,
    VERTEX_AI_API_KEY_ENV_REF,
    LLM_Config,
    build_default_embedding_llm_config,
    build_default_inference_llm_config,
)
from gosmart_dictionary_service.model.system import DEFAULT_SYSTEM_SERIAL_NUMBER, SystemNode
from gosmart_dictionary_service.service.llm_config_service import LLMConfigService
from gosmart_dictionary_service.service.system_service import (
    DEFAULT_GROUP_CODE,
    DEFAULT_GROUP_NAME,
    TEST_CLEANUP_LABELS,
    SystemService,
    resolve_root_group_code,
)
from tests.conftest import make_test_settings


def test_settings_default_llm_and_embedding_values() -> None:
    settings = make_test_settings()
    assert settings.VECTOR_EMBED_PROVIDER == "ollama"
    assert settings.VECTOR_EMBED_MODEL == "embeddinggemma-2:270m"
    assert settings.VECTOR_EMBED_DIMENSIONS == 768
    assert settings.VECTOR_EMBED_TAG == DEFAULT_EMBEDDING_CONFIG_TAG
    assert settings.INFERENCE_PROVIDER == "ollama"
    assert settings.INFERENCE_MODEL == "gemma4:e2b"
    assert settings.INFERENCE_TAG == DEFAULT_INFERENCE_CONFIG_TAG
    assert settings.VERTEX_AI_LOCATION == "us-central1"


def test_settings_llm_overrides() -> None:
    settings = make_test_settings(
        VECTOR_EMBED_PROVIDER="ollama",
        VECTOR_EMBED_SERVER_URL="http://m5.local.:11434",
        VECTOR_EMBED_MODEL="embeddinggemma-2:270m",
        VECTOR_EMBED_DIMENSIONS=768,
        INFERENCE_PROVIDER="vertexai",
        INFERENCE_MODEL="gemini-2.5-flash",
        VERTEX_AI_API_KEY=SecretStr("secret-test-key"),
    )
    assert settings.VECTOR_EMBED_SERVER_URL == "http://m5.local.:11434"
    assert settings.INFERENCE_PROVIDER == "vertexai"
    assert settings.INFERENCE_MODEL == "gemini-2.5-flash"
    assert settings.VERTEX_AI_API_KEY is not None
    assert settings.VERTEX_AI_API_KEY.get_secret_value() == "secret-test-key"


def test_llm_api_key_encryption_masking_and_env_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    assert classify_llm_api_key("") == "none"
    assert classify_llm_api_key(VERTEX_AI_API_KEY_ENV_REF) == "env"
    assert is_env_var_reference(VERTEX_AI_API_KEY_ENV_REF)
    assert mask_llm_api_key(VERTEX_AI_API_KEY_ENV_REF) == VERTEX_AI_API_KEY_ENV_REF

    # Fernet round-trip with explicit passphrase
    secret = "AQ.test-secret-value-123456"
    encrypted = encrypt_llm_api_key(secret, encryption_key="my-test-passphrase")
    assert encrypted.startswith("enc:fernet:")
    assert classify_llm_api_key(encrypted) == "encrypted"
    assert mask_llm_api_key(encrypted) == "enc:fernet:********"
    assert resolve_llm_api_key(encrypted, encryption_key="my-test-passphrase") == secret

    # Env var placeholders stay literal when passed through encrypt_llm_api_key
    assert encrypt_llm_api_key(VERTEX_AI_API_KEY_ENV_REF, encryption_key="my-test-passphrase") == (
        VERTEX_AI_API_KEY_ENV_REF
    )

    # Resolve ${ENV_VAR} from os.environ
    monkeypatch.setenv("CUSTOM_TEST_LLM_KEY", "env-key-999")
    assert resolve_llm_api_key("${CUSTOM_TEST_LLM_KEY}") == "env-key-999"

    # Resolve ${GOSMART_DS_VERTEX_AI_API_KEY} via Settings fallback even when unset in os.environ
    monkeypatch.delenv("GOSMART_DS_VERTEX_AI_API_KEY", raising=False)
    fake_settings = make_test_settings(VERTEX_AI_API_KEY=SecretStr("from-dotenv-settings"))
    get_settings.cache_clear()
    monkeypatch.setattr("gosmart_dictionary_service.core.security.get_settings", lambda: fake_settings)
    try:
        assert resolve_llm_api_key(VERTEX_AI_API_KEY_ENV_REF) == "from-dotenv-settings"
    finally:
        get_settings.cache_clear()


def test_system_and_group_deterministic_identity_and_test_id_isolation() -> None:
    sys_prod_a = SystemNode()
    sys_prod_b = SystemNode()
    sys_test = SystemNode(test_id="test_phase1")
    assert sys_prod_a.serial_number == DEFAULT_SYSTEM_SERIAL_NUMBER
    assert sys_prod_a.id == sys_prod_b.id
    assert sys_prod_a.id != sys_test.id
    assert isinstance(sys_prod_a.to_neo4j_properties()["installed_at"], str)

    assert resolve_root_group_code(None) == DEFAULT_GROUP_CODE
    assert resolve_root_group_code("   ") == DEFAULT_GROUP_CODE
    assert resolve_root_group_code("test-case 01") == "GOSMART_HEALTH__TEST_CASE_01"

    grp_prod = GroupNode(name=DEFAULT_GROUP_NAME, code=DEFAULT_GROUP_CODE)
    grp_test = GroupNode(name=DEFAULT_GROUP_NAME, code=resolve_root_group_code("test_phase1"), test_id="test_phase1")
    assert grp_prod.id != grp_test.id

    rel_prod = BelongsToRelation(start_node_id=grp_prod.id, end_node_id=sys_prod_a.id)
    rel_test = BelongsToRelation(start_node_id=grp_test.id, end_node_id=sys_test.id, test_id="test_phase1")
    assert rel_prod.id != rel_test.id


def test_llm_config_builders_for_ollama_and_cloud_gemini() -> None:
    group_id = GroupNode(name=DEFAULT_GROUP_NAME, code=DEFAULT_GROUP_CODE).id

    ollama_settings = make_test_settings(
        VECTOR_EMBED_PROVIDER="ollama",
        VECTOR_EMBED_SERVER_URL="http://m5.local.:11434",
        VECTOR_EMBED_MODEL="embeddinggemma-2:270m",
        VECTOR_EMBED_DIMENSIONS=768,
        INFERENCE_PROVIDER="ollama",
        INFERENCE_SERVER_URL="http://manabusb3.local.:11434",
        INFERENCE_MODEL="gemma4:e2b",
        VERTEX_AI_API_KEY=SecretStr("unused-for-ollama"),
    )
    embed_cfg = build_default_embedding_llm_config(group_id, settings=ollama_settings)
    infer_cfg = build_default_inference_llm_config(group_id, settings=ollama_settings)

    assert embed_cfg.tag == DEFAULT_EMBEDDING_CONFIG_TAG
    assert embed_cfg.provider == "ollama"
    assert embed_cfg.model == "embeddinggemma-2:270m"
    assert embed_cfg.url == "http://m5.local.:11434"
    assert embed_cfg.dimensions == 768
    assert embed_cfg.api_key == ""
    assert embed_cfg.id == LLM_Config.compute_group_scoped_id(DEFAULT_EMBEDDING_CONFIG_TAG, group_id)

    assert infer_cfg.tag == DEFAULT_INFERENCE_CONFIG_TAG
    assert infer_cfg.provider == "ollama"
    assert infer_cfg.model == "gemma4:e2b"
    assert infer_cfg.url == "http://manabusb3.local.:11434"
    assert infer_cfg.api_key == ""
    assert infer_cfg.id != embed_cfg.id

    # Cloud Gemini switches to ${GOSMART_DS_VERTEX_AI_API_KEY} and clears Ollama server URL
    cloud_settings = make_test_settings(
        VECTOR_EMBED_PROVIDER="vertexai",
        VECTOR_EMBED_SERVER_URL="http://m5.local.:11434",
        VECTOR_EMBED_MODEL="gemini-embedding-001",
        VECTOR_EMBED_DIMENSIONS=768,
        INFERENCE_PROVIDER="googleai",
        INFERENCE_SERVER_URL="http://manabusb3.local.:11434",
        INFERENCE_MODEL="gemini-2.5-flash",
        VERTEX_AI_API_KEY=SecretStr("cloud-secret"),
        VERTEX_AI_LOCATION="us-central1",
    )
    cloud_embed = build_default_embedding_llm_config(group_id, settings=cloud_settings)
    cloud_infer = build_default_inference_llm_config(group_id, settings=cloud_settings)

    assert cloud_embed.api_key == VERTEX_AI_API_KEY_ENV_REF
    assert cloud_embed.url == ""
    assert cloud_infer.api_key == VERTEX_AI_API_KEY_ENV_REF
    assert cloud_infer.url == ""

    with pytest.raises(ValidationError):
        LLM_Config(
            tag="bad-dims",
            provider="ollama",
            model="embeddinggemma-2:270m",
            dimensions=768,
            dimensions_truncated=1024,
        )


def test_embedding_and_embeds_relation_with_dictionary_field_tags() -> None:
    vec = [1.0 / math.sqrt(DEFAULT_EMBEDDING_VECTOR_DIMENSIONS)] * DEFAULT_EMBEDDING_VECTOR_DIMENSIONS
    emb_name = Embedding(
        modelTag=DEFAULT_EMBEDDING_CONFIG_TAG,
        vector=vec,
        text="Patient's Name",
        targetFieldTag=DictionaryEmbeddingFieldTag.NAME,
        target_node_id="elem-00100010",
    )
    emb_desc = Embedding(
        modelTag=DEFAULT_EMBEDDING_CONFIG_TAG,
        vector=vec,
        text="Patient's full legal name.",
        targetFieldTag=DictionaryEmbeddingFieldTag.DESCRIPTION,
        target_node_id="elem-00100010",
    )
    assert emb_name.size == 768
    assert emb_name.targetFieldTag == "NAME"
    assert emb_desc.targetFieldTag == "DESCRIPTION"
    assert emb_name.id == Embedding.compute_target_scoped_id(
        "elem-00100010", DEFAULT_EMBEDDING_CONFIG_TAG, DictionaryEmbeddingFieldTag.NAME
    )
    assert emb_name.id != emb_desc.id

    emb_test = Embedding(
        modelTag=DEFAULT_EMBEDDING_CONFIG_TAG,
        vector=vec,
        text="Patient's Name",
        targetFieldTag=DictionaryEmbeddingFieldTag.NAME,
        target_node_id="elem-00100010",
        test_id="test_embed_iso",
    )
    assert emb_test.id != emb_name.id

    rel = EmbedsRelation(
        start_node_id=emb_name.id,
        end_node_id="elem-00100010",
        aspect=DictionaryEmbeddingFieldTag.NAME,
    )
    assert rel.aspect == "NAME"

    with pytest.raises(ValidationError):
        Embedding(
            modelTag="embedding-v1",
            vector=[0.1, float("nan")],
            text="Invalid",
            targetFieldTag=DictionaryEmbeddingFieldTag.NAME,
        )


def test_schema_loader_parses_all_allowlisted_scripts_and_executes() -> None:
    assert SCHEMA_SCRIPT_FILES == (
        "system_schema.cypher",
        "embedding_schema.cypher",
        "dicom_schema.cypher",
        "clinical_schema.cypher",
    )
    statements = load_schema_statements()
    sources = {stmt.source_file for stmt in statements}
    assert sources == set(SCHEMA_SCRIPT_FILES)
    assert len(statements) > 30

    # Verify System, Embedding, and DICOM constraints and full-text indexes are present
    joined = "\n".join(stmt.statement for stmt in statements)
    assert "llm_config_sys_id_unique" in joined
    assert "embedding_vector_idx" in joined
    assert "dicom_element_id_unique" in joined
    assert "dicom_element_fulltext_idx" in joined
    assert "dicom_structure_fulltext_idx" in joined
    assert "code_system_id_unique" in joined
    assert "concept_id_unique" in joined
    assert "concept_fulltext_idx" in joined

    # Async execution path
    mock_client = MagicMock()
    mock_client.execute_query = AsyncMock(return_value=[])
    count = asyncio.run(apply_schema_async(mock_client))
    assert count == len(statements)
    assert mock_client.execute_query.await_count == len(statements)

    # Async fail-closed error reporting
    failing_client = MagicMock()
    failing_client.execute_query = AsyncMock(side_effect=RuntimeError("Syntax error (line 1, column 7)"))
    with pytest.raises(SchemaInitializationError) as exc_info:
        asyncio.run(apply_schema_async(failing_client))
    assert exc_info.value.script_file == "system_schema.cypher"
    assert exc_info.value.error_column == 7

    # Sync execution path
    sync_client = MagicMock()
    sync_client.settings = make_test_settings()
    sync_session = MagicMock()
    sync_driver = MagicMock()
    sync_driver.session.return_value.__enter__.return_value = sync_session
    sync_client.get_sync_driver.return_value = sync_driver
    sync_count = apply_schema_sync(sync_client)
    assert sync_count == len(statements)
    assert sync_session.run.call_count == len(statements)


def test_system_service_and_llm_config_service_async_and_sync_mocks() -> None:
    test_settings = make_test_settings(
        VECTOR_EMBED_PROVIDER="ollama",
        VECTOR_EMBED_SERVER_URL="http://m5.local.:11434",
        VECTOR_EMBED_MODEL="embeddinggemma-2:270m",
        VECTOR_EMBED_DIMENSIONS=768,
        INFERENCE_PROVIDER="ollama",
        INFERENCE_SERVER_URL="http://manabusb3.local.:11434",
        INFERENCE_MODEL="gemma4:e2b",
    )

    mock_client = MagicMock()
    call_log: list[tuple[str, dict[str, Any]]] = []

    async def fake_execute_query(query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        p = params or {}
        call_log.append((query, p))
        if "MERGE (sys:System" in query:
            return [
                {
                    "system": {
                        "id": p["sys_id"],
                        "name": p["sys_name"],
                        "owner": p["sys_owner"],
                        "address": p["sys_address"],
                        "email": p["sys_email"],
                        "phone": p["sys_phone"],
                        "serial_number": p["sys_serial_number"],
                        "installed_at": p["sys_installed_at"],
                        "test_id": p["test_id"],
                        "Action": "CREATE",
                    },
                    "root_group": {
                        "id": p["grp_id"],
                        "name": p["grp_name"],
                        "code": p["grp_code"],
                        "group_type": p["grp_type"],
                        "test_id": p["test_id"],
                    },
                }
            ]
        if "MERGE (l:LLM_Config" in query:
            return [
                {
                    "config": {
                        "id": p["id"],
                        "tag": p["tag"],
                        "provider": p["provider"],
                        "model": p["model"],
                        "url": p["url"],
                        "dimensions": p["dimensions"],
                        "api_key": p["api_key"],
                        "location": p["location"],
                        "project_id": p["project_id"],
                        "description": p["description"],
                        "test_id": p["test_id"],
                        "Action": "CREATE",
                    },
                    "groups": [{"id": p["group_id"], "name": DEFAULT_GROUP_NAME, "code": DEFAULT_GROUP_CODE}],
                }
            ]
        if "MATCH (g:Group)-[r:HAS_LLM_CONFIG]->(l:LLM_Config {tag: $tag})" in query:
            return [
                {
                    "config": {
                        "id": "cfg-embed-1",
                        "tag": p["tag"],
                        "provider": "ollama",
                        "model": "embeddinggemma-2:270m",
                        "url": "http://m5.local.:11434",
                        "dimensions": 768,
                        "api_key": "",
                        "location": "us-central1",
                    }
                }
            ]
        if "DETACH DELETE node" in query:
            return [{"deleted": 1}]
        return []

    mock_client.execute_query = AsyncMock(side_effect=fake_execute_query)

    llm_svc = LLMConfigService(client=mock_client)
    sys_svc = SystemService(client=mock_client, llm_service=llm_svc)

    result = asyncio.run(
        sys_svc.bootstrap_system(
            acted_by="test_runner",
            test_id="unit_bootstrap",
            settings=test_settings,
            ensure_schema=False,
            seed_default_llm_configs=True,
        )
    )
    assert result["success"] is True
    assert result["root_group"]["name"] == DEFAULT_GROUP_NAME
    assert result["root_group"]["code"] == "GOSMART_HEALTH__UNIT_BOOTSTRAP"
    assert [cfg["tag"] for cfg in result["llm_configs"]] == [
        DEFAULT_EMBEDDING_CONFIG_TAG,
        DEFAULT_INFERENCE_CONFIG_TAG,
    ]
    assert result["llm_configs"][0]["model"] == "embeddinggemma-2:270m"
    assert result["llm_configs"][0]["dimensions"] == 768
    assert result["llm_configs"][1]["model"] == "gemma4:e2b"

    resolved_cfg = asyncio.run(
        llm_svc.get_llm_config_by_tag(
            DEFAULT_EMBEDDING_CONFIG_TAG,
            group_code="GOSMART_HEALTH__UNIT_BOOTSTRAP",
            test_id="unit_bootstrap",
        )
    )
    assert resolved_cfg is not None
    assert resolved_cfg.model == "embeddinggemma-2:270m"
    assert resolved_cfg.dimensions == 768

    # Cleanup refuses empty/blank test_id
    with pytest.raises(ValueError):
        asyncio.run(sys_svc.cleanup_test_data(""))
    with pytest.raises(ValueError):
        sys_svc.cleanup_test_data_sync("   ")

    # Scoped cleanup runs across all allowlisted labels
    cleanup_res = asyncio.run(sys_svc.cleanup_test_data("unit_bootstrap"))
    assert cleanup_res["success"] is True
    assert cleanup_res["total_deleted"] == len(TEST_CLEANUP_LABELS)
    assert cleanup_res["test_id"] == "unit_bootstrap"


def test_schema_loader_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(SchemaInitializationError):
        load_schema_statements(script_files=("../escape.cypher",), schema_dir=tmp_path)
