"""Service for managing dynamic, group-scoped LLM configurations in Neo4j."""

from __future__ import annotations

import json
from typing import Any, Dict, List, LiteralString, Optional, cast

from gosmart_dictionary_service.core.config import Settings, get_settings
from gosmart_dictionary_service.core.logging import get_logger
from gosmart_dictionary_service.core.security import (
    classify_llm_api_key,
    encrypt_llm_api_key,
    mask_llm_api_key,
)
from gosmart_dictionary_service.db.neo4j import Neo4jClient, neo4j_client
from gosmart_dictionary_service.model.llm_config import (
    HasLlmConfigRelation,
    LLM_Config,
    build_default_embedding_llm_config,
    build_default_inference_llm_config,
)

logger = get_logger("gosmart_ds.llm_config")

DEFAULT_LLM_GROUP_NAME = "GoSmart Health"
DEFAULT_LLM_GROUP_CODE = "GOSMART_HEALTH"

_UPSERT_LLM_CONFIG_CYPHER: LiteralString = """
MATCH (g:Group {id: $group_id})
MERGE (l:LLM_Config {id: $id})
ON CREATE SET l.tag = $tag,
              l.provider = $provider,
              l.model = $model,
              l.url = $url,
              l.dimensions = $dimensions,
              l.dimensions_truncated = $dimensions_truncated,
              l.api_key = $api_key,
              l.location = $location,
              l.project_id = $project_id,
              l.extra_config = $extra_config,
              l.description = $description,
              l.test_id = $test_id,
              l.Action = 'CREATE',
              l.ActedAt = datetime({timezone: 'UTC'}),
              l.ActedBy = $acted_by,
              l.CreatedAt = datetime({timezone: 'UTC'})
ON MATCH SET  l.tag = $tag,
              l.provider = $provider,
              l.model = $model,
              l.url = $url,
              l.dimensions = $dimensions,
              l.dimensions_truncated = $dimensions_truncated,
              l.api_key = $api_key,
              l.location = $location,
              l.project_id = $project_id,
              l.extra_config = $extra_config,
              l.description = $description,
              l.test_id = coalesce($test_id, l.test_id),
              l.Action = 'UPDATE',
              l.ActedAt = datetime({timezone: 'UTC'}),
              l.ActedBy = $acted_by,
              l.UpdatedAt = datetime({timezone: 'UTC'})
MERGE (g)-[r:HAS_LLM_CONFIG {id: $rel_id}]->(l)
ON CREATE SET r.test_id = $test_id,
              r.Action = 'CREATE',
              r.ActedAt = datetime({timezone: 'UTC'}),
              r.ActedBy = $acted_by,
              r.CreatedAt = datetime({timezone: 'UTC'})
ON MATCH SET  r.test_id = coalesce($test_id, r.test_id),
              r.Action = 'UPDATE',
              r.ActedAt = datetime({timezone: 'UTC'}),
              r.ActedBy = $acted_by,
              r.UpdatedAt = datetime({timezone: 'UTC'})
RETURN properties(l) AS config,
       collect(DISTINCT properties(g)) AS groups
"""

_GET_LLM_CONFIG_BY_TAG_CYPHER: LiteralString = """
MATCH (g:Group)-[r:HAS_LLM_CONFIG]->(l:LLM_Config {tag: $tag})
WHERE (l.Action IS NULL OR l.Action <> 'DELETE')
  AND (r.Action IS NULL OR r.Action <> 'DELETE')
  AND (g.Action IS NULL OR g.Action <> 'DELETE')
  AND ($group_id IS NULL OR g.id = $group_id)
  AND ($group_code IS NULL OR g.code = $group_code)
  AND ($group_name IS NULL OR g.name = $group_name)
  AND (($test_id IS NULL AND l.test_id IS NULL) OR ($test_id IS NOT NULL AND l.test_id = $test_id))
RETURN properties(l) AS config
LIMIT 1
"""

_LIST_LLM_CONFIGS_CYPHER: LiteralString = """
MATCH (g:Group)-[r:HAS_LLM_CONFIG]->(l:LLM_Config)
WHERE ($include_deleted = true OR (l.Action IS NULL OR l.Action <> 'DELETE'))
  AND (r.Action IS NULL OR r.Action <> 'DELETE')
  AND ($group_id IS NULL OR g.id = $group_id)
  AND ($group_code IS NULL OR g.code = $group_code)
  AND (($test_id IS NULL AND l.test_id IS NULL) OR ($test_id IS NOT NULL AND l.test_id = $test_id))
RETURN properties(l) AS config,
       collect(DISTINCT properties(g)) AS groups
ORDER BY l.tag ASC
"""

_SOFT_DELETE_LLM_CONFIG_CYPHER: LiteralString = """
MATCH (l:LLM_Config {id: $config_id})
SET l.Action = 'DELETE',
    l.ActedAt = datetime({timezone: 'UTC'}),
    l.ActedBy = $acted_by,
    l.UpdatedAt = datetime({timezone: 'UTC'})
RETURN l.id AS id
"""


def _clean_neo4j_value(val: Any) -> Any:
    """Normalize Neo4j-specific types (e.g. neo4j.time.DateTime) to standard Python types."""
    if hasattr(val, "to_native"):
        return val.to_native()
    if isinstance(val, dict):
        return {k: _clean_neo4j_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_clean_neo4j_value(v) for v in val]
    return val


def _parse_extra_config_value(raw_val: Any) -> Optional[Dict[str, Any]]:
    if raw_val is None or raw_val == "":
        return None
    if isinstance(raw_val, dict):
        return raw_val
    if isinstance(raw_val, str):
        try:
            parsed = json.loads(raw_val)
            return parsed if isinstance(parsed, dict) else None
        except Exception as exc:
            logger.error("llm_config_extra_config_parse_failed", error=str(exc))
            return None
    return None


def _dict_to_llm_config_model(data: Dict[str, Any]) -> LLM_Config:
    """Convert raw Neo4j node properties into a typed ``LLM_Config`` domain model."""
    clean = _clean_neo4j_value(data)
    if "extra_config" in clean:
        clean["extra_config"] = _parse_extra_config_value(clean["extra_config"])
    return LLM_Config(**clean)


def _dict_to_llm_config_response(
    data: Dict[str, Any],
    groups: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Map raw Neo4j LLM_Config properties into a sanitized ``LLMConfigResponse`` dictionary."""
    clean = _clean_neo4j_value(data)
    raw_api_key = clean.get("api_key") or ""
    extra_cfg = _parse_extra_config_value(clean.get("extra_config"))

    group_summaries: List[Dict[str, Any]] = []
    group_ids: List[str] = []
    for grp in groups or []:
        if not grp:
            continue
        clean_grp = _clean_neo4j_value(grp)
        gid = clean_grp.get("id")
        if gid and gid not in group_ids:
            group_ids.append(gid)
            group_summaries.append(
                {
                    "id": gid,
                    "name": clean_grp.get("name", ""),
                    "code": clean_grp.get("code", ""),
                    "group_type": clean_grp.get("group_type"),
                }
            )

    return {
        "id": clean.get("id", ""),
        "tag": clean.get("tag", ""),
        "provider": clean.get("provider", ""),
        "model": clean.get("model", ""),
        "url": clean.get("url") or "",
        "dimensions": clean.get("dimensions"),
        "dimensions_truncated": clean.get("dimensions_truncated"),
        "api_key": mask_llm_api_key(raw_api_key),
        "has_api_key": bool(str(raw_api_key).strip()),
        "api_key_source": classify_llm_api_key(raw_api_key),
        "location": clean.get("location") or "us-central1",
        "project_id": clean.get("project_id"),
        "extra_config": extra_cfg,
        "description": clean.get("description"),
        "group_ids": group_ids,
        "groups": group_summaries,
        "test_id": clean.get("test_id"),
        "action": clean.get("Action", "CREATE"),
        "created_at": clean.get("CreatedAt"),
        "updated_at": clean.get("UpdatedAt"),
    }


def _normalize_test_id(test_id: Optional[str]) -> Optional[str]:
    if test_id and isinstance(test_id, str):
        stripped = test_id.strip()
        return stripped if stripped else None
    return None


def _build_upsert_params(
    config: LLM_Config,
    group_id: str,
    *,
    acted_by: str = "system",
) -> Dict[str, Any]:
    """Prepare Cypher parameters for an idempotent group-scoped LLM_Config upsert."""
    norm_test_id = _normalize_test_id(config.test_id)
    scoped_id = LLM_Config.compute_group_scoped_id(
        tag=config.tag,
        group_id=group_id,
        test_id=norm_test_id,
    )
    stored_api_key = encrypt_llm_api_key(config.api_key)
    extra_config_json = json.dumps(config.extra_config, sort_keys=True) if config.extra_config is not None else None
    rel = HasLlmConfigRelation(
        start_node_id=group_id,
        end_node_id=scoped_id,
        test_id=norm_test_id,
        ActedBy=acted_by,
    )
    return {
        "group_id": group_id,
        "id": scoped_id,
        "rel_id": rel.id,
        "tag": config.tag,
        "provider": config.provider,
        "model": config.model,
        "url": config.url,
        "dimensions": config.dimensions,
        "dimensions_truncated": config.dimensions_truncated,
        "api_key": stored_api_key,
        "location": config.location,
        "project_id": config.project_id,
        "extra_config": extra_config_json,
        "description": config.description,
        "test_id": norm_test_id,
        "acted_by": acted_by,
    }


class LLMConfigService:
    """Async and sync Neo4j service for group-scoped ``LLM_Config`` records."""

    def __init__(self, client: Optional[Neo4jClient] = None) -> None:
        self.client = client or neo4j_client

    async def upsert_llm_config(
        self,
        config: LLM_Config,
        group_id: str,
        *,
        acted_by: str = "system",
    ) -> Dict[str, Any]:
        """Idempotently persist an ``LLM_Config`` node linked to ``group_id`` asynchronously."""
        params = _build_upsert_params(config, group_id, acted_by=acted_by)
        records = await self.client.execute_query(_UPSERT_LLM_CONFIG_CYPHER, params)
        if not records:
            raise ValueError(f"Target Group '{group_id}' was not found while upserting LLM_Config '{config.tag}'.")
        row = records[0]
        return _dict_to_llm_config_response(row["config"], row.get("groups"))

    def upsert_llm_config_sync(
        self,
        config: LLM_Config,
        group_id: str,
        *,
        acted_by: str = "system",
    ) -> Dict[str, Any]:
        """Idempotently persist an ``LLM_Config`` node linked to ``group_id`` synchronously."""
        params = _build_upsert_params(config, group_id, acted_by=acted_by)

        def _tx(tx: Any) -> Optional[Dict[str, Any]]:
            result = tx.run(cast(LiteralString, _UPSERT_LLM_CONFIG_CYPHER), params)
            record = result.single()
            return record.data() if record is not None else None

        row = self.client.execute_write_tx_sync(_tx)
        if not row:
            raise ValueError(f"Target Group '{group_id}' was not found while upserting LLM_Config '{config.tag}'.")
        return _dict_to_llm_config_response(row["config"], row.get("groups"))

    async def seed_default_configs(
        self,
        group_id: str,
        *,
        settings: Optional[Settings] = None,
        acted_by: str = "system",
        test_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Seed the default embedding and inference ``LLM_Config`` nodes from Settings asynchronously."""
        cfg = settings or get_settings()
        defaults = [
            build_default_embedding_llm_config(group_id, settings=cfg, test_id=test_id, acted_by=acted_by),
            build_default_inference_llm_config(group_id, settings=cfg, test_id=test_id, acted_by=acted_by),
        ]
        results: List[Dict[str, Any]] = []
        for item in defaults:
            results.append(await self.upsert_llm_config(item, group_id, acted_by=acted_by))
        logger.info(
            "default_llm_configs_seeded",
            group_id=group_id,
            tags=[item["tag"] for item in results],
            test_id=_normalize_test_id(test_id),
        )
        return results

    def seed_default_configs_sync(
        self,
        group_id: str,
        *,
        settings: Optional[Settings] = None,
        acted_by: str = "system",
        test_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Seed the default embedding and inference ``LLM_Config`` nodes from Settings synchronously."""
        cfg = settings or get_settings()
        defaults = [
            build_default_embedding_llm_config(group_id, settings=cfg, test_id=test_id, acted_by=acted_by),
            build_default_inference_llm_config(group_id, settings=cfg, test_id=test_id, acted_by=acted_by),
        ]
        results: List[Dict[str, Any]] = []
        for item in defaults:
            results.append(self.upsert_llm_config_sync(item, group_id, acted_by=acted_by))
        logger.info(
            "default_llm_configs_seeded",
            group_id=group_id,
            tags=[item["tag"] for item in results],
            test_id=_normalize_test_id(test_id),
        )
        return results

    async def get_llm_config_by_tag(
        self,
        tag: str,
        *,
        group_id: Optional[str] = None,
        group_code: Optional[str] = None,
        group_name: Optional[str] = DEFAULT_LLM_GROUP_NAME,
        test_id: Optional[str] = None,
    ) -> Optional[LLM_Config]:
        """Resolve an active ``LLM_Config`` domain model by purpose ``tag`` asynchronously."""
        effective_name = None if (group_id or group_code) else group_name
        params = {
            "tag": tag.strip(),
            "group_id": group_id,
            "group_code": group_code.strip().upper() if group_code else None,
            "group_name": effective_name,
            "test_id": _normalize_test_id(test_id),
        }
        records = await self.client.execute_query(_GET_LLM_CONFIG_BY_TAG_CYPHER, params)
        if not records:
            return None
        return _dict_to_llm_config_model(records[0]["config"])

    def get_llm_config_by_tag_sync(
        self,
        tag: str,
        *,
        group_id: Optional[str] = None,
        group_code: Optional[str] = None,
        group_name: Optional[str] = DEFAULT_LLM_GROUP_NAME,
        test_id: Optional[str] = None,
    ) -> Optional[LLM_Config]:
        """Resolve an active ``LLM_Config`` domain model by purpose ``tag`` synchronously (for Pykka actors)."""
        effective_name = None if (group_id or group_code) else group_name
        params = {
            "tag": tag.strip(),
            "group_id": group_id,
            "group_code": group_code.strip().upper() if group_code else None,
            "group_name": effective_name,
            "test_id": _normalize_test_id(test_id),
        }
        driver = self.client.get_sync_driver()
        with driver.session(database=self.client.settings.DB_DATABASE) as session:
            result = session.run(cast(LiteralString, _GET_LLM_CONFIG_BY_TAG_CYPHER), params)
            record = result.single()
            if record is None:
                return None
            return _dict_to_llm_config_model(record["config"])

    async def list_llm_configs(
        self,
        *,
        group_id: Optional[str] = None,
        group_code: Optional[str] = None,
        test_id: Optional[str] = None,
        include_deleted: bool = False,
    ) -> List[Dict[str, Any]]:
        """List sanitized ``LLMConfigResponse`` entries for a group."""
        params = {
            "group_id": group_id,
            "group_code": group_code.strip().upper() if group_code else None,
            "test_id": _normalize_test_id(test_id),
            "include_deleted": include_deleted,
        }
        records = await self.client.execute_query(_LIST_LLM_CONFIGS_CYPHER, params)
        return [_dict_to_llm_config_response(row["config"], row.get("groups")) for row in records]

    async def delete_llm_config(self, config_id: str, *, acted_by: str = "system") -> bool:
        """Soft-delete an ``LLM_Config`` node (non-destructive lifecycle)."""
        records = await self.client.execute_query(
            _SOFT_DELETE_LLM_CONFIG_CYPHER,
            {"config_id": config_id, "acted_by": acted_by},
        )
        return bool(records)


llm_config_service = LLMConfigService()

__all__ = [
    "DEFAULT_LLM_GROUP_NAME",
    "DEFAULT_LLM_GROUP_CODE",
    "LLMConfigService",
    "llm_config_service",
]
