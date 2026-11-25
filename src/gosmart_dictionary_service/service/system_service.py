"""Service for root System node, top-level 'GoSmart Health' Group, and scoped test cleanup."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, LiteralString, Optional, Tuple, cast

from gosmart_dictionary_service.core.config import Settings, get_settings
from gosmart_dictionary_service.core.logging import get_logger
from gosmart_dictionary_service.db.neo4j import Neo4jClient, neo4j_client
from gosmart_dictionary_service.db.schema_loader import apply_schema_async, apply_schema_sync
from gosmart_dictionary_service.model.auth import BelongsToRelation, GroupNode
from gosmart_dictionary_service.model.system import (
    DEFAULT_SYSTEM_ADDRESS,
    DEFAULT_SYSTEM_EMAIL,
    DEFAULT_SYSTEM_NAME,
    DEFAULT_SYSTEM_OWNER,
    DEFAULT_SYSTEM_PHONE,
    DEFAULT_SYSTEM_SERIAL_NUMBER,
    SystemNode,
)
from gosmart_dictionary_service.service.llm_config_service import LLMConfigService

logger = get_logger("gosmart_ds.system")

DEFAULT_GROUP_NAME = "GoSmart Health"
DEFAULT_GROUP_CODE = "GOSMART_HEALTH"
DEFAULT_GROUP_TYPE = "HEALTH_SYSTEM"
DEFAULT_GROUP_DESCRIPTION = "Top-level GoSmart Health platform organization owning system LLM configurations"

TEST_CLEANUP_LABELS: Tuple[str, ...] = (
    "Embedding",
    "DicomElement",
    "DicomCIOD",
    "DicomModule",
    "DicomMacro",
    "DicomSOPClass",
    "DicomConfidentialityProfile",
    "DicomStandardReference",
    "LLM_Config",
    "Group",
    "System",
    "User",
    "Identity",
    "Authenticator",
    "AuditLog",
)

_BOOTSTRAP_SYSTEM_AND_GROUP_CYPHER: LiteralString = """
MERGE (sys:System {id: $sys_id})
ON CREATE SET sys.name = $sys_name,
              sys.owner = $sys_owner,
              sys.address = $sys_address,
              sys.email = $sys_email,
              sys.phone = $sys_phone,
              sys.serial_number = $sys_serial_number,
              sys.installed_at = datetime($sys_installed_at),
              sys.test_id = $test_id,
              sys.Action = 'CREATE',
              sys.ActedAt = datetime({timezone: 'UTC'}),
              sys.ActedBy = $acted_by,
              sys.CreatedAt = datetime({timezone: 'UTC'})
ON MATCH SET  sys.name = coalesce(sys.name, $sys_name),
              sys.owner = coalesce(sys.owner, $sys_owner),
              sys.test_id = coalesce($test_id, sys.test_id),
              sys.Action = 'UPDATE',
              sys.ActedAt = datetime({timezone: 'UTC'}),
              sys.ActedBy = $acted_by,
              sys.UpdatedAt = datetime({timezone: 'UTC'})

MERGE (g:Group {id: $grp_id})
ON CREATE SET g.name = $grp_name,
              g.code = $grp_code,
              g.group_type = $grp_type,
              g.is_active = true,
              g.description = $grp_description,
              g.test_id = $test_id,
              g.Action = 'CREATE',
              g.ActedAt = datetime({timezone: 'UTC'}),
              g.ActedBy = $acted_by,
              g.CreatedAt = datetime({timezone: 'UTC'})
ON MATCH SET  g.name = $grp_name,
              g.code = $grp_code,
              g.group_type = $grp_type,
              g.is_active = true,
              g.description = coalesce(g.description, $grp_description),
              g.test_id = coalesce($test_id, g.test_id),
              g.Action = 'UPDATE',
              g.ActedAt = datetime({timezone: 'UTC'}),
              g.ActedBy = $acted_by,
              g.UpdatedAt = datetime({timezone: 'UTC'})

MERGE (g)-[r:BELONGS_TO {id: $rel_id}]->(sys)
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

RETURN properties(sys) AS system,
       properties(g) AS root_group
"""

_GET_SYSTEM_CYPHER: LiteralString = """
MATCH (sys:System)
WHERE (sys.Action IS NULL OR sys.Action <> 'DELETE')
  AND (($test_id IS NULL AND sys.test_id IS NULL) OR ($test_id IS NOT NULL AND sys.test_id = $test_id))
RETURN properties(sys) AS system
LIMIT 1
"""

_GET_ROOT_GROUP_CYPHER: LiteralString = """
MATCH (g:Group {code: $code})
WHERE (g.Action IS NULL OR g.Action <> 'DELETE')
  AND (($test_id IS NULL AND g.test_id IS NULL) OR ($test_id IS NOT NULL AND g.test_id = $test_id))
RETURN properties(g) AS root_group
LIMIT 1
"""


def clean_neo4j_value(val: Any) -> Any:
    """Normalize Neo4j-specific types (e.g. neo4j.time.DateTime) to standard Python types."""
    if hasattr(val, "to_native"):
        return val.to_native()
    if isinstance(val, dict):
        return {k: clean_neo4j_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [clean_neo4j_value(v) for v in val]
    return val


def normalize_test_id(test_id: Optional[str]) -> Optional[str]:
    """Normalize ``test_id`` so empty strings or whitespace become ``None``."""
    if test_id and isinstance(test_id, str):
        stripped = test_id.strip()
        return stripped if stripped else None
    return None


def resolve_root_group_code(test_id: Optional[str] = None) -> str:
    """Return the root Group code, scoping it for test runs to avoid ``group_code_unique`` collisions."""
    norm_test_id = normalize_test_id(test_id)
    if not norm_test_id:
        return DEFAULT_GROUP_CODE
    safe_suffix = re.sub(r"[^A-Za-z0-9_]+", "_", norm_test_id).strip("_").upper() or "TEST"
    return f"{DEFAULT_GROUP_CODE}__{safe_suffix}"


def _dict_to_system_response(data: Dict[str, Any]) -> Dict[str, Any]:
    clean = clean_neo4j_value(data)
    installed_at = clean.get("installed_at")
    if isinstance(installed_at, str):
        installed_at = datetime.fromisoformat(installed_at)
    if not isinstance(installed_at, datetime):
        installed_at = datetime.now(timezone.utc)
    return {
        "id": clean.get("id", ""),
        "name": clean.get("name", DEFAULT_SYSTEM_NAME),
        "owner": clean.get("owner", DEFAULT_SYSTEM_OWNER),
        "address": clean.get("address", DEFAULT_SYSTEM_ADDRESS),
        "email": clean.get("email", DEFAULT_SYSTEM_EMAIL),
        "phone": clean.get("phone", DEFAULT_SYSTEM_PHONE),
        "serial_number": clean.get("serial_number", DEFAULT_SYSTEM_SERIAL_NUMBER),
        "installed_at": installed_at,
        "test_id": clean.get("test_id"),
        "action": clean.get("Action", "CREATE"),
        "created_at": clean.get("CreatedAt"),
        "updated_at": clean.get("UpdatedAt"),
    }


def _build_bootstrap_entities(
    *,
    acted_by: str = "system",
    test_id: Optional[str] = None,
) -> Tuple[SystemNode, GroupNode, BelongsToRelation, Dict[str, Any]]:
    """Construct deterministic System, root 'GoSmart Health' Group, and BELONGS_TO params."""
    norm_test_id = normalize_test_id(test_id)
    sys_node = SystemNode(test_id=norm_test_id, ActedBy=acted_by)
    grp_node = GroupNode(
        name=DEFAULT_GROUP_NAME,
        code=resolve_root_group_code(norm_test_id),
        group_type=DEFAULT_GROUP_TYPE,
        is_active=True,
        description=DEFAULT_GROUP_DESCRIPTION,
        test_id=norm_test_id,
        ActedBy=acted_by,
    )
    rel = BelongsToRelation(
        start_node_id=grp_node.id,
        end_node_id=sys_node.id,
        test_id=norm_test_id,
        ActedBy=acted_by,
    )
    params: Dict[str, Any] = {
        "sys_id": sys_node.id,
        "sys_name": sys_node.name,
        "sys_owner": sys_node.owner,
        "sys_address": sys_node.address,
        "sys_email": sys_node.email,
        "sys_phone": sys_node.phone,
        "sys_serial_number": sys_node.serial_number,
        "sys_installed_at": sys_node.installed_at.isoformat(),
        "grp_id": grp_node.id,
        "grp_name": grp_node.name,
        "grp_code": grp_node.code,
        "grp_type": grp_node.group_type,
        "grp_description": grp_node.description,
        "rel_id": rel.id,
        "test_id": norm_test_id,
        "acted_by": acted_by,
    }
    return sys_node, grp_node, rel, params


def _build_label_cleanup_cypher(label: str) -> LiteralString:
    return cast(
        LiteralString,
        (
            f"MATCH (n:{label} {{test_id: $test_id}}) "
            "WITH collect(n) AS nodes "
            "FOREACH (node IN nodes | DETACH DELETE node) "
            "RETURN size(nodes) AS deleted"
        ),
    )


class SystemService:
    """Manages root ``:System`` + ``GoSmart Health`` ``:Group`` bootstrap and scoped test cleanup."""

    def __init__(
        self,
        client: Optional[Neo4jClient] = None,
        llm_service: Optional[LLMConfigService] = None,
    ) -> None:
        self.client = client or neo4j_client
        self.llm_service = llm_service or LLMConfigService(client=self.client)

    async def bootstrap_system(
        self,
        *,
        acted_by: str = "system",
        test_id: Optional[str] = None,
        settings: Optional[Settings] = None,
        ensure_schema: bool = True,
        seed_default_llm_configs: bool = True,
    ) -> Dict[str, Any]:
        """Idempotently bootstrap ``:System``, ``GoSmart Health`` ``:Group``, and default ``:LLM_Config`` nodes."""
        if ensure_schema:
            await apply_schema_async(self.client)

        norm_test_id = normalize_test_id(test_id)
        _, grp_node, _, params = _build_bootstrap_entities(acted_by=acted_by, test_id=norm_test_id)
        records = await self.client.execute_query(_BOOTSTRAP_SYSTEM_AND_GROUP_CYPHER, params)
        if not records:
            raise RuntimeError("Failed to bootstrap root System and GoSmart Health Group nodes in Neo4j.")

        row = records[0]
        system_dict = _dict_to_system_response(row["system"])
        root_group_dict = clean_neo4j_value(row["root_group"])
        llm_configs: List[Dict[str, Any]] = []
        if seed_default_llm_configs:
            llm_configs = await self.llm_service.seed_default_configs(
                grp_node.id,
                settings=settings or get_settings(),
                acted_by=acted_by,
                test_id=norm_test_id,
            )

        logger.info(
            "system_bootstrap_completed",
            system_id=system_dict["id"],
            root_group_id=root_group_dict.get("id"),
            root_group_code=root_group_dict.get("code"),
            llm_config_count=len(llm_configs),
            test_id=norm_test_id,
        )
        return {
            "success": True,
            "message": "System, GoSmart Health Group, and default LLM configurations bootstrapped successfully.",
            "system": system_dict,
            "root_group": root_group_dict,
            "llm_configs": llm_configs,
        }

    def bootstrap_system_sync(
        self,
        *,
        acted_by: str = "system",
        test_id: Optional[str] = None,
        settings: Optional[Settings] = None,
        ensure_schema: bool = True,
        seed_default_llm_configs: bool = True,
    ) -> Dict[str, Any]:
        """Synchronously bootstrap ``:System``, ``GoSmart Health`` ``:Group``, and default ``:LLM_Config`` nodes."""
        if ensure_schema:
            apply_schema_sync(self.client)

        norm_test_id = normalize_test_id(test_id)
        _, grp_node, _, params = _build_bootstrap_entities(acted_by=acted_by, test_id=norm_test_id)

        def _tx(tx: Any) -> Optional[Dict[str, Any]]:
            result = tx.run(cast(LiteralString, _BOOTSTRAP_SYSTEM_AND_GROUP_CYPHER), params)
            record = result.single()
            return record.data() if record is not None else None

        row = self.client.execute_write_tx_sync(_tx)
        if not row:
            raise RuntimeError("Failed to bootstrap root System and GoSmart Health Group nodes in Neo4j.")

        system_dict = _dict_to_system_response(row["system"])
        root_group_dict = clean_neo4j_value(row["root_group"])
        llm_configs: List[Dict[str, Any]] = []
        if seed_default_llm_configs:
            llm_configs = self.llm_service.seed_default_configs_sync(
                grp_node.id,
                settings=settings or get_settings(),
                acted_by=acted_by,
                test_id=norm_test_id,
            )

        logger.info(
            "system_bootstrap_completed",
            system_id=system_dict["id"],
            root_group_id=root_group_dict.get("id"),
            root_group_code=root_group_dict.get("code"),
            llm_config_count=len(llm_configs),
            test_id=norm_test_id,
        )
        return {
            "success": True,
            "message": "System, GoSmart Health Group, and default LLM configurations bootstrapped successfully.",
            "system": system_dict,
            "root_group": root_group_dict,
            "llm_configs": llm_configs,
        }

    async def get_system(self, *, test_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch the active root ``:System`` node."""
        records = await self.client.execute_query(_GET_SYSTEM_CYPHER, {"test_id": normalize_test_id(test_id)})
        if not records:
            return None
        return _dict_to_system_response(records[0]["system"])

    async def get_root_group(self, *, test_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch the active top-level ``GoSmart Health`` ``:Group`` node asynchronously."""
        norm_test_id = normalize_test_id(test_id)
        records = await self.client.execute_query(
            _GET_ROOT_GROUP_CYPHER,
            {"code": resolve_root_group_code(norm_test_id), "test_id": norm_test_id},
        )
        if not records:
            return None
        return clean_neo4j_value(records[0]["root_group"])

    def get_root_group_sync(self, *, test_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch the active top-level ``GoSmart Health`` ``:Group`` node synchronously."""
        norm_test_id = normalize_test_id(test_id)
        params = {"code": resolve_root_group_code(norm_test_id), "test_id": norm_test_id}
        driver = self.client.get_sync_driver()
        with driver.session(database=self.client.settings.DB_DATABASE) as session:
            result = session.run(cast(LiteralString, _GET_ROOT_GROUP_CYPHER), params)
            record = result.single()
            return clean_neo4j_value(record["root_group"]) if record is not None else None

    async def cleanup_test_data(self, test_id: str) -> Dict[str, Any]:
        """Safely detach-delete ONLY nodes and relationships carrying the given non-empty ``test_id``."""
        norm_test_id = normalize_test_id(test_id)
        if not norm_test_id:
            raise ValueError(
                "cleanup_test_data requires a non-empty test_id to prevent accidental production deletion."
            )

        deleted_by_label: Dict[str, int] = {}
        total_deleted = 0
        for label in TEST_CLEANUP_LABELS:
            query = _build_label_cleanup_cypher(label)
            records = await self.client.execute_query(query, {"test_id": norm_test_id})
            count = int(records[0]["deleted"]) if records and records[0].get("deleted") else 0
            if count > 0:
                deleted_by_label[label] = count
                total_deleted += count

        return {
            "success": True,
            "message": f"Deleted {total_deleted} test nodes for test_id='{norm_test_id}'.",
            "total_deleted": total_deleted,
            "deleted_by_label": deleted_by_label,
            "labels_cleaned": list(TEST_CLEANUP_LABELS),
            "test_id": norm_test_id,
        }

    def cleanup_test_data_sync(self, test_id: str) -> Dict[str, Any]:
        """Synchronously detach-delete ONLY nodes carrying the given non-empty ``test_id``."""
        norm_test_id = normalize_test_id(test_id)
        if not norm_test_id:
            raise ValueError(
                "cleanup_test_data_sync requires a non-empty test_id to prevent accidental production deletion."
            )

        def _tx(tx: Any) -> Dict[str, int]:
            counts: Dict[str, int] = {}
            for label in TEST_CLEANUP_LABELS:
                query = _build_label_cleanup_cypher(label)
                rec = tx.run(query, {"test_id": norm_test_id}).single()
                cnt = int(rec["deleted"]) if rec is not None and rec.get("deleted") else 0
                if cnt > 0:
                    counts[label] = cnt
            return counts

        deleted_by_label = self.client.execute_write_tx_sync(_tx)
        total_deleted = sum(deleted_by_label.values())
        return {
            "success": True,
            "message": f"Deleted {total_deleted} test nodes for test_id='{norm_test_id}'.",
            "total_deleted": total_deleted,
            "deleted_by_label": deleted_by_label,
            "labels_cleaned": list(TEST_CLEANUP_LABELS),
            "test_id": norm_test_id,
        }


system_service = SystemService()

__all__ = [
    "DEFAULT_GROUP_NAME",
    "DEFAULT_GROUP_CODE",
    "DEFAULT_GROUP_TYPE",
    "DEFAULT_GROUP_DESCRIPTION",
    "TEST_CLEANUP_LABELS",
    "clean_neo4j_value",
    "normalize_test_id",
    "resolve_root_group_code",
    "SystemService",
    "system_service",
]
