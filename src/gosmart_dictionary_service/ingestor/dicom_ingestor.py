"""Idempotent batched Neo4j ingestion pipeline for the Innolitics DICOM standard dictionary."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, LiteralString, Optional, Tuple, cast

from pydantic import BaseModel, Field, field_validator

from gosmart_dictionary_service.core.config import Settings, get_settings
from gosmart_dictionary_service.core.logging import get_logger
from gosmart_dictionary_service.db.neo4j import Neo4jClient, neo4j_client
from gosmart_dictionary_service.db.schema_loader import apply_schema_sync
from gosmart_dictionary_service.model.actor_message import EmbeddingItem
from gosmart_dictionary_service.model.dicom import (
    DicomAttributeSeed,
    DicomAuditableNode,
    DicomAuditableRelation,
    DicomCIODNode,
    DicomCiodSeed,
    DicomCiodToFuncGroupMacroSeed,
    DicomCiodToModuleSeed,
    DicomConfidentialityProfileAttributeSeed,
    DicomConfidentialityProfileNode,
    DicomElementNode,
    DicomMacroNode,
    DicomMacroSeed,
    DicomMacroToAttributeSeed,
    DicomModuleNode,
    DicomModuleSeed,
    DicomModuleToAttributeSeed,
    DicomReferencesSeed,
    DicomSOPClassNode,
    DicomSopClassSeed,
    DicomStandardReferenceNode,
    HasConfidentialityProfileRelation,
    HasFuncGroupMacroRelation,
    HasModuleRelation,
    ImplementsCiodRelation,
    ReferencesStandardRelation,
    UsesElementRelation,
    normalize_dicom_tag_hex,
)
from gosmart_dictionary_service.model.embedding import DictionaryEmbeddingFieldTag
from gosmart_dictionary_service.model.llm_config import DEFAULT_EMBEDDING_CONFIG_TAG
from gosmart_dictionary_service.service.embedding_service import (
    EmbeddingBatchReport,
    EmbeddingService,
)
from gosmart_dictionary_service.service.system_service import (
    DEFAULT_GROUP_NAME,
    SystemService,
    normalize_test_id,
)

logger = get_logger("gosmart_ds.dicom_ingestor")

_AUDIT_KEYS = frozenset({"id", "Action", "ActedAt", "ActedBy", "CreatedAt", "UpdatedAt", "test_id"})

_ALLOWED_NODE_LABELS = frozenset(
    {
        "DicomElement",
        "DicomCIOD",
        "DicomModule",
        "DicomMacro",
        "DicomSOPClass",
        "DicomConfidentialityProfile",
        "DicomStandardReference",
    }
)

_ALLOWED_REL_SPECS = frozenset(
    {
        ("DicomSOPClass", "IMPLEMENTS_CIOD", "DicomCIOD"),
        ("DicomCIOD", "HAS_MODULE", "DicomModule"),
        ("DicomCIOD", "HAS_FUNC_GROUP_MACRO", "DicomMacro"),
        ("DicomModule", "USES_ELEMENT", "DicomElement"),
        ("DicomMacro", "USES_ELEMENT", "DicomElement"),
        ("DicomElement", "HAS_CONFIDENTIALITY_PROFILE", "DicomConfidentialityProfile"),
        ("DicomModule", "REFERENCES_STANDARD", "DicomStandardReference"),
        ("DicomMacro", "REFERENCES_STANDARD", "DicomStandardReference"),
    }
)


class DicomIngestionOptions(BaseModel):
    """Configuration options controlling a DICOM standard ingestion run."""

    source_dir: Path = Field(
        default=Path("./data-source/dicom"),
        description="Directory containing the 11 Innolitics dicom-standard JSON files",
    )
    batch_size: int = Field(default=1000, gt=0, description="Rows per Neo4j UNWIND MERGE write transaction")
    acted_by: str = Field(default="dicom_ingestor", description="Actor identifier recorded in audit fields")
    test_id: Optional[str] = Field(default=None, description="Optional test isolation identifier")
    ensure_schema: bool = Field(default=True, description="Apply Neo4j DDL schema before writing nodes")
    bootstrap_system: bool = Field(
        default=False,
        description="Bootstrap System + GoSmart Health Group + default LLM_Config before ingesting",
    )
    include_references: bool = Field(default=True, description="Ingest references.json nodes")
    include_reference_edges: bool = Field(
        default=True,
        description="Ingest :REFERENCES_STANDARD edges from modules/macros to references",
    )
    dedupe_reference_edges_per_owner: bool = Field(
        default=True,
        description="Deduplicate :REFERENCES_STANDARD edges per (owner_node_id, source_url)",
    )
    compute_embeddings: bool = Field(
        default=False,
        description="Compute and persist :Embedding nodes for DICOM node name/description fields",
    )
    embedding_tag: str = Field(
        default=DEFAULT_EMBEDDING_CONFIG_TAG,
        description="LLM_Config purpose tag for embedding computation",
    )
    embedding_group_code: Optional[str] = Field(
        default=None,
        description="Optional Group code override for resolving embedding LLM_Config",
    )
    embedding_group_name: Optional[str] = Field(
        default=DEFAULT_GROUP_NAME,
        description="Sponsoring Group name for resolving embedding LLM_Config",
    )
    embedding_batch_size: int = Field(default=64, gt=0, description="Texts per embedding provider batch")
    skip_existing_embeddings: bool = Field(
        default=True,
        description="Skip re-embedding nodes whose text has not changed",
    )
    use_contextual_embedding_text: bool = Field(
        default=True,
        description="Include lightweight DICOM entity context in embedding text (Option A)",
    )
    dry_run: bool = Field(
        default=False,
        description="Load and validate all seed files without writing to Neo4j or computing embeddings",
    )

    @field_validator("source_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: Any) -> Path:
        if isinstance(v, Path):
            return v
        if isinstance(v, str) and v.strip():
            return Path(v.strip())
        raise ValueError("source_dir must be a non-empty path.")


class DicomIngestionReport(BaseModel):
    """Structured summary of a completed DICOM ingestion run."""

    source_dir: str = Field(..., description="Resolved source directory path")
    dry_run: bool = Field(default=False, description="True if no database writes were executed")
    test_id: Optional[str] = Field(default=None, description="Optional test isolation identifier")
    node_counts: Dict[str, int] = Field(default_factory=dict, description="Nodes merged/validated by Neo4j label")
    relation_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Relationships merged/validated by Neo4j relationship type",
    )
    skipped_edges: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of skipped edges due to wildcards or unmatched references",
    )
    embedding_candidates_count: int = Field(
        default=0,
        description="Total embeddable (node, NAME|DESCRIPTION) text items extracted",
    )
    embedding_report: Optional[EmbeddingBatchReport] = Field(
        default=None,
        description="Embedding computation summary when compute_embeddings=True and dry_run=False",
    )
    duration_ms: float = Field(default=0.0, description="Total elapsed duration in milliseconds")


@dataclass
class DicomParsedBundle:
    """In-memory collection of validated, deterministic DICOM graph nodes and relationships."""

    elements: List[DicomElementNode] = field(default_factory=list)
    ciods: List[DicomCIODNode] = field(default_factory=list)
    modules: List[DicomModuleNode] = field(default_factory=list)
    macros: List[DicomMacroNode] = field(default_factory=list)
    sop_classes: List[DicomSOPClassNode] = field(default_factory=list)
    confidentiality_profiles: List[DicomConfidentialityProfileNode] = field(default_factory=list)
    standard_references: List[DicomStandardReferenceNode] = field(default_factory=list)

    implements_ciod: List[ImplementsCiodRelation] = field(default_factory=list)
    has_module: List[HasModuleRelation] = field(default_factory=list)
    has_func_group_macro: List[HasFuncGroupMacroRelation] = field(default_factory=list)
    module_uses_element: List[UsesElementRelation] = field(default_factory=list)
    macro_uses_element: List[UsesElementRelation] = field(default_factory=list)
    has_confidentiality_profile: List[HasConfidentialityProfileRelation] = field(default_factory=list)
    module_references_standard: List[ReferencesStandardRelation] = field(default_factory=list)
    macro_references_standard: List[ReferencesStandardRelation] = field(default_factory=list)

    skipped_edges: Dict[str, int] = field(default_factory=dict)


def _read_json_file(path: Path) -> Any:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required DICOM seed file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _increment_counter(counter: Dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def load_dicom_seed_bundle(
    source_dir: Path,
    *,
    test_id: Optional[str] = None,
    acted_by: str = "dicom_ingestor",
    include_references: bool = True,
    include_reference_edges: bool = True,
    dedupe_reference_edges_per_owner: bool = True,
) -> DicomParsedBundle:
    """Read, validate, and build deterministic DICOM graph entities from the Innolitics JSON files."""
    base = Path(source_dir).resolve()
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"DICOM source directory not found: {base}")

    norm_test_id = normalize_test_id(test_id)
    bundle = DicomParsedBundle()

    # 1. Data Elements (attributes.json)
    elements_by_hex: Dict[str, DicomElementNode] = {}
    for raw in _read_json_file(base / "attributes.json"):
        seed = DicomAttributeSeed.model_validate(raw)
        node = DicomElementNode.from_seed(seed, test_id=norm_test_id, ActedBy=acted_by)
        elements_by_hex[node.tag_hex] = node
        bundle.elements.append(node)

    # 2. Composite IODs (ciods.json)
    ciods_by_id: Dict[str, DicomCIODNode] = {}
    ciods_by_name: Dict[str, DicomCIODNode] = {}
    for raw in _read_json_file(base / "ciods.json"):
        seed = DicomCiodSeed.model_validate(raw)
        node = DicomCIODNode.from_seed(seed, test_id=norm_test_id, ActedBy=acted_by)
        ciods_by_id[node.ciod_id] = node
        ciods_by_name[node.name] = node
        bundle.ciods.append(node)

    # 3. Modules (modules.json)
    modules_by_id: Dict[str, DicomModuleNode] = {}
    for raw in _read_json_file(base / "modules.json"):
        seed = DicomModuleSeed.model_validate(raw)
        node = DicomModuleNode.from_seed(seed, test_id=norm_test_id, ActedBy=acted_by)
        modules_by_id[node.module_id] = node
        bundle.modules.append(node)

    # 4. Macros (macros.json)
    macros_by_id: Dict[str, DicomMacroNode] = {}
    for raw in _read_json_file(base / "macros.json"):
        seed = DicomMacroSeed.model_validate(raw)
        node = DicomMacroNode.from_seed(seed, test_id=norm_test_id, ActedBy=acted_by)
        macros_by_id[node.macro_id] = node
        bundle.macros.append(node)

    # 5. SOP Classes (sops.json) + IMPLEMENTS_CIOD
    for raw in _read_json_file(base / "sops.json"):
        seed = DicomSopClassSeed.model_validate(raw)
        node = DicomSOPClassNode.from_seed(seed, test_id=norm_test_id, ActedBy=acted_by)
        bundle.sop_classes.append(node)
        target_ciod = ciods_by_name.get(node.ciod_name) or ciods_by_id.get(node.ciod_name)
        if target_ciod is not None:
            bundle.implements_ciod.append(
                ImplementsCiodRelation.from_nodes(node, target_ciod, test_id=norm_test_id, ActedBy=acted_by)
            )
        else:
            _increment_counter(bundle.skipped_edges, "IMPLEMENTS_CIOD_MISSING_CIOD")

    # 6. Confidentiality Profiles (confidentiality_profile_attributes.json) + HAS_CONFIDENTIALITY_PROFILE
    for raw in _read_json_file(base / "confidentiality_profile_attributes.json"):
        seed = DicomConfidentialityProfileAttributeSeed.model_validate(raw)
        profile_node = DicomConfidentialityProfileNode.from_seed(seed, test_id=norm_test_id, ActedBy=acted_by)
        bundle.confidentiality_profiles.append(profile_node)
        element_node = elements_by_hex.get(profile_node.attribute_id)
        if element_node is not None:
            bundle.has_confidentiality_profile.append(
                HasConfidentialityProfileRelation.from_nodes(
                    element_node,
                    profile_node,
                    test_id=norm_test_id,
                    ActedBy=acted_by,
                )
            )
        else:
            _increment_counter(bundle.skipped_edges, "HAS_CONFIDENTIALITY_PROFILE_UNMATCHED_WILDCARD")

    # 7. Standard References (references.json)
    references_by_url: Dict[str, DicomStandardReferenceNode] = {}
    if include_references:
        ref_seed = DicomReferencesSeed.model_validate(_read_json_file(base / "references.json"))
        for source_url, raw_content in ref_seed.items():
            ref_node = DicomStandardReferenceNode.from_seed(
                source_url,
                raw_content,
                test_id=norm_test_id,
                ActedBy=acted_by,
            )
            references_by_url[ref_node.source_url] = ref_node
            bundle.standard_references.append(ref_node)

    # 8. CIOD -> Module (ciod_to_modules.json)
    for raw in _read_json_file(base / "ciod_to_modules.json"):
        seed = DicomCiodToModuleSeed.model_validate(raw)
        ciod_node = ciods_by_id.get(seed.ciod_id)
        mod_node = modules_by_id.get(seed.module_id)
        if ciod_node is None or mod_node is None:
            _increment_counter(bundle.skipped_edges, "HAS_MODULE_MISSING_ENDPOINT")
            continue
        bundle.has_module.append(
            HasModuleRelation.from_seed(
                seed,
                start_node_id=ciod_node.id,
                end_node_id=mod_node.id,
                test_id=norm_test_id,
                ActedBy=acted_by,
            )
        )

    # 9. CIOD -> Functional Group Macro (ciod_to_func_group_macros.json)
    for raw in _read_json_file(base / "ciod_to_func_group_macros.json"):
        seed = DicomCiodToFuncGroupMacroSeed.model_validate(raw)
        ciod_node = ciods_by_id.get(seed.ciod_id)
        macro_node = macros_by_id.get(seed.macro_id)
        if ciod_node is None or macro_node is None:
            _increment_counter(bundle.skipped_edges, "HAS_FUNC_GROUP_MACRO_MISSING_ENDPOINT")
            continue
        bundle.has_func_group_macro.append(
            HasFuncGroupMacroRelation.from_seed(
                seed,
                start_node_id=ciod_node.id,
                end_node_id=macro_node.id,
                test_id=norm_test_id,
                ActedBy=acted_by,
            )
        )

    # 10. Module -> Attribute (module_to_attributes.json) + REFERENCES_STANDARD
    seen_module_refs: set[Tuple[str, str]] = set()
    for raw in _read_json_file(base / "module_to_attributes.json"):
        seed = DicomModuleToAttributeSeed.model_validate(raw)
        mod_node = modules_by_id.get(seed.module_id)
        tag_hex = normalize_dicom_tag_hex(seed.tag)
        elem_node = elements_by_hex.get(tag_hex)
        if mod_node is None or elem_node is None:
            _increment_counter(bundle.skipped_edges, "MODULE_USES_ELEMENT_MISSING_ENDPOINT")
            continue
        bundle.module_uses_element.append(
            UsesElementRelation.from_seed(
                seed,
                start_node_id=mod_node.id,
                end_node_id=elem_node.id,
                test_id=norm_test_id,
                ActedBy=acted_by,
            )
        )
        if include_references and include_reference_edges and seed.external_references:
            for ext_ref in seed.external_references:
                ref_node = references_by_url.get(ext_ref.source_url)
                if ref_node is None:
                    _increment_counter(bundle.skipped_edges, "REFERENCES_STANDARD_MISSING_TARGET")
                    continue
                if dedupe_reference_edges_per_owner:
                    key = (mod_node.id, ref_node.source_url)
                    if key in seen_module_refs:
                        continue
                    seen_module_refs.add(key)
                    edge_path = None
                else:
                    edge_path = seed.path
                bundle.module_references_standard.append(
                    ReferencesStandardRelation.from_seed(
                        ext_ref,
                        start_node_id=mod_node.id,
                        reference=ref_node,
                        path=edge_path,
                        test_id=norm_test_id,
                        ActedBy=acted_by,
                    )
                )

    # 11. Macro -> Attribute (macro_to_attributes.json) + REFERENCES_STANDARD
    seen_macro_refs: set[Tuple[str, str]] = set()
    for raw in _read_json_file(base / "macro_to_attributes.json"):
        seed = DicomMacroToAttributeSeed.model_validate(raw)
        macro_node = macros_by_id.get(seed.macro_id)
        tag_hex = normalize_dicom_tag_hex(seed.tag)
        elem_node = elements_by_hex.get(tag_hex)
        if macro_node is None or elem_node is None:
            _increment_counter(bundle.skipped_edges, "MACRO_USES_ELEMENT_MISSING_ENDPOINT")
            continue
        bundle.macro_uses_element.append(
            UsesElementRelation.from_seed(
                seed,
                start_node_id=macro_node.id,
                end_node_id=elem_node.id,
                test_id=norm_test_id,
                ActedBy=acted_by,
            )
        )
        if include_references and include_reference_edges and seed.external_references:
            for ext_ref in seed.external_references:
                ref_node = references_by_url.get(ext_ref.source_url)
                if ref_node is None:
                    _increment_counter(bundle.skipped_edges, "REFERENCES_STANDARD_MISSING_TARGET")
                    continue
                if dedupe_reference_edges_per_owner:
                    key = (macro_node.id, ref_node.source_url)
                    if key in seen_macro_refs:
                        continue
                    seen_macro_refs.add(key)
                    edge_path = None
                else:
                    edge_path = seed.path
                bundle.macro_references_standard.append(
                    ReferencesStandardRelation.from_seed(
                        ext_ref,
                        start_node_id=macro_node.id,
                        reference=ref_node,
                        path=edge_path,
                        test_id=norm_test_id,
                        ActedBy=acted_by,
                    )
                )

    return bundle


def extract_dicom_embedding_items(
    bundle: DicomParsedBundle,
    *,
    contextual: bool = True,
) -> List[EmbeddingItem]:
    """Extract ``EmbeddingItem`` targets for every DICOM node carrying a non-empty ``name`` or ``description``."""
    items: List[EmbeddingItem] = []

    for elem in bundle.elements:
        if elem.name:
            if contextual:
                prefix = f"DICOM Element {elem.tag} {elem.keyword}".strip()
                name_text = f"{prefix}: {elem.name}"
            else:
                name_text = elem.name
            items.append(
                EmbeddingItem(
                    target_node_id=elem.id,
                    target_label=elem.__label__,
                    target_field_tag=DictionaryEmbeddingFieldTag.NAME,
                    text=name_text,
                )
            )
        if elem.description:
            desc_text = f"DICOM Element {elem.tag} {elem.name}: {elem.description}" if contextual else elem.description
            items.append(
                EmbeddingItem(
                    target_node_id=elem.id,
                    target_label=elem.__label__,
                    target_field_tag=DictionaryEmbeddingFieldTag.DESCRIPTION,
                    text=desc_text,
                )
            )

    for ciod in bundle.ciods:
        if ciod.name:
            items.append(
                EmbeddingItem(
                    target_node_id=ciod.id,
                    target_label=ciod.__label__,
                    target_field_tag=DictionaryEmbeddingFieldTag.NAME,
                    text=f"DICOM CIOD: {ciod.name}" if contextual else ciod.name,
                )
            )
        if ciod.description:
            items.append(
                EmbeddingItem(
                    target_node_id=ciod.id,
                    target_label=ciod.__label__,
                    target_field_tag=DictionaryEmbeddingFieldTag.DESCRIPTION,
                    text=f"DICOM CIOD {ciod.name}: {ciod.description}" if contextual else ciod.description,
                )
            )

    for module in bundle.modules:
        if module.name:
            items.append(
                EmbeddingItem(
                    target_node_id=module.id,
                    target_label=module.__label__,
                    target_field_tag=DictionaryEmbeddingFieldTag.NAME,
                    text=f"DICOM Module: {module.name}" if contextual else module.name,
                )
            )
        if module.description:
            items.append(
                EmbeddingItem(
                    target_node_id=module.id,
                    target_label=module.__label__,
                    target_field_tag=DictionaryEmbeddingFieldTag.DESCRIPTION,
                    text=f"DICOM Module {module.name}: {module.description}" if contextual else module.description,
                )
            )

    for macro in bundle.macros:
        if macro.name:
            items.append(
                EmbeddingItem(
                    target_node_id=macro.id,
                    target_label=macro.__label__,
                    target_field_tag=DictionaryEmbeddingFieldTag.NAME,
                    text=f"DICOM Macro: {macro.name}" if contextual else macro.name,
                )
            )
        if macro.description:
            items.append(
                EmbeddingItem(
                    target_node_id=macro.id,
                    target_label=macro.__label__,
                    target_field_tag=DictionaryEmbeddingFieldTag.DESCRIPTION,
                    text=f"DICOM Macro {macro.name}: {macro.description}" if contextual else macro.description,
                )
            )

    for sop in bundle.sop_classes:
        if sop.name:
            sop_text = f"DICOM SOP Class {sop.sop_class_uid} ({sop.ciod_name}): {sop.name}" if contextual else sop.name
            items.append(
                EmbeddingItem(
                    target_node_id=sop.id,
                    target_label=sop.__label__,
                    target_field_tag=DictionaryEmbeddingFieldTag.NAME,
                    text=sop_text,
                )
            )

    for prof in bundle.confidentiality_profiles:
        if prof.name:
            prof_text = (
                f"DICOM Confidentiality Profile {prof.tag} (Basic={prof.basic_profile}): {prof.name}"
                if contextual
                else prof.name
            )
            items.append(
                EmbeddingItem(
                    target_node_id=prof.id,
                    target_label=prof.__label__,
                    target_field_tag=DictionaryEmbeddingFieldTag.NAME,
                    text=prof_text,
                )
            )

    return items


def _build_node_merge_cypher(label: str) -> LiteralString:
    if label not in _ALLOWED_NODE_LABELS:
        raise ValueError(f"Unsupported DICOM node label: '{label}'")
    return cast(
        LiteralString,
        f"""
UNWIND $rows AS row
MERGE (n:{label} {{id: row.id}})
ON CREATE SET n += row.props,
              n.test_id = row.test_id,
              n.Action = 'CREATE',
              n.ActedAt = datetime({{timezone: 'UTC'}}),
              n.ActedBy = row.acted_by,
              n.CreatedAt = datetime({{timezone: 'UTC'}})
ON MATCH SET  n += row.props,
              n.test_id = coalesce(row.test_id, n.test_id),
              n.Action = 'UPDATE',
              n.ActedAt = datetime({{timezone: 'UTC'}}),
              n.ActedBy = row.acted_by,
              n.UpdatedAt = datetime({{timezone: 'UTC'}})
RETURN count(n) AS written
""",
    )


def _build_rel_merge_cypher(start_label: str, rel_type: str, end_label: str) -> LiteralString:
    spec = (start_label, rel_type, end_label)
    if spec not in _ALLOWED_REL_SPECS:
        raise ValueError(f"Unsupported DICOM relationship spec: {spec}")
    return cast(
        LiteralString,
        f"""
UNWIND $rows AS row
MATCH (a:{start_label} {{id: row.start_node_id}})
MATCH (b:{end_label} {{id: row.end_node_id}})
MERGE (a)-[r:{rel_type} {{id: row.id}}]->(b)
ON CREATE SET r += row.props,
              r.test_id = row.test_id,
              r.Action = 'CREATE',
              r.ActedAt = datetime({{timezone: 'UTC'}}),
              r.ActedBy = row.acted_by,
              r.CreatedAt = datetime({{timezone: 'UTC'}})
ON MATCH SET  r += row.props,
              r.test_id = coalesce(row.test_id, r.test_id),
              r.Action = 'UPDATE',
              r.ActedAt = datetime({{timezone: 'UTC'}}),
              r.ActedBy = row.acted_by,
              r.UpdatedAt = datetime({{timezone: 'UTC'}})
RETURN count(r) AS written
""",
    )


def _node_to_row(node: DicomAuditableNode, *, acted_by: str, test_id: Optional[str]) -> Dict[str, Any]:
    raw_props = node.to_neo4j_properties()
    domain_props = {k: v for k, v in raw_props.items() if k not in _AUDIT_KEYS}
    return {
        "id": node.id,
        "props": domain_props,
        "test_id": test_id,
        "acted_by": acted_by,
    }


def _rel_to_row(rel: DicomAuditableRelation, *, acted_by: str, test_id: Optional[str]) -> Dict[str, Any]:
    raw_props = rel.to_neo4j_properties(include_endpoints=False)
    domain_props = {k: v for k, v in raw_props.items() if k not in _AUDIT_KEYS}
    return {
        "id": rel.id,
        "start_node_id": rel.start_node_id,
        "end_node_id": rel.end_node_id,
        "props": domain_props,
        "test_id": test_id,
        "acted_by": acted_by,
    }


def _chunked(items: Sequence[Any], batch_size: int) -> List[Sequence[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


class DicomIngesterService:
    """Service executing idempotent batched Neo4j ingestion and optional embedding for DICOM standards."""

    def __init__(
        self,
        *,
        client: Optional[Neo4jClient] = None,
        system_service: Optional[SystemService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.client = client or neo4j_client
        self.settings = settings or get_settings()
        self.system_service = system_service or SystemService(client=self.client)
        self.embedding_service = embedding_service or EmbeddingService(
            client=self.client,
            llm_config_service=self.system_service.llm_service,
            settings=self.settings,
        )

    def _merge_nodes_sync(
        self,
        label: str,
        nodes: Sequence[DicomAuditableNode],
        *,
        batch_size: int,
        acted_by: str,
        test_id: Optional[str],
    ) -> int:
        if not nodes:
            return 0
        query = _build_node_merge_cypher(label)
        total = 0
        for chunk in _chunked(nodes, batch_size):
            rows = [_node_to_row(node, acted_by=acted_by, test_id=test_id) for node in chunk]

            def _tx(tx: Any, q: LiteralString = query, batch_rows: List[Dict[str, Any]] = rows) -> int:
                rec = tx.run(q, {"rows": batch_rows}).single()
                return int(rec["written"]) if rec is not None and rec.get("written") else 0

            total += int(self.client.execute_write_tx_sync(_tx))
        return total

    def _merge_relations_sync(
        self,
        start_label: str,
        rel_type: str,
        end_label: str,
        rels: Sequence[DicomAuditableRelation],
        *,
        batch_size: int,
        acted_by: str,
        test_id: Optional[str],
    ) -> int:
        if not rels:
            return 0
        query = _build_rel_merge_cypher(start_label, rel_type, end_label)
        total = 0
        for chunk in _chunked(rels, batch_size):
            rows = [_rel_to_row(rel, acted_by=acted_by, test_id=test_id) for rel in chunk]

            def _tx(tx: Any, q: LiteralString = query, batch_rows: List[Dict[str, Any]] = rows) -> int:
                rec = tx.run(q, {"rows": batch_rows}).single()
                return int(rec["written"]) if rec is not None and rec.get("written") else 0

            total += int(self.client.execute_write_tx_sync(_tx))
        return total

    def ingest_sync(
        self,
        options: DicomIngestionOptions,
        *,
        embed_callback: Optional[Callable[[List[EmbeddingItem], DicomIngestionOptions], EmbeddingBatchReport]] = None,
    ) -> DicomIngestionReport:
        """Synchronously ingest the DICOM dictionary (and optionally compute node embeddings)."""
        start_time = time.perf_counter()
        norm_test_id = normalize_test_id(options.test_id)

        bundle = load_dicom_seed_bundle(
            options.source_dir,
            test_id=norm_test_id,
            acted_by=options.acted_by,
            include_references=options.include_references,
            include_reference_edges=options.include_reference_edges,
            dedupe_reference_edges_per_owner=options.dedupe_reference_edges_per_owner,
        )
        embedding_items = extract_dicom_embedding_items(
            bundle,
            contextual=options.use_contextual_embedding_text,
        )

        if options.dry_run:
            return DicomIngestionReport(
                source_dir=str(options.source_dir.resolve()),
                dry_run=True,
                test_id=norm_test_id,
                node_counts={
                    "DicomElement": len(bundle.elements),
                    "DicomCIOD": len(bundle.ciods),
                    "DicomModule": len(bundle.modules),
                    "DicomMacro": len(bundle.macros),
                    "DicomSOPClass": len(bundle.sop_classes),
                    "DicomConfidentialityProfile": len(bundle.confidentiality_profiles),
                    "DicomStandardReference": len(bundle.standard_references),
                },
                relation_counts={
                    "IMPLEMENTS_CIOD": len(bundle.implements_ciod),
                    "HAS_MODULE": len(bundle.has_module),
                    "HAS_FUNC_GROUP_MACRO": len(bundle.has_func_group_macro),
                    "USES_ELEMENT": len(bundle.module_uses_element) + len(bundle.macro_uses_element),
                    "HAS_CONFIDENTIALITY_PROFILE": len(bundle.has_confidentiality_profile),
                    "REFERENCES_STANDARD": len(bundle.module_references_standard)
                    + len(bundle.macro_references_standard),
                },
                skipped_edges=dict(bundle.skipped_edges),
                embedding_candidates_count=len(embedding_items),
                embedding_report=None,
                duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )

        if options.bootstrap_system:
            self.system_service.bootstrap_system_sync(
                acted_by=options.acted_by,
                test_id=norm_test_id,
                settings=self.settings,
                ensure_schema=options.ensure_schema,
                seed_default_llm_configs=True,
            )
        elif options.ensure_schema:
            apply_schema_sync(self.client)

        node_counts: Dict[str, int] = {
            "DicomElement": self._merge_nodes_sync(
                "DicomElement",
                bundle.elements,
                batch_size=options.batch_size,
                acted_by=options.acted_by,
                test_id=norm_test_id,
            ),
            "DicomCIOD": self._merge_nodes_sync(
                "DicomCIOD",
                bundle.ciods,
                batch_size=options.batch_size,
                acted_by=options.acted_by,
                test_id=norm_test_id,
            ),
            "DicomModule": self._merge_nodes_sync(
                "DicomModule",
                bundle.modules,
                batch_size=options.batch_size,
                acted_by=options.acted_by,
                test_id=norm_test_id,
            ),
            "DicomMacro": self._merge_nodes_sync(
                "DicomMacro",
                bundle.macros,
                batch_size=options.batch_size,
                acted_by=options.acted_by,
                test_id=norm_test_id,
            ),
            "DicomSOPClass": self._merge_nodes_sync(
                "DicomSOPClass",
                bundle.sop_classes,
                batch_size=options.batch_size,
                acted_by=options.acted_by,
                test_id=norm_test_id,
            ),
            "DicomConfidentialityProfile": self._merge_nodes_sync(
                "DicomConfidentialityProfile",
                bundle.confidentiality_profiles,
                batch_size=options.batch_size,
                acted_by=options.acted_by,
                test_id=norm_test_id,
            ),
            "DicomStandardReference": self._merge_nodes_sync(
                "DicomStandardReference",
                bundle.standard_references,
                batch_size=options.batch_size,
                acted_by=options.acted_by,
                test_id=norm_test_id,
            ),
        }

        mod_uses_count = self._merge_relations_sync(
            "DicomModule",
            "USES_ELEMENT",
            "DicomElement",
            bundle.module_uses_element,
            batch_size=options.batch_size,
            acted_by=options.acted_by,
            test_id=norm_test_id,
        )
        macro_uses_count = self._merge_relations_sync(
            "DicomMacro",
            "USES_ELEMENT",
            "DicomElement",
            bundle.macro_uses_element,
            batch_size=options.batch_size,
            acted_by=options.acted_by,
            test_id=norm_test_id,
        )
        mod_refs_count = self._merge_relations_sync(
            "DicomModule",
            "REFERENCES_STANDARD",
            "DicomStandardReference",
            bundle.module_references_standard,
            batch_size=options.batch_size,
            acted_by=options.acted_by,
            test_id=norm_test_id,
        )
        macro_refs_count = self._merge_relations_sync(
            "DicomMacro",
            "REFERENCES_STANDARD",
            "DicomStandardReference",
            bundle.macro_references_standard,
            batch_size=options.batch_size,
            acted_by=options.acted_by,
            test_id=norm_test_id,
        )

        relation_counts: Dict[str, int] = {
            "IMPLEMENTS_CIOD": self._merge_relations_sync(
                "DicomSOPClass",
                "IMPLEMENTS_CIOD",
                "DicomCIOD",
                bundle.implements_ciod,
                batch_size=options.batch_size,
                acted_by=options.acted_by,
                test_id=norm_test_id,
            ),
            "HAS_MODULE": self._merge_relations_sync(
                "DicomCIOD",
                "HAS_MODULE",
                "DicomModule",
                bundle.has_module,
                batch_size=options.batch_size,
                acted_by=options.acted_by,
                test_id=norm_test_id,
            ),
            "HAS_FUNC_GROUP_MACRO": self._merge_relations_sync(
                "DicomCIOD",
                "HAS_FUNC_GROUP_MACRO",
                "DicomMacro",
                bundle.has_func_group_macro,
                batch_size=options.batch_size,
                acted_by=options.acted_by,
                test_id=norm_test_id,
            ),
            "USES_ELEMENT": mod_uses_count + macro_uses_count,
            "HAS_CONFIDENTIALITY_PROFILE": self._merge_relations_sync(
                "DicomElement",
                "HAS_CONFIDENTIALITY_PROFILE",
                "DicomConfidentialityProfile",
                bundle.has_confidentiality_profile,
                batch_size=options.batch_size,
                acted_by=options.acted_by,
                test_id=norm_test_id,
            ),
            "REFERENCES_STANDARD": mod_refs_count + macro_refs_count,
        }

        embedding_report: Optional[EmbeddingBatchReport] = None
        if options.compute_embeddings and embedding_items:
            if embed_callback is not None:
                embedding_report = embed_callback(embedding_items, options)
            else:
                embedding_report = self.embedding_service.embed_items_sync(
                    embedding_items,
                    tag=options.embedding_tag,
                    group_code=options.embedding_group_code,
                    group_name=options.embedding_group_name,
                    batch_size=options.embedding_batch_size,
                    skip_existing=options.skip_existing_embeddings,
                    test_id=norm_test_id,
                    acted_by=options.acted_by,
                )

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "dicom_ingestion_completed",
            source_dir=str(options.source_dir.resolve()),
            total_nodes=sum(node_counts.values()),
            total_relations=sum(relation_counts.values()),
            embedding_candidates=len(embedding_items),
            embeddings_persisted=embedding_report.persisted_count if embedding_report else 0,
            test_id=norm_test_id,
            duration_ms=duration_ms,
        )
        return DicomIngestionReport(
            source_dir=str(options.source_dir.resolve()),
            dry_run=False,
            test_id=norm_test_id,
            node_counts=node_counts,
            relation_counts=relation_counts,
            skipped_edges=dict(bundle.skipped_edges),
            embedding_candidates_count=len(embedding_items),
            embedding_report=embedding_report,
            duration_ms=duration_ms,
        )

    async def ingest(self, options: DicomIngestionOptions) -> DicomIngestionReport:
        """Async wrapper executing ``ingest_sync`` off the event loop."""
        return await asyncio.to_thread(self.ingest_sync, options)

    def cleanup_test_data_sync(self, test_id: str) -> Dict[str, Any]:
        """Safely purge ONLY nodes carrying the given non-empty ``test_id``."""
        return self.system_service.cleanup_test_data_sync(test_id)

    async def cleanup_test_data(self, test_id: str) -> Dict[str, Any]:
        """Async safe purge of ONLY nodes carrying the given non-empty ``test_id``."""
        return await self.system_service.cleanup_test_data(test_id)


dicom_ingester_service = DicomIngesterService()

__all__ = [
    "DicomIngestionOptions",
    "DicomIngestionReport",
    "DicomParsedBundle",
    "load_dicom_seed_bundle",
    "extract_dicom_embedding_items",
    "DicomIngesterService",
    "dicom_ingester_service",
]
