"""Idempotent streaming batched Neo4j ingestion pipeline for OHDSI Athena clinical terminologies."""

from __future__ import annotations

import asyncio
import csv
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, LiteralString, Optional, Tuple, cast

from pydantic import BaseModel, Field, field_validator

from gosmart_dictionary_service.core.config import Settings, get_settings
from gosmart_dictionary_service.core.logging import get_logger
from gosmart_dictionary_service.db.neo4j import Neo4jClient, neo4j_client
from gosmart_dictionary_service.db.schema_loader import apply_schema_sync
from gosmart_dictionary_service.model.actor_message import EmbeddingItem
from gosmart_dictionary_service.model.clinical import (
    DEFAULT_CLINICAL_VOCABULARIES,
    DEFAULT_SYNONYM_LANGUAGE_IDS,
    DEMO_CORE_DOMAINS,
    AthenaRelationshipSeed,
    AthenaVocabularySeed,
    ClinicalIngestionProfile,
    CodeSystemNode,
    classify_clinical_relationship,
    compute_clinical_relation_id,
    compute_concept_id,
    parse_athena_date,
    resolve_code_system_uri,
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

logger = get_logger("gosmart_ds.clinical_terms_ingestor")

_AUDIT_KEYS = frozenset({"id", "Action", "ActedAt", "ActedBy", "CreatedAt", "UpdatedAt", "test_id"})
_ALLOWED_CONCEPT_REL_TYPES = frozenset({"IS_A", "MAPS_TO", "HAS_ATTRIBUTE", "REPLACED_BY"})
_PROGRESS_LOG_INTERVAL = 250_000


class ClinicalTermsIngestionOptions(BaseModel):
    """Configuration options controlling a streaming OHDSI Athena clinical terms ingestion run."""

    source_dir: Path = Field(
        default=Path("./data-source/clinical-terms"),
        description="Directory containing OHDSI Athena tab-delimited CSV files",
    )
    profile: ClinicalIngestionProfile = Field(
        default=ClinicalIngestionProfile.FULL,
        description="Preset ingestion scope ('full', 'valid', or 'demo-core')",
    )
    vocabularies: Optional[List[str]] = Field(
        default=None,
        description="Optional explicit vocabulary_id allowlist override (defaults to SNOMED, LOINC, ICD10CM, UCUM)",
    )
    all_vocabularies: bool = Field(
        default=False,
        description="Ingest all vocabularies in the Athena export (overrides default clinical allowlist)",
    )
    domains: Optional[List[str]] = Field(
        default=None,
        description="Optional explicit domain_id filter override",
    )
    include_deprecated: Optional[bool] = Field(
        default=None,
        description="Override whether deprecated/upgraded ('D'/'U') concepts are included",
    )
    include_synonyms: bool = Field(default=True, description="Attach synonyms from CONCEPT_SYNONYM.csv")
    synonym_language_ids: Optional[List[int]] = Field(
        default=None,
        description="Optional language_concept_id allowlist for synonyms (default: English 4180186)",
    )
    all_synonym_languages: bool = Field(
        default=False,
        description="Include synonyms across all languages in CONCEPT_SYNONYM.csv",
    )
    max_synonyms_per_concept: int = Field(
        default=25,
        gt=0,
        description="Maximum deduplicated synonyms stored per Concept node",
    )
    include_relationships: bool = Field(
        default=True,
        description="Ingest canonical relationships from CONCEPT_RELATIONSHIP.csv",
    )
    include_attribute_edges: bool = Field(
        default=True,
        description="Ingest :HAS_ATTRIBUTE defining ontology relationships",
    )
    include_rf2_metadata_edges: bool = Field(
        default=False,
        description="Include SNOMED RF2 administrative edges ('Has status', 'Has Module')",
    )
    include_replaced_by_edges: bool = Field(
        default=True,
        description="Include historical replacement links (:REPLACED_BY)",
    )
    max_concepts: Optional[int] = Field(
        default=None,
        gt=0,
        description="Optional cap on ingested Concept nodes (useful for dev/smoke testing)",
    )
    batch_size: int = Field(default=5000, gt=0, description="Rows per Neo4j UNWIND MERGE write transaction")
    acted_by: str = Field(default="clinical_terms_ingestor", description="Actor identifier recorded in audit fields")
    test_id: Optional[str] = Field(default=None, description="Optional test isolation identifier")
    ensure_schema: bool = Field(default=True, description="Apply Neo4j DDL schema before writing nodes")
    bootstrap_system: bool = Field(
        default=False,
        description="Bootstrap System + GoSmart Health Group + default LLM_Config before ingesting",
    )
    compute_embeddings: bool = Field(
        default=False,
        description="Compute and persist :Embedding nodes for ingested :Concept nodes",
    )
    embed_standard_only: bool = Field(
        default=True,
        description="Restrict automatic embeddings to valid Standard ('S') or Classification ('C') concepts",
    )
    embed_synonyms: bool = Field(
        default=False,
        description="Also emit separate :Embedding items for concept synonyms",
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
    embedding_dispatch_chunk_size: int = Field(
        default=1000,
        gt=0,
        description="Maximum EmbeddingItem entries dispatched per chunk to EmbeddingActor",
    )
    skip_existing_embeddings: bool = Field(
        default=True,
        description="Skip re-embedding nodes whose text has not changed",
    )
    use_contextual_embedding_text: bool = Field(
        default=True,
        description="Prefix embedding text with vocabulary, domain, and code context",
    )
    dry_run: bool = Field(
        default=False,
        description="Stream and validate CSV files without writing to Neo4j or computing embeddings",
    )

    @field_validator("source_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: Any) -> Path:
        if isinstance(v, Path):
            return v
        if isinstance(v, str) and v.strip():
            return Path(v.strip())
        raise ValueError("source_dir must be a non-empty path.")

    def effective_vocabularies(self) -> Optional[frozenset[str]]:
        """Resolve active vocabulary_id allowlist (``None`` means all vocabularies)."""
        if self.all_vocabularies:
            return None
        if self.vocabularies is not None:
            cleaned = frozenset(v.strip() for v in self.vocabularies if v and v.strip())
            return cleaned if cleaned else frozenset(DEFAULT_CLINICAL_VOCABULARIES)
        return frozenset(DEFAULT_CLINICAL_VOCABULARIES)

    def effective_domains(self) -> Optional[frozenset[str]]:
        """Resolve active domain_id filter (``None`` means all domains)."""
        if self.domains is not None:
            cleaned = frozenset(d.strip() for d in self.domains if d and d.strip())
            return cleaned if cleaned else None
        if self.profile == ClinicalIngestionProfile.DEMO_CORE:
            return frozenset(DEMO_CORE_DOMAINS)
        return None

    def effective_include_deprecated(self) -> bool:
        """Resolve whether deprecated/upgraded ('D'/'U') concepts should be ingested."""
        if self.include_deprecated is not None:
            return self.include_deprecated
        return self.profile == ClinicalIngestionProfile.FULL

    def effective_synonym_language_ids(self) -> Optional[frozenset[int]]:
        """Resolve language_concept_id filter for CONCEPT_SYNONYM.csv (``None`` means all languages)."""
        if self.all_synonym_languages:
            return None
        if self.synonym_language_ids is not None:
            return frozenset(self.synonym_language_ids)
        return frozenset(DEFAULT_SYNONYM_LANGUAGE_IDS)


class ClinicalTermsIngestionReport(BaseModel):
    """Structured summary of a completed OHDSI Athena clinical terms ingestion run."""

    source_dir: str = Field(..., description="Resolved source directory path")
    profile: str = Field(..., description="Ingestion profile applied ('full', 'valid', or 'demo-core')")
    dry_run: bool = Field(default=False, description="True if no database writes were executed")
    test_id: Optional[str] = Field(default=None, description="Optional test isolation identifier")
    vocabularies_ingested: List[str] = Field(
        default_factory=list,
        description="Sorted list of vocabulary_ids ingested as :CodeSystem / :Concept entities",
    )
    node_counts: Dict[str, int] = Field(default_factory=dict, description="Nodes merged/validated by Neo4j label")
    relation_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Relationships merged/validated by Neo4j relationship type",
    )
    skipped_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Counts of skipped rows by reason (vocab filter, self-loops, mirror edges, etc.)",
    )
    synonyms_attached: int = Field(default=0, description="Total deduplicated synonyms attached to ingested Concepts")
    embedding_candidates_count: int = Field(
        default=0,
        description="Total embeddable (Concept, DISPLAY|SYNONYM) text items extracted",
    )
    embedding_report: Optional[EmbeddingBatchReport] = Field(
        default=None,
        description="Aggregated embedding summary when compute_embeddings=True and dry_run=False",
    )
    duration_ms: float = Field(default=0.0, description="Total elapsed duration in milliseconds")


@dataclass
class _RelMetadata:
    equivalence: Optional[str]
    confidence: Optional[float]
    mapping_source: Optional[str]


_MERGE_CODE_SYSTEMS_CYPHER: LiteralString = """
UNWIND $rows AS row
MERGE (n:CodeSystem {id: row.id})
ON CREATE SET n += row.props,
              n.test_id = row.test_id,
              n.Action = 'CREATE',
              n.ActedAt = datetime({timezone: 'UTC'}),
              n.ActedBy = row.acted_by,
              n.CreatedAt = datetime({timezone: 'UTC'})
ON MATCH SET  n += row.props,
              n.test_id = coalesce(row.test_id, n.test_id),
              n.Action = 'UPDATE',
              n.ActedAt = datetime({timezone: 'UTC'}),
              n.ActedBy = row.acted_by,
              n.UpdatedAt = datetime({timezone: 'UTC'})
RETURN count(n) AS written
"""

_MERGE_CONCEPTS_AND_SYSTEM_LINKS_CYPHER: LiteralString = """
UNWIND $rows AS row
MATCH (cs:CodeSystem {id: row.code_system_id})
MERGE (n:Concept {id: row.id})
ON CREATE SET n += row.props,
              n.test_id = row.test_id,
              n.Action = 'CREATE',
              n.ActedAt = datetime({timezone: 'UTC'}),
              n.ActedBy = row.acted_by,
              n.CreatedAt = datetime({timezone: 'UTC'})
ON MATCH SET  n += row.props,
              n.test_id = coalesce(row.test_id, n.test_id),
              n.Action = 'UPDATE',
              n.ActedAt = datetime({timezone: 'UTC'}),
              n.ActedBy = row.acted_by,
              n.UpdatedAt = datetime({timezone: 'UTC'})
MERGE (n)-[r:BELONGS_TO_SYSTEM {id: row.belongs_rel_id}]->(cs)
ON CREATE SET r.test_id = row.test_id,
              r.Action = 'CREATE',
              r.ActedAt = datetime({timezone: 'UTC'}),
              r.ActedBy = row.acted_by,
              r.CreatedAt = datetime({timezone: 'UTC'})
ON MATCH SET  r.test_id = coalesce(row.test_id, r.test_id),
              r.Action = 'UPDATE',
              r.ActedAt = datetime({timezone: 'UTC'}),
              r.ActedBy = row.acted_by,
              r.UpdatedAt = datetime({timezone: 'UTC'})
RETURN count(n) AS concepts_written, count(r) AS links_written
"""


def _build_concept_rel_merge_cypher(rel_type: str) -> LiteralString:
    if rel_type not in _ALLOWED_CONCEPT_REL_TYPES:
        raise ValueError(f"Unsupported clinical relationship type: '{rel_type}'")
    return cast(
        LiteralString,
        f"""
UNWIND $rows AS row
MATCH (a:Concept {{id: row.start_node_id}})
MATCH (b:Concept {{id: row.end_node_id}})
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


def _increment_counter(counter: Dict[str, int], key: str, delta: int = 1) -> None:
    counter[key] = counter.get(key, 0) + delta


def _iter_tsv_rows(
    path: Path,
    required_columns: Sequence[str],
) -> Iterator[Tuple[Dict[str, int], List[str]]]:
    """Stream a tab-delimited Athena CSV file, validating required header columns."""
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required Athena clinical terms file not found: {path}")
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Athena CSV file is empty (missing header): {path}")
        col_idx = {col.strip(): i for i, col in enumerate(header)}
        missing = [col for col in required_columns if col not in col_idx]
        if missing:
            raise ValueError(f"Athena CSV file '{path.name}' missing required columns: {missing}")
        for row in reader:
            if not row:
                continue
            yield col_idx, row


def _safe_cell(row: List[str], idx: int) -> str:
    return row[idx].strip() if idx < len(row) else ""


def _load_vocabularies(
    base: Path,
    *,
    test_id: Optional[str],
    acted_by: str,
) -> Dict[str, CodeSystemNode]:
    """Load and validate ``VOCABULARY.csv`` into ``CodeSystemNode`` instances keyed by ``vocabulary_id``."""
    required = (
        "vocabulary_id",
        "vocabulary_name",
        "vocabulary_reference",
        "vocabulary_version",
        "vocabulary_concept_id",
    )
    vocab_nodes: Dict[str, CodeSystemNode] = {}
    for col, row in _iter_tsv_rows(base / "VOCABULARY.csv", required):
        seed = AthenaVocabularySeed(
            vocabulary_id=_safe_cell(row, col["vocabulary_id"]),
            vocabulary_name=_safe_cell(row, col["vocabulary_name"]),
            vocabulary_reference=_safe_cell(row, col["vocabulary_reference"]),
            vocabulary_version=_safe_cell(row, col["vocabulary_version"]),
            vocabulary_concept_id=int(_safe_cell(row, col["vocabulary_concept_id"])),
        )
        node = CodeSystemNode.from_seed(seed, test_id=test_id, ActedBy=acted_by)
        vocab_nodes[node.vocabulary_id] = node
    return vocab_nodes


def _load_relationship_registry(base: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load ``RELATIONSHIP.csv`` into ``(reverse_rel_map, rel_name_map)``."""
    required = ("relationship_id", "relationship_name", "reverse_relationship_id", "relationship_concept_id")
    reverse_map: Dict[str, str] = {}
    name_map: Dict[str, str] = {}
    for col, row in _iter_tsv_rows(base / "RELATIONSHIP.csv", required):
        seed = AthenaRelationshipSeed(
            relationship_id=_safe_cell(row, col["relationship_id"]),
            relationship_name=_safe_cell(row, col["relationship_name"]),
            is_hierarchical=_safe_cell(row, col["is_hierarchical"]) if "is_hierarchical" in col else "0",
            defines_ancestry=_safe_cell(row, col["defines_ancestry"]) if "defines_ancestry" in col else "0",
            reverse_relationship_id=_safe_cell(row, col["reverse_relationship_id"]),
            relationship_concept_id=int(_safe_cell(row, col["relationship_concept_id"])),
        )
        reverse_map[seed.relationship_id] = seed.reverse_relationship_id
        name_map[seed.relationship_id] = seed.relationship_name
    return reverse_map, name_map


def _load_concept_metadata(base: Path) -> Dict[int, str]:
    """Load optional ``CONCEPT_METADATA.csv`` mapping ``concept_id -> concept_category``."""
    path = base / "CONCEPT_METADATA.csv"
    if not path.exists() or not path.is_file():
        return {}
    result: Dict[int, str] = {}
    for col, row in _iter_tsv_rows(path, ("concept_id", "concept_category")):
        cid_str = _safe_cell(row, col["concept_id"])
        cat = _safe_cell(row, col["concept_category"])
        if cid_str and cat:
            result[int(cid_str)] = cat
    return result


def _load_relationship_metadata(base: Path) -> Dict[Tuple[int, int, str], _RelMetadata]:
    """Load optional ``CONCEPT_RELATIONSHIP_METADATA.csv`` mapping ``(c1, c2, rel_id) -> _RelMetadata``."""
    path = base / "CONCEPT_RELATIONSHIP_METADATA.csv"
    if not path.exists() or not path.is_file():
        return {}
    result: Dict[Tuple[int, int, str], _RelMetadata] = {}
    required = ("concept_id_1", "concept_id_2", "relationship_id")
    for col, row in _iter_tsv_rows(path, required):
        c1_str = _safe_cell(row, col["concept_id_1"])
        c2_str = _safe_cell(row, col["concept_id_2"])
        rid = _safe_cell(row, col["relationship_id"])
        if not c1_str or not c2_str or not rid:
            continue
        equiv = _safe_cell(row, col["relationship_predicate_id"]) if "relationship_predicate_id" in col else ""
        src = _safe_cell(row, col["mapping_source"]) if "mapping_source" in col else ""
        conf_raw = _safe_cell(row, col["confidence"]) if "confidence" in col else ""
        conf: Optional[float] = None
        if conf_raw:
            try:
                conf = float(conf_raw)
            except ValueError:
                conf = None
        if equiv or src or conf is not None:
            result[(int(c1_str), int(c2_str), rid)] = _RelMetadata(
                equivalence=equiv or None,
                confidence=conf,
                mapping_source=src or None,
            )
    return result


def _prescan_selected_concept_ids(
    concept_path: Path,
    *,
    allowed_vocabs: Optional[frozenset[str]],
    allowed_domains: Optional[frozenset[str]],
    include_deprecated: bool,
    max_concepts: Optional[int],
) -> set[int]:
    """Fast pre-scan of ``CONCEPT.csv`` to collect eligible ``concept_id``s before loading synonyms."""
    required = ("concept_id", "domain_id", "vocabulary_id", "invalid_reason")
    selected: set[int] = set()
    for col, row in _iter_tsv_rows(concept_path, required):
        vid = _safe_cell(row, col["vocabulary_id"])
        if allowed_vocabs is not None and vid not in allowed_vocabs:
            continue
        did = _safe_cell(row, col["domain_id"])
        if allowed_domains is not None and did not in allowed_domains:
            continue
        inv = _safe_cell(row, col["invalid_reason"])
        if inv and not include_deprecated:
            continue
        selected.add(int(_safe_cell(row, col["concept_id"])))
        if max_concepts is not None and len(selected) >= max_concepts:
            break
    return selected


def _load_synonyms_for_selected(
    base: Path,
    selected_cids: set[int],
    *,
    allowed_lang_ids: Optional[frozenset[int]],
    max_per_concept: int,
    skipped_counts: Dict[str, int],
) -> Dict[int, List[str]]:
    """Stream ``CONCEPT_SYNONYM.csv`` keeping only deduplicated synonyms for selected concepts."""
    path = base / "CONCEPT_SYNONYM.csv"
    if not path.exists() or not path.is_file() or not selected_cids:
        return {}
    required = ("concept_id", "concept_synonym_name", "language_concept_id")
    synonyms: Dict[int, List[str]] = {}
    scanned = 0
    for col, row in _iter_tsv_rows(path, required):
        scanned += 1
        cid = int(_safe_cell(row, col["concept_id"]))
        if cid not in selected_cids:
            _increment_counter(skipped_counts, "SYNONYM_UNSELECTED_CONCEPT")
            continue
        if allowed_lang_ids is not None:
            lang_id = int(_safe_cell(row, col["language_concept_id"]))
            if lang_id not in allowed_lang_ids:
                _increment_counter(skipped_counts, "SYNONYM_FILTERED_LANGUAGE")
                continue
        syn = _safe_cell(row, col["concept_synonym_name"])
        if not syn:
            continue
        bucket = synonyms.get(cid)
        if bucket is None:
            synonyms[cid] = [syn]
        elif len(bucket) < max_per_concept and syn not in bucket:
            bucket.append(syn)
        else:
            _increment_counter(skipped_counts, "SYNONYM_CAPPED_OR_DUPLICATE")
        if scanned % _PROGRESS_LOG_INTERVAL == 0:
            logger.debug("clinical_synonyms_scan_progress", rows_scanned=scanned, concepts_with_synonyms=len(synonyms))
    return synonyms


def build_concept_embedding_items(
    *,
    concept_id: str,
    vocabulary_id: str,
    domain_id: str,
    code: str,
    display: str,
    synonyms: Sequence[str],
    contextual: bool = True,
    embed_synonyms: bool = False,
) -> List[EmbeddingItem]:
    """Build ``EmbeddingItem`` payloads for a single clinical ``:Concept`` node."""
    items: List[EmbeddingItem] = []
    if display:
        display_text = f"{vocabulary_id} {domain_id} ({code}): {display}" if contextual else display
        items.append(
            EmbeddingItem(
                target_node_id=concept_id,
                target_label="Concept",
                target_field_tag=DictionaryEmbeddingFieldTag.DISPLAY,
                text=display_text,
            )
        )
    if embed_synonyms and synonyms:
        joined = "; ".join(synonyms[:10])
        syn_text = f"{vocabulary_id} {domain_id} ({code}) synonyms: {joined}" if contextual else joined
        items.append(
            EmbeddingItem(
                target_node_id=concept_id,
                target_label="Concept",
                target_field_tag=DictionaryEmbeddingFieldTag.SYNONYM,
                text=syn_text,
            )
        )
    return items


def _merge_embedding_reports(
    acc: Optional[EmbeddingBatchReport],
    part: EmbeddingBatchReport,
) -> EmbeddingBatchReport:
    """Combine chunk-level ``EmbeddingBatchReport`` summaries into a single aggregate report."""
    if acc is None:
        return part.model_copy(deep=True)
    return EmbeddingBatchReport(
        total_items=acc.total_items + part.total_items,
        embedded_count=acc.embedded_count + part.embedded_count,
        skipped_existing_count=acc.skipped_existing_count + part.skipped_existing_count,
        persisted_count=acc.persisted_count + part.persisted_count,
        model_tag=part.model_tag or acc.model_tag,
        provider=part.provider or acc.provider,
        model=part.model or acc.model,
        dimensions=part.dimensions or acc.dimensions,
        test_id=acc.test_id or part.test_id,
        duration_ms=round(acc.duration_ms + part.duration_ms, 2),
    )


def _code_system_to_row(node: CodeSystemNode, *, acted_by: str, test_id: Optional[str]) -> Dict[str, Any]:
    raw_props = node.to_neo4j_properties()
    domain_props = {k: v for k, v in raw_props.items() if k not in _AUDIT_KEYS}
    return {
        "id": node.id,
        "props": domain_props,
        "test_id": test_id,
        "acted_by": acted_by,
    }


class ClinicalTermsIngesterService:
    """Streaming, idempotent batched Neo4j ingestor for OHDSI Athena clinical terminologies."""

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

    def _write_code_systems_sync(
        self,
        nodes: Sequence[CodeSystemNode],
        *,
        acted_by: str,
        test_id: Optional[str],
    ) -> int:
        if not nodes:
            return 0
        rows = [_code_system_to_row(n, acted_by=acted_by, test_id=test_id) for n in nodes]

        def _tx(tx: Any) -> int:
            rec = tx.run(_MERGE_CODE_SYSTEMS_CYPHER, {"rows": rows}).single()
            return int(rec["written"]) if rec is not None and rec.get("written") else 0

        return int(self.client.execute_write_tx_sync(_tx))

    def _write_concept_batch_sync(self, rows: List[Dict[str, Any]]) -> Tuple[int, int]:
        if not rows:
            return 0, 0

        def _tx(tx: Any) -> Tuple[int, int]:
            rec = tx.run(_MERGE_CONCEPTS_AND_SYSTEM_LINKS_CYPHER, {"rows": rows}).single()
            if rec is None:
                return 0, 0
            return int(rec["concepts_written"] or 0), int(rec["links_written"] or 0)

        return cast(Tuple[int, int], self.client.execute_write_tx_sync(_tx))

    def _write_concept_rel_batch_sync(self, rel_type: str, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        query = _build_concept_rel_merge_cypher(rel_type)

        def _tx(tx: Any) -> int:
            rec = tx.run(query, {"rows": rows}).single()
            return int(rec["written"]) if rec is not None and rec.get("written") else 0

        return int(self.client.execute_write_tx_sync(_tx))

    def _dispatch_embedding_chunk(
        self,
        items: List[EmbeddingItem],
        options: ClinicalTermsIngestionOptions,
        norm_test_id: Optional[str],
        embed_callback: Optional[Callable[[List[EmbeddingItem], ClinicalTermsIngestionOptions], EmbeddingBatchReport]],
    ) -> EmbeddingBatchReport:
        if embed_callback is not None:
            return embed_callback(items, options)
        return self.embedding_service.embed_items_sync(
            items,
            tag=options.embedding_tag,
            group_code=options.embedding_group_code,
            group_name=options.embedding_group_name,
            batch_size=options.embedding_batch_size,
            skip_existing=options.skip_existing_embeddings,
            test_id=norm_test_id,
            acted_by=options.acted_by,
        )

    def ingest_sync(
        self,
        options: ClinicalTermsIngestionOptions,
        *,
        embed_callback: Optional[
            Callable[[List[EmbeddingItem], ClinicalTermsIngestionOptions], EmbeddingBatchReport]
        ] = None,
    ) -> ClinicalTermsIngestionReport:
        """Synchronously stream and ingest OHDSI Athena clinical terminologies into Neo4j."""
        start_time = time.perf_counter()
        base = Path(options.source_dir).resolve()
        if not base.exists() or not base.is_dir():
            raise FileNotFoundError(f"Clinical terms source directory not found: {base}")

        norm_test_id = normalize_test_id(options.test_id)
        allowed_vocabs = options.effective_vocabularies()
        allowed_domains = options.effective_domains()
        include_deprecated = options.effective_include_deprecated()
        allowed_lang_ids = options.effective_synonym_language_ids()

        skipped_counts: Dict[str, int] = {}
        node_counts: Dict[str, int] = {"CodeSystem": 0, "Concept": 0}
        relation_counts: Dict[str, int] = {
            "BELONGS_TO_SYSTEM": 0,
            "IS_A": 0,
            "MAPS_TO": 0,
            "HAS_ATTRIBUTE": 0,
            "REPLACED_BY": 0,
        }

        # Apply schema / bootstrap before writes
        if not options.dry_run:
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

        # Pass 1: Reference registries (VOCABULARY, RELATIONSHIP, optional metadata CSVs)
        vocab_registry = _load_vocabularies(base, test_id=norm_test_id, acted_by=options.acted_by)
        reverse_rel_map, rel_name_map = _load_relationship_registry(base)
        concept_category_map = _load_concept_metadata(base)
        rel_metadata_map = _load_relationship_metadata(base) if options.include_relationships else {}

        # Precompute O(1) relationship routing table for Pass 4
        rel_routing_cache: Dict[str, Optional[str]] = {}
        for rid, rev in reverse_rel_map.items():
            routed = classify_clinical_relationship(
                rid,
                rev,
                include_rf2_metadata_edges=options.include_rf2_metadata_edges,
                include_attribute_edges=options.include_attribute_edges,
            )
            if routed == "REPLACED_BY" and not options.include_replaced_by_edges:
                routed = None
            rel_routing_cache[rid] = routed

        # Pre-resolve CodeSystem nodes for known target vocabularies
        persisted_code_systems: Dict[str, str] = {}
        initial_code_systems: List[CodeSystemNode] = []
        for vid, cs_node in vocab_registry.items():
            if allowed_vocabs is None or vid in allowed_vocabs:
                persisted_code_systems[vid] = cs_node.id
                initial_code_systems.append(cs_node)

        if initial_code_systems:
            if options.dry_run:
                node_counts["CodeSystem"] += len(initial_code_systems)
            else:
                node_counts["CodeSystem"] += self._write_code_systems_sync(
                    initial_code_systems,
                    acted_by=options.acted_by,
                    test_id=norm_test_id,
                )

        # Pass 2: Pre-scan eligible concept_ids & load filtered synonyms
        concept_path = base / "CONCEPT.csv"
        synonyms_by_cid: Dict[int, List[str]] = {}
        if options.include_synonyms:
            selected_cids = _prescan_selected_concept_ids(
                concept_path,
                allowed_vocabs=allowed_vocabs,
                allowed_domains=allowed_domains,
                include_deprecated=include_deprecated,
                max_concepts=options.max_concepts,
            )
            synonyms_by_cid = _load_synonyms_for_selected(
                base,
                selected_cids,
                allowed_lang_ids=allowed_lang_ids,
                max_per_concept=options.max_synonyms_per_concept,
                skipped_counts=skipped_counts,
            )
            del selected_cids

        # Pass 3: Stream CONCEPT.csv -> fused (:Concept) + (:BELONGS_TO_SYSTEM) batches
        concept_id_map: Dict[int, str] = {}
        ingested_vocabs: set[str] = set()
        concept_batch: List[Dict[str, Any]] = []
        embedding_buffer: List[EmbeddingItem] = []
        embedding_candidates_count = 0
        embedding_report: Optional[EmbeddingBatchReport] = None
        synonyms_attached = 0
        concepts_scanned = 0
        concepts_accepted = 0

        concept_required = (
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
        )

        def _flush_concept_batch() -> None:
            nonlocal concept_batch, embedding_buffer, embedding_report
            if not concept_batch:
                return
            if options.dry_run:
                node_counts["Concept"] += len(concept_batch)
                relation_counts["BELONGS_TO_SYSTEM"] += len(concept_batch)
            else:
                c_written, l_written = self._write_concept_batch_sync(concept_batch)
                node_counts["Concept"] += c_written
                relation_counts["BELONGS_TO_SYSTEM"] += l_written
            concept_batch = []

            if (
                options.compute_embeddings
                and not options.dry_run
                and len(embedding_buffer) >= options.embedding_dispatch_chunk_size
            ):
                part = self._dispatch_embedding_chunk(embedding_buffer, options, norm_test_id, embed_callback)
                embedding_report = _merge_embedding_reports(embedding_report, part)
                embedding_buffer = []

        for col, row in _iter_tsv_rows(concept_path, concept_required):
            concepts_scanned += 1
            vid = _safe_cell(row, col["vocabulary_id"])
            if allowed_vocabs is not None and vid not in allowed_vocabs:
                _increment_counter(skipped_counts, "CONCEPT_FILTERED_VOCABULARY")
                continue

            did = _safe_cell(row, col["domain_id"])
            if allowed_domains is not None and did not in allowed_domains:
                _increment_counter(skipped_counts, "CONCEPT_FILTERED_DOMAIN")
                continue

            inv = _safe_cell(row, col["invalid_reason"]) or None
            if inv is not None and not include_deprecated:
                _increment_counter(skipped_counts, "CONCEPT_FILTERED_DEPRECATED")
                continue

            cid = int(_safe_cell(row, col["concept_id"]))
            display = _safe_cell(row, col["concept_name"])
            raw_code = _safe_cell(row, col["concept_code"])
            code = f"OMOP:{cid}" if raw_code == "OMOP generated" else raw_code
            std = _safe_cell(row, col["standard_concept"]) or None
            concept_class_id = _safe_cell(row, col["concept_class_id"])
            valid_start = parse_athena_date(_safe_cell(row, col["valid_start_date"]))
            valid_end = parse_athena_date(_safe_cell(row, col["valid_end_date"]))

            # Ensure CodeSystem exists even if missing from VOCABULARY.csv
            cs_id = persisted_code_systems.get(vid)
            if cs_id is None:
                fallback_cs = CodeSystemNode(
                    vocabulary_id=vid,
                    system_uri=resolve_code_system_uri(vid),
                    name=vid,
                    test_id=norm_test_id,
                    ActedBy=options.acted_by,
                )
                cs_id = fallback_cs.id
                persisted_code_systems[vid] = cs_id
                if options.dry_run:
                    node_counts["CodeSystem"] += 1
                else:
                    node_counts["CodeSystem"] += self._write_code_systems_sync(
                        [fallback_cs],
                        acted_by=options.acted_by,
                        test_id=norm_test_id,
                    )

            system_uri = resolve_code_system_uri(vid)
            node_id = compute_concept_id(system_uri, code, test_id=norm_test_id)
            belongs_rel_id = compute_clinical_relation_id(
                node_id,
                "BELONGS_TO_SYSTEM",
                cs_id,
                test_id=norm_test_id,
            )

            raw_syns = synonyms_by_cid.get(cid)
            syns: List[str] = [s for s in raw_syns if s != display] if raw_syns else []
            synonyms_attached += len(syns)

            concept_props: Dict[str, Any] = {
                "athena_concept_id": cid,
                "system_uri": system_uri,
                "vocabulary_id": vid,
                "code": code,
                "display": display,
                "synonyms": syns,
                "definition": "",
                "domain_id": did,
                "concept_class_id": concept_class_id,
                "standard_concept": std,
                "is_deprecated": inv is not None,
                "invalid_reason": inv,
                "valid_start_date": valid_start,
                "valid_end_date": valid_end,
                "concept_category": concept_category_map.get(cid),
                "coding_key": f"{system_uri}|{code}",
                "coding_macro": f"{system_uri}|{code}|{display}",
            }

            concept_batch.append(
                {
                    "id": node_id,
                    "code_system_id": cs_id,
                    "belongs_rel_id": belongs_rel_id,
                    "props": concept_props,
                    "test_id": norm_test_id,
                    "acted_by": options.acted_by,
                }
            )
            concept_id_map[cid] = node_id
            ingested_vocabs.add(vid)
            concepts_accepted += 1

            # Extract embedding candidates when eligible
            eligible_for_embed = (inv is None and std in ("S", "C")) if options.embed_standard_only else True
            if eligible_for_embed:
                items = build_concept_embedding_items(
                    concept_id=node_id,
                    vocabulary_id=vid,
                    domain_id=did,
                    code=code,
                    display=display,
                    synonyms=syns,
                    contextual=options.use_contextual_embedding_text,
                    embed_synonyms=options.embed_synonyms,
                )
                embedding_candidates_count += len(items)
                if options.compute_embeddings and not options.dry_run:
                    embedding_buffer.extend(items)

            if len(concept_batch) >= options.batch_size:
                _flush_concept_batch()

            if not options.dry_run and concepts_scanned % _PROGRESS_LOG_INTERVAL == 0:
                logger.info(
                    "clinical_concepts_scan_progress",
                    rows_scanned=concepts_scanned,
                    concepts_accepted=concepts_accepted,
                )

            if options.max_concepts is not None and concepts_accepted >= options.max_concepts:
                break

        _flush_concept_batch()

        # Flush any remaining buffered embeddings (all Concept nodes are now persisted)
        if options.compute_embeddings and not options.dry_run and embedding_buffer:
            part = self._dispatch_embedding_chunk(embedding_buffer, options, norm_test_id, embed_callback)
            embedding_report = _merge_embedding_reports(embedding_report, part)
            embedding_buffer = []

        # Free Pass 2/3 auxiliary dicts before streaming 11.6M relationship rows
        synonyms_by_cid.clear()
        concept_category_map.clear()

        # Pass 4: Stream CONCEPT_RELATIONSHIP.csv -> batched canonical ontology edges
        if options.include_relationships and concept_id_map:
            rel_path = base / "CONCEPT_RELATIONSHIP.csv"
            rel_required = (
                "concept_id_1",
                "concept_id_2",
                "relationship_id",
                "valid_start_date",
                "valid_end_date",
                "invalid_reason",
            )
            rel_buffers: Dict[str, List[Dict[str, Any]]] = {rt: [] for rt in _ALLOWED_CONCEPT_REL_TYPES}
            rels_scanned = 0

            def _flush_rel_buffer(rel_type: str) -> None:
                buf = rel_buffers[rel_type]
                if not buf:
                    return
                if options.dry_run:
                    relation_counts[rel_type] += len(buf)
                else:
                    relation_counts[rel_type] += self._write_concept_rel_batch_sync(rel_type, buf)
                rel_buffers[rel_type] = []

            for col, row in _iter_tsv_rows(rel_path, rel_required):
                rels_scanned += 1
                inv = _safe_cell(row, col["invalid_reason"])
                if inv:
                    _increment_counter(skipped_counts, "REL_INVALID")
                    continue

                c1 = int(_safe_cell(row, col["concept_id_1"]))
                c2 = int(_safe_cell(row, col["concept_id_2"]))
                if c1 == c2:
                    _increment_counter(skipped_counts, "REL_SELF_LOOP")
                    continue

                start_id = concept_id_map.get(c1)
                if start_id is None:
                    _increment_counter(skipped_counts, "REL_UNSELECTED_ENDPOINT")
                    continue
                end_id = concept_id_map.get(c2)
                if end_id is None:
                    _increment_counter(skipped_counts, "REL_UNSELECTED_ENDPOINT")
                    continue

                rid = _safe_cell(row, col["relationship_id"])
                if rid in rel_routing_cache:
                    neo_rel_type = rel_routing_cache[rid]
                else:
                    neo_rel_type = classify_clinical_relationship(
                        rid,
                        reverse_rel_map.get(rid, ""),
                        include_rf2_metadata_edges=options.include_rf2_metadata_edges,
                        include_attribute_edges=options.include_attribute_edges,
                    )
                    if neo_rel_type == "REPLACED_BY" and not options.include_replaced_by_edges:
                        neo_rel_type = None
                    rel_routing_cache[rid] = neo_rel_type

                if neo_rel_type is None:
                    _increment_counter(skipped_counts, "REL_MIRROR_OR_FILTERED")
                    continue

                valid_start = parse_athena_date(_safe_cell(row, col["valid_start_date"]))
                valid_end = parse_athena_date(_safe_cell(row, col["valid_end_date"]))

                if neo_rel_type == "IS_A":
                    rel_id = compute_clinical_relation_id(start_id, "IS_A", end_id, test_id=norm_test_id)
                    props: Dict[str, Any] = {
                        "valid_start_date": valid_start,
                        "valid_end_date": valid_end,
                    }
                elif neo_rel_type == "MAPS_TO":
                    rel_id = compute_clinical_relation_id(
                        start_id,
                        "MAPS_TO",
                        end_id,
                        relationship_id=rid,
                        test_id=norm_test_id,
                    )
                    meta = rel_metadata_map.get((c1, c2, rid))
                    props = {
                        "relationship_id": rid,
                        "equivalence": meta.equivalence if meta else None,
                        "confidence": meta.confidence if meta else None,
                        "mapping_source": meta.mapping_source if meta else None,
                        "valid_start_date": valid_start,
                        "valid_end_date": valid_end,
                    }
                elif neo_rel_type == "REPLACED_BY":
                    rel_id = compute_clinical_relation_id(
                        start_id,
                        "REPLACED_BY",
                        end_id,
                        relationship_id=rid,
                        test_id=norm_test_id,
                    )
                    props = {
                        "relationship_id": rid,
                        "valid_start_date": valid_start,
                        "valid_end_date": valid_end,
                    }
                else:  # HAS_ATTRIBUTE
                    rel_id = compute_clinical_relation_id(
                        start_id,
                        "HAS_ATTRIBUTE",
                        end_id,
                        relationship_id=rid,
                        test_id=norm_test_id,
                    )
                    props = {
                        "relationship_id": rid,
                        "relationship_name": rel_name_map.get(rid, rid),
                        "valid_start_date": valid_start,
                        "valid_end_date": valid_end,
                    }

                rel_buffers[neo_rel_type].append(
                    {
                        "id": rel_id,
                        "start_node_id": start_id,
                        "end_node_id": end_id,
                        "props": props,
                        "test_id": norm_test_id,
                        "acted_by": options.acted_by,
                    }
                )
                if len(rel_buffers[neo_rel_type]) >= options.batch_size:
                    _flush_rel_buffer(neo_rel_type)

                if not options.dry_run and rels_scanned % _PROGRESS_LOG_INTERVAL == 0:
                    logger.info(
                        "clinical_relationships_scan_progress",
                        rows_scanned=rels_scanned,
                        relations_written=dict(relation_counts),
                    )

            for rt in sorted(_ALLOWED_CONCEPT_REL_TYPES):
                _flush_rel_buffer(rt)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        log_fn = logger.debug if options.dry_run else logger.info
        log_fn(
            "clinical_terms_ingestion_completed",
            source_dir=str(base),
            profile=options.profile.value,
            dry_run=options.dry_run,
            vocabularies=sorted(ingested_vocabs),
            total_nodes=sum(node_counts.values()),
            total_relations=sum(relation_counts.values()),
            synonyms_attached=synonyms_attached,
            embedding_candidates=embedding_candidates_count,
            embeddings_persisted=embedding_report.persisted_count if embedding_report else 0,
            test_id=norm_test_id,
            duration_ms=duration_ms,
        )
        return ClinicalTermsIngestionReport(
            source_dir=str(base),
            profile=options.profile.value,
            dry_run=options.dry_run,
            test_id=norm_test_id,
            vocabularies_ingested=sorted(ingested_vocabs),
            node_counts=node_counts,
            relation_counts=relation_counts,
            skipped_counts=skipped_counts,
            synonyms_attached=synonyms_attached,
            embedding_candidates_count=embedding_candidates_count,
            embedding_report=embedding_report,
            duration_ms=duration_ms,
        )

    async def ingest(self, options: ClinicalTermsIngestionOptions) -> ClinicalTermsIngestionReport:
        """Async wrapper executing ``ingest_sync`` off the event loop."""
        return await asyncio.to_thread(self.ingest_sync, options)

    def cleanup_test_data_sync(self, test_id: str) -> Dict[str, Any]:
        """Safely purge ONLY nodes carrying the given non-empty ``test_id``."""
        return self.system_service.cleanup_test_data_sync(test_id)

    async def cleanup_test_data(self, test_id: str) -> Dict[str, Any]:
        """Async safe purge of ONLY nodes carrying the given non-empty ``test_id``."""
        return await self.system_service.cleanup_test_data(test_id)


clinical_terms_ingester_service = ClinicalTermsIngesterService()

__all__ = [
    "ClinicalTermsIngestionOptions",
    "ClinicalTermsIngestionReport",
    "build_concept_embedding_items",
    "ClinicalTermsIngesterService",
    "clinical_terms_ingester_service",
]
