"""Provider-agnostic vector embedding computation and Neo4j persistence service."""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Dict, List, LiteralString, Optional, Tuple, cast

from genkit import Genkit
from genkit_google_genai import EmbeddingConfig, GoogleAI, VertexAI
from genkit_ollama import Ollama
from pydantic import BaseModel, Field

from gosmart_dictionary_service.core.config import Settings, get_settings
from gosmart_dictionary_service.core.logging import get_logger
from gosmart_dictionary_service.db.neo4j import Neo4jClient, neo4j_client
from gosmart_dictionary_service.model.actor_message import ALLOWED_EMBEDDING_TARGET_LABELS, EmbeddingItem
from gosmart_dictionary_service.model.embedding import Embedding, EmbedsRelation
from gosmart_dictionary_service.model.llm_config import DEFAULT_EMBEDDING_CONFIG_TAG, LLM_Config
from gosmart_dictionary_service.service.llm_config_service import DEFAULT_LLM_GROUP_NAME, LLMConfigService
from gosmart_dictionary_service.service.system_service import normalize_test_id, resolve_root_group_code

logger = get_logger("gosmart_ds.embedding")

EmbedVectorsFn = Callable[[Sequence[str], LLM_Config], Sequence[Sequence[float]]]

_FIND_EXISTING_EMBEDDINGS_CYPHER: LiteralString = """
UNWIND $ids AS emb_id
MATCH (e:Embedding {id: emb_id})
WHERE e.Action IS NULL OR e.Action <> 'DELETE'
RETURN e.id AS id, e.text AS text
"""


def _build_upsert_embeddings_cypher(target_label: str) -> LiteralString:
    if target_label not in ALLOWED_EMBEDDING_TARGET_LABELS:
        raise ValueError(f"Unsupported embedding target label: '{target_label}'")
    return cast(
        LiteralString,
        f"""
UNWIND $rows AS row
MATCH (t:{target_label} {{id: row.target_node_id}})
MERGE (e:Embedding {{id: row.embedding_id}})
ON CREATE SET e.modelTag = row.modelTag,
              e.size = row.size,
              e.vector = row.vector,
              e.text = row.text,
              e.targetFieldTag = row.targetFieldTag,
              e.test_id = row.test_id,
              e.Action = 'CREATE',
              e.ActedAt = datetime({{timezone: 'UTC'}}),
              e.ActedBy = row.acted_by,
              e.CreatedAt = datetime({{timezone: 'UTC'}})
ON MATCH SET  e.modelTag = row.modelTag,
              e.size = row.size,
              e.vector = row.vector,
              e.text = row.text,
              e.targetFieldTag = row.targetFieldTag,
              e.test_id = coalesce(row.test_id, e.test_id),
              e.Action = 'UPDATE',
              e.ActedAt = datetime({{timezone: 'UTC'}}),
              e.ActedBy = row.acted_by,
              e.UpdatedAt = datetime({{timezone: 'UTC'}})
MERGE (e)-[r:EMBEDS {{id: row.rel_id}}]->(t)
ON CREATE SET r.aspect = row.aspect,
              r.test_id = row.test_id,
              r.Action = 'CREATE',
              r.ActedAt = datetime({{timezone: 'UTC'}}),
              r.ActedBy = row.acted_by,
              r.CreatedAt = datetime({{timezone: 'UTC'}})
ON MATCH SET  r.aspect = row.aspect,
              r.test_id = coalesce(row.test_id, r.test_id),
              r.Action = 'UPDATE',
              r.ActedAt = datetime({{timezone: 'UTC'}}),
              r.ActedBy = row.acted_by,
              r.UpdatedAt = datetime({{timezone: 'UTC'}})
RETURN count(e) AS written
""",
    )


class EmbeddingBatchReport(BaseModel):
    """Structured summary returned after an embedding batch run."""

    total_items: int = Field(default=0, description="Total deduplicated embedding items received")
    embedded_count: int = Field(default=0, description="Number of items newly embedded via provider")
    skipped_existing_count: int = Field(
        default=0,
        description="Number of items skipped because identical text was already embedded",
    )
    persisted_count: int = Field(default=0, description="Number of (:Embedding)-[:EMBEDS]->(target) edges written")
    model_tag: str = Field(default=DEFAULT_EMBEDDING_CONFIG_TAG, description="Resolved LLM_Config tag")
    provider: str = Field(default="", description="Resolved embedding provider (e.g. 'ollama', 'vertexai')")
    model: str = Field(default="", description="Resolved embedding model name")
    dimensions: Optional[int] = Field(default=None, description="Output vector dimensionality")
    test_id: Optional[str] = Field(default=None, description="Optional test isolation identifier")
    duration_ms: float = Field(default=0.0, description="Elapsed processing duration in milliseconds")


def normalize_and_truncate_vector(
    vector: Sequence[float],
    *,
    dimensions_truncated: Optional[int] = None,
    expected_dimensions: Optional[int] = None,
) -> List[float]:
    """Validate finiteness, optionally apply Matryoshka truncation, and L2-normalize a vector."""
    values: List[float] = []
    for idx, elem in enumerate(vector):
        if isinstance(elem, bool):
            raise ValueError(f"Embedding vector element at index {idx} cannot be bool.")
        fval = float(elem)
        if not math.isfinite(fval):
            raise ValueError(f"Embedding vector element at index {idx} is not finite: {elem!r}")
        values.append(fval)

    if not values:
        raise ValueError("Embedding vector must not be empty.")

    if dimensions_truncated is not None:
        if dimensions_truncated > len(values):
            raise ValueError(
                f"dimensions_truncated ({dimensions_truncated}) exceeds returned vector length ({len(values)})."
            )
        values = values[:dimensions_truncated]

    if expected_dimensions is not None and len(values) != expected_dimensions:
        raise ValueError(f"Embedding vector length ({len(values)}) does not match expected ({expected_dimensions}).")

    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("Cannot L2-normalize a zero-magnitude embedding vector.")

    return [v / norm for v in values]


def _chunked(items: Sequence[Any], size: int) -> List[Sequence[Any]]:
    if size <= 0:
        raise ValueError("Chunk size must be a positive integer.")
    return [items[i : i + size] for i in range(0, len(items), size)]


class EmbeddingService:
    """Computes L2-normalized embeddings using dynamic Neo4j ``LLM_Config`` records and persists them."""

    def __init__(
        self,
        *,
        client: Optional[Neo4jClient] = None,
        llm_config_service: Optional[LLMConfigService] = None,
        settings: Optional[Settings] = None,
        embed_fn: Optional[EmbedVectorsFn] = None,
    ) -> None:
        self.client = client or neo4j_client
        self.llm_config_service = llm_config_service or LLMConfigService(client=self.client)
        self.settings = settings or get_settings()
        self._embed_fn = embed_fn
        self._client_lock = threading.Lock()
        self._genkit_clients: Dict[Tuple[Any, ...], Genkit] = {}

    def _get_or_create_genkit_client(self, config: LLM_Config) -> Tuple[Genkit, Any]:
        """Build or reuse a Genkit client + embedder reference for the given ``LLM_Config``."""
        provider = config.provider.strip().lower()
        resolved_key = config.resolve_api_key()
        project_id = config.project_id or self.settings.VERTEX_AI_PROJECT_ID
        location = config.location or self.settings.VERTEX_AI_LOCATION
        fingerprint = config.client_fingerprint(resolved_api_key=resolved_key, fallback_project_id=project_id)

        with self._client_lock:
            ai = self._genkit_clients.get(fingerprint)
            if ai is None:
                if provider == "ollama":
                    plugin: Any = Ollama(base_url=config.url or None)
                elif provider in {"googleai", "gemini"}:
                    plugin = GoogleAI(api_key=resolved_key or None)
                elif provider == "vertexai":
                    plugin = VertexAI(
                        api_key=resolved_key or None,
                        project=project_id,
                        location=location,
                    )
                else:
                    raise ValueError(
                        f"Unsupported embedding provider '{provider}'. Supported providers: ollama, googleai, vertexai."
                    )
                ai = Genkit(plugins=[plugin])
                self._genkit_clients[fingerprint] = ai

        output_dim = config.dimensions_truncated or config.dimensions
        if provider == "ollama":
            embedder_ref: Any = config.model if config.model.startswith("ollama/") else f"ollama/{config.model}"
        elif provider in {"googleai", "gemini"}:
            clean_model = config.model.removeprefix("googleai/")
            embedder_ref = GoogleAI.embedder(
                clean_model,
                config=EmbeddingConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=output_dim),
            )
        elif provider == "vertexai":
            clean_model = config.model.removeprefix("vertexai/")
            embedder_ref = VertexAI.embedder(
                clean_model,
                config=EmbeddingConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=output_dim),
            )
        else:
            raise ValueError(f"Unsupported embedding provider '{provider}'.")

        return ai, embedder_ref

    async def compute_vectors(
        self,
        texts: Sequence[str],
        config: LLM_Config,
    ) -> List[List[float]]:
        """Compute L2-normalized vectors for a batch of texts asynchronously."""
        if not texts:
            return []

        if self._embed_fn is not None:
            raw_vectors = list(self._embed_fn(texts, config))
        else:
            ai, embedder_ref = self._get_or_create_genkit_client(config)
            embeddings = await ai.embed_many(embedder=embedder_ref, content=list(texts))
            raw_vectors = [emb.embedding for emb in embeddings]

        if len(raw_vectors) != len(texts):
            raise RuntimeError(
                f"Embedding provider '{config.provider}/{config.model}' returned {len(raw_vectors)} vectors "
                f"for {len(texts)} input texts."
            )

        expected_dim = config.dimensions_truncated or config.dimensions
        return [
            normalize_and_truncate_vector(
                vec,
                dimensions_truncated=config.dimensions_truncated,
                expected_dimensions=expected_dim,
            )
            for vec in raw_vectors
        ]

    def compute_vectors_sync(
        self,
        texts: Sequence[str],
        config: LLM_Config,
    ) -> List[List[float]]:
        """Compute L2-normalized vectors synchronously (safe for Pykka worker threads)."""
        if not texts:
            return []
        if self._embed_fn is not None:
            raw_vectors = list(self._embed_fn(texts, config))
            if len(raw_vectors) != len(texts):
                raise RuntimeError(
                    f"Embedding provider '{config.provider}/{config.model}' returned {len(raw_vectors)} vectors "
                    f"for {len(texts)} input texts."
                )
            expected_dim = config.dimensions_truncated or config.dimensions
            return [
                normalize_and_truncate_vector(
                    vec,
                    dimensions_truncated=config.dimensions_truncated,
                    expected_dimensions=expected_dim,
                )
                for vec in raw_vectors
            ]
        return asyncio.run(self.compute_vectors(texts, config))

    def resolve_embedding_config_sync(
        self,
        *,
        tag: str = DEFAULT_EMBEDDING_CONFIG_TAG,
        group_id: Optional[str] = None,
        group_code: Optional[str] = None,
        group_name: Optional[str] = DEFAULT_LLM_GROUP_NAME,
        test_id: Optional[str] = None,
        config_override: Optional[LLM_Config] = None,
    ) -> LLM_Config:
        """Resolve the active embedding ``LLM_Config`` synchronously from Neo4j."""
        if config_override is not None:
            return config_override

        norm_test_id = normalize_test_id(test_id)
        effective_group_code = group_code
        if not group_id and not effective_group_code and norm_test_id:
            effective_group_code = resolve_root_group_code(norm_test_id)

        cfg = self.llm_config_service.get_llm_config_by_tag_sync(
            tag,
            group_id=group_id,
            group_code=effective_group_code,
            group_name=group_name,
            test_id=norm_test_id,
        )
        if cfg is None:
            raise LookupError(
                f"Active LLM_Config with tag='{tag}' not found "
                f"(group_id={group_id!r}, group_code={effective_group_code!r}, "
                f"group_name={group_name!r}, test_id={norm_test_id!r}). "
                "Run system bootstrap first."
            )
        return cfg

    def find_existing_embedding_texts_sync(
        self,
        embedding_ids: Sequence[str],
        *,
        batch_size: int = 1000,
    ) -> Dict[str, str]:
        """Fetch ``{embedding_id: text}`` for existing active ``:Embedding`` nodes in Neo4j."""
        if not embedding_ids:
            return {}
        found: Dict[str, str] = {}
        driver = self.client.get_sync_driver()
        with driver.session(database=self.client.settings.DB_DATABASE) as session:
            for chunk in _chunked(list(embedding_ids), batch_size):
                result = session.run(_FIND_EXISTING_EMBEDDINGS_CYPHER, {"ids": list(chunk)})
                for record in result:
                    emb_id = record.get("id")
                    emb_text = record.get("text")
                    if isinstance(emb_id, str) and isinstance(emb_text, str):
                        found[emb_id] = emb_text
        return found

    def persist_embeddings_sync(
        self,
        rows_by_label: Dict[str, List[Dict[str, Any]]],
        *,
        batch_size: int = 500,
    ) -> int:
        """Persist ``(:Embedding)-[:EMBEDS]->(target)`` batches in Neo4j via ``UNWIND MERGE``."""
        total_written = 0
        for label, rows in rows_by_label.items():
            if not rows:
                continue
            query = _build_upsert_embeddings_cypher(label)
            for chunk in _chunked(rows, batch_size):

                def _tx(tx: Any, q: LiteralString = query, batch: Sequence[Dict[str, Any]] = chunk) -> int:
                    rec = tx.run(q, {"rows": list(batch)}).single()
                    return int(rec["written"]) if rec is not None and rec.get("written") else 0

                total_written += int(self.client.execute_write_tx_sync(_tx))
        return total_written

    def embed_items_sync(
        self,
        items: Sequence[EmbeddingItem],
        *,
        tag: str = DEFAULT_EMBEDDING_CONFIG_TAG,
        group_id: Optional[str] = None,
        group_code: Optional[str] = None,
        group_name: Optional[str] = DEFAULT_LLM_GROUP_NAME,
        batch_size: int = 64,
        persist_batch_size: int = 500,
        skip_existing: bool = True,
        test_id: Optional[str] = None,
        acted_by: str = "embedding_service",
        config_override: Optional[LLM_Config] = None,
    ) -> EmbeddingBatchReport:
        """Resolve config, compute L2-normalized vectors in batches, and persist ``:Embedding`` nodes."""
        start_time = time.perf_counter()
        norm_test_id = normalize_test_id(test_id)
        config = self.resolve_embedding_config_sync(
            tag=tag,
            group_id=group_id,
            group_code=group_code,
            group_name=group_name,
            test_id=norm_test_id,
            config_override=config_override,
        )
        model_tag = config.tag

        # Deduplicate items preserving first occurrence per (target_node_id, target_field_tag)
        deduped: List[EmbeddingItem] = []
        seen_keys: set[Tuple[str, str]] = set()
        for item in items:
            key = (item.target_node_id, item.target_field_tag)
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(item)

        if not deduped:
            return EmbeddingBatchReport(
                total_items=0,
                embedded_count=0,
                skipped_existing_count=0,
                persisted_count=0,
                model_tag=model_tag,
                provider=config.provider,
                model=config.model,
                dimensions=config.dimensions_truncated or config.dimensions,
                test_id=norm_test_id,
                duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )

        id_map: Dict[Tuple[str, str], str] = {
            (item.target_node_id, item.target_field_tag): Embedding.compute_target_scoped_id(
                item.target_node_id,
                model_tag,
                item.target_field_tag,
                test_id=norm_test_id,
            )
            for item in deduped
        }

        existing_texts: Dict[str, str] = {}
        if skip_existing:
            existing_texts = self.find_existing_embedding_texts_sync(list(id_map.values()))

        pending: List[EmbeddingItem] = []
        skipped_count = 0
        for item in deduped:
            emb_id = id_map[(item.target_node_id, item.target_field_tag)]
            if skip_existing and existing_texts.get(emb_id) == item.text:
                skipped_count += 1
            else:
                pending.append(item)

        embedded_count = 0
        persisted_count = 0
        actual_dimensions: Optional[int] = config.dimensions_truncated or config.dimensions

        for batch in _chunked(pending, batch_size):
            batch_items = list(batch)
            vectors = self.compute_vectors_sync([it.text for it in batch_items], config)
            rows_by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

            for item, vec in zip(batch_items, vectors, strict=True):
                emb_node = Embedding(
                    modelTag=model_tag,
                    vector=vec,
                    text=item.text,
                    targetFieldTag=item.target_field_tag,
                    target_node_id=item.target_node_id,
                    test_id=norm_test_id,
                    ActedBy=acted_by,
                )
                rel = EmbedsRelation(
                    start_node_id=emb_node.id,
                    end_node_id=item.target_node_id,
                    aspect=item.target_field_tag,
                    test_id=norm_test_id,
                    ActedBy=acted_by,
                )
                actual_dimensions = emb_node.size
                rows_by_label[item.target_label].append(
                    {
                        "target_node_id": item.target_node_id,
                        "embedding_id": emb_node.id,
                        "rel_id": rel.id,
                        "modelTag": emb_node.modelTag,
                        "size": emb_node.size,
                        "vector": emb_node.vector,
                        "text": emb_node.text,
                        "targetFieldTag": emb_node.targetFieldTag,
                        "aspect": rel.aspect,
                        "test_id": norm_test_id,
                        "acted_by": acted_by,
                    }
                )
                embedded_count += 1

            persisted_count += self.persist_embeddings_sync(rows_by_label, batch_size=persist_batch_size)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "embedding_batch_completed",
            total_items=len(deduped),
            embedded_count=embedded_count,
            skipped_existing_count=skipped_count,
            persisted_count=persisted_count,
            model_tag=model_tag,
            provider=config.provider,
            model=config.model,
            dimensions=actual_dimensions,
            test_id=norm_test_id,
            duration_ms=duration_ms,
        )
        return EmbeddingBatchReport(
            total_items=len(deduped),
            embedded_count=embedded_count,
            skipped_existing_count=skipped_count,
            persisted_count=persisted_count,
            model_tag=model_tag,
            provider=config.provider,
            model=config.model,
            dimensions=actual_dimensions,
            test_id=norm_test_id,
            duration_ms=duration_ms,
        )

    async def embed_items(
        self,
        items: Sequence[EmbeddingItem],
        **kwargs: Any,
    ) -> EmbeddingBatchReport:
        """Async wrapper executing ``embed_items_sync`` off the event loop."""
        return await asyncio.to_thread(self.embed_items_sync, items, **kwargs)


embedding_service = EmbeddingService()

__all__ = [
    "EmbedVectorsFn",
    "EmbeddingBatchReport",
    "normalize_and_truncate_vector",
    "EmbeddingService",
    "embedding_service",
]
