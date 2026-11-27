"""Pydantic message models for GoSmart DS Pykka actor communication."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from gosmart_dictionary_service.model.clinical.constants import ClinicalIngestionProfile
from gosmart_dictionary_service.model.embedding import DictionaryEmbeddingFieldTag
from gosmart_dictionary_service.model.llm_config import DEFAULT_EMBEDDING_CONFIG_TAG

ALLOWED_EMBEDDING_TARGET_LABELS: frozenset[str] = frozenset(
    {
        "DicomElement",
        "DicomCIOD",
        "DicomModule",
        "DicomMacro",
        "DicomSOPClass",
        "DicomConfidentialityProfile",
        "DicomStandardReference",
        "CodeSystem",
        "Concept",
    }
)


class EmbeddingItem(BaseModel):
    """Single embeddable text aspect targeting a specific graph node."""

    target_node_id: str = Field(..., min_length=1, description="Deterministic ID of the target Neo4j node")
    target_label: str = Field(..., min_length=1, description="Allowlisted Neo4j label of the target node")
    target_field_tag: str = Field(
        default=DictionaryEmbeddingFieldTag.NAME.value,
        description="Aspect/field tag represented by this embedding (e.g. 'NAME', 'DESCRIPTION', 'DISPLAY', 'SYNONYM')",
    )
    text: str = Field(..., min_length=1, description="Source text to embed")

    @field_validator("target_node_id", "text", mode="before")
    @classmethod
    def _validate_non_empty_str(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Field must be a non-empty string.")
        return v.strip()

    @field_validator("target_label", mode="before")
    @classmethod
    def _validate_target_label(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("target_label must be a non-empty string.")
        cleaned = v.strip()
        if cleaned not in ALLOWED_EMBEDDING_TARGET_LABELS:
            raise ValueError(
                f"Unsupported embedding target_label '{cleaned}'. "
                f"Allowed labels: {sorted(ALLOWED_EMBEDDING_TARGET_LABELS)}"
            )
        return cleaned

    @field_validator("target_field_tag", mode="before")
    @classmethod
    def _normalize_field_tag(cls, v: Any) -> str:
        if isinstance(v, DictionaryEmbeddingFieldTag):
            return v.value
        if not isinstance(v, str) or not v.strip():
            raise ValueError("target_field_tag must be a non-empty string.")
        return v.strip()


class EmbeddingBatchMessage(BaseModel):
    """Validated message payload for ``EmbeddingActor`` batch embedding requests."""

    type: Literal["EMBED_BATCH"] = Field(default="EMBED_BATCH", description="Actor message discriminator")
    items: list[EmbeddingItem] = Field(default_factory=list, description="Embeddable target node items")
    tag: str = Field(
        default=DEFAULT_EMBEDDING_CONFIG_TAG,
        description="LLM_Config purpose tag (default: 'embedding-v1')",
    )
    group_id: Optional[str] = Field(default=None, description="Optional sponsoring Group deterministic ID")
    group_code: Optional[str] = Field(default=None, description="Optional sponsoring Group code")
    group_name: Optional[str] = Field(default="GoSmart Health", description="Default sponsoring Group name")
    batch_size: int = Field(default=64, gt=0, description="Number of texts embedded per provider batch request")
    skip_existing: bool = Field(
        default=True,
        description="Skip re-embedding items whose deterministic Embedding already stores identical text",
    )
    test_id: Optional[str] = Field(default=None, description="Optional test isolation identifier")
    acted_by: str = Field(default="embedding_actor", description="Actor or user identity for audit fields")

    @field_validator("tag", mode="before")
    @classmethod
    def _normalize_tag(cls, v: Any) -> str:
        if v is None or (isinstance(v, str) and not v.strip()):
            return DEFAULT_EMBEDDING_CONFIG_TAG
        if isinstance(v, str):
            return v.strip()
        raise ValueError("tag must be a string.")

    @field_validator("group_id", "group_code", "group_name", "test_id", mode="before")
    @classmethod
    def _normalize_optional_str(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            return stripped if stripped else None
        return str(v).strip() or None

    @field_validator("acted_by", mode="before")
    @classmethod
    def _normalize_acted_by(cls, v: Any) -> str:
        if v is None or (isinstance(v, str) and not v.strip()):
            return "embedding_actor"
        return str(v).strip()


class DicomIngestMessage(BaseModel):
    """Validated message payload for ``DicomIngestionActor`` ingestion requests."""

    type: Literal["INGEST_DICOM"] = Field(default="INGEST_DICOM", description="Actor message discriminator")
    source_dir: str = Field(
        default="./data-source/dicom",
        min_length=1,
        description="Directory containing Innolitics DICOM standard JSON files",
    )
    batch_size: int = Field(default=1000, gt=0, description="Neo4j UNWIND MERGE batch size")
    acted_by: str = Field(default="dicom_ingestor", description="Audit actor identifier")
    test_id: Optional[str] = Field(default=None, description="Optional test isolation identifier")
    ensure_schema: bool = Field(default=True, description="Apply Neo4j DDL schema before ingesting")
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
        description="LLM_Config tag used for embedding computation",
    )
    embedding_group_code: Optional[str] = Field(
        default=None,
        description="Optional Group code override for resolving embedding LLM_Config",
    )
    embedding_group_name: Optional[str] = Field(
        default="GoSmart Health",
        description="Sponsoring Group name for resolving embedding LLM_Config",
    )
    embedding_batch_size: int = Field(default=64, gt=0, description="Texts per embedding provider batch")
    skip_existing_embeddings: bool = Field(
        default=True,
        description="Skip re-embedding nodes whose text has not changed",
    )
    use_contextual_embedding_text: bool = Field(
        default=True,
        description="Prefix embedding text with lightweight DICOM entity context (Option A)",
    )
    dry_run: bool = Field(
        default=False,
        description="Validate seed files and return counts without writing to Neo4j or calling LLMs",
    )

    @field_validator("source_dir", mode="before")
    @classmethod
    def _validate_source_dir(cls, v: Any) -> str:
        if isinstance(v, Path):
            v = str(v)
        if not isinstance(v, str) or not v.strip():
            raise ValueError("source_dir must be a non-empty path string.")
        return v.strip()

    @field_validator("test_id", "embedding_group_code", "embedding_group_name", mode="before")
    @classmethod
    def _normalize_optional_str(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            return stripped if stripped else None
        return str(v).strip() or None

    @field_validator("acted_by", mode="before")
    @classmethod
    def _normalize_acted_by(cls, v: Any) -> str:
        if v is None or (isinstance(v, str) and not v.strip()):
            return "dicom_ingestor"
        return str(v).strip()


class DicomCleanupMessage(BaseModel):
    """Validated message payload for ``DicomIngestionActor`` test-scoped cleanup."""

    type: Literal["CLEANUP_TEST_DATA"] = Field(
        default="CLEANUP_TEST_DATA",
        description="Actor message discriminator",
    )
    test_id: str = Field(..., min_length=1, description="Mandatory non-empty test identifier to purge")

    @field_validator("test_id", mode="before")
    @classmethod
    def _validate_test_id(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("test_id must be a non-empty string.")
        return v.strip()


class ClinicalTermsIngestMessage(BaseModel):
    """Validated message payload for ``ClinicalTermsIngestionActor`` ingestion requests."""

    type: Literal["INGEST_CLINICAL_TERMS"] = Field(
        default="INGEST_CLINICAL_TERMS",
        description="Actor message discriminator",
    )
    source_dir: str = Field(
        default="./data-source/clinical-terms",
        min_length=1,
        description="Directory containing OHDSI Athena tab-delimited CSV files",
    )
    profile: ClinicalIngestionProfile = Field(
        default=ClinicalIngestionProfile.FULL,
        description="Preset ingestion scope ('full', 'valid', or 'demo-core')",
    )
    vocabularies: Optional[List[str]] = Field(
        default=None,
        description="Optional explicit vocabulary_id allowlist override",
    )
    all_vocabularies: bool = Field(
        default=False,
        description="Ingest all vocabularies present in the Athena export (overrides default clinical allowlist)",
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
        description="Optional language_concept_id filter for synonyms (default: English 4180186)",
    )
    all_synonym_languages: bool = Field(
        default=False,
        description="Include synonyms across all languages",
    )
    max_synonyms_per_concept: int = Field(
        default=25,
        gt=0,
        description="Maximum deduplicated synonyms stored per Concept node",
    )
    include_relationships: bool = Field(
        default=True,
        description="Ingest canonical ontology relationships from CONCEPT_RELATIONSHIP.csv",
    )
    include_attribute_edges: bool = Field(
        default=True,
        description="Ingest :HAS_ATTRIBUTE clinical defining edges",
    )
    include_rf2_metadata_edges: bool = Field(
        default=False,
        description="Include SNOMED RF2 administrative edges ('Has status', 'Has Module')",
    )
    include_replaced_by_edges: bool = Field(
        default=True,
        description="Include historical replacement edges (:REPLACED_BY)",
    )
    max_concepts: Optional[int] = Field(
        default=None,
        gt=0,
        description="Optional cap on ingested Concept nodes (useful for dev/smoke runs)",
    )
    batch_size: int = Field(default=5000, gt=0, description="Rows per Neo4j UNWIND MERGE write transaction")
    acted_by: str = Field(default="clinical_terms_ingestor", description="Audit actor identifier")
    test_id: Optional[str] = Field(default=None, description="Optional test isolation identifier")
    ensure_schema: bool = Field(default=True, description="Apply Neo4j DDL schema before ingesting")
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
        description="Also generate separate :Embedding nodes for concept synonyms",
    )
    embedding_tag: str = Field(
        default=DEFAULT_EMBEDDING_CONFIG_TAG,
        description="LLM_Config tag used for embedding computation",
    )
    embedding_group_code: Optional[str] = Field(
        default=None,
        description="Optional Group code override for resolving embedding LLM_Config",
    )
    embedding_group_name: Optional[str] = Field(
        default="GoSmart Health",
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
        description="Prefix embedding text with vocabulary/domain/code context",
    )
    dry_run: bool = Field(
        default=False,
        description="Stream and validate CSV files and return counts without writing to Neo4j or calling LLMs",
    )

    @field_validator("source_dir", mode="before")
    @classmethod
    def _validate_source_dir(cls, v: Any) -> str:
        if isinstance(v, Path):
            v = str(v)
        if not isinstance(v, str) or not v.strip():
            raise ValueError("source_dir must be a non-empty path string.")
        return v.strip()

    @field_validator("test_id", "embedding_group_code", "embedding_group_name", mode="before")
    @classmethod
    def _normalize_optional_str(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            return stripped if stripped else None
        return str(v).strip() or None

    @field_validator("acted_by", mode="before")
    @classmethod
    def _normalize_acted_by(cls, v: Any) -> str:
        if v is None or (isinstance(v, str) and not v.strip()):
            return "clinical_terms_ingestor"
        return str(v).strip()


class ClinicalTermsCleanupMessage(BaseModel):
    """Validated message payload for ``ClinicalTermsIngestionActor`` test-scoped cleanup."""

    type: Literal["CLEANUP_TEST_DATA"] = Field(
        default="CLEANUP_TEST_DATA",
        description="Actor message discriminator",
    )
    test_id: str = Field(..., min_length=1, description="Mandatory non-empty test identifier to purge")

    @field_validator("test_id", mode="before")
    @classmethod
    def _validate_test_id(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("test_id must be a non-empty string.")
        return v.strip()


__all__ = [
    "ALLOWED_EMBEDDING_TARGET_LABELS",
    "EmbeddingItem",
    "EmbeddingBatchMessage",
    "DicomIngestMessage",
    "DicomCleanupMessage",
    "ClinicalTermsIngestMessage",
    "ClinicalTermsCleanupMessage",
]
