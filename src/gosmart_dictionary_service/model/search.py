"""Unified search, ontology traversal, and NLU contract models for GoSmart Dictionary Service."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field

from gosmart_dictionary_service.model.llm_config import DEFAULT_EMBEDDING_CONFIG_TAG


class SearchEntityType(StrEnum):
    """Supported searchable graph entity labels in GoSmart Dictionary Service."""

    CONCEPT = "Concept"
    CODE_SYSTEM = "CodeSystem"
    DICOM_ELEMENT = "DicomElement"
    DICOM_CIOD = "DicomCIOD"
    DICOM_MODULE = "DicomModule"
    DICOM_MACRO = "DicomMacro"
    DICOM_SOP_CLASS = "DicomSOPClass"
    DICOM_CONFIDENTIALITY_PROFILE = "DicomConfidentialityProfile"


CLINICAL_ENTITY_TYPES: FrozenSet[SearchEntityType] = frozenset({SearchEntityType.CONCEPT})
DICOM_ENTITY_TYPES: FrozenSet[SearchEntityType] = frozenset(
    {
        SearchEntityType.DICOM_ELEMENT,
        SearchEntityType.DICOM_CIOD,
        SearchEntityType.DICOM_MODULE,
        SearchEntityType.DICOM_MACRO,
        SearchEntityType.DICOM_SOP_CLASS,
        SearchEntityType.DICOM_CONFIDENTIALITY_PROFILE,
    }
)
DEFAULT_UNIFIED_ENTITY_TYPES: FrozenSet[SearchEntityType] = frozenset(
    {
        SearchEntityType.CONCEPT,
        SearchEntityType.DICOM_ELEMENT,
    }
)


class ResponseFormat(StrEnum):
    """Dual-representation wire projection format (REST default ``fhir`` vs MCP default ``compact``)."""

    FHIR = "fhir"
    COMPACT = "compact"


class RelationDirection(StrEnum):
    """Graph traversal direction for related-concept aggregation."""

    FORWARD = "forward"
    REVERSE = "reverse"
    BOTH = "both"


class MatchSource(StrEnum):
    """Provenance tag indicating how a hit or related node was retrieved."""

    EXACT_CODE = "exact_code"
    FULLTEXT = "fulltext"
    VECTOR = "vector"
    HYBRID = "hybrid"
    LLM_EXPANSION = "llm_expansion"
    IS_A_PARENT = "is_a_parent"
    IS_A_CHILD = "is_a_child"
    MAPS_TO = "maps_to"
    MAPPED_FROM = "mapped_from"
    REPLACED_BY = "replaced_by"
    REPLACES = "replaces"
    HAS_ATTRIBUTE = "has_attribute"
    USED_AS_ATTRIBUTE_BY = "used_as_attribute_by"
    SHARED_ATTRIBUTE = "shared_attribute"
    SIBLING = "sibling"
    DICOM_RELATION = "dicom_relation"


class NLUIntent(StrEnum):
    """Parsed high-level query intent produced by the NLU layer."""

    SEARCH = "search"
    EXACT_LOOKUP = "exact_lookup"
    RELATED_LOOKUP = "related_lookup"
    CROSSWALK = "crosswalk"
    EXPAND_SUBTYPES = "expand_subtypes"


SubsumptionOutcome = Literal["equivalent", "subsumes", "subsumed-by", "not-subsumed"]


class ExtractedCodeEntity(BaseModel):
    """Explicit terminology code or DICOM identifier extracted from a user query."""

    vocabulary_id: str = Field(
        ...,
        description="Normalized code system / vocabulary identifier (e.g. 'LOINC', 'SNOMED', 'ICD10CM', 'DICOM_TAG')",
    )
    code: str = Field(..., description="Extracted code value (e.g. '87279-6', '233604007', '00180050')")
    raw_span: str = Field(default="", description="Original matched substring from the user query")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score")
    start_char: Optional[int] = Field(default=None, description="Start character offset of raw_span in cleaned query")
    end_char: Optional[int] = Field(default=None, description="End character offset of raw_span in cleaned query")
    resolved: Optional[bool] = Field(
        default=None,
        description="Whether the extracted code resolved against the active Neo4j dictionary",
    )
    resolved_display: Optional[str] = Field(
        default=None,
        description="Reverse-looked-up human-readable display/name for query rewriting",
    )
    is_schema_loaded: Optional[bool] = Field(
        default=None,
        description="Whether the coding schema is loaded in the active Neo4j database",
    )


class NLUParseResult(BaseModel):
    """Structured parse output produced by an ``NLUParserProtocol`` implementation."""

    raw_query: str = Field(..., description="Original user query string")
    normalized_query: str = Field(..., description="Whitespace-normalized search phrase stripped of intent scaffolding")
    rewritten_query: Optional[str] = Field(
        default=None,
        description="Query string with resolved code spans replaced by their reverse-looked-up display text",
    )
    synthesized_phrases: List[str] = Field(
        default_factory=list,
        description="Alternate clinical/DICOM search phrases synthesized via the inference LLM",
    )
    intent: NLUIntent = Field(default=NLUIntent.SEARCH, description="Detected routing intent")
    nlu_confidence: Optional[float] = Field(
        default=None,
        description="Intent confidence score returned by the trained spaCy NLU pipeline when available",
    )
    extracted_codes: List[ExtractedCodeEntity] = Field(
        default_factory=list,
        description="Explicit vocabulary/code or DICOM tag/UID entities extracted from the query",
    )
    inferred_entity_types: List[SearchEntityType] = Field(
        default_factory=list,
        description="Optional entity-type narrowing inferred from query hints",
    )
    inferred_vocabularies: List[str] = Field(
        default_factory=list,
        description="Optional clinical vocabulary allowlist inferred from query hints",
    )
    inferred_domains: List[str] = Field(
        default_factory=list,
        description="Optional clinical domain hints inferred from the query",
    )
    target_vocabularies: List[str] = Field(
        default_factory=list,
        description="Target vocabularies for CROSSWALK intent (e.g. ['SNOMED'])",
    )
    unsupported_vocabularies: List[str] = Field(
        default_factory=list,
        description="Extracted coding schemas that are not loaded in the active Neo4j database",
    )
    modifier_text: Optional[str] = Field(
        default=None,
        description="Optional free-text qualifier accompanying a RELATED_LOOKUP (e.g. 'with contrast')",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal NLU warnings (unloaded schemas, unknown codes, or LLM fallback notes)",
    )
    provider: str = Field(default="stub", description="NLU implementation identifier (e.g. 'stub', 'gosmart-nlu')")


class FHIRCoding(BaseModel):
    """Canonical FHIR-aligned Coding tuple."""

    system: str = Field(..., description="Canonical FHIR CodeSystem URI")
    code: str = Field(..., description="Canonical code within the system")
    display: str = Field(..., description="Human-readable display name")
    version: Optional[str] = Field(default=None, description="Optional terminology release version")


class RelatedConceptHit(BaseModel):
    """Related clinical concept or DICOM entity aggregated via forward/reverse/2-hop traversal."""

    id: str = Field(..., description="Deterministic UUIDv5 id of the related node")
    entity_type: SearchEntityType = Field(default=SearchEntityType.CONCEPT, description="Graph label of related node")
    direction: RelationDirection = Field(..., description="Traversal direction relative to seed node")
    relation_reason: MatchSource = Field(..., description="Semantic category of the relationship")
    relationship_id: str = Field(default="", description="Underlying Neo4j/Athena relationship predicate")
    shared_axes: List[str] = Field(
        default_factory=list,
        description="Shared defining axis displays when retrieved via 2-hop shared-attribute overlap",
    )
    score: float = Field(default=1.0, description="Relative relevance score for ordering related hits")
    coding: FHIRCoding = Field(..., description="Canonical FHIR Coding tuple")
    macro: str = Field(..., description="Precomputed gosmart-graphite Coding Macro (<system>|<code>|<display>)")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Domain-specific metadata properties")

    def to_projection(self, fmt: ResponseFormat = ResponseFormat.FHIR) -> Dict[str, Any]:
        """Project into FHIR envelope (REST default) or token-efficient compact dict (MCP default)."""
        if fmt == ResponseFormat.COMPACT:
            payload: Dict[str, Any] = {
                "relation": self.relation_reason.value,
                "direction": self.direction.value,
                "predicate": self.relationship_id,
                "macro": self.macro,
            }
            vocab = self.properties.get("vocabulary")
            if vocab:
                payload["vocab"] = vocab
            domain = self.properties.get("domain")
            if domain:
                payload["domain"] = domain
            if self.shared_axes:
                payload["shared_axes"] = self.shared_axes
            return payload

        return {
            "direction": self.direction.value,
            "relation_reason": self.relation_reason.value,
            "relationship_id": self.relationship_id,
            "shared_axes": self.shared_axes,
            "score": round(self.score, 4),
            "coding": self.coding.model_dump(exclude_none=True),
            "macro": self.macro,
            "properties": self.properties,
        }


class SearchHit(BaseModel):
    """Unified ranked search result across Clinical Terminology and DICOM Dictionary entities."""

    id: str = Field(..., description="Deterministic UUIDv5 id of the matched node")
    entity_type: SearchEntityType = Field(..., description="Neo4j label of the matched entity")
    score: float = Field(..., description="Final combined ranking score")
    lexical_score: Optional[float] = Field(default=None, description="Raw or normalized Lucene BM25 score")
    vector_score: Optional[float] = Field(default=None, description="Cosine similarity score in [-1.0, 1.0]")
    match_sources: List[MatchSource] = Field(
        default_factory=list,
        description="One or more provenance sources that surfaced this hit",
    )
    coding: FHIRCoding = Field(..., description="Canonical FHIR Coding tuple")
    macro: str = Field(..., description="Precomputed gosmart-graphite Coding Macro (<system>|<code>|<display>)")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Flat entity metadata properties")
    related: List[RelatedConceptHit] = Field(
        default_factory=list,
        description="Optional related concepts/entities attached via forward, reverse, or 2-hop traversal",
    )

    def to_projection(self, fmt: ResponseFormat = ResponseFormat.FHIR) -> Dict[str, Any]:
        """Project hit into REST ``fhir`` envelope or MCP token-efficient ``compact`` dict."""
        if fmt == ResponseFormat.COMPACT:
            if self.entity_type == SearchEntityType.DICOM_ELEMENT:
                compact: Dict[str, Any] = {
                    "entity_type": self.entity_type.value,
                    "tag": self.properties.get("tag_hex") or self.coding.code,
                    "keyword": self.properties.get("keyword"),
                    "display": self.coding.display,
                    "vr": self.properties.get("vr"),
                    "vm": self.properties.get("vm"),
                    "retired": self.properties.get("retired", False),
                    "score": round(self.score, 4),
                }
            elif self.entity_type == SearchEntityType.CONCEPT:
                compact = {
                    "entity_type": self.entity_type.value,
                    "macro": self.macro,
                    "vocab": self.properties.get("vocabulary"),
                    "domain": self.properties.get("domain"),
                    "standard": self.properties.get("standard"),
                    "score": round(self.score, 4),
                }
            else:
                compact = {
                    "entity_type": self.entity_type.value,
                    "code": self.coding.code,
                    "display": self.coding.display,
                    "macro": self.macro,
                    "score": round(self.score, 4),
                }
            if self.related:
                compact["related"] = [rel.to_projection(ResponseFormat.COMPACT) for rel in self.related]
            return {k: v for k, v in compact.items() if v is not None}

        payload: Dict[str, Any] = {
            "id": self.id,
            "entity_type": self.entity_type.value,
            "score": round(self.score, 4),
            "match_sources": [src.value for src in self.match_sources],
            "coding": self.coding.model_dump(exclude_none=True),
            "macro": self.macro,
            "properties": self.properties,
        }
        if self.lexical_score is not None:
            payload["lexical_score"] = round(self.lexical_score, 4)
        if self.vector_score is not None:
            payload["vector_score"] = round(self.vector_score, 4)
        if self.related:
            payload["related"] = [rel.to_projection(ResponseFormat.FHIR) for rel in self.related]
        return payload


class SearchOptions(BaseModel):
    """Structured search configuration options accepted by ``SearchService.search``."""

    entity_types: Optional[List[SearchEntityType]] = Field(
        default=None,
        description="Optional entity-type allowlist; None auto-routes via NLU or defaults to Concept + DicomElement",
    )
    vocabularies: Optional[List[str]] = Field(
        default=None,
        description="Optional clinical vocabulary allowlist (e.g. ['SNOMED', 'LOINC', 'ICD10CM', 'UCUM'])",
    )
    domains: Optional[List[str]] = Field(
        default=None,
        description="Optional clinical domain allowlist (e.g. ['Condition', 'Procedure', 'Measurement'])",
    )
    valid_only: bool = Field(default=True, description="Exclude deprecated/retired concepts when True")
    standard_only: bool = Field(default=False, description="Restrict clinical concepts to standard_concept='S'")
    include_related: bool = Field(
        default=False,
        description="Attach forward/reverse/2-hop related nodes to returned hits",
    )
    related_direction: RelationDirection = Field(
        default=RelationDirection.BOTH,
        description="Direction for related-concept aggregation when include_related or RELATED_LOOKUP is active",
    )
    related_limit: int = Field(default=12, ge=1, le=100, description="Maximum related items per seed hit")
    include_shared_attributes: bool = Field(
        default=True,
        description="Include 2-hop shared-attribute sibling discovery during related-concept aggregation",
    )
    rerank_with_embeddings: bool = Field(
        default=True,
        description="Use on-the-fly embedding-v1 reranking for clinical hits and HNSW vector index for DICOM hits",
    )
    expand_with_llm: Optional[bool] = Field(
        default=None,
        description="Synthesize alternate search phrases via the inference LLM (defaults to NLU_LLM_EXPANSION_ENABLED)",
    )
    embedding_tag: str = Field(
        default=DEFAULT_EMBEDDING_CONFIG_TAG,
        description="Active LLM_Config tag for vector embeddings",
    )
    limit: int = Field(default=10, ge=1, le=200, description="Maximum top-level hits to return")
    response_format: ResponseFormat = Field(
        default=ResponseFormat.FHIR,
        description="Default wire projection format for to_wire()",
    )
    test_id: Optional[str] = Field(
        default=None,
        description="Optional test_id scope (None queries production test_id IS NULL records)",
    )


class SearchResponse(BaseModel):
    """Top-level response envelope returned by ``SearchService.search``."""

    query: str = Field(..., description="Original query string")
    nlu: NLUParseResult = Field(..., description="NLU parse metadata explaining how the query was routed")
    total_hits: int = Field(default=0, description="Number of hits returned")
    hits: List[SearchHit] = Field(default_factory=list, description="Ranked search results")
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal operational notes (e.g. embedding fallback when provider offline)",
    )
    duration_ms: float = Field(default=0.0, description="End-to-end query latency in milliseconds")

    def to_wire(self, fmt: Optional[ResponseFormat] = None) -> Dict[str, Any]:
        """Serialize response for REST (``fhir``) or MCP (``compact``) consumers."""
        target_fmt = fmt or ResponseFormat.FHIR
        payload: Dict[str, Any] = {
            "query": self.query,
            "intent": self.nlu.intent.value,
            "total_hits": self.total_hits,
            "hits": [hit.to_projection(target_fmt) for hit in self.hits],
            "duration_ms": round(self.duration_ms, 2),
        }
        if self.warnings:
            payload["warnings"] = self.warnings
        if target_fmt == ResponseFormat.FHIR:
            payload["nlu"] = self.nlu.model_dump()
        return payload


def normalize_entity_types(
    entity_types: Optional[Sequence[SearchEntityType | str]],
) -> Optional[List[SearchEntityType]]:
    """Coerce string or enum entity type arguments into deduplicated ``SearchEntityType`` values."""
    if entity_types is None:
        return None
    normalized: List[SearchEntityType] = []
    seen: set[SearchEntityType] = set()
    alias_map: Dict[str, Sequence[SearchEntityType]] = {
        "clinical": sorted(CLINICAL_ENTITY_TYPES, key=lambda x: x.value),
        "dicom": sorted(DICOM_ENTITY_TYPES, key=lambda x: x.value),
        "all": sorted(CLINICAL_ENTITY_TYPES | DICOM_ENTITY_TYPES, key=lambda x: x.value),
    }
    for raw in entity_types:
        if isinstance(raw, SearchEntityType):
            items: Sequence[SearchEntityType] = (raw,)
        else:
            key = str(raw).strip()
            lowered = key.lower()
            if lowered in alias_map:
                items = alias_map[lowered]
            else:
                items = (SearchEntityType(key),)
        for item in items:
            if item not in seen:
                seen.add(item)
                normalized.append(item)
    return normalized


__all__ = [
    "CLINICAL_ENTITY_TYPES",
    "DEFAULT_UNIFIED_ENTITY_TYPES",
    "DICOM_ENTITY_TYPES",
    "ExtractedCodeEntity",
    "FHIRCoding",
    "MatchSource",
    "NLUIntent",
    "NLUParseResult",
    "RelatedConceptHit",
    "RelationDirection",
    "ResponseFormat",
    "SearchEntityType",
    "SearchHit",
    "SearchOptions",
    "SearchResponse",
    "SubsumptionOutcome",
    "normalize_entity_types",
]
