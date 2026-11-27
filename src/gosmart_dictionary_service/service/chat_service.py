"""Grounded Dictionary AI Chat service combining Unified SearchService retrieval with Genkit inference."""

from __future__ import annotations

import inspect
import re
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from genkit import Genkit
from genkit_google_genai import GoogleAI, VertexAI
from genkit_ollama import Ollama
from pydantic import BaseModel, Field

from gosmart_dictionary_service.core.config import Settings, get_settings
from gosmart_dictionary_service.core.logging import get_logger
from gosmart_dictionary_service.db.neo4j import Neo4jClient, neo4j_client
from gosmart_dictionary_service.model.llm_config import (
    DEFAULT_EMBEDDING_CONFIG_TAG,
    DEFAULT_INFERENCE_CONFIG_TAG,
    LLM_Config,
    build_default_inference_llm_config,
)
from gosmart_dictionary_service.model.search import (
    MatchSource,
    NLUParseResult,
    RelatedConceptHit,
    SearchEntityType,
    SearchHit,
    build_drilldown_url,
)
from gosmart_dictionary_service.service.embedding_service import EmbeddingService
from gosmart_dictionary_service.service.llm_config_service import DEFAULT_LLM_GROUP_NAME, LLMConfigService
from gosmart_dictionary_service.service.search_service import SearchService

logger = get_logger("gosmart_ds.chat")

_DICOM_TAG_RE = re.compile(r"\(?([0-9A-Fa-f]{4})\s*,?\s*([0-9A-Fa-f]{4})\)?")
_KEYWORD_TOKEN_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]{3,})\b")
_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9]{2,}")
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "what",
        "which",
        "where",
        "when",
        "how",
        "does",
        "are",
        "was",
        "were",
        "have",
        "has",
        "can",
        "could",
        "about",
        "into",
        "tell",
        "show",
        "find",
        "give",
        "please",
        "dicom",
        "tag",
        "tags",
        "attribute",
        "element",
        "code",
    }
)

ChatSearchScope = Literal["auto", "clinical", "dicom"]
GenerateReplyFn = Callable[[str, str, LLM_Config], Union[str, Awaitable[str]]]
SearchContextFn = Callable[[str, int], Union[List["ChatCitation"], Awaitable[List["ChatCitation"]]]]


class ChatProviderError(RuntimeError):
    """Raised when the configured LLM inference provider cannot produce a response."""


class ChatMessage(BaseModel):
    """Single conversational turn sent as chat history."""

    role: Literal["user", "assistant"] = Field(..., description="Speaker role")
    content: str = Field(..., min_length=1, max_length=8000, description="Message text")


class ChatRelatedItem(BaseModel):
    """Compact related clinical concept or DICOM entity attached to a grounded citation."""

    id: str = Field(..., description="Deterministic UUIDv5 id of the related node")
    entity_type: str = Field(..., description="Neo4j label of the related node")
    identifier: str = Field(..., description="Display code/tag identifier (e.g. 'SNOMED:233604007' or 'image-plane')")
    display: str = Field(..., description="Human-readable name of the related node")
    vocabulary: Optional[str] = Field(default=None, description="Clinical vocabulary when applicable")
    relation: str = Field(..., description="Semantic relationship category (e.g. 'maps_to', 'is_a_parent')")
    relationship_id: str = Field(default="", description="Underlying Neo4j/Athena relationship predicate")
    direction: str = Field(default="forward", description="Direction relative to the seed citation")
    macro: str = Field(default="", description="Precomputed gosmart-graphite Coding Macro (<system>|<code>|<display>)")
    drilldown_url: str = Field(..., description="Relative /explore URL to drill down into this related entity")


class ChatCitation(BaseModel):
    """Grounded dictionary entity retrieved from Neo4j to support an AI Chat answer."""

    id: Optional[str] = Field(default=None, description="Deterministic UUIDv5 node id when available")
    entity_type: str = Field(..., description="Neo4j entity label (e.g. Concept, DicomElement, DicomCIOD, DicomModule)")
    identifier: str = Field(
        ...,
        description="Canonical display identifier (e.g. 'LOINC:87279-6' or '(0018,0050)')",
    )
    code: Optional[str] = Field(default=None, description="Raw canonical code/tag_hex/slug within its CodeSystem")
    name: str = Field(..., description="Human-readable entity display/name")
    vocabulary: Optional[str] = Field(default=None, description="Clinical vocabulary (e.g. SNOMED, LOINC, ICD10CM)")
    domain: Optional[str] = Field(default=None, description="Clinical domain (e.g. Condition, Measurement, Procedure)")
    standard_concept: Optional[str] = Field(default=None, description="Athena standard_concept flag ('S', 'C', etc.)")
    keyword: Optional[str] = Field(default=None, description="DICOM keyword when applicable")
    vr: Optional[str] = Field(default=None, description="DICOM Value Representation when applicable")
    vm: Optional[str] = Field(default=None, description="DICOM Value Multiplicity when applicable")
    retired: Optional[bool] = Field(default=None, description="Whether the entity is retired or deprecated")
    snippet: Optional[str] = Field(default=None, description="Short description, definition, or synonym snippet")
    score: Optional[float] = Field(default=None, description="Retrieval similarity/relevance score")
    source: str = Field(
        default="fulltext", description="Primary retrieval stage: 'exact', 'hybrid', 'vector', 'fulltext'"
    )
    match_sources: List[str] = Field(default_factory=list, description="All provenance match sources for this hit")
    coding_system: Optional[str] = Field(default=None, description="Canonical FHIR CodeSystem URI")
    macro: Optional[str] = Field(
        default=None,
        description="Precomputed gosmart-graphite Coding Macro (<system>|<code>|<display>)",
    )
    drilldown_url: Optional[str] = Field(
        default=None,
        description="Relative /explore URL to inspect this entity and its graph neighborhood",
    )
    related: List[ChatRelatedItem] = Field(
        default_factory=list,
        description="Optional 1-hop/2-hop related concepts or DICOM structures",
    )


class ChatNLUSummary(BaseModel):
    """Compact NLU routing & query-rewriting metadata returned to the Chat UI."""

    intent: str = Field(..., description="Detected NLU intent (search, exact_lookup, crosswalk, related_lookup, etc.)")
    provider: str = Field(default="stub", description="Active NLU parser provider")
    normalized_query: str = Field(default="", description="Normalized query phrase")
    rewritten_query: Optional[str] = Field(
        default=None,
        description="Reverse-looked-up rewritten query when codes were expanded to display text",
    )
    synthesized_phrases: List[str] = Field(
        default_factory=list,
        description="LLM-synthesized alternate clinical/DICOM search phrases",
    )
    extracted_codes: List[str] = Field(
        default_factory=list,
        description="Formatted extracted code/tag entities (e.g. ['LOINC:87279-6'])",
    )
    inferred_vocabularies: List[str] = Field(default_factory=list, description="Inferred clinical vocabularies")
    target_vocabularies: List[str] = Field(default_factory=list, description="Target vocabularies for crosswalks")


class ChatRequest(BaseModel):
    """Payload for the grounded Dictionary AI Chat endpoint."""

    message: str = Field(..., min_length=1, max_length=4000, description="User question or lookup prompt")
    history: List[ChatMessage] = Field(
        default_factory=list,
        max_length=20,
        description="Optional prior conversation turns (most recent turns kept)",
    )
    include_context: bool = Field(
        default=True,
        description="Whether to ground the prompt with Neo4j dictionary lookup context",
    )
    context_limit: int = Field(
        default=8,
        ge=1,
        le=25,
        description="Maximum number of grounded dictionary citations to retrieve",
    )
    scope: ChatSearchScope = Field(
        default="auto",
        description="Search domain scope: 'auto' (Clinical + DICOM via NLU), 'clinical', or 'dicom'",
    )
    include_related: bool = Field(
        default=False,
        description="Attach 1-hop/2-hop related ontology concepts & DICOM structures to grounded hits",
    )
    test_id: Optional[str] = Field(
        default=None,
        description="Optional test_id scope for isolated test runs (None queries production data)",
    )


class ChatResponse(BaseModel):
    """Structured response returned by the grounded Dictionary AI Chat endpoint."""

    reply: str = Field(..., description="Assistant answer text")
    model: str = Field(..., description="Inference model name used")
    provider: str = Field(..., description="Inference provider used")
    citations: List[ChatCitation] = Field(
        default_factory=list,
        description="Grounded dictionary items retrieved from Neo4j for this turn",
    )
    nlu: Optional[ChatNLUSummary] = Field(
        default=None,
        description="NLU routing & query-expansion summary when dictionary grounding ran",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal NLU/search operational warnings (e.g. unloaded schemas or unknown codes)",
    )
    search_duration_ms: Optional[float] = Field(
        default=None,
        description="Dictionary search & NLU latency in milliseconds",
    )


@dataclass
class _GroundingBundle:
    """Internal container holding grounded citations, NLU metadata, warnings, and search latency."""

    citations: List[ChatCitation] = field(default_factory=list)
    nlu: Optional[ChatNLUSummary] = None
    warnings: List[str] = field(default_factory=list)
    duration_ms: Optional[float] = None


def build_lucene_query(user_query: str) -> str:
    """Convert raw user text into a safe, non-failing Lucene full-text query string."""
    raw_tokens = _WORD_TOKEN_RE.findall(user_query)
    filtered: List[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        lower = token.lower()
        if lower in _STOPWORDS or lower in seen:
            continue
        seen.add(lower)
        filtered.append(token)
        if len(filtered) >= 12:
            break

    if not filtered:
        for token in raw_tokens[:6]:
            lower = token.lower()
            if lower not in seen:
                seen.add(lower)
                filtered.append(token)

    return " OR ".join(filtered)


def extract_dicom_lookup_hints(user_query: str) -> Tuple[List[str], List[str]]:
    """Extract candidate 8-char DICOM hex tags and PascalCase keywords from free text."""
    tag_hexes: List[str] = []
    seen_tags: set[str] = set()
    for grp, elem in _DICOM_TAG_RE.findall(user_query):
        tag_hex = f"{grp}{elem}".upper()
        if tag_hex not in seen_tags:
            seen_tags.add(tag_hex)
            tag_hexes.append(tag_hex)

    keywords: List[str] = []
    seen_kw: set[str] = set()
    for kw in _KEYWORD_TOKEN_RE.findall(user_query):
        if kw not in seen_kw and kw.lower() not in _STOPWORDS:
            seen_kw.add(kw)
            keywords.append(kw)

    return tag_hexes, keywords


def _truncate_snippet(text: Optional[str], max_len: int = 240) -> Optional[str]:
    if not text:
        return None
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[: max_len - 1]}…"


def _primary_source_label(match_sources: Sequence[MatchSource]) -> str:
    if MatchSource.EXACT_CODE in match_sources:
        return "exact"
    if MatchSource.HYBRID in match_sources:
        return "hybrid"
    if MatchSource.VECTOR in match_sources and MatchSource.FULLTEXT not in match_sources:
        return "vector"
    if MatchSource.FULLTEXT in match_sources:
        return "fulltext"
    if match_sources:
        return match_sources[0].value
    return "fulltext"


def _related_hit_to_chat_item(rel: RelatedConceptHit) -> ChatRelatedItem:
    vocab = str(rel.properties["vocabulary"]) if rel.properties.get("vocabulary") else None
    tag_fmt = rel.properties.get("tag")
    if rel.entity_type == SearchEntityType.CONCEPT and vocab:
        ident = f"{vocab}:{rel.coding.code}"
    elif rel.entity_type == SearchEntityType.DICOM_ELEMENT and tag_fmt:
        ident = str(tag_fmt)
    else:
        ident = rel.coding.code
    return ChatRelatedItem(
        id=rel.id,
        entity_type=rel.entity_type.value,
        identifier=ident,
        display=rel.coding.display,
        vocabulary=vocab,
        relation=rel.relation_reason.value,
        relationship_id=rel.relationship_id or rel.relation_reason.value,
        direction=rel.direction.value,
        macro=rel.macro,
        drilldown_url=build_drilldown_url(
            node_id=rel.id,
            entity_type=rel.entity_type,
            vocabulary=vocab,
            code=rel.coding.code,
        ),
    )


def search_hit_to_citation(hit: SearchHit, *, related_limit: int = 8) -> ChatCitation:
    """Convert a unified ``SearchHit`` (Clinical Concept or DICOM entity) into a ``ChatCitation``."""
    props = hit.properties or {}
    vocab = str(props["vocabulary"]) if props.get("vocabulary") else None
    domain = str(props["domain"]) if props.get("domain") else None
    std_flag = str(props["standard"]) if props.get("standard") else None
    source_label = _primary_source_label(hit.match_sources)
    match_source_vals = [m.value for m in hit.match_sources]
    drilldown_url = build_drilldown_url(
        node_id=hit.id,
        entity_type=hit.entity_type,
        vocabulary=vocab,
        code=hit.coding.code,
    )
    related_items = [_related_hit_to_chat_item(r) for r in hit.related[:related_limit]]

    if hit.entity_type == SearchEntityType.CONCEPT:
        identifier = f"{vocab}:{hit.coding.code}" if vocab else hit.coding.code
        raw_snippet = props.get("definition")
        if not raw_snippet and props.get("synonyms"):
            syns = [str(s) for s in props["synonyms"][:4] if s]
            if syns:
                raw_snippet = "Synonyms: " + "; ".join(syns)
        return ChatCitation(
            id=hit.id,
            entity_type=hit.entity_type.value,
            identifier=identifier,
            code=hit.coding.code,
            name=hit.coding.display or identifier,
            vocabulary=vocab,
            domain=domain,
            standard_concept=std_flag,
            retired=bool(props.get("deprecated", False)),
            snippet=_truncate_snippet(raw_snippet),
            score=round(hit.score, 4),
            source=source_label,
            match_sources=match_source_vals,
            coding_system=hit.coding.system or None,
            macro=hit.macro or None,
            drilldown_url=drilldown_url,
            related=related_items,
        )

    if hit.entity_type == SearchEntityType.DICOM_ELEMENT:
        identifier = str(props.get("tag") or hit.coding.code)
        return ChatCitation(
            id=hit.id,
            entity_type=hit.entity_type.value,
            identifier=identifier,
            code=hit.coding.code,
            name=hit.coding.display or str(props.get("keyword") or "DICOM Element"),
            keyword=str(props["keyword"]) if props.get("keyword") else None,
            vr=str(props["vr"]) if props.get("vr") else None,
            vm=str(props["vm"]) if props.get("vm") else None,
            retired=bool(props.get("retired", False)),
            snippet=_truncate_snippet(props.get("description")),
            score=round(hit.score, 4),
            source=source_label,
            match_sources=match_source_vals,
            coding_system=hit.coding.system or None,
            macro=hit.macro or None,
            drilldown_url=drilldown_url,
            related=related_items,
        )

    # Other DICOM structures (DicomCIOD, DicomModule, DicomMacro, DicomSOPClass, DicomConfidentialityProfile)
    desc_parts: List[str] = []
    if props.get("ciod_name"):
        desc_parts.append(f"CIOD: {props['ciod_name']}")
    if props.get("basic_profile"):
        desc_parts.append(f"Basic Profile: {props['basic_profile']}")
    if props.get("description"):
        desc_parts.append(str(props["description"]))
    snippet_text = " — ".join(desc_parts) if desc_parts else None

    return ChatCitation(
        id=hit.id,
        entity_type=hit.entity_type.value,
        identifier=hit.coding.code,
        code=hit.coding.code,
        name=hit.coding.display or hit.coding.code,
        snippet=_truncate_snippet(snippet_text),
        score=round(hit.score, 4),
        source=source_label,
        match_sources=match_source_vals,
        coding_system=hit.coding.system or None,
        macro=hit.macro or None,
        drilldown_url=drilldown_url,
        related=related_items,
    )


def build_nlu_summary(nlu: NLUParseResult) -> ChatNLUSummary:
    """Project a full ``NLUParseResult`` into a UI-friendly ``ChatNLUSummary``."""
    extracted = [f"{e.vocabulary_id}:{e.code}" for e in nlu.extracted_codes]
    rewritten = nlu.rewritten_query
    if rewritten and rewritten.strip() == nlu.raw_query.strip():
        rewritten = None
    return ChatNLUSummary(
        intent=nlu.intent.value,
        provider=nlu.provider,
        normalized_query=nlu.normalized_query,
        rewritten_query=rewritten,
        synthesized_phrases=list(nlu.synthesized_phrases),
        extracted_codes=extracted,
        inferred_vocabularies=list(nlu.inferred_vocabularies),
        target_vocabularies=list(nlu.target_vocabularies),
    )


def _scope_to_entity_types(scope: ChatSearchScope) -> Optional[List[str]]:
    if scope == "clinical":
        return ["clinical"]
    if scope == "dicom":
        return ["dicom"]
    return None


class DictionaryChatService:
    """Grounded AI Chat assistant over GoSmart DS Neo4j dictionary entities and Genkit LLM flows."""

    SYSTEM_PROMPT = (
        "You are the GoSmart Health Dictionary Service (GoSmart DS) AI assistant, specializing in "
        "clinical terminologies (SNOMED CT, LOINC, ICD-10-CM, UCUM), DICOM standard data elements, "
        "CIODs, modules, SOP classes, confidentiality profiles, and FHIR coding representations.\n\n"
        "Guidelines:\n"
        "1. Base your answer on the Grounded Dictionary Context retrieved from Neo4j whenever it is provided.\n"
        "2. For Clinical Concepts, cite the exact Vocabulary, Code, Display Name, Domain, and FHIR Coding Macro "
        "(`system|code|display`) when helpful.\n"
        "3. For DICOM attributes, cite the exact tag `(gggg,eeee)`, Keyword, VR, and VM from context.\n"
        "4. Never fabricate or guess standard clinical codes or DICOM tags if they are not known with certainty; "
        "if Dictionary Warnings indicate an unloaded coding schema or unknown code, "
        "clearly communicate that to the user.\n"
        "5. Keep narrative explanations clear, concise, and actionable—structured tabular citations are shown "
        "directly below your reply in the UI."
    )

    def __init__(
        self,
        *,
        client: Optional[Neo4jClient] = None,
        llm_config_service: Optional[LLMConfigService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        search_service: Optional[SearchService] = None,
        settings: Optional[Settings] = None,
        generate_fn: Optional[GenerateReplyFn] = None,
        search_fn: Optional[SearchContextFn] = None,
    ) -> None:
        self.client = client or neo4j_client
        self.llm_config_service = llm_config_service or LLMConfigService(client=self.client)
        self.embedding_service = embedding_service or EmbeddingService(
            client=self.client,
            llm_config_service=self.llm_config_service,
            settings=settings,
        )
        self._search_service = search_service
        self._settings = settings
        self._generate_fn = generate_fn
        self._search_fn = search_fn
        self._client_lock = threading.Lock()
        self._genkit_clients: Dict[Tuple[Any, ...], Genkit] = {}

    @property
    def settings(self) -> Settings:
        return self._settings or get_settings()

    @property
    def search_service(self) -> SearchService:
        """Lazily initialize ``SearchService`` sharing this chat service's Neo4j and embedding clients."""
        if self._search_service is None:
            svc = SearchService(
                client=self.client,
                embedding_service=self.embedding_service,
                settings=self._settings,
            )
            svc.llm_config_service = self.llm_config_service
            self._search_service = svc
        return self._search_service

    async def resolve_inference_config(
        self,
        *,
        tag: Optional[str] = None,
        group_name: Optional[str] = DEFAULT_LLM_GROUP_NAME,
    ) -> LLM_Config:
        """Resolve the active inference ``LLM_Config`` from Neo4j, falling back to env defaults."""
        target_tag = tag or self.settings.INFERENCE_TAG or DEFAULT_INFERENCE_CONFIG_TAG
        try:
            cfg = await self.llm_config_service.get_llm_config_by_tag(
                target_tag,
                group_name=group_name,
                test_id=None,
            )
            if cfg is not None:
                return cfg
        except Exception as exc:
            logger.warning("inference_llm_config_lookup_fallback", tag=target_tag, error=str(exc))

        return build_default_inference_llm_config("GOSMART_HEALTH", settings=self.settings)

    def _get_or_create_genkit_client(self, config: LLM_Config) -> Tuple[Genkit, str]:
        """Build or reuse a Genkit client and canonical model reference for ``config``."""
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
                    raise ChatProviderError(
                        f"Unsupported inference provider '{provider}'. Supported providers: ollama, googleai, vertexai."
                    )
                ai = Genkit(plugins=[plugin])
                self._genkit_clients[fingerprint] = ai

        if provider == "ollama":
            model_ref = config.model if config.model.startswith("ollama/") else f"ollama/{config.model}"
        elif provider in {"googleai", "gemini"}:
            clean_model = config.model.removeprefix("googleai/")
            model_ref = f"googleai/{clean_model}"
        elif provider == "vertexai":
            clean_model = config.model.removeprefix("vertexai/")
            model_ref = f"vertexai/{clean_model}"
        else:
            raise ChatProviderError(f"Unsupported inference provider '{provider}'.")

        return ai, model_ref

    @staticmethod
    def _ensure_citation_drilldown(item: ChatCitation) -> ChatCitation:
        if item.drilldown_url:
            return item
        code_val = item.code or item.identifier
        if item.vocabulary and code_val.startswith(f"{item.vocabulary}:"):
            code_val = code_val.split(":", 1)[1]
        url = build_drilldown_url(
            node_id=item.id,
            entity_type=item.entity_type,
            vocabulary=item.vocabulary,
            code=code_val,
        )
        return item.model_copy(update={"drilldown_url": url})

    async def _retrieve_grounding_bundle(
        self,
        user_query: str,
        *,
        limit: int = 8,
        scope: ChatSearchScope = "auto",
        include_related: bool = False,
        test_id: Optional[str] = None,
    ) -> _GroundingBundle:
        """Execute unified ``SearchService.search`` (or custom ``search_fn``) and return grounding metadata."""
        if self._search_fn is not None:
            custom = self._search_fn(user_query, limit)
            raw_citations = await custom if inspect.isawaitable(custom) else list(custom)
            enriched = [self._ensure_citation_drilldown(c) for c in raw_citations[:limit]]
            return _GroundingBundle(citations=enriched)

        embed_tag = self.settings.VECTOR_EMBED_TAG or DEFAULT_EMBEDDING_CONFIG_TAG
        try:
            search_resp = await self.search_service.search(
                user_query,
                entity_types=_scope_to_entity_types(scope),
                include_related=include_related,
                related_limit=8,
                embedding_tag=embed_tag,
                limit=limit,
                test_id=test_id,
            )
        except Exception as exc:
            logger.warning("chat_unified_search_failed", error=str(exc))
            return _GroundingBundle(
                warnings=[f"Dictionary search unavailable ({exc.__class__.__name__}); answering without Neo4j context."]
            )

        citations = [search_hit_to_citation(hit, related_limit=8) for hit in search_resp.hits[:limit]]
        return _GroundingBundle(
            citations=citations,
            nlu=build_nlu_summary(search_resp.nlu),
            warnings=list(search_resp.warnings),
            duration_ms=search_resp.duration_ms,
        )

    async def retrieve_dictionary_context(
        self,
        user_query: str,
        limit: int = 8,
        *,
        scope: ChatSearchScope = "auto",
        include_related: bool = False,
        test_id: Optional[str] = None,
    ) -> List[ChatCitation]:
        """Retrieve grounded Clinical & DICOM dictionary citations via ``SearchService``."""
        bundle = await self._retrieve_grounding_bundle(
            user_query,
            limit=limit,
            scope=scope,
            include_related=include_related,
            test_id=test_id,
        )
        return bundle.citations

    @staticmethod
    def format_context_block(citations: Sequence[ChatCitation]) -> str:
        """Render retrieved Clinical & DICOM citations into a compact grounded context block for the LLM."""
        if not citations:
            return "No matching dictionary entries were found in Neo4j for this query."

        lines: List[str] = []
        for idx, item in enumerate(citations, start=1):
            parts = [f"[{idx}] {item.entity_type} {item.identifier} — {item.name}"]
            if item.vocabulary:
                parts.append(f"Vocabulary={item.vocabulary}")
            if item.domain:
                parts.append(f"Domain={item.domain}")
            if item.standard_concept == "S":
                parts.append("Standard=Yes")
            if item.keyword:
                parts.append(f"Keyword={item.keyword}")
            if item.vr:
                parts.append(f"VR={item.vr}")
            if item.vm:
                parts.append(f"VM={item.vm}")
            if item.retired is True:
                parts.append("DEPRECATED/RETIRED")
            if item.macro:
                parts.append(f"Macro={item.macro}")
            header = " | ".join(parts)

            entry_lines = [header]
            if item.snippet:
                entry_lines.append(f"    Details: {item.snippet}")
            if item.related:
                rel_summaries = [
                    f"{r.relationship_id or r.relation} -> {r.identifier} ({r.display})" for r in item.related[:6]
                ]
                entry_lines.append("    Related: " + "; ".join(rel_summaries))
            lines.append("\n".join(entry_lines))
        return "\n".join(lines)

    @staticmethod
    def build_prompt(
        message: str,
        history: Sequence[ChatMessage],
        citations: Sequence[ChatCitation],
        *,
        nlu: Optional[ChatNLUSummary] = None,
        warnings: Sequence[str] = (),
    ) -> str:
        """Compose the grounded prompt with NLU context, warnings, and recent conversation history."""
        sections: List[str] = []

        if nlu is not None:
            nlu_bits = [f"Intent={nlu.intent}"]
            if nlu.extracted_codes:
                nlu_bits.append("ExtractedCodes=" + ", ".join(nlu.extracted_codes))
            if nlu.rewritten_query:
                nlu_bits.append(f"RewrittenQuery={nlu.rewritten_query!r}")
            if nlu.target_vocabularies:
                nlu_bits.append("TargetVocabularies=" + ", ".join(nlu.target_vocabularies))
            sections.append("### NLU Routing Metadata\n" + " | ".join(nlu_bits))

        if warnings:
            warn_text = "\n".join(f"- {w}" for w in warnings)
            sections.append("### Dictionary Warnings & Guardrails\n" + warn_text)

        if citations:
            sections.append(
                "### Grounded Dictionary Context (Neo4j)\n" + DictionaryChatService.format_context_block(citations)
            )

        if history:
            recent = history[-10:]
            transcript = "\n".join(f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in recent)
            sections.append("### Recent Conversation\n" + transcript)

        sections.append(f"### Current User Question\n{message}")
        return "\n\n".join(sections)

    async def _invoke_llm(self, system_prompt: str, prompt: str, config: LLM_Config) -> str:
        if self._generate_fn is not None:
            result = self._generate_fn(system_prompt, prompt, config)
            return await result if inspect.isawaitable(result) else str(result)

        try:
            ai, model_ref = self._get_or_create_genkit_client(config)
            response = await ai.generate(
                model=model_ref,
                system=system_prompt,
                prompt=prompt,
                config=config.extra_config or None,
            )
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                raise ChatProviderError("Inference provider returned an empty response.")
            return text
        except ChatProviderError:
            raise
        except Exception as exc:
            logger.error(
                "chat_inference_failed",
                provider=config.provider,
                model=config.model,
                error=str(exc),
            )
            raise ChatProviderError(f"LLM inference failed for {config.provider}/{config.model}: {exc}") from exc

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Run a grounded Dictionary AI Chat turn powered by ``SearchService`` + Genkit."""
        bundle = _GroundingBundle()
        if request.include_context:
            bundle = await self._retrieve_grounding_bundle(
                request.message,
                limit=request.context_limit,
                scope=request.scope,
                include_related=request.include_related,
                test_id=request.test_id,
            )

        config = await self.resolve_inference_config()
        prompt = self.build_prompt(
            request.message,
            request.history,
            bundle.citations,
            nlu=bundle.nlu,
            warnings=bundle.warnings,
        )
        reply = await self._invoke_llm(self.SYSTEM_PROMPT, prompt, config)

        return ChatResponse(
            reply=reply,
            model=config.model,
            provider=config.provider,
            citations=bundle.citations,
            nlu=bundle.nlu,
            warnings=bundle.warnings,
            search_duration_ms=bundle.duration_ms,
        )


chat_service = DictionaryChatService()

__all__ = [
    "ChatCitation",
    "ChatMessage",
    "ChatNLUSummary",
    "ChatProviderError",
    "ChatRelatedItem",
    "ChatRequest",
    "ChatResponse",
    "ChatSearchScope",
    "DictionaryChatService",
    "build_lucene_query",
    "build_nlu_summary",
    "chat_service",
    "extract_dicom_lookup_hints",
    "search_hit_to_citation",
]
