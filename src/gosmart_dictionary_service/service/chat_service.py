"""Grounded Dictionary AI Chat service combining Neo4j DICOM retrieval with Genkit inference."""

from __future__ import annotations

import inspect
import re
import threading
from collections.abc import Awaitable, Callable, Sequence
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
from gosmart_dictionary_service.service.embedding_service import EmbeddingService
from gosmart_dictionary_service.service.llm_config_service import DEFAULT_LLM_GROUP_NAME, LLMConfigService

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

GenerateReplyFn = Callable[[str, str, LLM_Config], Union[str, Awaitable[str]]]
SearchContextFn = Callable[[str, int], Union[List["ChatCitation"], Awaitable[List["ChatCitation"]]]]


class ChatProviderError(RuntimeError):
    """Raised when the configured LLM inference provider cannot produce a response."""


class ChatMessage(BaseModel):
    """Single conversational turn sent as chat history."""

    role: Literal["user", "assistant"] = Field(..., description="Speaker role")
    content: str = Field(..., min_length=1, max_length=8000, description="Message text")


class ChatCitation(BaseModel):
    """Grounded dictionary entity retrieved from Neo4j to support an AI Chat answer."""

    entity_type: str = Field(..., description="Neo4j entity label (e.g. DicomElement, DicomCIOD, DicomModule)")
    identifier: str = Field(..., description="Canonical tag/slug/UID identifier (e.g. '(0018,0050)')")
    name: str = Field(..., description="Human-readable entity name")
    keyword: Optional[str] = Field(default=None, description="DICOM keyword when applicable")
    vr: Optional[str] = Field(default=None, description="DICOM Value Representation when applicable")
    vm: Optional[str] = Field(default=None, description="DICOM Value Multiplicity when applicable")
    retired: Optional[bool] = Field(default=None, description="Whether the DICOM attribute is retired")
    snippet: Optional[str] = Field(default=None, description="Short description snippet")
    score: Optional[float] = Field(default=None, description="Retrieval similarity/relevance score")
    source: str = Field(default="fulltext", description="Retrieval stage: 'exact', 'fulltext', or 'vector'")


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
        default=6,
        ge=1,
        le=20,
        description="Maximum number of grounded dictionary citations to retrieve",
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


class DictionaryChatService:
    """Grounded AI Chat assistant over GoSmart DS Neo4j dictionary entities and Genkit LLM flows."""

    SYSTEM_PROMPT = (
        "You are the GoSmart Health Dictionary Service (GoSmart DS) AI assistant, specializing in "
        "clinical terminologies and DICOM standard data elements, CIODs, modules, SOP classes, "
        "and FHIR coding representations.\n\n"
        "Guidelines:\n"
        "1. Prefer the Grounded Dictionary Context retrieved from Neo4j whenever it is provided.\n"
        "2. When citing DICOM attributes, include exact tag `(gggg,eeee)`, Keyword, VR, and VM from context.\n"
        "3. Never fabricate or guess standard DICOM tags or clinical codes if they are not known with certainty; "
        "clearly state when a code was not found in the loaded dictionary.\n"
        "4. Keep answers clear, concise, and practical for developers, clinicians, and integrators."
    )

    def __init__(
        self,
        *,
        client: Optional[Neo4jClient] = None,
        llm_config_service: Optional[LLMConfigService] = None,
        embedding_service: Optional[EmbeddingService] = None,
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
        self._settings = settings
        self._generate_fn = generate_fn
        self._search_fn = search_fn
        self._client_lock = threading.Lock()
        self._genkit_clients: Dict[Tuple[Any, ...], Genkit] = {}

    @property
    def settings(self) -> Settings:
        return self._settings or get_settings()

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

    async def _search_exact_dicom(self, user_query: str, limit: int) -> List[ChatCitation]:
        tag_hexes, keywords = extract_dicom_lookup_hints(user_query)
        if not tag_hexes and not keywords:
            return []

        query = """
        MATCH (n:DicomElement)
        WHERE (n.Action IS NULL OR n.Action <> 'DELETE')
          AND (n.test_id IS NULL OR n.test_id = '')
          AND (n.tag_hex IN $tag_hexes OR n.keyword IN $keywords)
        RETURN n.tag AS tag,
               n.tag_hex AS tag_hex,
               n.name AS name,
               n.keyword AS keyword,
               n.vr AS vr,
               n.vm AS vm,
               n.is_retired AS retired,
               n.description AS description
        LIMIT $limit
        """
        records = await self.client.execute_query(
            query,
            {"tag_hexes": tag_hexes, "keywords": keywords, "limit": limit},
        )
        return [
            ChatCitation(
                entity_type="DicomElement",
                identifier=str(rec.get("tag") or rec.get("tag_hex") or ""),
                name=str(rec.get("name") or rec.get("keyword") or "DICOM Element"),
                keyword=rec.get("keyword"),
                vr=rec.get("vr"),
                vm=rec.get("vm"),
                retired=rec.get("retired"),
                snippet=_truncate_snippet(rec.get("description")),
                score=1.0,
                source="exact",
            )
            for rec in records
        ]

    async def _search_fulltext_dicom(self, user_query: str, limit: int) -> List[ChatCitation]:
        lucene_query = build_lucene_query(user_query)
        if not lucene_query:
            return []

        citations: List[ChatCitation] = []
        elem_query = """
        CALL db.index.fulltext.queryNodes('dicom_element_fulltext_idx', $q)
        YIELD node AS n, score
        WHERE (n.Action IS NULL OR n.Action <> 'DELETE')
          AND (n.test_id IS NULL OR n.test_id = '')
        RETURN n.tag AS tag,
               n.tag_hex AS tag_hex,
               n.name AS name,
               n.keyword AS keyword,
               n.vr AS vr,
               n.vm AS vm,
               n.is_retired AS retired,
               n.description AS description,
               score
        ORDER BY score DESC
        LIMIT $limit
        """
        elem_records = await self.client.execute_query(elem_query, {"q": lucene_query, "limit": limit})
        for rec in elem_records:
            citations.append(
                ChatCitation(
                    entity_type="DicomElement",
                    identifier=str(rec.get("tag") or rec.get("tag_hex") or ""),
                    name=str(rec.get("name") or rec.get("keyword") or "DICOM Element"),
                    keyword=rec.get("keyword"),
                    vr=rec.get("vr"),
                    vm=rec.get("vm"),
                    retired=rec.get("retired"),
                    snippet=_truncate_snippet(rec.get("description")),
                    score=round(float(rec["score"]), 4) if rec.get("score") is not None else None,
                    source="fulltext",
                )
            )

        struct_limit = max(2, limit // 2)
        struct_query = """
        CALL db.index.fulltext.queryNodes('dicom_structure_fulltext_idx', $q)
        YIELD node AS n, score
        WHERE (n.Action IS NULL OR n.Action <> 'DELETE')
          AND (n.test_id IS NULL OR n.test_id = '')
        RETURN head(labels(n)) AS label,
               coalesce(n.ciod_id, n.module_id, n.macro_id, n.sop_class_uid, n.id) AS identifier,
               n.name AS name,
               n.description AS description,
               score
        ORDER BY score DESC
        LIMIT $limit
        """
        struct_records = await self.client.execute_query(struct_query, {"q": lucene_query, "limit": struct_limit})
        for rec in struct_records:
            citations.append(
                ChatCitation(
                    entity_type=str(rec.get("label") or "DicomStructure"),
                    identifier=str(rec.get("identifier") or ""),
                    name=str(rec.get("name") or rec.get("identifier") or "DICOM Structure"),
                    snippet=_truncate_snippet(rec.get("description")),
                    score=round(float(rec["score"]), 4) if rec.get("score") is not None else None,
                    source="fulltext",
                )
            )
        return citations

    async def _search_vector_dicom(self, user_query: str, limit: int) -> List[ChatCitation]:
        embed_tag = self.settings.VECTOR_EMBED_TAG or DEFAULT_EMBEDDING_CONFIG_TAG
        embed_cfg = await self.llm_config_service.get_llm_config_by_tag(
            embed_tag,
            group_name=DEFAULT_LLM_GROUP_NAME,
            test_id=None,
        )
        if embed_cfg is None:
            return []

        vectors = await self.embedding_service.compute_vectors([user_query], embed_cfg)
        if not vectors:
            return []

        query = """
        CALL db.index.vector.queryNodes('embedding_vector_idx', $k, $vec)
        YIELD node AS e, score
        WHERE (e.Action IS NULL OR e.Action <> 'DELETE')
          AND (e.test_id IS NULL OR e.test_id = '')
        MATCH (e)-[:EMBEDS]->(n)
        WHERE (n.Action IS NULL OR n.Action <> 'DELETE')
          AND (n.test_id IS NULL OR n.test_id = '')
        RETURN head(labels(n)) AS label,
               coalesce(n.tag, n.tag_hex, n.ciod_id, n.module_id, n.macro_id, n.sop_class_uid, n.id) AS identifier,
               n.name AS name,
               n.keyword AS keyword,
               n.vr AS vr,
               n.vm AS vm,
               n.is_retired AS retired,
               coalesce(n.description, e.text) AS description,
               score
        ORDER BY score DESC
        LIMIT $limit
        """
        records = await self.client.execute_query(
            query,
            {"k": max(limit * 3, 12), "vec": vectors[0], "limit": limit},
        )
        return [
            ChatCitation(
                entity_type=str(rec.get("label") or "DicomElement"),
                identifier=str(rec.get("identifier") or ""),
                name=str(rec.get("name") or rec.get("keyword") or "DICOM Entity"),
                keyword=rec.get("keyword"),
                vr=rec.get("vr"),
                vm=rec.get("vm"),
                retired=rec.get("retired"),
                snippet=_truncate_snippet(rec.get("description")),
                score=round(float(rec["score"]), 4) if rec.get("score") is not None else None,
                source="vector",
            )
            for rec in records
        ]

    async def retrieve_dictionary_context(self, user_query: str, limit: int = 6) -> List[ChatCitation]:
        """Retrieve grounded DICOM dictionary citations using exact, fulltext, and optional vector search."""
        if self._search_fn is not None:
            custom = self._search_fn(user_query, limit)
            return await custom if inspect.isawaitable(custom) else list(custom)

        collected: List[ChatCitation] = []
        seen: set[Tuple[str, str]] = set()

        def _append_unique(items: Sequence[ChatCitation]) -> None:
            for item in items:
                key = (item.entity_type, item.identifier)
                if not item.identifier or key in seen:
                    continue
                seen.add(key)
                collected.append(item)
                if len(collected) >= limit:
                    return

        try:
            exact_hits = await self._search_exact_dicom(user_query, limit=limit)
            _append_unique(exact_hits)
        except Exception as exc:
            logger.warning("chat_exact_dicom_lookup_failed", error=str(exc))

        if len(collected) < limit:
            try:
                fulltext_hits = await self._search_fulltext_dicom(user_query, limit=limit)
                _append_unique(fulltext_hits)
            except Exception as exc:
                logger.warning("chat_fulltext_dicom_lookup_failed", error=str(exc))

        if len(collected) < limit:
            try:
                vector_hits = await self._search_vector_dicom(user_query, limit=limit)
                _append_unique(vector_hits)
            except Exception as exc:
                logger.debug("chat_vector_dicom_lookup_skipped", error=str(exc))

        return collected[:limit]

    @staticmethod
    def format_context_block(citations: Sequence[ChatCitation]) -> str:
        """Render retrieved citations into a compact grounded context block for the LLM."""
        if not citations:
            return "No matching dictionary entries were found in Neo4j for this query."

        lines: List[str] = []
        for idx, item in enumerate(citations, start=1):
            parts = [f"[{idx}] {item.entity_type} {item.identifier} — {item.name}"]
            if item.keyword:
                parts.append(f"Keyword={item.keyword}")
            if item.vr:
                parts.append(f"VR={item.vr}")
            if item.vm:
                parts.append(f"VM={item.vm}")
            if item.retired is True:
                parts.append("RETIRED")
            header = " | ".join(parts)
            if item.snippet:
                lines.append(f"{header}\n    Description: {item.snippet}")
            else:
                lines.append(header)
        return "\n".join(lines)

    @staticmethod
    def build_prompt(
        message: str,
        history: Sequence[ChatMessage],
        citations: Sequence[ChatCitation],
    ) -> str:
        """Compose the grounded prompt with optional recent conversation history."""
        sections: List[str] = []

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
        """Run a grounded Dictionary AI Chat turn."""
        citations: List[ChatCitation] = []
        if request.include_context:
            citations = await self.retrieve_dictionary_context(request.message, limit=request.context_limit)

        config = await self.resolve_inference_config()
        prompt = self.build_prompt(request.message, request.history, citations)
        reply = await self._invoke_llm(self.SYSTEM_PROMPT, prompt, config)

        return ChatResponse(
            reply=reply,
            model=config.model,
            provider=config.provider,
            citations=citations,
        )


chat_service = DictionaryChatService()

__all__ = [
    "ChatCitation",
    "ChatMessage",
    "ChatProviderError",
    "ChatRequest",
    "ChatResponse",
    "DictionaryChatService",
    "build_lucene_query",
    "chat_service",
    "extract_dicom_lookup_hints",
]
