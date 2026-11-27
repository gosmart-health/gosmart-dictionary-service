"""Model Context Protocol (MCP) server for GoSmart Dictionary Service.

Mirrors the ``/web`` portal's Natural-Language Ontology Explorer (``/explore``) and Grounded
Dictionary AI Chat (``/chat``) workflows while exposing granular ``SearchService`` ontology
traversal primitives, interactive per-session context retention, and role-gated ``LLM_Config``
administration over ``stdio``, ``streamable-http`` (``/mcp``), and ``sse`` (``/mcp-sse``).
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from http.cookies import SimpleCookie
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from fastapi import HTTPException
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from gosmart_dictionary_service.core.auth_dependency import (
    _authenticate_jwt_token,
    get_active_group_id,
    reset_active_group_id,
    set_active_group_id,
)
from gosmart_dictionary_service.core.config import Settings, get_settings
from gosmart_dictionary_service.core.logging import bind_log_context, get_logger
from gosmart_dictionary_service.db.neo4j import neo4j_client
from gosmart_dictionary_service.model.llm_config import (
    DEFAULT_EMBEDDING_CONFIG_TAG,
    DEFAULT_INFERENCE_CONFIG_TAG,
    LLM_Config,
)
from gosmart_dictionary_service.model.search import (
    EntityNeighborhood,
    MatchSource,
    RelatedConceptHit,
    RelationDirection,
    ResponseFormat,
    SearchEntityType,
    build_drilldown_url,
)
from gosmart_dictionary_service.service.chat_service import (
    ChatCitation,
    ChatMessage,
    ChatRequest,
    ChatSearchScope,
    chat_service,
    search_hit_to_citation,
)
from gosmart_dictionary_service.service.llm_config_service import (
    _dict_to_llm_config_response,
    llm_config_service,
)
from gosmart_dictionary_service.service.nlu_service import ClinicalNLUService
from gosmart_dictionary_service.service.search_service import search_service
from gosmart_dictionary_service.service.system_service import normalize_test_id, system_service
from gosmart_dictionary_service.service.user_service import user_service

logger = get_logger("gosmart_ds.mcp")

# Context variables propagating authenticated HTTP/SSE caller identity and transport session ID
_current_mcp_user: ContextVar[Optional[Dict[str, Any]]] = ContextVar("_current_mcp_user", default=None)
_current_mcp_session_id: ContextVar[Optional[str]] = ContextVar("_current_mcp_session_id", default=None)

_SSE_SESSION_ID_RE = re.compile(r"session_id=([A-Za-z0-9_-]+)")
_VALID_SCOPES: Dict[str, ChatSearchScope] = {
    "auto": "auto",
    "clinical": "clinical",
    "dicom": "dicom",
}

MCP_INSTRUCTIONS = (
    "GoSmart Health Dictionary Service (GoSmart DS) MCP Server.\n"
    "Provides natural-language terminology & DICOM search (`search_dictionary`), grounded AI Q&A with "
    "automatic multi-turn session memory (`ask_dictionary_ai`), neighborhood inspection (`inspect_entity`), "
    "exact O(1) lookups (`lookup_concept`, `lookup_dicom`), ontology graph traversal (`find_related_concepts`, "
    "`expand_descendants`, `crosswalk_codes`, `check_subsumption`), interactive session management "
    "(`manage_mcp_session`), health checks (`dictionary_health_check`), and `LLM_Config` administration.\n"
    "Defaults to `format='compact'` for token efficiency; pass `format='fhir'` when full FHIR Coding + "
    "properties envelopes are needed."
)


def get_current_mcp_user() -> Optional[Dict[str, Any]]:
    """Return the authenticated User dict bound to the current MCP HTTP/SSE request, or ``None`` on STDIO."""
    return _current_mcp_user.get()


def get_current_mcp_session_id() -> Optional[str]:
    """Return the transport-level MCP session identifier bound to the current request, if any."""
    return _current_mcp_session_id.get()


def _acted_by() -> str:
    user = _current_mcp_user.get()
    if user and user.get("id"):
        return str(user["id"])
    return "mcp_stdio"


def _require_admin_if_authenticated() -> Optional[Dict[str, Any]]:
    """Enforce ``PLATFORM_ADMIN`` or ``ADMIN`` role when invoked over authenticated HTTP/SSE transports.

    Local CLI ``STDIO`` invocations (where ``_current_mcp_user`` is ``None``) run with local operator privileges.
    """
    user = _current_mcp_user.get()
    if user is None:
        return None
    roles = {str(r).upper() for r in (user.get("system_roles") or [])}
    if "PLATFORM_ADMIN" not in roles and "ADMIN" not in roles:
        raise PermissionError("Operation requires PLATFORM_ADMIN or ADMIN system role.")
    return user


def _parse_response_format(raw_format: Optional[str]) -> ResponseFormat:
    cleaned = (raw_format or ResponseFormat.COMPACT.value).strip().lower()
    if cleaned == ResponseFormat.COMPACT.value:
        return ResponseFormat.COMPACT
    if cleaned == ResponseFormat.FHIR.value:
        return ResponseFormat.FHIR
    raise ValueError(f"Unsupported format '{raw_format}'. Expected 'compact' or 'fhir'.")


def _parse_direction(raw_direction: Optional[str]) -> RelationDirection:
    cleaned = (raw_direction or RelationDirection.BOTH.value).strip().lower()
    try:
        return RelationDirection(cleaned)
    except ValueError as exc:
        raise ValueError(f"Unsupported direction '{raw_direction}'. Expected 'forward', 'reverse', or 'both'.") from exc


def _parse_scope(raw_scope: Optional[str], *, fallback: ChatSearchScope = "auto") -> ChatSearchScope:
    if raw_scope is None or not str(raw_scope).strip():
        return fallback
    cleaned = str(raw_scope).strip().lower()
    if cleaned not in _VALID_SCOPES:
        raise ValueError(f"Unsupported scope '{raw_scope}'. Expected 'auto', 'clinical', or 'dicom'.")
    return _VALID_SCOPES[cleaned]


def _scope_to_entity_types(scope: ChatSearchScope) -> Optional[List[str]]:
    if scope == "clinical":
        return ["clinical"]
    if scope == "dicom":
        return ["dicom"]
    return None


def _project_citation(item: ChatCitation, fmt: ResponseFormat) -> Dict[str, Any]:
    if fmt == ResponseFormat.FHIR:
        return item.model_dump(exclude_none=True)

    compact: Dict[str, Any] = {
        "entity_type": item.entity_type,
        "identifier": item.identifier,
        "name": item.name,
        "source": item.source,
    }
    if item.vocabulary:
        compact["vocab"] = item.vocabulary
    if item.domain:
        compact["domain"] = item.domain
    if item.standard_concept:
        compact["standard"] = item.standard_concept
    if item.keyword:
        compact["keyword"] = item.keyword
    if item.vr:
        compact["vr"] = item.vr
    if item.vm:
        compact["vm"] = item.vm
    if item.retired:
        compact["retired"] = True
    if item.score is not None:
        compact["score"] = item.score
    if item.macro:
        compact["macro"] = item.macro
    if item.snippet:
        compact["snippet"] = item.snippet
    if item.drilldown_url:
        compact["drilldown_url"] = item.drilldown_url
    if item.related:
        compact["related"] = [
            {
                "entity_type": r.entity_type,
                "identifier": r.identifier,
                "display": r.display,
                "relation": r.relation,
                "predicate": r.relationship_id,
                "direction": r.direction,
                **({"macro": r.macro} if r.macro else {}),
            }
            for r in item.related
        ]
    return compact


def _format_neighborhood_relation(rel: RelatedConceptHit, fmt: ResponseFormat) -> Dict[str, Any]:
    if fmt == ResponseFormat.FHIR:
        row = rel.to_projection(ResponseFormat.FHIR)
        vocab = str(rel.properties["vocabulary"]) if rel.properties.get("vocabulary") else None
        row["id"] = rel.id
        row["entity_type"] = rel.entity_type.value
        row["drilldown_url"] = build_drilldown_url(
            node_id=rel.id,
            entity_type=rel.entity_type,
            vocabulary=vocab,
            code=rel.coding.code,
        )
        return row

    props = rel.properties or {}
    vocab = str(props["vocabulary"]) if props.get("vocabulary") else None
    if rel.entity_type == SearchEntityType.CONCEPT and vocab:
        identifier = f"{vocab}:{rel.coding.code}"
    elif rel.entity_type == SearchEntityType.DICOM_ELEMENT:
        identifier = str(props.get("tag") or rel.coding.code)
    else:
        identifier = rel.coding.code

    compact: Dict[str, Any] = {
        "id": rel.id,
        "entity_type": rel.entity_type.value,
        "identifier": identifier,
        "display": rel.coding.display or identifier,
        "relation": rel.relation_reason.value,
        "predicate": rel.relationship_id or rel.relation_reason.value,
        "direction": rel.direction.value,
        "macro": rel.macro,
        "drilldown_url": build_drilldown_url(
            node_id=rel.id,
            entity_type=rel.entity_type,
            vocabulary=vocab,
            code=rel.coding.code,
        ),
    }
    if vocab:
        compact["vocab"] = vocab
    if props.get("domain"):
        compact["domain"] = props["domain"]
    if props.get("equivalence"):
        compact["equivalence"] = props["equivalence"]
    if rel.shared_axes:
        compact["shared_axes"] = rel.shared_axes[:4]
    if props.get("keyword"):
        compact["keyword"] = props["keyword"]
    if props.get("vr"):
        compact["vr"] = props["vr"]
    if props.get("requirement_type"):
        compact["requirement_type"] = props["requirement_type"]
    if props.get("basic_profile"):
        compact["basic_profile"] = props["basic_profile"]
    return compact


def _group_neighborhood_relationships(
    neighborhood: EntityNeighborhood,
    fmt: ResponseFormat,
) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize ``neighborhood.seed.related`` into the same semantic buckets used by ``/explore``."""
    hierarchy: List[Dict[str, Any]] = []
    crosswalks: List[Dict[str, Any]] = []
    attributes: List[Dict[str, Any]] = []
    dicom_links: List[Dict[str, Any]] = []

    for rel in neighborhood.seed.related:
        item = _format_neighborhood_relation(rel, fmt)
        if rel.relation_reason in {MatchSource.IS_A_PARENT, MatchSource.IS_A_CHILD}:
            hierarchy.append(item)
        elif rel.relation_reason in {
            MatchSource.MAPS_TO,
            MatchSource.MAPPED_FROM,
            MatchSource.REPLACED_BY,
            MatchSource.REPLACES,
        }:
            crosswalks.append(item)
        elif rel.relation_reason == MatchSource.DICOM_RELATION:
            dicom_links.append(item)
        else:
            attributes.append(item)

    groups: Dict[str, List[Dict[str, Any]]] = {}
    if hierarchy:
        groups["hierarchy"] = hierarchy
    if crosswalks:
        groups["crosswalks"] = crosswalks
    if attributes:
        groups["attributes"] = attributes
    if dicom_links:
        groups["dicom_links"] = dicom_links
    return groups


# -----------------------------------------------------------------------------
# Interactive Stateful Session Context Store
# -----------------------------------------------------------------------------


@dataclass
class MCPSessionContext:
    """In-memory conversational and entity memory bound to an interactive MCP session."""

    user_scope: str
    session_id: str
    default_scope: ChatSearchScope = "auto"
    chat_history: List[ChatMessage] = field(default_factory=list)
    recent_entities: List[ChatCitation] = field(default_factory=list)
    last_query: Optional[str] = None
    last_intent: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class MCPSessionContextStore:
    """Thread-safe, TTL- and LRU-bounded session context store isolated by ``(user_scope, session_id)``."""

    def __init__(
        self,
        *,
        ttl_sec: Optional[int] = None,
        max_sessions: Optional[int] = None,
        max_history_turns: Optional[int] = None,
        max_recent_entities: int = 15,
    ) -> None:
        self._ttl_sec = ttl_sec
        self._max_sessions = max_sessions
        self._max_history_turns = max_history_turns
        self._max_recent_entities = max(1, max_recent_entities)
        self._lock = threading.RLock()
        self._sessions: OrderedDict[Tuple[str, str], MCPSessionContext] = OrderedDict()

    @property
    def ttl_sec(self) -> int:
        if self._ttl_sec is not None:
            return max(1, self._ttl_sec)
        return max(60, get_settings().MCP_SESSION_TTL_SEC)

    @property
    def max_sessions(self) -> int:
        if self._max_sessions is not None:
            return max(1, self._max_sessions)
        return max(1, get_settings().MCP_MAX_SESSIONS)

    @property
    def max_history_turns(self) -> int:
        if self._max_history_turns is not None:
            return max(2, self._max_history_turns)
        return max(2, get_settings().MCP_MAX_HISTORY_TURNS)

    def _prune_expired_locked(self, now: float) -> None:
        cutoff = now - float(self.ttl_sec)
        expired_keys = [key for key, sess in self._sessions.items() if sess.updated_at < cutoff]
        for key in expired_keys:
            self._sessions.pop(key, None)

    def _evict_overflow_locked(self) -> None:
        cap = self.max_sessions
        while len(self._sessions) > cap:
            self._sessions.popitem(last=False)

    def resolve_identity(
        self,
        explicit_session_id: Optional[str] = None,
        ctx: Optional[Context[Any, Any]] = None,
    ) -> Tuple[str, str]:
        """Resolve ``(user_scope, session_id)`` from explicit args, HTTP ContextVars, or MCP request context."""
        user = _current_mcp_user.get()
        user_scope = str(user["id"]) if user and user.get("id") else "stdio"

        if explicit_session_id and str(explicit_session_id).strip():
            return user_scope, str(explicit_session_id).strip()

        bound_sid = _current_mcp_session_id.get()
        if bound_sid and bound_sid.strip():
            return user_scope, bound_sid.strip()

        if ctx is not None:
            try:
                hdrs = ctx.headers
                if hdrs:
                    for hdr_name in ("mcp-session-id", "x-mcp-session-id"):
                        hdr_val = hdrs.get(hdr_name)
                        if hdr_val and str(hdr_val).strip():
                            return user_scope, str(hdr_val).strip()
            except Exception:
                pass
            try:
                sess_obj = ctx.session
                if sess_obj is not None:
                    return user_scope, f"mcp-conn-{id(sess_obj)}"
            except Exception:
                pass

        return user_scope, "stdio-default"

    def get_or_create(self, user_scope: str, session_id: str) -> MCPSessionContext:
        now = time.time()
        key = (user_scope, session_id)
        with self._lock:
            self._prune_expired_locked(now)
            sess = self._sessions.get(key)
            if sess is None:
                sess = MCPSessionContext(
                    user_scope=user_scope,
                    session_id=session_id,
                    created_at=now,
                    updated_at=now,
                )
                self._sessions[key] = sess
                self._evict_overflow_locked()
            else:
                sess.updated_at = now
                self._sessions.move_to_end(key)
            return sess

    def peek(self, user_scope: str, session_id: str) -> Optional[MCPSessionContext]:
        now = time.time()
        key = (user_scope, session_id)
        with self._lock:
            self._prune_expired_locked(now)
            sess = self._sessions.get(key)
            if sess is not None:
                sess.updated_at = now
                self._sessions.move_to_end(key)
            return sess

    def record_entities(
        self,
        user_scope: str,
        session_id: str,
        citations: Sequence[ChatCitation],
        *,
        query: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> MCPSessionContext:
        sess = self.get_or_create(user_scope, session_id)
        with self._lock:
            if query is not None:
                sess.last_query = query
            if intent is not None:
                sess.last_intent = intent
            merged: List[ChatCitation] = []
            seen: set[str] = set()
            for item in (*citations, *sess.recent_entities):
                dedup_key = item.id or f"{item.entity_type}:{item.identifier}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                merged.append(item)
                if len(merged) >= self._max_recent_entities:
                    break
            sess.recent_entities = merged
            sess.updated_at = time.time()
            return sess

    def record_chat_turn(
        self,
        user_scope: str,
        session_id: str,
        *,
        user_message: str,
        assistant_reply: str,
        citations: Sequence[ChatCitation] = (),
        intent: Optional[str] = None,
    ) -> MCPSessionContext:
        sess = self.record_entities(
            user_scope,
            session_id,
            citations,
            query=user_message,
            intent=intent,
        )
        with self._lock:
            sess.chat_history.append(ChatMessage(role="user", content=user_message[:8000]))
            sess.chat_history.append(ChatMessage(role="assistant", content=assistant_reply[:8000]))
            max_msgs = self.max_history_turns
            if len(sess.chat_history) > max_msgs:
                sess.chat_history = sess.chat_history[-max_msgs:]
            sess.updated_at = time.time()
            return sess

    def set_scope(self, user_scope: str, session_id: str, scope: ChatSearchScope) -> MCPSessionContext:
        sess = self.get_or_create(user_scope, session_id)
        with self._lock:
            sess.default_scope = scope
            sess.updated_at = time.time()
            return sess

    def clear(self, user_scope: str, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop((user_scope, session_id), None) is not None

    def clear_all(self) -> None:
        with self._lock:
            self._sessions.clear()

    def stats(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            self._prune_expired_locked(now)
            return {
                "active_sessions": len(self._sessions),
                "ttl_sec": self.ttl_sec,
                "max_sessions": self.max_sessions,
                "max_history_turns": self.max_history_turns,
            }


mcp_session_store = MCPSessionContextStore()

# Initialize Official MCP v2 Server
mcp = MCPServer(
    "gosmart-dictionary-service",
    title="GoSmart Dictionary Service",
    instructions=MCP_INSTRUCTIONS,
)


# -----------------------------------------------------------------------------
# A. Core /web-Mirrored Natural-Language & Explorer MCP Tools
# -----------------------------------------------------------------------------


@mcp.tool()
async def search_dictionary(
    query: str,
    scope: Optional[str] = None,
    entity_types: Optional[List[str]] = None,
    vocabularies: Optional[List[str]] = None,
    domains: Optional[List[str]] = None,
    valid_only: bool = True,
    standard_only: bool = False,
    include_related: bool = False,
    related_direction: str = "both",
    related_limit: int = 12,
    rerank_with_embeddings: bool = True,
    expand_with_llm: Optional[bool] = None,
    limit: int = 10,
    format: str = "compact",
    session_id: Optional[str] = None,
    test_id: Optional[str] = None,
    ctx: Context[Any, Any] | None = None,
) -> Dict[str, Any]:
    """Natural-language and code-aware search across Clinical Terminologies and the DICOM Standard.

    Mirrors the ``/explore`` portal workflow: runs spaCy clinical/DICOM NLU intent & code extraction,
    exact O(1) resolution, reverse-lookup query rewriting, Genkit alternate phrase synthesis, and
    hybrid Lucene BM25 + 768-d vector cosine reranking. Top hits are cached in the interactive
    session context for follow-up ``ask_dictionary_ai`` questions.

    Args:
        query: Natural-language phrase, clinical code (e.g. ``"LOINC 87279-6"``), or DICOM tag/keyword.
        scope: Domain filter: ``"auto"`` (Clinical + DICOM via NLU), ``"clinical"``, or ``"dicom"``.
            Defaults to the session's sticky scope or ``"auto"``.
        entity_types: Optional explicit entity labels (e.g. ``["Concept", "DicomElement", "DicomModule"]``).
        vocabularies: Optional clinical vocabulary allowlist (e.g. ``["SNOMED", "LOINC", "ICD10CM", "UCUM"]``).
        domains: Optional Athena domain allowlist (e.g. ``["Condition", "Measurement", "Procedure"]``).
        valid_only: Exclude deprecated/retired concepts and DICOM tags when True.
        standard_only: Restrict clinical hits to ``standard_concept == 'S'``.
        include_related: Attach 1-hop/2-hop related ontology concepts or DICOM structures to each hit.
        related_direction: Traversal direction for related hits (``"forward"``, ``"reverse"``, or ``"both"``).
        related_limit: Maximum related items per hit (1–100).
        rerank_with_embeddings: Enable 768-d vector cosine similarity reranking.
        expand_with_llm: Override ``NLU_LLM_EXPANSION_ENABLED`` for alternate search phrase synthesis.
        limit: Maximum top-level hits to return (1–200).
        format: Wire format: ``"compact"`` (token-efficient MCP default) or ``"fhir"`` (full Coding envelope).
        session_id: Optional interactive session identifier override.
        test_id: Optional isolated test scope (``None`` queries production data).
    """
    fmt = _parse_response_format(format)
    direction = _parse_direction(related_direction)
    user_scope, resolved_sid = mcp_session_store.resolve_identity(session_id, ctx)
    existing_sess = mcp_session_store.peek(user_scope, resolved_sid)
    fallback_scope: ChatSearchScope = existing_sess.default_scope if existing_sess else "auto"
    resolved_scope = _parse_scope(scope, fallback=fallback_scope)

    effective_entity_types = entity_types if entity_types is not None else _scope_to_entity_types(resolved_scope)
    bounded_limit = max(1, min(int(limit), 200))
    bounded_rel_limit = max(1, min(int(related_limit), 100))

    resp = await search_service.search(
        query,
        entity_types=effective_entity_types,
        vocabularies=vocabularies,
        domains=domains,
        valid_only=valid_only,
        standard_only=standard_only,
        include_related=include_related,
        related_direction=direction,
        related_limit=bounded_rel_limit,
        rerank_with_embeddings=rerank_with_embeddings,
        expand_with_llm=expand_with_llm,
        limit=bounded_limit,
        test_id=test_id,
    )

    citations = [search_hit_to_citation(hit, related_limit=8) for hit in resp.hits[:10]]
    mcp_session_store.record_entities(
        user_scope,
        resolved_sid,
        citations,
        query=query,
        intent=resp.nlu.intent.value,
    )

    payload = resp.to_wire(fmt)
    payload["scope"] = resolved_scope
    payload["session_id"] = resolved_sid
    return payload


@mcp.tool()
async def ask_dictionary_ai(
    message: str,
    scope: Optional[str] = None,
    include_context: bool = True,
    include_related: bool = False,
    context_limit: int = 8,
    use_session_context: bool = True,
    reset_session: bool = False,
    format: str = "compact",
    session_id: Optional[str] = None,
    test_id: Optional[str] = None,
    ctx: Context[Any, Any] | None = None,
) -> Dict[str, Any]:
    """Ask a natural-language question answered by the Grounded Dictionary AI Assistant.

    Mirrors ``/chat`` and ``POST /api/v1/chat``: retrieves grounded Neo4j clinical & DICOM entities
    via ``SearchService``, merges prior interactive session entities and multi-turn conversation
    history when ``use_session_context=True``, and generates a grounded answer via the active
    ``inference-v1`` Genkit model.

    Args:
        message: Natural-language clinical terminology, ontology crosswalk, or DICOM standard question.
        scope: Domain scope: ``"auto"``, ``"clinical"``, or ``"dicom"`` (defaults to sticky session scope).
        include_context: Ground the LLM prompt with Neo4j dictionary search citations.
        include_related: Attach 1-hop/2-hop related ontology/DICOM neighbors to grounded citations.
        context_limit: Maximum grounded dictionary citations included in the prompt (1–25).
        use_session_context: Retain and inject multi-turn chat history and recent entities for this session.
        reset_session: Clear prior session context before running this turn.
        format: Citation wire format: ``"compact"`` (default) or ``"fhir"``.
        session_id: Optional interactive session identifier override.
        test_id: Optional isolated test scope (``None`` queries production data).
    """
    fmt = _parse_response_format(format)
    user_scope, resolved_sid = mcp_session_store.resolve_identity(session_id, ctx)
    if reset_session:
        mcp_session_store.clear(user_scope, resolved_sid)

    existing_sess = mcp_session_store.peek(user_scope, resolved_sid) if use_session_context else None
    fallback_scope: ChatSearchScope = existing_sess.default_scope if existing_sess else "auto"
    resolved_scope = _parse_scope(scope, fallback=fallback_scope)

    prior_history = list(existing_sess.chat_history) if existing_sess else []
    prior_entities = list(existing_sess.recent_entities) if (existing_sess and include_context) else None
    bounded_limit = max(1, min(int(context_limit), 25))

    chat_req = ChatRequest(
        message=message,
        history=prior_history[-20:],
        include_context=include_context,
        context_limit=bounded_limit,
        scope=resolved_scope,
        include_related=include_related,
        test_id=test_id,
    )
    chat_resp = await chat_service.chat(chat_req, extra_citations=prior_entities)

    turn_count = 1
    if use_session_context:
        updated_sess = mcp_session_store.record_chat_turn(
            user_scope,
            resolved_sid,
            user_message=message,
            assistant_reply=chat_resp.reply,
            citations=chat_resp.citations,
            intent=chat_resp.nlu.intent if chat_resp.nlu else None,
        )
        turn_count = len(updated_sess.chat_history) // 2

    payload: Dict[str, Any] = {
        "reply": chat_resp.reply,
        "model": chat_resp.model,
        "provider": chat_resp.provider,
        "scope": resolved_scope,
        "citations": [_project_citation(c, fmt) for c in chat_resp.citations],
        "session_id": resolved_sid,
        "turn_count": turn_count,
    }
    if chat_resp.nlu is not None:
        payload["nlu"] = chat_resp.nlu.model_dump(exclude_none=True)
    if chat_resp.warnings:
        payload["warnings"] = chat_resp.warnings
    if chat_resp.search_duration_ms is not None:
        payload["search_duration_ms"] = round(chat_resp.search_duration_ms, 2)
    return payload


@mcp.tool()
async def inspect_entity(
    code: Optional[str] = None,
    vocabulary: Optional[str] = None,
    entity_type: Optional[str] = None,
    node_id: Optional[str] = None,
    related_limit: int = 40,
    include_graph: bool = False,
    format: str = "compact",
    session_id: Optional[str] = None,
    test_id: Optional[str] = None,
    ctx: Context[Any, Any] | None = None,
) -> Dict[str, Any]:
    """Inspect a Clinical Concept or DICOM entity and its categorized 1-hop/2-hop graph neighborhood.

    Mirrors ``GET /explore?id=...&vocab=...&code=...``: returns full seed entity details, precomputed
    ``gosmart-graphite`` Coding Macro, categorized relationship groups (``hierarchy``, ``crosswalks``,
    ``attributes``, ``dicom_links``), and optional graph ``nodes``/``edges``. Also caches the entity
    in the interactive session context.

    Args:
        code: Concept code, DICOM tag (e.g. ``"(0018,0050)"`` or ``"00180050"``), keyword, slug, or UID.
        vocabulary: Clinical vocabulary when inspecting by code (e.g. ``"SNOMED"``, ``"LOINC"``, ``"ICD10CM"``).
        entity_type: Optional Neo4j label hint (e.g. ``"Concept"``, ``"DicomElement"``, ``"DicomModule"``).
        node_id: Optional deterministic UUIDv5 ``id`` for direct node lookup.
        related_limit: Maximum related entities to retrieve (1–100).
        include_graph: Include deduplicated ``nodes`` and ``edges`` arrays alongside categorized groups.
        format: Wire format: ``"compact"`` (default) or ``"fhir"``.
        session_id: Optional interactive session identifier override.
        test_id: Optional isolated test scope (``None`` queries production data).
    """
    if not (node_id and str(node_id).strip()) and not (code and str(code).strip()):
        raise ValueError("Provide at least `code` (with optional `vocabulary` / `entity_type`) or `node_id`.")

    fmt = _parse_response_format(format)
    bounded_limit = max(1, min(int(related_limit), 100))
    neighborhood = await search_service.inspect_entity(
        node_id=node_id,
        entity_type=entity_type,
        vocabulary_id=vocabulary,
        code=code,
        related_limit=bounded_limit,
        test_id=test_id,
    )
    user_scope, resolved_sid = mcp_session_store.resolve_identity(session_id, ctx)
    if neighborhood is None:
        return {
            "found": False,
            "code": code,
            "vocabulary": vocabulary,
            "entity_type": entity_type,
            "node_id": node_id,
            "session_id": resolved_sid,
        }

    mcp_session_store.record_entities(
        user_scope,
        resolved_sid,
        [search_hit_to_citation(neighborhood.seed, related_limit=8)],
        query=code or node_id,
        intent="inspect_entity",
    )

    seed_without_dup_rels = neighborhood.seed.model_copy(update={"related": []})
    payload: Dict[str, Any] = {
        "found": True,
        "id": neighborhood.seed.id,
        "entity_type": neighborhood.seed.entity_type.value,
        "macro": neighborhood.seed.macro,
        "drilldown_url": neighborhood.drilldown_url,
        "seed": seed_without_dup_rels.to_projection(fmt),
        "total_related": len(neighborhood.seed.related),
        "relationship_groups": _group_neighborhood_relationships(neighborhood, fmt),
        "session_id": resolved_sid,
    }
    if fmt == ResponseFormat.COMPACT and neighborhood.seed.properties:
        # Preserve essential seed properties (e.g. description/synonyms/definition) during deep inspection
        payload["properties"] = neighborhood.seed.properties
    if include_graph:
        payload["nodes"] = [n.model_dump(exclude_none=True) for n in neighborhood.nodes]
        payload["edges"] = [e.model_dump(exclude_none=True) for e in neighborhood.edges]
    return payload


# -----------------------------------------------------------------------------
# B. Granular SearchService Lookup & Ontology Traversal Tools
# -----------------------------------------------------------------------------


@mcp.tool()
async def lookup_concept(
    vocabulary: str,
    code: str,
    resolve_deprecated: bool = True,
    include_related: bool = False,
    related_direction: str = "both",
    include_shared_attributes: bool = True,
    related_limit: int = 12,
    format: str = "compact",
    session_id: Optional[str] = None,
    test_id: Optional[str] = None,
    ctx: Context[Any, Any] | None = None,
) -> Dict[str, Any]:
    """Exact O(1) indexed lookup of a clinical ``:Concept`` by ``vocabulary`` and ``code``.

    Automatically follows ``[:REPLACED_BY]`` / ``[:MAPS_TO]`` upgrade targets on deprecated codes
    when ``resolve_deprecated=True``.
    """
    fmt = _parse_response_format(format)
    direction = _parse_direction(related_direction)
    bounded_limit = max(1, min(int(related_limit), 100))
    hit = await search_service.lookup_code(
        vocabulary,
        code,
        resolve_deprecated=resolve_deprecated,
        include_related=include_related,
        related_direction=direction,
        include_shared_attributes=include_shared_attributes,
        related_limit=bounded_limit,
        test_id=test_id,
    )
    user_scope, resolved_sid = mcp_session_store.resolve_identity(session_id, ctx)
    if hit is None:
        return {
            "found": False,
            "vocabulary": vocabulary,
            "code": code,
            "session_id": resolved_sid,
        }

    mcp_session_store.record_entities(
        user_scope,
        resolved_sid,
        [search_hit_to_citation(hit, related_limit=8)],
        query=f"{vocabulary}:{code}",
        intent="exact_lookup",
    )
    return {
        "found": True,
        "hit": hit.to_projection(fmt),
        "session_id": resolved_sid,
    }


@mcp.tool()
async def lookup_dicom(
    tag_or_keyword: str,
    include_related: bool = False,
    related_limit: int = 12,
    format: str = "compact",
    session_id: Optional[str] = None,
    test_id: Optional[str] = None,
    ctx: Context[Any, Any] | None = None,
) -> Dict[str, Any]:
    """Exact lookup of a DICOM Data Element (tag ``(gggg,eeee)``, 8-char hex, or keyword) or SOP Class UID."""
    fmt = _parse_response_format(format)
    bounded_limit = max(1, min(int(related_limit), 100))
    hit = await search_service.lookup_dicom(
        tag_or_keyword,
        include_related=include_related,
        related_limit=bounded_limit,
        test_id=test_id,
    )
    user_scope, resolved_sid = mcp_session_store.resolve_identity(session_id, ctx)
    if hit is None:
        return {
            "found": False,
            "tag_or_keyword": tag_or_keyword,
            "session_id": resolved_sid,
        }

    mcp_session_store.record_entities(
        user_scope,
        resolved_sid,
        [search_hit_to_citation(hit, related_limit=8)],
        query=tag_or_keyword,
        intent="exact_lookup",
    )
    return {
        "found": True,
        "hit": hit.to_projection(fmt),
        "session_id": resolved_sid,
    }


@mcp.tool()
async def find_related_concepts(
    vocabulary: str,
    code: str,
    direction: str = "both",
    include_shared_attributes: bool = True,
    valid_only: bool = True,
    min_shared_axes: int = 2,
    limit: int = 20,
    format: str = "compact",
    test_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Discover forward, reverse, and 2-hop shared-attribute related concepts for a clinical code."""
    fmt = _parse_response_format(format)
    rel_dir = _parse_direction(direction)
    bounded_limit = max(1, min(int(limit), 100))
    related = await search_service.find_related_concepts(
        vocabulary,
        code,
        direction=rel_dir,
        include_shared_attributes=include_shared_attributes,
        valid_only=valid_only,
        min_shared_axes=max(1, int(min_shared_axes)),
        limit=bounded_limit,
        test_id=test_id,
    )
    return {
        "vocabulary": vocabulary,
        "code": code,
        "direction": rel_dir.value,
        "total_related": len(related),
        "related": [item.to_projection(fmt) for item in related],
    }


@mcp.tool()
async def expand_descendants(
    vocabulary: str,
    code: str,
    max_depth: int = 2,
    valid_only: bool = True,
    limit: int = 100,
    format: str = "compact",
    test_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Expand valid ``[:IS_A*1..max_depth]`` descendant subtypes beneath a clinical root concept (FHIR ``$expand``)."""
    fmt = _parse_response_format(format)
    bounded_depth = max(1, min(int(max_depth), 8))
    bounded_limit = max(1, min(int(limit), 500))
    hits = await search_service.expand_descendants(
        vocabulary,
        code,
        max_depth=bounded_depth,
        valid_only=valid_only,
        limit=bounded_limit,
        test_id=test_id,
    )
    return {
        "vocabulary": vocabulary,
        "code": code,
        "max_depth": bounded_depth,
        "total_descendants": len(hits),
        "macros": [hit.macro for hit in hits],
        "descendants": [hit.to_projection(fmt) for hit in hits],
    }


@mcp.tool()
async def crosswalk_codes(
    vocabulary: str,
    code: str,
    target_vocabularies: Optional[List[str]] = None,
    max_hops: int = 2,
    limit: int = 25,
    format: str = "compact",
    test_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Translate a clinical concept across code systems via direct and 2-hop ``[:MAPS_TO]`` crosswalk edges."""
    fmt = _parse_response_format(format)
    bounded_hops = max(1, min(int(max_hops), 2))
    bounded_limit = max(1, min(int(limit), 100))
    mappings = await search_service.crosswalk(
        vocabulary,
        code,
        target_vocabularies=target_vocabularies,
        max_hops=bounded_hops,
        limit=bounded_limit,
        test_id=test_id,
    )
    return {
        "source_vocabulary": vocabulary,
        "source_code": code,
        "target_vocabularies": target_vocabularies,
        "max_hops": bounded_hops,
        "total_mappings": len(mappings),
        "mappings": [item.to_projection(fmt) for item in mappings],
    }


@mcp.tool()
async def check_subsumption(
    ancestor_vocabulary: str,
    ancestor_code: str,
    descendant_vocabulary: str,
    descendant_code: str,
    max_depth: int = 10,
    test_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate FHIR Terminology Service ``$subsumes`` between two clinical concepts.

    Returns ``outcome`` in ``{"equivalent", "subsumes", "subsumed-by", "not-subsumed"}``.
    """
    bounded_depth = max(1, min(int(max_depth), 15))
    outcome = await search_service.subsumes(
        ancestor_vocabulary,
        ancestor_code,
        descendant_vocabulary,
        descendant_code,
        max_depth=bounded_depth,
        test_id=test_id,
    )
    return {
        "outcome": outcome,
        "ancestor": f"{ancestor_vocabulary}:{ancestor_code}",
        "descendant": f"{descendant_vocabulary}:{descendant_code}",
        "max_depth": bounded_depth,
    }


# -----------------------------------------------------------------------------
# C. Interactive Session Management, Health & Admin LLM_Config Tools
# -----------------------------------------------------------------------------


@mcp.tool()
async def manage_mcp_session(
    action: str = "status",
    scope: Optional[str] = None,
    session_id: Optional[str] = None,
    ctx: Context[Any, Any] | None = None,
) -> Dict[str, Any]:
    """Inspect, configure sticky domain scope, or clear the interactive MCP session context.

    Args:
        action: One of ``"status"`` (inspect turn count & cached entities), ``"clear"`` (reset chat
            history and cached entities, mirroring the ``/chat`` Clear button), or ``"set_scope"``
            (persist sticky ``"auto" | "clinical" | "dicom"`` scope for subsequent calls).
        scope: Target domain scope when ``action == "set_scope"``.
        session_id: Optional interactive session identifier override.
    """
    clean_action = (action or "status").strip().lower()
    user_scope, resolved_sid = mcp_session_store.resolve_identity(session_id, ctx)

    if clean_action == "clear":
        cleared = mcp_session_store.clear(user_scope, resolved_sid)
        return {
            "action": "clear",
            "cleared": cleared,
            "session_id": resolved_sid,
            "turn_count": 0,
            "recent_entity_count": 0,
            "default_scope": "auto",
        }

    if clean_action == "set_scope":
        if scope is None or not str(scope).strip():
            raise ValueError("`scope` ('auto', 'clinical', or 'dicom') is required when action='set_scope'.")
        target_scope = _parse_scope(scope)
        sess = mcp_session_store.set_scope(user_scope, resolved_sid, target_scope)
        return {
            "action": "set_scope",
            "session_id": resolved_sid,
            "default_scope": sess.default_scope,
            "turn_count": len(sess.chat_history) // 2,
            "recent_entity_count": len(sess.recent_entities),
        }

    if clean_action != "status":
        raise ValueError(f"Unsupported action '{action}'. Expected 'status', 'clear', or 'set_scope'.")

    sess_opt = mcp_session_store.peek(user_scope, resolved_sid)
    if sess_opt is None:
        return {
            "action": "status",
            "active": False,
            "session_id": resolved_sid,
            "default_scope": "auto",
            "turn_count": 0,
            "message_count": 0,
            "recent_entities": [],
        }

    return {
        "action": "status",
        "active": True,
        "session_id": resolved_sid,
        "default_scope": sess_opt.default_scope,
        "turn_count": len(sess_opt.chat_history) // 2,
        "message_count": len(sess_opt.chat_history),
        "last_query": sess_opt.last_query,
        "last_intent": sess_opt.last_intent,
        "recent_entities": [
            {
                "entity_type": item.entity_type,
                "identifier": item.identifier,
                "name": item.name,
                **({"macro": item.macro} if item.macro else {}),
            }
            for item in sess_opt.recent_entities
        ],
    }


@mcp.tool()
async def dictionary_health_check(test_id: Optional[str] = None) -> Dict[str, Any]:
    """Check Neo4j connectivity, loaded dictionary coding schemas, active LLM tags, and MCP session stats."""
    settings = get_settings()
    connected = False
    try:
        connected = await neo4j_client.verify_connectivity()
    except Exception as exc:
        logger.warning("mcp_health_neo4j_check_failed", error=str(exc))

    loaded_schemas: List[str] = []
    if connected and isinstance(search_service.nlu_service, ClinicalNLUService):
        try:
            schemas = await search_service.nlu_service.get_loaded_schemas(test_id=normalize_test_id(test_id))
            if schemas:
                loaded_schemas = sorted(schemas)
        except Exception as exc:
            logger.warning("mcp_health_schema_check_failed", error=str(exc))

    return {
        "status": "healthy" if connected else "degraded",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "neo4j_connected": connected,
        "loaded_schemas": loaded_schemas,
        "default_embedding_tag": settings.VECTOR_EMBED_TAG or DEFAULT_EMBEDDING_CONFIG_TAG,
        "default_inference_tag": settings.INFERENCE_TAG or DEFAULT_INFERENCE_CONFIG_TAG,
        "mcp_sessions": mcp_session_store.stats(),
    }


@mcp.tool()
async def list_llm_configurations(
    group_id: Optional[str] = None,
    group_code: Optional[str] = None,
    include_deleted: bool = False,
    test_id: Optional[str] = None,
) -> Dict[str, Any]:
    """List group-scoped ``:LLM_Config`` records with secret API keys masked."""
    effective_group_id = group_id or get_active_group_id()
    configs = await llm_config_service.list_llm_configs(
        group_id=effective_group_id,
        group_code=group_code,
        test_id=test_id,
        include_deleted=include_deleted,
    )
    return {
        "count": len(configs),
        "configs": configs,
    }


@mcp.tool()
async def get_llm_configuration(
    tag: str,
    group_id: Optional[str] = None,
    group_code: Optional[str] = None,
    test_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch an active ``:LLM_Config`` by purpose ``tag`` (e.g. ``embedding-v1``) with ``api_key`` masked."""
    effective_group_id = group_id or get_active_group_id()
    cfg = await llm_config_service.get_llm_config_by_tag(
        tag,
        group_id=effective_group_id,
        group_code=group_code,
        test_id=test_id,
    )
    if cfg is None:
        return {"found": False, "tag": tag}
    return {
        "found": True,
        "config": _dict_to_llm_config_response(cfg.model_dump()),
    }


@mcp.tool()
async def upsert_llm_configuration(
    tag: str,
    provider: str,
    model: str,
    group_id: Optional[str] = None,
    url: str = "",
    dimensions: Optional[int] = None,
    dimensions_truncated: Optional[int] = None,
    api_key: str = "",
    location: str = "us-central1",
    project_id: Optional[str] = None,
    extra_config: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    test_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create or update a group-scoped ``:LLM_Config`` node (requires ``PLATFORM_ADMIN`` or ``ADMIN`` over HTTP/SSE).

    Defaults ``group_id`` to the active ``X-Group-ID`` context or the root ``GoSmart Health`` ``:Group``.
    """
    _require_admin_if_authenticated()
    target_group_id = group_id or get_active_group_id()
    if not target_group_id:
        root_group = await system_service.get_root_group(test_id=test_id)
        if not root_group or not root_group.get("id"):
            raise ValueError(
                "Root 'GoSmart Health' Group was not found in Neo4j. Run `gosmart-ds bootstrap` or specify `group_id`."
            )
        target_group_id = str(root_group["id"])

    actor = _acted_by()
    cfg_model = LLM_Config(
        tag=tag,
        provider=provider,
        model=model,
        url=url,
        dimensions=dimensions,
        dimensions_truncated=dimensions_truncated,
        api_key=api_key,
        location=location,
        project_id=project_id,
        extra_config=extra_config,
        description=description,
        test_id=normalize_test_id(test_id),
        ActedBy=actor,
    )
    saved = await llm_config_service.upsert_llm_config(
        cfg_model,
        target_group_id,
        acted_by=actor,
    )
    return {
        "success": True,
        "config": saved,
    }


@mcp.tool()
async def delete_llm_configuration(config_id: str) -> Dict[str, Any]:
    """Non-destructively soft-delete (``Action = 'DELETE'``) an ``:LLM_Config`` node by ``config_id``."""
    _require_admin_if_authenticated()
    deleted = await llm_config_service.delete_llm_config(config_id, acted_by=_acted_by())
    return {
        "success": deleted,
        "config_id": config_id,
    }


# -----------------------------------------------------------------------------
# Transport Security, Authentication Middleware & ASGI Mount Wrappers
# -----------------------------------------------------------------------------


def build_mcp_transport_security(settings: Optional[Settings] = None) -> TransportSecuritySettings:
    """Build DNS-rebinding ``TransportSecuritySettings`` for localhost, ``PUBLIC_BASE_URL``, and allowed hosts."""
    cfg = settings or get_settings()
    extra_raw = [h.strip() for h in (cfg.MCP_ALLOWED_HOSTS or "").split(",") if h and h.strip()]
    if "*" in extra_raw:
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)

    allowed_hosts: List[str] = [
        "127.0.0.1",
        "127.0.0.1:*",
        "localhost",
        "localhost:*",
        "[::1]",
        "[::1]:*",
        "0.0.0.0",
        "0.0.0.0:*",
        "testserver",
        "testserver:*",
    ]
    allowed_origins: List[str] = [
        "http://127.0.0.1:*",
        "https://127.0.0.1:*",
        "http://localhost:*",
        "https://localhost:*",
        "http://[::1]:*",
        "http://testserver",
        "http://testserver:*",
    ]

    if cfg.PUBLIC_BASE_URL:
        parsed = urlparse(cfg.PUBLIC_BASE_URL)
        host = parsed.hostname
        if isinstance(host, str) and host:
            host_wildcard = f"{host}:*"
            for candidate in (host, host_wildcard):
                if candidate not in allowed_hosts:
                    allowed_hosts.append(candidate)
            for scheme in ("http", "https"):
                for origin in (f"{scheme}://{host}", f"{scheme}://{host}:*"):
                    if origin not in allowed_origins:
                        allowed_origins.append(origin)

    for host_pattern in extra_raw:
        if host_pattern not in allowed_hosts:
            allowed_hosts.append(host_pattern)
        if ":" not in host_pattern and f"{host_pattern}:*" not in allowed_hosts:
            allowed_hosts.append(f"{host_pattern}:*")
        base_host = host_pattern.split(":", 1)[0]
        for scheme in ("http", "https"):
            for origin in (f"{scheme}://{base_host}", f"{scheme}://{base_host}:*"):
                if origin not in allowed_origins:
                    allowed_origins.append(origin)

    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )


class MCPAuthMiddleware:
    """Unified ASGI authentication & session-binding middleware for ``/mcp`` and ``/mcp-sse``.

    Supports:
    1. ``X-API-Key`` header, ``Authorization: Bearer gs_dict_...``, or ``?api_key=...``
    2. ``Authorization: Bearer <jwt>`` or ``?token=<jwt>``
    3. ``HttpOnly`` browser session cookie (``settings.SESSION_COOKIE_NAME``)
    4. Active authenticated transport session (``mcp-session-id`` header or SSE ``?session_id=...``)
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._lock = threading.RLock()
        # session_id -> (user_dict, active_group_id, expires_at_epoch)
        self._authenticated_sessions: OrderedDict[str, Tuple[Dict[str, Any], Optional[str], float]] = OrderedDict()

    def _prune_sessions_locked(self, now: float) -> None:
        expired = [sid for sid, (_, _, exp) in self._authenticated_sessions.items() if exp < now]
        for sid in expired:
            self._authenticated_sessions.pop(sid, None)

    def _lookup_session(self, session_id: Optional[str]) -> Optional[Tuple[Dict[str, Any], Optional[str]]]:
        if not session_id:
            return None
        now = time.time()
        ttl = float(max(60, get_settings().MCP_SESSION_TTL_SEC))
        with self._lock:
            self._prune_sessions_locked(now)
            entry = self._authenticated_sessions.get(session_id)
            if entry is None:
                return None
            user, group_id, _ = entry
            self._authenticated_sessions[session_id] = (user, group_id, now + ttl)
            self._authenticated_sessions.move_to_end(session_id)
            return user, group_id

    def _bind_session(self, session_id: str, user: Dict[str, Any], group_id: Optional[str]) -> None:
        if not session_id:
            return
        settings = get_settings()
        now = time.time()
        ttl = float(max(60, settings.MCP_SESSION_TTL_SEC))
        max_cap = max(1, settings.MCP_MAX_SESSIONS)
        with self._lock:
            self._prune_sessions_locked(now)
            self._authenticated_sessions[session_id] = (user, group_id, now + ttl)
            self._authenticated_sessions.move_to_end(session_id)
            while len(self._authenticated_sessions) > max_cap:
                self._authenticated_sessions.popitem(last=False)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        settings = get_settings()
        headers: Dict[str, str] = {
            k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])
        }
        query_params = parse_qs(scope.get("query_string", b"").decode("latin-1"))

        transport_sid = (
            headers.get("mcp-session-id")
            or headers.get("x-mcp-session-id")
            or (query_params.get("session_id", [None])[0])
        )
        if transport_sid:
            transport_sid = transport_sid.strip() or None

        req_group_id: Optional[str] = headers.get("x-group-id") or query_params.get("group_id", [None])[0]
        if req_group_id:
            req_group_id = req_group_id.strip() or None

        auth_header = headers.get("authorization") or ""
        raw_bearer: Optional[str] = None
        if auth_header.lower().startswith("bearer "):
            raw_bearer = auth_header[7:].strip() or None

        api_key: Optional[str] = headers.get("x-api-key") or query_params.get("api_key", [None])[0]
        if api_key:
            api_key = api_key.strip() or None
        if not api_key and raw_bearer and raw_bearer.startswith(settings.API_KEY_PREFIX):
            api_key = raw_bearer
            raw_bearer = None

        jwt_token: Optional[str] = raw_bearer or query_params.get("token", [None])[0]
        if jwt_token:
            jwt_token = jwt_token.strip() or None

        cookie_token: Optional[str] = None
        raw_cookie = headers.get("cookie")
        if raw_cookie:
            jar = SimpleCookie()
            try:
                jar.load(raw_cookie)
                morsel = jar.get(settings.SESSION_COOKIE_NAME)
                if morsel and morsel.value.strip():
                    cookie_token = morsel.value.strip()
            except Exception:
                cookie_token = None

        user: Optional[Dict[str, Any]] = None
        auth_error_detail = "Authentication required for GoSmart DS MCP endpoint."

        # 1. Explicit API Key
        if api_key:
            user = await user_service.authenticate_api_key(api_key)
            if not user:
                auth_error_detail = "Invalid, revoked, or expired API Key."

        # 2. Explicit JWT Bearer / query token
        elif jwt_token:
            try:
                user = await _authenticate_jwt_token(jwt_token, auth_type="mcp_jwt")
            except HTTPException as exc:
                auth_error_detail = str(exc.detail)

        # 3. Browser HttpOnly Session Cookie
        elif cookie_token:
            try:
                user = await _authenticate_jwt_token(cookie_token, auth_type="mcp_cookie")
            except HTTPException as exc:
                auth_error_detail = str(exc.detail)

        # 4. Previously authenticated MCP transport session
        if user is None and transport_sid and not (api_key or jwt_token):
            cached = self._lookup_session(transport_sid)
            if cached is not None:
                user, cached_group_id = cached
                if not req_group_id:
                    req_group_id = cached_group_id

        if user is None:
            resp = JSONResponse(
                {"detail": auth_error_detail},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await resp(scope, receive, send)
            return

        authed_user: Dict[str, Any] = user
        bound_group_id: Optional[str] = req_group_id

        if transport_sid:
            self._bind_session(transport_sid, authed_user, bound_group_id)

        bind_log_context(user_id=authed_user["id"], mcp_session_id=transport_sid)
        user_token = _current_mcp_user.set(authed_user)
        sid_token = _current_mcp_session_id.set(transport_sid)
        grp_token = set_active_group_id(bound_group_id)

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                for k, v in message.get("headers", []):
                    if k.decode("latin-1").lower() == "mcp-session-id":
                        issued_sid = v.decode("latin-1").strip()
                        if issued_sid:
                            self._bind_session(issued_sid, authed_user, bound_group_id)
                            _current_mcp_session_id.set(issued_sid)
            elif message["type"] == "http.response.body":
                body_bytes = message.get("body", b"")
                if body_bytes and b"session_id=" in body_bytes:
                    match = _SSE_SESSION_ID_RE.search(body_bytes.decode("utf-8", errors="ignore"))
                    if match:
                        self._bind_session(match.group(1), authed_user, bound_group_id)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            reset_active_group_id(grp_token)
            _current_mcp_session_id.reset(sid_token)
            _current_mcp_user.reset(user_token)


class DynamicMCPStreamableApp:
    """Reusable ASGI proxy that rebuilds ``mcp.streamable_http_app`` per FastAPI lifespan cycle.

    ``StreamableHTTPSessionManager.run()`` is single-use per app instance; refreshing inside FastAPI's
    ``lifespan`` guarantees clean startup/shutdown across both production servers and multiple
    ``TestClient`` lifespans in pytest.
    """

    def __init__(self, streamable_http_path: str = "/") -> None:
        self.streamable_http_path = streamable_http_path
        self._app: Optional[ASGIApp] = None

    def refresh(self, settings: Optional[Settings] = None) -> ASGIApp:
        cfg = settings or get_settings()
        raw_app = mcp.streamable_http_app(
            streamable_http_path=self.streamable_http_path,
            session_idle_timeout=float(max(60, cfg.MCP_SESSION_TTL_SEC)),
            max_sessions=max(1, cfg.MCP_MAX_SESSIONS),
            transport_security=build_mcp_transport_security(cfg),
        )
        wrapped: ASGIApp = MCPAuthMiddleware(raw_app)
        self._app = wrapped
        return wrapped

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        app = self._app
        if app is None:
            app = self.refresh()
        await app(scope, receive, send)


class DirectMCPMountAlias:
    """ASGI endpoint allowing ``POST /mcp`` (without trailing slash) to route directly without ``307`` redirect."""

    def __init__(self, target: ASGIApp, mount_prefix: str) -> None:
        self.target = target
        self.mount_prefix = mount_prefix.rstrip("/")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        patched = dict(
            scope,
            path="/",
            root_path=f"{scope.get('root_path', '')}{self.mount_prefix}",
        )
        await self.target(patched, receive, send)


dynamic_mcp_streamable_app = DynamicMCPStreamableApp(streamable_http_path="/")


def refresh_mcp_streamable_app(settings: Optional[Settings] = None) -> ASGIApp:
    """Rebuild the Streamable HTTP ASGI app and initialize ``mcp.session_manager`` for FastAPI lifespan."""
    return dynamic_mcp_streamable_app.refresh(settings=settings)


def get_mcp_streamable_http_app(streamable_http_path: str = "/") -> ASGIApp:
    """Return the authenticated Streamable HTTP ASGI app for mounting at ``/mcp``."""
    if streamable_http_path == "/":
        if dynamic_mcp_streamable_app._app is None:
            dynamic_mcp_streamable_app.refresh()
        return dynamic_mcp_streamable_app
    cfg = get_settings()
    return MCPAuthMiddleware(
        mcp.streamable_http_app(
            streamable_http_path=streamable_http_path,
            session_idle_timeout=float(max(60, cfg.MCP_SESSION_TTL_SEC)),
            max_sessions=max(1, cfg.MCP_MAX_SESSIONS),
            transport_security=build_mcp_transport_security(cfg),
        )
    )


def get_mcp_sse_app(settings: Optional[Settings] = None) -> ASGIApp:
    """Return the authenticated SSE ASGI app for mounting at ``/mcp-sse``."""
    cfg = settings or get_settings()
    return MCPAuthMiddleware(
        mcp.sse_app(
            transport_security=build_mcp_transport_security(cfg),
        )
    )


def run_mcp_server(
    transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
    *,
    host: Optional[str] = None,
    port: Optional[int] = None,
    transport_security: Optional[TransportSecuritySettings] = None,
) -> None:
    """Run the GoSmart DS MCP server synchronously over ``stdio``, ``streamable-http``, or ``sse``."""
    logger.info("mcp_server_starting", transport=transport, host=host, port=port)
    if transport == "stdio":
        mcp.run(transport="stdio")
        return

    cfg = get_settings()
    resolved_host = host or "127.0.0.1"
    resolved_port = port if port is not None else 8000
    security = transport_security or build_mcp_transport_security(cfg)
    if transport == "sse":
        mcp.run(
            transport="sse",
            host=resolved_host,
            port=resolved_port,
            transport_security=security,
        )
    else:
        mcp.run(
            transport="streamable-http",
            host=resolved_host,
            port=resolved_port,
            session_idle_timeout=float(max(60, cfg.MCP_SESSION_TTL_SEC)),
            max_sessions=max(1, cfg.MCP_MAX_SESSIONS),
            transport_security=security,
        )


__all__ = [
    "DirectMCPMountAlias",
    "DynamicMCPStreamableApp",
    "MCPAuthMiddleware",
    "MCPSessionContext",
    "MCPSessionContextStore",
    "ask_dictionary_ai",
    "build_mcp_transport_security",
    "check_subsumption",
    "crosswalk_codes",
    "delete_llm_configuration",
    "dictionary_health_check",
    "dynamic_mcp_streamable_app",
    "expand_descendants",
    "find_related_concepts",
    "get_current_mcp_session_id",
    "get_current_mcp_user",
    "get_llm_configuration",
    "get_mcp_sse_app",
    "get_mcp_streamable_http_app",
    "inspect_entity",
    "list_llm_configurations",
    "lookup_concept",
    "lookup_dicom",
    "manage_mcp_session",
    "mcp",
    "mcp_session_store",
    "refresh_mcp_streamable_app",
    "run_mcp_server",
    "search_dictionary",
    "upsert_llm_configuration",
]
