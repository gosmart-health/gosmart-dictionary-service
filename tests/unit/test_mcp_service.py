"""Modular unit tests for the GoSmart DS MCP Server, interactive session store, RBAC, and transports."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from gosmart_dictionary_service.__main__ import _build_parser
from gosmart_dictionary_service.__main__ import main as cli_main
from gosmart_dictionary_service.core import auth_dependency
from gosmart_dictionary_service.core.security import create_access_token
from gosmart_dictionary_service.model.llm_config import LLM_Config
from gosmart_dictionary_service.model.search import (
    EntityNeighborhood,
    FHIRCoding,
    GraphEdgeView,
    GraphNodeView,
    MatchSource,
    NLUIntent,
    NLUParseResult,
    RelatedConceptHit,
    RelationDirection,
    SearchEntityType,
    SearchHit,
    SearchResponse,
)
from gosmart_dictionary_service.service import mcp_service
from gosmart_dictionary_service.service.chat_service import (
    ChatCitation,
    ChatNLUSummary,
    ChatRequest,
    ChatResponse,
    search_hit_to_citation,
)
from gosmart_dictionary_service.service.mcp_service import (
    MCPSessionContextStore,
    _current_mcp_session_id,
    _current_mcp_user,
    ask_dictionary_ai,
    check_subsumption,
    crosswalk_codes,
    delete_llm_configuration,
    dictionary_health_check,
    expand_descendants,
    find_related_concepts,
    get_llm_configuration,
    inspect_entity,
    list_llm_configurations,
    lookup_concept,
    lookup_dicom,
    manage_mcp_session,
    mcp,
    mcp_session_store,
    search_dictionary,
    upsert_llm_configuration,
)
from gosmart_dictionary_service.service.nlu_service import ClinicalNLUService

EXPECTED_TOOLS = {
    "search_dictionary",
    "ask_dictionary_ai",
    "inspect_entity",
    "lookup_concept",
    "lookup_dicom",
    "find_related_concepts",
    "expand_descendants",
    "crosswalk_codes",
    "check_subsumption",
    "manage_mcp_session",
    "dictionary_health_check",
    "list_llm_configurations",
    "get_llm_configuration",
    "upsert_llm_configuration",
    "delete_llm_configuration",
}

LEGACY_PACS_TOOLS = {
    "list_dicom_nodes",
    "search_patients",
    "get_patient_studies",
    "trigger_c_echo",
    "trigger_c_find",
}


def _set_service_stub(monkeypatch: pytest.MonkeyPatch, target: Any, attr_name: str, stub: Any) -> None:
    """Bind a test stub onto a module or service object via pytest ``MonkeyPatch``."""
    monkeypatch.setattr(target, attr_name, stub)


@pytest.fixture(autouse=True)
def _reset_mcp_session_state() -> Any:
    """Isolate interactive MCP session state across every unit test."""
    mcp_session_store.clear_all()
    yield
    mcp_session_store.clear_all()


def _sample_concept_hit(
    *,
    code: str = "233604007",
    vocab: str = "SNOMED",
    display: str = "Pneumonia",
    related: Optional[List[RelatedConceptHit]] = None,
) -> SearchHit:
    system_uri = "http://snomed.info/sct" if vocab == "SNOMED" else "http://loinc.org"
    return SearchHit(
        id=f"concept-{vocab.lower()}-{code}",
        entity_type=SearchEntityType.CONCEPT,
        score=0.96,
        lexical_score=0.91,
        vector_score=0.88,
        match_sources=[MatchSource.HYBRID],
        coding=FHIRCoding(system=system_uri, code=code, display=display),
        macro=f"{system_uri}|{code}|{display}",
        properties={
            "vocabulary": vocab,
            "domain": "Condition",
            "standard": "S",
            "deprecated": False,
            "synonyms": ["Lung inflammation"],
        },
        related=related or [],
    )


def _sample_dicom_hit() -> SearchHit:
    return SearchHit(
        id="dicom-element-00180050",
        entity_type=SearchEntityType.DICOM_ELEMENT,
        score=1.0,
        match_sources=[MatchSource.EXACT_CODE],
        coding=FHIRCoding(
            system="http://gosmart.health/terminology/dicom-tag",
            code="00180050",
            display="Slice Thickness",
        ),
        macro="http://gosmart.health/terminology/dicom-tag|00180050|Slice Thickness",
        properties={
            "tag": "(0018,0050)",
            "tag_hex": "00180050",
            "keyword": "SliceThickness",
            "vr": "DS",
            "vm": "1",
            "retired": False,
            "description": "Nominal slice thickness, in mm.",
        },
    )


def _parse_mcp_http_payload(raw_text: str) -> Dict[str, Any]:
    """Decode either JSON or SSE ``data: {...}`` frames returned by StreamableHTTP."""
    stripped = raw_text.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    last_data: Optional[Dict[str, Any]] = None
    for line in stripped.splitlines():
        if line.startswith("data:"):
            candidate = line[len("data:") :].strip()
            if candidate:
                last_data = json.loads(candidate)
    if last_data is None:
        raise AssertionError(f"Could not parse MCP response payload: {raw_text!r}")
    return last_data


def _extract_tool_output(call_json: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap MCP v2 ``tools/call`` JSON-RPC result into the returned tool dict."""
    res = call_json["result"]
    assert res.get("isError") is not True
    sc = res.get("structuredContent")
    if isinstance(sc, dict):
        if set(sc.keys()) == {"result"} and isinstance(sc["result"], dict):
            return sc["result"]
        return sc
    return json.loads(res["content"][0]["text"])


def test_mcp_tool_catalog_matches_dictionary_design() -> None:
    tools = asyncio.run(mcp.list_tools())
    registered_names = {t.name for t in tools}
    assert registered_names == EXPECTED_TOOLS
    assert registered_names.isdisjoint(LEGACY_PACS_TOOLS)

    for tool in tools:
        props = tool.input_schema.get("properties", {})
        assert "ctx" not in props, f"`ctx` leaked into input_schema for {tool.name}"


def test_search_dictionary_compact_and_fhir_projections(monkeypatch: pytest.MonkeyPatch) -> None:
    hit = _sample_concept_hit()
    fake_resp = SearchResponse(
        query="acute pneumonia",
        nlu=NLUParseResult(raw_query="acute pneumonia", normalized_query="acute pneumonia", intent=NLUIntent.SEARCH),
        total_hits=1,
        hits=[hit],
        duration_ms=12.4,
    )
    search_mock = AsyncMock(return_value=fake_resp)
    _set_service_stub(monkeypatch, mcp_service, "search_service", SimpleNamespace(search=search_mock))

    compact_out = asyncio.run(
        search_dictionary(
            query="acute pneumonia",
            scope="clinical",
            format="compact",
            session_id="sess-search-1",
        )
    )
    assert compact_out["total_hits"] == 1
    assert compact_out["scope"] == "clinical"
    assert compact_out["session_id"] == "sess-search-1"
    assert compact_out["hits"][0]["macro"] == "http://snomed.info/sct|233604007|Pneumonia"
    assert "nlu" not in compact_out
    search_mock.assert_awaited_once()
    assert search_mock.await_args.kwargs["entity_types"] == ["clinical"]

    fhir_out = asyncio.run(
        search_dictionary(
            query="acute pneumonia",
            format="fhir",
            session_id="sess-search-1",
        )
    )
    assert "nlu" in fhir_out
    assert fhir_out["hits"][0]["coding"]["code"] == "233604007"


def test_interactive_session_retains_entities_and_chat_history_across_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    concept_hit = _sample_concept_hit(code="87279-6", vocab="LOINC", display="CT Chest")
    _set_service_stub(
        monkeypatch,
        mcp_service,
        "search_service",
        SimpleNamespace(
            search=AsyncMock(
                return_value=SearchResponse(
                    query="LOINC 87279-6",
                    nlu=NLUParseResult(
                        raw_query="LOINC 87279-6",
                        normalized_query="LOINC 87279-6",
                        intent=NLUIntent.EXACT_LOOKUP,
                    ),
                    total_hits=1,
                    hits=[concept_hit],
                    duration_ms=5.0,
                )
            )
        ),
    )

    captured_calls: List[Dict[str, Any]] = []

    async def _fake_chat(
        request: ChatRequest,
        *,
        extra_citations: Optional[Sequence[ChatCitation]] = None,
    ) -> ChatResponse:
        captured_calls.append(
            {
                "request": request,
                "extra_citations": list(extra_citations or []),
            }
        )
        merged = list(extra_citations or []) or [search_hit_to_citation(concept_hit)]
        return ChatResponse(
            reply=f"Answered: {request.message}",
            model="gemma4:e2b",
            provider="ollama",
            citations=merged,
            nlu=ChatNLUSummary(intent="search", provider="gosmart-nlu", normalized_query=request.message),
            search_duration_ms=4.2,
        )

    _set_service_stub(monkeypatch, mcp_service, "chat_service", SimpleNamespace(chat=_fake_chat))

    async def _scenario() -> None:
        # Step 1: Sticky scope configuration on session "interactive-1"
        scope_res = await manage_mcp_session(action="set_scope", scope="clinical", session_id="interactive-1")
        assert scope_res["default_scope"] == "clinical"

        # Step 2: Natural-language search caches LOINC 87279-6 in recent_entities
        await search_dictionary(query="LOINC 87279-6", session_id="interactive-1")

        # Step 3: Turn 1 AI question receives recent_entities from prior search_dictionary call
        turn1 = await ask_dictionary_ai(
            message="Explain that CT Chest code",
            session_id="interactive-1",
        )
        assert turn1["turn_count"] == 1
        assert turn1["scope"] == "clinical"
        assert len(captured_calls[0]["request"].history) == 0
        assert [c.identifier for c in captured_calls[0]["extra_citations"]] == ["LOINC:87279-6"]

        # Step 4: Turn 2 follow-up automatically receives prior user+assistant ChatMessage history
        turn2 = await ask_dictionary_ai(
            message="What domain does it belong to?",
            session_id="interactive-1",
        )
        assert turn2["turn_count"] == 2
        sent_history = captured_calls[1]["request"].history
        assert len(sent_history) == 2
        assert sent_history[0].role == "user"
        assert sent_history[0].content == "Explain that CT Chest code"
        assert sent_history[1].role == "assistant"
        assert "Answered:" in sent_history[1].content

        # Step 5: Inspect session status, then clear
        status_res = await manage_mcp_session(action="status", session_id="interactive-1")
        assert status_res["active"] is True
        assert status_res["turn_count"] == 2
        assert status_res["recent_entities"][0]["identifier"] == "LOINC:87279-6"

        clear_res = await manage_mcp_session(action="clear", session_id="interactive-1")
        assert clear_res["cleared"] is True

        after_clear = await manage_mcp_session(action="status", session_id="interactive-1")
        assert after_clear["active"] is False
        assert after_clear["turn_count"] == 0

    asyncio.run(_scenario())


def test_session_store_user_isolation_ttl_and_lru_eviction() -> None:
    store = MCPSessionContextStore(ttl_sec=1, max_sessions=2, max_history_turns=4)
    cit = search_hit_to_citation(_sample_concept_hit())

    store.record_entities("user-a", "shared-sid", [cit], query="pneumonia")
    assert store.peek("user-a", "shared-sid") is not None
    assert store.peek("user-b", "shared-sid") is None  # Isolated by user_scope

    # Exceed max_sessions=2 -> LRU ("user-a", "shared-sid") is evicted
    store.get_or_create("user-b", "sid-2")
    store.get_or_create("user-c", "sid-3")
    assert store.peek("user-a", "shared-sid") is None
    assert store.peek("user-b", "sid-2") is not None

    # History cap & TTL expiry
    store.record_chat_turn("user-b", "sid-2", user_message="q1", assistant_reply="a1")
    store.record_chat_turn("user-b", "sid-2", user_message="q2", assistant_reply="a2")
    store.record_chat_turn("user-b", "sid-2", user_message="q3", assistant_reply="a3")
    sess = store.peek("user-b", "sid-2")
    assert sess is not None
    assert len(sess.chat_history) == 4  # Capped at max_history_turns=4

    sess.updated_at = time.time() - 10.0
    assert store.peek("user-b", "sid-2") is None


def test_inspect_entity_groups_relationships_and_optional_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    rels = [
        RelatedConceptHit(
            id="rel-parent",
            entity_type=SearchEntityType.CONCEPT,
            direction=RelationDirection.FORWARD,
            relation_reason=MatchSource.IS_A_PARENT,
            relationship_id="Is a",
            coding=FHIRCoding(system="http://snomed.info/sct", code="64572001", display="Disease"),
            macro="http://snomed.info/sct|64572001|Disease",
            properties={"vocabulary": "SNOMED", "domain": "Condition"},
        ),
        RelatedConceptHit(
            id="rel-map",
            entity_type=SearchEntityType.CONCEPT,
            direction=RelationDirection.REVERSE,
            relation_reason=MatchSource.MAPPED_FROM,
            relationship_id="Maps to",
            coding=FHIRCoding(system="http://hl7.org/fhir/sid/icd-10-cm", code="J18.9", display="Pneumonia"),
            macro="http://hl7.org/fhir/sid/icd-10-cm|J18.9|Pneumonia",
            properties={"vocabulary": "ICD10CM", "domain": "Condition"},
        ),
    ]
    seed = _sample_concept_hit(related=rels)
    neighborhood = EntityNeighborhood(
        seed=seed,
        drilldown_url="/explore?id=concept-snomed-233604007&entity_type=Concept&vocab=SNOMED&code=233604007",
        nodes=[
            GraphNodeView(
                id=seed.id,
                entity_type=SearchEntityType.CONCEPT,
                code="233604007",
                display="Pneumonia",
                vocabulary="SNOMED",
                is_seed=True,
                drilldown_url="/explore?code=233604007",
                macro=seed.macro,
            )
        ],
        edges=[
            GraphEdgeView(
                source_id=seed.id,
                target_id="rel-parent",
                predicate="Is a",
                category=MatchSource.IS_A_PARENT,
                direction=RelationDirection.FORWARD,
            )
        ],
    )
    _set_service_stub(
        monkeypatch,
        mcp_service,
        "search_service",
        SimpleNamespace(inspect_entity=AsyncMock(return_value=neighborhood)),
    )

    result = asyncio.run(
        inspect_entity(
            vocabulary="SNOMED",
            code="233604007",
            include_graph=True,
            session_id="inspect-sess",
        )
    )
    assert result["found"] is True
    assert "hierarchy" in result["relationship_groups"]
    assert "crosswalks" in result["relationship_groups"]
    assert len(result["nodes"]) == 1
    assert len(result["edges"]) == 1

    with pytest.raises(ValueError, match="Provide at least"):
        asyncio.run(inspect_entity())


def test_granular_search_service_tools_and_health_check(monkeypatch: pytest.MonkeyPatch) -> None:
    concept = _sample_concept_hit()
    dicom = _sample_dicom_hit()
    rel = RelatedConceptHit(
        id="rel-1",
        entity_type=SearchEntityType.CONCEPT,
        direction=RelationDirection.FORWARD,
        relation_reason=MatchSource.MAPS_TO,
        relationship_id="Maps to",
        coding=FHIRCoding(system="http://snomed.info/sct", code="233604007", display="Pneumonia"),
        macro="http://snomed.info/sct|233604007|Pneumonia",
        properties={"vocabulary": "SNOMED"},
    )

    nlu_stub = MagicMock(spec=ClinicalNLUService)
    nlu_stub.get_loaded_schemas = AsyncMock(return_value={"SNOMED", "LOINC", "DICOM"})
    _set_service_stub(
        monkeypatch,
        mcp_service,
        "search_service",
        SimpleNamespace(
            lookup_code=AsyncMock(return_value=concept),
            lookup_dicom=AsyncMock(return_value=dicom),
            find_related_concepts=AsyncMock(return_value=[rel]),
            expand_descendants=AsyncMock(return_value=[concept]),
            crosswalk=AsyncMock(return_value=[rel]),
            subsumes=AsyncMock(return_value="subsumes"),
            nlu_service=nlu_stub,
        ),
    )
    _set_service_stub(
        monkeypatch,
        mcp_service,
        "neo4j_client",
        SimpleNamespace(verify_connectivity=AsyncMock(return_value=True)),
    )

    async def _run_all() -> None:
        c_res = await lookup_concept(vocabulary="SNOMED", code="233604007")
        assert c_res["found"] is True
        assert c_res["hit"]["macro"] == concept.macro

        d_res = await lookup_dicom(tag_or_keyword="(0018,0050)")
        assert d_res["found"] is True
        assert d_res["hit"]["keyword"] == "SliceThickness"

        rel_res = await find_related_concepts(vocabulary="ICD10CM", code="J18.9")
        assert rel_res["total_related"] == 1

        exp_res = await expand_descendants(vocabulary="SNOMED", code="64572001", max_depth=2)
        assert exp_res["total_descendants"] == 1
        assert exp_res["macros"] == [concept.macro]

        cw_res = await crosswalk_codes(vocabulary="ICD10CM", code="J18.9", target_vocabularies=["SNOMED"])
        assert cw_res["total_mappings"] == 1

        sub_res = await check_subsumption(
            ancestor_vocabulary="SNOMED",
            ancestor_code="64572001",
            descendant_vocabulary="SNOMED",
            descendant_code="233604007",
        )
        assert sub_res["outcome"] == "subsumes"

        health = await dictionary_health_check()
        assert health["status"] == "healthy"
        assert health["neo4j_connected"] is True
        assert health["loaded_schemas"] == ["DICOM", "LOINC", "SNOMED"]

    asyncio.run(_run_all())


def test_llm_config_tools_mask_keys_and_enforce_admin_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_obj = LLM_Config(
        tag="inference-v1",
        provider="googleai",
        model="gemini-2.5-flash",
        api_key="AIzaSySuperSecretKeyValue123456",
    )
    masked_row = {"id": "cfg-1", "tag": "inference-v1", "api_key": "AIza****3456"}
    _set_service_stub(
        monkeypatch,
        mcp_service,
        "llm_config_service",
        SimpleNamespace(
            get_llm_config_by_tag=AsyncMock(return_value=cfg_obj),
            list_llm_configs=AsyncMock(return_value=[masked_row]),
            upsert_llm_config=AsyncMock(return_value=masked_row),
            delete_llm_config=AsyncMock(return_value=True),
        ),
    )
    _set_service_stub(
        monkeypatch,
        mcp_service,
        "system_service",
        SimpleNamespace(get_root_group=AsyncMock(return_value={"id": "grp-root", "code": "GOSMART_HEALTH"})),
    )

    # Read tools mask api_key
    got = asyncio.run(get_llm_configuration(tag="inference-v1"))
    assert got["found"] is True
    assert "SuperSecret" not in got["config"]["api_key"]

    listed = asyncio.run(list_llm_configurations())
    assert listed["count"] == 1

    # Non-admin HTTP user cannot mutate LLM_Config
    non_admin_token = _current_mcp_user.set({"id": "user-1", "system_roles": ["USER"]})
    try:
        with pytest.raises(PermissionError, match="PLATFORM_ADMIN"):
            asyncio.run(upsert_llm_configuration(tag="inference-v1", provider="ollama", model="gemma4:e2b"))
        with pytest.raises(PermissionError, match="PLATFORM_ADMIN"):
            asyncio.run(delete_llm_configuration(config_id="cfg-1"))
    finally:
        _current_mcp_user.reset(non_admin_token)

    # PLATFORM_ADMIN user succeeds
    admin_token = _current_mcp_user.set({"id": "admin-1", "system_roles": ["PLATFORM_ADMIN"]})
    try:
        upserted = asyncio.run(upsert_llm_configuration(tag="inference-v1", provider="ollama", model="gemma4:e2b"))
        assert upserted["success"] is True
        deleted = asyncio.run(delete_llm_configuration(config_id="cfg-1"))
        assert deleted["success"] is True
    finally:
        _current_mcp_user.reset(admin_token)


def test_streamable_http_mcp_auth_and_session_handshake(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_user = {
        "id": "usr-mcp-001",
        "primary_email": "clinician@gosmart.health",
        "is_active": True,
        "email_verified": True,
        "system_roles": ["USER"],
    }
    _set_service_stub(
        monkeypatch,
        mcp_service,
        "user_service",
        SimpleNamespace(
            authenticate_api_key=AsyncMock(side_effect=lambda k: fake_user if k == "gs_dict_valid_test_key" else None),
        ),
    )
    _set_service_stub(
        monkeypatch,
        auth_dependency,
        "user_service",
        SimpleNamespace(get_user_by_id=AsyncMock(return_value=fake_user)),
    )
    _set_service_stub(
        monkeypatch,
        mcp_service,
        "search_service",
        SimpleNamespace(lookup_dicom=AsyncMock(return_value=_sample_dicom_hit())),
    )

    mcp_headers = {
        "Accept": "application/json, text/event-stream",
        "X-API-Key": "gs_dict_valid_test_key",
    }

    # 1. Initialize over POST /mcp (zero 307 redirect!)
    init_resp = client.post(
        "/mcp",
        headers=mcp_headers,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "pytest-mcp-client", "version": "1.0.0"},
            },
        },
        follow_redirects=False,
    )
    assert init_resp.status_code == 200
    session_id = init_resp.headers.get("mcp-session-id")
    assert session_id
    init_json = _parse_mcp_http_payload(init_resp.text)
    assert init_json["result"]["serverInfo"]["name"] == "gosmart-dictionary-service"

    # 2. Send notifications/initialized + call lookup_dicom using ONLY mcp-session-id (cached auth binding!)
    notif_resp = client.post(
        "/mcp",
        headers={
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": session_id,
        },
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        follow_redirects=False,
    )
    assert notif_resp.status_code in (200, 202)

    call_resp = client.post(
        "/mcp",
        headers={
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": session_id,
        },
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "lookup_dicom",
                "arguments": {"tag_or_keyword": "SliceThickness"},
            },
        },
        follow_redirects=False,
    )
    assert call_resp.status_code == 200
    call_json = _parse_mcp_http_payload(call_resp.text)
    structured = _extract_tool_output(call_json)
    assert structured["found"] is True
    assert structured["hit"]["tag"] == "00180050"
    assert structured["session_id"] == session_id

    # Verify session context was recorded under (fake_user["id"], session_id)
    status_resp = client.post(
        "/mcp",
        headers={
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": session_id,
        },
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "manage_mcp_session",
                "arguments": {"action": "status"},
            },
        },
        follow_redirects=False,
    )
    assert status_resp.status_code == 200
    status_out = _extract_tool_output(_parse_mcp_http_payload(status_resp.text))
    assert status_out["active"] is True
    assert status_out["session_id"] == session_id
    assert status_out["recent_entities"][0]["identifier"] == "(0018,0050)"

    # 3. Verify JWT Bearer auth also works on /mcp
    jwt_token = create_access_token({"sub": fake_user["id"], "email": fake_user["primary_email"]})
    jwt_init = client.post(
        "/mcp",
        headers={
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {jwt_token}",
        },
        json={
            "jsonrpc": "2.0",
            "id": 4,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "jwt-mcp-client", "version": "1.0.0"},
            },
        },
        follow_redirects=False,
    )
    assert jwt_init.status_code == 200
    assert _current_mcp_session_id.get() is None  # ContextVar cleanly reset after request


def test_cli_mcp_subcommand_routes_stdio_logs_to_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = _build_parser()
    parsed = parser.parse_args(["mcp", "--transport", "streamable-http", "--host", "127.0.0.1", "--port", "8901"])
    assert parsed.command == "mcp"
    assert parsed.transport == "streamable-http"
    assert parsed.host == "127.0.0.1"
    assert parsed.port == 8901

    setup_log_mock = MagicMock()
    run_mcp_mock = MagicMock()
    monkeypatch.setattr("gosmart_dictionary_service.__main__.setup_logging", setup_log_mock)
    monkeypatch.setattr("gosmart_dictionary_service.__main__.run_mcp_server", run_mcp_mock)
    monkeypatch.setattr("gosmart_dictionary_service.db.neo4j.Neo4jClient.close", AsyncMock())

    exit_code = cli_main(["mcp"])
    assert exit_code == 0
    setup_log_mock.assert_called_once_with(stream=sys.stderr)
    run_mcp_mock.assert_called_once_with(transport="stdio")
