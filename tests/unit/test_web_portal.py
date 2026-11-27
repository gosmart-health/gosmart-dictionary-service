"""Modular unit tests for the multi-page server-rendered HTML web portal (/chat, /explore, /keys, auth)."""

from __future__ import annotations

import re
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient

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
    build_drilldown_url,
)
from gosmart_dictionary_service.service.chat_service import ChatNLUSummary, ChatResponse, search_hit_to_citation
from tests.unit.test_auth_api import _patch_services


def _extract_csrf(html_text: str) -> str:
    match = re.search(r'name="csrf_token"\s+value="([^"]+)"', html_text)
    assert match is not None, "Expected hidden csrf_token input in HTML form"
    return match.group(1)


class _PortalStubSearchService:
    """Stub SearchService for /explore route unit tests."""

    def __init__(self, neighborhood: EntityNeighborhood, search_resp: SearchResponse) -> None:
        self._neighborhood = neighborhood
        self._search_resp = search_resp

    async def inspect_entity(
        self,
        *,
        node_id: Optional[str] = None,
        code: Optional[str] = None,
        **_kwargs: Any,
    ) -> Optional[EntityNeighborhood]:
        if node_id == "missing-id" or code == "MISSING":
            return None
        return self._neighborhood

    async def search(self, *_args: Any, **_kwargs: Any) -> SearchResponse:
        return self._search_resp


class _RichChatStub:
    """Stub DictionaryChatService returning NLU metadata, warnings, and related chips."""

    def __init__(self, response: ChatResponse) -> None:
        self._response = response

    async def resolve_inference_config(self, **_kwargs: Any) -> Any:
        from gosmart_dictionary_service.model.llm_config import LLM_Config

        return LLM_Config(tag="inference-v1", provider="ollama", model="gemma4:e2b")

    async def chat(self, _req: Any) -> ChatResponse:
        return self._response


def test_root_content_negotiation_and_static_assets(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_services(monkeypatch, dev_mode=True)

    # Default / JSON API client gets service metadata JSON
    json_resp = client.get("/", headers={"Accept": "application/json"})
    assert json_resp.status_code == 200
    assert json_resp.json()["name"] == "GoSmart DS"

    # Browser Accept: text/html gets multi-page HTML home page
    html_resp = client.get("/", headers={"Accept": "text/html,application/xhtml+xml"})
    assert html_resp.status_code == 200
    assert "text/html" in html_resp.headers["content-type"]
    assert "GoSmart Health Dictionary Service" in html_resp.text
    assert 'href="/register"' in html_resp.text
    assert 'href="/login"' in html_resp.text

    # Static CSS and small JS assets are served cleanly
    css = client.get("/static/style.css")
    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]

    js = client.get("/static/portal.js")
    assert js.status_code == 200


def test_unauthenticated_keys_chat_and_explore_redirect_to_login(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_services(monkeypatch, dev_mode=True)

    keys_resp = client.get("/keys", follow_redirects=False)
    assert keys_resp.status_code == 303
    assert keys_resp.headers["location"] == "/login?next=/keys"

    chat_resp = client.get("/chat", follow_redirects=False)
    assert chat_resp.status_code == 303
    assert chat_resp.headers["location"] == "/login?next=/chat"

    explore_resp = client.get("/explore?id=uuid-loinc-87279-6", follow_redirects=False)
    assert explore_resp.status_code == 303
    assert explore_resp.headers["location"] == "/login?next=/explore?id=uuid-loinc-87279-6"


def test_full_html_portal_journey_register_verify_login_keys_chat_logout(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_services(monkeypatch, dev_mode=True)

    # 1. Load /register and extract CSRF token
    reg_page = client.get("/register")
    assert reg_page.status_code == 200
    csrf = _extract_csrf(reg_page.text)

    # 2. Missing CSRF is rejected (403)
    bad_csrf = client.post(
        "/register",
        data={
            "email": "webuser@example.org",
            "display_name": "Web User",
            "password": "StrongPassword123!",
            "password_confirm": "StrongPassword123!",
            "terms_accepted": "on",
            "csrf_token": "tampered-token",
        },
    )
    assert bad_csrf.status_code == 403

    # 3. Password mismatch returns 400 with helpful error
    mismatch = client.post(
        "/register",
        data={
            "email": "webuser@example.org",
            "display_name": "Web User",
            "password": "StrongPassword123!",
            "password_confirm": "DifferentPassword!",
            "terms_accepted": "on",
            "csrf_token": csrf,
        },
    )
    assert mismatch.status_code == 400
    assert "Passwords do not match" in mismatch.text

    # 4. Valid registration displays DEV_MODE local verification link
    reg_ok = client.post(
        "/register",
        data={
            "email": "webuser@example.org",
            "display_name": "Web User",
            "password": "StrongPassword123!",
            "password_confirm": "StrongPassword123!",
            "terms_accepted": "on",
            "csrf_token": csrf,
        },
    )
    assert reg_ok.status_code == 201
    assert "DEV_MODE Local Verification Link" in reg_ok.text
    link_match = re.search(r"/verify-email\?token=([A-Za-z0-9_\-]+)", reg_ok.text)
    assert link_match is not None
    verify_path = link_match.group(0)

    # 5. Click verification link -> account activated
    verify_page = client.get(verify_path)
    assert verify_page.status_code == 200
    assert "Email verified!" in verify_page.text
    assert "webuser@example.org" in verify_page.text

    # 6. Log in via /login -> receives HttpOnly session cookie and redirects to /keys
    login_page = client.get("/login")
    assert login_page.status_code == 200
    login_csrf = _extract_csrf(login_page.text)

    login_submit = client.post(
        "/login",
        data={
            "email": "webuser@example.org",
            "password": "StrongPassword123!",
            "next_url": "/keys",
            "csrf_token": login_csrf,
        },
        follow_redirects=False,
    )
    assert login_submit.status_code == 303
    assert login_submit.headers["location"] == "/keys"
    assert "gosmart_ds_session" in login_submit.cookies

    # 7. Visit /keys and generate an API key
    keys_page = client.get("/keys")
    assert keys_page.status_code == 200
    assert "webuser@example.org" in keys_page.text
    keys_csrf = _extract_csrf(keys_page.text)

    create_key = client.post(
        "/keys/create",
        data={"name": "Research CLI", "expires_in_days": "90", "csrf_token": keys_csrf},
    )
    assert create_key.status_code == 201
    assert "Save your new API key now" in create_key.text
    assert "Research CLI" in create_key.text
    key_match = re.search(r'id="generated-api-key">([^<]+)<', create_key.text)
    assert key_match is not None
    generated_key = key_match.group(1).strip()
    assert generated_key.startswith("gs_dict_")

    # Verify generated key works on REST /api/v1/auth/me
    me_via_key = client.get("/api/v1/auth/me", headers={"X-API-Key": generated_key})
    assert me_via_key.status_code == 200
    assert me_via_key.json()["email"] == "webuser@example.org"

    # 8. Revoke the API key via HTML form
    revoke_match = re.search(r'action="/keys/([^/]+)/revoke"', create_key.text)
    assert revoke_match is not None
    key_id = revoke_match.group(1)
    revoke_submit = client.post(
        f"/keys/{key_id}/revoke",
        data={"csrf_token": keys_csrf},
        follow_redirects=True,
    )
    assert revoke_submit.status_code == 200
    assert "API key revoked successfully" in revoke_submit.text

    # 9. Visit /chat (with prefill query & scope controls) and submit via HTML form + /api/v1/chat
    chat_page = client.get("/chat?q=What+is+(0018,0050)%3F&scope=dicom")
    assert chat_page.status_code == 200
    assert "Dictionary AI Chat" in chat_page.text
    assert 'name="scope"' in chat_page.text
    assert 'name="include_related"' in chat_page.text
    assert "What is (0018,0050)?" in chat_page.text
    chat_csrf = _extract_csrf(chat_page.text)

    chat_html_post = client.post(
        "/chat",
        data={
            "message": "What is (0018,0050)?",
            "scope": "auto",
            "include_context": "on",
            "include_related": "on",
            "csrf_token": chat_csrf,
        },
    )
    assert chat_html_post.status_code == 200
    assert "citation-drawer" in chat_html_post.text
    assert "citation-table" in chat_html_post.text
    assert "/explore?" in chat_html_post.text
    assert "SliceThickness" in chat_html_post.text

    chat_api_cookie = client.post(
        "/api/v1/chat",
        json={"message": "What is (0018,0050)?", "include_context": True, "scope": "dicom", "include_related": True},
    )
    assert chat_api_cookie.status_code == 200
    assert chat_api_cookie.json()["citations"][0]["identifier"] == "(0018,0050)"
    assert chat_api_cookie.json()["citations"][0]["drilldown_url"].startswith("/explore?")

    # 10. Log out via POST /logout -> session cookie cleared, /keys redirects to /login again
    logout_resp = client.post(
        "/logout",
        data={"csrf_token": chat_csrf},
        follow_redirects=False,
    )
    assert logout_resp.status_code == 303
    assert logout_resp.headers["location"] == "/login?logged_out=1"

    after_logout = client.get("/keys", follow_redirects=False)
    assert after_logout.status_code == 303
    assert after_logout.headers["location"] == "/login?next=/keys"


def test_chat_nlu_warnings_and_explore_entity_drilldown_page(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_users = _patch_services(monkeypatch, dev_mode=True)

    # Create & verify user + log in
    reg_page = client.get("/register")
    csrf = _extract_csrf(reg_page.text)
    reg_ok = client.post(
        "/register",
        data={
            "email": "explorer@example.org",
            "display_name": "Explorer User",
            "password": "StrongPassword123!",
            "password_confirm": "StrongPassword123!",
            "terms_accepted": "on",
            "csrf_token": csrf,
        },
    )
    verify_match = re.search(r"/verify-email\?token=([A-Za-z0-9_\-]+)", reg_ok.text)
    assert verify_match is not None
    client.get(verify_match.group(0))
    login_csrf = _extract_csrf(client.get("/login").text)
    client.post(
        "/login",
        data={
            "email": "explorer@example.org",
            "password": "StrongPassword123!",
            "next_url": "/explore",
            "csrf_token": login_csrf,
        },
        follow_redirects=False,
    )
    assert fake_users is not None

    seed_hit = SearchHit(
        id="uuid-loinc-87279-6",
        entity_type=SearchEntityType.CONCEPT,
        score=1.0,
        match_sources=[MatchSource.EXACT_CODE],
        coding=FHIRCoding(system="http://loinc.org", code="87279-6", display="CT Chest for screening"),
        macro="http://loinc.org|87279-6|CT Chest for screening",
        properties={
            "vocabulary": "LOINC",
            "domain": "Measurement",
            "concept_class": "Rad",
            "standard": "S",
            "deprecated": False,
            "synonyms": ["Low-dose screening CT thorax"],
        },
        related=[
            RelatedConceptHit(
                id="uuid-snomed-169069000",
                entity_type=SearchEntityType.CONCEPT,
                direction=RelationDirection.FORWARD,
                relation_reason=MatchSource.MAPS_TO,
                relationship_id="Maps to",
                score=0.96,
                coding=FHIRCoding(
                    system="http://snomed.info/sct",
                    code="169069000",
                    display="Computed tomography of chest",
                ),
                macro="http://snomed.info/sct|169069000|Computed tomography of chest",
                properties={"vocabulary": "SNOMED", "domain": "Procedure", "standard": "S"},
            )
        ],
    )
    seed_url = build_drilldown_url(
        node_id=seed_hit.id,
        entity_type=seed_hit.entity_type,
        vocabulary="LOINC",
        code="87279-6",
    )
    neighborhood = EntityNeighborhood(
        seed=seed_hit,
        drilldown_url=seed_url,
        nodes=[
            GraphNodeView(
                id=seed_hit.id,
                entity_type=SearchEntityType.CONCEPT,
                code="87279-6",
                display="CT Chest for screening",
                vocabulary="LOINC",
                is_seed=True,
                drilldown_url=seed_url,
                macro=seed_hit.macro,
            ),
            GraphNodeView(
                id="uuid-snomed-169069000",
                entity_type=SearchEntityType.CONCEPT,
                code="169069000",
                display="Computed tomography of chest",
                vocabulary="SNOMED",
                is_seed=False,
                drilldown_url="/explore?id=uuid-snomed-169069000",
                macro="http://snomed.info/sct|169069000|Computed tomography of chest",
            ),
        ],
        edges=[
            GraphEdgeView(
                source_id=seed_hit.id,
                target_id="uuid-snomed-169069000",
                predicate="Maps to",
                category=MatchSource.MAPS_TO,
                direction=RelationDirection.FORWARD,
            )
        ],
    )
    search_resp = SearchResponse(
        query="chest CT",
        nlu=NLUParseResult(raw_query="chest CT", normalized_query="chest CT", intent=NLUIntent.SEARCH),
        total_hits=1,
        hits=[seed_hit],
        duration_ms=9.4,
    )
    for target in ("gosmart_dictionary_service.web.router.search_service",):
        monkeypatch.setattr(target, _PortalStubSearchService(neighborhood, search_resp))

    # 1. Inspect /explore entity drill-down page -> renders properties, macro, clickable crosswalk row & graph JSON
    explore_resp = client.get("/explore?id=uuid-loinc-87279-6&entity_type=Concept&vocab=LOINC&code=87279-6")
    assert explore_resp.status_code == 200
    assert "LOINC:87279-6" in explore_resp.text
    assert "CT Chest for screening" in explore_resp.text
    assert "http://loinc.org|87279-6|CT Chest for screening" in explore_resp.text
    assert "SNOMED:169069000" in explore_resp.text
    assert "Crosswalks &amp; Lifecycle Mappings" in explore_resp.text
    assert 'id="entity-graph-data"' in explore_resp.text
    assert "Graph Neighborhood: 2 nodes" in explore_resp.text

    # 2. Search directly on /explore?q=...
    explore_search = client.get("/explore?q=chest+CT&scope=clinical")
    assert explore_search.status_code == 200
    assert "Search Results (1)" in explore_search.text
    assert "LOINC:87279-6" in explore_search.text

    # 3. Unknown entity returns clean 404 alert page
    missing_resp = client.get("/explore?id=missing-id")
    assert missing_resp.status_code == 404
    assert "was not found in the active Neo4j database" in missing_resp.text

    # 4. Verify /chat server-rendered NLU bar + warnings + related chips
    rich_resp = ChatResponse(
        reply="LOINC 87279-6 maps to SNOMED 169069000.",
        model="gemma4:e2b",
        provider="ollama",
        citations=[search_hit_to_citation(seed_hit)],
        nlu=ChatNLUSummary(
            intent="crosswalk",
            provider="gosmart-nlu",
            normalized_query="LOINC 87279-6",
            rewritten_query="CT Chest for screening",
            extracted_codes=["LOINC:87279-6"],
            target_vocabularies=["SNOMED"],
        ),
        warnings=["Schema 'RXNORM' is not loaded in active Neo4j."],
        search_duration_ms=11.2,
    )
    for target in ("gosmart_dictionary_service.web.router.chat_service",):
        monkeypatch.setattr(target, _RichChatStub(rich_resp))

    chat_csrf = _extract_csrf(client.get("/chat").text)
    rich_post = client.post(
        "/chat",
        data={"message": "Crosswalk LOINC 87279-6", "include_context": "on", "csrf_token": chat_csrf},
    )
    assert rich_post.status_code == 200
    assert "CROSSWALK" in rich_post.text
    assert "CT Chest for screening" in rich_post.text
    assert "RXNORM" in rich_post.text
    assert "SNOMED:169069000" in rich_post.text
