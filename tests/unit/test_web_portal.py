"""Modular unit tests for the multi-page server-rendered HTML web portal."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from tests.unit.test_auth_api import _patch_services


def _extract_csrf(html_text: str) -> str:
    match = re.search(r'name="csrf_token"\s+value="([^"]+)"', html_text)
    assert match is not None, "Expected hidden csrf_token input in HTML form"
    return match.group(1)


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


def test_unauthenticated_keys_and_chat_redirect_to_login(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_services(monkeypatch, dev_mode=True)

    keys_resp = client.get("/keys", follow_redirects=False)
    assert keys_resp.status_code == 303
    assert keys_resp.headers["location"] == "/login?next=/keys"

    chat_resp = client.get("/chat", follow_redirects=False)
    assert chat_resp.status_code == 303
    assert chat_resp.headers["location"] == "/login?next=/chat"


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

    # 9. Visit /chat and submit via HTML form fallback (works without JS) AND via /api/v1/chat cookie auth
    chat_page = client.get("/chat")
    assert chat_page.status_code == 200
    assert "Dictionary AI Chat" in chat_page.text
    chat_csrf = _extract_csrf(chat_page.text)

    chat_html_post = client.post(
        "/chat",
        data={"message": "What is (0018,0050)?", "include_context": "on", "csrf_token": chat_csrf},
    )
    assert chat_html_post.status_code == 200
    assert "SliceThickness" in chat_html_post.text

    chat_api_cookie = client.post(
        "/api/v1/chat",
        json={"message": "What is (0018,0050)?", "include_context": True},
    )
    assert chat_api_cookie.status_code == 200
    assert chat_api_cookie.json()["citations"][0]["identifier"] == "(0018,0050)"

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
