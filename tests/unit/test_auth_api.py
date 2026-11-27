"""Modular unit tests for REST /api/v1/auth/* and /api/v1/chat endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytest
from fastapi.testclient import TestClient

from gosmart_dictionary_service.core.security import (
    generate_api_key,
    generate_verification_token,
    hash_api_key,
    hash_password,
    hash_token,
    verify_password,
)
from gosmart_dictionary_service.model.auth import UserNode
from gosmart_dictionary_service.model.llm_config import LLM_Config
from gosmart_dictionary_service.service.chat_service import (
    ChatCitation,
    DictionaryChatService,
)
from gosmart_dictionary_service.service.email_service import EmailService
from gosmart_dictionary_service.service.user_service import (
    EmailAlreadyRegisteredError,
    EmailNotVerifiedError,
    InvalidVerificationTokenError,
)
from tests.conftest import make_test_settings


class InMemoryUserService:
    """Isolated per-test in-memory UserService double implementing the exact service contract."""

    def __init__(self) -> None:
        self.users_by_id: Dict[str, Dict[str, Any]] = {}
        self.passwords_by_uid: Dict[str, str] = {}
        self.token_hash_to_uid: Dict[str, str] = {}
        self.api_keys_by_id: Dict[str, Dict[str, Any]] = {}

    async def register_user(
        self,
        *,
        email: str,
        password: str,
        display_name: Optional[str] = None,
        terms_accepted: bool = False,
        test_id: Optional[str] = None,
        acted_by: str = "self_register",
    ) -> Tuple[Dict[str, Any], str]:
        if not terms_accepted:
            raise ValueError("You must accept the terminology terms of use to register.")
        norm_email = email.strip().lower()
        existing = await self.get_user_by_email(norm_email)
        if existing and existing.get("email_verified"):
            raise EmailAlreadyRegisteredError("An account with this email address is already registered.")

        node = UserNode(
            display_name=(display_name or "").strip() or norm_email.split("@")[0],
            primary_email=norm_email,
            is_active=True,
            email_verified=False,
            test_id=test_id,
            ActedBy=acted_by,
        )
        raw_token, token_hash = generate_verification_token()
        user_dict = {
            "id": node.id,
            "primary_email": node.primary_email,
            "display_name": node.display_name,
            "is_active": True,
            "email_verified": False,
            "tier": "PUBLIC",
            "system_roles": ["USER"],
        }
        self.users_by_id[node.id] = user_dict
        self.passwords_by_uid[node.id] = hash_password(password)
        self.token_hash_to_uid[token_hash] = node.id
        return dict(user_dict), raw_token

    async def verify_email_token(self, raw_token: str, *, acted_by: str = "email_verification") -> Dict[str, Any]:
        digest = hash_token((raw_token or "").strip())
        uid = self.token_hash_to_uid.pop(digest, None)
        if not uid or uid not in self.users_by_id:
            raise InvalidVerificationTokenError("Verification link is invalid or has already been used.")
        self.users_by_id[uid]["email_verified"] = True
        return dict(self.users_by_id[uid])

    async def authenticate_local(
        self,
        identifier: str,
        password: str,
        *,
        require_verified: bool = True,
    ) -> Optional[Dict[str, Any]]:
        user = await self.get_user_by_email(identifier)
        if not user:
            return None
        pwd_hash = self.passwords_by_uid.get(user["id"], "")
        if not verify_password(password, pwd_hash):
            return None
        if require_verified and not user.get("email_verified"):
            raise EmailNotVerifiedError("Please verify your email address before logging in.")
        return dict(user)

    async def create_api_key(
        self,
        user_id: str,
        name: str,
        scopes: Optional[List[str]] = None,
        expires_at: Optional[datetime] = None,
        test_id: Optional[str] = None,
        acted_by: str = "system",
    ) -> Dict[str, Any]:
        raw_key, visible_prefix, key_hash = generate_api_key()
        key_id = f"key-{len(self.api_keys_by_id) + 1}"
        now_iso = datetime.now(timezone.utc).isoformat()
        rec = {
            "id": key_id,
            "user_id": user_id,
            "name": name.strip(),
            "prefix": visible_prefix,
            "secret_hash": key_hash,
            "scopes": scopes or ["*"],
            "created_at": now_iso,
            "last_used_at": None,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "is_revoked": False,
        }
        self.api_keys_by_id[key_id] = rec
        return {**rec, "raw_api_key": raw_key}

    async def list_api_keys(self, user_id: str, *, include_revoked: bool = False) -> List[Dict[str, Any]]:
        out = []
        for rec in self.api_keys_by_id.values():
            if rec["user_id"] != user_id:
                continue
            if not include_revoked and rec["is_revoked"]:
                continue
            out.append({k: v for k, v in rec.items() if k not in {"secret_hash", "user_id"}})
        return list(reversed(out))

    async def revoke_api_key(self, user_id: str, key_id: str, *, acted_by: str = "user") -> bool:
        rec = self.api_keys_by_id.get(key_id)
        if not rec or rec["user_id"] != user_id or rec["is_revoked"]:
            return False
        rec["is_revoked"] = True
        return True

    async def authenticate_api_key(self, raw_api_key: str) -> Optional[Dict[str, Any]]:
        digest = hash_api_key(raw_api_key.strip())
        for rec in self.api_keys_by_id.values():
            if rec["secret_hash"] == digest and not rec["is_revoked"]:
                user = self.users_by_id.get(rec["user_id"])
                if user and user.get("is_active") and user.get("email_verified"):
                    rec["last_used_at"] = datetime.now(timezone.utc).isoformat()
                    return dict(user)
        return None

    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        user = self.users_by_id.get(user_id)
        return dict(user) if user else None

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        norm = email.strip().lower()
        for user in self.users_by_id.values():
            if user["primary_email"] == norm:
                return dict(user)
        return None


def _patch_services(monkeypatch: pytest.MonkeyPatch, dev_mode: bool = True) -> InMemoryUserService:
    fake_users = InMemoryUserService()
    test_settings = make_test_settings(DEV_MODE=dev_mode, BREVO_API_KEY=None)
    fake_email = EmailService(settings=test_settings)
    fake_chat = DictionaryChatService(
        settings=test_settings,
        search_fn=lambda _q, _lim: [
            ChatCitation(
                entity_type="DicomElement",
                identifier="(0018,0050)",
                name="Slice Thickness",
                keyword="SliceThickness",
                vr="DS",
                vm="1",
                source="exact",
            )
        ],
        generate_fn=lambda _sys, _prompt, cfg: f"Mocked reply from {cfg.model}: (0018,0050) SliceThickness is VR=DS.",
    )
    monkeypatch.setattr(
        fake_chat,
        "resolve_inference_config",
        lambda **_kw: _async_val(LLM_Config(tag="inference-v1", provider="ollama", model="gemma4:e2b")),
    )

    for target in (
        "gosmart_dictionary_service.api.v1.endpoints.auth.user_service",
        "gosmart_dictionary_service.core.auth_dependency.user_service",
        "gosmart_dictionary_service.web.router.user_service",
    ):
        monkeypatch.setattr(target, fake_users)

    for target in (
        "gosmart_dictionary_service.api.v1.endpoints.auth.email_service",
        "gosmart_dictionary_service.web.router.email_service",
    ):
        monkeypatch.setattr(target, fake_email)

    for target in (
        "gosmart_dictionary_service.api.v1.endpoints.chat.chat_service",
        "gosmart_dictionary_service.web.router.chat_service",
    ):
        monkeypatch.setattr(target, fake_chat)

    return fake_users


async def _async_val(val: Any) -> Any:
    return val


def test_rest_auth_register_verify_login_keys_and_chat_flow(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_services(monkeypatch, dev_mode=True)

    # 1. Register fails if terms not accepted
    bad_reg = client.post(
        "/api/v1/auth/register",
        json={"email": "doc@example.org", "password": "StrongPassword123!", "terms_accepted": False},
    )
    assert bad_reg.status_code == 400

    # 2. Register succeeds in DEV_MODE and returns local verification link
    reg = client.post(
        "/api/v1/auth/register",
        json={
            "email": "doc@example.org",
            "password": "StrongPassword123!",
            "display_name": "Dr. House",
            "terms_accepted": True,
        },
    )
    assert reg.status_code == 201
    reg_body = reg.json()
    assert reg_body["email"] == "doc@example.org"
    assert reg_body["email_verified"] is False
    assert reg_body["verification_delivery_mode"] == "dev_local"
    assert reg_body["dev_verification_url"]
    raw_token = reg_body["dev_verification_url"].split("token=")[1]

    # 3. Login blocked until email is verified
    unverified_login = client.post(
        "/api/v1/auth/login",
        json={"email": "doc@example.org", "password": "StrongPassword123!"},
    )
    assert unverified_login.status_code == 403

    # 4. Verify email token
    verify_resp = client.post("/api/v1/auth/verify-email", json={"token": raw_token})
    assert verify_resp.status_code == 200
    assert verify_resp.json()["verified"] is True

    # 5. Duplicate registration of verified email returns 409
    dup = client.post(
        "/api/v1/auth/register",
        json={"email": "doc@example.org", "password": "StrongPassword123!", "terms_accepted": True},
    )
    assert dup.status_code == 409

    # 6. Login succeeds and returns JWT access token
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "doc@example.org", "password": "StrongPassword123!"},
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    bearer_headers = {"Authorization": f"Bearer {token}"}

    # 7. GET /me with JWT Bearer
    me_resp = client.get("/api/v1/auth/me", headers=bearer_headers)
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == "doc@example.org"

    # 8. Issue an API key, verify plaintext shown once and omitted from list
    key_resp = client.post(
        "/api/v1/auth/keys",
        json={"name": "MCP Integration", "expires_in_days": 30},
        headers=bearer_headers,
    )
    assert key_resp.status_code == 201
    key_data = key_resp.json()
    raw_api_key = key_data["raw_api_key"]
    key_id = key_data["id"]
    assert raw_api_key.startswith(f"{key_data['prefix']}_")

    list_resp = client.get("/api/v1/auth/keys", headers=bearer_headers)
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1
    assert "raw_api_key" not in list_resp.json()[0]

    # 9. Unauthenticated chat rejected; X-API-Key chat succeeds with grounded citations
    assert client.post("/api/v1/chat", json={"message": "What is (0018,0050)?"}).status_code == 401
    chat_resp = client.post(
        "/api/v1/chat",
        json={"message": "What is (0018,0050)?", "include_context": True},
        headers={"X-API-Key": raw_api_key},
    )
    assert chat_resp.status_code == 200
    chat_body = chat_resp.json()
    assert "(0018,0050)" in chat_body["reply"]
    assert chat_body["citations"][0]["keyword"] == "SliceThickness"

    # 10. Revoke API key and verify it can no longer authenticate
    revoke_resp = client.delete(f"/api/v1/auth/keys/{key_id}", headers=bearer_headers)
    assert revoke_resp.status_code == 200
    assert revoke_resp.json()["revoked"] is True

    revoked_chat = client.post(
        "/api/v1/chat",
        json={"message": "Should fail"},
        headers={"X-API-Key": raw_api_key},
    )
    assert revoked_chat.status_code == 401


def test_rest_register_returns_503_in_prod_when_brevo_key_unset(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_services(monkeypatch, dev_mode=False)
    resp = client.post(
        "/api/v1/auth/register",
        json={"email": "prod@example.org", "password": "StrongPassword123!", "terms_accepted": True},
    )
    assert resp.status_code == 503
    assert "GOSMART_DS_BREVO_API_KEY" in resp.json()["detail"]
