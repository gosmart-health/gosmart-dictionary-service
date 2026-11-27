"""Unit tests for Auth models, security helpers, Brevo EmailService, and UserService."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx
import pytest
from pydantic import SecretStr

from gosmart_dictionary_service.core.security import (
    generate_api_key,
    generate_csrf_token,
    generate_verification_token,
    hash_password,
    hash_token,
    verify_api_key,
    verify_csrf_token,
)
from gosmart_dictionary_service.model.auth import AuthenticatorNode, UserNode
from gosmart_dictionary_service.service.email_service import (
    EmailConfigurationError,
    EmailDeliveryError,
    EmailService,
)
from gosmart_dictionary_service.service.user_service import (
    EmailAlreadyRegisteredError,
    EmailNotVerifiedError,
    InvalidVerificationTokenError,
    UserService,
)
from tests.conftest import make_test_settings


class _MockNeo4jClient:
    """Minimal scripted Neo4j async client for isolated UserService unit tests."""

    def __init__(self, responses: List[List[Dict[str, Any]]]) -> None:
        self._responses = list(responses)
        self.calls: List[tuple[str, Dict[str, Any]]] = []

    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        self.calls.append((query, dict(parameters or {})))
        if not self._responses:
            return []
        return self._responses.pop(0)


def test_user_and_authenticator_deterministic_ids() -> None:
    u1 = UserNode(display_name="Jane Doe", primary_email="  Jane.Doe@Example.ORG ")
    u2 = UserNode(display_name="Dr. Jane", primary_email="jane.doe@example.org")
    assert u1.primary_email == "jane.doe@example.org"
    assert u1.id == u2.id

    u_test = UserNode(display_name="Jane Doe", primary_email="jane.doe@example.org", test_id="unit_auth")
    assert u_test.id != u1.id

    raw_key, visible_prefix, key_hash = generate_api_key()
    assert raw_key.startswith(f"{visible_prefix}_")
    assert visible_prefix.startswith("gs_dict_")
    assert verify_api_key(raw_key, key_hash) is True
    assert verify_api_key(raw_key + "x", key_hash) is False

    a1 = AuthenticatorNode(authenticator_type="api_key", secret_hash=key_hash, prefix=visible_prefix, name="CLI")
    a2 = AuthenticatorNode(authenticator_type="api_key", secret_hash=key_hash, prefix=visible_prefix, name="CLI")
    assert a1.id == a2.id


def test_verification_token_and_csrf_helpers() -> None:
    raw, digest = generate_verification_token()
    assert raw != digest
    assert hash_token(raw) == digest

    csrf = generate_csrf_token()
    assert verify_csrf_token(csrf, csrf) is True
    assert verify_csrf_token(csrf, "different") is False
    assert verify_csrf_token(None, csrf) is False


@pytest.mark.anyio
async def test_email_service_dev_mode_local_fallback_when_brevo_key_unset() -> None:
    settings = make_test_settings(
        DEV_MODE=True,
        BREVO_API_KEY=None,
        PUBLIC_BASE_URL="http://localhost:8081/",
    )
    svc = EmailService(settings=settings)
    result = await svc.send_verification_email(
        to_email="dev@example.org",
        display_name="Dev User",
        raw_token="tok_abc123",
    )
    assert result.sent is False
    assert result.mode == "dev_local"
    assert result.verification_url == "http://localhost:8081/verify-email?token=tok_abc123"


@pytest.mark.anyio
async def test_email_service_fails_closed_in_prod_when_brevo_key_unset() -> None:
    settings = make_test_settings(DEV_MODE=False, BREVO_API_KEY=None)
    svc = EmailService(settings=settings)
    with pytest.raises(EmailConfigurationError):
        await svc.send_verification_email(
            to_email="prod@example.org",
            display_name="Prod User",
            raw_token="tok_abc123",
        )


@pytest.mark.anyio
async def test_email_service_dispatches_brevo_api_request() -> None:
    captured: Dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["api_key"] = request.headers.get("api-key")
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(201, json={"messageId": "<brevo-msg-123>"})

    settings = make_test_settings(
        DEV_MODE=False,
        BREVO_API_KEY=SecretStr("xkeysib-test-secret"),
        BREVO_SENDER_EMAIL="noreply@gosmart.health",
        BREVO_SENDER_NAME="GoSmart DS",
        PUBLIC_BASE_URL="https://dict.gosmart.health",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as http_client:
        svc = EmailService(settings=settings, http_client=http_client)
        result = await svc.send_verification_email(
            to_email="clinician@example.org",
            display_name="Dr. <Alice>",
            raw_token="raw_tok_999",
        )

    assert result.sent is True
    assert result.mode == "brevo"
    assert result.message_id == "<brevo-msg-123>"
    assert captured["url"] == "https://api.brevo.com/v3/smtp/email"
    assert captured["api_key"] == "xkeysib-test-secret"
    assert captured["body"]["sender"] == {"name": "GoSmart DS", "email": "noreply@gosmart.health"}
    assert captured["body"]["to"][0]["email"] == "clinician@example.org"
    assert "Dr. &lt;Alice&gt;" in captured["body"]["htmlContent"]
    assert "https://dict.gosmart.health/verify-email?token=raw_tok_999" in captured["body"]["htmlContent"]


@pytest.mark.anyio
async def test_email_service_raises_delivery_error_on_brevo_http_failure() -> None:
    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Unauthorized"})

    settings = make_test_settings(DEV_MODE=False, BREVO_API_KEY=SecretStr("bad-key"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as http_client:
        svc = EmailService(settings=settings, http_client=http_client)
        with pytest.raises(EmailDeliveryError):
            await svc.send_verification_email(
                to_email="user@example.org",
                display_name="User",
                raw_token="tok_fail",
            )


@pytest.mark.anyio
async def test_user_service_register_and_verify_flow() -> None:
    settings = make_test_settings(EMAIL_VERIFICATION_EXPIRE_HOURS=24)
    mock_client = _MockNeo4jClient(
        [
            [],  # get_user_by_email -> not found
            [
                {
                    "user": {
                        "id": "u-1",
                        "primary_email": "alice@example.org",
                        "display_name": "Alice",
                        "email_verified": False,
                        "is_active": True,
                    }
                }
            ],
        ]
    )
    svc = UserService(client=mock_client, settings=settings)  # type: ignore[arg-type]

    user, raw_token = await svc.register_user(
        email="Alice@Example.org",
        password="StrongPassword123!",
        display_name="Alice",
        terms_accepted=True,
    )
    assert user["primary_email"] == "alice@example.org"
    assert user["email_verified"] is False
    assert raw_token
    # Verify raw token was never passed directly into Cypher parameters
    merge_params = mock_client.calls[1][1]
    assert merge_params["user"]["verification_token_hash"] == hash_token(raw_token)
    assert merge_params["user"]["verification_token_hash"] != raw_token


@pytest.mark.anyio
async def test_user_service_register_rejects_duplicate_verified_and_unaccepted_terms() -> None:
    mock_client = _MockNeo4jClient(
        [
            [{"user": {"id": "u-1", "primary_email": "alice@example.org", "email_verified": True}}],
        ]
    )
    svc = UserService(client=mock_client, settings=make_test_settings())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="terms"):
        await svc.register_user(email="a@b.com", password="StrongPassword123!", terms_accepted=False)

    with pytest.raises(EmailAlreadyRegisteredError):
        await svc.register_user(email="alice@example.org", password="StrongPassword123!", terms_accepted=True)


@pytest.mark.anyio
async def test_user_service_verify_email_token_valid_and_expired() -> None:
    future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()

    valid_client = _MockNeo4jClient(
        [
            [{"user": {"id": "u-1", "primary_email": "alice@example.org", "verification_expires_at": future}}],
            [{"user": {"id": "u-1", "primary_email": "alice@example.org", "email_verified": True}}],
        ]
    )
    svc_valid = UserService(client=valid_client, settings=make_test_settings())  # type: ignore[arg-type]
    verified = await svc_valid.verify_email_token("tok_valid_123")
    assert verified["email_verified"] is True

    expired_client = _MockNeo4jClient(
        [
            [{"user": {"id": "u-1", "primary_email": "alice@example.org", "verification_expires_at": past}}],
        ]
    )
    svc_expired = UserService(client=expired_client, settings=make_test_settings())  # type: ignore[arg-type]
    with pytest.raises(InvalidVerificationTokenError, match="expired"):
        await svc_expired.verify_email_token("tok_expired")


@pytest.mark.anyio
async def test_user_service_authenticate_local_checks_password_and_verification() -> None:
    pwd_hash = hash_password("CorrectPassword123!")
    unverified_client = _MockNeo4jClient(
        [
            [
                {
                    "user": {
                        "id": "u-1",
                        "primary_email": "alice@example.org",
                        "email_verified": False,
                        "is_active": True,
                    },
                    "authenticator": {"id": "a-1", "secret_hash": pwd_hash},
                    "auth_id": "a-1",
                }
            ],
            [
                {
                    "user": {
                        "id": "u-1",
                        "primary_email": "alice@example.org",
                        "email_verified": False,
                        "is_active": True,
                    },
                    "authenticator": {"id": "a-1", "secret_hash": pwd_hash},
                    "auth_id": "a-1",
                }
            ],
        ]
    )
    svc = UserService(client=unverified_client, settings=make_test_settings())  # type: ignore[arg-type]

    # Wrong password returns None without leaking verification status
    assert await svc.authenticate_local("alice@example.org", "WrongPassword!") is None

    # Correct password on unverified account raises EmailNotVerifiedError
    with pytest.raises(EmailNotVerifiedError):
        await svc.authenticate_local("alice@example.org", "CorrectPassword123!")
