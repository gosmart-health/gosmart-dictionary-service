"""Optional live Neo4j integration test for registration, email verification, login, and API key lifecycle.

Follows AGENTS.md testing policy:
- Skips cleanly if local Neo4j is unreachable.
- Uses an isolated ``test_id`` so production data is never touched.
- Cleans up prior ``test_id`` artifacts **before** running the test (NOT after) so post-run
  graph state remains inspectable in Neo4j Browser.
"""

from __future__ import annotations

import pytest

from gosmart_dictionary_service.core.config import get_settings
from gosmart_dictionary_service.db.neo4j import Neo4jClient
from gosmart_dictionary_service.db.schema_loader import apply_schema_async
from gosmart_dictionary_service.service.system_service import SystemService
from gosmart_dictionary_service.service.user_service import (
    EmailNotVerifiedError,
    UserService,
)

AUTH_WEB_INTEGRATION_TEST_ID = "test_auth_web_integration"


def _get_live_client_or_skip() -> Neo4jClient:
    get_settings.cache_clear()
    settings = get_settings()
    client = Neo4jClient(settings=settings)
    try:
        driver = client.get_sync_driver()
        driver.verify_connectivity()
    except Exception as exc:
        if client._sync_driver is not None:
            client._sync_driver.close()
        pytest.skip(f"Live Neo4j at {settings.DB_CONNECTION_URI} is unavailable: {exc}")
    return client


@pytest.mark.anyio
async def test_user_registration_verification_and_api_key_lifecycle_in_neo4j() -> None:
    client = _get_live_client_or_skip()
    try:
        await client.connect()
        await apply_schema_async(client)

        sys_service = SystemService(client=client)
        # Clean up ONLY prior test data BEFORE running the test (leave post-test data for inspection)
        sys_service.cleanup_test_data_sync(AUTH_WEB_INTEGRATION_TEST_ID)

        svc = UserService(client=client, settings=client.settings)
        email = f"{AUTH_WEB_INTEGRATION_TEST_ID}@example.org"
        password = "IntegrationPassword123!"

        # 1. Register user (unverified)
        user, raw_token = await svc.register_user(
            email=email,
            password=password,
            display_name="Integration Registrant",
            terms_accepted=True,
            test_id=AUTH_WEB_INTEGRATION_TEST_ID,
            acted_by="pytest_auth_integration",
        )
        assert user["primary_email"] == email
        assert user["email_verified"] is False

        # 2. Unverified login fails with EmailNotVerifiedError
        with pytest.raises(EmailNotVerifiedError):
            await svc.authenticate_local(email, password, require_verified=True)

        # 3. Verify email token
        verified_user = await svc.verify_email_token(raw_token, acted_by="pytest_auth_integration")
        assert verified_user["email_verified"] is True

        # 4. Login now succeeds
        authed = await svc.authenticate_local(email, password, require_verified=True)
        assert authed is not None
        assert authed["id"] == user["id"]

        # 5. Create API key, authenticate with it, list it, then revoke it
        created_key = await svc.create_api_key(
            user_id=user["id"],
            name="Integration CLI Key",
            test_id=AUTH_WEB_INTEGRATION_TEST_ID,
            acted_by="pytest_auth_integration",
        )
        raw_api_key = created_key["raw_api_key"]
        key_id = created_key["id"]

        api_user = await svc.authenticate_api_key(raw_api_key)
        assert api_user is not None
        assert api_user["id"] == user["id"]

        keys = await svc.list_api_keys(user["id"])
        assert any(k["id"] == key_id and not k["is_revoked"] for k in keys)

        assert await svc.revoke_api_key(user["id"], key_id, acted_by="pytest_auth_integration") is True
        assert await svc.authenticate_api_key(raw_api_key) is None
    finally:
        await client.close()
        if client._sync_driver is not None:
            client._sync_driver.close()
            client._sync_driver = None
