"""Service for managing registrant Users, Identities, Password Credentials, and API Keys in Neo4j."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from gosmart_dictionary_service.core.config import Settings, get_settings
from gosmart_dictionary_service.core.logging import get_logger
from gosmart_dictionary_service.core.security import (
    generate_api_key,
    generate_verification_token,
    hash_api_key,
    hash_password,
    hash_token,
    verify_password,
)
from gosmart_dictionary_service.db.neo4j import Neo4jClient, neo4j_client
from gosmart_dictionary_service.model.auth import (
    AuthenticatorNode,
    HasAuthenticatorRelation,
    HasIdentityRelation,
    IdentityNode,
    UserNode,
    UsesAuthenticatorRelation,
)

logger = get_logger("gosmart_ds.user_service")


class EmailAlreadyRegisteredError(ValueError):
    """Raised when attempting to register an email address that is already verified."""


class EmailNotVerifiedError(PermissionError):
    """Raised when a user enters valid credentials before verifying their email address."""


class InvalidVerificationTokenError(ValueError):
    """Raised when an email verification token is invalid, already used, or expired."""


def _clean_neo4j_value(val: Any) -> Any:
    """Normalize Neo4j-specific types (e.g. neo4j.time.DateTime) to standard Python types."""
    if hasattr(val, "to_native"):
        return val.to_native()
    if isinstance(val, dict):
        return {k: _clean_neo4j_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_clean_neo4j_value(v) for v in val]
    return val


def _to_iso_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val)


def _parse_utc_datetime(val: Any) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
    if isinstance(val, str) and val.strip():
        parsed = datetime.fromisoformat(val.strip().replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return None


class UserService:
    """Service for managing Users, Identities, Authenticators, and API Keys in Neo4j."""

    def __init__(
        self,
        client: Optional[Neo4jClient] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.client = client or neo4j_client
        self._settings = settings

    @property
    def settings(self) -> Settings:
        return self._settings or get_settings()

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
        """Register a new local registrant account with ``email_verified=False``.

        Returns ``(user_dict, raw_verification_token)``. If an unverified account with the same
        email already exists, refreshes its verification token and password cleanly; if an
        already-verified account exists, raises ``EmailAlreadyRegisteredError``.
        """
        if not terms_accepted:
            raise ValueError("You must accept the terminology terms of use to register.")

        normalized_email = email.strip().lower()
        clean_display_name = (display_name or "").strip() or normalized_email.split("@")[0]

        existing = await self.get_user_by_email(normalized_email)
        if existing and existing.get("email_verified") is True:
            raise EmailAlreadyRegisteredError("An account with this email address is already registered.")

        raw_token, token_hash = generate_verification_token()
        now_utc = datetime.now(timezone.utc)
        expires_at = now_utc + timedelta(hours=self.settings.EMAIL_VERIFICATION_EXPIRE_HOURS)
        pwd_hash = hash_password(password)

        user_node = UserNode(
            display_name=clean_display_name,
            primary_email=normalized_email,
            is_active=True,
            is_shadow=False,
            email_verified=False,
            verification_token_hash=token_hash,
            verification_expires_at=expires_at,
            terms_accepted_at=now_utc,
            tier="PUBLIC",
            system_roles=["USER"],
            test_id=test_id,
            ActedBy=acted_by,
        )

        identity_node = IdentityNode(
            provider="local",
            external_sub=normalized_email,
            identifier=normalized_email,
            email=normalized_email,
            email_verified=False,
            test_id=test_id,
            ActedBy=acted_by,
        )

        auth_node = AuthenticatorNode(
            authenticator_type="password",
            secret_hash=pwd_hash,
            name="Primary Password",
            test_id=test_id,
            ActedBy=acted_by,
        )

        rel_has_id = HasIdentityRelation(
            start_node_id=user_node.id,
            end_node_id=identity_node.id,
            test_id=test_id,
            ActedBy=acted_by,
        )
        rel_uses_auth = UsesAuthenticatorRelation(
            start_node_id=identity_node.id,
            end_node_id=auth_node.id,
            test_id=test_id,
            ActedBy=acted_by,
        )

        # If an unverified user already exists, soft-retire any prior password authenticators
        # before attaching the newly hashed password credential.
        if existing:
            retire_old_pwd_query = """
            MATCH (u:User {id: $user_id})-[:HAS_IDENTITY]->(i:Identity {provider: 'local'})
                  -[:USES_AUTHENTICATOR]->(old_a:Authenticator {authenticator_type: 'password'})
            WHERE old_a.id <> $new_auth_id
            SET old_a.Action = 'DELETE',
                old_a.is_revoked = true,
                old_a.UpdatedAt = datetime({timezone: 'UTC'})
            """
            await self.client.execute_query(
                retire_old_pwd_query,
                {"user_id": existing["id"], "new_auth_id": auth_node.id},
            )

        query = """
        MERGE (u:User {id: $user.id})
        ON CREATE SET u.display_name = $user.display_name,
                      u.primary_email = $user.primary_email,
                      u.is_active = true,
                      u.is_shadow = false,
                      u.email_verified = false,
                      u.verification_token_hash = $user.verification_token_hash,
                      u.verification_expires_at = datetime($user.verification_expires_at),
                      u.terms_accepted_at = datetime($user.terms_accepted_at),
                      u.tier = $user.tier,
                      u.system_roles = $user.system_roles,
                      u.test_id = $user.test_id,
                      u.Action = 'CREATE',
                      u.ActedAt = datetime({timezone: 'UTC'}),
                      u.ActedBy = $user.ActedBy,
                      u.CreatedAt = datetime({timezone: 'UTC'})
        ON MATCH SET  u.display_name = $user.display_name,
                      u.is_active = true,
                      u.email_verified = false,
                      u.verification_token_hash = $user.verification_token_hash,
                      u.verification_expires_at = datetime($user.verification_expires_at),
                      u.terms_accepted_at = datetime($user.terms_accepted_at),
                      u.test_id = coalesce($user.test_id, u.test_id),
                      u.Action = 'UPDATE',
                      u.ActedAt = datetime({timezone: 'UTC'}),
                      u.ActedBy = $user.ActedBy,
                      u.UpdatedAt = datetime({timezone: 'UTC'})

        MERGE (i:Identity {id: $identity.id})
        ON CREATE SET i.provider = $identity.provider,
                      i.external_sub = $identity.external_sub,
                      i.identifier = $identity.identifier,
                      i.email = $identity.email,
                      i.email_verified = false,
                      i.test_id = $identity.test_id,
                      i.Action = 'CREATE',
                      i.ActedAt = datetime({timezone: 'UTC'}),
                      i.ActedBy = $identity.ActedBy,
                      i.CreatedAt = datetime({timezone: 'UTC'})
        ON MATCH SET  i.email_verified = false,
                      i.test_id = coalesce($identity.test_id, i.test_id),
                      i.Action = 'UPDATE',
                      i.ActedAt = datetime({timezone: 'UTC'}),
                      i.ActedBy = $identity.ActedBy,
                      i.UpdatedAt = datetime({timezone: 'UTC'})

        MERGE (a:Authenticator {id: $auth.id})
        ON CREATE SET a.authenticator_type = $auth.authenticator_type,
                      a.secret_hash = $auth.secret_hash,
                      a.name = $auth.name,
                      a.scopes = $auth.scopes,
                      a.is_revoked = false,
                      a.test_id = $auth.test_id,
                      a.Action = 'CREATE',
                      a.ActedAt = datetime({timezone: 'UTC'}),
                      a.ActedBy = $auth.ActedBy,
                      a.CreatedAt = datetime({timezone: 'UTC'})
        ON MATCH SET  a.secret_hash = $auth.secret_hash,
                      a.is_revoked = false,
                      a.Action = 'UPDATE',
                      a.ActedAt = datetime({timezone: 'UTC'}),
                      a.ActedBy = $auth.ActedBy,
                      a.UpdatedAt = datetime({timezone: 'UTC'})

        MERGE (u)-[r_hi:HAS_IDENTITY {id: $rel_hi.id}]->(i)
        ON CREATE SET r_hi.test_id = $rel_hi.test_id,
                      r_hi.Action = 'CREATE',
                      r_hi.ActedAt = datetime({timezone: 'UTC'}),
                      r_hi.ActedBy = $rel_hi.ActedBy,
                      r_hi.CreatedAt = datetime({timezone: 'UTC'})

        MERGE (i)-[r_ua:USES_AUTHENTICATOR {id: $rel_ua.id}]->(a)
        ON CREATE SET r_ua.test_id = $rel_ua.test_id,
                      r_ua.Action = 'CREATE',
                      r_ua.ActedAt = datetime({timezone: 'UTC'}),
                      r_ua.ActedBy = $rel_ua.ActedBy,
                      r_ua.CreatedAt = datetime({timezone: 'UTC'})

        RETURN properties(u) AS user
        """
        params = {
            "user": {
                "id": user_node.id,
                "display_name": user_node.display_name,
                "primary_email": user_node.primary_email,
                "verification_token_hash": token_hash,
                "verification_expires_at": expires_at.isoformat(),
                "terms_accepted_at": now_utc.isoformat(),
                "tier": user_node.tier,
                "system_roles": user_node.system_roles,
                "test_id": user_node.test_id,
                "ActedBy": user_node.ActedBy,
            },
            "identity": {
                "id": identity_node.id,
                "provider": identity_node.provider,
                "external_sub": identity_node.external_sub,
                "identifier": identity_node.identifier,
                "email": identity_node.email,
                "test_id": identity_node.test_id,
                "ActedBy": identity_node.ActedBy,
            },
            "auth": {
                "id": auth_node.id,
                "authenticator_type": auth_node.authenticator_type,
                "secret_hash": auth_node.secret_hash,
                "name": auth_node.name,
                "scopes": auth_node.scopes,
                "test_id": auth_node.test_id,
                "ActedBy": auth_node.ActedBy,
            },
            "rel_hi": {
                "id": rel_has_id.id,
                "test_id": rel_has_id.test_id,
                "ActedBy": rel_has_id.ActedBy,
            },
            "rel_ua": {
                "id": rel_uses_auth.id,
                "test_id": rel_uses_auth.test_id,
                "ActedBy": rel_uses_auth.ActedBy,
            },
        }
        records = await self.client.execute_query(query, params)
        user_data = _clean_neo4j_value(records[0].get("user", records[0])) if records else user_node.model_dump()
        logger.info("user_registered_pending_verification", user_id=user_data.get("id"), email=normalized_email)
        return user_data, raw_token

    async def verify_email_token(
        self,
        raw_token: str,
        *,
        acted_by: str = "email_verification",
    ) -> Dict[str, Any]:
        """Confirm an email verification token and activate the registrant account."""
        cleaned_token = (raw_token or "").strip()
        if not cleaned_token:
            raise InvalidVerificationTokenError("Verification token is required.")

        token_hash = hash_token(cleaned_token)
        lookup_query = """
        MATCH (u:User {verification_token_hash: $token_hash})
        WHERE (u.Action IS NULL OR u.Action <> 'DELETE')
          AND u.is_active = true
        RETURN properties(u) AS user
        LIMIT 1
        """
        records = await self.client.execute_query(lookup_query, {"token_hash": token_hash})
        if not records:
            raise InvalidVerificationTokenError("Verification link is invalid or has already been used.")

        user_data = _clean_neo4j_value(records[0].get("user", records[0]))
        expires_at = _parse_utc_datetime(user_data.get("verification_expires_at"))
        if expires_at is not None and expires_at <= datetime.now(timezone.utc):
            raise InvalidVerificationTokenError("Verification link has expired. Please register again to resend.")

        activate_query = """
        MATCH (u:User {id: $user_id})
        SET u.email_verified = true,
            u.verification_token_hash = null,
            u.verification_expires_at = null,
            u.Action = 'UPDATE',
            u.ActedAt = datetime({timezone: 'UTC'}),
            u.ActedBy = $acted_by,
            u.UpdatedAt = datetime({timezone: 'UTC'})
        WITH u
        OPTIONAL MATCH (u)-[:HAS_IDENTITY]->(i:Identity {provider: 'local'})
        SET i.email_verified = true,
            i.UpdatedAt = datetime({timezone: 'UTC'})
        RETURN properties(u) AS user
        """
        updated_records = await self.client.execute_query(
            activate_query,
            {"user_id": user_data["id"], "acted_by": acted_by},
        )
        verified_user = (
            _clean_neo4j_value(updated_records[0].get("user", updated_records[0])) if updated_records else user_data
        )
        verified_user["email_verified"] = True
        logger.info("user_email_verified", user_id=verified_user.get("id"), email=verified_user.get("primary_email"))
        return verified_user

    async def create_local_user(
        self,
        username: str,
        email: str,
        password: str,
        display_name: Optional[str] = None,
        system_roles: Optional[List[str]] = None,
        email_verified: bool = True,
        tier: str = "PUBLIC",
        test_id: Optional[str] = None,
        acted_by: str = "system",
    ) -> Dict[str, Any]:
        """Create a local user with password authenticator (used by admin/CLI bootstrap)."""
        normalized_email = email.strip().lower()
        normalized_username = username.strip()
        user_name = display_name or normalized_username
        roles = system_roles or ["USER"]

        user_node = UserNode(
            display_name=user_name,
            primary_email=normalized_email,
            is_active=True,
            is_shadow=False,
            email_verified=email_verified,
            tier=tier,
            system_roles=roles,
            test_id=test_id,
            ActedBy=acted_by,
        )

        identity_node = IdentityNode(
            provider="local",
            external_sub=normalized_username,
            identifier=normalized_username,
            email=normalized_email,
            email_verified=email_verified,
            test_id=test_id,
            ActedBy=acted_by,
        )

        pwd_hash = hash_password(password)
        auth_node = AuthenticatorNode(
            authenticator_type="password",
            secret_hash=pwd_hash,
            name="Primary Password",
            test_id=test_id,
            ActedBy=acted_by,
        )

        rel_has_id = HasIdentityRelation(
            start_node_id=user_node.id,
            end_node_id=identity_node.id,
            test_id=test_id,
            ActedBy=acted_by,
        )
        rel_uses_auth = UsesAuthenticatorRelation(
            start_node_id=identity_node.id,
            end_node_id=auth_node.id,
            test_id=test_id,
            ActedBy=acted_by,
        )

        query = """
        MERGE (u:User {id: $user.id})
        ON CREATE SET u.display_name = $user.display_name,
                      u.primary_email = $user.primary_email,
                      u.is_active = $user.is_active,
                      u.is_shadow = $user.is_shadow,
                      u.email_verified = $user.email_verified,
                      u.tier = $user.tier,
                      u.system_roles = $user.system_roles,
                      u.test_id = $user.test_id,
                      u.Action = 'CREATE',
                      u.ActedAt = datetime({timezone: 'UTC'}),
                      u.ActedBy = $user.ActedBy,
                      u.CreatedAt = datetime({timezone: 'UTC'})
        ON MATCH SET  u.display_name = $user.display_name,
                      u.is_active = $user.is_active,
                      u.is_shadow = $user.is_shadow,
                      u.email_verified = $user.email_verified,
                      u.tier = $user.tier,
                      u.system_roles = $user.system_roles,
                      u.test_id = coalesce($user.test_id, u.test_id),
                      u.Action = 'UPDATE',
                      u.ActedAt = datetime({timezone: 'UTC'}),
                      u.ActedBy = $user.ActedBy,
                      u.UpdatedAt = datetime({timezone: 'UTC'})

        MERGE (i:Identity {id: $identity.id})
        ON CREATE SET i.provider = $identity.provider,
                      i.external_sub = $identity.external_sub,
                      i.identifier = $identity.identifier,
                      i.email = $identity.email,
                      i.email_verified = $identity.email_verified,
                      i.test_id = $identity.test_id,
                      i.Action = 'CREATE',
                      i.ActedAt = datetime({timezone: 'UTC'}),
                      i.ActedBy = $identity.ActedBy,
                      i.CreatedAt = datetime({timezone: 'UTC'})

        MERGE (a:Authenticator {id: $auth.id})
        ON CREATE SET a.authenticator_type = $auth.authenticator_type,
                      a.secret_hash = $auth.secret_hash,
                      a.name = $auth.name,
                      a.scopes = $auth.scopes,
                      a.is_revoked = false,
                      a.test_id = $auth.test_id,
                      a.Action = 'CREATE',
                      a.ActedAt = datetime({timezone: 'UTC'}),
                      a.ActedBy = $auth.ActedBy,
                      a.CreatedAt = datetime({timezone: 'UTC'})

        MERGE (u)-[r_hi:HAS_IDENTITY {id: $rel_hi.id}]->(i)
        ON CREATE SET r_hi.test_id = $rel_hi.test_id,
                      r_hi.Action = 'CREATE',
                      r_hi.ActedAt = datetime({timezone: 'UTC'}),
                      r_hi.ActedBy = $rel_hi.ActedBy,
                      r_hi.CreatedAt = datetime({timezone: 'UTC'})

        MERGE (i)-[r_ua:USES_AUTHENTICATOR {id: $rel_ua.id}]->(a)
        ON CREATE SET r_ua.test_id = $rel_ua.test_id,
                      r_ua.Action = 'CREATE',
                      r_ua.ActedAt = datetime({timezone: 'UTC'}),
                      r_ua.ActedBy = $rel_ua.ActedBy,
                      r_ua.CreatedAt = datetime({timezone: 'UTC'})

        RETURN properties(u) AS user
        """
        params = {
            "user": {
                "id": user_node.id,
                "display_name": user_node.display_name,
                "primary_email": user_node.primary_email,
                "is_active": user_node.is_active,
                "is_shadow": user_node.is_shadow,
                "email_verified": user_node.email_verified,
                "tier": user_node.tier,
                "system_roles": user_node.system_roles,
                "test_id": user_node.test_id,
                "ActedBy": user_node.ActedBy,
            },
            "identity": {
                "id": identity_node.id,
                "provider": identity_node.provider,
                "external_sub": identity_node.external_sub,
                "identifier": identity_node.identifier,
                "email": identity_node.email,
                "email_verified": identity_node.email_verified,
                "test_id": identity_node.test_id,
                "ActedBy": identity_node.ActedBy,
            },
            "auth": {
                "id": auth_node.id,
                "authenticator_type": auth_node.authenticator_type,
                "secret_hash": auth_node.secret_hash,
                "name": auth_node.name,
                "scopes": auth_node.scopes,
                "test_id": auth_node.test_id,
                "ActedBy": auth_node.ActedBy,
            },
            "rel_hi": {
                "id": rel_has_id.id,
                "test_id": rel_has_id.test_id,
                "ActedBy": rel_has_id.ActedBy,
            },
            "rel_ua": {
                "id": rel_uses_auth.id,
                "test_id": rel_uses_auth.test_id,
                "ActedBy": rel_uses_auth.ActedBy,
            },
        }
        records = await self.client.execute_query(query, params)
        return _clean_neo4j_value(records[0].get("user", records[0])) if records else {}

    async def authenticate_local(
        self,
        identifier: str,
        password: str,
        *,
        require_verified: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Authenticate user by local email or username and password.

        Returns ``None`` on invalid credentials, or raises ``EmailNotVerifiedError`` when the
        password matches an account whose email address has not been verified yet.
        """
        norm_identifier = identifier.strip()
        norm_email = identifier.strip().lower()

        query = """
        MATCH (u:User)-[:HAS_IDENTITY]->(i:Identity {provider: 'local'})
              -[:USES_AUTHENTICATOR]->(a:Authenticator {authenticator_type: 'password'})
        WHERE (i.identifier = $identifier OR toLower(u.primary_email) = $email)
          AND u.is_active = true
          AND (u.Action IS NULL OR u.Action <> 'DELETE')
          AND (a.Action IS NULL OR a.Action <> 'DELETE')
          AND coalesce(a.is_revoked, false) = false
        RETURN properties(u) AS user, properties(a) AS authenticator, a.id AS auth_id
        ORDER BY a.CreatedAt DESC
        LIMIT 1
        """
        records = await self.client.execute_query(query, {"identifier": norm_identifier, "email": norm_email})
        if not records:
            return None

        record = records[0]
        user_data = _clean_neo4j_value(record.get("user"))
        auth_data = _clean_neo4j_value(record.get("authenticator"))

        if not user_data or not auth_data or not auth_data.get("secret_hash"):
            return None

        if not verify_password(password, auth_data["secret_hash"]):
            return None

        if require_verified and user_data.get("email_verified") is False:
            raise EmailNotVerifiedError("Please verify your email address before logging in.")

        auth_id = record.get("auth_id") or auth_data.get("id")
        if auth_id:
            update_query = """
            MATCH (a:Authenticator {id: $auth_id})
            SET a.last_used_at = datetime({timezone: 'UTC'})
            """
            await self.client.execute_query(update_query, {"auth_id": auth_id})

        return user_data

    async def create_api_key(
        self,
        user_id: str,
        name: str,
        scopes: Optional[List[str]] = None,
        expires_at: Optional[datetime] = None,
        test_id: Optional[str] = None,
        acted_by: str = "system",
    ) -> Dict[str, Any]:
        """Generate and attach a secure API Key Authenticator directly to a User."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("API key name cannot be empty.")

        raw_key, visible_prefix, key_hash = generate_api_key()
        scopes_list = scopes or ["*"]

        auth_node = AuthenticatorNode(
            authenticator_type="api_key",
            secret_hash=key_hash,
            prefix=visible_prefix,
            name=clean_name,
            scopes=scopes_list,
            expires_at=expires_at,
            is_revoked=False,
            test_id=test_id,
            ActedBy=acted_by,
        )

        rel = HasAuthenticatorRelation(
            start_node_id=user_id,
            end_node_id=auth_node.id,
            test_id=test_id,
            ActedBy=acted_by,
        )

        query = """
        MATCH (u:User {id: $user_id})
        WHERE (u.Action IS NULL OR u.Action <> 'DELETE')
        MERGE (a:Authenticator {id: $auth.id})
        ON CREATE SET a.authenticator_type = 'api_key',
                      a.secret_hash = $auth.secret_hash,
                      a.prefix = $auth.prefix,
                      a.name = $auth.name,
                      a.scopes = $auth.scopes,
                      a.expires_at = $auth.expires_at,
                      a.is_revoked = false,
                      a.test_id = $auth.test_id,
                      a.Action = 'CREATE',
                      a.ActedAt = datetime({timezone: 'UTC'}),
                      a.ActedBy = $acted_by,
                      a.CreatedAt = datetime({timezone: 'UTC'})

        MERGE (u)-[r:HAS_AUTHENTICATOR {id: $rel_id}]->(a)
        ON CREATE SET r.test_id = $rel_test_id,
                      r.Action = 'CREATE',
                      r.ActedAt = datetime({timezone: 'UTC'}),
                      r.ActedBy = $acted_by,
                      r.CreatedAt = datetime({timezone: 'UTC'})

        RETURN properties(a) AS authenticator
        """
        params = {
            "user_id": user_id,
            "auth": {
                "id": auth_node.id,
                "secret_hash": auth_node.secret_hash,
                "prefix": auth_node.prefix,
                "name": auth_node.name,
                "scopes": auth_node.scopes,
                "expires_at": auth_node.expires_at.isoformat() if auth_node.expires_at else None,
                "test_id": test_id,
            },
            "rel_id": rel.id,
            "rel_test_id": test_id,
            "acted_by": acted_by,
        }
        records = await self.client.execute_query(query, params)
        created_props = _clean_neo4j_value(records[0].get("authenticator", {})) if records else {}
        created_at = _to_iso_str(created_props.get("CreatedAt")) or datetime.now(timezone.utc).isoformat()

        return {
            "id": auth_node.id,
            "authenticator_id": auth_node.id,
            "raw_api_key": raw_key,
            "prefix": visible_prefix,
            "name": clean_name,
            "scopes": scopes_list,
            "created_at": created_at,
            "last_used_at": None,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "is_revoked": False,
        }

    async def list_api_keys(
        self,
        user_id: str,
        *,
        include_revoked: bool = False,
    ) -> List[Dict[str, Any]]:
        """List non-secret API key metadata owned by ``user_id``."""
        query = """
        MATCH (u:User {id: $user_id})-[:HAS_AUTHENTICATOR]->(a:Authenticator {authenticator_type: 'api_key'})
        WHERE (
            $include_revoked = true
            OR ((a.Action IS NULL OR a.Action <> 'DELETE') AND coalesce(a.is_revoked, false) = false)
        )
        RETURN properties(a) AS authenticator
        ORDER BY a.CreatedAt DESC
        """
        records = await self.client.execute_query(
            query,
            {"user_id": user_id, "include_revoked": include_revoked},
        )
        items: List[Dict[str, Any]] = []
        for rec in records:
            props = _clean_neo4j_value(rec.get("authenticator", rec))
            is_revoked = bool(props.get("is_revoked")) or props.get("Action") == "DELETE"
            items.append(
                {
                    "id": props.get("id", ""),
                    "name": props.get("name") or "API Key",
                    "prefix": props.get("prefix") or self.settings.API_KEY_PREFIX,
                    "scopes": list(props.get("scopes") or ["*"]),
                    "created_at": _to_iso_str(props.get("CreatedAt")),
                    "last_used_at": _to_iso_str(props.get("last_used_at")),
                    "expires_at": _to_iso_str(props.get("expires_at")),
                    "is_revoked": is_revoked,
                }
            )
        return items

    async def revoke_api_key(
        self,
        user_id: str,
        key_id: str,
        *,
        acted_by: str = "user",
    ) -> bool:
        """Non-destructively revoke an API key belonging to ``user_id``."""
        query = """
        MATCH (u:User {id: $user_id})-[:HAS_AUTHENTICATOR]->(
            a:Authenticator {id: $key_id, authenticator_type: 'api_key'}
        )
        WHERE (a.Action IS NULL OR a.Action <> 'DELETE')
          AND coalesce(a.is_revoked, false) = false
        SET a.is_revoked = true,
            a.Action = 'DELETE',
            a.ActedAt = datetime({timezone: 'UTC'}),
            a.ActedBy = $acted_by,
            a.UpdatedAt = datetime({timezone: 'UTC'})
        RETURN a.id AS id
        """
        records = await self.client.execute_query(
            query,
            {"user_id": user_id, "key_id": key_id, "acted_by": acted_by},
        )
        revoked = bool(records)
        if revoked:
            logger.info("api_key_revoked", user_id=user_id, key_id=key_id, acted_by=acted_by)
        return revoked

    async def authenticate_api_key(self, raw_api_key: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user by incoming ``X-API-Key`` header value."""
        cleaned_key = (raw_api_key or "").strip()
        if not cleaned_key:
            return None

        key_hash = hash_api_key(cleaned_key)
        query = """
        MATCH (u:User)-[:HAS_AUTHENTICATOR]->(a:Authenticator {authenticator_type: 'api_key', secret_hash: $key_hash})
        WHERE u.is_active = true
          AND coalesce(u.email_verified, true) = true
          AND (u.Action IS NULL OR u.Action <> 'DELETE')
          AND (a.Action IS NULL OR a.Action <> 'DELETE')
          AND coalesce(a.is_revoked, false) = false
          AND (a.expires_at IS NULL OR datetime(a.expires_at) > datetime({timezone: 'UTC'}))
        SET a.last_used_at = datetime({timezone: 'UTC'})
        RETURN properties(u) AS user, properties(a) AS authenticator
        LIMIT 1
        """
        records = await self.client.execute_query(query, {"key_hash": key_hash})
        if not records:
            return None

        record = records[0]
        user_data = _clean_neo4j_value(record.get("user"))
        auth_data = _clean_neo4j_value(record.get("authenticator"))
        if not user_data:
            return None

        user_data["api_key_scopes"] = (auth_data or {}).get("scopes", ["*"])
        user_data["api_key_name"] = (auth_data or {}).get("name")
        user_data["api_key_id"] = (auth_data or {}).get("id")
        return user_data

    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetch an active user by deterministic ID."""
        query = """
        MATCH (u:User {id: $user_id})
        WHERE (u.Action IS NULL OR u.Action <> 'DELETE')
        RETURN properties(u) AS user
        LIMIT 1
        """
        records = await self.client.execute_query(query, {"user_id": user_id})
        if records:
            return _clean_neo4j_value(records[0].get("user", records[0]))
        return None

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Fetch an active user by normalized primary email."""
        norm_email = email.strip().lower()
        query = """
        MATCH (u:User {primary_email: $email})
        WHERE (u.Action IS NULL OR u.Action <> 'DELETE')
        RETURN properties(u) AS user
        LIMIT 1
        """
        records = await self.client.execute_query(query, {"email": norm_email})
        if records:
            return _clean_neo4j_value(records[0].get("user", records[0]))
        return None

    async def list_users(
        self,
        include_inactive: bool = False,
        include_tests: bool = True,
        test_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List users."""
        query = """
        MATCH (u:User)
        WHERE (u.Action IS NULL OR u.Action <> 'DELETE')
          AND ($include_inactive = true OR u.is_active = true)
          AND ($include_tests = true OR u.test_id IS NULL OR u.test_id = '')
          AND ($test_id IS NULL OR u.test_id = $test_id)
        RETURN properties(u) AS user
        ORDER BY u.display_name ASC
        LIMIT $limit
        """
        records = await self.client.execute_query(
            query,
            {
                "include_inactive": include_inactive,
                "include_tests": include_tests,
                "test_id": test_id,
                "limit": limit,
            },
        )
        return [_clean_neo4j_value(r.get("user", r)) for r in records]


user_service = UserService()

__all__ = [
    "EmailAlreadyRegisteredError",
    "EmailNotVerifiedError",
    "InvalidVerificationTokenError",
    "UserService",
    "user_service",
]
