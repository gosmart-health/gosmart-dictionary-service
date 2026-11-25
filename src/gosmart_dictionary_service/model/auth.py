"""Graph entity models and API schemas for users, identities, authenticators, and groups."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from gosmart_dictionary_service.model.auditable import AuditableNode, AuditableRelation

_BASIC_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _validate_email_format(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Email must be a string.")
    normalized = value.strip().lower()
    if not _BASIC_EMAIL_RE.match(normalized):
        raise ValueError("Please provide a valid email address.")
    return normalized


class UserNode(AuditableNode):
    """Graph node representing an authenticated principal (human user or service account)."""

    __label__: ClassVar[str] = "User"

    display_name: str = Field(..., description="Full display name (e.g. 'Dr. Jane Doe')")
    primary_email: str = Field(..., description="Canonical primary email address")
    is_active: bool = Field(default=True, description="Whether the user account is active")
    is_shadow: bool = Field(
        default=False,
        description="Whether this is a shadow account provisioned automatically via OAuth without local password",
    )
    email_verified: bool = Field(
        default=False,
        description="Whether the registrant's email address has been verified via Brevo confirmation link",
    )
    verification_token_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hex digest of active email verification token (raw token is never stored)",
    )
    verification_expires_at: Optional[datetime] = Field(
        default=None,
        description="UTC expiration timestamp for the active email verification token",
    )
    terms_accepted_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when registrant accepted terminology terms of use",
    )
    tier: str = Field(
        default="PUBLIC",
        description="Access tier: PUBLIC | INTERNAL_SERVICE | ADMIN",
    )
    system_roles: List[str] = Field(
        default_factory=lambda: ["USER"],
        description="Global system roles (e.g. ['USER'], ['PLATFORM_ADMIN'])",
    )
    avatar_url: Optional[str] = Field(default=None, description="URL or avatar image endpoint")

    @field_validator("primary_email", mode="before")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower() if isinstance(v, str) else v

    def get_identity_subset(self) -> Dict[str, Any]:
        subset: Dict[str, Any] = {
            "entity": self.__label__,
            "primary_email": self.primary_email.lower().strip(),
        }
        if self.test_id:
            subset["test_id"] = self.test_id
        return subset


class IdentityNode(AuditableNode):
    """Graph node representing an external or local identity provider claim."""

    __label__: ClassVar[str] = "Identity"

    provider: str = Field(..., description="Identity provider name: 'google', 'local', 'github', 'saml_sso'")
    external_sub: str = Field(..., description="Unique provider subject/user identifier (e.g. Google sub)")
    identifier: str = Field(..., description="User login identifier or username")
    email: Optional[str] = Field(default=None, description="Provider-reported email address")
    email_verified: bool = Field(default=False, description="Whether the provider has verified the email")
    profile_metadata: Optional[str] = Field(
        default=None,
        description="JSON serialized extra claims from provider (avatar, given_name, etc.)",
    )

    @field_validator("provider", mode="before")
    @classmethod
    def normalize_provider(cls, v: str) -> str:
        return v.strip().lower() if isinstance(v, str) else v

    def get_identity_subset(self) -> Dict[str, Any]:
        subset: Dict[str, Any] = {
            "entity": self.__label__,
            "provider": self.provider.lower().strip(),
            "external_sub": self.external_sub.strip(),
        }
        if self.test_id:
            subset["test_id"] = self.test_id
        return subset


class AuthenticatorNode(AuditableNode):
    """Graph node representing a credential or verification factor."""

    __label__: ClassVar[str] = "Authenticator"

    authenticator_type: str = Field(
        ...,
        description="Type of authenticator: 'password', 'api_key', 'passkey', 'totp'",
    )
    secret_hash: str = Field(..., description="Argon2 hash for password or SHA-256 hash for API key")
    prefix: Optional[str] = Field(default=None, description="Visible key prefix (e.g. 'gs_dict_a1b2c3d4')")
    name: Optional[str] = Field(default=None, description="Human-readable label (e.g. 'MacBook Pro CLI', 'MCP Agent')")
    scopes: List[str] = Field(default_factory=lambda: ["*"], description="Authorized permission scopes")
    expires_at: Optional[datetime] = Field(default=None, description="Credential expiration timestamp")
    last_used_at: Optional[datetime] = Field(default=None, description="Timestamp of last successful authentication")
    is_revoked: bool = Field(default=False, description="Whether the credential has been soft-revoked")

    def get_identity_subset(self) -> Dict[str, Any]:
        subset: Dict[str, Any] = {
            "entity": self.__label__,
            "authenticator_type": self.authenticator_type,
            "secret_hash": self.secret_hash[:32],
        }
        if self.test_id:
            subset["test_id"] = self.test_id
        return subset


class GroupNode(AuditableNode):
    """Graph node representing an organization, health system, or platform root group."""

    __label__: ClassVar[str] = "Group"

    name: str = Field(..., description="Organization or group name (e.g. 'GoSmart Health')")
    code: str = Field(..., description="Unique alphanumeric code/slug (e.g. 'GOSMART_HEALTH')")
    group_type: str = Field(
        default="HEALTH_SYSTEM",
        description="Group type: HEALTH_SYSTEM | HOSPITAL | FACILITY | DEPARTMENT | CARE_TEAM | SYSTEM",
    )
    is_active: bool = Field(default=True, description="Whether the group is active")
    description: Optional[str] = Field(default=None, description="Detailed description")

    @field_validator("code", mode="before")
    @classmethod
    def normalize_code(cls, v: str) -> str:
        return v.strip().upper() if isinstance(v, str) else v

    def get_identity_subset(self) -> Dict[str, Any]:
        subset: Dict[str, Any] = {
            "entity": self.__label__,
            "code": self.code.upper().strip(),
        }
        if self.test_id:
            subset["test_id"] = self.test_id
        return subset

    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Serialize model fields into Neo4j-compatible primitives."""
        data = self.model_dump()
        for key, val in list(data.items()):
            if isinstance(val, datetime):
                data[key] = val.isoformat()
        return data


Group = GroupNode


class HasIdentityRelation(AuditableRelation):
    """Relationship connecting User -> Identity."""

    __type__: ClassVar[str] = "HAS_IDENTITY"


class UsesAuthenticatorRelation(AuditableRelation):
    """Relationship connecting Identity -> Authenticator."""

    __type__: ClassVar[str] = "USES_AUTHENTICATOR"


class HasAuthenticatorRelation(AuditableRelation):
    """Relationship connecting User -> Authenticator (direct API keys, Passkeys, TOTP)."""

    __type__: ClassVar[str] = "HAS_AUTHENTICATOR"


class MemberOfRelation(AuditableRelation):
    """Relationship connecting User -> Group with group-specific roles, permissions, and lifecycle status."""

    __type__: ClassVar[str] = "MEMBER_OF"

    roles: List[str] = Field(
        default_factory=lambda: ["Viewer"],
        description="Group-specific roles (e.g. ['Admin'], ['Viewer'], ['Editor'])",
    )
    permissions: List[str] = Field(
        default_factory=list,
        description="Fine-grained granular permissions within this group",
    )
    status: str = Field(
        default="ACTIVE",
        description="Affiliation lifecycle status: 'ACTIVE' | 'INACTIVE' | 'SUSPENDED' | 'PENDING'",
    )
    is_primary: bool = Field(default=False, description="Whether this is the user's primary/default group affiliation")


class BelongsToRelation(AuditableRelation):
    """Relationship connecting Root Group -> System or Child Group -> Parent Group."""

    __type__: ClassVar[str] = "BELONGS_TO"


# ---------------------------------------------------------------------------
# REST API Request / Response Schemas for Auth & API Key Management
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    """Payload for self-service registrant signup."""

    email: str = Field(..., description="Registrant email address")
    password: str = Field(..., min_length=8, max_length=128, description="Account password (minimum 8 characters)")
    display_name: Optional[str] = Field(default=None, max_length=120, description="Optional full display name")
    terms_accepted: bool = Field(default=False, description="Must be true to accept terminology terms of use")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return _validate_email_format(v)


class RegisterResponse(BaseModel):
    """Response returned after successful registration."""

    user_id: str
    email: str
    display_name: str
    email_verified: bool
    verification_delivery_mode: str = Field(
        description="Email delivery mode: 'brevo' or 'dev_local'",
    )
    message: str
    dev_verification_url: Optional[str] = Field(
        default=None,
        description="Populated ONLY in DEV_MODE when GOSMART_DS_BREVO_API_KEY is unset",
    )


class VerifyEmailRequest(BaseModel):
    """Payload for confirming a registrant email verification token."""

    token: str = Field(..., min_length=8, description="Raw email verification token")


class VerifyEmailResponse(BaseModel):
    """Response returned after email verification."""

    verified: bool
    email: str
    message: str


class LoginRequest(BaseModel):
    """Payload for email/username + password authentication."""

    email: str = Field(..., description="Registered email address or username")
    password: str = Field(..., min_length=1, description="Account password")


class UserProfileResponse(BaseModel):
    """Public profile of an authenticated user."""

    id: str
    email: str
    display_name: str
    email_verified: bool
    is_active: bool
    tier: str = "PUBLIC"
    system_roles: List[str] = Field(default_factory=lambda: ["USER"])


class TokenResponse(BaseModel):
    """JWT access token response returned on login."""

    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    user: UserProfileResponse


class ApiKeyCreateRequest(BaseModel):
    """Payload to issue a new X-API-Key credential."""

    name: str = Field(..., min_length=1, max_length=80, description="Human-readable label for the API key")
    expires_in_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=3650,
        description="Optional expiration window in days (null = non-expiring)",
    )
    scopes: Optional[List[str]] = Field(default=None, description="Optional permission scopes (default: ['*'])")


class ApiKeySummary(BaseModel):
    """Non-secret metadata for an issued API key."""

    id: str
    name: str
    prefix: str
    scopes: List[str] = Field(default_factory=lambda: ["*"])
    created_at: Optional[str] = None
    last_used_at: Optional[str] = None
    expires_at: Optional[str] = None
    is_revoked: bool = False


class ApiKeyCreateResponse(ApiKeySummary):
    """Response returned once upon API key creation, including the plaintext key."""

    raw_api_key: str = Field(..., description="Plaintext API key (displayed once; store securely)")


__all__ = [
    "UserNode",
    "IdentityNode",
    "AuthenticatorNode",
    "GroupNode",
    "Group",
    "HasIdentityRelation",
    "UsesAuthenticatorRelation",
    "HasAuthenticatorRelation",
    "MemberOfRelation",
    "BelongsToRelation",
    "RegisterRequest",
    "RegisterResponse",
    "VerifyEmailRequest",
    "VerifyEmailResponse",
    "LoginRequest",
    "UserProfileResponse",
    "TokenResponse",
    "ApiKeyCreateRequest",
    "ApiKeySummary",
    "ApiKeyCreateResponse",
]
