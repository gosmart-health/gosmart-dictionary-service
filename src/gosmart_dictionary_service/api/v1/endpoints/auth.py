"""REST API endpoints for registrant account registration, email verification, login, and API keys."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, status

from gosmart_dictionary_service.core.auth_dependency import get_current_user
from gosmart_dictionary_service.core.config import get_settings
from gosmart_dictionary_service.core.security import create_access_token
from gosmart_dictionary_service.model.auth import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeySummary,
    LoginRequest,
    RegisterRequest,
    RegisterResponse,
    TokenResponse,
    UserProfileResponse,
    VerifyEmailRequest,
    VerifyEmailResponse,
)
from gosmart_dictionary_service.service.email_service import (
    EmailConfigurationError,
    EmailDeliveryError,
    email_service,
)
from gosmart_dictionary_service.service.user_service import (
    EmailAlreadyRegisteredError,
    EmailNotVerifiedError,
    InvalidVerificationTokenError,
    user_service,
)

router = APIRouter(prefix="/auth", tags=["Auth & API Keys"])


def _to_user_profile(user: Dict[str, Any]) -> UserProfileResponse:
    return UserProfileResponse(
        id=str(user["id"]),
        email=str(user.get("primary_email", "")),
        display_name=str(user.get("display_name", "")),
        email_verified=bool(user.get("email_verified", False)),
        is_active=bool(user.get("is_active", True)),
        tier=str(user.get("tier") or "PUBLIC"),
        system_roles=list(user.get("system_roles") or ["USER"]),
    )


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register_account(payload: RegisterRequest) -> RegisterResponse:
    """Register a new account and dispatch an email verification link via Brevo."""
    if not payload.terms_accepted:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You must accept the terminology terms of use to register.",
        )

    try:
        user, raw_token = await user_service.register_user(
            email=payload.email,
            password=payload.password,
            display_name=payload.display_name,
            terms_accepted=payload.terms_accepted,
            acted_by=f"register:{payload.email}",
        )
    except EmailAlreadyRegisteredError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        dispatch = await email_service.send_verification_email(
            to_email=user["primary_email"],
            display_name=user["display_name"],
            raw_token=raw_token,
        )
    except EmailConfigurationError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except EmailDeliveryError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    if dispatch.mode == "brevo":
        msg = "Registration received. Please check your inbox for the verification link."
        dev_url = None
    else:
        msg = "Registration received (DEV_MODE: Brevo key unset). Use the local verification link to activate."
        dev_url = dispatch.verification_url

    return RegisterResponse(
        user_id=str(user["id"]),
        email=str(user["primary_email"]),
        display_name=str(user["display_name"]),
        email_verified=bool(user.get("email_verified", False)),
        verification_delivery_mode=dispatch.mode,
        message=msg,
        dev_verification_url=dev_url,
    )


@router.post("/verify-email", response_model=VerifyEmailResponse)
async def verify_email(payload: VerifyEmailRequest) -> VerifyEmailResponse:
    """Verify a registrant's email address using the token from their verification link."""
    try:
        user = await user_service.verify_email_token(payload.token)
    except InvalidVerificationTokenError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return VerifyEmailResponse(
        verified=True,
        email=str(user.get("primary_email", "")),
        message="Email address verified. You can now log in and generate API keys.",
    )


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest) -> TokenResponse:
    """Authenticate with email/username and password and return a signed JWT access token."""
    try:
        user = await user_service.authenticate_local(
            identifier=payload.email,
            password=payload.password,
            require_verified=True,
        )
    except EmailNotVerifiedError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    settings = get_settings()
    token = create_access_token(
        {
            "sub": user["id"],
            "email": user["primary_email"],
            "roles": user.get("system_roles", ["USER"]),
        }
    )
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in_minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
        user=_to_user_profile(user),
    )


@router.get("/me", response_model=UserProfileResponse)
async def get_me(current_user: Dict[str, Any] = Depends(get_current_user)) -> UserProfileResponse:
    """Return the authenticated user's profile."""
    return _to_user_profile(current_user)


@router.post("/keys", response_model=ApiKeyCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    payload: ApiKeyCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ApiKeyCreateResponse:
    """Generate a new X-API-Key credential for the authenticated user (plaintext shown once)."""
    expires_at = None
    if payload.expires_in_days is not None:
        expires_at = datetime.now(timezone.utc) + timedelta(days=payload.expires_in_days)

    try:
        created = await user_service.create_api_key(
            user_id=str(current_user["id"]),
            name=payload.name,
            scopes=payload.scopes,
            expires_at=expires_at,
            acted_by=f"user:{current_user.get('primary_email', current_user['id'])}",
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return ApiKeyCreateResponse(
        id=str(created["id"]),
        name=str(created["name"]),
        prefix=str(created["prefix"]),
        scopes=list(created["scopes"]),
        created_at=created.get("created_at"),
        last_used_at=created.get("last_used_at"),
        expires_at=created.get("expires_at"),
        is_revoked=bool(created.get("is_revoked", False)),
        raw_api_key=str(created["raw_api_key"]),
    )


@router.get("/keys", response_model=List[ApiKeySummary])
async def list_api_keys(
    include_revoked: bool = Query(default=False, description="Include revoked API keys"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> List[ApiKeySummary]:
    """List API keys belonging to the authenticated user (non-secret metadata only)."""
    items = await user_service.list_api_keys(str(current_user["id"]), include_revoked=include_revoked)
    return [ApiKeySummary(**item) for item in items]


@router.delete("/keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Revoke an active API key owned by the authenticated user."""
    revoked = await user_service.revoke_api_key(
        user_id=str(current_user["id"]),
        key_id=key_id,
        acted_by=f"user:{current_user.get('primary_email', current_user['id'])}",
    )
    if not revoked:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Active API key not found.")
    return {"revoked": True, "id": key_id}
