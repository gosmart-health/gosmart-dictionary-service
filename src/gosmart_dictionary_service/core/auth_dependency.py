"""FastAPI authentication dependencies supporting X-API-Key, Bearer JWT, and HttpOnly session cookies."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from gosmart_dictionary_service.core.config import get_settings
from gosmart_dictionary_service.core.logging import bind_log_context, get_logger
from gosmart_dictionary_service.core.security import decode_token
from gosmart_dictionary_service.service.user_service import user_service

logger = get_logger("gosmart_ds.auth_dep")

# Optional Bearer schema for Swagger UI & header extraction
bearer_scheme = HTTPBearer(auto_error=False)

# Request/MCP-scoped active Group ID for downstream services (e.g. dynamic LLM resolution)
_active_group_id_ctx: ContextVar[Optional[str]] = ContextVar("_active_group_id_ctx", default=None)


def get_active_group_id() -> Optional[str]:
    """Return the active Group ID bound to the current execution context, if any."""
    return _active_group_id_ctx.get()


def set_active_group_id(group_id: Optional[str]) -> Token[Optional[str]]:
    """Bind an active Group ID to the current execution context."""
    cleaned = group_id.strip() if isinstance(group_id, str) and group_id.strip() else None
    return _active_group_id_ctx.set(cleaned)


def reset_active_group_id(token: Token[Optional[str]]) -> None:
    """Reset the active Group ID context variable using its token."""
    _active_group_id_ctx.reset(token)


async def _authenticate_jwt_token(raw_token: str, *, auth_type: str) -> Dict[str, Any]:
    try:
        payload = decode_token(raw_token)
    except Exception as exc:
        logger.warning("jwt_validation_failed", auth_type=auth_type, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    if payload.get("token_type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Provided token is not an access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload: missing 'sub'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await user_service.get_user_by_id(str(user_id))
    if not user or not user.get("is_active") or user.get("email_verified") is False:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive, unverified, or no longer exists",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user["token_claims"] = payload
    return user


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Dict[str, Any]:
    """Extract and validate the authenticated User from ``X-API-Key``, JWT Bearer, or session cookie.

    Raises HTTP 401 Unauthorized if missing, expired, or invalid.
    """
    settings = get_settings()
    raw_bearer: Optional[str] = None
    if credentials and credentials.credentials:
        raw_bearer = credentials.credentials.strip()

    # 1. API Key Authentication (via X-API-Key header or Bearer gs_dict_...)
    api_key = x_api_key.strip() if isinstance(x_api_key, str) and x_api_key.strip() else None
    if not api_key and raw_bearer and raw_bearer.startswith(settings.API_KEY_PREFIX):
        api_key = raw_bearer

    if api_key:
        user = await user_service.authenticate_api_key(api_key)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid, revoked, or expired API Key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        request.state.current_user = user
        bind_log_context(user_id=user["id"], auth_type="api_key")
        return user

    # 2. JWT Bearer Token Authentication
    if raw_bearer:
        user = await _authenticate_jwt_token(raw_bearer, auth_type="jwt_bearer")
        request.state.current_user = user
        bind_log_context(user_id=user["id"], user_email=user.get("primary_email"), auth_type="jwt")
        return user

    # 3. HttpOnly Browser Session Cookie Authentication
    cookie_token = request.cookies.get(settings.SESSION_COOKIE_NAME)
    if cookie_token and cookie_token.strip():
        user = await _authenticate_jwt_token(cookie_token.strip(), auth_type="session_cookie")
        request.state.current_user = user
        bind_log_context(user_id=user["id"], user_email=user.get("primary_email"), auth_type="cookie")
        return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication credentials were not provided",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Optional[Dict[str, Any]]:
    """Optionally extract the authenticated User if valid credentials are provided."""
    try:
        return await get_current_user(request=request, credentials=credentials, x_api_key=x_api_key)
    except HTTPException:
        return None


def require_roles(allowed_roles: List[str]) -> Callable[..., Any]:
    """Dependency factory checking that ``current_user`` has at least one of the allowed system roles."""

    async def _role_checker(
        current_user: Dict[str, Any] = Depends(get_current_user),
    ) -> Dict[str, Any]:
        user_roles = set(current_user.get("system_roles", []))
        if "PLATFORM_ADMIN" in user_roles or "ADMIN" in user_roles:
            return current_user

        allowed_set = {r.lower() for r in allowed_roles}
        user_set = {r.lower() for r in user_roles}
        if not allowed_set.intersection(user_set):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires one of the following roles: {allowed_roles}",
            )
        return current_user

    return _role_checker


__all__ = [
    "bearer_scheme",
    "get_active_group_id",
    "set_active_group_id",
    "reset_active_group_id",
    "get_current_user",
    "get_optional_user",
    "require_roles",
]
