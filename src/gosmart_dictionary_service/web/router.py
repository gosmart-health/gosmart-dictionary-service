"""Multi-page server-rendered HTML routes for registration, login, API keys, and AI Chat."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, Query, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from gosmart_dictionary_service.core.auth_dependency import get_optional_user
from gosmart_dictionary_service.core.config import get_settings
from gosmart_dictionary_service.core.security import (
    create_access_token,
    generate_csrf_token,
    verify_csrf_token,
)
from gosmart_dictionary_service.model.auth import _validate_email_format
from gosmart_dictionary_service.service.chat_service import (
    ChatProviderError,
    ChatRequest,
    chat_service,
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

WEB_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
web_router = APIRouter(include_in_schema=False)

CSRF_COOKIE_NAME = "gosmart_ds_csrf"


def _wants_html(request: Request) -> bool:
    """Return True when the HTTP client explicitly accepts ``text/html``."""
    accept = (request.headers.get("accept") or "").lower()
    return "text/html" in accept


def _safe_next_url(next_url: Optional[str], default: str = "/keys") -> str:
    """Allow only same-origin relative redirect paths."""
    if not next_url:
        return default
    candidate = next_url.strip()
    if candidate.startswith("/") and not candidate.startswith("//"):
        return candidate
    return default


def _get_or_create_csrf_token(request: Request) -> str:
    existing = request.cookies.get(CSRF_COOKIE_NAME)
    if existing and len(existing.strip()) >= 16:
        return existing.strip()
    return generate_csrf_token()


def _set_csrf_cookie(response: Response, csrf_token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=csrf_token,
        httponly=False,
        samesite="lax",
        secure=settings.SESSION_COOKIE_SECURE,
        path="/",
    )


def _render(
    request: Request,
    template_name: str,
    context: Dict[str, Any],
    *,
    current_user: Optional[Dict[str, Any]] = None,
    status_code: int = status.HTTP_200_OK,
) -> Response:
    settings = get_settings()
    csrf_token = _get_or_create_csrf_token(request)
    merged: Dict[str, Any] = {
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION,
        "api_v1_prefix": settings.API_V1_PREFIX,
        "dev_mode": settings.DEV_MODE,
        "current_user": current_user,
        "csrf_token": csrf_token,
        "active_page": "",
        **context,
    }
    response = templates.TemplateResponse(
        request=request,
        name=template_name,
        context=merged,
        status_code=status_code,
    )
    _set_csrf_cookie(response, csrf_token)
    return response


def _redirect_to_login(next_path: str) -> RedirectResponse:
    return RedirectResponse(
        url=f"/login?next={quote(next_path, safe='/')}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


def _validate_csrf_or_error(request: Request, csrf_token: str) -> Optional[str]:
    cookie_val = request.cookies.get(CSRF_COOKIE_NAME)
    if not verify_csrf_token(cookie_val, csrf_token):
        return "Security token validation failed. Please reload the page and try again."
    return None


@web_router.get("/", response_model=None)
async def home_page(
    request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> Response:
    """Serve the HTML home page to browsers, or JSON service metadata to API clients."""
    settings = get_settings()
    if not _wants_html(request):
        return JSONResponse(
            content={
                "name": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "docs_url": "/docs",
                "api_v1": settings.API_V1_PREFIX,
                "db_connection": settings.DB_CONNECTION_URI,
            }
        )
    return _render(
        request,
        "home.html",
        {"active_page": "home"},
        current_user=current_user,
    )


@web_router.get("/register", response_class=HTMLResponse)
async def register_page(
    request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> Response:
    """Render the self-service account registration page."""
    if current_user:
        return RedirectResponse(url="/keys", status_code=status.HTTP_303_SEE_OTHER)
    return _render(
        request,
        "register.html",
        {
            "active_page": "register",
            "form_email": "",
            "form_display_name": "",
            "error": None,
            "success": None,
            "dev_verification_url": None,
        },
    )


@web_router.post("/register", response_class=HTMLResponse)
async def register_submit(
    request: Request,
    email: str = Form(default=""),
    display_name: str = Form(default=""),
    password: str = Form(default=""),
    password_confirm: str = Form(default=""),
    terms_accepted: Optional[str] = Form(default=None),
    csrf_token: str = Form(default=""),
) -> Response:
    """Process account registration form submission and dispatch Brevo verification email."""
    base_ctx: Dict[str, Any] = {
        "active_page": "register",
        "form_email": email.strip(),
        "form_display_name": display_name.strip(),
        "error": None,
        "success": None,
        "dev_verification_url": None,
    }

    csrf_err = _validate_csrf_or_error(request, csrf_token)
    if csrf_err:
        base_ctx["error"] = csrf_err
        return _render(request, "register.html", base_ctx, status_code=status.HTTP_403_FORBIDDEN)

    try:
        normalized_email = _validate_email_format(email)
    except ValueError as exc:
        base_ctx["error"] = str(exc)
        return _render(request, "register.html", base_ctx, status_code=status.HTTP_400_BAD_REQUEST)

    if len(password) < 8:
        base_ctx["error"] = "Password must be at least 8 characters long."
        return _render(request, "register.html", base_ctx, status_code=status.HTTP_400_BAD_REQUEST)

    if password != password_confirm:
        base_ctx["error"] = "Passwords do not match."
        return _render(request, "register.html", base_ctx, status_code=status.HTTP_400_BAD_REQUEST)

    accepted = bool(terms_accepted and terms_accepted.lower() in {"on", "true", "1", "yes"})
    if not accepted:
        base_ctx["error"] = "You must accept the terminology terms of use to register."
        return _render(request, "register.html", base_ctx, status_code=status.HTTP_400_BAD_REQUEST)

    try:
        user, raw_token = await user_service.register_user(
            email=normalized_email,
            password=password,
            display_name=display_name.strip() or None,
            terms_accepted=True,
            acted_by=f"web_register:{normalized_email}",
        )
    except EmailAlreadyRegisteredError as exc:
        base_ctx["error"] = str(exc)
        return _render(request, "register.html", base_ctx, status_code=status.HTTP_409_CONFLICT)
    except ValueError as exc:
        base_ctx["error"] = str(exc)
        return _render(request, "register.html", base_ctx, status_code=status.HTTP_400_BAD_REQUEST)

    try:
        dispatch = await email_service.send_verification_email(
            to_email=user["primary_email"],
            display_name=user["display_name"],
            raw_token=raw_token,
        )
    except EmailConfigurationError as exc:
        base_ctx["error"] = str(exc)
        return _render(request, "register.html", base_ctx, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    except EmailDeliveryError as exc:
        base_ctx["error"] = str(exc)
        return _render(request, "register.html", base_ctx, status_code=status.HTTP_502_BAD_GATEWAY)

    if dispatch.mode == "brevo":
        base_ctx["success"] = (
            f"Account created for {user['primary_email']}. "
            "Check your inbox for the Brevo verification email to activate your account."
        )
    else:
        base_ctx["success"] = (
            f"Account created for {user['primary_email']} in DEV_MODE (GOSMART_DS_BREVO_API_KEY is unset)."
        )
        base_ctx["dev_verification_url"] = dispatch.verification_url

    base_ctx["form_email"] = ""
    base_ctx["form_display_name"] = ""
    return _render(request, "register.html", base_ctx, status_code=status.HTTP_201_CREATED)


@web_router.get("/verify-email", response_class=HTMLResponse)
async def verify_email_page(
    request: Request,
    token: Optional[str] = Query(default=None),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> Response:
    """Confirm email verification when ``?token=...`` is provided, or render manual token form."""
    ctx: Dict[str, Any] = {
        "active_page": "verify",
        "verified": False,
        "verified_email": None,
        "error": None,
    }
    if token and token.strip():
        try:
            user = await user_service.verify_email_token(token.strip())
            ctx["verified"] = True
            ctx["verified_email"] = user.get("primary_email")
        except InvalidVerificationTokenError as exc:
            ctx["error"] = str(exc)
            return _render(
                request,
                "verify_email.html",
                ctx,
                current_user=current_user,
                status_code=status.HTTP_400_BAD_REQUEST,
            )
    return _render(request, "verify_email.html", ctx, current_user=current_user)


@web_router.post("/verify-email", response_class=HTMLResponse)
async def verify_email_submit(
    request: Request,
    token: str = Form(default=""),
    csrf_token: str = Form(default=""),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> Response:
    """Verify an email token submitted via HTML form."""
    ctx: Dict[str, Any] = {
        "active_page": "verify",
        "verified": False,
        "verified_email": None,
        "error": None,
    }
    csrf_err = _validate_csrf_or_error(request, csrf_token)
    if csrf_err:
        ctx["error"] = csrf_err
        return _render(
            request,
            "verify_email.html",
            ctx,
            current_user=current_user,
            status_code=status.HTTP_403_FORBIDDEN,
        )

    try:
        user = await user_service.verify_email_token(token.strip())
        ctx["verified"] = True
        ctx["verified_email"] = user.get("primary_email")
    except InvalidVerificationTokenError as exc:
        ctx["error"] = str(exc)
        return _render(
            request,
            "verify_email.html",
            ctx,
            current_user=current_user,
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    return _render(request, "verify_email.html", ctx, current_user=current_user)


@web_router.get("/login", response_class=HTMLResponse)
async def login_page(
    request: Request,
    next: Optional[str] = Query(default=None),
    logged_out: Optional[str] = Query(default=None),
    verified: Optional[str] = Query(default=None),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> Response:
    """Render the login page."""
    target = _safe_next_url(next, default="/keys")
    if current_user:
        return RedirectResponse(url=target, status_code=status.HTTP_303_SEE_OTHER)

    notice = None
    if logged_out:
        notice = "You have been logged out."
    elif verified:
        notice = "Your email is verified. Please log in below."

    return _render(
        request,
        "login.html",
        {
            "active_page": "login",
            "next_url": target,
            "form_email": "",
            "error": None,
            "notice": notice,
        },
    )


@web_router.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    email: str = Form(default=""),
    password: str = Form(default=""),
    next_url: str = Form(default="/keys"),
    csrf_token: str = Form(default=""),
) -> Response:
    """Authenticate credentials and issue an HttpOnly signed JWT session cookie."""
    target = _safe_next_url(next_url, default="/keys")
    ctx: Dict[str, Any] = {
        "active_page": "login",
        "next_url": target,
        "form_email": email.strip(),
        "error": None,
        "notice": None,
    }

    csrf_err = _validate_csrf_or_error(request, csrf_token)
    if csrf_err:
        ctx["error"] = csrf_err
        return _render(request, "login.html", ctx, status_code=status.HTTP_403_FORBIDDEN)

    if not email.strip() or not password:
        ctx["error"] = "Please enter both email and password."
        return _render(request, "login.html", ctx, status_code=status.HTTP_400_BAD_REQUEST)

    try:
        user = await user_service.authenticate_local(
            identifier=email.strip(),
            password=password,
            require_verified=True,
        )
    except EmailNotVerifiedError as exc:
        ctx["error"] = str(exc)
        return _render(request, "login.html", ctx, status_code=status.HTTP_403_FORBIDDEN)

    if not user:
        ctx["error"] = "Invalid email or password."
        return _render(request, "login.html", ctx, status_code=status.HTTP_401_UNAUTHORIZED)

    settings = get_settings()
    access_token = create_access_token(
        {
            "sub": user["id"],
            "email": user["primary_email"],
            "roles": user.get("system_roles", ["USER"]),
        }
    )
    response = RedirectResponse(url=target, status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        key=settings.SESSION_COOKIE_NAME,
        value=access_token,
        max_age=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=True,
        samesite="lax",
        secure=settings.SESSION_COOKIE_SECURE,
        path="/",
    )
    return response


@web_router.post("/logout")
async def logout_submit(
    request: Request,
    csrf_token: str = Form(default=""),
) -> Response:
    """Clear the HttpOnly session cookie and redirect to the login page."""
    csrf_err = _validate_csrf_or_error(request, csrf_token)
    if csrf_err:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    settings = get_settings()
    response = RedirectResponse(url="/login?logged_out=1", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(key=settings.SESSION_COOKIE_NAME, path="/")
    return response


@web_router.get("/keys", response_class=HTMLResponse)
async def keys_page(
    request: Request,
    revoked: Optional[str] = Query(default=None),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> Response:
    """Render the self-service API key management page for logged-in users."""
    if not current_user:
        return _redirect_to_login("/keys")

    keys = await user_service.list_api_keys(str(current_user["id"]), include_revoked=True)
    return _render(
        request,
        "keys.html",
        {
            "active_page": "keys",
            "keys": keys,
            "new_api_key": None,
            "new_api_key_name": None,
            "error": None,
            "notice": "API key revoked successfully." if revoked else None,
        },
        current_user=current_user,
    )


@web_router.post("/keys/create", response_class=HTMLResponse)
async def keys_create_submit(
    request: Request,
    name: str = Form(default=""),
    expires_in_days: str = Form(default=""),
    csrf_token: str = Form(default=""),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> Response:
    """Create a new API key and display its plaintext secret once."""
    if not current_user:
        return _redirect_to_login("/keys")

    csrf_err = _validate_csrf_or_error(request, csrf_token)
    if csrf_err:
        keys = await user_service.list_api_keys(str(current_user["id"]), include_revoked=True)
        return _render(
            request,
            "keys.html",
            {
                "active_page": "keys",
                "keys": keys,
                "new_api_key": None,
                "new_api_key_name": None,
                "error": csrf_err,
                "notice": None,
            },
            current_user=current_user,
            status_code=status.HTTP_403_FORBIDDEN,
        )

    clean_name = name.strip()
    error: Optional[str] = None
    expires_at: Optional[datetime] = None

    if not clean_name:
        error = "Please provide a descriptive name for the API key."
    elif expires_in_days.strip():
        try:
            days = int(expires_in_days.strip())
            if days < 1 or days > 3650:
                error = "Expiration days must be between 1 and 3650."
            else:
                expires_at = datetime.now(timezone.utc) + timedelta(days=days)
        except ValueError:
            error = "Expiration days must be a whole number."

    new_raw_key: Optional[str] = None
    new_key_name: Optional[str] = None
    status_code = status.HTTP_200_OK

    if error:
        status_code = status.HTTP_400_BAD_REQUEST
    else:
        created = await user_service.create_api_key(
            user_id=str(current_user["id"]),
            name=clean_name,
            expires_at=expires_at,
            acted_by=f"web_user:{current_user.get('primary_email', current_user['id'])}",
        )
        new_raw_key = str(created["raw_api_key"])
        new_key_name = str(created["name"])
        status_code = status.HTTP_201_CREATED

    keys = await user_service.list_api_keys(str(current_user["id"]), include_revoked=True)
    return _render(
        request,
        "keys.html",
        {
            "active_page": "keys",
            "keys": keys,
            "new_api_key": new_raw_key,
            "new_api_key_name": new_key_name,
            "error": error,
            "notice": None,
        },
        current_user=current_user,
        status_code=status_code,
    )


@web_router.post("/keys/{key_id}/revoke")
async def keys_revoke_submit(
    request: Request,
    key_id: str,
    csrf_token: str = Form(default=""),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> Response:
    """Revoke an active API key owned by the logged-in user."""
    if not current_user:
        return _redirect_to_login("/keys")

    csrf_err = _validate_csrf_or_error(request, csrf_token)
    if csrf_err:
        return RedirectResponse(url="/keys", status_code=status.HTTP_303_SEE_OTHER)

    await user_service.revoke_api_key(
        user_id=str(current_user["id"]),
        key_id=key_id,
        acted_by=f"web_user:{current_user.get('primary_email', current_user['id'])}",
    )
    return RedirectResponse(url="/keys?revoked=1", status_code=status.HTTP_303_SEE_OTHER)


@web_router.get("/chat", response_class=HTMLResponse)
async def chat_page(
    request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> Response:
    """Render the Grounded Dictionary AI Chat page."""
    if not current_user:
        return _redirect_to_login("/chat")

    cfg = await chat_service.resolve_inference_config()
    return _render(
        request,
        "chat.html",
        {
            "active_page": "chat",
            "inference_provider": cfg.provider,
            "inference_model": cfg.model,
            "messages": [],
            "error": None,
        },
        current_user=current_user,
    )


@web_router.post("/chat", response_class=HTMLResponse)
async def chat_submit(
    request: Request,
    message: str = Form(default=""),
    include_context: Optional[str] = Form(default="on"),
    csrf_token: str = Form(default=""),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> Response:
    """Standard HTML form fallback for AI Chat (works even with JavaScript disabled)."""
    if not current_user:
        return _redirect_to_login("/chat")

    cfg = await chat_service.resolve_inference_config()
    messages: List[Dict[str, Any]] = []
    error: Optional[str] = None
    status_code = status.HTTP_200_OK

    csrf_err = _validate_csrf_or_error(request, csrf_token)
    if csrf_err:
        error = csrf_err
        status_code = status.HTTP_403_FORBIDDEN
    elif not message.strip():
        error = "Please enter a question."
        status_code = status.HTTP_400_BAD_REQUEST
    else:
        use_ctx = bool(include_context and include_context.lower() in {"on", "true", "1", "yes"})
        messages.append({"role": "user", "content": message.strip(), "citations": []})
        try:
            chat_resp = await chat_service.chat(
                ChatRequest(message=message.strip(), include_context=use_ctx),
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": chat_resp.reply,
                    "citations": [c.model_dump() for c in chat_resp.citations],
                    "meta": f"{chat_resp.provider}/{chat_resp.model}",
                }
            )
        except ChatProviderError as exc:
            error = str(exc)
            status_code = status.HTTP_502_BAD_GATEWAY

    return _render(
        request,
        "chat.html",
        {
            "active_page": "chat",
            "inference_provider": cfg.provider,
            "inference_model": cfg.model,
            "messages": messages,
            "error": error,
        },
        current_user=current_user,
        status_code=status_code,
    )
