"""Multi-page server-rendered HTML routes for registration, login, API keys, AI Chat, and Entity Explorer."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlsplit

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
from gosmart_dictionary_service.model.search import (
    EntityNeighborhood,
    MatchSource,
    RelatedConceptHit,
    SearchEntityType,
    build_drilldown_url,
)
from gosmart_dictionary_service.service.chat_service import (
    ChatProviderError,
    ChatRequest,
    ChatSearchScope,
    chat_service,
    search_hit_to_citation,
)
from gosmart_dictionary_service.service.email_service import (
    EmailConfigurationError,
    EmailDeliveryError,
    email_service,
)
from gosmart_dictionary_service.service.search_service import search_service
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
_VALID_CHAT_SCOPES: Dict[str, ChatSearchScope] = {
    "auto": "auto",
    "clinical": "clinical",
    "dicom": "dicom",
}

MCP_SERVER_ALIAS = "gosmart-dictionary"
MCP_API_KEY_PLACEHOLDER = "<YOUR_GOSMART_API_KEY>"

_MCP_AGENT_TABS: Tuple[Dict[str, str], ...] = (
    {"id": "cursor", "label": "Cursor", "filename": "mcp.json"},
    {"id": "claude-code", "label": "Claude Code", "filename": ".mcp.json"},
    {"id": "claude-desktop", "label": "Claude Desktop", "filename": "claude_desktop_config.json"},
    {"id": "gemini", "label": "Gemini CLI", "filename": "settings.json"},
    {"id": "antigravity", "label": "Antigravity", "filename": "mcp_config.json"},
)
_MCP_AGENT_ALIASES: Dict[str, str] = {
    "cursor": "cursor",
    "claude": "claude-code",
    "claude-code": "claude-code",
    "claude_code": "claude-code",
    "claude-desktop": "claude-desktop",
    "claude_desktop": "claude-desktop",
    "gemini": "gemini",
    "gemini-cli": "gemini",
    "gemini_cli": "gemini",
    "antigravity": "antigravity",
}


def normalize_mcp_agent(raw_agent: Optional[str]) -> str:
    """Normalize an agent selector value to one of the supported profile keys."""
    cleaned = (raw_agent or "").strip().lower()
    return _MCP_AGENT_ALIASES.get(cleaned, "cursor")


def _is_local_host_authority(authority: str) -> bool:
    """Return True when ``authority`` targets loopback or a local dev host."""
    host_part = authority.strip().lower()
    if host_part.startswith("["):
        bracket_end = host_part.find("]")
        host_only = host_part[1:bracket_end] if bracket_end != -1 else host_part.lstrip("[")
    else:
        host_only = host_part.split(":", 1)[0]
    return host_only in {"localhost", "127.0.0.1", "::1", "0.0.0.0", "testserver"} or host_only.endswith(
        (".local", ".localhost")
    )


def _resolve_default_server_origin(request: Request) -> str:
    """Determine the default public origin for the current GoSmart DS instance."""
    settings = get_settings()
    forwarded_host = (request.headers.get("x-forwarded-host") or "").split(",", 1)[0].strip()
    if forwarded_host:
        forwarded_proto = (request.headers.get("x-forwarded-proto") or "").split(",", 1)[0].strip().lower()
        scheme = (
            forwarded_proto
            if forwarded_proto in {"http", "https"}
            else ("http" if _is_local_host_authority(forwarded_host) else "https")
        )
        return f"{scheme}://{forwarded_host}".rstrip("/")

    configured_public = (settings.PUBLIC_BASE_URL or "").strip().rstrip("/")
    if configured_public and configured_public != "http://127.0.0.1:8081":
        return configured_public

    return str(request.base_url).rstrip("/")


def normalize_mcp_endpoint_url(
    raw_host: Optional[str],
    default_origin: str,
    mount_path: str = "/mcp",
    sse_mount_path: str = "/mcp-sse",
) -> Tuple[str, str]:
    """Normalize a bare hostname, ``host:port``, or full URL into ``(origin_base, mcp_endpoint_url)``."""
    candidate = (raw_host or "").strip() or default_origin.strip() or "http://127.0.0.1:8081"
    candidate = candidate.rstrip("/")

    lower_candidate = candidate.lower()
    if not (lower_candidate.startswith("http://") or lower_candidate.startswith("https://")):
        authority = candidate.split("/", 1)[0]
        inferred_scheme = "http" if _is_local_host_authority(authority) else "https"
        candidate = f"{inferred_scheme}://{candidate}"

    parsed = urlsplit(candidate)
    scheme = parsed.scheme.lower() if parsed.scheme else "https"
    netloc = parsed.netloc
    path = parsed.path.rstrip("/")

    clean_mount = "/" + (mount_path or "/mcp").strip("/")
    clean_sse = "/" + (sse_mount_path or "/mcp-sse").strip("/")
    for suffix in (f"{clean_sse}/sse", clean_sse, clean_mount):
        if suffix and path.lower().endswith(suffix.lower()):
            path = path[: -len(suffix)].rstrip("/")
            break

    normalized_base = f"{scheme}://{netloc}{path}".rstrip("/")
    endpoint_url = f"{normalized_base}{clean_mount}"
    return normalized_base, endpoint_url


def build_mcp_client_guide(
    *,
    agent: Optional[str],
    raw_host: Optional[str],
    default_origin: str,
    api_key: Optional[str] = None,
    mount_path: str = "/mcp",
    sse_mount_path: str = "/mcp-sse",
) -> Dict[str, Any]:
    """Build the agent-specific MCP installation JSON, target file paths, and setup instructions."""
    selected_agent = normalize_mcp_agent(agent)
    normalized_base, endpoint_url = normalize_mcp_endpoint_url(
        raw_host,
        default_origin=default_origin,
        mount_path=mount_path,
        sse_mount_path=sse_mount_path,
    )
    host_input = (raw_host or "").strip() or normalized_base
    clean_key = (api_key or "").strip()
    has_real_key = bool(clean_key)
    effective_key = clean_key if has_real_key else MCP_API_KEY_PLACEHOLDER

    cli_command: Optional[str] = None
    if selected_agent == "claude-code":
        label = "Claude Code"
        download_filename = ".mcp.json"
        server_entry: Dict[str, Any] = {
            "type": "http",
            "url": endpoint_url,
            "headers": {"X-API-Key": effective_key},
        }
        config_locations = [
            {"scope": "Project (shared / repo root)", "path": ".mcp.json"},
            {"scope": "Global (macOS / Linux)", "path": "~/.claude.json"},
            {"scope": "Global (Windows)", "path": r"%USERPROFILE%\.claude.json"},
        ]
        merge_hint = (
            'Save as .mcp.json in your project root, or merge the "gosmart-dictionary" entry inside '
            'the existing "mcpServers" object in ~/.claude.json (or run the CLI command below).'
        )
        cli_command = (
            f'claude mcp add --transport http {MCP_SERVER_ALIAS} {endpoint_url} --header "X-API-Key: {effective_key}"'
        )
        steps = [
            "Place or merge the JSON into .mcp.json (project) or ~/.claude.json (global), "
            "or run the claude mcp add command below.",
            'Keep "type": "http" intact—Claude Code requires an explicit transport type for remote HTTP MCP servers.',
            "Run /mcp inside an interactive Claude Code session to verify gosmart-dictionary is connected.",
        ]
    elif selected_agent == "claude-desktop":
        label = "Claude Desktop"
        download_filename = "claude_desktop_config.json"
        server_entry = {
            "command": "npx",
            "args": [
                "-y",
                "mcp-remote",
                endpoint_url,
                "--header",
                "X-API-Key:${GOSMART_API_KEY}",
            ],
            "env": {
                "GOSMART_API_KEY": effective_key,
            },
        }
        config_locations = [
            {"scope": "macOS", "path": "~/Library/Application Support/Claude/claude_desktop_config.json"},
            {"scope": "Windows", "path": r"%APPDATA%\Claude\claude_desktop_config.json"},
            {"scope": "Linux", "path": "~/.config/Claude/claude_desktop_config.json"},
        ]
        merge_hint = (
            "Open via Claude Desktop → Settings → Developer → Edit Config, and merge "
            '"gosmart-dictionary" under "mcpServers".'
        )
        steps = [
            "Ensure Node.js 18+ (npx) is installed—claude_desktop_config.json bridges remote HTTP "
            "headers via mcp-remote.",
            "Save or merge the JSON block into claude_desktop_config.json at your OS path above.",
            "Restart Claude Desktop completely to load the GoSmart Dictionary tools.",
        ]
    elif selected_agent == "gemini":
        label = "Gemini CLI"
        download_filename = "settings.json"
        server_entry = {
            "httpUrl": endpoint_url,
            "headers": {"X-API-Key": effective_key},
        }
        config_locations = [
            {"scope": "Project (workspace)", "path": ".gemini/settings.json"},
            {"scope": "Global (macOS / Linux)", "path": "~/.gemini/settings.json"},
            {"scope": "Global (Windows)", "path": r"%USERPROFILE%\.gemini\settings.json"},
        ]
        merge_hint = (
            'settings.json also stores general Gemini CLI settings—merge "gosmart-dictionary" inside '
            'its "mcpServers" object rather than overwriting an existing settings.json '
            "(or run the CLI command below)."
        )
        cli_command = (
            f'gemini mcp add --transport http --header "X-API-Key: {effective_key}" {MCP_SERVER_ALIAS} {endpoint_url}'
        )
        steps = [
            "Merge the JSON block into ~/.gemini/settings.json or .gemini/settings.json, "
            "or run the gemini mcp add command below.",
            'Keep "httpUrl" (Gemini CLI reserves "url" for legacy SSE) and the hyphenated alias "gosmart-dictionary".',
            "Run /mcp inside an interactive gemini session to confirm gosmart-dictionary is active.",
        ]
    elif selected_agent == "antigravity":
        label = "Antigravity"
        download_filename = "mcp_config.json"
        server_entry = {
            "serverUrl": endpoint_url,
            "headers": {"X-API-Key": effective_key},
        }
        config_locations = [
            {"scope": "Workspace", "path": ".agents/mcp_config.json"},
            {"scope": "Global (macOS / Linux)", "path": "~/.gemini/config/mcp_config.json"},
            {"scope": "Global (Windows)", "path": r"%USERPROFILE%\.gemini\config\mcp_config.json"},
        ]
        merge_hint = (
            "Save directly as .agents/mcp_config.json for a single workspace, or merge "
            '"gosmart-dictionary" under "mcpServers" in ~/.gemini/config/mcp_config.json '
            "(also accessible via MCP Servers → Manage MCP Servers → View raw config)."
        )
        steps = [
            "Open MCP Servers → Manage MCP Servers → View raw config in Antigravity, "
            "or open ~/.gemini/config/mcp_config.json (or <workspace>/.agents/mcp_config.json).",
            'Merge "gosmart-dictionary" under "mcpServers" and save '
            '(Antigravity uses "serverUrl" and does not accept Gemini CLI\'s "httpUrl").',
            "Refresh MCP servers in Antigravity to load the 15 GoSmart Dictionary tools.",
        ]
    else:
        label = "Cursor"
        download_filename = "mcp.json"
        server_entry = {
            "url": endpoint_url,
            "headers": {"X-API-Key": effective_key},
        }
        config_locations = [
            {"scope": "Project (workspace)", "path": ".cursor/mcp.json"},
            {"scope": "Global (macOS / Linux)", "path": "~/.cursor/mcp.json"},
            {"scope": "Global (Windows)", "path": r"%USERPROFILE%\.cursor\mcp.json"},
        ]
        merge_hint = (
            'Save directly as a new mcp.json file, or merge "gosmart-dictionary" inside '
            'your existing "mcpServers" object.'
        )
        steps = [
            "Save or merge the JSON into .cursor/mcp.json (project) or ~/.cursor/mcp.json (global), "
            "or open Cursor Settings → Tools & MCP.",
            "Save the file and confirm gosmart-dictionary shows a green status indicator.",
            'Ask Cursor a terminology or DICOM question (e.g. "Search SNOMED for acute pneumonia").',
        ]

    config_payload = {"mcpServers": {MCP_SERVER_ALIAS: server_entry}}
    json_config = json.dumps(config_payload, indent=2)
    download_query = f"agent={quote(selected_agent)}&mcp_host={quote(host_input)}"

    return {
        "agent": selected_agent,
        "agent_label": label,
        "agent_tabs": [dict(tab) for tab in _MCP_AGENT_TABS],
        "server_alias": MCP_SERVER_ALIAS,
        "host_input": host_input,
        "normalized_origin": normalized_base,
        "endpoint_url": endpoint_url,
        "has_real_key": has_real_key,
        "api_key_placeholder": MCP_API_KEY_PLACEHOLDER,
        "download_filename": download_filename,
        "download_url": f"/keys/mcp-config.json?{download_query}",
        "config_locations": config_locations,
        "merge_hint": merge_hint,
        "cli_command": cli_command,
        "steps": steps,
        "json_config": json_config,
    }


def _build_request_mcp_guide(
    request: Request,
    *,
    agent: Optional[str] = None,
    raw_host: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    settings = get_settings()
    default_origin = _resolve_default_server_origin(request)
    return build_mcp_client_guide(
        agent=agent,
        raw_host=raw_host,
        default_origin=default_origin,
        api_key=api_key,
        mount_path=settings.MCP_MOUNT_PATH,
        sse_mount_path=settings.MCP_SSE_MOUNT_PATH,
    )


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
        url=f"/login?next={quote(next_path, safe='/?=&:')}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


def _validate_csrf_or_error(request: Request, csrf_token: str) -> Optional[str]:
    cookie_val = request.cookies.get(CSRF_COOKIE_NAME)
    if not verify_csrf_token(cookie_val, csrf_token):
        return "Security token validation failed. Please reload the page and try again."
    return None


def _normalize_scope(raw_scope: Optional[str]) -> ChatSearchScope:
    key = (raw_scope or "auto").strip().lower()
    return _VALID_CHAT_SCOPES.get(key, "auto")


def _format_related_row(rel: RelatedConceptHit) -> Dict[str, Any]:
    props = rel.properties or {}
    vocab = str(props["vocabulary"]) if props.get("vocabulary") else None
    if rel.entity_type == SearchEntityType.CONCEPT and vocab:
        identifier = f"{vocab}:{rel.coding.code}"
        schema_badge = vocab
    elif rel.entity_type == SearchEntityType.DICOM_ELEMENT:
        identifier = str(props.get("tag") or rel.coding.code)
        schema_badge = "DICOM Tag"
    else:
        identifier = rel.coding.code
        schema_badge = rel.entity_type.value.removeprefix("Dicom") or rel.entity_type.value

    detail_bits: List[str] = []
    if props.get("domain"):
        detail_bits.append(f"Domain: {props['domain']}")
    if props.get("standard") == "S":
        detail_bits.append("Standard")
    if props.get("equivalence"):
        detail_bits.append(f"Equiv: {props['equivalence']}")
    if props.get("hops") and int(props["hops"]) > 1:
        detail_bits.append(f"{props['hops']}-hop")
    if rel.shared_axes:
        detail_bits.append("Shared axes: " + ", ".join(rel.shared_axes[:3]))
    if props.get("keyword"):
        detail_bits.append(str(props["keyword"]))
    if props.get("vr"):
        detail_bits.append(f"VR={props['vr']}")
    if props.get("requirement_type"):
        detail_bits.append(f"Req={props['requirement_type']}")
    if props.get("usage"):
        detail_bits.append(f"Usage={props['usage']}")
    if props.get("information_entity"):
        detail_bits.append(f"IE={props['information_entity']}")
    if props.get("basic_profile"):
        detail_bits.append(f"BasicProfile={props['basic_profile']}")
    if props.get("path"):
        detail_bits.append(f"Path={props['path']}")

    drilldown_url = build_drilldown_url(
        node_id=rel.id,
        entity_type=rel.entity_type,
        vocabulary=vocab,
        code=rel.coding.code,
    )
    chat_prompt = f"Explain {identifier} ({rel.coding.display}) and show related concepts"
    return {
        "id": rel.id,
        "entity_type": rel.entity_type.value,
        "schema_badge": schema_badge,
        "identifier": identifier,
        "display": rel.coding.display or identifier,
        "predicate": rel.relationship_id or rel.relation_reason.value,
        "direction": rel.direction.value,
        "reason": rel.relation_reason.value,
        "details": " · ".join(detail_bits),
        "macro": rel.macro,
        "deprecated": bool(props.get("deprecated", False)),
        "drilldown_url": drilldown_url,
        "ask_chat_url": f"/chat?q={quote(chat_prompt)}",
    }


def _group_neighborhood_tables(neighborhood: EntityNeighborhood) -> List[Dict[str, Any]]:
    """Organize ``neighborhood.seed.related`` into scannable relationship tables for ``explore.html``."""
    hierarchy: List[Dict[str, Any]] = []
    crosswalks: List[Dict[str, Any]] = []
    attributes: List[Dict[str, Any]] = []
    dicom_links: List[Dict[str, Any]] = []

    for rel in neighborhood.seed.related:
        formatted = _format_related_row(rel)
        if rel.relation_reason in {MatchSource.IS_A_PARENT, MatchSource.IS_A_CHILD}:
            hierarchy.append(formatted)
        elif rel.relation_reason in {
            MatchSource.MAPS_TO,
            MatchSource.MAPPED_FROM,
            MatchSource.REPLACED_BY,
            MatchSource.REPLACES,
        }:
            crosswalks.append(formatted)
        elif rel.relation_reason == MatchSource.DICOM_RELATION:
            dicom_links.append(formatted)
        else:
            attributes.append(formatted)

    groups: List[Dict[str, Any]] = []
    if hierarchy:
        groups.append(
            {
                "title": "Ontology Hierarchy (IS_A Parents & Subtypes)",
                "subtitle": "Direct supertype and subtype concepts in the terminology graph.",
                "rows": hierarchy,
            }
        )
    if crosswalks:
        groups.append(
            {
                "title": "Crosswalks & Lifecycle Mappings (MAPS_TO / REPLACED_BY)",
                "subtitle": "Cross-vocabulary translations and active replacement concepts.",
                "rows": crosswalks,
            }
        )
    if attributes:
        groups.append(
            {
                "title": "Defining Attributes & Shared-Axis Siblings",
                "subtitle": "1-hop defining attributes and 2-hop sibling concepts sharing clinical axes.",
                "rows": attributes,
            }
        )
    if dicom_links:
        groups.append(
            {
                "title": "DICOM Standard Structure & Confidentiality Links",
                "subtitle": "Connected CIODs, SOP Classes, Modules, Macros, Data Elements, and PS3.15 rules.",
                "rows": dicom_links,
            }
        )
    return groups


@web_router.get("/", response_model=None)
async def home_page(
    request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> Response:
    """Serve the HTML home page to browsers, or JSON service metadata to API clients."""
    settings = get_settings()
    if not _wants_html(request):
        metadata: Dict[str, Any] = {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs_url": "/docs",
            "api_v1": settings.API_V1_PREFIX,
            "db_connection": settings.DB_CONNECTION_URI,
        }
        if settings.MCP_ENABLED:
            metadata["mcp"] = settings.MCP_MOUNT_PATH
            metadata["mcp_sse"] = settings.MCP_SSE_MOUNT_PATH
        return JSONResponse(content=metadata)
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
    agent: Optional[str] = Query(default=None),
    mcp_host: Optional[str] = Query(default=None),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> Response:
    """Render the self-service API key management page for logged-in users."""
    if not current_user:
        return _redirect_to_login("/keys")

    settings = get_settings()
    mcp_guide = _build_request_mcp_guide(
        request,
        agent=agent,
        raw_host=mcp_host,
        api_key=None,
    )
    keys = await user_service.list_api_keys(str(current_user["id"]), include_revoked=True)
    return _render(
        request,
        "keys.html",
        {
            "active_page": "keys",
            "keys": keys,
            "new_api_key": None,
            "new_api_key_name": None,
            "mcp_enabled": settings.MCP_ENABLED,
            "mcp_mount_path": settings.MCP_MOUNT_PATH,
            "mcp_sse_mount_path": settings.MCP_SSE_MOUNT_PATH,
            "mcp_guide": mcp_guide,
            "error": None,
            "notice": "API key revoked successfully." if revoked else None,
        },
        current_user=current_user,
    )


@web_router.get("/keys/mcp-config.json")
async def keys_download_mcp_config(
    request: Request,
    agent: Optional[str] = Query(default=None),
    mcp_host: Optional[str] = Query(default=None),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> Response:
    """Download the MCP client JSON file for the selected agent (progressive fallback)."""
    if not current_user:
        return _redirect_to_login("/keys")

    mcp_guide = _build_request_mcp_guide(
        request,
        agent=agent,
        raw_host=mcp_host,
        api_key=None,
    )
    filename = str(mcp_guide["download_filename"])
    return Response(
        content=str(mcp_guide["json_config"]) + "\n",
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@web_router.post("/keys/create", response_class=HTMLResponse)
async def keys_create_submit(
    request: Request,
    name: str = Form(default=""),
    expires_in_days: str = Form(default=""),
    agent: str = Form(default="cursor"),
    mcp_host: str = Form(default=""),
    csrf_token: str = Form(default=""),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> Response:
    """Create a new API key and display its plaintext secret once."""
    if not current_user:
        return _redirect_to_login("/keys")

    settings = get_settings()
    csrf_err = _validate_csrf_or_error(request, csrf_token)
    if csrf_err:
        mcp_guide = _build_request_mcp_guide(
            request,
            agent=agent,
            raw_host=mcp_host,
            api_key=None,
        )
        keys = await user_service.list_api_keys(str(current_user["id"]), include_revoked=True)
        return _render(
            request,
            "keys.html",
            {
                "active_page": "keys",
                "keys": keys,
                "new_api_key": None,
                "new_api_key_name": None,
                "mcp_enabled": settings.MCP_ENABLED,
                "mcp_mount_path": settings.MCP_MOUNT_PATH,
                "mcp_sse_mount_path": settings.MCP_SSE_MOUNT_PATH,
                "mcp_guide": mcp_guide,
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

    mcp_guide = _build_request_mcp_guide(
        request,
        agent=agent,
        raw_host=mcp_host,
        api_key=new_raw_key,
    )
    keys = await user_service.list_api_keys(str(current_user["id"]), include_revoked=True)
    return _render(
        request,
        "keys.html",
        {
            "active_page": "keys",
            "keys": keys,
            "new_api_key": new_raw_key,
            "new_api_key_name": new_key_name,
            "mcp_enabled": settings.MCP_ENABLED,
            "mcp_mount_path": settings.MCP_MOUNT_PATH,
            "mcp_sse_mount_path": settings.MCP_SSE_MOUNT_PATH,
            "mcp_guide": mcp_guide,
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
    q: Optional[str] = Query(default=None),
    scope: Optional[str] = Query(default="auto"),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> Response:
    """Render the Grounded Dictionary AI Chat page."""
    if not current_user:
        next_target = "/chat"
        if request.url.query:
            next_target = f"/chat?{request.url.query}"
        return _redirect_to_login(next_target)

    cfg = await chat_service.resolve_inference_config()
    return _render(
        request,
        "chat.html",
        {
            "active_page": "chat",
            "inference_provider": cfg.provider,
            "inference_model": cfg.model,
            "prefill_message": (q or "").strip(),
            "selected_scope": _normalize_scope(scope),
            "include_context_checked": True,
            "include_related_checked": False,
            "messages": [],
            "error": None,
        },
        current_user=current_user,
    )


@web_router.post("/chat", response_class=HTMLResponse)
async def chat_submit(
    request: Request,
    message: str = Form(default=""),
    scope: str = Form(default="auto"),
    include_context: Optional[str] = Form(default=None),
    include_related: Optional[str] = Form(default=None),
    csrf_token: str = Form(default=""),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> Response:
    """Standard HTML form fallback for AI Chat (works even with JavaScript disabled)."""
    if not current_user:
        return _redirect_to_login("/chat")

    cfg = await chat_service.resolve_inference_config()
    resolved_scope = _normalize_scope(scope)
    use_ctx = bool(include_context and include_context.lower() in {"on", "true", "1", "yes"})
    use_related = bool(include_related and include_related.lower() in {"on", "true", "1", "yes"})

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
        messages.append({"role": "user", "content": message.strip(), "citations": []})
        try:
            chat_resp = await chat_service.chat(
                ChatRequest(
                    message=message.strip(),
                    include_context=use_ctx,
                    scope=resolved_scope,
                    include_related=use_related,
                ),
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": chat_resp.reply,
                    "citations": [c.model_dump() for c in chat_resp.citations],
                    "nlu": chat_resp.nlu.model_dump() if chat_resp.nlu else None,
                    "warnings": list(chat_resp.warnings),
                    "search_duration_ms": chat_resp.search_duration_ms,
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
            "prefill_message": "",
            "selected_scope": resolved_scope,
            "include_context_checked": use_ctx,
            "include_related_checked": use_related,
            "messages": messages,
            "error": error,
        },
        current_user=current_user,
        status_code=status_code,
    )


@web_router.get("/explore", response_class=HTMLResponse)
async def explore_page(
    request: Request,
    id: Optional[str] = Query(default=None),
    entity_type: Optional[str] = Query(default=None),
    vocab: Optional[str] = Query(default=None),
    code: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None),
    scope: Optional[str] = Query(default="auto"),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user),
) -> Response:
    """Render the Entity Drill-Down & Ontology Explorer page (Graph-UI ready)."""
    if not current_user:
        next_target = "/explore"
        if request.url.query:
            next_target = f"/explore?{request.url.query}"
        return _redirect_to_login(next_target)

    resolved_scope = _normalize_scope(scope)
    has_lookup = bool((id and id.strip()) or (code and code.strip()))
    has_search = bool(q and q.strip())

    neighborhood: Optional[EntityNeighborhood] = None
    seed_citation: Optional[Dict[str, Any]] = None
    relation_groups: List[Dict[str, Any]] = []
    fhir_coding_json: Optional[str] = None
    graph_payload_json: Optional[str] = None
    ask_chat_url: Optional[str] = None
    search_hits: List[Dict[str, Any]] = []
    search_nlu: Optional[Dict[str, Any]] = None
    warnings: List[str] = []
    error: Optional[str] = None
    status_code = status.HTTP_200_OK

    if has_lookup:
        try:
            neighborhood = await search_service.inspect_entity(
                node_id=id,
                entity_type=entity_type,
                vocabulary_id=vocab,
                code=code,
                related_limit=50,
            )
        except Exception as exc:
            error = f"Unable to inspect entity from Neo4j: {exc}"
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        if neighborhood is None and error is None:
            target_desc = code or id or "requested entity"
            error = f"Dictionary entity '{target_desc}' was not found in the active Neo4j database."
            status_code = status.HTTP_404_NOT_FOUND
        elif neighborhood is not None:
            cit = search_hit_to_citation(neighborhood.seed, related_limit=50)
            seed_citation = cit.model_dump()
            relation_groups = _group_neighborhood_tables(neighborhood)
            fhir_coding_json = json.dumps(neighborhood.seed.coding.model_dump(exclude_none=True), indent=2)
            graph_payload_json = json.dumps(
                {
                    "seed_id": neighborhood.seed.id,
                    "nodes": [n.model_dump() for n in neighborhood.nodes],
                    "edges": [e.model_dump() for e in neighborhood.edges],
                }
            )
            ask_prompt = f"Explain {cit.identifier} ({cit.name}) and its related ontology concepts"
            ask_chat_url = f"/chat?q={quote(ask_prompt)}"

    elif has_search:
        entity_filter = (
            ["clinical"] if resolved_scope == "clinical" else (["dicom"] if resolved_scope == "dicom" else None)
        )
        try:
            resp = await search_service.search(
                q.strip(),  # type: ignore[union-attr]
                entity_types=entity_filter,
                include_related=False,
                limit=20,
            )
            search_hits = [search_hit_to_citation(h).model_dump() for h in resp.hits]
            search_nlu = {
                "intent": resp.nlu.intent.value,
                "rewritten_query": resp.nlu.rewritten_query,
                "duration_ms": resp.duration_ms,
            }
            warnings = list(resp.warnings)
        except Exception as exc:
            error = f"Dictionary search failed: {exc}"
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return _render(
        request,
        "explore.html",
        {
            "active_page": "explore",
            "query_text": (q or "").strip(),
            "selected_scope": resolved_scope,
            "neighborhood": neighborhood,
            "seed": neighborhood.seed if neighborhood else None,
            "seed_citation": seed_citation,
            "relation_groups": relation_groups,
            "fhir_coding_json": fhir_coding_json,
            "graph_payload_json": graph_payload_json,
            "graph_node_count": len(neighborhood.nodes) if neighborhood else 0,
            "graph_edge_count": len(neighborhood.edges) if neighborhood else 0,
            "ask_chat_url": ask_chat_url,
            "search_hits": search_hits,
            "search_nlu": search_nlu,
            "warnings": warnings,
            "error": error,
        },
        current_user=current_user,
        status_code=status_code,
    )
