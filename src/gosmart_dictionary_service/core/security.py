import base64
import hashlib
import hmac
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from cryptography.fernet import Fernet, InvalidToken

from gosmart_dictionary_service.core.config import get_settings

# Standard Argon2id password hasher (OWASP recommended defaults)
_password_hasher = PasswordHasher()

_ENV_VAR_EXACT_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_ENV_VAR_SUB_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_FERNET_PREFIX = "enc:fernet:"


def hash_password(password: str) -> str:
    """Hash a plaintext password using Argon2id."""
    return _password_hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against an Argon2id hash."""
    try:
        return _password_hasher.verify(hashed_password, plain_password)
    except VerifyMismatchError, Exception:
        return False


def hash_token(raw_token: str) -> str:
    """Compute a deterministic SHA-256 hex digest of an opaque token or API key."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def hash_api_key(raw_api_key: str) -> str:
    """Compute a deterministic SHA-256 hex digest of a raw API key."""
    return hash_token(raw_api_key)


def generate_verification_token() -> Tuple[str, str]:
    """Generate a URL-safe email verification token, returning ``(raw_token, sha256_hash)``."""
    raw_token = secrets.token_urlsafe(32)
    return raw_token, hash_token(raw_token)


def generate_csrf_token() -> str:
    """Generate a random URL-safe CSRF token for HTML form protection."""
    return secrets.token_urlsafe(32)


def verify_csrf_token(expected_token: Optional[str], provided_token: Optional[str]) -> bool:
    """Verify a submitted CSRF token against the cookie value using constant-time comparison."""
    if not expected_token or not provided_token:
        return False
    return hmac.compare_digest(expected_token.strip(), provided_token.strip())


def generate_api_key(prefix: Optional[str] = None) -> Tuple[str, str, str]:
    """Generate a secure, random API key with visible identifier prefix.

    Returns ``(raw_key, visible_prefix, sha256_hash)`` where:
    - ``raw_key`` has the form ``gs_dict_<8hex>_<secret>``
    - ``visible_prefix`` has the form ``gs_dict_<8hex>`` for non-secret UI display
    """
    settings = get_settings()
    base_prefix = prefix or settings.API_KEY_PREFIX
    key_id_part = secrets.token_hex(4)
    visible_prefix = f"{base_prefix}{key_id_part}"
    secret_part = secrets.token_urlsafe(32)
    raw_key = f"{visible_prefix}_{secret_part}"
    key_hash = hash_api_key(raw_key)
    return raw_key, visible_prefix, key_hash


def verify_api_key(raw_api_key: str, stored_hash: str) -> bool:
    """Verify a raw API key against a stored SHA-256 hash using constant-time comparison."""
    calculated_hash = hash_api_key(raw_api_key)
    return hmac.compare_digest(calculated_hash, stored_hash)


def _get_effective_llm_encryption_key(explicit_key: Optional[str] = None) -> Optional[str]:
    if explicit_key is not None:
        cleaned = explicit_key.strip()
        return cleaned if cleaned else None
    settings = get_settings()
    if settings.LLM_CONFIG_ENCRYPTION_KEY is not None:
        cleaned = settings.LLM_CONFIG_ENCRYPTION_KEY.get_secret_value().strip()
        return cleaned if cleaned else None
    env_fallback = os.environ.get("GOSMART_DS_LLM_CONFIG_ENCRYPTION_KEY", "").strip()
    return env_fallback if env_fallback else None


def _build_fernet(secret_key: str) -> Fernet:
    raw_bytes = secret_key.encode("utf-8")
    try:
        return Fernet(raw_bytes)
    except Exception:
        derived = base64.urlsafe_b64encode(hashlib.sha256(raw_bytes).digest())
        return Fernet(derived)


def is_env_var_reference(value: Optional[str]) -> bool:
    """Return True if value is a ${ENV_VAR} placeholder."""
    if not value:
        return False
    return bool(_ENV_VAR_EXACT_RE.match(value.strip()))


def classify_llm_api_key(stored_value: Optional[str]) -> str:
    """Classify the storage format of an LLM_Config api_key: none, env, encrypted, or plaintext."""
    if not stored_value or not stored_value.strip():
        return "none"
    val = stored_value.strip()
    if is_env_var_reference(val):
        return "env"
    if val.startswith(_FERNET_PREFIX):
        return "encrypted"
    return "plaintext"


def encrypt_llm_api_key(raw_value: Optional[str], encryption_key: Optional[str] = None) -> str:
    """Prepare an LLM API key for Neo4j storage.

    - Empty strings stay empty.
    - `${ENV_VAR}` references are stored literally so runtime reads from env / Settings.
    - Already-encrypted `enc:fernet:...` values are preserved as-is.
    - Literal secrets are encrypted with Fernet when `LLM_CONFIG_ENCRYPTION_KEY` is configured.
    """
    if not raw_value or not raw_value.strip():
        return ""
    val = raw_value.strip()
    if is_env_var_reference(val) or val.startswith(_FERNET_PREFIX):
        return val

    effective_key = _get_effective_llm_encryption_key(encryption_key)
    if not effective_key:
        return val

    fernet = _build_fernet(effective_key)
    token = fernet.encrypt(val.encode("utf-8")).decode("utf-8")
    return f"{_FERNET_PREFIX}{token}"


def _lookup_env_or_settings(var_name: str) -> Optional[str]:
    """Resolve an env var from ``os.environ`` first, falling back to ``Settings`` for ``GOSMART_DS_*``."""
    resolved = os.environ.get(var_name)
    if resolved is not None and resolved.strip():
        return resolved.strip()
    if var_name.startswith("GOSMART_DS_"):
        attr_name = var_name.removeprefix("GOSMART_DS_")
        settings_val = getattr(get_settings(), attr_name, None)
        if settings_val is not None:
            raw = settings_val.get_secret_value() if hasattr(settings_val, "get_secret_value") else str(settings_val)
            if raw.strip():
                return raw.strip()
    return None


def resolve_llm_api_key(stored_value: Optional[str], encryption_key: Optional[str] = None) -> str:
    """Resolve a stored LLM_Config api_key into a runtime credential.

    Decrypts `enc:fernet:...` payloads and expands `${ENV_VAR}` placeholders from
    `os.environ` (or `.env`-loaded `Settings` for `GOSMART_DS_*` variables).
    """
    if not stored_value or not stored_value.strip():
        return ""
    val = stored_value.strip()

    if val.startswith(_FERNET_PREFIX):
        effective_key = _get_effective_llm_encryption_key(encryption_key)
        if not effective_key:
            raise ValueError("Cannot decrypt stored LLM api_key: GOSMART_DS_LLM_CONFIG_ENCRYPTION_KEY is not set.")
        token = val[len(_FERNET_PREFIX) :].encode("utf-8")
        try:
            fernet = _build_fernet(effective_key)
            val = fernet.decrypt(token).decode("utf-8")
        except InvalidToken as exc:
            raise ValueError("Failed to decrypt stored LLM api_key with configured encryption key.") from exc

    def _replace_env(match: re.Match[str]) -> str:
        var_name = match.group(1)
        resolved = _lookup_env_or_settings(var_name)
        if resolved is None:
            raise ValueError(f"Environment variable '{var_name}' referenced by LLM_Config api_key is not set.")
        return resolved

    if "${" in val:
        val = _ENV_VAR_SUB_RE.sub(_replace_env, val)

    return val.strip()


def mask_llm_api_key(stored_value: Optional[str]) -> str:
    """Return a redacted representation of an LLM_Config api_key safe for API/MCP/CLI display."""
    if not stored_value or not stored_value.strip():
        return ""
    val = stored_value.strip()
    if is_env_var_reference(val):
        return val
    if val.startswith(_FERNET_PREFIX):
        return f"{_FERNET_PREFIX}********"
    if len(val) <= 8:
        return "****"
    return f"{val[:4]}****{val[-4:]}"


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Generate a signed JWT Access Token containing user claims and expiration."""
    settings = get_settings()
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update(
        {
            "iat": int(now.timestamp()),
            "exp": int(expire.timestamp()),
            "token_type": "access",
        }
    )
    secret = settings.JWT_SECRET_KEY.get_secret_value()
    return jwt.encode(to_encode, secret, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Generate a signed JWT Refresh Token with extended validity."""
    settings = get_settings()
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode.update(
        {
            "iat": int(now.timestamp()),
            "exp": int(expire.timestamp()),
            "token_type": "refresh",
        }
    )
    secret = settings.JWT_SECRET_KEY.get_secret_value()
    return jwt.encode(to_encode, secret, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str, verify_exp: bool = True) -> Dict[str, Any]:
    """Decode and validate a signed JWT token.

    Raises jwt.PyJWTError (ExpiredSignatureError, InvalidTokenError, etc.) on invalid tokens.
    """
    settings = get_settings()
    secret = settings.JWT_SECRET_KEY.get_secret_value()
    options = {"verify_exp": verify_exp}
    return jwt.decode(
        token,
        secret,
        algorithms=[settings.JWT_ALGORITHM],
        options=options,
    )
