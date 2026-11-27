import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ROOT_ENV_FILE = PROJECT_ROOT / ".env"
_TESTS_ENV_FILE = PROJECT_ROOT / "tests" / ".env"


def _resolve_env_files() -> tuple[Path, ...]:
    """Resolve .env files: project-root .env first, then tests/.env during pytest runs."""
    env_files: list[Path] = [_ROOT_ENV_FILE]
    if "pytest" in sys.modules:
        env_files.append(_TESTS_ENV_FILE)
    cwd_env = Path(".env").resolve()
    if cwd_env not in (_ROOT_ENV_FILE, _TESTS_ENV_FILE):
        env_files.append(cwd_env)
    return tuple(env_files)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GOSMART_DS_",
        env_file=_resolve_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server Core Settings
    APP_NAME: str = Field(default="GoSmart DS", description="Application Name")
    APP_VERSION: str = Field(default="0.1.0", description="API Version")
    DEBUG: bool = Field(default=False, description="Debug mode flag")
    DEV_MODE: bool = Field(
        default=False,
        description="Enable development mode (relaxes Neo4j offline connectivity failures during startup)",
    )
    HOST: str = Field(default="0.0.0.0", description="FastAPI host binding")
    PORT: int = Field(default=8081, description="FastAPI HTTP port")
    RELOAD: bool = Field(default=True, description="Uvicorn auto-reload")
    API_V1_PREFIX: str = Field(default="/api/v1", description="API v1 route prefix")

    # Logging Configuration (structlog & rotation)
    LOG_LEVEL: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    LOG_FORMAT: str = Field(default="json", description="Console log format: 'json' or 'console'")
    LOG_DIR: str = Field(default="./logs", description="Directory where rotating log files are saved")
    LOG_FILE: str = Field(default="gosmart_ds.log", description="Base name for log file")
    LOG_ROTATION_DAYS: int = Field(default=7, description="Auto-rotate log files every N days (default: 7)")
    LOG_BACKUP_COUNT: int = Field(default=8, description="Number of rotated 7-day backup files to keep")
    LOG_TO_FILE: bool = Field(default=True, description="Enable rotating file logging")

    # Graph Database (Neo4j)
    DB_CONNECTION_URI: str = Field(
        default="bolt://localhost:7688",
        validation_alias=AliasChoices(
            "GOSMART_DS_DB_CONNECTION_URI",
            "NEO4J_URI",
        ),
        description="Neo4j connection URI (e.g. bolt://localhost:7688 or neo4j://...)",
    )
    DB_USER: str = Field(
        default="neo4j",
        validation_alias=AliasChoices(
            "GOSMART_DS_DB_USER",
            "NEO4J_USERNAME",
        ),
        description="Neo4j username",
    )
    #
    # NOTE: To read SecretStr from environment variables, use .get_secret_value() method,
    # e.g. get_settings().DB_PASSWORD.get_secret_value()
    DB_PASSWORD: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices(
            "GOSMART_DS_DB_PASSWORD",
            "NEO4J_PASSWORD",
        ),
        description="Neo4j password (empty string when NEO4J_AUTH=none)",
    )
    DB_DATABASE: str = Field(
        default="neo4j",
        validation_alias=AliasChoices(
            "GOSMART_DS_DB_DATABASE",
            "NEO4J_DATABASE",
        ),
        description="Neo4j database name",
    )
    DB_MAX_CONNECTION_LIFETIME: int = Field(default=3600, description="Max connection lifetime in seconds")
    DB_MAX_CONNECTION_POOL_SIZE: int = Field(default=50, description="Max connection pool size")
    DB_CONNECT_RETRY_TIMEOUT_SEC: int = Field(
        default=600,
        ge=0,
        description=(
            "Maximum total seconds to retry Neo4j connectivity in production mode "
            "(when DEV_MODE is False) before failing startup; 0 retries indefinitely"
        ),
    )
    DB_CONNECT_RETRY_INITIAL_DELAY_SEC: float = Field(
        default=2.0,
        gt=0.0,
        description="Initial backoff delay in seconds between Neo4j connectivity retries",
    )
    DB_CONNECT_RETRY_MAX_DELAY_SEC: float = Field(
        default=30.0,
        gt=0.0,
        description="Maximum backoff delay in seconds between Neo4j connectivity retries",
    )
    DB_AURA_INSTANCE_ID: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "GOSMART_DS_DB_AURA_INSTANCE_ID",
            "GOSMART_DS_AURA_INSTANCEID",
            "AURA_INSTANCEID",
        ),
        description="Neo4j Aura Instance ID (optional metadata)",
    )
    DB_AURA_INSTANCE_NAME: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "GOSMART_DS_DB_AURA_INSTANCE_NAME",
            "GOSMART_DS_AURA_INSTANCENAME",
            "AURA_INSTANCENAME",
        ),
        description="Neo4j Aura Instance Name (optional metadata)",
    )

    # Default LLM & Vector Embedding Bootstrap Configuration
    VECTOR_EMBED_PROVIDER: str = Field(
        default="ollama",
        description="Default vector embedding LLM provider (e.g. 'ollama', 'vertexai', 'googleai')",
    )
    VECTOR_EMBED_SERVER_URL: str = Field(
        default="http://localhost:11434",
        description="Default vector embedding server base URL (for Ollama or custom endpoints)",
    )
    VECTOR_EMBED_MODEL: str = Field(
        default="embeddinggemma-2:270m",
        description="Default vector embedding model name (e.g. 'embeddinggemma-2:270m', 'gemini-embedding-001')",
    )
    VECTOR_EMBED_DIMENSIONS: int = Field(
        default=768,
        gt=0,
        description="Default output vector dimensionality (default: 768)",
    )
    VECTOR_EMBED_TAG: str = Field(
        default="embedding-v1",
        description="Default LLM_Config tag used for vector embedding computation",
    )
    INFERENCE_PROVIDER: str = Field(
        default="ollama",
        description="Default inference LLM provider (e.g. 'ollama', 'vertexai', 'googleai')",
    )
    INFERENCE_SERVER_URL: str = Field(
        default="http://localhost:11434",
        description="Default inference LLM server base URL",
    )
    INFERENCE_MODEL: str = Field(
        default="gemma4:e2b",
        description="Default inference LLM model name (e.g. 'gemma4:e2b', 'gemini-2.5-flash')",
    )
    INFERENCE_TAG: str = Field(
        default="inference-v1",
        description="Default LLM_Config tag used for inference and disambiguation flows",
    )
    VERTEX_AI_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="Optional Google Cloud / Vertex AI API key (referenced via '${GOSMART_DS_VERTEX_AI_API_KEY}')",
    )
    VERTEX_AI_LOCATION: str = Field(
        default="us-central1",
        description="Default Google Cloud / Vertex AI region",
    )
    VERTEX_AI_PROJECT_ID: Optional[str] = Field(
        default=None,
        description="Optional Google Cloud Project ID for Vertex AI",
    )

    # Clinical NLU & Query Expansion Configuration
    NLU_MODEL_PATH: str = Field(
        default="./nlu_models/intent_models.zip",
        description="Path to the trained gosmart-clinical-nlu spaCy model bundle (.zip)",
    )
    NLU_LLM_EXPANSION_ENABLED: bool = Field(
        default=True,
        description="Synthesize alternate clinical/DICOM search phrases via the inference LLM during NLU parsing",
    )
    NLU_MAX_ALTERNATE_PHRASES: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of alternate search phrases synthesized per query (default: 3)",
    )
    NLU_SCHEMA_CACHE_TTL_SEC: float = Field(
        default=60.0,
        ge=0.0,
        description="TTL in seconds for caching loaded Neo4j coding schemas during NLU schema checks",
    )

    # Dynamic LLM Configuration Security Settings
    LLM_CONFIG_ENCRYPTION_KEY: Optional[SecretStr] = Field(
        default=None,
        description=(
            "Optional Fernet key or passphrase for encrypting stored LLM_Config api_key values at rest in Neo4j"
        ),
    )

    # Authentication & JWT Security Settings
    JWT_SECRET_KEY: SecretStr = Field(
        default=SecretStr("gosmart-ds-dev-jwt-secret-key-change-in-production-min32chars"),
        description="Secret key for signing and verifying JWT tokens",
    )
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT signing algorithm (e.g. HS256, RS256)")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=60, description="Access token expiration time in minutes (default: 60)"
    )
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=30, description="Refresh token expiration time in days (default: 30)"
    )
    API_KEY_PREFIX: str = Field(default="gs_dict_", description="Standard prefix for generated API keys")
    SESSION_COOKIE_NAME: str = Field(
        default="gosmart_ds_session",
        description="HttpOnly browser session cookie name for the web portal",
    )
    SESSION_COOKIE_SECURE: bool = Field(
        default=False,
        description="Mark session cookies as Secure (set true when served over HTTPS / Cloudflare)",
    )
    GOOGLE_OAUTH_CLIENT_ID: Optional[str] = Field(
        default=None, description="Google OAuth2 / OIDC Client ID for federated authentication"
    )
    GOOGLE_OAUTH_CLIENT_SECRET: Optional[SecretStr] = Field(default=None, description="Google OAuth2 Client Secret")

    # Brevo Transactional Email & Self-Service Registration Settings
    BREVO_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="Brevo transactional email API key (sent via api-key header); optional in DEV_MODE",
    )
    BREVO_API_URL: str = Field(
        default="https://api.brevo.com/v3/smtp/email",
        description="Brevo transactional email API endpoint URL",
    )
    BREVO_SENDER_EMAIL: str = Field(
        default="no-reply@gosmart.health",
        description="Verified sender email address used for Brevo transactional emails",
    )
    BREVO_SENDER_NAME: str = Field(
        default="GoSmart Health Dictionary Service",
        description="Sender display name used for Brevo transactional emails",
    )
    PUBLIC_BASE_URL: str = Field(
        default="http://127.0.0.1:8081",
        description="Public base URL used to construct email verification links",
    )
    EMAIL_VERIFICATION_EXPIRE_HOURS: int = Field(
        default=24,
        gt=0,
        description="Hours before an account email verification token expires (default: 24)",
    )

    # Initial Platform Admin Bootstrapping & Recovery
    INITIAL_ADMIN_EMAIL: Optional[str] = Field(
        default="admin@gosmart.health",
        description="Initial bootstrap Platform Admin email",
    )
    INITIAL_ADMIN_PASSWORD: Optional[SecretStr] = Field(
        default=None,
        description="Initial bootstrap Platform Admin password (if set, auto-creates superuser on startup)",
    )
    INITIAL_ADMIN_USERNAME: str = Field(
        default="admin",
        description="Initial bootstrap Platform Admin username",
    )
    INITIAL_ADMIN_DISPLAY_NAME: str = Field(
        default="System Administrator",
        description="Initial Platform Admin display name",
    )
    FORCE_ADMIN_PASSWORD_RESET: bool = Field(
        default=False,
        description="Force reset admin password to INITIAL_ADMIN_PASSWORD on startup if account already exists",
    )

    # Audit Tracking Settings
    AUDIT_RETENTION_DAYS: int = Field(
        default=365,
        description="Audit log retention period in days (default: 365 days / 1 year)",
    )
    AUDIT_LOG_TO_DB: bool = Field(
        default=True,
        description="Enable Neo4j database persistence for audit logs",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
