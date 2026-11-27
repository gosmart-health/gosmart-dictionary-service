"""Transactional email delivery service backed by Brevo with DEV_MODE local fallback."""

from __future__ import annotations

import html
from typing import Literal, Optional
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, Field

from gosmart_dictionary_service.core.config import Settings, get_settings
from gosmart_dictionary_service.core.logging import get_logger

logger = get_logger("gosmart_ds.email")


class EmailConfigurationError(RuntimeError):
    """Raised when transactional email is required (non-DEV_MODE) but Brevo is unconfigured."""


class EmailDeliveryError(RuntimeError):
    """Raised when the Brevo API returns a non-success response or network error."""


class EmailDispatchResult(BaseModel):
    """Structured outcome of dispatching an account verification email."""

    sent: bool = Field(..., description="True when delivered to Brevo API; False in DEV_MODE local fallback")
    mode: Literal["brevo", "dev_local"] = Field(..., description="Delivery mode used")
    verification_url: str = Field(..., description="Verification URL generated for the registrant")
    message_id: Optional[str] = Field(default=None, description="Provider message ID returned by Brevo, if any")


class EmailService:
    """Dispatches transactional registration emails via Brevo HTTP API."""

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self._settings = settings
        self._http_client = http_client
        self._timeout_seconds = timeout_seconds

    @property
    def settings(self) -> Settings:
        return self._settings or get_settings()

    def build_verification_url(self, raw_token: str, base_url: Optional[str] = None) -> str:
        """Build the clickable email verification URL for the HTML web portal."""
        origin = (base_url or self.settings.PUBLIC_BASE_URL).strip().rstrip("/")
        query = urlencode({"token": raw_token})
        return f"{origin}/verify-email?{query}"

    def _get_brevo_api_key(self) -> Optional[str]:
        secret = self.settings.BREVO_API_KEY
        if secret is None:
            return None
        value = secret.get_secret_value().strip()
        return value if value else None

    async def send_verification_email(
        self,
        *,
        to_email: str,
        display_name: str,
        raw_token: str,
        base_url: Optional[str] = None,
    ) -> EmailDispatchResult:
        """Send an email verification link via Brevo, or fall back locally when ``DEV_MODE=True``."""
        verification_url = self.build_verification_url(raw_token, base_url=base_url)
        api_key = self._get_brevo_api_key()

        if not api_key:
            if self.settings.DEV_MODE:
                logger.warning(
                    "brevo_api_key_unset_dev_fallback",
                    to_email=to_email,
                    verification_url=verification_url,
                )
                return EmailDispatchResult(
                    sent=False,
                    mode="dev_local",
                    verification_url=verification_url,
                    message_id=None,
                )
            logger.error("brevo_api_key_missing_production", to_email=to_email)
            raise EmailConfigurationError(
                "Email verification is not configured on this server (GOSMART_DS_BREVO_API_KEY is unset)."
            )

        safe_name = html.escape(display_name or to_email)
        safe_url = html.escape(verification_url, quote=True)
        expire_hours = self.settings.EMAIL_VERIFICATION_EXPIRE_HOURS
        app_name = self.settings.APP_NAME

        payload = {
            "sender": {
                "name": self.settings.BREVO_SENDER_NAME,
                "email": self.settings.BREVO_SENDER_EMAIL,
            },
            "to": [
                {
                    "email": to_email,
                    "name": display_name or to_email,
                }
            ],
            "subject": f"Verify your {app_name} account",
            "htmlContent": (
                f"<p>Hello {safe_name},</p>"
                f"<p>Thank you for registering with <strong>{html.escape(app_name)}</strong>.</p>"
                f"<p>Please confirm your email address to activate your account and generate API keys:</p>"
                f'<p><a href="{safe_url}">Verify Email Address</a></p>'
                f"<p>Or copy and paste this link into your browser:<br><code>{safe_url}</code></p>"
                f"<p>This verification link expires in {expire_hours} hours.</p>"
            ),
            "textContent": (
                f"Hello {display_name or to_email},\n\n"
                f"Thank you for registering with {app_name}.\n"
                f"Verify your email address by visiting:\n{verification_url}\n\n"
                f"This link expires in {expire_hours} hours.\n"
            ),
        }
        headers = {
            "api-key": api_key,
            "accept": "application/json",
            "content-type": "application/json",
        }

        try:
            if self._http_client is not None:
                response = await self._http_client.post(
                    self.settings.BREVO_API_URL,
                    json=payload,
                    headers=headers,
                    timeout=self._timeout_seconds,
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                    response = await client.post(
                        self.settings.BREVO_API_URL,
                        json=payload,
                        headers=headers,
                    )
        except httpx.HTTPError as exc:
            logger.error("brevo_request_failed", to_email=to_email, error=str(exc))
            raise EmailDeliveryError(f"Failed to connect to Brevo email service: {exc}") from exc

        if response.status_code < 200 or response.status_code >= 300:
            logger.error(
                "brevo_api_error",
                to_email=to_email,
                status_code=response.status_code,
                response_text=response.text[:300],
            )
            raise EmailDeliveryError(f"Brevo API rejected verification email (HTTP {response.status_code}).")

        message_id: Optional[str] = None
        try:
            data = response.json()
            if isinstance(data, dict) and data.get("messageId"):
                message_id = str(data["messageId"])
        except Exception:
            message_id = None

        logger.info("brevo_verification_email_sent", to_email=to_email, message_id=message_id)
        return EmailDispatchResult(
            sent=True,
            mode="brevo",
            verification_url=verification_url,
            message_id=message_id,
        )


email_service = EmailService()

__all__ = [
    "EmailConfigurationError",
    "EmailDeliveryError",
    "EmailDispatchResult",
    "EmailService",
    "email_service",
]
