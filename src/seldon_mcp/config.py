from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class SeldonConfig(BaseSettings):
    """Configuration for the Seldon MCP server, loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    neuralk_api_key: str | None = None
    neuralk_prediction_url: str = "https://api.prediction.neuralk-ai.com"
    seldon_default_model: str = "seldon-small"
    # Retention tier (in days) for uploaded datasets, one of 1/7/30/90. None
    # leaves it to the Neuralk server default (90 days). Applied by upload_data.
    seldon_upload_ttl_days: int | None = None
    # Where full-prediction download files are written (defaults to a temp dir),
    # and how long they live before the sweeper deletes them.
    seldon_download_dir: str | None = None
    seldon_download_ttl_seconds: int = 300
    # Hosted mode. Every MCP request must carry the client's own key
    # (x-neuralk-api-key or Authorization: Bearer); the server's NEURALK_API_KEY,
    # if any, is never used on a client's behalf, and an MCP request without a
    # key is refused with 401 before it reaches a tool.
    require_client_api_key: bool = False
    # Public base URL of this server, e.g. https://mcp.neuralk.ai. Download
    # links are built from it; empty = derive from each request's own URL.
    seldon_public_url: str | None = None
    # API-key validation against the neuralk-saas auth API (GET /api/v1/auth/whoami).
    # base URL reuses neuralk_prediction_url (same host).
    skip_api_key_validation: bool = False
    api_key_validation_ttl_s: int = 300
    api_key_validation_timeout_s: float = 5.0

    @property
    def public_url(self) -> str | None:
        """The configured public base URL, or None."""
        return self.seldon_public_url or None
