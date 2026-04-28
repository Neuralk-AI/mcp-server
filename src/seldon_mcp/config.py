from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class SeldonConfig(BaseSettings):
    """Configuration for the Seldon MCP server, loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    neuralk_api_key: str | None = None
    neuralk_host: str | None = None
    seldon_default_model: str = "seldon-small"
    seldon_data_dir: str = "."
