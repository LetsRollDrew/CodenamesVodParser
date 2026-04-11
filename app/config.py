"""Application settings and configuration helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SQLITE_URL = "sqlite:///data/codenames_vod_parser.db"


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing."""


class Settings(BaseSettings):
    """Typed application settings loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
    )

    twitch_client_id: str | None = None
    twitch_access_token: str | None = None
    twitch_client_secret: str | None = None
    twitch_streamer_login: str = "tiewhy"
    ffmpeg_path: str = "ffmpeg"
    sqlite_url: str = DEFAULT_SQLITE_URL
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("twitch_client_id", "twitch_access_token", "twitch_client_secret", mode="before")
    @classmethod
    def _normalize_optional_string(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        return value

    @field_validator("twitch_streamer_login", "ffmpeg_path", "sqlite_url", mode="before")
    @classmethod
    def _normalize_required_string(cls, value: object, info: ValidationInfo) -> object:
        if not isinstance(value, str):
            return value

        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{info.field_name} must not be blank")
        return cleaned

    @field_validator("twitch_streamer_login")
    @classmethod
    def _normalize_streamer_login(cls, value: str) -> str:
        return value.lower()

    @property
    def sqlite_path(self) -> Path | None:
        """Return a filesystem path for sqlite URLs that use the sqlite:/// scheme."""

        prefix = "sqlite:///"
        if not self.sqlite_url.startswith(prefix):
            return None
        return Path(self.sqlite_url.removeprefix(prefix))

    def ensure_runtime_directories(self) -> None:
        """Create local runtime directories for path-based SQLite databases."""

        sqlite_path = self.sqlite_path
        if sqlite_path is None:
            return
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    def require_twitch_api_credentials(self) -> None:
        """Validate that the Twitch API credentials required for Helix access are set."""

        missing: list[str] = []
        if not self.twitch_client_id:
            missing.append("TWITCH_CLIENT_ID")
        if not self.twitch_access_token:
            missing.append("TWITCH_ACCESS_TOKEN")

        if missing:
            names = ", ".join(missing)
            raise ConfigurationError(f"Missing required Twitch API settings: {names}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()
