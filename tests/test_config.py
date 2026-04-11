from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import ConfigurationError, Settings


def test_settings_load_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "TWITCH_CLIENT_ID=test-client",
                "TWITCH_ACCESS_TOKEN=test-token",
                "TWITCH_STREAMER_LOGIN=TieWhy",
                "FFMPEG_PATH= C:/ffmpeg/bin/ffmpeg.exe ",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.twitch_client_id == "test-client"
    assert settings.twitch_access_token == "test-token"
    assert settings.twitch_streamer_login == "tiewhy"
    assert settings.ffmpeg_path == "C:/ffmpeg/bin/ffmpeg.exe"


def test_require_twitch_api_credentials_raises_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("TWITCH_CLIENT_ID", raising=False)
    monkeypatch.delenv("TWITCH_ACCESS_TOKEN", raising=False)

    settings = Settings(
        _env_file=None,
        twitch_client_id=None,
        twitch_access_token=None,
    )

    with pytest.raises(ConfigurationError, match="TWITCH_CLIENT_ID, TWITCH_ACCESS_TOKEN"):
        settings.require_twitch_api_credentials()


def test_blank_ffmpeg_path_is_rejected() -> None:
    with pytest.raises(ValidationError, match="ffmpeg_path must not be blank"):
        Settings(ffmpeg_path="   ")
