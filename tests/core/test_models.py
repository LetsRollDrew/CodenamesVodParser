from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.core.models import GameRecord, VodMeta


def test_typed_models_validate_core_constraints() -> None:
    VodMeta(
        vod_id="vod-001",
        streamer_login="tiewhy",
        created_at=datetime.now(timezone.utc),
        title="Test VOD",
        url="https://twitch.tv/videos/1",
        duration_seconds=3600,
    )

    with pytest.raises(ValidationError):
        GameRecord(
            vod_id="vod-001",
            game_index=0,
            start_sec=0,
            end_sec=10,
            parse_confidence=1.5,
        )
