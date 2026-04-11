import json
from typing import Mapping

import pytest

from app.config import Settings
from app.twitch_client import HttpResponse, TwitchClient, TwitchClientError, parse_twitch_duration


class FakeTransport:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self._responses = list(responses)
        self.urls: list[str] = []
        self.headers: list[Mapping[str, str]] = []

    def __call__(self, url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
        del timeout
        self.urls.append(url)
        self.headers.append(headers)
        if not self._responses:
            raise AssertionError("No fake responses left for Twitch transport")
        return self._responses.pop(0)


def _json_response(payload: dict, *, status_code: int = 200, headers: dict[str, str] | None = None) -> HttpResponse:
    return HttpResponse(
        status_code=status_code,
        headers=headers or {},
        body=json.dumps(payload).encode("utf-8"),
    )


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        twitch_client_id="client-id",
        twitch_access_token="access-token",
    )


@pytest.mark.parametrize(
    ("duration_text", "expected_seconds"),
    [
        ("6h26m14s", 23174),
        ("3h4m", 11040),
        ("55m32s", 3332),
    ],
)
def test_parse_twitch_duration(duration_text: str, expected_seconds: int) -> None:
    assert parse_twitch_duration(duration_text) == expected_seconds


def test_get_broadcaster_id_returns_first_matching_user_id() -> None:
    transport = FakeTransport(
        [
            _json_response({"data": [{"id": "8675309", "login": "tiewhy"}]}),
        ]
    )
    client = TwitchClient(settings=_settings(), transport=transport)

    broadcaster_id = client.get_broadcaster_id("TieWhy")

    assert broadcaster_id == "8675309"
    assert "login=tiewhy" in transport.urls[0]


def test_list_archive_vods_paginates_across_multiple_pages() -> None:
    transport = FakeTransport(
        [
            _json_response({"data": [{"id": "42", "login": "tiewhy"}]}),
            _json_response(
                {
                    "data": [
                        {
                            "id": "v1",
                            "created_at": "2026-04-10T12:00:00Z",
                            "title": "stream one",
                            "url": "https://twitch.tv/videos/v1",
                            "duration": "1h1m1s",
                        }
                    ],
                    "pagination": {"cursor": "page-2"},
                }
            ),
            _json_response(
                {
                    "data": [
                        {
                            "id": "v2",
                            "created_at": "2026-04-11T12:00:00Z",
                            "title": "stream two",
                            "url": "https://twitch.tv/videos/v2",
                            "duration": "55m32s",
                        }
                    ],
                    "pagination": {},
                }
            ),
        ]
    )
    client = TwitchClient(settings=_settings(), transport=transport)

    vods = client.list_archive_vods("tiewhy")

    assert [vod.vod_id for vod in vods] == ["v1", "v2"]
    assert vods[0].duration_seconds == 3661
    assert "type=archive" in transport.urls[1]
    assert "after=page-2" in transport.urls[2]


def test_rate_limited_request_retries_then_succeeds() -> None:
    sleep_calls: list[float] = []
    transport = FakeTransport(
        [
            _json_response({}, status_code=429, headers={"Retry-After": "0.25"}),
            _json_response({"data": [{"id": "123"}]}),
        ]
    )
    client = TwitchClient(
        settings=_settings(),
        transport=transport,
        sleep_fn=sleep_calls.append,
    )

    broadcaster_id = client.get_broadcaster_id("tiewhy")

    assert broadcaster_id == "123"
    assert sleep_calls == [0.25]


def test_missing_broadcaster_raises() -> None:
    transport = FakeTransport([_json_response({"data": []})])
    client = TwitchClient(settings=_settings(), transport=transport)

    with pytest.raises(TwitchClientError, match="No Twitch broadcaster found"):
        client.get_broadcaster_id("missing-user")
