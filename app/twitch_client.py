"""Twitch Helix client for broadcaster lookup and archive VOD discovery."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.config import Settings, get_settings
from app.models import VodMeta

HELIX_BASE_URL = "https://api.twitch.tv/helix"
MAX_PAGE_SIZE = 100
_DURATION_PATTERN = re.compile(r"^(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?(?:(?P<seconds>\d+)s)?$")


class TwitchClientError(RuntimeError):
    """Raised when the Twitch Helix client cannot fulfill a request."""


@dataclass(frozen=True)
class HttpResponse:
    """A minimal HTTP response container used by the injected transport."""

    status_code: int
    headers: Mapping[str, str]
    body: bytes


Transport = Callable[[str, Mapping[str, str], float], HttpResponse]
SleepFn = Callable[[float], None]
TimeFn = Callable[[], float]


def parse_twitch_duration(duration_text: str) -> int:
    """Convert Twitch duration strings like `6h26m14s` into seconds."""

    match = _DURATION_PATTERN.fullmatch(duration_text.strip())
    if match is None:
        raise ValueError(f"Invalid Twitch duration: {duration_text!r}")

    parts = match.groupdict(default="0")
    total_seconds = (
        int(parts["hours"]) * 3600
        + int(parts["minutes"]) * 60
        + int(parts["seconds"])
    )
    if total_seconds == 0 and duration_text.strip() != "0s":
        raise ValueError(f"Invalid Twitch duration: {duration_text!r}")
    return total_seconds


def _default_transport(url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
    request = Request(url=url, headers=dict(headers), method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return HttpResponse(
                status_code=response.status,
                headers=dict(response.headers.items()),
                body=response.read(),
            )
    except HTTPError as exc:
        return HttpResponse(
            status_code=exc.code,
            headers=dict(exc.headers.items()) if exc.headers is not None else {},
            body=exc.read(),
        )
    except URLError as exc:
        raise TwitchClientError(f"Failed to reach Twitch Helix: {exc.reason}") from exc


class TwitchClient:
    """Thin Helix client for broadcaster resolution and archive VOD listing."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        base_url: str = HELIX_BASE_URL,
        timeout_sec: float = 15.0,
        max_retries: int = 2,
        transport: Transport | None = None,
        sleep_fn: SleepFn = time.sleep,
        time_fn: TimeFn = time.time,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.require_twitch_api_credentials()
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.transport = transport or _default_transport
        self.sleep_fn = sleep_fn
        self.time_fn = time_fn

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.twitch_access_token}",
            "Client-Id": self.settings.twitch_client_id or "",
        }

    def get_broadcaster_id(self, login: str) -> str:
        """Resolve a Twitch login name to a broadcaster ID."""

        payload = self._request_json("/users", {"login": login.lower()})
        users = payload.get("data", [])
        if not users:
            raise TwitchClientError(f"No Twitch broadcaster found for login {login!r}")
        return str(users[0]["id"])

    def list_archive_vods(self, login: str, limit: int | None = None) -> list[VodMeta]:
        """List archive VOD metadata for the supplied broadcaster login."""

        broadcaster_id = self.get_broadcaster_id(login)
        normalized_login = login.lower()
        remaining = limit
        cursor: str | None = None
        vods: list[VodMeta] = []

        while remaining is None or remaining > 0:
            page_size = MAX_PAGE_SIZE if remaining is None else min(remaining, MAX_PAGE_SIZE)
            params: dict[str, str | int] = {
                "user_id": broadcaster_id,
                "type": "archive",
                "sort": "time",
                "first": page_size,
            }
            if cursor is not None:
                params["after"] = cursor

            payload = self._request_json("/videos", params)
            page_data = payload.get("data", [])
            for item in page_data:
                vods.append(
                    VodMeta(
                        vod_id=str(item["id"]),
                        streamer_login=normalized_login,
                        created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
                        title=item.get("title", ""),
                        url=item["url"],
                        duration_seconds=parse_twitch_duration(item["duration"]),
                    )
                )
                if remaining is not None:
                    remaining -= 1
                    if remaining == 0:
                        return vods

            cursor = payload.get("pagination", {}).get("cursor")
            if not cursor or not page_data:
                break

        return vods

    def _request_json(self, path: str, params: Mapping[str, str | int]) -> dict[str, Any]:
        query_string = urlencode({key: value for key, value in params.items() if value is not None})
        url = f"{self.base_url}{path}?{query_string}"
        attempt = 0

        while True:
            response = self.transport(url, self._headers, self.timeout_sec)
            headers = {key.lower(): value for key, value in response.headers.items()}

            if response.status_code == 429 and attempt < self.max_retries:
                self.sleep_fn(self._compute_retry_delay_seconds(headers))
                attempt += 1
                continue

            if response.status_code < 200 or response.status_code >= 300:
                raise TwitchClientError(
                    f"Twitch Helix request failed with status {response.status_code}: {response.body.decode('utf-8', errors='replace')}"
                )

            try:
                return json.loads(response.body.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise TwitchClientError("Twitch Helix returned invalid JSON") from exc

    def _compute_retry_delay_seconds(self, headers: Mapping[str, str]) -> float:
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass

        rate_limit_reset = headers.get("ratelimit-reset")
        if rate_limit_reset is not None:
            try:
                return max(0.0, float(rate_limit_reset) - self.time_fn())
            except ValueError:
                pass

        return 1.0
