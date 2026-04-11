from __future__ import annotations

import io

import pytest
from PIL import Image

from app.vod_source import VodSource, VodSourceError


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 1), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeWritable(io.BytesIO):
    def __init__(self) -> None:
        super().__init__()
        self.written = b""

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.written = self.getvalue()
        super().close()


class FakeProcess:
    def __init__(self, command: list[str], stdout_bytes: bytes, *, returncode: int = 0) -> None:
        self.command = command
        self.stdin = FakeWritable()
        self.stdout = io.BytesIO(stdout_bytes)
        self.stderr = io.BytesIO()
        self.returncode = returncode
        self.killed = False

    def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
        del timeout
        return self.stdout.getvalue(), self.stderr.getvalue()

    def kill(self) -> None:
        self.killed = True

    def poll(self) -> int | None:
        return self.returncode


class FakeReadable(io.BytesIO):
    pass


class FakeUrlStream:
    def __init__(self, url: str) -> None:
        self.url = url

    def open(self) -> io.BytesIO:
        raise AssertionError("open() should not be used when a direct URL is available")


class FakePipeStream:
    def __init__(self, payload: bytes) -> None:
        self.handle = FakeReadable(payload)

    def open(self) -> FakeReadable:
        return self.handle


class FakeSession:
    def __init__(self, stream: object) -> None:
        self.stream = stream
        self.requested_urls: list[str] = []

    def streams(self, vod_url: str) -> dict[str, object]:
        self.requested_urls.append(vod_url)
        return {"best": self.stream}


def test_iter_window_frames_uses_stream_url_when_available() -> None:
    frames_blob = _jpeg_bytes((255, 0, 0)) + _jpeg_bytes((0, 255, 0))
    processes: list[FakeProcess] = []

    def popen_factory(command: list[str], **kwargs) -> FakeProcess:
        del kwargs
        process = FakeProcess(command, frames_blob)
        process.stdin = None  # type: ignore[assignment]
        processes.append(process)
        return process

    source = VodSource(
        ffmpeg_path="ffmpeg-test",
        stream_session_factory=lambda: FakeSession(FakeUrlStream("https://media.example/vod.m3u8")),
        popen_factory=popen_factory,
    )

    frames = list(
        source.iter_window_frames(
            "https://twitch.tv/videos/1",
            start_sec=10.0,
            duration_sec=1.0,
            fps=2.0,
        )
    )

    assert len(frames) == 2
    assert frames[0].frame_index_in_window == 0
    assert frames[1].timestamp_sec == pytest.approx(10.5)
    assert frames[0].frame_bgr.shape == (1, 2, 3)
    assert "https://media.example/vod.m3u8" in processes[0].command


def test_iter_window_frames_falls_back_to_pipe_input() -> None:
    frames_blob = _jpeg_bytes((10, 20, 30))
    stream = FakePipeStream(b"stream-bytes")
    process_holder: dict[str, FakeProcess] = {}

    def popen_factory(command: list[str], **kwargs) -> FakeProcess:
        del kwargs
        process = FakeProcess(command, frames_blob)
        process_holder["process"] = process
        return process

    source = VodSource(
        ffmpeg_path="ffmpeg-test",
        stream_session_factory=lambda: FakeSession(stream),
        popen_factory=popen_factory,
    )

    frames = list(
        source.iter_window_frames(
            "https://twitch.tv/videos/2",
            start_sec=5.0,
            duration_sec=1.0,
            fps=1.0,
        )
    )

    assert len(frames) == 1
    assert process_holder["process"].stdin.written == b"stream-bytes"
    assert process_holder["process"].stdin.closed
    assert stream.handle.closed
    assert "pipe:0" in process_holder["process"].command


def test_iter_window_frames_raises_on_ffmpeg_error() -> None:
    def popen_factory(command: list[str], **kwargs) -> FakeProcess:
        del kwargs
        process = FakeProcess(command, b"", returncode=1)
        process.stderr = io.BytesIO(b"bad ffmpeg run")
        return process

    source = VodSource(
        ffmpeg_path="ffmpeg-test",
        stream_session_factory=lambda: FakeSession(FakeUrlStream("https://media.example/vod.m3u8")),
        popen_factory=popen_factory,
    )

    with pytest.raises(VodSourceError, match="ffmpeg exited with status 1"):
        list(
            source.iter_window_frames(
                "https://twitch.tv/videos/3",
                start_sec=0.0,
                duration_sec=1.0,
                fps=1.0,
            )
        )
