"""Bounded frame access for Twitch VODs using Streamlink and ffmpeg."""

from __future__ import annotations

import io
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterator, Mapping, Protocol

import numpy as np
from PIL import Image

from app.config import get_settings

StreamSessionFactory = Callable[[], object]
PopenFactory = Callable[..., subprocess.Popen]


class VodSourceError(RuntimeError):
    """Raised when a VOD frame window cannot be decoded."""


@dataclass(frozen=True)
class FrameSample:
    """A decoded frame from a requested VOD time window."""

    timestamp_sec: float
    frame_bgr: np.ndarray
    frame_index_in_window: int


class SupportsOpen(Protocol):
    """Protocol for Streamlink stream objects used by VodSource."""

    def open(self) -> BinaryIO:
        ...


def _default_stream_session_factory() -> object:
    from streamlink import Streamlink

    return Streamlink()


class VodSource:
    """Decode bounded frame windows from Twitch VOD streams."""

    def __init__(
        self,
        *,
        ffmpeg_path: str | None = None,
        stream_quality: str = "best",
        stream_session_factory: StreamSessionFactory | None = None,
        popen_factory: PopenFactory = subprocess.Popen,
        chunk_size: int = 64 * 1024,
        process_timeout_sec: float = 60.0,
    ) -> None:
        settings = get_settings()
        self.ffmpeg_path = ffmpeg_path or settings.ffmpeg_path
        self.stream_quality = stream_quality
        self.stream_session_factory = stream_session_factory or _default_stream_session_factory
        self.popen_factory = popen_factory
        self.chunk_size = chunk_size
        self.process_timeout_sec = process_timeout_sec

    def iter_window_frames(
        self,
        vod_url: str,
        start_sec: float,
        duration_sec: float,
        fps: float,
    ) -> Iterator[FrameSample]:
        """Yield decoded BGR frames for a bounded VOD window."""

        if start_sec < 0:
            raise ValueError("start_sec must be >= 0")
        if duration_sec <= 0:
            raise ValueError("duration_sec must be > 0")
        if fps <= 0:
            raise ValueError("fps must be > 0")

        local_path = Path(vod_url)
        if local_path.exists():
            yield from self._iter_frames_from_input_source(
                input_source=str(local_path),
                start_sec=start_sec,
                duration_sec=duration_sec,
                fps=fps,
                use_seekable_input=True,
            )
            return

        session = self.stream_session_factory()
        available_streams = self._get_streams(session, vod_url)
        stream = available_streams.get(self.stream_quality)
        if stream is None:
            raise VodSourceError(
                f"Requested stream quality {self.stream_quality!r} is not available for {vod_url!r}"
            )

        input_source = getattr(stream, "url", None)
        stream_handle: BinaryIO | None = None
        feed_thread: threading.Thread | None = None

        command = self._build_ffmpeg_command(
            input_source="pipe:0" if input_source is None else str(input_source),
            start_sec=start_sec,
            duration_sec=duration_sec,
            fps=fps,
            use_seekable_input=input_source is not None,
        )

        yield from self._iter_frames_from_input_source(
            input_source=str(input_source) if input_source is not None else None,
            start_sec=start_sec,
            duration_sec=duration_sec,
            fps=fps,
            use_seekable_input=input_source is not None,
            stream=stream if input_source is None else None,
        )

    def _iter_frames_from_input_source(
        self,
        *,
        input_source: str | None,
        start_sec: float,
        duration_sec: float,
        fps: float,
        use_seekable_input: bool,
        stream: SupportsOpen | None = None,
    ) -> Iterator[FrameSample]:
        stream_handle: BinaryIO | None = None
        feed_thread: threading.Thread | None = None

        command = self._build_ffmpeg_command(
            input_source="pipe:0" if input_source is None else input_source,
            start_sec=start_sec,
            duration_sec=duration_sec,
            fps=fps,
            use_seekable_input=use_seekable_input,
        )

        process = self.popen_factory(
            command,
            stdin=subprocess.PIPE if input_source is None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            if input_source is None:
                if stream is None:
                    raise VodSourceError("A stream handle is required for non-seekable input")
                stream_handle = self._open_stream(stream)
                stdin_handle = process.stdin
                if stdin_handle is None:
                    raise VodSourceError("ffmpeg stdin pipe is unavailable")
                feed_thread = threading.Thread(
                    target=self._feed_stream,
                    args=(stream_handle, stdin_handle),
                    daemon=True,
                )
                feed_thread.start()

            stdout_data, stderr_data = process.communicate(timeout=self.process_timeout_sec)

            if feed_thread is not None:
                feed_thread.join(timeout=self.process_timeout_sec)

            if process.returncode not in (0, None):
                raise VodSourceError(
                    f"ffmpeg exited with status {process.returncode}: {stderr_data.decode('utf-8', errors='replace')}"
                )

            for index, frame_bytes in enumerate(self._extract_jpeg_frames(stdout_data)):
                frame_rgb = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
                frame_bgr = np.asarray(frame_rgb)[:, :, ::-1].copy()
                yield FrameSample(
                    timestamp_sec=start_sec + (index / fps),
                    frame_bgr=frame_bgr,
                    frame_index_in_window=index,
                )
        except subprocess.TimeoutExpired as exc:
            process.kill()
            raise VodSourceError("ffmpeg window decode timed out") from exc
        finally:
            if stream_handle is not None and not stream_handle.closed:
                stream_handle.close()
            if process.stdout is not None and not process.stdout.closed:
                process.stdout.close()
            if process.stderr is not None and not process.stderr.closed:
                process.stderr.close()
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.kill()

    def _build_ffmpeg_command(
        self,
        *,
        input_source: str,
        start_sec: float,
        duration_sec: float,
        fps: float,
        use_seekable_input: bool,
    ) -> list[str]:
        base_command = [self.ffmpeg_path, "-hide_banner", "-loglevel", "error"]
        if use_seekable_input:
            base_command.extend(["-ss", f"{start_sec:.3f}"])
        base_command.extend(["-i", input_source])
        if not use_seekable_input:
            base_command.extend(["-ss", f"{start_sec:.3f}"])
        base_command.extend(
            [
                "-t",
                f"{duration_sec:.3f}",
                "-vf",
                f"fps={fps}",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "pipe:1",
            ]
        )
        return base_command

    def _get_streams(self, session: object, vod_url: str) -> Mapping[str, SupportsOpen]:
        streams = getattr(session, "streams")(vod_url)
        if not isinstance(streams, Mapping):
            raise VodSourceError("Streamlink did not return a stream mapping")
        return streams

    def _open_stream(self, stream: SupportsOpen) -> BinaryIO:
        stream_handle = stream.open()
        if not hasattr(stream_handle, "read"):
            raise VodSourceError("Streamlink stream handle is not readable")
        return stream_handle

    def _feed_stream(self, stream_handle: BinaryIO, ffmpeg_stdin: BinaryIO) -> None:
        try:
            while True:
                chunk = stream_handle.read(self.chunk_size)
                if not chunk:
                    break
                ffmpeg_stdin.write(chunk)
                ffmpeg_stdin.flush()
        finally:
            if not ffmpeg_stdin.closed:
                ffmpeg_stdin.close()

    @staticmethod
    def _extract_jpeg_frames(buffer: bytes) -> Iterator[bytes]:
        start_marker = b"\xff\xd8"
        end_marker = b"\xff\xd9"
        scan_offset = 0

        while True:
            frame_start = buffer.find(start_marker, scan_offset)
            if frame_start == -1:
                return
            frame_end = buffer.find(end_marker, frame_start + 2)
            if frame_end == -1:
                return

            yield buffer[frame_start : frame_end + 2]
            scan_offset = frame_end + 2
