"""Best-effort publication of the latest benchmark camera frame."""

from __future__ import annotations

import io
import logging
import select
import socket
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from manifold.core.sensor import CameraOrientation, ChannelOrder
from manifold.wire import FrameChannel

if TYPE_CHECKING:
    from manifold.core.sensor import Camera
    from manifold.core.values import Observation

# Avoid upscaling frames whose smaller edge already fits the live view.
_MIN_EDGE_PIXELS = 256
# Balance live-view detail against transfer size.
_JPEG_QUALITY = 80
# Keep encoded frames within the receiver's JPEG limit.
_MAX_JPEG_BYTES = 2 * 1024 * 1024
# Throttle each connection independently.
_PUBLISH_INTERVAL_SECONDS = 1.0
# Bound shutdown latency while waiting for socket input.
_SOCKET_POLL_SECONDS = 0.1
# Avoid a tight connection-failure loop.
_RECONNECT_SECONDS = 1.0
# Reconnect when the receiver stops acknowledging frames.
_ACK_TIMEOUT_SECONDS = 5.0

_logger = logging.getLogger(__name__)


class LiveViewPublisher:
    """Keep one latest camera frame and publish it from a background thread."""

    def __init__(self, host: str, port: int, camera: Camera) -> None:
        self._host = host
        self._port = port
        self._camera = camera
        self._condition = threading.Condition()
        self._latest: _PendingFrame | None = None
        # Label each frame because delivery can be out of order; consumers discard any
        # frame with a lower sequence than the latest frame processed.
        self._sequence = 0
        self._consumed_sequence = 0
        self._closed = False
        self._socket: socket.socket | None = None
        self._thread = threading.Thread(target=self._run, name="manifold-live-view", daemon=True)

    def __enter__(self) -> LiveViewPublisher:
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            active_socket = self._socket
        if active_socket is not None:
            with suppress(OSError):
                active_socket.shutdown(socket.SHUT_RDWR)
        self._thread.join()

    def publish(
        self,
        observation: Observation,
        *,
        episode_idx: int,
        step: int,
        task: str,
    ) -> None:
        """Copy the selected camera frame into the latest-value slot without encoding or I/O."""
        image = observation.sensors.get(self._camera.name)
        if image is None:
            return
        self._camera.validate_value(image)
        copied = np.array(image, copy=True)
        with self._condition:
            self._sequence += 1
            self._latest = _PendingFrame(
                sequence=self._sequence,
                episode_idx=episode_idx,
                step=step,
                task=task,
                captured_at=time.time(),
                image=copied,
            )
            self._condition.notify_all()

    def _run(self) -> None:
        while not self._closed:
            try:
                with socket.create_connection((self._host, self._port), timeout=1.0) as sock:
                    with self._condition:
                        self._socket = sock
                    try:
                        self._publish_connected(sock)
                    finally:
                        with self._condition:
                            if self._socket is sock:
                                self._socket = None
            except OSError:
                if self._wait(_RECONNECT_SECONDS):
                    return
            except Exception:
                _logger.exception("live view publication failed")
                if self._wait(_RECONNECT_SECONDS):
                    return

    def _publish_connected(self, sock: socket.socket) -> None:
        sock.settimeout(_ACK_TIMEOUT_SECONDS)
        channel = FrameChannel.from_socket(sock, max_frame_bytes=1024)
        watched = False
        in_flight: int | None = None
        ack_deadline: float | None = None
        next_publish_at = 0.0

        while not self._closed:
            pending = self._pending_after(self._consumed_sequence)
            now = time.monotonic()
            if watched and in_flight is None and pending is not None and now >= next_publish_at:
                encoded = _encode_frame(pending, self._camera)
                if encoded is None:
                    self._consumed_sequence = pending.sequence
                    continue
                channel.send("frame", encoded)
                self._consumed_sequence = pending.sequence
                in_flight = pending.sequence
                ack_deadline = time.monotonic() + _ACK_TIMEOUT_SECONDS
                next_publish_at = now + _PUBLISH_INTERVAL_SECONDS

            if ack_deadline is not None and time.monotonic() >= ack_deadline:
                raise TimeoutError("live view acknowledgement timed out")

            readable, _, _ = select.select((sock,), (), (), _SOCKET_POLL_SECONDS)
            if len(readable) == 0:
                continue
            message = channel.recv()
            if message is None:
                return
            next_watched, acknowledged = _read_control(message, in_flight=in_flight)
            if next_watched is not None:
                watched = next_watched
            if acknowledged is not None:
                in_flight = None
                ack_deadline = None

    def _pending_after(self, sequence: int) -> _PendingFrame | None:
        with self._condition:
            if self._latest is None or self._latest.sequence <= sequence:
                return None
            return self._latest

    def _wait(self, seconds: float) -> bool:
        with self._condition:
            self._condition.wait_for(lambda: self._closed, timeout=seconds)
            return self._closed


@dataclass(frozen=True, eq=False)
class _PendingFrame:
    """One copied camera frame awaiting optional JPEG encoding."""

    sequence: int
    episode_idx: int
    step: int
    task: str
    captured_at: float
    image: np.ndarray


def _read_control(
    message: dict[str, Any], *, in_flight: int | None
) -> tuple[bool | None, int | None]:
    frame_type = message.get("type")
    payload = message.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("live view control payload must be a dict")  # noqa: TRY004
    if frame_type == "watch":
        watched = payload.get("watched")
        if not isinstance(watched, bool):
            raise ValueError("live view watch value must be a bool")
        return watched, None
    if frame_type == "ack":
        acknowledged = payload.get("sequence")
        if (
            not isinstance(acknowledged, int)
            or isinstance(acknowledged, bool)
            or acknowledged != in_flight
        ):
            raise ValueError("live view acknowledgement does not match the frame")
        return None, acknowledged
    raise ValueError(f"unsupported live view control type {frame_type!r}")


def _encode_frame(frame: _PendingFrame, camera: Camera) -> dict[str, Any] | None:
    try:
        image = _display_image(frame.image, camera)
        jpeg, width, height = _jpeg(image)
    except Exception:
        _logger.exception("live view frame encoding failed")
        return None
    if len(jpeg) > _MAX_JPEG_BYTES:
        _logger.warning("live view JPEG is too large: %d bytes", len(jpeg))
        return None
    return {
        "sequence": frame.sequence,
        "episode_idx": frame.episode_idx,
        "step": frame.step,
        "task": frame.task,
        "captured_at": frame.captured_at,
        "width": width,
        "height": height,
        "jpeg": jpeg,
    }


def _display_image(image: np.ndarray, camera: Camera) -> np.ndarray:
    camera.validate_value(image)
    displayed = image
    if camera.orientation in (
        CameraOrientation.FLIPPED_VERTICAL,
        CameraOrientation.ROTATED_180,
    ):
        displayed = displayed[::-1, :, :]
    if camera.orientation in (
        CameraOrientation.FLIPPED_HORIZONTAL,
        CameraOrientation.ROTATED_180,
    ):
        displayed = displayed[:, ::-1, :]
    if camera.channel_order is ChannelOrder.BGR:
        displayed = displayed[:, :, ::-1]
    return np.ascontiguousarray(displayed)


def _jpeg(image: np.ndarray) -> tuple[bytes, int, int]:
    rendered = Image.fromarray(image).convert("RGB")
    width, height = rendered.size
    if min(width, height) > _MIN_EDGE_PIXELS:
        scale = _MIN_EDGE_PIXELS / min(width, height)
        rendered = rendered.resize(
            (round(width * scale), round(height * scale)),
            resample=Image.Resampling.BILINEAR,
        )
    buffer = io.BytesIO()
    rendered.save(buffer, format="JPEG", quality=_JPEG_QUALITY)
    width, height = rendered.size
    return buffer.getvalue(), width, height


__all__ = [
    "LiveViewPublisher",
]
