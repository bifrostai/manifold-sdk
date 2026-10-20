"""Tests for the live view publisher in manifold.benchmark._live_view."""

from __future__ import annotations

import io
import socket
import threading
from contextlib import suppress
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from manifold.benchmark._live_view import live_view_publisher
from manifold.core.benchmark import Benchmark
from manifold.core.sensor import Camera, Modality
from manifold.core.values import Observation
from manifold.embodiments import FRANKA_EE_DELTA
from manifold.sensors import CameraName
from manifold.wire import FrameChannel

_SERVER = "127.0.0.1:9001"


def _benchmark(camera: Camera) -> Benchmark:
    return Benchmark(name="bench-1", embodiment=FRANKA_EE_DELTA, sensors=[camera])


def _frame(image: np.ndarray) -> Observation:
    return Observation(sensors={CameraName.AGENTVIEW: image})


def test_the_publisher_sends_the_frame_as_a_jpeg_with_its_episode(monkeypatch):
    # arrange
    live_client, live_server = socket.socketpair()
    received: list[dict] = []
    received_frame = threading.Event()

    def receive_live_view() -> None:
        with live_server:
            channel = FrameChannel.from_socket(live_server)
            channel.send("watch", {"watched": True})
            frame = channel.recv()
            assert frame is not None
            received.append(frame["payload"])
            received_frame.set()
            channel.send("ack", {"sequence": frame["payload"]["sequence"]})

    receiver = threading.Thread(target=receive_live_view)
    receiver.start()
    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: live_client)
    source = np.zeros((360, 640, 3), dtype=np.uint8)
    source[:180, :320, 0] = 255
    source[:180, 320:, 1] = 255
    source[180:, :320, 2] = 255
    benchmark = _benchmark(Camera(name=CameraName.AGENTVIEW, shape=source.shape))

    # act
    with live_view_publisher(benchmark, _SERVER, CameraName.AGENTVIEW) as publisher:
        assert publisher is not None
        publisher.publish(_frame(source), episode_idx=7, step=0, task="pick", task_id="PickTask")
        assert received_frame.wait(timeout=2.0)
    receiver.join(timeout=2.0)

    # assert
    assert not receiver.is_alive()
    assert len(received) == 1
    payload = received[0]
    assert payload["episode_idx"] == 7
    assert payload["step"] == 0
    assert payload["task"] == "pick"
    assert payload["task_id"] == "PickTask"
    assert (payload["width"], payload["height"]) == (455, 256)
    displayed = np.asarray(Image.open(io.BytesIO(payload["jpeg"])))
    assert displayed.shape == (256, 455, 3)
    assert int(np.argmax(displayed[32, 32])) == 0
    assert int(np.argmax(displayed[32, -32])) == 1
    assert int(np.argmax(displayed[-32, 32])) == 2


def test_the_publisher_does_not_repeat_a_frame_after_reconnecting(monkeypatch):
    # arrange
    first_client, first_server = socket.socketpair()
    second_client, second_server = socket.socketpair()
    clients = iter((first_client, second_client))
    frames: list[dict] = []
    publish_next = threading.Event()
    received_second = threading.Event()
    second_connected = threading.Event()

    def receive_live_view() -> None:
        with first_server:
            first_channel = FrameChannel.from_socket(first_server)
            first_channel.send("watch", {"watched": True})
            first = first_channel.recv()
            assert first is not None
            frames.append(first["payload"])

        with second_server:
            second_channel = FrameChannel.from_socket(second_server)
            second_channel.send("watch", {"watched": True})
            assert second_connected.wait(timeout=2.0)
            second_server.settimeout(0.2)
            with pytest.raises(TimeoutError):
                second_channel.recv()
            publish_next.set()
            second_server.settimeout(None)
            second = second_channel.recv()
            assert second is not None
            frames.append(second["payload"])
            second_channel.send("ack", {"sequence": second["payload"]["sequence"]})
            received_second.set()

    receiver = threading.Thread(target=receive_live_view)
    receiver.start()

    def connect(*_args, **_kwargs):
        client = next(clients)
        if client is second_client:
            second_connected.set()
        return client

    monkeypatch.setattr(socket, "create_connection", connect)
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    benchmark = _benchmark(Camera(name=CameraName.AGENTVIEW, shape=image.shape))

    # act
    with live_view_publisher(benchmark, _SERVER, CameraName.AGENTVIEW) as publisher:
        assert publisher is not None
        publisher.publish(_frame(image), episode_idx=0, step=0, task="task", task_id="task")
        assert publish_next.wait(timeout=2.0)
        publisher.publish(_frame(image), episode_idx=0, step=1, task="task", task_id="task")
        assert received_second.wait(timeout=2.0)
    receiver.join(timeout=2.0)

    # assert
    assert not receiver.is_alive()
    assert [frame["sequence"] for frame in frames] == [1, 2]


def test_the_publisher_sends_a_frame_again_when_sending_fails(monkeypatch):
    from manifold.benchmark import _live_view

    first_client, first_server = socket.socketpair()
    second_client, second_server = socket.socketpair()
    clients = iter((first_client, second_client))
    failed_send = threading.Event()
    received_frame = threading.Event()
    received: list[dict] = []

    class FailFirstFrame:
        def __init__(self, channel: FrameChannel) -> None:
            self._channel = channel

        def send(self, frame_type: str, payload: dict) -> None:
            if frame_type == "frame" and not failed_send.is_set():
                failed_send.set()
                raise OSError("send failed")
            self._channel.send(frame_type, payload)

        def recv(self):
            return self._channel.recv()

    def publisher_channel(sock, *, max_frame_bytes):
        channel = FrameChannel.from_socket(sock, max_frame_bytes=max_frame_bytes)
        return FailFirstFrame(channel)

    def receive_live_view() -> None:
        with first_server:
            channel = FrameChannel.from_socket(first_server)
            channel.send("watch", {"watched": True})
            assert failed_send.wait(timeout=2.0)
            assert first_server.recv(1) == b""

        with second_server:
            channel = FrameChannel.from_socket(second_server)
            channel.send("watch", {"watched": True})
            frame = channel.recv()
            assert frame is not None
            received.append(frame["payload"])
            channel.send("ack", {"sequence": frame["payload"]["sequence"]})
            received_frame.set()

    receiver = threading.Thread(target=receive_live_view)
    receiver.start()
    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: next(clients))
    monkeypatch.setattr(_live_view, "FrameChannel", SimpleNamespace(from_socket=publisher_channel))
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    benchmark = _benchmark(Camera(name=CameraName.AGENTVIEW, shape=image.shape))

    with live_view_publisher(benchmark, _SERVER, CameraName.AGENTVIEW) as publisher:
        assert publisher is not None
        publisher.publish(_frame(image), episode_idx=0, step=0, task="task", task_id="task")
        assert received_frame.wait(timeout=2.0)
    receiver.join(timeout=2.0)

    assert not receiver.is_alive()
    assert [frame["sequence"] for frame in received] == [1]


def test_the_publisher_rejects_a_boolean_acknowledgement(monkeypatch):
    live_client, live_server = socket.socketpair()
    acknowledgement_rejected = threading.Event()

    def receive_live_view() -> None:
        with live_server:
            channel = FrameChannel.from_socket(live_server)
            channel.send("watch", {"watched": True})
            frame = channel.recv()
            assert frame is not None
            assert frame["payload"]["sequence"] == 1
            channel.send("ack", {"sequence": True})
            assert live_server.recv(1) == b""
            acknowledgement_rejected.set()

    receiver = threading.Thread(target=receive_live_view)
    receiver.start()
    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: live_client)
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    benchmark = _benchmark(Camera(name=CameraName.AGENTVIEW, shape=image.shape))

    with live_view_publisher(benchmark, _SERVER, CameraName.AGENTVIEW) as publisher:
        assert publisher is not None
        publisher.publish(_frame(image), episode_idx=0, step=0, task="task", task_id="task")
        assert acknowledgement_rejected.wait(timeout=2.0)
    receiver.join(timeout=2.0)

    assert not receiver.is_alive()


def test_a_live_view_server_requires_a_camera():
    benchmark = _benchmark(Camera(name=CameraName.AGENTVIEW, shape=(8, 8, 3)))

    with pytest.raises(ValueError, match="camera is required"):
        live_view_publisher(benchmark, _SERVER, None)


def test_the_live_view_rejects_a_depth_camera():
    benchmark = _benchmark(
        Camera(
            name=CameraName.AGENTVIEW_DEPTH,
            shape=(256, 256, 1),
            dtype="float32",
            modality=Modality.DEPTH,
        )
    )

    with pytest.raises(ValueError, match="must be RGB"):
        live_view_publisher(benchmark, _SERVER, CameraName.AGENTVIEW_DEPTH)


def test_the_publisher_checks_a_frame_before_copying_it(monkeypatch):
    live_client, live_server = socket.socketpair()
    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: live_client)
    benchmark = _benchmark(Camera(name=CameraName.AGENTVIEW, shape=(256, 256, 3)))
    too_large = np.zeros((1000, 1000, 3), dtype=np.uint8)

    with (
        live_server,
        live_view_publisher(benchmark, _SERVER, CameraName.AGENTVIEW) as publisher,
        pytest.raises(ValueError, match="shape mismatch"),
    ):
        assert publisher is not None
        publisher.publish(_frame(too_large), episode_idx=0, step=0, task="task", task_id="task")


def test_the_publisher_closes_its_socket_during_a_partial_control_frame(monkeypatch):
    live_client, live_server = socket.socketpair()
    partial_sent = threading.Event()

    def receive_live_view() -> None:
        with live_server:
            live_server.sendall(b"\x00")
            partial_sent.set()
            with suppress(ConnectionResetError):
                assert live_server.recv(1) == b""

    receiver = threading.Thread(target=receive_live_view)
    receiver.start()
    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: live_client)
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    benchmark = _benchmark(Camera(name=CameraName.AGENTVIEW, shape=image.shape))

    with live_view_publisher(benchmark, _SERVER, CameraName.AGENTVIEW) as publisher:
        assert publisher is not None
        publisher.publish(_frame(image), episode_idx=0, step=0, task="task", task_id="task")
        assert partial_sent.wait(timeout=2.0)
    receiver.join(timeout=1.0)

    assert not receiver.is_alive()
