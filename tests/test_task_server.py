"""Behaviour of the task server protocol messages on the wire."""

from __future__ import annotations

from typing import Any

import pytest

from manifold.wire import (
    FrameChannel,
    TaskServerAccepted,
    TaskServerClaim,
    TaskServerComplete,
    TaskServerDone,
    TaskServerError,
    TaskServerHello,
    TaskServerReady,
    TaskServerWait,
    TaskServerWork,
    WorkerArtifact,
    WorkerEpisode,
)
from manifold.wire.bridge import pack_stream_frame

_RESULT = {"episode_idx": 3, "task_id": "3", "success": True}

_MESSAGES = [
    (
        TaskServerHello(
            worker_id="worker-1",
            benchmark_name="libero",
            episodes=(WorkerEpisode(episode_idx=3, task_id="3", task_config={"seed": 2}),),
        ),
        ["worker_id", "benchmark_name", "episodes"],
    ),
    (TaskServerReady(), []),
    (TaskServerClaim(request_id="request-1"), ["request_id"]),
    (
        TaskServerWork(
            request_id="request-1",
            item_id="item-3",
            attempt_id="attempt-3",
            task_id="3",
            task_config={"suite": "libero_10", "seed": 2},
            episode_indices=(3,),
            output_dir="/tmp/results/attempts/attempt-3",
        ),
        [
            "request_id",
            "item_id",
            "attempt_id",
            "task_id",
            "task_config",
            "episode_indices",
            "output_dir",
        ],
    ),
    (TaskServerWait(request_id="request-1", retry_after_sec=1), ["request_id", "retry_after_sec"]),
    (TaskServerDone(request_id="request-1"), ["request_id"]),
    (
        TaskServerComplete(
            item_id="item-3",
            attempt_id="attempt-3",
            results=(_RESULT,),
            artifacts=(WorkerArtifact(episode_idx=3, path="replay.mfr", kind="replay"),),
        ),
        ["item_id", "attempt_id", "results", "artifacts"],
    ),
    (TaskServerAccepted(item_id="item-3", attempt_id="attempt-3"), ["item_id", "attempt_id"]),
    (TaskServerError(message="worker identity does not match assignment"), ["message"]),
]


def _pipe() -> FrameChannel:
    buffer = bytearray()

    def read_exactly(n: int) -> bytes | None:
        if len(buffer) < n:
            return None
        chunk = bytes(buffer[:n])
        del buffer[:n]
        return chunk

    return FrameChannel(read_exactly, buffer.extend)


@pytest.mark.parametrize(("message", "keys"), _MESSAGES)
def test_message_round_trips_through_a_frame_channel(message: Any, keys: list[str]) -> None:
    channel = _pipe()
    channel.send(message.frame_type, message.to_payload())
    frame = channel.recv()
    assert frame is not None
    assert frame["type"] == message.frame_type.value
    assert list(frame["payload"]) == keys
    assert type(message).from_payload(frame["payload"]) == message


@pytest.mark.parametrize(
    ("message", "frame_type", "payload"),
    [
        (
            TaskServerHello(
                worker_id="worker-1",
                benchmark_name="libero",
                episodes=(WorkerEpisode(3, "3", {"task_id": 3}),),
            ),
            "hello",
            {
                "worker_id": "worker-1",
                "benchmark_name": "libero",
                "episodes": [{"episode_idx": 3, "task_id": "3", "task_config": {"task_id": 3}}],
            },
        ),
        (TaskServerClaim(request_id="request-1"), "claim", {"request_id": "request-1"}),
        (
            TaskServerComplete(
                item_id="item-3",
                attempt_id="attempt-3",
                results=(_RESULT,),
                artifacts=(WorkerArtifact(3, "replay.mfr", "replay"),),
            ),
            "complete",
            {
                "item_id": "item-3",
                "attempt_id": "attempt-3",
                "results": [_RESULT],
                "artifacts": [{"episode_idx": 3, "path": "replay.mfr", "kind": "replay"}],
            },
        ),
    ],
)
def test_worker_message_frames_match_the_literal_payloads(
    message: Any, frame_type: str, payload: dict[str, Any]
) -> None:
    frame = pack_stream_frame(message.frame_type, message.to_payload(), seq=4, timestamp=0.0)
    assert frame == pack_stream_frame(frame_type, payload, seq=4, timestamp=0.0)


def test_work_refuses_a_task_config_that_is_not_a_mapping() -> None:
    payload = TaskServerWork(
        request_id="request-1",
        item_id="item-3",
        attempt_id="attempt-3",
        task_id="3",
        task_config={},
        episode_indices=(3,),
        output_dir="/tmp/results/attempts/attempt-3",
    ).to_payload()
    payload["task_config"] = ["[not a mapping]"]
    with pytest.raises(ValueError, match="work payload task_config must be a mapping"):
        TaskServerWork.from_payload(payload)
