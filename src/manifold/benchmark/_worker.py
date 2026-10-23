"""A connection to the task server, and the tasks and episode records that it sends.

`WorkerSession` registers the worker with the task server, claims tasks and completes
them. A task covers one or more episodes. `WorkerSession.complete` sends the results
of all its episodes in a single `complete` message.
"""

from __future__ import annotations

import socket
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from manifold.wire import FrameChannel

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class WorkerEpisodeRecord:
    """A finished episode, as the benchmark observed it.

    `elapsed_sec` is computed from `started_at` and `ended_at`, so the three values
    agree. `task_name` is the episode's instruction, or the benchmark's name for a
    benchmark without instructions. `episode_idx` is the index that the task server
    assigned to the episode, unique across the run.

    `task_id` is the benchmark's own identifier for the task, such as a class name.
    Two tasks in a suite can share an instruction, and `task_id` separates them.
    `attributes` labels the episode for result tables. Both values come from the
    benchmark's `ResetResult`.
    """

    episode_idx: int
    task_name: str
    success: bool
    steps: int
    initialization_sec: float
    started_at: datetime
    ended_at: datetime
    task_id: str
    attributes: dict[str, str]

    @property
    def elapsed_sec(self) -> float:
        """The episode's wall-clock duration, in seconds."""
        return (self.ended_at - self.started_at).total_seconds()


@dataclass(frozen=True)
class WorkerTask:
    """An attempt at a task that the task server assigned.

    It contains the task's config, its episode indices and its output directory.
    """

    item_id: str
    attempt_id: str
    task_id: str
    task_config: dict[str, Any]
    episode_indices: tuple[int, ...]
    output_dir: Path


class WorkerSession:
    """A connection to the task server, which claims and completes tasks.

    The constructor connects, sends `hello` and waits for `ready`. Leaving the context
    closes the connection. Any thread can use the session, but only a single thread at a
    time can call `claim` or `complete`.
    """

    def __init__(
        self,
        task_server_address: str,
        *,
        worker_id: str,
    ) -> None:
        host, separator, port = task_server_address.rpartition(":")
        if not separator or not host:
            raise ValueError("task_server_address must be host:port")
        self._sock = socket.create_connection((host, int(port)))
        try:
            self._channel = FrameChannel.from_socket(self._sock)
            self._channel.send("hello", {"worker_id": worker_id})
            _receive_task_server_frame(self._channel, "ready")
        except BaseException:
            self._sock.close()
            raise

    def __enter__(self) -> WorkerSession:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the task server connection."""
        self._sock.close()

    def claim(self) -> WorkerTask | None:
        """Claim the next task, or return `None` once the task server replies `done`.

        While the task server replies `wait`, sleep for the delay that it sends and claim
        again.
        """
        while True:
            request_id = str(uuid4())
            self._channel.send("claim", {"request_id": request_id})
            reply = _receive_task_server_frame(self._channel)
            payload = reply["payload"]
            if payload["request_id"] != request_id:
                raise ValueError("task server claim request ID mismatch")
            if reply["type"] == "done":
                return None
            if reply["type"] == "wait":
                delay = payload["retry_after_sec"]
                if not isinstance(delay, (int, float)) or not 0 <= delay <= 60:
                    raise ValueError("task server retry_after_sec must be between 0 and 60")
                time.sleep(delay)
                continue
            if reply["type"] != "work":
                raise ValueError(f"unexpected task server response {reply['type']}")
            return _parse_task_assignment(payload)

    def complete(
        self,
        task: WorkerTask,
        records: Sequence[WorkerEpisodeRecord],
        artifacts: Mapping[int, Sequence[Path]],
    ) -> None:
        """Send the results of the episodes in `task`, and wait for the server to accept them.

        `records` contains a record for each index in `task.episode_indices`. `artifacts`
        maps an episode index to the replay files of that episode under
        `task.output_dir`. Close each replay before the call.
        """
        results = []
        for record in records:
            result = asdict(record)
            result["started_at"] = record.started_at.isoformat()
            result["ended_at"] = record.ended_at.isoformat()
            results.append(result)
        artifact_entries = []
        for episode_idx, paths in artifacts.items():
            for path in paths:
                resolved = path.resolve(strict=True)
                relative = resolved.relative_to(task.output_dir.resolve())
                if not resolved.is_file():
                    raise ValueError("worker artifact must be a file")
                artifact_entries.append(
                    {"episode_idx": episode_idx, "path": str(relative), "kind": "replay"}
                )
        self._channel.send(
            "complete",
            {
                "item_id": task.item_id,
                "attempt_id": task.attempt_id,
                "results": results,
                "artifacts": artifact_entries,
            },
        )
        accepted = _receive_task_server_frame(self._channel, "accepted")["payload"]
        if accepted["item_id"] != task.item_id or accepted["attempt_id"] != task.attempt_id:
            raise ValueError("task server accepted a different item or attempt")


def _receive_task_server_frame(
    channel: FrameChannel, expected: str | None = None
) -> dict[str, Any]:
    frame = channel.recv()
    if frame is None:
        raise RuntimeError("task server disconnected")
    if frame["type"] == "error":
        raise RuntimeError(frame["payload"]["message"])
    if expected is not None and frame["type"] != expected:
        raise ValueError(f"expected task server {expected}, got {frame['type']}")
    return frame


def _parse_task_assignment(payload: dict[str, Any]) -> WorkerTask:
    indices = payload["episode_indices"]
    output_dir = Path(payload["output_dir"])
    if not output_dir.is_absolute():
        raise ValueError("worker output_dir must be absolute")
    for key in ("item_id", "attempt_id", "task_id"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise ValueError(f"worker {key} must be a non-empty string")
    if not isinstance(payload["task_config"], dict):
        raise TypeError("worker task_config must be a mapping")
    return WorkerTask(
        item_id=payload["item_id"],
        attempt_id=payload["attempt_id"],
        task_id=payload["task_id"],
        task_config=payload["task_config"],
        episode_indices=tuple(indices),
        output_dir=output_dir,
    )


__all__ = [
    "WorkerEpisodeRecord",
    "WorkerSession",
    "WorkerTask",
]
