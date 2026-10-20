"""Runner-assigned benchmark task execution.

`WorkerSession` registers the workload with the task server, claims tasks and
completes them. A task covers one or more episodes, and `WorkerSession.complete`
sends the results of all of them in one `complete` message.
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
    """One finished episode, as the benchmark observed it.

    `started_at` and `ended_at` bracket the rollout, so `elapsed_sec` derives from
    them rather than being measured separately and free to disagree. `task_name` is
    the reset observation's instruction where the benchmark publishes one, and the
    benchmark's own name otherwise. `episode_idx` is the global episode id that the
    task server assigned.

    `task_id` is the benchmark's own identifier for the task, such as a class
    name. Two tasks in a suite may share an instruction, and `task_id` separates
    them. `attributes` labels the episode for result tables. Both come from the
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
    """One attempt assigned by the task server, its episodes and its output directory."""

    item_id: str
    attempt_id: str
    task_id: str
    task_config: dict[str, Any]
    episode_indices: tuple[int, ...]
    output_dir: Path


class WorkerSession:
    """A task server connection for claiming and completing tasks.

    Connect, send `hello` and wait for `ready` on construction; close the
    connection on exit. Use the session from any thread, but call `claim` and
    `complete` from one thread at a time.
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

        Sleep and claim again while the task server replies `wait`.
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
        """Send the results of every episode in `task` and wait for acceptance.

        Pass one record per index in `task.episode_indices`, and key `artifacts` by
        episode index with that episode's replay files under `task.output_dir`.
        Close each replay before calling this.
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
