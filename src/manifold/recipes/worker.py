"""Runner-assigned benchmark task execution.

`WorkerSession` registers the workload with the task server, claims tasks and
completes them. A task covers one or more episodes, and `WorkerSession.complete`
sends the results of all of them in one `complete` message.

`run_worker` executes one-episode tasks on a `WorkerSession`. It reuses one
policy connection to execute every task, reports the result, and waits for task
server acceptance before claiming the next task.
"""

from __future__ import annotations

import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from manifold.recipes.serving import BenchmarkResult, ResetResult, run_episodes
from manifold.wire import FrameChannel

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from manifold.core.benchmark import Benchmark
    from manifold.core.values import Action, Observation
    from manifold.recipes.recording import EpisodeRecorder
    from manifold.recipes.serving import EpisodeRecord, StepResult
    from manifold.wire import ImageFormat


@dataclass(frozen=True)
class WorkerEpisode:
    """One episode in the workload's ordered task and seed manifest."""

    episode_idx: int
    task_id: str
    task_config: dict[str, Any]


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
        benchmark_name: str,
        episodes: Sequence[WorkerEpisode],
    ) -> None:
        host, separator, port = task_server_address.rpartition(":")
        if not separator or not host:
            raise ValueError("task_server_address must be host:port")
        self._sock = socket.create_connection((host, int(port)))
        try:
            self._channel = FrameChannel.from_socket(self._sock)
            self._channel.send(
                "hello",
                {
                    "worker_id": worker_id,
                    "benchmark_name": benchmark_name,
                    "episodes": [asdict(episode) for episode in episodes],
                },
            )
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
        records: Sequence[EpisodeRecord],
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


def run_worker(
    benchmark: Benchmark,
    reset: Callable[[WorkerTask], Observation],
    step: Callable[[Action], StepResult],
    *,
    server: str,
    task_server_address: str,
    worker_id: str,
    episodes: Sequence[WorkerEpisode],
    max_steps: int,
    recorder: Callable[[WorkerTask], EpisodeRecorder],
    artifacts: Callable[[WorkerTask], Sequence[Path]],
    image_format: ImageFormat = "raw",
    live_view_server: str | None = None,
    live_view_camera: str | None = None,
) -> BenchmarkResult:
    """Execute one-episode task server assignments with one policy connection.

    Refuse a task that covers more than one episode. Send results after closing
    each replay, then wait for task server acceptance before claiming again.
    `live_view_server` and `live_view_camera` configure the optional latest-value
    channel as they do for `run_benchmark`.
    """
    with WorkerSession(
        task_server_address,
        worker_id=worker_id,
        benchmark_name=benchmark.name,
        episodes=episodes,
    ) as session:
        worker_run = _WorkerRun(session, reset, recorder, artifacts)
        run_episodes(
            benchmark,
            worker_run.reset_episode,
            step,
            server=server,
            episode_ids=worker_run.claim_episode_ids(),
            max_steps=max_steps,
            image_format=image_format,
            recorder=worker_run,
            on_episode=worker_run.report_episode,
            live_view_server=live_view_server,
            live_view_camera=live_view_camera,
        )
        return BenchmarkResult(records=tuple(worker_run.records))


class _WorkerRun:
    def __init__(
        self,
        session: WorkerSession,
        reset: Callable[[WorkerTask], Observation],
        recorder: Callable[[WorkerTask], EpisodeRecorder],
        artifacts: Callable[[WorkerTask], Sequence[Path]],
    ) -> None:
        self.session = session
        self.reset_task = reset
        self.recorder_factory = recorder
        self.artifacts_for_task = artifacts
        self.current_task: WorkerTask | None = None
        self.current_recorder: EpisodeRecorder | None = None
        self.records: list[EpisodeRecord] = []

    def claim_episode_ids(self) -> Iterator[int]:
        while (task := self.session.claim()) is not None:
            if len(task.episode_indices) != 1:
                raise ValueError(
                    f"run_worker requires one episode per task, got {len(task.episode_indices)}; "
                    "use WorkerSession to claim tasks that cover several episodes"
                )
            self.current_task = task
            self.current_recorder = self.recorder_factory(task)
            yield task.episode_indices[0]
            self.current_task = None
            self.current_recorder = None

    def reset_episode(self, _episode_idx: int) -> ResetResult:
        assert self.current_task is not None
        task = self.current_task
        return ResetResult(self.reset_task(task), task_id=task.task_id)

    def begin(self, episode_id: int) -> None:
        assert self.current_recorder is not None
        self.current_recorder.begin(episode_id)

    def record(self, action: Action, *, success: bool, done: bool) -> None:
        assert self.current_recorder is not None
        self.current_recorder.record(action, success=success, done=done)

    def end(self) -> None:
        assert self.current_recorder is not None
        self.current_recorder.end()

    def report_episode(self, record: EpisodeRecord) -> None:
        assert self.current_task is not None
        task = self.current_task
        # The episode loop closes the recorder before invoking this callback.
        self.session.complete(task, [record], {record.episode_idx: self.artifacts_for_task(task)})
        self.records.append(record)


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
    "WorkerEpisode",
    "WorkerSession",
    "WorkerTask",
    "run_worker",
]
