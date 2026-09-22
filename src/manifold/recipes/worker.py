"""Runner-assigned benchmark task execution.

`run_worker` registers the workload with the runner and claims one task at a
time. It reuses one policy connection to execute every task, reports the
result, and waits for scheduler acceptance before claiming the next task.
"""

from __future__ import annotations

import socket
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from manifold.recipes.serving import BenchmarkResult, run_episodes
from manifold.wire import FrameChannel

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

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
    """One scheduler-assigned attempt and its output directory."""

    item_id: str
    attempt_id: str
    task_id: str
    task_config: dict[str, Any]
    episode_idx: int
    output_dir: Path


def run_worker(
    benchmark: Benchmark,
    reset: Callable[[WorkerTask], Observation],
    step: Callable[[Action], StepResult],
    *,
    server: str,
    scheduler_address: str,
    worker_id: str,
    episodes: Sequence[WorkerEpisode],
    max_steps: int,
    recorder: Callable[[WorkerTask], EpisodeRecorder],
    artifacts: Callable[[WorkerTask], Sequence[Path]],
    image_format: ImageFormat = "raw",
) -> BenchmarkResult:
    """Execute scheduler assignments with one policy connection.

    Send results after closing each replay, then wait for the scheduler to accept
    or discard the attempt before claiming again. A new attempt may repeat a global
    episode index. Attempt IDs must be distinct, and output directories must not
    overlap. Omit a result when the scheduler discards its attempt after lease loss,
    then claim again.
    """
    host, separator, port = scheduler_address.rpartition(":")
    if not separator or not host:
        raise ValueError("scheduler_address must be host:port")
    with socket.create_connection((host, int(port))) as sock:
        channel = FrameChannel.from_socket(sock)
        channel.send(
            "hello",
            {
                "worker_id": worker_id,
                "benchmark_name": benchmark.name,
                "episodes": [asdict(episode) for episode in episodes],
            },
        )
        _receive_scheduler_frame(channel, "ready")
        worker_run = _WorkerRun(channel, reset, recorder, artifacts)
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
        )
        return BenchmarkResult(records=tuple(worker_run.records))


class _WorkerRun:
    def __init__(
        self,
        channel: FrameChannel,
        reset: Callable[[WorkerTask], Observation],
        recorder: Callable[[WorkerTask], EpisodeRecorder],
        artifacts: Callable[[WorkerTask], Sequence[Path]],
    ) -> None:
        self.channel = channel
        self.reset_task = reset
        self.recorder_factory = recorder
        self.artifacts_for_task = artifacts
        self.current_task: WorkerTask | None = None
        self.current_recorder: EpisodeRecorder | None = None
        self.records: list[EpisodeRecord] = []
        self.attempt_ids: set[str] = set()
        self.output_dirs: set[Path] = set()

    def assign_task(self, payload: dict[str, Any]) -> None:
        task = _parse_task_assignment(payload)
        if task.attempt_id in self.attempt_ids:
            raise ValueError("scheduler repeated an executed attempt")
        output_dir = task.output_dir.resolve()
        if any(
            output_dir.is_relative_to(previous) or previous.is_relative_to(output_dir)
            for previous in self.output_dirs
        ):
            raise ValueError("scheduler reused an attempt output directory")
        self.attempt_ids.add(task.attempt_id)
        self.output_dirs.add(output_dir)
        self.current_task = task
        self.current_recorder = self.recorder_factory(task)

    def claim_episode_ids(self) -> Iterator[int]:
        while True:
            request_id = str(uuid4())
            self.channel.send("claim", {"request_id": request_id})
            reply = _receive_scheduler_frame(self.channel)
            payload = reply["payload"]
            if payload["request_id"] != request_id:
                raise ValueError("scheduler claim request ID mismatch")
            if reply["type"] == "done":
                return
            if reply["type"] == "wait":
                delay = payload["retry_after_sec"]
                if not isinstance(delay, (int, float)) or not 0 <= delay <= 60:
                    raise ValueError("scheduler retry_after_sec must be between 0 and 60")
                time.sleep(delay)
                continue
            if reply["type"] != "work":
                raise ValueError(f"unexpected scheduler response {reply['type']}")
            self.assign_task(payload)
            # The generic recipe requires distinct IDs. Restore the global index
            # before recording or reporting each attempt.
            yield len(self.attempt_ids) - 1
            self.current_task = None
            self.current_recorder = None

    def reset_episode(self, _episode_idx: int) -> Observation:
        assert self.current_task is not None
        return self.reset_task(self.current_task)

    def begin(self, episode_id: int) -> None:
        assert self.current_task is not None
        assert self.current_recorder is not None
        self.current_recorder.begin(self.current_task.episode_idx)

    def record(self, action: Action, *, success: bool, done: bool) -> None:
        assert self.current_recorder is not None
        self.current_recorder.record(action, success=success, done=done)

    def end(self) -> None:
        assert self.current_recorder is not None
        self.current_recorder.end()

    def report_episode(self, record: EpisodeRecord) -> None:
        assert self.current_task is not None
        task = self.current_task
        record = replace(record, episode_idx=task.episode_idx, task_id=task.task_id)
        result = asdict(record)
        result["started_at"] = record.started_at.isoformat()
        result["ended_at"] = record.ended_at.isoformat()
        artifacts = []
        for path in self.artifacts_for_task(task):
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(task.output_dir.resolve())
            if not resolved.is_file():
                raise ValueError("worker artifact must be a file")
            artifacts.append(
                {"episode_idx": record.episode_idx, "path": str(relative), "kind": "replay"}
            )
        # The episode loop closes the recorder before invoking this callback.
        self.channel.send(
            "complete",
            {
                "item_id": task.item_id,
                "attempt_id": task.attempt_id,
                "results": [result],
                "artifacts": artifacts,
            },
        )
        reply = _receive_scheduler_frame(self.channel)
        if reply["type"] not in ("accepted", "discarded"):
            raise ValueError(f"expected scheduler accepted or discarded, got {reply['type']}")
        receipt = reply["payload"]
        if receipt["item_id"] != task.item_id or receipt["attempt_id"] != task.attempt_id:
            raise ValueError("scheduler replied for a different item or attempt")
        if reply["type"] == "discarded":
            if receipt["reason"] != "lease_lost":
                raise ValueError("unexpected scheduler discard reason")
            return
        self.records.append(record)


def _receive_scheduler_frame(channel: FrameChannel, expected: str | None = None) -> dict[str, Any]:
    frame = channel.recv()
    if frame is None:
        raise RuntimeError("scheduler disconnected")
    if frame["type"] == "error":
        raise RuntimeError(frame["payload"]["message"])
    if expected is not None and frame["type"] != expected:
        raise ValueError(f"expected scheduler {expected}, got {frame['type']}")
    return frame


def _parse_task_assignment(payload: dict[str, Any]) -> WorkerTask:
    indices = payload["episode_indices"]
    if len(indices) != 1 or type(indices[0]) is not int or indices[0] < 0:
        raise ValueError("worker requires one non-negative episode index per item")
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
        episode_idx=indices[0],
        output_dir=output_dir,
    )


__all__ = [
    "WorkerEpisode",
    "WorkerTask",
    "run_worker",
]
