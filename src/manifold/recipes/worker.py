"""Runner-assigned benchmark task execution.

`run_worker` registers the workload with the runner and claims one task at a
time. It reuses one policy connection to execute every task, reports the
result, and waits for task server acceptance before claiming the next task.
"""

from __future__ import annotations

import socket
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from manifold.lib.compat import assert_never
from manifold.recipes.serving import BenchmarkResult, run_episodes
from manifold.wire import (
    FrameChannel,
    TaskServerAccepted,
    TaskServerClaim,
    TaskServerComplete,
    TaskServerDone,
    TaskServerError,
    TaskServerFrameType,
    TaskServerHello,
    TaskServerWait,
    TaskServerWork,
    WorkerArtifact,
    WorkerEpisode,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from manifold.core.benchmark import Benchmark
    from manifold.core.values import Action, Observation
    from manifold.recipes.recording import EpisodeRecorder
    from manifold.recipes.serving import EpisodeRecord, StepResult
    from manifold.wire import ImageFormat


@dataclass(frozen=True)
class WorkerTask:
    """One attempt assigned by the task server and its output directory."""

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
    task_server_address: str,
    worker_id: str,
    episodes: Sequence[WorkerEpisode],
    max_steps: int,
    recorder: Callable[[WorkerTask], EpisodeRecorder],
    artifacts: Callable[[WorkerTask], Sequence[Path]],
    image_format: ImageFormat = "raw",
) -> BenchmarkResult:
    """Execute task server assignments with one policy connection.

    Send results after closing each replay, then wait for task server acceptance
    before claiming again.
    """
    host, separator, port = task_server_address.rpartition(":")
    if not separator or not host:
        raise ValueError("task_server_address must be host:port")
    with socket.create_connection((host, int(port))) as sock:
        channel = FrameChannel.from_socket(sock)
        hello = TaskServerHello(
            worker_id=worker_id, benchmark_name=benchmark.name, episodes=tuple(episodes)
        )
        channel.send(hello.frame_type, hello.to_payload())
        _receive_task_server_frame(channel, TaskServerFrameType.READY)
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

    def claim_episode_ids(self) -> Iterator[int]:
        while True:
            claim = TaskServerClaim(request_id=str(uuid4()))
            self.channel.send(claim.frame_type, claim.to_payload())
            response = _parse_claim_response(_receive_task_server_frame(self.channel))
            if response.request_id != claim.request_id:
                raise ValueError("task server claim request ID mismatch")
            if isinstance(response, TaskServerDone):
                return
            elif isinstance(response, TaskServerWait):
                delay = response.retry_after_sec
                if not 0 <= delay <= 60:
                    raise ValueError("task server retry_after_sec must be between 0 and 60")
                time.sleep(delay)
            elif isinstance(response, TaskServerWork):
                self.current_task = _worker_task_from_work(response)
                self.current_recorder = self.recorder_factory(self.current_task)
                yield self.current_task.episode_idx
                self.current_task = None
                self.current_recorder = None
            else:
                assert_never(response)

    def reset_episode(self, _episode_idx: int) -> Observation:
        assert self.current_task is not None
        return self.reset_task(self.current_task)

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
        record = replace(record, task_id=task.task_id)
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
                WorkerArtifact(episode_idx=record.episode_idx, path=str(relative), kind="replay")
            )
        # The episode loop closes the recorder before invoking this callback.
        complete = TaskServerComplete(
            item_id=task.item_id,
            attempt_id=task.attempt_id,
            results=(result,),
            artifacts=tuple(artifacts),
        )
        self.channel.send(complete.frame_type, complete.to_payload())
        accepted = TaskServerAccepted.from_payload(
            _receive_task_server_frame(self.channel, TaskServerFrameType.ACCEPTED)["payload"]
        )
        if accepted.item_id != task.item_id or accepted.attempt_id != task.attempt_id:
            raise ValueError("task server accepted a different item or attempt")
        self.records.append(record)


def _receive_task_server_frame(
    channel: FrameChannel, expected: TaskServerFrameType | None = None
) -> dict[str, Any]:
    frame = channel.recv()
    if frame is None:
        raise RuntimeError("task server disconnected")
    if frame["type"] == TaskServerFrameType.ERROR:
        raise RuntimeError(TaskServerError.from_payload(frame["payload"]).message)
    if expected is not None and frame["type"] != expected:
        raise ValueError(f"expected task server {expected}, got {frame['type']}")
    return frame


def _parse_claim_response(
    frame: dict[str, Any],
) -> TaskServerDone | TaskServerWait | TaskServerWork:
    if frame["type"] == TaskServerFrameType.DONE:
        return TaskServerDone.from_payload(frame["payload"])
    if frame["type"] == TaskServerFrameType.WAIT:
        return TaskServerWait.from_payload(frame["payload"])
    if frame["type"] == TaskServerFrameType.WORK:
        return TaskServerWork.from_payload(frame["payload"])
    raise ValueError(f"unexpected task server response {frame['type']}")


def _worker_task_from_work(work: TaskServerWork) -> WorkerTask:
    indices = work.episode_indices
    if len(indices) != 1 or indices[0] < 0:
        raise ValueError("worker requires one non-negative episode index per item")
    output_dir = Path(work.output_dir)
    if not output_dir.is_absolute():
        raise ValueError("worker output_dir must be absolute")
    for key, value in (
        ("item_id", work.item_id),
        ("attempt_id", work.attempt_id),
        ("task_id", work.task_id),
    ):
        if not value:
            raise ValueError(f"worker {key} must be a non-empty string")
    return WorkerTask(
        item_id=work.item_id,
        attempt_id=work.attempt_id,
        task_id=work.task_id,
        task_config=work.task_config,
        episode_idx=indices[0],
        output_dir=output_dir,
    )


__all__ = [
    "WorkerEpisode",
    "WorkerTask",
    "run_worker",
]
