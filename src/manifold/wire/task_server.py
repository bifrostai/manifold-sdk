"""The task server protocol: the messages a benchmark worker and a task server exchange.

The worker opens with HELLO and its episode manifest, and the task server replies
READY. The worker then sends CLAIM, and the task server replies WORK, WAIT or DONE
with the claim's `request_id`. After each WORK, the worker sends COMPLETE and waits
for ACCEPTED before it claims again. The task server sends ERROR when it fails to
handle a message, then closes the connection.

Each message is sent as one `FrameChannel` frame: the frame type is the message's
`frame_type`, and the payload is `to_payload()`. `from_payload` parses an untrusted
payload, so a missing key is a `KeyError` and a value of the wrong type is a
`ValueError`, as in `bridge.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from manifold.lib.compat import StrEnum


def _read_str(payload: dict[str, Any], key: str, *, where: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise ValueError(f"{where} {key} must be a string")  # noqa: TRY004
    return value


def _read_int(payload: dict[str, Any], key: str, *, where: str) -> int:
    value = payload[key]
    if type(value) is not int:
        raise ValueError(f"{where} {key} must be an integer")
    return value


def _read_number(payload: dict[str, Any], key: str, *, where: str) -> float:
    value = payload[key]
    if not isinstance(value, (int, float)):
        raise ValueError(f"{where} {key} must be a number")  # noqa: TRY004
    return value


def _read_mapping(payload: dict[str, Any], key: str, *, where: str) -> dict[str, Any]:
    value = payload[key]
    if not isinstance(value, dict):
        raise ValueError(f"{where} {key} must be a mapping")  # noqa: TRY004
    return value


def _read_list(payload: dict[str, Any], key: str, *, where: str) -> list[Any]:
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"{where} {key} must be a list")  # noqa: TRY004
    return value


def _read_entries(payload: dict[str, Any], key: str, *, where: str) -> list[dict[str, Any]]:
    entries = _read_list(payload, key, where=where)
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{where} {key} must contain mappings")  # noqa: TRY004
    return entries


class TaskServerFrameType(StrEnum):
    """The kinds of frame a benchmark worker and a task server exchange."""

    HELLO = "hello"  # worker -> task server, once, with the episode manifest.
    READY = "ready"  # task server -> worker, after it accepts the manifest.
    CLAIM = "claim"  # worker -> task server, to request the next task.
    WORK = "work"  # task server -> worker, one attempt in reply to a claim.
    WAIT = "wait"  # task server -> worker, claim again after a delay.
    DONE = "done"  # task server -> worker, every task is assigned.
    COMPLETE = "complete"  # worker -> task server, the results of one attempt.
    ACCEPTED = "accepted"  # task server -> worker, the results are accepted.
    ERROR = "error"  # task server -> worker, before it closes the connection.


@dataclass(frozen=True)
class WorkerEpisode:
    """One episode in the workload's ordered task and seed manifest."""

    episode_idx: int
    task_id: str
    task_config: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        """The manifest entry for a HELLO payload."""
        return {
            "episode_idx": self.episode_idx,
            "task_id": self.task_id,
            "task_config": self.task_config,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> WorkerEpisode:
        """Read one manifest entry from a HELLO payload."""
        where = "hello episode"
        return cls(
            episode_idx=_read_int(payload, "episode_idx", where=where),
            task_id=_read_str(payload, "task_id", where=where),
            task_config=_read_mapping(payload, "task_config", where=where),
        )


@dataclass(frozen=True)
class WorkerArtifact:
    """One file the worker wrote under an attempt's output directory."""

    episode_idx: int
    path: str
    kind: str

    def to_payload(self) -> dict[str, Any]:
        """The artifact entry for a COMPLETE payload."""
        return {"episode_idx": self.episode_idx, "path": self.path, "kind": self.kind}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> WorkerArtifact:
        """Read one artifact entry from a COMPLETE payload."""
        where = "complete artifact"
        return cls(
            episode_idx=_read_int(payload, "episode_idx", where=where),
            path=_read_str(payload, "path", where=where),
            kind=_read_str(payload, "kind", where=where),
        )


@dataclass(frozen=True)
class TaskServerHello:
    """The worker's opening message and its episode manifest."""

    frame_type: ClassVar[TaskServerFrameType] = TaskServerFrameType.HELLO

    worker_id: str
    benchmark_name: str
    episodes: tuple[WorkerEpisode, ...]

    def to_payload(self) -> dict[str, Any]:
        """The HELLO frame payload."""
        return {
            "worker_id": self.worker_id,
            "benchmark_name": self.benchmark_name,
            "episodes": [episode.to_payload() for episode in self.episodes],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TaskServerHello:
        """Read a HELLO frame payload."""
        where = "hello payload"
        return cls(
            worker_id=_read_str(payload, "worker_id", where=where),
            benchmark_name=_read_str(payload, "benchmark_name", where=where),
            episodes=tuple(
                WorkerEpisode.from_payload(entry)
                for entry in _read_entries(payload, "episodes", where=where)
            ),
        )


@dataclass(frozen=True)
class TaskServerReady:
    """The task server's reply to HELLO."""

    frame_type: ClassVar[TaskServerFrameType] = TaskServerFrameType.READY

    def to_payload(self) -> dict[str, Any]:
        """The READY frame payload."""
        return {}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TaskServerReady:
        """Read a READY frame payload."""
        return cls()


@dataclass(frozen=True)
class TaskServerClaim:
    """The worker's request for its next task."""

    frame_type: ClassVar[TaskServerFrameType] = TaskServerFrameType.CLAIM

    request_id: str

    def to_payload(self) -> dict[str, Any]:
        """The CLAIM frame payload."""
        return {"request_id": self.request_id}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TaskServerClaim:
        """Read a CLAIM frame payload."""
        return cls(request_id=_read_str(payload, "request_id", where="claim payload"))


@dataclass(frozen=True)
class TaskServerWork:
    """One attempt the task server assigns in reply to a claim."""

    frame_type: ClassVar[TaskServerFrameType] = TaskServerFrameType.WORK

    request_id: str
    item_id: str
    attempt_id: str
    task_id: str
    task_config: dict[str, Any]
    episode_indices: tuple[int, ...]
    output_dir: str

    def to_payload(self) -> dict[str, Any]:
        """The WORK frame payload."""
        return {
            "request_id": self.request_id,
            "item_id": self.item_id,
            "attempt_id": self.attempt_id,
            "task_id": self.task_id,
            "task_config": self.task_config,
            "episode_indices": list(self.episode_indices),
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TaskServerWork:
        """Read a WORK frame payload."""
        where = "work payload"
        episode_indices = _read_list(payload, "episode_indices", where=where)
        if any(type(index) is not int for index in episode_indices):
            raise ValueError(f"{where} episode_indices must contain integers")
        return cls(
            request_id=_read_str(payload, "request_id", where=where),
            item_id=_read_str(payload, "item_id", where=where),
            attempt_id=_read_str(payload, "attempt_id", where=where),
            task_id=_read_str(payload, "task_id", where=where),
            task_config=_read_mapping(payload, "task_config", where=where),
            episode_indices=tuple(episode_indices),
            output_dir=_read_str(payload, "output_dir", where=where),
        )


@dataclass(frozen=True)
class TaskServerWait:
    """The task server's reply to a claim when it cannot assign a task yet."""

    frame_type: ClassVar[TaskServerFrameType] = TaskServerFrameType.WAIT

    request_id: str
    retry_after_sec: float

    def to_payload(self) -> dict[str, Any]:
        """The WAIT frame payload."""
        return {"request_id": self.request_id, "retry_after_sec": self.retry_after_sec}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TaskServerWait:
        """Read a WAIT frame payload."""
        where = "wait payload"
        return cls(
            request_id=_read_str(payload, "request_id", where=where),
            retry_after_sec=_read_number(payload, "retry_after_sec", where=where),
        )


@dataclass(frozen=True)
class TaskServerDone:
    """The task server's reply to a claim once every task is assigned."""

    frame_type: ClassVar[TaskServerFrameType] = TaskServerFrameType.DONE

    request_id: str

    def to_payload(self) -> dict[str, Any]:
        """The DONE frame payload."""
        return {"request_id": self.request_id}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TaskServerDone:
        """Read a DONE frame payload."""
        return cls(request_id=_read_str(payload, "request_id", where="done payload"))


@dataclass(frozen=True)
class TaskServerComplete:
    """The worker's results and artifacts for one attempt."""

    frame_type: ClassVar[TaskServerFrameType] = TaskServerFrameType.COMPLETE

    item_id: str
    attempt_id: str
    results: tuple[dict[str, Any], ...]
    artifacts: tuple[WorkerArtifact, ...]

    def to_payload(self) -> dict[str, Any]:
        """The COMPLETE frame payload."""
        return {
            "item_id": self.item_id,
            "attempt_id": self.attempt_id,
            "results": list(self.results),
            "artifacts": [artifact.to_payload() for artifact in self.artifacts],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TaskServerComplete:
        """Read a COMPLETE frame payload."""
        where = "complete payload"
        return cls(
            item_id=_read_str(payload, "item_id", where=where),
            attempt_id=_read_str(payload, "attempt_id", where=where),
            results=tuple(_read_entries(payload, "results", where=where)),
            artifacts=tuple(
                WorkerArtifact.from_payload(entry)
                for entry in _read_entries(payload, "artifacts", where=where)
            ),
        )


@dataclass(frozen=True)
class TaskServerAccepted:
    """The task server's reply once it accepts an attempt's results."""

    frame_type: ClassVar[TaskServerFrameType] = TaskServerFrameType.ACCEPTED

    item_id: str
    attempt_id: str

    def to_payload(self) -> dict[str, Any]:
        """The ACCEPTED frame payload."""
        return {"item_id": self.item_id, "attempt_id": self.attempt_id}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TaskServerAccepted:
        """Read an ACCEPTED frame payload."""
        where = "accepted payload"
        return cls(
            item_id=_read_str(payload, "item_id", where=where),
            attempt_id=_read_str(payload, "attempt_id", where=where),
        )


@dataclass(frozen=True)
class TaskServerError:
    """The task server's reply to a message it fails to handle."""

    frame_type: ClassVar[TaskServerFrameType] = TaskServerFrameType.ERROR

    message: str

    def to_payload(self) -> dict[str, Any]:
        """The ERROR frame payload."""
        return {"message": self.message}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> TaskServerError:
        """Read an ERROR frame payload."""
        return cls(message=_read_str(payload, "message", where="error payload"))


__all__ = [
    "TaskServerAccepted",
    "TaskServerClaim",
    "TaskServerComplete",
    "TaskServerDone",
    "TaskServerError",
    "TaskServerFrameType",
    "TaskServerHello",
    "TaskServerReady",
    "TaskServerWait",
    "TaskServerWork",
    "WorkerArtifact",
    "WorkerEpisode",
]
