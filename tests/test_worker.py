"""Behaviour of task server assignments and result acceptance."""

from __future__ import annotations

import socket
import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from manifold.benchmark import WorkerEpisodeRecord, WorkerSession, WorkerTask
from manifold.benchmark import _worker as worker
from manifold.wire import FrameChannel


class _TaskServer:
    def __init__(
        self,
        root: Path,
        events: list,
        *,
        ack="accepted",
        wait=False,
        tasks: Sequence[list[int]] = ([3], [7]),
    ):
        self.root, self.events, self.ack, self.wait = root, events, ack, wait
        self.tasks = tasks
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.assigned = 0

    def send(self, kind, payload):
        self.sent.append((kind, payload))
        self.events.append(kind)

    def recv(self):
        kind, payload = self.sent[-1]
        if kind == "hello":
            return {"type": "ready", "payload": {}}
        if kind == "complete":
            return {
                "type": self.ack,
                "payload": {"item_id": payload["item_id"], "attempt_id": payload["attempt_id"]},
            }
        request_id = payload["request_id"]
        if self.wait:
            self.wait = False
            return {"type": "wait", "payload": {"request_id": request_id, "retry_after_sec": 0}}
        if self.assigned == len(self.tasks):
            return {"type": "done", "payload": {"request_id": request_id}}
        indices = self.tasks[self.assigned]
        idx = indices[0]
        self.assigned += 1
        return {
            "type": "work",
            "payload": {
                "request_id": request_id,
                "item_id": f"item-{idx}",
                "attempt_id": f"attempt-{idx}",
                "task_id": str(idx),
                "task_config": {"suite": "libero_10", "task_id": idx, "seed": 2},
                "episode_indices": indices,
                "output_dir": str(self.root / f"attempt-{idx}"),
            },
        }


def _open_session(monkeypatch, task_server):
    task_server_socket = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(worker.socket, "create_connection", lambda address: task_server_socket)
    monkeypatch.setattr(FrameChannel, "from_socket", lambda sock, **kwargs: task_server)
    return WorkerSession(
        "task-server:9001",
        worker_id="worker-1",
    )


def _record(episode_idx):
    now = datetime.now(timezone.utc)
    return WorkerEpisodeRecord(
        episode_idx=episode_idx,
        task_name="task 3",
        success=True,
        steps=1,
        initialization_sec=0.0,
        started_at=now,
        ended_at=now,
        task_id="3",
        attributes={},
    )


def test_session_completes_a_multi_episode_task_in_one_message_from_another_thread(
    monkeypatch, tmp_path
):
    task_server = _TaskServer(tmp_path, [], tasks=[[3, 4, 5]])
    claimed = []
    errors = []

    def coordinate():
        try:
            with _open_session(monkeypatch, task_server) as session:
                task = session.claim()
                assert task is not None
                claimed.append(task)
                task.output_dir.mkdir(parents=True)
                replays = {}
                for idx in task.episode_indices:
                    replay = task.output_dir / f"episode-{idx}.mfr"
                    replay.write_bytes(b"closed")
                    replays[idx] = [replay]
                session.complete(task, [_record(idx) for idx in task.episode_indices], replays)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=coordinate)
    thread.start()
    thread.join(timeout=5)

    assert errors == []
    assert [task.episode_indices for task in claimed] == [(3, 4, 5)]
    completions = [payload for kind, payload in task_server.sent if kind == "complete"]
    assert len(completions) == 1
    assert [result["episode_idx"] for result in completions[0]["results"]] == [3, 4, 5]
    assert [(a["episode_idx"], a["path"]) for a in completions[0]["artifacts"]] == [
        (3, "episode-3.mfr"),
        (4, "episode-4.mfr"),
        (5, "episode-5.mfr"),
    ]


def test_session_claims_again_after_wait(monkeypatch, tmp_path):
    task_server = _TaskServer(tmp_path, [], wait=True)
    with _open_session(monkeypatch, task_server) as session:
        task = session.claim()
    assert task is not None
    assert task.episode_indices == (3,)
    assert [kind for kind, _ in task_server.sent] == ["hello", "claim", "claim"]


def test_session_claim_returns_none_on_done(monkeypatch, tmp_path):
    task_server = _TaskServer(tmp_path, [], tasks=[])
    with _open_session(monkeypatch, task_server) as session:
        assert session.claim() is None


def _replays(task: WorkerTask) -> dict[int, list[Path]]:
    """A closed replay file for each episode of `task`, under its output directory."""
    task.output_dir.mkdir(parents=True)
    replays = {}
    for idx in task.episode_indices:
        replay = task.output_dir / f"episode-{idx}.mfr"
        replay.write_bytes(b"closed")
        replays[idx] = [replay]
    return replays


def _claim_and_complete(session: WorkerSession) -> None:
    task = session.claim()
    assert task is not None
    session.complete(task, [_record(idx) for idx in task.episode_indices], _replays(task))


def test_session_refuses_a_reply_other_than_accepted(monkeypatch, tmp_path):
    task_server = _TaskServer(tmp_path, [], ack="done")
    with (
        _open_session(monkeypatch, task_server) as session,
        pytest.raises(ValueError, match="expected task server accepted"),
    ):
        _claim_and_complete(session)
    assert [kind for kind, _ in task_server.sent] == ["hello", "claim", "complete"]


def test_session_refuses_an_acceptance_of_another_attempt(monkeypatch, tmp_path):
    task_server = _TaskServer(tmp_path, [])
    receive = task_server.recv

    def stale_ack():
        reply = receive()
        if reply["type"] == "accepted":
            reply["payload"]["attempt_id"] = "expired-attempt"
        return reply

    monkeypatch.setattr(task_server, "recv", stale_ack)
    with (
        _open_session(monkeypatch, task_server) as session,
        pytest.raises(ValueError, match="different item or attempt"),
    ):
        _claim_and_complete(session)
    assert [kind for kind, _ in task_server.sent] == ["hello", "claim", "complete"]


def test_session_refuses_a_replay_outside_the_attempt_directory(monkeypatch, tmp_path):
    task_server = _TaskServer(tmp_path, [])
    outside = tmp_path / "another-attempt.mfr"
    outside.write_bytes(b"other attempt")
    with _open_session(monkeypatch, task_server) as session:
        task = session.claim()
        assert task is not None
        replays = _replays(task)
        replay = replays[3][0]
        replay.unlink()
        replay.symlink_to(outside)
        with pytest.raises(ValueError):
            session.complete(task, [_record(3)], replays)
    assert [kind for kind, _ in task_server.sent] == ["hello", "claim"]


def test_session_uses_length_prefixed_task_server_frames(monkeypatch, tmp_path):
    client, server = socket.socketpair()
    server_channel = FrameChannel.from_socket(server)
    task_server = _TaskServer(tmp_path, [])
    errors = []

    def serve():
        try:
            with server:
                while True:
                    frame = server_channel.recv()
                    if frame is None:
                        return
                    task_server.send(frame["type"], frame["payload"])
                    reply = task_server.recv()
                    if frame["type"] == "complete":
                        task = frame["payload"]
                        artifact = task["artifacts"][0]
                        path = tmp_path / task["attempt_id"] / artifact["path"]
                        assert path.read_bytes() == b"closed"
                    server_channel.send(reply["type"], reply["payload"])
                    if reply["type"] == "done":
                        return
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(worker.socket, "create_connection", lambda address: client)
    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        with WorkerSession(
            "task-server:9001",
            worker_id="worker-1",
        ) as session:
            _claim_and_complete(session)
            _claim_and_complete(session)
            assert session.claim() is None
    finally:
        client.close()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors == []
    completions = [payload for kind, payload in task_server.sent if kind == "complete"]
    assert [p["results"][0]["episode_idx"] for p in completions] == [3, 7]
