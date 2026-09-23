"""Behaviour of scheduler assignments and result acceptance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from manifold.core.values import Observation
from manifold.recipes import StepResult, WorkerEpisode, run_worker, serving, worker
from manifold.wire import FrameChannel, FrameType, bridge
from tests.test_serving import _fake_benchmark, _FakeSocket


class _Scheduler:
    def __init__(self, root: Path, events: list, *, ack="accepted", wait=False):
        self.root, self.events, self.ack, self.wait = root, events, ack, wait
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
        if self.assigned == 2:
            return {"type": "done", "payload": {"request_id": request_id}}
        idx = [3, 7][self.assigned]
        self.assigned += 1
        return {
            "type": "work",
            "payload": {
                "request_id": request_id,
                "item_id": f"item-{idx}",
                "attempt_id": f"attempt-{idx}",
                "task_id": str(idx),
                "task_config": {"suite": "libero_10", "task_id": idx, "seed": 2},
                "episode_indices": [idx],
                "output_dir": str(self.root / f"attempt-{idx}"),
            },
        }


class _Recorder:
    def __init__(self, task, events, fail_close):
        self.task, self.events, self.fail_close = task, events, fail_close

    def begin(self, episode_id):
        self.events.append(("begin", episode_id))

    def record(self, action, *, success, done):
        pass

    def end(self):
        self.events.append("close")
        if self.fail_close:
            raise OSError("replay close failed")
        self.task.output_dir.mkdir(parents=True)
        (self.task.output_dir / "replay.mfr").write_bytes(b"closed")


class _Policy:
    def __init__(self):
        self.sent = []
        self.width = 0

    def send(self, kind, payload):
        from manifold.core.benchmark import Benchmark

        self.sent.append(kind)
        if kind == FrameType.HELLO:
            # Answer in the width the benchmark declares, as a real policy would.
            benchmark = Benchmark.model_validate(payload["benchmark"])
            self.width = benchmark.embodiment.action.expected_length()

    def recv(self):
        from manifold.core.values import Action

        if self.sent[-1] == FrameType.HELLO:
            return {"type": FrameType.READY, "payload": {}}
        action = Action.from_array([0.0] * self.width)
        return {"type": FrameType.ACTION, "payload": bridge.encode_action(action)}


def _execute(
    monkeypatch, tmp_path, *, ack="accepted", wait=False, fail_close=False, fail_task=False
):
    events = []
    scheduler = _Scheduler(tmp_path, events, ack=ack, wait=wait)
    policy = _Policy()
    scheduler_socket = _FakeSocket()
    connections = []

    def policy_socket(*args):
        connections.append("policy")
        return _FakeSocket()

    monkeypatch.setattr(worker.socket, "create_connection", lambda address: scheduler_socket)
    monkeypatch.setattr(serving.socket, "socket", policy_socket)
    monkeypatch.setattr(
        FrameChannel,
        "from_socket",
        lambda sock, **kwargs: scheduler if sock is scheduler_socket else policy,
    )

    def reset(task):
        events.append(("reset", task.episode_idx, task.task_config))
        return Observation(instruction=f"task {task.task_id}")

    def step(action):
        if fail_task:
            raise RuntimeError("task failed")
        return StepResult(Observation(), success=True, done=True)

    def run():
        return run_worker(
            _fake_benchmark(),
            reset,
            step,
            server="policy:9000",
            scheduler_address="scheduler:9001",
            worker_id="worker-1",
            episodes=[WorkerEpisode(i, str(i), {"task_id": i}) for i in [3, 7]],
            max_steps=1,
            recorder=lambda task: _Recorder(task, events, fail_close),
            artifacts=lambda task: [task.output_dir / "replay.mfr"],
        )

    return run, scheduler, policy, connections, events


def test_worker_keeps_policy_connection_and_closes_replay_before_scheduler_acceptance(
    monkeypatch, tmp_path
):
    run, scheduler, policy, connections, events = _execute(monkeypatch, tmp_path, wait=True)
    result = run()
    assert connections == ["policy"]
    assert [r.episode_idx for r in result.records] == [3, 7]
    assert policy.sent.count(FrameType.HELLO) == 1
    assert policy.sent.count(FrameType.RESET) == 2
    assert policy.sent[-1] == FrameType.BYE
    completions = [payload for kind, payload in scheduler.sent if kind == "complete"]
    assert [p["results"][0]["task_id"] for p in completions] == ["3", "7"]
    assert [p["artifacts"][0]["path"] for p in completions] == ["replay.mfr", "replay.mfr"]
    assert events.index("close") < events.index("complete")
    first_complete = events.index("complete")
    assert events[first_complete + 1] == "claim"
    assert scheduler.sent[0][1]["episodes"][0]["episode_idx"] == 3


@pytest.mark.parametrize("fail_close,fail_task", [(True, False), (False, True)])
def test_worker_does_not_submit_or_retry_after_task_or_replay_failure(
    monkeypatch, tmp_path, fail_close, fail_task
):
    run, scheduler, _, _, events = _execute(
        monkeypatch, tmp_path, fail_close=fail_close, fail_task=fail_task
    )
    with pytest.raises((OSError, RuntimeError)):
        run()
    assert [kind for kind, _ in scheduler.sent] == ["hello", "claim"]
    assert events.count("close") == 1


def test_worker_does_not_claim_without_acceptance(monkeypatch, tmp_path):
    run, scheduler, _, _, _ = _execute(monkeypatch, tmp_path, ack="done")
    with pytest.raises(ValueError, match="expected scheduler accepted"):
        run()
    assert [kind for kind, _ in scheduler.sent] == ["hello", "claim", "complete"]


def test_worker_rejects_stale_acceptance(monkeypatch, tmp_path):
    run, scheduler, _, _, _ = _execute(monkeypatch, tmp_path)
    receive = scheduler.recv

    def stale_ack():
        reply = receive()
        if reply["type"] == "accepted":
            reply["payload"]["attempt_id"] = "expired-attempt"
        return reply

    monkeypatch.setattr(scheduler, "recv", stale_ack)
    with pytest.raises(ValueError, match="different item or attempt"):
        run()
    assert [kind for kind, _ in scheduler.sent] == ["hello", "claim", "complete"]


def test_worker_rejects_artifact_outside_attempt_directory(monkeypatch, tmp_path):
    run, scheduler, _, _, _ = _execute(monkeypatch, tmp_path)
    outside = tmp_path / "another-attempt.mfr"
    outside.write_bytes(b"other attempt")
    original_end = _Recorder.end

    def close_with_symlink(self):
        original_end(self)
        replay = self.task.output_dir / "replay.mfr"
        replay.unlink()
        replay.symlink_to(outside)

    monkeypatch.setattr(_Recorder, "end", close_with_symlink)
    with pytest.raises(ValueError):
        run()
    assert [kind for kind, _ in scheduler.sent] == ["hello", "claim"]


def test_worker_uses_length_prefixed_scheduler_frames(monkeypatch, tmp_path):
    import socket
    import threading

    client, server = socket.socketpair()
    server_channel = FrameChannel.from_socket(server)
    from_socket = FrameChannel.from_socket
    policy = _Policy()
    events = []
    scheduler = _Scheduler(tmp_path, events)
    errors = []

    def schedule():
        try:
            with server:
                while True:
                    frame = server_channel.recv()
                    if frame is None:
                        return
                    scheduler.send(frame["type"], frame["payload"])
                    reply = scheduler.recv()
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
    monkeypatch.setattr(serving.socket, "socket", lambda *args: _FakeSocket())
    monkeypatch.setattr(
        FrameChannel,
        "from_socket",
        lambda sock, **kwargs: from_socket(sock, **kwargs) if sock is client else policy,
    )
    thread = threading.Thread(target=schedule, daemon=True)
    thread.start()
    try:
        result = run_worker(
            _fake_benchmark(),
            lambda task: Observation(instruction="task"),
            lambda action: StepResult(Observation(), success=True, done=True),
            server="policy:9000",
            scheduler_address="scheduler:9001",
            worker_id="worker-1",
            episodes=[WorkerEpisode(i, str(i), {"task_id": i}) for i in [3, 7]],
            max_steps=1,
            recorder=lambda task: _Recorder(task, events, False),
            artifacts=lambda task: [task.output_dir / "replay.mfr"],
        )
    finally:
        client.close()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors == []
    assert [record.task_id for record in result.records] == ["3", "7"]
