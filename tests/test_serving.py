"""Unit tests for the open-loop chunk-queue Session base in recipes.serving.

ActionQueue holds one raw model chunk and serves it
`signature.execution_steps` times before running a fresh forward — the reusable
half of the serve loop a policy backend subclasses. These exercise that horizon
logic against a fake endpoint (no model, no torch).
"""

from __future__ import annotations

import socket
import threading
from contextlib import suppress
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from manifold.benchmark import ResetResult, StepResult
from manifold.recipes import ActionQueue
from manifold.recipes.serving import PolicyEndpoint
from manifold.sensors import CameraName


def _endpoint(execution_steps: int) -> PolicyEndpoint:
    """A fake endpoint that carries only the signature field that the queue reads."""
    return cast(
        PolicyEndpoint,
        SimpleNamespace(signature=SimpleNamespace(execution_steps=execution_steps)),
    )


class _CountingQueue(ActionQueue):
    """A queue whose `_forward` returns a fresh sentinel and counts its calls."""

    def __init__(self, execution_steps: int) -> None:
        super().__init__(_endpoint(execution_steps))
        self.forwards = 0

    def _forward(self, native, /):
        self.forwards += 1
        return f"chunk{self.forwards}"


def test_advance_forwards_on_drain_then_serves_buffered_steps():
    q = _CountingQueue(execution_steps=3)
    # First advance: empty buffer -> forward, step 0.
    assert q.advance({}) == ("chunk1", 0)
    # Next two: serve the buffered chunk at steps 1, 2 — no new forward.
    assert q.advance({}) == ("chunk1", 1)
    assert q.advance({}) == ("chunk1", 2)
    assert q.forwards == 1
    # execution_steps=3 served -> horizon drained -> fresh forward, step resets to 0.
    assert q.advance({}) == ("chunk2", 0)
    assert q.forwards == 2


def test_reset_clears_buffer_so_next_advance_forwards():
    q = _CountingQueue(execution_steps=5)
    q.advance({})  # forward -> chunk1, step 0
    q.advance({})  # buffered -> chunk1, step 1
    assert q.forwards == 1
    q.reset()
    # Buffer cleared: the next advance forwards again from step 0 mid-horizon.
    assert q.advance({}) == ("chunk2", 0)
    assert q.forwards == 2


def test_base_forward_raises_so_a_backend_cannot_silently_serve_stale():
    q = ActionQueue(_endpoint(2))
    with pytest.raises(NotImplementedError):
        q.advance({})


def test_infer_is_the_unused_no_packing_path():
    q = _CountingQueue(execution_steps=2)
    with pytest.raises(NotImplementedError):
        q.infer(object())


def test_close_is_a_noop_for_the_stateless_base():
    q = _CountingQueue(execution_steps=2)
    assert q.close() is None


# --- launch_server ----------------------------------------------------------------


def _pairing(profile, benchmark="B", pipeline="PIPE"):
    # Build via read_pairing so the fake (str) benchmark/pipeline launder through the
    # module's Any-typed attributes, rather than constructing the typed Pairing with
    # mismatched literals.
    from manifold.recipes import read_pairing

    module = cast(
        ModuleType,
        SimpleNamespace(
            __name__="test.pairing", PROFILE=profile, BENCHMARK=benchmark, PIPELINE=pipeline
        ),
    )
    return read_pairing(module)


def test_launch_server_loads_the_shared_profile_and_serves_the_single_pipeline(monkeypatch):
    from manifold.recipes import launch_server, serving

    captured: dict = {}
    monkeypatch.setattr(
        serving, "serve", lambda endpoint, **kw: captured.update(endpoint=endpoint, **kw)
    )
    sentinel_endpoint = object()
    profile = SimpleNamespace(
        default_weights="DEFAULT",
        load=lambda weights, device: (
            captured.update(weights=weights, device=device) or sentinel_endpoint
        ),
    )

    launch_server([_pairing(profile)], port=9000, max_workers=3)

    assert captured["endpoint"] is sentinel_endpoint
    assert captured["pipeline"] == "PIPE"  # a single pairing serves its own pipeline
    assert captured["weights"] == "DEFAULT"  # weights=None -> the profile's default
    assert captured["port"] == 9000


def test_launch_server_honors_an_explicit_weights_override(monkeypatch):
    from manifold.recipes import launch_server, serving

    captured: dict = {}
    monkeypatch.setattr(serving, "serve", lambda endpoint, **kw: None)
    profile = SimpleNamespace(
        default_weights="DEFAULT",
        load=lambda weights, device: captured.update(weights=weights) or object(),
    )

    launch_server([_pairing(profile)], weights="OVERRIDE", port=9000)

    assert captured["weights"] == "OVERRIDE"


def test_launch_server_rejects_pairings_that_do_not_share_one_profile(monkeypatch):
    from manifold.recipes import launch_server, serving

    monkeypatch.setattr(serving, "serve", lambda *a, **k: None)
    profile_a = SimpleNamespace(default_weights="w", load=lambda *a: object())
    profile_b = SimpleNamespace(default_weights="w", load=lambda *a: object())  # distinct object

    with pytest.raises(ValueError):
        launch_server([_pairing(profile_a), _pairing(profile_b)], port=9000)


# serve


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return cast(int, probe.getsockname()[1])


def test_serve_accepts_connections_beyond_max_workers(monkeypatch):
    from manifold.recipes import serve, serving

    # Each connection goes to a handler that holds it open until the test ends.
    # The accept loop reaches the last connection because `max_workers` no longer
    # limits how many connections are served at once.
    connections = 12
    port = _free_port()
    accepted = threading.Semaphore(0)
    accepted_count = 0
    listening = threading.Event()
    release = threading.Event()

    def hold(*args: Any) -> None:
        nonlocal accepted_count
        accepted_count += 1
        accepted.release()
        release.wait(timeout=10)

    def record(event: str) -> None:
        if event.startswith("listening on"):
            listening.set()

    monkeypatch.setattr(serving, "_serve_connection", hold)
    server = threading.Thread(
        target=serve,
        args=(cast(Any, SimpleNamespace()),),  # never read: no connection sends HELLO
        kwargs={"host": "127.0.0.1", "port": port, "max_workers": 1, "on_event": record},
        daemon=True,
    )
    server.start()
    assert listening.wait(timeout=5), "the server never bound its port"

    clients = []
    try:
        for _ in range(connections):
            clients.append(socket.create_connection(("127.0.0.1", port), timeout=5))
        for accepted_so_far in range(connections):
            assert accepted.acquire(timeout=5), (
                f"the accept loop stopped after {accepted_so_far} of {connections} connections"
            )
    finally:
        release.set()
        for client in clients:
            with suppress(OSError):
                client.close()
    assert accepted_count == connections


def test_serve_logs_nothing_for_a_connection_that_sends_no_frame():
    """The Manifold CLI connects and closes to check that the server is up.
    That check is not a benchmark, so it leaves no log lines."""
    from manifold.recipes import serve
    from manifold.wire.bridge import FrameChannel, FrameType

    port = _free_port()
    events: list[str] = []
    listening = threading.Event()
    spoke_closed = threading.Event()

    def record(event: str) -> None:
        events.append(event)
        if event.startswith("listening on"):
            listening.set()
        if event.startswith("connection from") and event.endswith("closed"):
            spoke_closed.set()

    threading.Thread(
        target=serve,
        args=(cast(Any, SimpleNamespace()),),  # never read: no connection sends HELLO
        kwargs={"host": "127.0.0.1", "port": port, "on_event": record},
        daemon=True,
    ).start()
    assert listening.wait(timeout=5), "the server never bound its port"

    probe = socket.create_connection(("127.0.0.1", port), timeout=5)
    probe_addr = str(probe.getsockname())
    probe.close()
    with socket.create_connection(("127.0.0.1", port), timeout=5) as speaker:
        FrameChannel.from_socket(speaker).send(FrameType.BYE, {})
        assert spoke_closed.wait(timeout=5), "the server never closed the second connection"

    assert not any(probe_addr in event for event in events)
    assert any(event.startswith("benchmark connected from") for event in events)


def _install_sdk_from(monkeypatch: pytest.MonkeyPatch, direct_url: str) -> None:
    """Make the installed manifold-sdk report `direct_url` as its install origin."""
    from manifold.recipes import serving

    installed = SimpleNamespace(read_text=lambda name: direct_url)
    monkeypatch.setattr(serving, "distribution", lambda name: installed)


def test_sdk_commit_reads_the_commit_of_a_git_install(monkeypatch):
    from manifold.recipes.serving import _sdk_commit

    _install_sdk_from(
        monkeypatch,
        '{"url": "https://github.com/bifrostai/manifold-sdk.git",'
        ' "vcs_info": {"vcs": "git", "commit_id": "3cba8f4e0c7d"}}',
    )
    assert _sdk_commit() == "3cba8f4e0c7d"


def test_sdk_commit_is_none_for_a_local_install(monkeypatch):
    from manifold.recipes.serving import _sdk_commit

    _install_sdk_from(
        monkeypatch, '{"url": "file:///src/manifold-sdk", "dir_info": {"editable": true}}'
    )
    assert _sdk_commit() is None


def test_serve_replies_to_get_signature_with_the_signature(monkeypatch):
    """The Manifold CLI asks for the signature once the server is up, and sends
    it to Manifold. The server does not log the request, because the request
    does not come from a benchmark."""
    from manifold.core import EEActionSpace, PolicySignature, RotationFormat
    from manifold.recipes import serve
    from manifold.wire.bridge import FrameChannel, FrameType

    _install_sdk_from(
        monkeypatch,
        '{"url": "https://github.com/bifrostai/manifold-sdk.git",'
        ' "vcs_info": {"vcs": "git", "commit_id": "3cba8f4e0c7d"}}',
    )
    signature = PolicySignature(
        action_space=EEActionSpace(rotation=RotationFormat.AXIS_ANGLE, delta=True),
        chunk_size=10,
        execution_steps=5,
    )
    port = _free_port()
    events: list[str] = []
    listening = threading.Event()

    def record(event: str) -> None:
        events.append(event)
        if event.startswith("listening on"):
            listening.set()

    threading.Thread(
        target=serve,
        args=(cast(Any, SimpleNamespace(signature=signature)),),
        kwargs={"host": "127.0.0.1", "port": port, "on_event": record},
        daemon=True,
    ).start()
    assert listening.wait(timeout=5), "the server never bound its port"

    with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
        channel = FrameChannel.from_socket(client)
        channel.send(FrameType.GET_SIGNATURE, {})
        reply = channel.recv()

    assert reply is not None
    assert reply["type"] == FrameType.SIGNATURE
    assert reply["payload"] == {
        "signature": signature.model_dump(mode="json"),
        "sdk_version": "3cba8f4e0c7d",
        # The thread starts `serve` directly. The stack holds only threading
        # code, so the script is null.
        "script": None,
    }
    assert events == [event for event in events if event.startswith("listening on")]


def test_serve_replies_to_get_signature_with_the_serving_script(monkeypatch):
    """`manifold policy serve` uploads the serving script, so the run page shows
    the code that served the run. Here the serving script is this test file."""
    from manifold.core import EEActionSpace, PolicySignature, RotationFormat
    from manifold.recipes import serve
    from manifold.wire.bridge import FrameChannel, FrameType

    monkeypatch.chdir(Path(__file__).parents[1])
    signature = PolicySignature(action_space=EEActionSpace(rotation=RotationFormat.AXIS_ANGLE))
    port = _free_port()
    listening = threading.Event()

    def record(event: str) -> None:
        if event.startswith("listening on"):
            listening.set()

    def serving_script() -> None:
        serve(
            cast(Any, SimpleNamespace(signature=signature)),
            host="127.0.0.1",
            port=port,
            on_event=record,
        )

    threading.Thread(target=serving_script, daemon=True).start()
    assert listening.wait(timeout=5), "the server never bound its port"

    with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
        channel = FrameChannel.from_socket(client)
        channel.send(FrameType.GET_SIGNATURE, {})
        reply = channel.recv()

    assert reply is not None
    assert reply["payload"]["script"] == {
        "path": "tests/test_serving.py",
        "text": Path(__file__).read_text(encoding="utf-8"),
    }


def test_the_serving_script_is_none_for_a_file_longer_than_the_backend_stores(
    monkeypatch,
):
    from manifold.recipes import serving

    monkeypatch.setattr(serving, "MAX_SCRIPT_CHARS", 10)

    assert serving._serving_script() is None


def test_serve_reads_the_next_frame_after_replying_to_get_signature():
    """A benchmark can ask for the signature, choose its cameras, and then send
    HELLO on the same connection."""
    from manifold.core import EEActionSpace, PolicySignature, RotationFormat
    from manifold.recipes import serve
    from manifold.wire.bridge import FrameChannel, FrameType

    signature = PolicySignature(action_space=EEActionSpace(rotation=RotationFormat.AXIS_ANGLE))
    port = _free_port()
    listening = threading.Event()
    spoke_closed = threading.Event()
    events: list[str] = []

    def record(event: str) -> None:
        events.append(event)
        if event.startswith("listening on"):
            listening.set()
        if event.startswith("connection from") and event.endswith("closed"):
            spoke_closed.set()

    threading.Thread(
        target=serve,
        args=(cast(Any, SimpleNamespace(signature=signature)),),
        kwargs={"host": "127.0.0.1", "port": port, "on_event": record},
        daemon=True,
    ).start()
    assert listening.wait(timeout=5), "the server never bound its port"

    with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
        channel = FrameChannel.from_socket(client)
        channel.send(FrameType.GET_SIGNATURE, {})
        assert channel.recv() is not None
        channel.send(FrameType.BYE, {})
        assert spoke_closed.wait(timeout=5), "the server never read the second frame"

    assert any(event.startswith("benchmark connected from") for event in events)


def test_serve_pairs_with_a_benchmark_publishing_a_camera_it_has_no_name_for():
    """A benchmark on a newer SDK may publish a camera this one does not list; a
    policy that does not read it still gets READY."""
    from manifold.core import Benchmark, Camera, PolicySignature
    from manifold.embodiments import FRANKA_EE_DELTA
    from manifold.recipes import serve
    from manifold.wire import BRIDGE_PROTOCOL_VERSION
    from manifold.wire.bridge import FrameChannel, FrameType

    signature = PolicySignature(action_space=FRANKA_EE_DELTA.action, instruction=False)
    advertised = Benchmark(
        name="suite",
        embodiment=FRANKA_EE_DELTA,
        sensors=[Camera(name=CameraName.AGENTVIEW, shape=(8, 8, 3))],
        instruction=False,
    ).model_dump(mode="json")
    advertised["sensors"].append({**advertised["sensors"][0], "name": "chest"})
    endpoint = SimpleNamespace(
        signature=signature, session=lambda: SimpleNamespace(close=lambda: None)
    )
    port = _free_port()
    listening = threading.Event()
    events: list[str] = []

    def record(event: str) -> None:
        events.append(event)
        if event.startswith("listening on"):
            listening.set()

    threading.Thread(
        target=serve,
        args=(cast(Any, endpoint),),
        kwargs={"host": "127.0.0.1", "port": port, "on_event": record},
        daemon=True,
    ).start()
    assert listening.wait(timeout=5), "the server never bound its port"

    with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
        channel = FrameChannel.from_socket(client)
        channel.send(
            FrameType.HELLO,
            {"protocol_version": BRIDGE_PROTOCOL_VERSION, "benchmark": advertised},
        )
        reply = channel.recv()
        channel.send(FrameType.BYE, {})

    assert reply is not None and reply.get("type") == FrameType.READY, events
    assert "ignoring cameras this SDK does not list: chest" in events


def test_serve_sends_its_protocol_version_in_ready():
    from manifold.core import Benchmark, PolicySignature
    from manifold.embodiments import FRANKA_EE_DELTA
    from manifold.recipes import serve
    from manifold.wire import BRIDGE_PROTOCOL_VERSION
    from manifold.wire.bridge import FrameChannel, FrameType

    signature = PolicySignature(action_space=FRANKA_EE_DELTA.action, instruction=False)
    benchmark = Benchmark(name="suite", embodiment=FRANKA_EE_DELTA, instruction=False)
    endpoint = SimpleNamespace(
        signature=signature, session=lambda: SimpleNamespace(close=lambda: None)
    )
    port = _free_port()
    listening = threading.Event()

    def record(event: str) -> None:
        if event.startswith("listening on"):
            listening.set()

    threading.Thread(
        target=serve,
        args=(cast(Any, endpoint),),
        kwargs={"host": "127.0.0.1", "port": port, "on_event": record},
        daemon=True,
    ).start()
    assert listening.wait(timeout=5), "the server never bound its port"

    with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
        channel = FrameChannel.from_socket(client)
        channel.send(
            FrameType.HELLO,
            {
                "protocol_version": BRIDGE_PROTOCOL_VERSION,
                "benchmark": benchmark.model_dump(mode="json"),
            },
        )
        reply = channel.recv()
        channel.send(FrameType.BYE, {})

    assert reply is not None and reply.get("type") == FrameType.READY
    assert reply["payload"] == {"protocol_version": BRIDGE_PROTOCOL_VERSION}


# --- evaluate's episode records --------------------------------------------------
#
# `evaluate` measures each episode, so these run it with a policy that answers every
# observation with a zero action, and check the records that it returns.


def _evaluate(*, step, episodes, max_steps, instruction=None):
    """Run `evaluate` against a policy that answers each observation with zeros."""
    from manifold.core import PolicySignature
    from manifold.core.benchmark import Benchmark
    from manifold.core.values import Action, Observation
    from manifold.embodiments import FRANKA_EE_DELTA
    from manifold.recipes import evaluate

    width = FRANKA_EE_DELTA.action.expected_length()
    session = SimpleNamespace(
        reset=lambda: None,
        infer=lambda _observation: Action.from_array([0.0] * width),
        close=lambda: None,
    )
    endpoint = SimpleNamespace(
        signature=PolicySignature(action_space=FRANKA_EE_DELTA.action, instruction=False),
        session=lambda: session,
    )
    benchmark = Benchmark(
        name="bench-1", embodiment=FRANKA_EE_DELTA, instruction=instruction is not None
    )
    return evaluate(
        cast(Any, endpoint),
        benchmark,
        lambda: ResetResult(
            Observation(),
            instruction=instruction,
            task_id="task",
            attributes={},
            events=(),
            max_steps=None,
        ),
        step,
        episodes=episodes,
        max_steps=max_steps,
        on_event=lambda _m: None,
    )


def test_evaluate_records_every_episode_with_its_own_step_count():
    from manifold.core.values import Observation

    # Succeed on the third step of every episode.
    taken = {"steps": 0}

    def step(_action) -> StepResult:
        taken["steps"] += 1
        done = taken["steps"] % 3 == 0
        return StepResult(observation=Observation(), success=done, done=done, events=())

    result = _evaluate(step=step, episodes=2, max_steps=10)

    assert [r.episode_idx for r in result.records] == [0, 1]
    assert [r.steps for r in result.records] == [3, 3]
    assert [r.success for r in result.records] == [True, True]
    assert (result.episodes, result.successes) == (2, 2)


def test_evaluate_records_an_episode_that_exhausts_its_steps_as_a_failure():
    from manifold.core.values import Observation

    def step(_action) -> StepResult:
        return StepResult(observation=Observation(), success=False, done=False, events=())

    result = _evaluate(step=step, episodes=1, max_steps=4)

    record = result.records[0]
    assert (record.success, record.steps) == (False, 4)
    assert result.success_rate == 0.0


def test_an_episode_record_brackets_the_rollout_and_derives_its_duration():
    from manifold.core.values import Observation

    def step(_action) -> StepResult:
        return StepResult(observation=Observation(), success=True, done=True, events=())

    result = _evaluate(step=step, episodes=1, max_steps=2)

    record = result.records[0]
    assert record.started_at <= record.ended_at
    # The duration is a view over the two instants, so the three cannot disagree.
    assert record.elapsed_sec == (record.ended_at - record.started_at).total_seconds()
    assert 0.0 <= record.initialization_sec <= record.elapsed_sec


def test_an_episode_takes_its_task_name_from_the_instruction_it_was_reset_with():
    from manifold.core.values import Observation

    def step(_action) -> StepResult:
        return StepResult(observation=Observation(), success=True, done=True, events=())

    result = _evaluate(step=step, episodes=1, max_steps=2, instruction="pick up the mug")

    assert result.records[0].task_name == "pick up the mug"


def test_an_episode_falls_back_to_the_benchmark_name_when_no_instruction_is_published():
    from manifold.core.values import Observation

    def step(_action) -> StepResult:
        return StepResult(observation=Observation(), success=True, done=True, events=())

    result = _evaluate(step=step, episodes=1, max_steps=2)

    assert result.records[0].task_name == "bench-1"


def test_a_server_refuses_a_benchmark_newer_than_its_protocol(monkeypatch) -> None:
    # A newer benchmark may include fields that this server would drop without an
    # error. The server refuses a newer benchmark before parsing the payload, so
    # the server does not pair against a benchmark that the model has misread. The
    # benchmark below is not a mapping, so the model rejects it. Without the
    # version check, the test fails.
    from manifold.recipes import serving
    from manifold.wire import BRIDGE_PROTOCOL_VERSION, FrameType

    class _NewerHello:
        def recv(self):
            payload = {"protocol_version": BRIDGE_PROTOCOL_VERSION + 1, "benchmark": "new"}
            return {"type": FrameType.HELLO, "payload": payload}

        def send(self, *_args) -> None:
            raise AssertionError("nothing is sent to a refused peer")

    class _Unreachable:
        @property
        def signature(self):
            raise AssertionError("a refused peer never reaches the gate")

    monkeypatch.setattr(
        serving.FrameChannel, "from_socket", staticmethod(lambda _conn: _NewerHello())
    )
    events: list[str] = []
    # The endpoint and socket are never touched: the refusal comes before either.
    closed = serving._serve_session(
        cast(Any, _Unreachable()), None, None, cast(Any, None), "peer", events.append
    )
    assert closed is True
    assert any("newer than this server" in event for event in events)
