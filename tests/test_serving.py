"""Unit tests for the open-loop chunk-queue Session base in recipes.serving.

OpenLoopChunkQueue holds one raw model chunk and serves it `exec_steps` times before
running a fresh forward — the reusable half of the serve loop a policy backend
subclasses. These exercise that horizon logic against a fake endpoint (no model, no
torch).
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

from manifold.recipes import OpenLoopChunkQueue, StepResult


class _CountingQueue(OpenLoopChunkQueue):
    """A queue whose `_forward` returns a fresh sentinel and counts its calls."""

    def __init__(self, exec_steps: int) -> None:
        super().__init__(SimpleNamespace(profile=SimpleNamespace(exec_steps=exec_steps)))
        self.forwards = 0

    def _forward(self, native, /):
        self.forwards += 1
        return f"chunk{self.forwards}"


def test_advance_forwards_on_drain_then_serves_buffered_steps():
    q = _CountingQueue(exec_steps=3)
    # First advance: empty buffer -> forward, step 0.
    assert q.advance({}) == ("chunk1", 0)
    # Next two: serve the buffered chunk at steps 1, 2 — no new forward.
    assert q.advance({}) == ("chunk1", 1)
    assert q.advance({}) == ("chunk1", 2)
    assert q.forwards == 1
    # exec_steps=3 served -> horizon drained -> fresh forward, step resets to 0.
    assert q.advance({}) == ("chunk2", 0)
    assert q.forwards == 2


def test_reset_clears_buffer_so_next_advance_forwards():
    q = _CountingQueue(exec_steps=5)
    q.advance({})  # forward -> chunk1, step 0
    q.advance({})  # buffered -> chunk1, step 1
    assert q.forwards == 1
    q.reset()
    # Buffer cleared: the next advance forwards again from step 0 mid-horizon.
    assert q.advance({}) == ("chunk2", 0)
    assert q.forwards == 2


def test_base_forward_raises_so_a_backend_cannot_silently_serve_stale():
    q = OpenLoopChunkQueue(SimpleNamespace(profile=SimpleNamespace(exec_steps=2)))
    with pytest.raises(NotImplementedError):
        q.advance({})


def test_infer_is_the_unused_no_packing_path():
    q = _CountingQueue(exec_steps=2)
    with pytest.raises(NotImplementedError):
        q.infer(object())


def test_close_is_a_noop_for_the_stateless_base():
    q = _CountingQueue(exec_steps=2)
    assert q.close() is None


def _records(episodes: int, *, successes: int):
    """`episodes` records, of which the first `successes` succeeded."""
    from manifold.recipes import EpisodeRecord

    moment = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return tuple(
        EpisodeRecord(
            episode_idx=idx,
            task_name="task",
            success=idx < successes,
            steps=1,
            initialization_sec=0.0,
            started_at=moment,
            ended_at=moment,
        )
        for idx in range(episodes)
    )


# --- BenchmarkResult.format_shard -------------------------------------------------


def test_format_shard_renders_the_per_shard_tally():
    from manifold.recipes import BenchmarkResult

    assert (
        BenchmarkResult(records=_records(4, successes=3)).format_shard(0, 2)
        == "[bench shard 0/2] 3/4 succeeded (75.0%)"
    )


def test_format_shard_handles_an_empty_run():
    from manifold.recipes import BenchmarkResult

    assert BenchmarkResult().format_shard(1, 1) == "[bench shard 1/1] 0/0 succeeded (0.0%)"


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

    launch_server([_pairing(profile)], port=9000)

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


# --- run_episodes and run_sharded_benchmark ---------------------------------------
#
# The dispatch wrappers own the run-mechanics (server parse, the ids, the stamping) so
# a runner doesn't. These drive the real loop over the scripted transport further
# down, since the ids reach the records through it.


def _step(_action) -> StepResult:
    return cast(StepResult, None)  # opaque: the id validation refuses before any step


def test_run_episodes_drives_the_ids_it_is_given_in_order(monkeypatch):
    from manifold.core.values import Observation
    from manifold.recipes import run_episodes

    sock = _install_fake_transport(monkeypatch)
    seeded: list[int] = []

    def reset_episode(episode_id: int) -> Observation:
        seeded.append(episode_id)
        return Observation()

    events: list[str] = []
    result = run_episodes(
        _fake_benchmark(),
        reset_episode,
        _one_step_episodes(succeeding=1),
        server="host.example:9000",
        episode_ids=[7, 2, 5],  # arbitrary: not a stride, and not sorted
        max_steps=50,
        on_event=events.append,
    )

    assert sock.address == ("host.example", 9000)
    assert seeded == [7, 2, 5]
    assert [r.episode_idx for r in result.records] == [7, 2, 5]
    # The shard tally belongs to the shard wrapper; a bare id list has no shard to
    # name, so this emits only what one connection's run does.
    assert not any(event.startswith("[bench shard") for event in events)


def test_run_episodes_rejects_ids_that_cannot_identify_an_episode():
    from manifold.core.benchmark import Benchmark
    from manifold.core.values import Observation
    from manifold.recipes import run_episodes

    def dispatch(episode_ids: list[int]) -> None:
        run_episodes(
            cast(Benchmark, object()),
            lambda _id: cast(Observation, object()),
            _step,
            server="h:1",
            episode_ids=episode_ids,
            max_steps=1,
        )

    with pytest.raises(ValueError):
        dispatch([0, 1, 0])  # a repeat would report one episode_idx twice
    with pytest.raises(ValueError):
        dispatch([-1])


def test_run_sharded_benchmark_parses_server_seeds_ids_and_emits_the_tally(monkeypatch):
    from manifold.core.values import Observation
    from manifold.recipes import run_sharded_benchmark

    sock = _install_fake_transport(monkeypatch)
    seeded: list[int] = []

    def reset_episode(episode_id: int) -> Observation:
        seeded.append(episode_id)
        return Observation()

    events: list[str] = []
    result = run_sharded_benchmark(
        _fake_benchmark(),
        reset_episode,
        _one_step_episodes(succeeding=2),
        server="host.example:9000",
        total_episodes=10,
        num_shards=2,
        shard_index=1,
        max_steps=50,
        on_event=events.append,
    )

    assert sock.address == ("host.example", 9000)
    assert seeded == [1, 3, 5, 7, 9]  # this shard's disjoint global ids, in order
    assert result.successes == 2
    assert events[-1] == "[bench shard 1/2] 2/5 succeeded (40.0%)"
    # Each record carries the id its episode ran, not this shard's position in its own
    # list: numbering by position would have every shard report episode 0.
    assert [r.episode_idx for r in result.records] == [1, 3, 5, 7, 9]


def test_run_sharded_benchmark_single_shard_runs_every_global_episode(monkeypatch):
    from manifold.core.values import Observation
    from manifold.recipes import run_sharded_benchmark

    _install_fake_transport(monkeypatch)
    seeded: list[int] = []

    def reset_episode(episode_id: int) -> Observation:
        seeded.append(episode_id)
        return Observation()

    run_sharded_benchmark(
        _fake_benchmark(),
        reset_episode,
        _one_step_episodes(succeeding=0),
        server="h:1",
        total_episodes=3,
        max_steps=1,
    )

    assert seeded == [0, 1, 2]  # unsharded -> the whole global set, in order


def test_run_sharded_benchmark_rejects_a_malformed_server():
    from manifold.core.benchmark import Benchmark
    from manifold.core.values import Observation
    from manifold.recipes import run_sharded_benchmark

    with pytest.raises(ValueError):  # parsed before any connection is attempted
        run_sharded_benchmark(
            cast(Benchmark, object()),
            lambda _id: cast(Observation, object()),
            _step,
            server="no-port",
            total_episodes=1,
            max_steps=1,
        )


def test_shard_episode_ids_rejects_bad_flags_as_a_catchable_valueerror():
    from manifold.recipes import shard_episode_ids

    with pytest.raises(ValueError):
        shard_episode_ids(10, 0, 0)  # num_shards < 1
    with pytest.raises(ValueError):
        shard_episode_ids(10, 2, 2)  # shard_index out of [0, num_shards)


# --- run_benchmark's episode records ----------------------------------------------
#
# The run loop is what measures an episode, so these drive it over a fake transport
# (no socket, no policy) and assert on the records it produces.


class _ScriptedTransport:
    """Answers the handshake with READY, then every observation with one action."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, frame_type, payload) -> None:
        self.sent.append(str(frame_type))

    def recv(self):
        from manifold.core.values import Action
        from manifold.wire import FrameType, bridge

        if self.sent[-1] == "hello":
            return {"type": FrameType.READY, "payload": {}}
        return {
            "type": FrameType.ACTION,
            "payload": bridge.encode_action(Action.from_array([0.0])),
        }


def _run_over_fake_transport(monkeypatch, *, step, episodes, max_steps, instruction=None, **kw):
    """Drive `run_benchmark` against a scripted transport, returning its result."""
    from manifold.core.values import Observation
    from manifold.recipes import run_benchmark

    _install_fake_transport(monkeypatch)
    return run_benchmark(
        _fake_benchmark(instruction=instruction),
        lambda: Observation(instruction=instruction),
        step,
        episodes=episodes,
        max_steps=max_steps,
        port=9000,
        on_event=lambda _m: None,
        **kw,
    )


def _install_fake_transport(monkeypatch) -> _FakeSocket:
    """Answer the dial-out with a scripted transport; return the socket it dialled on.

    Every run path in this module ends at one connection, so patching here drives the
    real episode loop - the numbering, the tally and the recorder included - with no
    socket and no policy behind it.
    """
    from manifold.recipes import serving

    sock = _FakeSocket()
    monkeypatch.setattr(serving.socket, "socket", lambda *a, **k: sock)
    monkeypatch.setattr(
        serving.FrameChannel, "from_socket", staticmethod(lambda _sock: _ScriptedTransport())
    )
    return sock


def _fake_benchmark(*, instruction: str | None = None):
    """The benchmark advertised in HELLO; only its name and instruction flag are read."""
    from manifold.core.benchmark import Benchmark
    from manifold.embodiments import FRANKA_EE

    return Benchmark(name="bench-1", embodiment=FRANKA_EE, instruction=instruction is not None)


def _one_step_episodes(*, succeeding: int):
    """A step ending every episode on its first, of which the first `succeeding` succeed."""
    from manifold.core.values import Observation
    from manifold.recipes import StepResult

    taken = {"steps": 0}

    def step(_action) -> StepResult:
        taken["steps"] += 1
        return StepResult(
            observation=Observation(), success=taken["steps"] <= succeeding, done=True
        )

    return step


class _FakeSocket:
    """A context-manager stand-in for the socket a run dials out on."""

    def __init__(self) -> None:
        self.address: tuple[str, int] | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def connect(self, address) -> None:
        self.address = address


def test_run_benchmark_records_every_episode_with_its_own_step_count(monkeypatch):
    from manifold.core.values import Observation
    from manifold.recipes import StepResult

    # Succeed on the third step of every episode.
    taken = {"steps": 0}

    def step(_action) -> StepResult:
        taken["steps"] += 1
        done = taken["steps"] % 3 == 0
        return StepResult(observation=Observation(), success=done, done=done)

    result = _run_over_fake_transport(monkeypatch, step=step, episodes=2, max_steps=10)

    assert [r.episode_idx for r in result.records] == [0, 1]
    assert [r.steps for r in result.records] == [3, 3]
    assert [r.success for r in result.records] == [True, True]
    assert (result.episodes, result.successes) == (2, 2)


def test_run_benchmark_records_an_episode_that_exhausts_its_steps_as_a_failure(monkeypatch):
    from manifold.core.values import Observation
    from manifold.recipes import StepResult

    def step(_action) -> StepResult:
        return StepResult(observation=Observation(), success=False, done=False)

    result = _run_over_fake_transport(monkeypatch, step=step, episodes=1, max_steps=4)

    record = result.records[0]
    assert (record.success, record.steps) == (False, 4)
    assert result.success_rate == 0.0


def test_an_episode_record_brackets_the_rollout_and_derives_its_duration(monkeypatch):
    from manifold.core.values import Observation
    from manifold.recipes import StepResult

    def step(_action) -> StepResult:
        return StepResult(observation=Observation(), success=True, done=True)

    result = _run_over_fake_transport(monkeypatch, step=step, episodes=1, max_steps=2)

    record = result.records[0]
    assert record.started_at <= record.ended_at
    # The duration is a view over the two instants, so the three cannot disagree.
    assert record.elapsed_sec == (record.ended_at - record.started_at).total_seconds()
    assert 0.0 <= record.initialization_sec <= record.elapsed_sec


def test_an_episode_takes_its_task_name_from_the_instruction_it_was_reset_with(monkeypatch):
    from manifold.core.values import Observation
    from manifold.recipes import StepResult

    def step(_action) -> StepResult:
        return StepResult(observation=Observation(), success=True, done=True)

    result = _run_over_fake_transport(
        monkeypatch, step=step, episodes=1, max_steps=2, instruction="pick up the mug"
    )

    assert result.records[0].task_name == "pick up the mug"


def test_an_episode_falls_back_to_the_benchmark_name_when_no_instruction_is_published(
    monkeypatch,
):
    from manifold.core.values import Observation
    from manifold.recipes import StepResult

    def step(_action) -> StepResult:
        return StepResult(observation=Observation(), success=True, done=True)

    result = _run_over_fake_transport(monkeypatch, step=step, episodes=1, max_steps=2)

    assert result.records[0].task_name == "bench-1"


# --- the episode recorder ----------------------------------------------------------
#
# The loop owns the episode lifecycle, so these assert on what it tells a recorder:
# which id began, which step was the last, and that an episode is ended once whatever
# happened inside it.


class _FakeRecorder:
    """Notes every lifecycle call the loop makes, in order."""

    def __init__(self) -> None:
        self.begun: list[int] = []
        self.steps: list[tuple[bool, bool]] = []
        self.ended = 0

    def begin(self, episode_id: int) -> None:
        self.begun.append(episode_id)

    def record(self, action, *, success: bool, done: bool) -> None:
        self.steps.append((success, done))

    def end(self) -> None:
        self.ended += 1


def test_a_recorder_is_begun_with_the_global_episode_id_of_each_episode(monkeypatch):
    from manifold.core.values import Observation
    from manifold.recipes import run_sharded_benchmark

    _install_fake_transport(monkeypatch)
    recorder = _FakeRecorder()

    run_sharded_benchmark(
        _fake_benchmark(),
        lambda _id: Observation(),
        _one_step_episodes(succeeding=0),
        server="h:1",
        total_episodes=10,
        num_shards=2,
        shard_index=1,
        max_steps=4,
        recorder=recorder,
    )

    # A shard writing a file per episode collides with its sibling unless it is told
    # the global id rather than its own position.
    assert recorder.begun == [1, 3, 5, 7, 9]
    assert recorder.ended == 5


def test_a_recorder_is_told_the_last_step_when_the_step_budget_runs_out(monkeypatch):
    from manifold.core.values import Observation
    from manifold.recipes import StepResult

    def step(_action) -> StepResult:
        return StepResult(observation=Observation(), success=False, done=False)

    recorder = _FakeRecorder()
    _run_over_fake_transport(monkeypatch, step=step, episodes=1, max_steps=3, recorder=recorder)

    # The driver never reports done, so the budget is what ends this episode - and the
    # recorder is told, which it could not work out for itself.
    assert recorder.steps == [(False, False), (False, False), (False, True)]
    assert recorder.ended == 1


def test_a_recorder_is_told_the_last_step_when_the_driver_reports_done(monkeypatch):
    from manifold.core.values import Observation
    from manifold.recipes import StepResult

    taken = {"steps": 0}

    def step(_action) -> StepResult:
        taken["steps"] += 1
        finished = taken["steps"] == 2
        return StepResult(observation=Observation(), success=finished, done=finished)

    recorder = _FakeRecorder()
    _run_over_fake_transport(monkeypatch, step=step, episodes=1, max_steps=9, recorder=recorder)

    assert recorder.steps == [(False, False), (True, True)]


def test_a_recorder_is_ended_when_a_step_raises(monkeypatch):
    def step(_action) -> StepResult:
        raise RuntimeError("the sim fell over")

    recorder = _FakeRecorder()
    with pytest.raises(RuntimeError):
        _run_over_fake_transport(monkeypatch, step=step, episodes=1, max_steps=2, recorder=recorder)

    # Whatever the recorder opened is closed, which is why the benchmark no longer
    # needs a `finally` of its own.
    assert (recorder.begun, recorder.ended) == ([0], 1)


def test_a_recorder_is_ended_when_its_own_begin_raises(monkeypatch):
    from manifold.core.values import Observation
    from manifold.recipes import StepResult

    class _FailingRecorder(_FakeRecorder):
        def begin(self, episode_id: int) -> None:
            super().begin(episode_id)
            raise RuntimeError("the log would not open")

    recorder = _FailingRecorder()
    with pytest.raises(RuntimeError):
        _run_over_fake_transport(
            monkeypatch,
            step=lambda _a: StepResult(observation=Observation(), success=True, done=True),
            episodes=1,
            max_steps=2,
            recorder=recorder,
        )

    # `begin` is where a recorder opens things, so a `begin` that raises part-way is
    # the path with most to release - it has to be inside the `finally`, not before it.
    assert recorder.ended == 1


# --- write_rollup ------------------------------------------------------------------


def test_write_rollup_leaves_the_records_where_a_runner_scans(tmp_path):
    import json

    from manifold.recipes import BenchmarkResult, EpisodeRecord, write_rollup

    record = EpisodeRecord(
        episode_idx=3,
        task_name="pick up the mug",
        success=True,
        steps=12,
        initialization_sec=3.0,
        started_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 1, 1, 12, 0, 30, tzinfo=timezone.utc),
    )

    path = write_rollup(BenchmarkResult(records=(record,)), tmp_path, benchmark_name="libero")

    assert path == tmp_path / "results" / "libero.json"
    assert json.loads(path.read_text()) == {
        "records": [
            {
                "episode_idx": 3,
                "task_name": "pick up the mug",
                "success": True,
                "steps": 12,
                "initialization_sec": 3.0,
                "started_at": "2026-01-01T12:00:00+00:00",
                "ended_at": "2026-01-01T12:00:30+00:00",
            }
        ]
    }


def test_write_rollup_round_trips_its_instants_with_their_timezone(tmp_path):
    import json

    from manifold.recipes import BenchmarkResult, write_rollup

    path = write_rollup(
        BenchmarkResult(records=_records(2, successes=1)), tmp_path, benchmark_name="bench"
    )

    (first, second) = json.loads(path.read_text())["records"]
    assert [first["episode_idx"], second["episode_idx"]] == [0, 1]
    assert datetime.fromisoformat(first["started_at"]).tzinfo is not None
