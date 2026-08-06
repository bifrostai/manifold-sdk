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


# --- run_sharded_benchmark --------------------------------------------------------
#
# The sharded-run wrapper owns the run-mechanics (server parse + sharding + cursor)
# so a runner doesn't, then delegates to the unchanged run_benchmark. These stub
# run_benchmark to capture what the wrapper forwards and to exhaust the seeding closure.


def _step(_action) -> StepResult:
    return cast(StepResult, None)  # opaque: the wrapper forwards step untouched


def test_run_sharded_benchmark_parses_server_seeds_ids_and_emits_the_tally(monkeypatch):
    from manifold.core.benchmark import Benchmark
    from manifold.core.values import Observation
    from manifold.recipes import BenchmarkResult, run_sharded_benchmark, serving

    captured: dict = {}
    seeded: list[int] = []

    def fake_run_benchmark(benchmark, reset, step, **kw):
        captured.update(**kw)
        for _ in range(kw["episodes"]):  # draining reset() proves the cursor seeding
            reset()
        return BenchmarkResult(records=_records(kw["episodes"], successes=2))

    monkeypatch.setattr(serving, "run_benchmark", fake_run_benchmark)

    def reset_episode(episode_id: int) -> Observation:
        seeded.append(episode_id)
        return cast(Observation, object())

    events: list[str] = []
    result = run_sharded_benchmark(
        cast(Benchmark, object()),
        reset_episode,
        _step,
        server="host.example:9000",
        total_episodes=10,
        num_shards=2,
        shard_index=1,
        max_steps=50,
        on_event=events.append,
    )

    assert (captured["host"], captured["port"]) == ("host.example", 9000)
    assert captured["episodes"] == 5  # shard 1 of 2 over 10 global episodes
    assert seeded == [1, 3, 5, 7, 9]  # this shard's disjoint global ids, in order
    assert result.successes == 2
    assert events[-1] == "[bench shard 1/2] 2/5 succeeded (40.0%)"
    # run_benchmark numbered these 0..4; the wrapper restamps the global ids, without
    # which every shard reports episode 0 and two shards collide on one index.
    assert [r.episode_idx for r in result.records] == [1, 3, 5, 7, 9]


def test_run_sharded_benchmark_single_shard_walks_a_monotone_counter(monkeypatch):
    from manifold.core.benchmark import Benchmark
    from manifold.core.values import Observation
    from manifold.recipes import BenchmarkResult, run_sharded_benchmark, serving

    seeded: list[int] = []

    def fake_run_benchmark(benchmark, reset, step, **kw):
        for _ in range(kw["episodes"]):
            reset()
        return BenchmarkResult(records=_records(kw["episodes"], successes=0))

    monkeypatch.setattr(serving, "run_benchmark", fake_run_benchmark)

    def reset_episode(episode_id: int) -> Observation:
        seeded.append(episode_id)
        return cast(Observation, object())

    run_sharded_benchmark(
        cast(Benchmark, object()),
        reset_episode,
        _step,
        server="h:1",
        total_episodes=3,
        max_steps=1,
    )

    assert seeded == [0, 1, 2]  # unsharded -> monotone 0,1,2 over all total_episodes


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


def _run_over_fake_transport(monkeypatch, *, step, episodes, max_steps, instruction=None):
    """Drive `run_benchmark` against a scripted transport, returning its result."""
    from manifold.core.benchmark import Benchmark
    from manifold.core.values import Observation
    from manifold.embodiments import FRANKA_EE
    from manifold.recipes import run_benchmark, serving

    monkeypatch.setattr(serving.socket, "socket", lambda *a, **k: _FakeSocket())
    monkeypatch.setattr(
        serving.FrameChannel, "from_socket", staticmethod(lambda _sock: _ScriptedTransport())
    )
    return run_benchmark(
        Benchmark(name="bench-1", embodiment=FRANKA_EE, instruction=instruction is not None),
        lambda: Observation(instruction=instruction),
        step,
        episodes=episodes,
        max_steps=max_steps,
        port=9000,
        on_event=lambda _m: None,
    )


class _FakeSocket:
    """A context-manager stand-in for the socket `run_benchmark` dials out on."""

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def connect(self, _address) -> None:
        return None


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


# --- write_rollup ------------------------------------------------------------------


def test_write_rollup_leaves_the_records_where_a_runner_scans(tmp_path):
    import json

    from manifold.recipes import BenchmarkResult, EpisodeRecord, write_rollup

    record = EpisodeRecord(
        episode_idx=3,
        task_name="pick up the mug",
        success=True,
        steps=12,
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
