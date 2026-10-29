from __future__ import annotations

import dataclasses
import json
import logging
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from pydantic import ValidationError

from manifold.benchmark import (
    BenchmarkConfig,
    HooksError,
    ResetResult,
    StepResult,
    WorkerTask,
    _client,
    load_config,
)
from manifold.benchmark.__main__ import main
from manifold.benchmark._client import Client, _ClientPolicyConnection, _upright_image_pipeline
from manifold.benchmark._replay import _ReplayRecorder
from manifold.benchmark._service import _log_unsupported_versions, make_service
from manifold.benchmark.hooks import _check_hooks, _fill_optional_hooks
from manifold.benchmark.simulators import Simulator
from manifold.core.benchmark import Benchmark
from manifold.core.embodiment import JointActionSpace, UnifiedActionSpace
from manifold.core.sensor import Camera, CameraOrientation, ChannelOrder
from manifold.core.values import Action, Observation
from manifold.embodiments import DROID_JOINT_ABSOLUTE, FRANKA_JOINT_ABSOLUTE
from manifold.replay import (
    GripperTouchingEvent,
    NoteEvent,
    OnEvent,
    ReplayMesh,
    ReplayPose,
    ReplaySceneBody,
    ReplayScenePart,
    RerunScalar,
    RerunTab,
    Severity,
    TurnedOnEvent,
    read_replay_log,
    replay_log_path,
)
from manifold.sensors import CameraName
from manifold.wire import FrameType, bridge

if TYPE_CHECKING:
    from manifold.benchmark import WorkerSession
    from manifold.benchmark._live_view import LiveViewPublisher

# A logger to pass where a test calls the Service without the Client.
_LOGGER = logging.LoggerAdapter(logging.getLogger("test"))

_SPEC = {
    "name": "toy",
    "embodiment": FRANKA_JOINT_ABSOLUTE.model_dump(mode="json"),
    "sensors": [{"name": "agentview", "shape": [4, 4, 3]}],
    "instruction": True,
}


def _raw_config(service: dict[str, Any], num_envs: int = 1, max_steps: int | None = 5) -> dict:
    return {
        "benchmark": {"version": "0.0.1", "rubric": "rubric.json"},
        "upstream": {
            "repo": "https://example.com/toy",
            "commit": "abc",
            "patches": [],
            "path": "/opt/upstream",
            "data": "/data/toy",
        },
        "service": service,
        "extras": {},
        "env": {"num_envs": num_envs},
        "run": {"seed": 0, "max_steps": max_steps, "step_seconds": 0.05, "image_format": "jpeg"},
        "simulator": None,
        "replay": {"cameras": ["agentview"]},
        "spec": _SPEC,
    }


def _config(
    service: dict[str, Any], num_envs: int = 1, max_steps: int | None = 5
) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(_raw_config(service, num_envs, max_steps))


def _observation(value: float) -> Observation:
    return Observation(
        sensors={"agentview": np.full((4, 4, 3), int(value) % 255, dtype=np.uint8)},
        state={"joint_pos": np.full(14, value, dtype=np.float32)},
    )


def _task(tmp_path: Path, indices: tuple[int, ...]) -> WorkerTask:
    return WorkerTask(
        item_id="item",
        attempt_id="attempt",
        task_id="toy/clean",
        task_config={"task_id": "toy/clean", "seed": 0, "num_envs": len(indices)},
        episode_indices=indices,
        output_dir=tmp_path,
    )


def _action(value: float) -> Action:
    return Action(values=np.full(14, value, dtype=np.float32))


# Checks of the config file and the hooks file.


def test_a_config_file_loads_into_a_benchmark_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_raw_config({"type": "batched", "reset_together": True}, 2)))
    config = load_config(path)
    assert config.service.type == "batched"
    assert config.env.num_envs == 2


def test_a_config_changes_only_the_values_that_it_lists_in_the_config_that_it_extends(
    tmp_path,
):
    (tmp_path / "config-base.json").write_text(json.dumps(_raw_config({"type": "single"})))
    leaf = {
        "extends": "config-base.json",
        "run": {"max_steps": 9},
        "extras": {"suite": "kitchen"},
    }
    (tmp_path / "config-leaf.json").write_text(json.dumps(leaf))

    config = load_config(tmp_path / "config-leaf.json")

    assert config.run.max_steps == 9
    assert config.run.step_seconds == 0.05
    assert config.extras == {"suite": "kitchen"}
    assert config.spec.name == "toy"


def test_a_null_in_a_config_removes_the_key_from_the_config_that_it_extends(tmp_path):
    base = _raw_config({"type": "single"})
    base["extras"] = {"suite": "kitchen", "tasks": 2}
    (tmp_path / "config-base.json").write_text(json.dumps(base))
    leaf = {"extends": "config-base.json", "extras": {"tasks": None}}
    (tmp_path / "config-leaf.json").write_text(json.dumps(leaf))

    assert load_config(tmp_path / "config-leaf.json").extras == {"suite": "kitchen"}


def test_configs_that_extend_each_other_in_a_loop_are_refused(tmp_path):
    (tmp_path / "config-a.json").write_text(json.dumps({"extends": "config-b.json"}))
    (tmp_path / "config-b.json").write_text(json.dumps({"extends": "config-a.json"}))

    with pytest.raises(ValueError, match="in a loop"):
        load_config(tmp_path / "config-a.json")


def test_a_config_takes_no_key_that_it_does_not_list():
    raw = _raw_config({"type": "single"})
    raw["run"]["episodes"] = 3
    with pytest.raises(ValidationError, match="episodes"):
        BenchmarkConfig.model_validate(raw)


def test_reset_together_is_only_for_a_batched_service():
    with pytest.raises(ValidationError, match="reset_together"):
        _config({"type": "single", "reset_together": True})


def test_replay_cameras_must_be_sensors():
    raw = _raw_config({"type": "single"})
    raw["replay"]["cameras"] = ["wrist"]
    with pytest.raises(ValidationError, match="wrist"):
        BenchmarkConfig.model_validate(raw)


def test_the_spec_is_an_sdk_benchmark():
    benchmark = _config({"type": "single"}).spec
    assert benchmark.name == "toy"
    assert benchmark.embodiment.action.expected_length() == 8


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda spec: spec.pop("instruction"), "lacks \\['instruction'\\]"),
        (lambda spec: spec.pop("sensors"), "lacks \\['sensors'\\]"),
        (lambda spec: spec.update(cameras=[]), "unknown keys \\['cameras'\\]"),
    ],
)
def test_the_spec_must_have_each_key_of_a_benchmark_whose_default_states_a_value(change, message):
    raw = _raw_config({"type": "single"})
    raw["spec"] = dict(raw["spec"])
    change(raw["spec"])
    with pytest.raises(ValidationError, match=message):
        BenchmarkConfig.model_validate(raw)


def test_the_spec_may_leave_out_the_action_range():
    assert "action_range" not in _SPEC
    assert _config({"type": "single"}).spec.action_range is None
    raw = _raw_config({"type": "single"})
    raw["spec"] = {**raw["spec"], "action_range": {"low": [-1.0] * 8, "high": [1.0] * 8}}
    spec = BenchmarkConfig.model_validate(raw).spec
    assert spec.action_range is not None
    assert spec.action_range.high == (1.0,) * 8


def test_the_spec_may_leave_out_the_cuda_architectures():
    assert "cuda_architectures" not in _SPEC
    assert _config({"type": "single"}).spec.cuda_architectures is None
    raw = _raw_config({"type": "single"})
    raw["spec"] = {**raw["spec"], "cuda_architectures": ["sm_86", "compute_86"]}
    spec = BenchmarkConfig.model_validate(raw).spec
    assert spec.cuda_architectures == ("sm_86", "compute_86")


def test_a_hooks_file_must_define_its_services_hooks():
    with pytest.raises(HooksError, match="must define on_reset"):
        _check_hooks(
            types.SimpleNamespace(on_load=lambda config, task_config, logger: None), "single"
        )


def test_a_hook_must_take_its_arguments():
    hooks = types.SimpleNamespace(
        on_load=lambda config, logger: None,
        on_reset=lambda config, env, task_config, episode_idx, logger: None,
        on_step=lambda config, env, task_config, action, logger: None,
    )
    with pytest.raises(HooksError, match="on_load must take"):
        _check_hooks(hooks, "single")


# The two Service types.


def _single_hooks() -> types.SimpleNamespace:
    """An episode that succeeds once the action's first value reaches 3."""

    def on_reset(config, env, task_config, episode_idx, logger):
        env.value = float(episode_idx)
        return ResetResult(
            _observation(env.value),
            instruction="lift the toy",
            task_id=task_config["task_id"],
            attributes={},
            events=(),
            max_steps=None,
        )

    def on_step(config, env, task_config, action, logger):
        env.value += 1
        success = bool(action.values[0] >= 3)
        return StepResult(_observation(env.value), success=success, done=success, events=())

    return types.SimpleNamespace(
        env_key=lambda config, task_config, logger: None,
        on_load=lambda config, task_config, logger: types.SimpleNamespace(),
        on_reset=on_reset,
        on_step=on_step,
    )


def test_a_single_service_moves_an_env_with_its_hooks(tmp_path):
    service = make_service(_config({"type": "single"}), _single_hooks())
    service.load(_task(tmp_path, (0,)))
    reset = service.reset({0: 7}, {0: _LOGGER})
    assert float(reset[0].observation.state["joint_pos"][0]) == 7.0
    assert service.step({0: _action(1)}, {0: _LOGGER})[0].done is False
    ended = service.step({0: _action(3)}, {0: _LOGGER})[0]
    assert ended.success is True
    assert ended.done is True


def test_each_hook_gets_the_config_and_the_tasks_config(tmp_path):
    seen = []
    hooks = _single_hooks()
    on_step = hooks.on_step

    def step(config, env, task_config, action, logger):
        seen.append((config, task_config))
        return on_step(config, env, task_config, action, logger)

    hooks.on_step = step
    config = _config({"type": "single"})
    task = _task(tmp_path, (0,))
    service = make_service(config, hooks)
    service.load(task)
    service.reset({0: 0}, {0: _LOGGER})
    service.step({0: _action(0)}, {0: _LOGGER})
    assert seen == [(config, task.task_config)]


def test_a_batched_service_steps_all_envs_in_one_call(tmp_path):
    calls = []

    def on_step(config, env, task_config, env_ids, actions, loggers):
        calls.append(list(env_ids))
        return [StepResult(_observation(0), success=True, done=True, events=()) for _ in env_ids]

    hooks = types.SimpleNamespace(
        env_key=lambda config, task_config, logger: None,
        on_load=lambda config, task_config, logger: object(),
        on_reset=lambda config, env, env_id, task_config, episode_idx, logger: ResetResult(
            _observation(episode_idx),
            instruction="lift the toy",
            task_id="toy/clean",
            attributes={},
            events=(),
            max_steps=None,
        ),
        on_step=on_step,
    )
    service = make_service(_config({"type": "batched", "reset_together": False}, 2), hooks)
    service.load(_task(tmp_path, (0, 1)))
    service.reset({0: 0, 1: 1}, {0: _LOGGER, 1: _LOGGER})
    service.step({0: _action(0), 1: _action(0)}, {0: _LOGGER, 1: _LOGGER})
    assert calls == [[0, 1]]


# The Client, run against a fake task session and a fake policy.


class _Session:
    def __init__(self, task: WorkerTask) -> None:
        self._task: WorkerTask | None = task
        self.completed: list[tuple[list, dict]] = []

    def claim(self) -> WorkerTask | None:
        task, self._task = self._task, None
        return task

    def complete(self, task, records, artifacts) -> None:
        self.completed.append((list(records), dict(artifacts)))


class _Policy:
    """Stands in for the Client's policy connection: answers with the first value plus 1."""

    def __init__(self, address: str, benchmark: object, image_format: str) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def infer(self, observation: Observation) -> Action:
        return _action(float(observation.state["joint_pos"][0]) + 1)

    def close(self) -> None:
        pass


def _run_client(
    config, hooks, session, monkeypatch, policy=_Policy, simulator=None, live_view=None
) -> list:
    policies: list = []

    def connect(*args, **kwargs):
        policies.append(policy(*args, **kwargs))
        return policies[-1]

    monkeypatch.setattr("manifold.benchmark._client._ClientPolicyConnection", connect)
    # `load_hooks` fills in the optional hooks that a hooks file leaves out.
    _fill_optional_hooks(hooks)
    service = make_service(config, hooks)
    service.simulator = simulator
    # `main` starts the Service before the Client runs and stops it after.
    try:
        service.start()
        Client(
            config,
            service,
            cast("WorkerSession", session),
            "test:0",
            cast("LiveViewPublisher", live_view),
        ).run()
    finally:
        service.stop()
    return policies


def test_the_client_runs_a_tasks_episodes_and_reports_them(tmp_path, monkeypatch):
    config = _config({"type": "single"}, num_envs=2)
    session = _Session(_task(tmp_path, (0, 1, 2)))
    policies = _run_client(config, _single_hooks(), session, monkeypatch)

    ((records, artifacts),) = session.completed
    assert [record.episode_idx for record in records] == [0, 1, 2]
    assert all(record.success for record in records)
    assert sum(policy.resets for policy in policies) == 3
    log = read_replay_log(artifacts[0][0])
    assert len(log.steps) == records[0].steps + 1


def test_the_client_ends_an_episode_at_max_steps(tmp_path, monkeypatch):
    config = _config({"type": "single"}, max_steps=1)
    session = _Session(_task(tmp_path, (0,)))

    class _Stuck(_Policy):
        def infer(self, observation):
            return _action(0)

    _run_client(config, _single_hooks(), session, monkeypatch, policy=_Stuck)
    ((records, _),) = session.completed
    assert (records[0].success, records[0].steps) == (False, 1)


def test_a_tasks_own_step_limit_ends_its_episode(tmp_path, monkeypatch):
    config = _config({"type": "single"}, max_steps=None)
    session = _Session(_task(tmp_path, (0,)))
    hooks = _single_hooks()
    on_reset = hooks.on_reset

    def reset(config, env, task_config, episode_idx, logger):
        return dataclasses.replace(
            on_reset(config, env, task_config, episode_idx, logger), max_steps=2
        )

    hooks.on_reset = reset

    class _Stuck(_Policy):
        def infer(self, observation):
            return _action(0)

    _run_client(config, hooks, session, monkeypatch, policy=_Stuck)
    ((records, _),) = session.completed
    assert (records[0].success, records[0].steps) == (False, 2)


def test_a_tasks_own_step_limit_is_refused_while_the_config_sets_one(tmp_path, monkeypatch):
    config = _config({"type": "single"}, max_steps=5)
    hooks = _single_hooks()
    on_reset = hooks.on_reset

    def reset(config, env, task_config, episode_idx, logger):
        return dataclasses.replace(
            on_reset(config, env, task_config, episode_idx, logger), max_steps=2
        )

    hooks.on_reset = reset

    with pytest.raises(ValueError, match=r"set run\.max_steps to null"):
        _run_client(config, hooks, _Session(_task(tmp_path, (0,))), monkeypatch)


def test_the_service_starts_the_simulator_once_and_the_envs_for_each_task(tmp_path, monkeypatch):
    calls: list[str] = []
    hooks = _single_hooks()
    hooks.on_start = lambda config, logger: calls.append("start")
    on_load = hooks.on_load

    def load(config, task_config, logger):
        calls.append("load")
        return on_load(config, task_config, logger)

    hooks.on_load = load
    hooks.on_unload = lambda config, env, logger: calls.append("unload")
    hooks.on_stop = lambda config, logger: calls.append("stop")

    class _TwoTasks(_Session):
        def __init__(self) -> None:
            super().__init__(_task(tmp_path, (0,)))
            self._tasks = [_task(tmp_path, (0,)), _task(tmp_path, (1,))]

        def claim(self) -> WorkerTask | None:
            return self._tasks.pop(0) if self._tasks else None

    config = _config({"type": "single"})
    _run_client(config, hooks, _TwoTasks(), monkeypatch)
    assert calls == ["start", "load", "unload", "load", "unload", "stop"]


def test_the_service_stops_after_on_start_raises(tmp_path, monkeypatch):
    calls: list[str] = []
    hooks = _single_hooks()

    def start(config, logger):
        raise RuntimeError("on_start failed")

    hooks.on_start = start
    hooks.on_stop = lambda config, logger: calls.append("stop")

    with pytest.raises(RuntimeError, match="on_start failed"):
        _run_client(
            _config({"type": "single"}), hooks, _Session(_task(tmp_path, (0,))), monkeypatch
        )
    assert calls == ["stop"]


def test_the_simulator_stops_after_on_stop_raises(tmp_path, monkeypatch):
    class _StopCounter(_Simulator):
        def __init__(self) -> None:
            super().__init__()
            self.stops = 0

        def stop(self) -> None:
            self.stops += 1

    simulator = _StopCounter()
    hooks = _single_hooks()

    def stop(config, logger):
        raise RuntimeError("on_stop failed")

    hooks.on_stop = stop

    with pytest.raises(RuntimeError, match="on_stop failed"):
        _run_client(
            _config({"type": "single"}),
            hooks,
            _Session(_task(tmp_path, (0,))),
            monkeypatch,
            simulator=simulator,
        )
    assert simulator.stops == 1


def test_the_service_keeps_the_env_while_env_key_is_the_same(tmp_path, monkeypatch):
    calls: list[str] = []
    hooks = _single_hooks()
    on_load = hooks.on_load

    def load(config, task_config, logger):
        calls.append(f"load {task_config['task_id']}")
        return on_load(config, task_config, logger)

    hooks.env_key = lambda config, task_config, logger: task_config["task_id"]
    hooks.on_load = load
    hooks.on_unload = lambda config, env, logger: calls.append("unload")
    hooks.on_stop = lambda config, logger: calls.append("stop")

    class _ThreeTasks(_Session):
        def __init__(self) -> None:
            super().__init__(_task(tmp_path, (0,)))
            self._tasks = [
                dataclasses.replace(
                    _task(tmp_path, (index,)),
                    task_config={"task_id": task_id, "seed": 0, "num_envs": 1},
                )
                for index, task_id in enumerate(("toy/clean", "toy/clean", "toy/stack"))
            ]

        def claim(self) -> WorkerTask | None:
            return self._tasks.pop(0) if self._tasks else None

    _run_client(_config({"type": "single"}), hooks, _ThreeTasks(), monkeypatch)
    assert calls == ["load toy/clean", "unload", "load toy/stack", "unload", "stop"]


def test_the_client_puts_the_reset_instruction_on_each_observation(tmp_path, monkeypatch):
    sent: list[str | None] = []

    class _InstructionPolicy(_Policy):
        def infer(self, observation: Observation) -> Action:
            sent.append(observation.instruction)
            return super().infer(observation)

    session = _Session(_task(tmp_path, (0,)))
    _run_client(
        _config({"type": "single"}), _single_hooks(), session, monkeypatch, _InstructionPolicy
    )
    # The hooks' observations carry no instruction, and the episode takes 3 steps.
    assert sent == ["lift the toy"] * 3


def test_a_message_that_the_hooks_log_is_written_once_in_an_episode(tmp_path, monkeypatch, caplog):
    hooks = _single_hooks()
    on_step = hooks.on_step

    def step(config, env, task_config, action, logger):
        logger.warning("goal predicates unreadable")
        return on_step(config, env, task_config, action, logger)

    hooks.on_step = step
    session = _Session(_task(tmp_path, (0, 1)))
    with caplog.at_level(logging.WARNING, logger="manifold.benchmark.hooks"):
        _run_client(_config({"type": "single"}), hooks, session, monkeypatch)

    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "manifold.benchmark.hooks"
    ] == [
        "env 0 episode 0: goal predicates unreadable",
        "env 0 episode 1: goal predicates unreadable",
    ]
    ((_, artifacts),) = session.completed
    notes = [
        (index, event)
        for index, step in enumerate(read_replay_log(artifacts[0][0]).steps)
        for event in step.events
        if isinstance(event, NoteEvent)
    ]
    assert notes == [(1, NoteEvent(text="goal predicates unreadable", severity=Severity.WARN))]


def test_the_hooks_outside_an_episode_log_with_the_container_or_the_task_in_front(
    tmp_path, monkeypatch, caplog
):
    hooks = _single_hooks()
    on_load = hooks.on_load

    def load(config, task_config, logger):
        logger.info("built the env")
        return on_load(config, task_config, logger)

    hooks.on_start = lambda config, logger: logger.info("started")
    hooks.on_load = load
    with caplog.at_level(logging.INFO, logger="manifold.benchmark.hooks"):
        _run_client(
            _config({"type": "single"}), hooks, _Session(_task(tmp_path, (0,))), monkeypatch
        )

    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "manifold.benchmark.hooks"
    ] == ["started", "task toy/clean: built the env"]


def test_the_client_records_each_fact_when_it_changes(tmp_path, monkeypatch):
    def facts(value):
        return (
            TurnedOnEvent(object="stove", is_active=True),
            OnEvent(object="bowl", reference="plate", is_active=value >= 2),
        )

    def on_reset(config, env, task_config, episode_idx, logger):
        env.value = float(episode_idx)
        return ResetResult(
            _observation(env.value),
            instruction="lift the toy",
            task_id=task_config["task_id"],
            attributes={},
            events=facts(env.value),
            max_steps=None,
        )

    def on_step(config, env, task_config, action, logger):
        env.value += 1
        success = bool(action.values[0] >= 3)
        return StepResult(
            _observation(env.value),
            success=success,
            done=success,
            events=facts(env.value),
        )

    hooks = types.SimpleNamespace(
        on_load=lambda config, task_config, logger: types.SimpleNamespace(),
        on_reset=on_reset,
        on_step=on_step,
    )
    session = _Session(_task(tmp_path, (0,)))
    _run_client(_config({"type": "single"}), hooks, session, monkeypatch)

    ((_, artifacts),) = session.completed
    log = read_replay_log(artifacts[0][0])
    recorded = [
        (index, type(event).__name__, event.is_active)
        for index, step in enumerate(log.steps)
        for event in step.events
        if isinstance(event, (TurnedOnEvent, OnEvent))
    ]
    assert recorded == [(0, "TurnedOnEvent", True), (2, "OnEvent", True)]


class _Gripper(_Policy):
    """Answers with the gripper open, then shut, then open: 1, 0, 1 in a binary gripper."""

    def __init__(self, address: str, benchmark: object, image_format: str) -> None:
        super().__init__(address, benchmark, image_format)
        self._commands = iter([1.0, 0.0, 1.0])

    def infer(self, observation: Observation) -> Action:
        values = np.zeros(8, dtype=np.float32)
        values[:7] = 0.5
        values[7] = next(self._commands)
        return Action(values=values)


def _touching_hooks() -> types.SimpleNamespace:
    """An episode of three steps in which the gripper touches the toy from step 1."""

    def on_reset(config, env, task_config, episode_idx, logger):
        env.value = 0.0
        return ResetResult(
            _observation(env.value),
            instruction="lift the toy",
            task_id=task_config["task_id"],
            attributes={"category": "camera"},
            events=(GripperTouchingEvent(object="toy", is_active=False),),
            max_steps=None,
        )

    def on_step(config, env, task_config, action, logger):
        env.value += 1
        done = env.value >= 3
        notes = (NoteEvent(text="sub-task passed"),) if env.value == 2 else ()
        return StepResult(
            _observation(env.value),
            success=done,
            done=done,
            events=(GripperTouchingEvent(object="toy", is_active=True), *notes),
        )

    return types.SimpleNamespace(
        on_load=lambda config, task_config, logger: types.SimpleNamespace(),
        on_reset=on_reset,
        on_step=on_step,
    )


def test_the_replay_records_the_gripper_commands_and_derives_each_grasp(tmp_path, monkeypatch):
    session = _Session(_task(tmp_path, (0,)))
    _run_client(_config({"type": "single"}), _touching_hooks(), session, monkeypatch, _Gripper)

    ((records, artifacts),) = session.completed
    log = read_replay_log(artifacts[0][0])
    assert records[0].attributes == {"category": "camera"}
    assert [[type(event).__name__ for event in step.events] for step in log.steps] == [
        ["EpisodeStartedEvent"],
        ["GripperTouchingEvent"],
        ["GripperCommandedEvent", "NoteEvent", "GraspedEvent"],
        ["GripperCommandedEvent", "GraspedEvent", "EpisodeEndedEvent"],
    ]
    assert [step.text.get("events") for step in log.steps] == [
        "episode started [category: camera]: lift the toy",
        None,
        "gripper close at step 2 / [INFO] sub-task passed / grasp: toy at step 2",
        "gripper open at step 3 / release at step 3 / episode ended at step 3: SUCCESS",
    ]
    assert [step.scalars["grasp"] for step in log.steps] == [0.0, 0.0, 1.0, 0.0]


def test_the_replay_plots_the_action_the_robot_and_the_task(tmp_path, monkeypatch):
    session = _Session(_task(tmp_path, (0,)))
    _run_client(_config({"type": "single"}), _touching_hooks(), session, monkeypatch, _Gripper)

    ((_, artifacts),) = session.completed
    log = read_replay_log(artifacts[0][0])
    groups = {channel.name: (channel.group, channel.row) for channel in log.channels}
    assert log.overview_group == "Action"
    assert groups["gripper"] == ("Action", 0)
    assert groups["joint_cmd/6"] == ("Action", 1)
    assert groups["track_j0/commanded"] == ("Robot", None)
    assert groups["facts/toy_gripper_touching"] == ("Task", None)
    assert [step.scalars["joint_cmd/0"] for step in log.steps] == [0.0, 0.5, 0.5, 0.5]
    assert [step.scalars["track_j0/observed"] for step in log.steps] == [0.0, 1.0, 2.0, 3.0]
    assert [step.scalars["facts/toy_gripper_touching"] for step in log.steps] == [
        0.0,
        1.0,
        1.0,
        1.0,
    ]
    assert [step.scalars["episode_success"] for step in log.steps] == [0.0, 0.0, 0.0, 1.0]


class _Simulator(Simulator):
    """Stands in for a simulator with one cube that moves along x, one step at a time."""

    def __init__(self) -> None:
        self.attached: list[list[int]] = []
        self._x = 0.0
        # The supported versions that a test sets.
        self.supported: dict[str, str] = {}

    @property
    def supported_versions(self) -> dict[str, str]:
        return self.supported

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def attach(self, env, env_ids) -> None:
        self.attached.append(list(env_ids))

    def replay_bodies(self, env_id):
        return (
            ReplaySceneBody(
                name="cube", parts=(ReplayScenePart(mesh=ReplayMesh.box([0.1, 0.1, 0.1])),)
            ),
        )

    def replay_poses(self, env_id):
        self._x += 1.0
        return {
            "cube": ReplayPose(
                position=np.array([self._x, 0.0, 0.0]), orientation=np.array([0.0, 0.0, 0.0, 1.0])
            )
        }

    def replay_extrinsics(self, env_id):
        return {}

    def replay_images(self, env_id):
        return {"egocentric": np.zeros((2, 2, 3), dtype=np.uint8)}

    def replay_scalars(self, env_id, ee_position):
        return [RerunScalar("joint_torques/0", RerunTab.PHYSICS, self._x, None)]


def test_the_replay_holds_the_simulators_scene_images_and_scalars(tmp_path, monkeypatch):
    simulator = _Simulator()
    session = _Session(_task(tmp_path, (0,)))
    _run_client(
        _config({"type": "single"}), _single_hooks(), session, monkeypatch, simulator=simulator
    )

    ((_, artifacts),) = session.completed
    log = read_replay_log(artifacts[0][0])
    assert simulator.attached == [[0]]
    assert [body.name for body in log.bodies] == ["cube"]
    assert "egocentric" in {channel.name for channel in log.channels}
    assert ("joint_torques/0", "Physics") in {
        (channel.name, channel.group) for channel in log.channels
    }
    assert [step.poses["cube"].position[0] for step in log.steps] == [
        float(index + 1) for index in range(len(log.steps))
    ]


def test_the_client_sends_env_0_to_the_live_view(tmp_path, monkeypatch):
    published: list[tuple[int, int]] = []
    live_view = types.SimpleNamespace(
        publish=lambda observation, *, episode_idx, step, task, task_id: published.append(
            (episode_idx, step)
        )
    )
    session = _Session(_task(tmp_path, (0, 1)))
    _run_client(
        _config({"type": "single"}, num_envs=2),
        _single_hooks(),
        session,
        monkeypatch,
        live_view=live_view,
    )

    assert {episode for episode, _ in published} == {0}
    assert [step for _, step in published] == list(range(len(published)))


def test_an_optional_hook_that_a_hooks_file_leaves_out_does_nothing():
    hooks = types.SimpleNamespace(on_stop=lambda config, logger: "stopped")
    _fill_optional_hooks(hooks)
    assert hooks.on_start(None, None) is None
    assert hooks.env_key(None, None, None) is None
    assert hooks.on_unload(None, None, None) is None
    assert hooks.on_stop(None, None) == "stopped"


def test_check_writes_the_spec_and_the_run_values(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(_raw_config({"type": "single"})))
    (tmp_path / "hooks.py").write_text(
        "def on_load(config, task_config, logger): ...\n"
        "def on_reset(config, env, task_config, episode_idx, logger): ...\n"
        "def on_step(config, env, task_config, action, logger): ...\n"
    )
    monkeypatch.setattr(
        sys, "argv", ["manifold.benchmark", "--check", str(tmp_path / "config.json")]
    )

    main()

    written = json.loads((tmp_path / "spec.json").read_text())
    assert written["run"] == {
        "seed": 0,
        "max_steps": 5,
        "step_seconds": 0.05,
        "image_format": "jpeg",
    }
    # A reader that knows only the spec's keys ignores `run`.
    spec = Benchmark.model_validate(written)
    assert spec.name == "toy"
    assert "run" not in spec.model_dump()


def test_the_client_refuses_an_action_that_the_benchmark_cannot_take(monkeypatch):
    # A policy on an older SDK can misread the benchmark's action width and answer in
    # the width that it expects. The Client checks each action against the
    # benchmark's spec, so the env never steps with an action that it cannot take.
    replies = [
        {"type": FrameType.READY, "payload": {}},
        {"type": FrameType.ACTION, "payload": bridge.encode_action(Action.from_array([0.0]))},
    ]
    channel = types.SimpleNamespace(send=lambda *_args: None, recv=lambda: replies.pop(0))
    monkeypatch.setattr(
        _client.socket,
        "create_connection",
        lambda *_args: types.SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(_client.FrameChannel, "from_socket", staticmethod(lambda _s: channel))
    benchmark = _config({"type": "single"}).spec
    connection = _ClientPolicyConnection("policy:9000", benchmark, "raw")

    with pytest.raises(RuntimeError, match="cannot take"):
        connection.infer(_observation(0.0))


@pytest.mark.parametrize(
    ("orientation", "channel_order"),
    [
        (CameraOrientation.UPRIGHT, ChannelOrder.RGB),
        (CameraOrientation.FLIPPED_VERTICAL, ChannelOrder.RGB),
        (CameraOrientation.FLIPPED_HORIZONTAL, ChannelOrder.RGB),
        (CameraOrientation.ROTATED_180, ChannelOrder.RGB),
        (CameraOrientation.UPRIGHT, ChannelOrder.BGR),
        (CameraOrientation.ROTATED_180, ChannelOrder.BGR),
    ],
)
def test_the_upright_image_pipeline_turns_a_camera_upright_and_into_rgb(orientation, channel_order):
    upright = np.zeros((4, 6, 3), dtype=np.uint8)
    upright[:2, :3, 0] = 255
    upright[:2, 3:, 1] = 255
    upright[2:, :3, 2] = 255
    stored = upright
    if orientation in (CameraOrientation.FLIPPED_VERTICAL, CameraOrientation.ROTATED_180):
        stored = stored[::-1, :, :]
    if orientation in (CameraOrientation.FLIPPED_HORIZONTAL, CameraOrientation.ROTATED_180):
        stored = stored[:, ::-1, :]
    if channel_order is ChannelOrder.BGR:
        stored = stored[:, :, ::-1]
    camera = Camera(
        name=CameraName.AGENTVIEW,
        shape=stored.shape,
        orientation=orientation,
        channel_order=channel_order,
    )
    benchmark = Benchmark(name="bench", embodiment=FRANKA_JOINT_ABSOLUTE, sensors=[camera])
    observation = Observation(sensors={CameraName.AGENTVIEW: np.ascontiguousarray(stored)})

    result = _upright_image_pipeline(benchmark).apply_observation(
        observation, source=benchmark.observation_space
    )

    np.testing.assert_array_equal(result.sensors[CameraName.AGENTVIEW], upright)
    np.testing.assert_array_equal(observation.sensors[CameraName.AGENTVIEW], stored)


def _recorder_for(action_space) -> _ReplayRecorder:
    embodiment = FRANKA_JOINT_ABSOLUTE.model_copy(update={"action": action_space})
    raw = _raw_config({"type": "single"})
    raw["spec"] = {**raw["spec"], "embodiment": embodiment.model_dump(mode="json")}
    return _ReplayRecorder(BenchmarkConfig.model_validate(raw), None, 0)


def test_a_rotation_of_zeros_is_charted_as_a_quaternion_of_zeros():
    raw = _raw_config({"type": "single"})
    raw["spec"] = {**raw["spec"], "embodiment": DROID_JOINT_ABSOLUTE.model_dump(mode="json")}
    recorder = _ReplayRecorder(BenchmarkConfig.model_validate(raw), None, 0)
    observation = Observation(
        sensors={}, state={"ee_pose": np.array([0.1, 0.2, 0.3, 0, 0, 0, 0])}, instruction=""
    )

    ee = recorder._ee_pose(observation)

    assert ee is not None
    position, quaternion = ee
    np.testing.assert_array_equal(position, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(quaternion, np.zeros(4))


def test_the_replay_plots_no_command_for_a_unified_action():
    payload = FRANKA_JOINT_ABSOLUTE.action
    assert isinstance(payload, JointActionSpace)
    recorder = _recorder_for(UnifiedActionSpace(width=32, payload=payload))
    values = np.zeros(32, dtype=np.float32)

    assert recorder._action_scalars(Action(values=values)) == []


def test_a_replay_without_action_plots_names_no_overview_group(tmp_path):
    recorder = _recorder_for(UnifiedActionSpace(width=32, payload=None))

    recorder.begin(tmp_path, 0, _observation(0), {}, ())
    recorder.end()

    assert read_replay_log(replay_log_path(tmp_path, 0)).overview_group is None


def test_a_fact_that_first_appears_after_the_reset_is_recorded_without_a_chart(
    tmp_path, monkeypatch, caplog
):
    hooks = _single_hooks()
    on_step = hooks.on_step

    def step(config, env, task_config, action, logger):
        result = on_step(config, env, task_config, action, logger)
        late = OnEvent(object="toy", reference="tray", is_active=True)
        return dataclasses.replace(result, events=(late,))

    hooks.on_step = step
    session = _Session(_task(tmp_path, (0,)))

    _run_client(_config({"type": "single"}), hooks, session, monkeypatch)

    ((_, artifacts),) = session.completed
    log = read_replay_log(artifacts[0][0])
    assert not any(channel.name.startswith("facts/") for channel in log.channels)
    assert any(isinstance(event, OnEvent) for step in log.steps for event in step.events)
    assert "after the reset" in caplog.text


def test_an_installed_version_outside_the_range_is_logged(caplog):
    simulator = _Simulator()
    simulator.supported = {"numpy": "<1.0", "pydantic": ">=2", "no-such-package": ">=1"}

    with caplog.at_level(logging.WARNING):
        _log_unsupported_versions(simulator)

    assert [record.getMessage().split(" ")[0] for record in caplog.records] == ["numpy"]
