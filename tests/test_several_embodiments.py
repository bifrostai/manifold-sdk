"""Tests for a benchmark version whose tasks use several embodiments."""

from __future__ import annotations

import socket
import threading
import types
from typing import Any, cast

import pytest
from pydantic import ValidationError

from manifold.benchmark import BenchmarkConfig, ResetResult, StepResult, _client
from manifold.benchmark.__main__ import _render_only_policy_cameras
from manifold.benchmark._client import (
    PolicyCameras,
    _ClientPolicyConnection,
    request_policy_cameras,
)
from manifold.benchmark._service import make_service
from manifold.benchmark.simulators.genesis import GenesisSimulator
from manifold.core.benchmark import Benchmark
from manifold.core.names import CameraName
from manifold.core.policy import PolicySignature
from manifold.embodiments import FRANKA_EE_DELTA, FRANKA_JOINT_ABSOLUTE
from manifold.recipes import serve
from manifold.replay import EpisodeStartedEvent, read_replay_log
from manifold.sensors import agentview, wrist
from manifold.wire import FrameType
from tests.test_benchmark_runtime import (
    _LOGGER,
    _observation,
    _Policy,
    _raw_config,
    _run_client,
    _Session,
    _task,
)

_GENESIS = {
    "name": "genesis",
    "path": "gym_env.unwrapped",
    "robot": "agent.robot",
    "arm_joints": ["joint1"],
    "gripper_joints": ["finger_joint1", "finger_joint2"],
    "touch_links": {"hand": "gripper"},
}

_PADS = [
    {"body": "left_finger", "offset": [0.0, 0.0, 0.05]},
    {"body": "right_finger", "offset": [0.0, 0.0, 0.05]},
]


def _arm_entry() -> dict[str, Any]:
    """The first embodiment: a Franka in joint positions with an agentview camera."""
    return {
        "embodiment": FRANKA_JOINT_ABSOLUTE.model_dump(mode="json"),
        "sensors": [{"name": "agentview", "shape": [4, 4, 3]}],
        "action_range": {"low": [-1.0] * 8, "high": [1.0] * 8},
        "end_effectors": [{"name": "hand", "end_effector_type": "gripper"}],
    }


def _ee_entry() -> dict[str, Any]:
    """The second embodiment: a Franka in end-effector deltas with a wrist camera too."""
    return {
        "embodiment": FRANKA_EE_DELTA.model_dump(mode="json"),
        "sensors": [
            {"name": "agentview", "shape": [4, 4, 3]},
            {"name": "wrist", "shape": [4, 4, 3]},
        ],
        "action_range": None,
        "end_effectors": None,
    }


def _spec(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """A spec that lists `entries`, with the first entry at the top level."""
    first = entries[0]
    return {"name": "ladder", **first, "instruction": True, "embodiments": entries}


def test_a_spec_of_several_embodiments_gives_the_spec_of_each():
    benchmark = Benchmark.model_validate({**_spec([_arm_entry(), _ee_entry()]), "max_steps": 9})

    ee = benchmark.for_embodiment("franka_ee_delta")

    assert benchmark.embodiment_names == ("franka_joint_absolute", "franka_ee_delta")
    assert ee.embodiment == FRANKA_EE_DELTA
    assert [str(sensor.name) for sensor in ee.sensors] == ["agentview", "wrist"]
    assert (ee.action_range, ee.end_effectors, ee.embodiments) == (None, None, None)
    assert (ee.name, ee.instruction, ee.max_steps) == ("ladder", True, 9)


def test_the_spec_of_the_first_embodiment_is_the_top_level():
    benchmark = Benchmark.model_validate(_spec([_arm_entry(), _ee_entry()]))

    assert benchmark.for_embodiment("franka_joint_absolute") == benchmark.model_copy(
        update={"embodiments": None}
    )


def test_a_spec_of_one_robot_is_its_own_spec_and_serialises_a_null_embodiments():
    benchmark = Benchmark.model_validate(_spec([_arm_entry()]) | {"embodiments": None})

    assert benchmark.for_embodiment("franka_joint_absolute") is benchmark
    assert benchmark.embodiment_names == ("franka_joint_absolute",)
    assert benchmark.model_dump(mode="json")["embodiments"] is None


def test_the_spec_of_an_unlisted_embodiment_is_refused():
    benchmark = Benchmark.model_validate(_spec([_arm_entry(), _ee_entry()]))

    with pytest.raises(ValueError, match="no embodiment 'widowx_ee_delta'"):
        benchmark.for_embodiment("widowx_ee_delta")


def test_two_entries_with_one_embodiment_name_are_refused():
    with pytest.raises(ValidationError, match="appear more than once"):
        Benchmark.model_validate(_spec([_arm_entry(), _ee_entry(), _ee_entry()]))


def test_a_top_level_that_differs_from_the_first_entry_is_refused():
    spec = _spec([_arm_entry(), _ee_entry()])
    spec["sensors"] = _ee_entry()["sensors"]

    with pytest.raises(ValidationError, match="\\['sensors'\\] must equal the first entry"):
        Benchmark.model_validate(spec)


def test_an_entry_whose_action_range_does_not_fit_its_action_is_refused():
    ee = _ee_entry() | {"action_range": {"low": [-1.0] * 8, "high": [1.0] * 8}}

    with pytest.raises(ValidationError, match="action_range has 8 values and the action has 7"):
        Benchmark.model_validate(_spec([_arm_entry(), ee]))


def _several_config(**changes: Any) -> dict[str, Any]:
    """A raw config with a Genesis simulator and a spec of two embodiments."""
    raw = _raw_config({"type": "single"})
    raw["simulator"] = _GENESIS
    raw["replay"] = {"cameras": ["agentview"], "finger_pads": _PADS}
    raw["spec"] = _spec([_arm_entry(), _ee_entry()])
    return raw | changes


def test_the_config_of_an_embodiment_takes_its_finger_pads_and_simulator_values():
    raw = _several_config(
        embodiments={
            "franka_ee_delta": {
                "finger_pads": [],
                "simulator": {"robot": "agents.1.robot", "touch_links": {"link7": "arm"}},
            }
        }
    )
    config = BenchmarkConfig.model_validate(raw)

    ee = config.for_embodiment("franka_ee_delta")
    arm = config.for_embodiment("franka_joint_absolute")

    assert ee.spec == config.spec.for_embodiment("franka_ee_delta")
    assert ee.replay.finger_pads == ()
    assert ee.simulator is not None
    assert ee.simulator.model_dump() == {
        **_GENESIS,
        "arm_joints": ("joint1",),
        "gripper_joints": ("finger_joint1", "finger_joint2"),
        "robot": "agents.1.robot",
        "touch_links": {"hand": "gripper", "link7": "arm"},
    }
    assert ee.embodiments is None
    assert arm.simulator == config.simulator
    assert arm.replay == config.replay


def test_the_config_refuses_values_for_an_embodiment_that_the_spec_does_not_list():
    raw = _several_config(embodiments={"widowx_ee_delta": {"finger_pads": [], "simulator": {}}})

    with pytest.raises(ValidationError, match="lists \\['widowx_ee_delta'\\], which spec"):
        BenchmarkConfig.model_validate(raw)


def test_the_config_refuses_simulator_values_that_break_the_simulator_config():
    raw = _several_config(
        embodiments={"franka_ee_delta": {"finger_pads": [], "simulator": {"robots": "x"}}}
    )

    with pytest.raises(ValidationError, match="robots"):
        BenchmarkConfig.model_validate(raw)


def test_the_config_refuses_values_for_each_embodiment_with_an_isaac_sim_simulator():
    raw = _several_config(embodiments={"franka_ee_delta": {"finger_pads": [], "simulator": {}}})
    raw["simulator"] = {
        "name": "isaacsim",
        "path": "env",
        "app": {},
        "robot": "robot",
        "max_position_iterations": None,
        "env_spacing": None,
        "camera_far_clip": None,
        "touch_links": None,
    }

    with pytest.raises(ValidationError, match="isaacsim simulator reads one robot"):
        BenchmarkConfig.model_validate(raw)


def test_the_replay_cameras_must_be_sensors_of_every_embodiment():
    raw = _several_config()
    raw["replay"] = {"cameras": ["wrist"], "finger_pads": _PADS}
    arm = _arm_entry() | {
        "sensors": [*_arm_entry()["sensors"], {"name": "wrist", "shape": [4, 4, 3]}]
    }
    ee = _ee_entry() | {"sensors": [{"name": "agentview", "shape": [4, 4, 3]}]}
    raw["spec"] = _spec([arm, ee])

    with pytest.raises(ValidationError, match=r"in spec\.sensors: \['wrist'\]"):
        BenchmarkConfig.model_validate(raw)


# The Client, run against a fake task session and fake policy connections.


def _client_config(num_envs: int, **changes: Any) -> BenchmarkConfig:
    """A config without a simulator, whose spec lists the two embodiments."""
    raw = _raw_config({"type": "single"}, num_envs=num_envs)
    raw["spec"] = _spec([_arm_entry(), _ee_entry()])
    return BenchmarkConfig.model_validate(raw | changes)


def _hooks(embodiment_of: Any) -> types.SimpleNamespace:
    """Hooks whose episode `i` uses the embodiment `embodiment_of(i)` and succeeds at once."""

    def on_reset(config, env, task_config, episode_idx, logger):
        return ResetResult(
            _observation(episode_idx),
            instruction="lift the toy",
            task_id=task_config["task_id"],
            attributes={"seed": str(episode_idx)},
            events=(),
            max_steps=None,
            embodiment=embodiment_of(episode_idx),
        )

    def on_step(config, env, task_config, action, logger):
        return StepResult(_observation(0), success=True, done=True, events=())

    return types.SimpleNamespace(
        env_key=lambda config, task_config, logger: None,
        on_load=lambda config, task_config, logger: types.SimpleNamespace(),
        on_reset=on_reset,
        on_step=on_step,
    )


class _NamedPolicy(_Policy):
    """A fake policy connection that keeps the embodiment of its spec and counts its calls."""

    def __init__(self, address: str, benchmark: Benchmark, image_format: str) -> None:
        super().__init__(address, benchmark, image_format)
        self.embodiment = str(benchmark.embodiment.name)
        self.embodiments = benchmark.embodiments
        self.infers = 0
        self.closed = False

    def infer(self, observation):
        self.infers += 1
        return super().infer(observation)

    def close(self) -> None:
        self.closed = True


def _by_parity(episode_idx: int) -> str:
    return "franka_joint_absolute" if episode_idx % 2 == 0 else "franka_ee_delta"


def test_the_client_opens_a_connection_for_each_env_and_embodiment(tmp_path, monkeypatch):
    session = _Session(_task(tmp_path, (0, 1, 2, 3)))

    policies = _run_client(
        _client_config(2), _hooks(_by_parity), session, monkeypatch, _NamedPolicy
    )

    assert [policy.embodiment for policy in policies] == [
        "franka_joint_absolute",
        "franka_ee_delta",
        "franka_joint_absolute",
        "franka_ee_delta",
    ]
    assert all(policy.embodiments is None for policy in policies)
    assert all(policy.closed for policy in policies)


def test_each_episode_runs_on_its_embodiments_connection_and_names_it(tmp_path, monkeypatch):
    session = _Session(_task(tmp_path, (0, 1, 2)))

    policies = _run_client(
        _client_config(1), _hooks(_by_parity), session, monkeypatch, _NamedPolicy
    )

    arm, ee = policies
    assert (arm.resets, arm.infers, ee.resets, ee.infers) == (2, 2, 1, 1)
    ((records, artifacts),) = session.completed
    assert [record.attributes for record in records] == [
        {"seed": "0", "embodiment": "franka_joint_absolute"},
        {"seed": "1", "embodiment": "franka_ee_delta"},
        {"seed": "2", "embodiment": "franka_joint_absolute"},
    ]
    started = read_replay_log(artifacts[1][0]).steps[0].events[0]
    assert isinstance(started, EpisodeStartedEvent)
    assert started.attributes == {"seed": "1", "embodiment": "franka_ee_delta"}


def test_a_spec_of_one_robot_runs_each_episode_on_its_only_embodiment(tmp_path, monkeypatch):
    session = _Session(_task(tmp_path, (0,)))
    raw = _raw_config({"type": "single"})

    policies = _run_client(
        BenchmarkConfig.model_validate(raw), _hooks(lambda _: None), session, monkeypatch
    )

    assert len(policies) == 1
    ((records, _),) = session.completed
    assert records[0].attributes == {"seed": "0", "embodiment": "franka_joint_absolute"}


@pytest.mark.parametrize(
    ("embodiment", "message"),
    [
        (None, "named no embodiment for task 'toy/clean'"),
        ("widowx_ee_delta", "embodiment 'widowx_ee_delta' for task 'toy/clean', which the spec"),
    ],
)
def test_an_episode_without_a_listed_embodiment_fails(tmp_path, monkeypatch, embodiment, message):
    session = _Session(_task(tmp_path, (0,)))

    with pytest.raises(ValueError, match=message):
        _run_client(_client_config(1), _hooks(lambda _: embodiment), session, monkeypatch)

    assert session.completed == []


def test_a_refused_embodiment_fails_the_run_before_any_episode(tmp_path, monkeypatch):
    session = _Session(_task(tmp_path, (0,)))
    opened: list[_NamedPolicy] = []
    resets: list[int] = []
    hooks = _hooks(_by_parity)
    on_reset = hooks.on_reset

    def refusing(address, benchmark, image_format):
        if benchmark.embodiment.name == "franka_ee_delta":
            raise RuntimeError("policy rejected the pairing with franka_ee_delta (no READY)")
        opened.append(_NamedPolicy(address, benchmark, image_format))
        return opened[-1]

    def reset(config, env, task_config, episode_idx, logger):
        resets.append(episode_idx)
        return on_reset(config, env, task_config, episode_idx, logger)

    hooks.on_reset = reset
    with pytest.raises(RuntimeError, match="rejected the pairing with franka_ee_delta"):
        _run_client(_client_config(1), hooks, session, monkeypatch, refusing)

    assert [policy.closed for policy in opened] == [True]
    assert resets == []
    assert session.completed == []


def test_the_replay_of_each_episode_has_its_embodiments_finger_pads(tmp_path, monkeypatch):
    session = _Session(_task(tmp_path, (0, 1)))
    raw = _raw_config({"type": "single"})
    raw["replay"] = {"cameras": ["agentview"], "finger_pads": _PADS}
    config = _client_config(
        1,
        replay=raw["replay"],
        embodiments={"franka_ee_delta": {"finger_pads": [], "simulator": {}}},
    )

    _run_client(config, _hooks(_by_parity), session, monkeypatch, _NamedPolicy)

    ((_, artifacts),) = session.completed
    arm, ee = (read_replay_log(artifacts[index][0]) for index in (0, 1))
    assert [pad.body for pad in arm.finger_pads] == ["left_finger", "right_finger"]
    assert ee.finger_pads == ()


def test_the_live_view_gets_each_episodes_embodiment(tmp_path, monkeypatch):
    session = _Session(_task(tmp_path, (0, 1)))
    published: list[str] = []
    live_view = types.SimpleNamespace(
        publish=lambda observation, *, embodiment, **_: published.append(embodiment)
    )

    _run_client(_client_config(1), _hooks(_by_parity), session, monkeypatch, live_view=live_view)

    assert published == ["franka_joint_absolute"] * 2 + ["franka_ee_delta"] * 2


def test_the_hello_leaves_out_the_embodiments(monkeypatch):
    sent = []
    replies = [{"type": FrameType.READY, "payload": {}}]
    channel = types.SimpleNamespace(
        send=lambda *args: sent.append(args), recv=lambda: replies.pop(0)
    )
    monkeypatch.setattr(
        _client.socket,
        "create_connection",
        lambda *_args: types.SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(_client.FrameChannel, "from_socket", staticmethod(lambda _s: channel))
    spec = _client_config(1).spec

    _ClientPolicyConnection("policy:9000", spec.for_embodiment("franka_ee_delta"), "raw")

    ((_, payload),) = sent
    assert "embodiments" not in payload["benchmark"]
    assert payload["benchmark"]["embodiment"]["name"] == "franka_ee_delta"


def test_the_service_attaches_the_simulator_of_each_embodiment_with_its_own_values(tmp_path):
    raw = _several_config(
        embodiments={
            "franka_ee_delta": {"finger_pads": [], "simulator": {"robot": "agents.1.robot"}}
        }
    )
    service = make_service(BenchmarkConfig.model_validate(raw), _hooks(_by_parity))

    service.load(_task(tmp_path, (0,)))
    service.reset({0: 0}, {0: _LOGGER})

    arm = service.simulator_for("franka_joint_absolute")
    ee = service.simulator_for("franka_ee_delta")
    assert arm is service.simulator
    assert isinstance(ee, GenesisSimulator)
    assert ee is not arm
    assert cast("Any", ee)._config.robot == "agents.1.robot"
    assert cast("Any", ee)._finger_pads == ()
    assert set(cast("Any", ee)._envs) == set(cast("Any", arm)._envs) == {0}


# Each embodiment renders only the cameras that its pairing reads.


def _camera_config() -> BenchmarkConfig:
    """Two embodiments that each publish the agentview, the wrist and a right shoulder."""
    cameras = [
        {"name": "agentview", "shape": [4, 4, 3]},
        {"name": "wrist", "shape": [4, 4, 3]},
        {"name": "over_shoulder_right", "shape": [4, 4, 3]},
    ]
    raw = _raw_config({"type": "single"})
    raw["spec"] = _spec([_arm_entry() | {"sensors": cameras}, _ee_entry() | {"sensors": cameras}])
    return BenchmarkConfig.model_validate(raw)


def _rendered(config: BenchmarkConfig, reply: PolicyCameras | None, monkeypatch) -> list[list[str]]:
    """The cameras that each embodiment renders after the policy sends `reply`."""
    monkeypatch.setattr("manifold.benchmark.__main__.request_policy_cameras", lambda address: reply)
    spec = _render_only_policy_cameras(config, "ws://policy:1").spec
    return [
        [str(sensor.name) for sensor in spec.for_embodiment(name).sensors]
        for name in spec.embodiment_names
    ]


_ALL = ["agentview", "wrist", "over_shoulder_right"]


def test_each_embodiment_renders_the_cameras_of_its_pairing_and_the_replay(monkeypatch):
    reply = PolicyCameras(
        cameras=frozenset({"wrist"}),
        by_embodiment={
            "franka_joint_absolute": frozenset({"agentview"}),
            "franka_ee_delta": frozenset({"wrist", "over_shoulder_right"}),
        },
    )

    assert _rendered(_camera_config(), reply, monkeypatch) == [["agentview"], _ALL]


@pytest.mark.parametrize(
    "by_embodiment",
    [
        None,
        {"franka_joint_absolute": frozenset({"wrist"})},
        {"franka_joint_absolute": frozenset({"wrist"}), "franka_ee_delta": frozenset({"head"})},
    ],
    ids=[
        "a reply without signatures",
        "no pairing for the embodiment",
        "a pairing that reads a camera the embodiment does not publish",
    ],
)
def test_an_embodiment_renders_every_camera(monkeypatch, by_embodiment):
    reply = PolicyCameras(cameras=frozenset({"wrist"}), by_embodiment=by_embodiment)

    rendered = _rendered(_camera_config(), reply, monkeypatch)

    assert rendered[1] == _ALL
    assert rendered[0] == (_ALL if by_embodiment is None else ["agentview", "wrist"])


def test_a_spec_of_one_robot_renders_the_cameras_of_its_pairing_before_the_signature(
    monkeypatch,
):
    raw = _raw_config({"type": "single"})
    raw["spec"] = _spec([_arm_entry() | {"sensors": _camera_config().spec.sensors}]) | {
        "embodiments": None
    }
    config = BenchmarkConfig.model_validate(raw)
    paired = PolicyCameras(
        cameras=frozenset({"wrist"}),
        by_embodiment={"franka_joint_absolute": frozenset({"over_shoulder_right"})},
    )
    unpaired = PolicyCameras(cameras=frozenset({"wrist"}), by_embodiment={})

    assert _rendered(config, paired, monkeypatch) == [["agentview", "over_shoulder_right"]]
    assert _rendered(config, unpaired, monkeypatch) == [["agentview", "wrist"]]


def test_with_embodiment_sensors_keeps_the_top_level_equal_to_the_first_entry():
    spec = _camera_config().spec

    narrowed = spec.with_embodiment_sensors({"franka_joint_absolute": {CameraName.WRIST}})

    assert [str(sensor.name) for sensor in narrowed.sensors] == ["wrist"]
    assert narrowed.for_embodiment("franka_joint_absolute").sensors == narrowed.sensors
    assert narrowed.for_embodiment("franka_ee_delta").sensors == spec.sensors


def test_the_signature_reply_carries_each_embodiments_signature():
    arm = PolicySignature(
        action_space=FRANKA_EE_DELTA.action, cameras=[agentview((8, 8, 3))], instruction=False
    )
    both = arm.model_copy(update={"cameras": [agentview((8, 8, 3)), wrist((8, 8, 3))]})
    endpoint = types.SimpleNamespace(signature=arm, session=None)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    listening = threading.Event()
    threading.Thread(
        target=serve,
        args=(endpoint,),
        kwargs={
            "port": port,
            "signatures": {"franka_joint_absolute": arm, "franka_ee_delta": both},
            "on_event": lambda event: event.startswith("listening on") and listening.set(),
        },
        daemon=True,
    ).start()
    assert listening.wait(timeout=5), "the server never bound its port"

    assert request_policy_cameras(f"127.0.0.1:{port}") == PolicyCameras(
        cameras=frozenset({"agentview"}),
        by_embodiment={
            "franka_joint_absolute": frozenset({"agentview"}),
            "franka_ee_delta": frozenset({"agentview", "wrist"}),
        },
    )
