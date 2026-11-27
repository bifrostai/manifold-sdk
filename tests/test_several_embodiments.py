"""Tests for a benchmark version whose tasks use several embodiments."""

from __future__ import annotations

import socket
import threading
import types
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest
from pydantic import ValidationError

from manifold.benchmark import BenchmarkConfig, ResetResult, StepResult, _client
from manifold.benchmark.__main__ import _render_only_policy_cameras
from manifold.benchmark._client import (
    Client,
    PolicyCameras,
    _ClientPolicyConnection,
    request_policy_cameras,
)
from manifold.benchmark._recorder import _Gripper as _ActionGripper
from manifold.benchmark._recorder import _gripper_pads
from manifold.benchmark._service import make_service
from manifold.benchmark.hooks import _fill_optional_hooks
from manifold.benchmark.simulators.genesis import GenesisSimulator
from manifold.core.benchmark import Benchmark
from manifold.core.conventions import GripperFormat
from manifold.core.names import CameraName
from manifold.core.pipeline import Pipeline
from manifold.core.policy import PolicySignature
from manifold.core.values import Action, Observation
from manifold.embodiments import DUAL_FRANKA_EE_DELTA, FRANKA_EE_DELTA, FRANKA_JOINT_ABSOLUTE
from manifold.recipes import (
    Pairing,
    launch_server,
    multi_pairing_pipeline,
    read_pairings,
    serve,
    serving,
)
from manifold.replay import EpisodeStartedEvent, GraspedEvent, GripperTouchingEvent, read_replay_log
from manifold.sensors import agentview, wrist
from manifold.wire import FrameType
from tests.test_benchmark_runtime import (
    _LOGGER,
    _observation,
    _PadSimulator,
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
        "other_robots": (),
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


# A policy server that pairs one model with each embodiment, each with its own signature.


def _signature(benchmark: Benchmark) -> PolicySignature:
    """A signature written out for `benchmark`'s embodiment, with the cameras it reads."""
    embodiment = benchmark.embodiment
    cameras = [agentview((4, 4, 3))]
    if embodiment.name == "franka_ee_delta":
        cameras.append(wrist((4, 4, 3)))
    return PolicySignature(
        action_space=embodiment.action,
        proprioception=embodiment.proprioception,
        cameras=cameras,
        instruction=True,
    )


def test_a_pairing_checks_its_own_signature_or_else_the_profiles():
    spec = _client_config(1).spec
    ee = spec.for_embodiment("franka_ee_delta")
    profile = cast("Any", types.SimpleNamespace(signature=_signature(spec)))

    own = Pairing(profile, ee, Pipeline(), signature=_signature(ee))
    shared = Pairing(profile, spec, Pipeline())

    assert own.checked_signature == _signature(ee)
    assert shared.checked_signature == _signature(spec)


def test_read_pairings_reads_a_modules_pairings_or_its_one_pairing():
    one = Pairing(cast("Any", "P"), cast("Any", "B"), Pipeline())
    listing = cast(ModuleType, types.SimpleNamespace(__name__="listing", PAIRINGS=[one, one]))
    single = cast(
        ModuleType,
        types.SimpleNamespace(__name__="single", PROFILE="P", BENCHMARK="B", PIPELINE="X"),
    )
    empty = cast(ModuleType, types.SimpleNamespace(__name__="empty", PAIRINGS=[]))

    assert read_pairings(listing) == [one, one]
    assert [pairing.pipeline for pairing in read_pairings(single)] == ["X"]
    with pytest.raises(ValueError, match=r"empty\.PAIRINGS must be a non-empty list"):
        read_pairings(empty)


def test_the_matchmaker_returns_the_pipeline_and_signature_of_each_embodiment():
    spec = _client_config(1).spec
    arm, ee = (spec.for_embodiment(name) for name in spec.embodiment_names)
    arm_pipeline, ee_pipeline = Pipeline(), Pipeline()

    match = multi_pairing_pipeline(
        [(arm, arm_pipeline, _signature(arm)), (ee, ee_pipeline, _signature(ee))]
    )

    assert match(ee) == (ee_pipeline, _signature(ee))
    assert match(arm)[0] is arm_pipeline


def test_the_signature_reply_leaves_out_an_embodiment_paired_with_two_signatures():
    spec = _client_config(1).spec
    ee = spec.for_embodiment("franka_ee_delta")
    profile = cast("Any", types.SimpleNamespace(signature=_signature(spec)))
    pairings = [
        Pairing(profile, spec.for_embodiment("franka_joint_absolute"), Pipeline()),
        Pairing(profile, ee, Pipeline(), signature=_signature(ee)),
        Pairing(profile, ee.model_copy(update={"name": "other"}), Pipeline()),
    ]

    assert serving._signatures_by_embodiment(pairings) == {
        "franka_joint_absolute": _signature(spec)
    }


class _SlicingSession:
    """A session whose model answers with zeros as wide as the action of the observation's robot.

    The joint-position robot reports `joint_pos` and takes 8 values. The end-effector
    robot reports only `ee_pose` and takes 7.
    """

    def reset(self) -> None:
        pass

    def infer(self, observation: Observation) -> Action:
        width = 8 if "joint_pos" in observation.state else 7
        return Action(values=np.zeros(width, dtype=np.float32))

    def close(self) -> None:
        pass


def _launch(pairings: list[Pairing]) -> int:
    """Serve `pairings` on a free port in a background thread, and return the port."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    listening = threading.Event()
    threading.Thread(
        target=launch_server,
        args=(pairings,),
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "on_event": lambda event: event.startswith("listening on") and listening.set(),
        },
        daemon=True,
    ).start()
    assert listening.wait(timeout=5), "the server never bound its port"
    return port


def _pairings(spec: Benchmark, names: tuple[str, ...]) -> list[Pairing]:
    """A pairing for each of `names`, each with its own signature, over one loaded model."""
    endpoint = types.SimpleNamespace(signature=_signature(spec), session=_SlicingSession)
    profile = cast(
        "Any",
        types.SimpleNamespace(
            default_weights="weights",
            signature=_signature(spec),
            load=lambda weights, device: endpoint,
        ),
    )
    return [
        Pairing(
            profile,
            spec.for_embodiment(name),
            Pipeline(),
            signature=_signature(spec.for_embodiment(name)),
        )
        for name in names
    ]


def _end_to_end_hooks() -> types.SimpleNamespace:
    """Hooks whose even episodes use the joint-position robot and succeed at the second step."""

    def observation(embodiment: str) -> Observation:
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        if embodiment == "franka_joint_absolute":
            return Observation(
                sensors={"agentview": image},
                state={
                    "joint_pos": np.zeros(8, dtype=np.float32),
                    "ee_pose": np.zeros(6, dtype=np.float32),
                },
            )
        return Observation(
            sensors={"agentview": image, "wrist": image},
            state={"ee_pose": np.zeros(9, dtype=np.float32)},
        )

    def on_reset(config, env, task_config, episode_idx, logger):
        env.embodiment = _by_parity(episode_idx)
        env.steps = 0
        return ResetResult(
            observation(env.embodiment),
            instruction="lift the toy",
            task_id=task_config["task_id"],
            attributes={},
            events=(),
            max_steps=None,
            embodiment=env.embodiment,
        )

    def on_step(config, env, task_config, action, logger):
        env.steps += 1
        env.widths = [*getattr(env, "widths", []), len(action.values)]
        done = env.steps == 2
        return StepResult(observation(env.embodiment), success=done, done=done, events=())

    envs: list[types.SimpleNamespace] = []
    hooks = types.SimpleNamespace(
        on_load=lambda config, task_config, logger: (
            envs.append(types.SimpleNamespace()) or envs[-1]
        ),
        on_reset=on_reset,
        on_step=on_step,
        envs=envs,
    )
    _fill_optional_hooks(hooks)
    return hooks


def test_a_two_embodiment_benchmark_runs_against_a_two_pairing_server(tmp_path):
    config = _client_config(1)
    port = _launch(_pairings(config.spec, config.spec.embodiment_names))
    hooks = _end_to_end_hooks()
    session = _Session(_task(tmp_path, (0, 1)))
    service = make_service(config, hooks)

    cameras = request_policy_cameras(f"127.0.0.1:{port}")
    service.start()
    try:
        Client(config, service, cast("Any", session), f"ws://127.0.0.1:{port}", None).run()
    finally:
        service.stop()

    assert cameras is not None
    assert cameras.by_embodiment == {
        "franka_joint_absolute": frozenset({"agentview"}),
        "franka_ee_delta": frozenset({"agentview", "wrist"}),
    }
    ((records, _),) = session.completed
    assert [(record.success, record.attributes["embodiment"]) for record in records] == [
        (True, "franka_joint_absolute"),
        (True, "franka_ee_delta"),
    ]
    (env,) = hooks.envs
    assert env.widths == [8, 8, 7, 7]


def test_a_server_without_a_pairing_for_one_embodiment_fails_the_run_at_start_up(tmp_path):
    config = _client_config(1)
    port = _launch(_pairings(config.spec, ("franka_joint_absolute",)))
    hooks = _end_to_end_hooks()
    session = _Session(_task(tmp_path, (0, 1)))
    service = make_service(config, hooks)

    service.start()
    try:
        with pytest.raises(RuntimeError, match="rejected the pairing with franka_ee_delta"):
            Client(config, service, cast("Any", session), f"127.0.0.1:{port}", None).run()
    finally:
        service.stop()

    assert hooks.envs == []
    assert session.completed == []


# Each gripper of a robot with two arms grasps with its own finger pads.


def _two_grippers() -> tuple[_ActionGripper, ...]:
    return (
        _ActionGripper("0", 6, GripperFormat.SIGNED),
        _ActionGripper("1", 13, GripperFormat.SIGNED),
    )


def test_a_pair_of_pads_for_each_gripper_belongs_to_that_gripper():
    pads = _gripper_pads(("l0", "r0", "l1", "r1"), _two_grippers())

    assert pads == {"0": ("l0", "r0"), "1": ("l1", "r1")}


def test_one_pair_of_pads_serves_every_gripper():
    assert _gripper_pads(("l", "r"), _two_grippers()) == {"0": ("l", "r"), "1": ("l", "r")}
    assert _gripper_pads((), _two_grippers()) == {}


def test_pairs_of_pads_that_do_not_match_the_grippers_are_refused():
    with pytest.raises(ValueError, match="3 pairs of finger pads for 2 grippers"):
        _gripper_pads(("a", "b", "c", "d", "e", "f"), _two_grippers())


def test_the_config_refuses_an_odd_number_of_finger_pads():
    raw = _raw_config({"type": "single"})
    raw["replay"]["finger_pads"] = _PADS[:1]

    with pytest.raises(ValidationError, match="two pads for each gripper, and lists 1"):
        BenchmarkConfig.model_validate(raw)


class _TwoArmPadSimulator(_PadSimulator):
    """A simulator whose two grippers each report their own pair of pads."""

    def replay_finger_pads(self):
        return ("left_pad", "right_pad", "left_pad_1", "right_pad_1")


class _ShutTwoArms(_Policy):
    """Tells both grippers to close at every step."""

    def infer(self, observation):
        return Action(values=np.full(14, -1.0, dtype=np.float32))


def test_each_gripper_grasps_only_what_both_of_its_own_pads_touch(tmp_path, monkeypatch):
    def on_reset(config, env, task_config, episode_idx, logger):
        return ResetResult(
            _observation(0),
            instruction="lift the toy",
            task_id=task_config["task_id"],
            attributes={},
            events=(),
            max_steps=None,
        )

    def on_step(config, env, task_config, action, logger):
        env.steps = getattr(env, "steps", 0) + 1
        touching = tuple(
            GripperTouchingEvent(gripper=gripper, object="toy", is_active=True)
            for gripper in ("0", "1")
        )
        done = env.steps == 3
        return StepResult(_observation(0), success=done, done=done, events=touching)

    hooks = types.SimpleNamespace(
        on_load=lambda config, task_config, logger: types.SimpleNamespace(),
        on_reset=on_reset,
        on_step=on_step,
    )
    raw = _raw_config({"type": "single"})
    raw["spec"]["embodiment"] = DUAL_FRANKA_EE_DELTA.model_dump(mode="json")
    simulator = _TwoArmPadSimulator()
    # Only the second gripper's pads pinch the toy.
    simulator.pad_contacts = [frozenset({("left_pad_1", "toy"), ("right_pad_1", "toy")})] * 3
    session = _Session(_task(tmp_path, (0,)))

    _run_client(
        BenchmarkConfig.model_validate(raw),
        hooks,
        session,
        monkeypatch,
        _ShutTwoArms,
        simulator=simulator,
    )

    ((_, artifacts),) = session.completed
    grasps = [
        (event.gripper, event.is_active)
        for step in read_replay_log(artifacts[0][0]).steps
        for event in step.events
        if isinstance(event, GraspedEvent)
    ]
    assert grasps == [("1", True)]
