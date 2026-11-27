"""Tests for a benchmark version whose tasks use several embodiments."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from manifold.benchmark import BenchmarkConfig
from manifold.core.benchmark import Benchmark
from manifold.embodiments import FRANKA_EE_DELTA, FRANKA_JOINT_ABSOLUTE
from tests.test_benchmark_runtime import _raw_config

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
