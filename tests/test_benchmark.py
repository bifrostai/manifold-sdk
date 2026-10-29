"""Tests for `Benchmark.gripper_format` and `Benchmark.action_range`.

The action space is a union (EE / joint / unified / composite); `gripper_format` must read
the gripper convention out of whichever variant is in play, and raise when the
variant has no gripper at all. `action_range` is optional, has one entry for each value
of the action, and does not take part in pairing.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from manifold.core import (
    ActionRange,
    ActionSpace,
    Benchmark,
    Compatibility,
    EEActionSpace,
    Embodiment,
    GripperFormat,
    JointActionSpace,
    PolicySignature,
    Proprioception,
    RotationFormat,
    UnifiedActionSpace,
    check_compatibility,
)
from manifold.core.names import EmbodimentName
from tests._benchmarks import TABLETOP

# The range of robosuite's OSC controller for an axis-angle action with a gripper.
_OSC_RANGE = {
    "low": [-1.0] * 7,
    "high": [1.0] * 7,
    "scale": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0],
}


def _benchmark(action: ActionSpace) -> Benchmark:
    """A minimal benchmark wrapping the given action space."""
    embodiment = Embodiment(
        name=EmbodimentName.FRANKA_EE_DELTA, action=action, proprioception=Proprioception()
    )
    return Benchmark(name="suite", embodiment=embodiment, instruction=False)


def test_ee_action_with_a_gripper_reports_its_format():
    action = EEActionSpace(rotation=RotationFormat.AXIS_ANGLE, gripper=GripperFormat.SIGNED)
    assert _benchmark(action).gripper_format is GripperFormat.SIGNED


def test_ee_action_without_a_gripper_raises():
    action = EEActionSpace(rotation=RotationFormat.AXIS_ANGLE)
    with pytest.raises(TypeError):
        _ = _benchmark(action).gripper_format


def test_joint_action_with_a_gripper_reports_its_format():
    action = JointActionSpace(dof=7, gripper=GripperFormat.UNSIGNED)
    assert _benchmark(action).gripper_format is GripperFormat.UNSIGNED


def test_unified_action_reports_its_ee_payloads_gripper():
    payload = EEActionSpace(rotation=RotationFormat.AXIS_ANGLE, gripper=GripperFormat.BINARY)
    action = UnifiedActionSpace(width=12, payload=payload)
    assert _benchmark(action).gripper_format is GripperFormat.BINARY


def test_unified_action_with_no_payload_raises():
    action = UnifiedActionSpace(width=12)
    with pytest.raises(TypeError):
        _ = _benchmark(action).gripper_format


def _tabletop_with_range(action_range: object) -> Benchmark:
    return Benchmark.model_validate(
        {**TABLETOP.model_dump(mode="json"), "action_range": action_range}
    )


def test_a_benchmark_declares_no_action_range_by_default():
    assert TABLETOP.action_range is None
    assert TABLETOP.model_dump(mode="json")["action_range"] is None


def test_an_action_range_round_trips_through_json():
    benchmark = _tabletop_with_range(_OSC_RANGE)
    assert benchmark.action_range == ActionRange(
        low=(-1.0,) * 7, high=(1.0,) * 7, scale=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 1.0)
    )
    assert Benchmark.model_validate_json(benchmark.model_dump_json()) == benchmark


def test_an_action_range_without_a_scale_is_in_metres_and_radians():
    benchmark = _tabletop_with_range({"low": [-1.0] * 7, "high": [1.0] * 7})
    assert benchmark.action_range is not None
    assert benchmark.action_range.scale is None


@pytest.mark.parametrize(
    ("action_range", "message"),
    [
        ({"low": [-1.0] * 6, "high": [1.0] * 6}, "action_range has 6 values and the action has 7"),
        ({"low": [-1.0] * 7, "high": [1.0] * 6}, "low has 7 values and high has 6"),
        ({**_OSC_RANGE, "scale": [1.0]}, "scale has 1 values and low has 7"),
        ({"low": [1.0] * 7, "high": [-1.0] * 7}, "low is above high"),
    ],
)
def test_an_action_range_must_fit_the_action(action_range, message):
    with pytest.raises(ValidationError, match=message):
        _tabletop_with_range(action_range)


def test_pairing_does_not_compare_the_action_range():
    benchmark = _tabletop_with_range(_OSC_RANGE)
    policy = PolicySignature(
        action_space=benchmark.action_space,
        proprioception=benchmark.observation_space.proprioception,
        cameras=list(benchmark.observation_space.cameras),
        instruction=benchmark.observation_space.instruction,
    )
    assert check_compatibility(policy, benchmark).status is Compatibility.COMPATIBLE


# The architectures of PyTorch 2.7 built for CUDA 12.8.
_TORCH_ARCHITECTURES = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]


def _tabletop_with_cuda_architectures(cuda_architectures: object) -> Benchmark:
    return Benchmark.model_validate(
        {**TABLETOP.model_dump(mode="json"), "cuda_architectures": cuda_architectures}
    )


def test_a_benchmark_declares_no_cuda_architectures_by_default():
    assert TABLETOP.cuda_architectures is None
    assert TABLETOP.model_dump(mode="json")["cuda_architectures"] is None


def test_cuda_architectures_round_trip_through_json():
    benchmark = _tabletop_with_cuda_architectures(_TORCH_ARCHITECTURES)
    assert benchmark.cuda_architectures == tuple(_TORCH_ARCHITECTURES)
    assert Benchmark.model_validate_json(benchmark.model_dump_json()) == benchmark


@pytest.mark.parametrize(
    "cuda_architectures", [[], ["8.6"], ["sm86"], ["ptx_120"], ["sm_86", "sm_86"]]
)
def test_cuda_architectures_must_be_a_nonempty_list_of_unique_architecture_names(
    cuda_architectures,
):
    with pytest.raises(ValidationError):
        _tabletop_with_cuda_architectures(cuda_architectures)


def test_pairing_does_not_compare_the_cuda_architectures():
    benchmark = _tabletop_with_cuda_architectures(["sm_90"])
    policy = PolicySignature(
        action_space=benchmark.action_space,
        proprioception=benchmark.observation_space.proprioception,
        cameras=list(benchmark.observation_space.cameras),
        instruction=benchmark.observation_space.instruction,
    )
    assert check_compatibility(policy, benchmark).status is Compatibility.COMPATIBLE
