"""Tests for `Benchmark.gripper_format` and `Benchmark.with_sensors`.

The action space is a union (EE / joint / unified); `gripper_format` must read
the gripper convention out of whichever variant is in play, and raise when the
variant has no gripper at all.
"""

from __future__ import annotations

import pytest

from manifold.core import (
    ActionSpace,
    Benchmark,
    EEActionSpace,
    Embodiment,
    GripperFormat,
    JointActionSpace,
    Proprioception,
    RotationFormat,
    UnifiedActionSpace,
)
from manifold.core.names import CameraName, EmbodimentName


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


def test_with_sensors_keeps_the_named_sensors_in_the_benchmark_order():
    from manifold.core.sensor import Modality
    from manifold.sensors.cameras import agentview, wrist

    depth = agentview((8, 8, 1), modality=Modality.DEPTH)
    benchmark = _benchmark(EEActionSpace(rotation=RotationFormat.AXIS_ANGLE)).model_copy(
        update={"sensors": [agentview((8, 8, 3)), depth, wrist((8, 8, 3))]}
    )

    selected = benchmark.with_sensors({CameraName.WRIST, depth.key})

    assert [sensor.key for sensor in selected.sensors] == ["agentview_depth", CameraName.WRIST]


def test_with_sensors_refuses_a_sensor_the_benchmark_does_not_publish():
    from manifold.sensors.cameras import wrist

    benchmark = _benchmark(EEActionSpace(rotation=RotationFormat.AXIS_ANGLE)).model_copy(
        update={"sensors": [wrist((8, 8, 3))]}
    )

    with pytest.raises(ValueError, match=CameraName.HEAD):
        benchmark.with_sensors({CameraName.WRIST, CameraName.HEAD})
