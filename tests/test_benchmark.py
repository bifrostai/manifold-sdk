"""Tests for `Benchmark.gripper_format`.

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


def _benchmark(action: ActionSpace) -> Benchmark:
    """A minimal benchmark wrapping the given action space."""
    embodiment = Embodiment(name="arm", action=action, proprioception=Proprioception())
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
