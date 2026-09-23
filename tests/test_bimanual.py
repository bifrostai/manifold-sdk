"""Tests for `arm_count`: how many arms a step carries, and where each one sits.

A single-arm robot has one arm, and every spec defaulted to that. A bimanual robot
has two, so the count is both a width and a layout: the value is `arm_count` arms
back to back, each ending in its gripper when `gripper` is set.

Arms are declared; grippers are not. There is one gripper on each arm when `gripper`
is set and none otherwise, so a gripper count is always derived, never a second field
that could disagree with the first.

The default of 1 is what keeps every existing declaration meaning what it did, so it is
pinned here rather than assumed.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from manifold.core import (
    EEActionSpace,
    GripperFormat,
    JointActionSpace,
    RotationFormat,
)
from manifold.core.conventions import ee_step_layout
from manifold.core.embodiment.proprioception import (
    EEObservationSpec,
    GripperObservationSpec,
    JointObservationSpec,
)

_Q = RotationFormat.QUATERNION


# --------------------------------------------------------------- joint space


def test_joint_default_is_one_arm() -> None:
    # The layout every existing embodiment declares: dof angles, then one gripper.
    space = JointActionSpace(dof=7, gripper=GripperFormat.BINARY_OPEN_LOW)
    assert space.arm_count == 1
    assert space.per_step_length() == 8


def test_joint_bimanual_is_two_arms() -> None:
    # ALOHA: [left joints (6), left gripper, right joints (6), right gripper].
    space = JointActionSpace(dof=12, gripper=GripperFormat.UNSIGNED, arm_count=2)
    assert space.per_arm_length() == 7
    assert space.per_step_length() == 14


def test_joint_observation_mirrors_the_action() -> None:
    # A robot reports what it commands; a 14-float action against a 13-float
    # observation would be an embodiment that cannot describe itself.
    spec = JointObservationSpec(dof=12, gripper=GripperFormat.UNSIGNED, arm_count=2)
    assert spec.expected_length() == 14


def test_joint_dof_must_divide_across_arms() -> None:
    # 13 joints cannot be two equal arms. Caught at declaration because a short arm
    # would put the second gripper one slot off on every step of every episode.
    with pytest.raises(ValidationError, match="does not divide into 2 equal arms"):
        JointActionSpace(dof=13, gripper=GripperFormat.UNSIGNED, arm_count=2)


def test_joint_observation_dof_must_divide_across_arms() -> None:
    # The observation spec is declared separately from the action one, so the action's
    # check does not cover it: a benchmark may publish proprioception it takes no
    # action in, and that reading would divide unevenly with nothing objecting.
    with pytest.raises(ValidationError, match="does not divide into 2 equal arms"):
        JointObservationSpec(dof=13, gripper=GripperFormat.UNSIGNED, arm_count=2)


def test_joint_arms_split_the_dof_even_without_a_gripper() -> None:
    # `arm_count` says where the arms split, which does not depend on a gripper, so
    # the rule holds without one. Were the count a gripper count it would be inert
    # here; it is an arm count, so it is not.
    with pytest.raises(ValidationError, match="does not divide into 2 equal arms"):
        JointActionSpace(dof=13, arm_count=2)


def test_joint_arms_add_no_width_without_a_gripper() -> None:
    # `dof` is already the total across arms, so arms alone add no floats; only a
    # gripper on each does.
    assert JointActionSpace(dof=12, arm_count=2).per_step_length() == 12


def test_joint_chunking_multiplies_the_whole_step() -> None:
    space = JointActionSpace(dof=12, gripper=GripperFormat.UNSIGNED, arm_count=2, chunk_size=4)
    assert space.expected_length() == 56


# ------------------------------------------------------------------ ee space


def test_ee_default_is_one_arm() -> None:
    space = EEActionSpace(rotation=_Q, gripper=GripperFormat.SIGNED)
    assert space.arm_count == 1
    assert space.per_step_length() == 8  # 3 + 4 + 1


def test_ee_bimanual_repeats_the_pose_per_arm() -> None:
    # Each arm carries its own pose, so the whole group repeats -- not just the gripper.
    space = EEActionSpace(rotation=_Q, gripper=GripperFormat.UNSIGNED, arm_count=2)
    assert space.per_arm_length() == 8
    assert space.per_step_length() == 16  # 2 x (3 + 4 + 1)


def test_ee_observation_keeps_dim_as_fingers_per_gripper() -> None:
    # `dim` counts fingers of ONE gripper; `arm_count` counts arms, each with one
    # gripper. Collapsing them would make two two-fingered hands indistinguishable
    # from one four-fingered hand.
    jaw = GripperObservationSpec(dim=2)
    assert EEObservationSpec(rotation=_Q, gripper=jaw).expected_length() == 9
    assert EEObservationSpec(rotation=_Q, gripper=jaw, arm_count=2).expected_length() == 18


def test_ee_arms_repeat_the_pose_without_a_gripper() -> None:
    # Unlike the joint case, a gripperless EE spec still doubles: every arm has a pose
    # of its own, and it is the pose -- not the gripper -- that repeats.
    assert EEObservationSpec(rotation=_Q, arm_count=2).expected_length() == 14


# -------------------------------------------------------- one arm's worth


def test_ee_step_layout_is_one_arm() -> None:
    # Every term is per arm, so the tuple sums to one arm's width; multiplying by the
    # arm count is the spec's job. A tuple mixing per-arm and per-robot terms could not
    # be summed at all.
    pos, rot, grip = ee_step_layout(_Q, GripperFormat.UNSIGNED)
    assert (pos, rot, grip) == (3, 4, 1)
    space = EEActionSpace(rotation=_Q, gripper=GripperFormat.UNSIGNED, arm_count=2)
    assert pos + rot + grip == space.per_arm_length()


@pytest.mark.parametrize("arm_count", [1, 2, 3])
def test_every_step_is_whole_arms(arm_count: int) -> None:
    # One arm's width times the arms is a step's width, on every spec -- so a slicer
    # that strides by `per_arm_length()` lands on every arm and nowhere between.
    jaw = GripperObservationSpec(dim=2)
    actions = [
        EEActionSpace(rotation=_Q, gripper=GripperFormat.SIGNED, arm_count=arm_count),
        JointActionSpace(dof=6 * arm_count, gripper=GripperFormat.UNSIGNED, arm_count=arm_count),
    ]
    observations = [
        EEObservationSpec(rotation=_Q, gripper=jaw, arm_count=arm_count),
        JointObservationSpec(
            dof=6 * arm_count, gripper=GripperFormat.UNSIGNED, arm_count=arm_count
        ),
    ]
    for spec in actions:
        assert spec.per_arm_length() * arm_count == spec.per_step_length(), spec
    for spec in observations:
        assert spec.per_arm_length() * arm_count == spec.expected_length(), spec


# ------------------------------------------------------------- the catalogue


def test_no_pre_existing_embodiment_changes_width() -> None:
    # The whole point of defaulting to 1: this change must be invisible to everything
    # already declared. Asserted per entry rather than over the whole catalogue, so
    # adding an embodiment later does not fail a test about not changing old ones.
    from manifold.embodiments import ALL

    widths = {embodiment.name: embodiment.action.expected_length() for embodiment in ALL}
    for name, expected in {
        "droid_joint_absolute": 8,
        "franka_ee_delta": 7,
        "franka_joint_absolute": 8,
        "panda_omron_whole_body": 12,
        "widowx_ee_delta": 7,
    }.items():
        assert widths[name] == expected, name


# ------------------------------------------------------------ drafting a spec


def test_a_lerobot_draft_is_told_to_choose_the_arm_count() -> None:
    # A checkpoint records its action width, never how many arms it spans, so a 14-dim
    # ALOHA checkpoint would otherwise draft as one fourteen-joint arm without a word.
    from manifold.recipes.lerobot import _ALWAYS_UNDETERMINED

    assert "arm_count" in {field.name for field in _ALWAYS_UNDETERMINED}
