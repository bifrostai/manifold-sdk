"""ROBOTWIN's contract: two arms of six joints, each ending in its gripper."""

from __future__ import annotations

from manifold.benchmarks.robotwin import ROBOTWIN
from manifold.core import JointActionSpace, Pipeline, PolicySignature, verify
from manifold.embodiments.aloha_agilex_joint_absolute import ALOHA_AGILEX_JOINT_ABSOLUTE


def _action() -> JointActionSpace:
    action = ALOHA_AGILEX_JOINT_ABSOLUTE.action
    assert isinstance(action, JointActionSpace)
    return action


def test_aloha_is_two_arms_of_six_joints_and_a_gripper() -> None:
    action = _action()
    assert action.arm_count == 2
    assert action.per_arm_length() == 7
    assert action.expected_length() == 14


def test_the_grippers_sit_where_robotwin_and_openpi_read_them() -> None:
    action = _action()
    arm = action.per_arm_length()
    assert [n * arm + arm - 1 for n in range(action.arm_count)] == [6, 13]


def test_the_robot_reports_the_layout_it_is_commanded_in() -> None:
    joint_pos = ALOHA_AGILEX_JOINT_ABSOLUTE.proprioception.joint_pos
    assert joint_pos is not None
    assert joint_pos.arm_count == _action().arm_count
    assert joint_pos.expected_length() == _action().expected_length()


def test_verify_exercises_both_arms_of_a_robotwin_pairing() -> None:
    # This policy declares exactly ROBOTWIN's contract. `verify` must pass and
    # report the `joints`, `gripper` and `joint_pos` checks for arm0 and arm1.
    policy = PolicySignature(
        action_space=ROBOTWIN.embodiment.action,
        proprioception=ROBOTWIN.embodiment.proprioception,
        cameras=list(ROBOTWIN.sensors),
        instruction=ROBOTWIN.instruction,
    )

    report = verify(policy, ROBOTWIN, Pipeline())

    assert report.ok, report.reasons
    names = {check.name for check in report.checks}
    for arm in (0, 1):
        assert f"action.joints.arm{arm}" in names
        assert f"action.gripper.arm{arm}" in names
        assert f"observation.joint_pos.arm{arm}" in names
