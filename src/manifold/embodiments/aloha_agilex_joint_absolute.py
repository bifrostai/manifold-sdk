"""Aloha-AgileX (dual AgileX arm) in absolute joint targets, RoboTwin's benchmark robot.

RoboTwin 2.0's default embodiment: `env_cfg/task_config/demo_clean.yml` and
`demo_randomized.yml` both set `embodiment: [aloha-agilex]`, and every score on the
public leaderboard is measured on it. The five embodiments RoboTwin ships
(`_embodiment_config.yml`: aloha-agilex, piper, franka-panda, ARX-X5, ur5-wsg) are
selectable per config, but only this one is registered here.

One step carries 14 floats, in the order `Base_Task.take_action` slices them
(`envs/_base_task.py:1587-1594`):

    left arm joints    actions[:, 0:6]
    left gripper       actions[:, 6]
    right arm joints   actions[:, 7:13]
    right gripper      actions[:, 13]

Each arm has six joints, per the joint ids that the docstring of `_init_task_env_`
lists (`left_arm_joint_id: [6, 14, 18, 22, 26, 30]`). So `arm_count=2`: twelve joints
split into two arms of six, and each arm ends in its own gripper. A single arm of
twelve joints with one gripper would place the left gripper at the index of the right
arm's first joint.

The targets are absolute joint angles. `take_action(action, action_type="qpos")` reads
each value as a joint angle. It then interpolates a path from the arm's current joint
state to that angle, so the commanded value is the angle itself.

Gripper polarity is UNSIGNED, so a high value means open. `together_open_gripper`
defaults to
`left_pos=1, right_pos=1` and `together_close_gripper` to `0, 0`
(`_base_task.py:781`, `:791`), and the value is normalised to [0, 1].

Proprioception is the same 14-value layout the action uses: `get_obs` publishes
`joint_action["vector"] = left_jointstate + right_jointstate`, each being six arm
angles followed by that arm's gripper (`_base_task.py:485-495`).

`ee_pose` is declared beside it. Both configs set `endpose: true`, so the same
`get_obs` call fills `endpose` with each arm's end-effector pose and gripper
(`_base_task.py:473-484`): three floats of position, four of rotation, and one
gripper value, per arm.

RoboTwin reports that rotation as SAPIEN's `(w, x, y, z)`; `RotationFormat.QUATERNION`
is `(qx, qy, qz, qw)`, so the runner reorders it before publishing. The gripper value
is `get_left_gripper_val`, a single scalar normalised to [0, 1] with high meaning
open, not finger qpos, which is why it declares `encoding` rather than claiming
`dim` finger positions.

Declaring both costs a policy nothing: `Proprioception.first_unmet` compares only a
component the policy itself declares, so one consuming joints alone still pairs.
"""

from __future__ import annotations

from manifold.core.conventions import GripperFormat, RotationFormat
from manifold.core.embodiment import (
    EEObservationSpec,
    Embodiment,
    GripperObservationSpec,
    JointActionSpace,
    JointObservationSpec,
    Proprioception,
)

ALOHA_AGILEX_JOINT_ABSOLUTE = Embodiment(
    name="aloha_agilex_joint_absolute",
    # 12 joints and 2 grippers, laid out as two arms of 6 joints and 1 gripper. The
    # grippers therefore sit at index 6 and 13, which is where RoboTwin reads them,
    # and where openpi's ALOHA policy reads them.
    action=JointActionSpace(
        dof=12,
        gripper=GripperFormat.UNSIGNED,
        arm_count=2,
        delta=False,
    ),
    proprioception=Proprioception(
        joint_pos=JointObservationSpec(
            dof=12,
            gripper=GripperFormat.UNSIGNED,
            arm_count=2,
        ),
        # 8 floats an arm: 3 position, 4 rotation, 1 gripper. 16 for both.
        ee_pose=EEObservationSpec(
            rotation=RotationFormat.QUATERNION,
            arm_count=2,
            gripper=GripperObservationSpec(dim=1, encoding=GripperFormat.UNSIGNED),
        ),
    ),
)
