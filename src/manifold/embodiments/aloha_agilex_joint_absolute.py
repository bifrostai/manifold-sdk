"""Aloha-AgileX (dual AgileX arm) in absolute joint targets — RoboTwin's benchmark robot.

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

Six joints per arm, per the arm-joint ids `_init_task_env_`'s own docstring lists
(`left_arm_joint_id: [6, 14, 18, 22, 26, 30]`). So `arm_count=2`: twelve joints split
into two arms of six, each ending in its own gripper. One arm of twelve joints and a
single gripper would put the left gripper where the right arm's first joint is.

The targets are absolute, not deltas. `take_action(action, action_type="qpos")` treats
each value as a joint angle and interpolates a path from the arm's *current* joint
state to it, so the commanded value is the angle itself.

Gripper polarity is UNSIGNED — high is open. `together_open_gripper` defaults to
`left_pos=1, right_pos=1` and `together_close_gripper` to `0, 0`
(`_base_task.py:781`, `:791`), and the value is normalised to [0, 1].

Proprioception is the same 14-value layout the action uses: `get_obs` publishes
`joint_action["vector"] = left_jointstate + right_jointstate`, each being six arm
angles followed by that arm's gripper (`_base_task.py:485-495`).

No `ee_pose` is declared, though the sim does report one. Both configs set
`endpose: true`, so the same `get_obs` call also fills `endpose` with each arm's
end-effector pose and normalised gripper value (`_base_task.py:473-484`). It is
left out because the runner gathers only the joint vector and the cameras into
its native layout, and an embodiment must not promise a field the observation
never carries — declaring it is a matter of shipping it first.
"""

from __future__ import annotations

from manifold.core.conventions import GripperFormat
from manifold.core.embodiment import (
    Embodiment,
    JointActionSpace,
    JointObservationSpec,
    Proprioception,
)

ALOHA_AGILEX_JOINT_ABSOLUTE = Embodiment(
    name="aloha_agilex_joint_absolute",
    # 12 joints and 2 grippers, laid out as two arms of (6 joints, 1 gripper) — so
    # the grippers land at index 6 and 13, which is where RoboTwin slices them and
    # where openpi's ALOHA policy expects them.
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
    ),
)
