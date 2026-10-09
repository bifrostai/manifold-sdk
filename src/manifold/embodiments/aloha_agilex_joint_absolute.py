"""Aloha-AgileX (dual AgileX arm) in absolute joint targets, RoboTwin's benchmark robot.

Both RoboTwin configs, `demo_clean.yml` and `demo_randomized.yml`, set
`embodiment: [aloha-agilex]`.

Each action is 14 floats, in the order `Base_Task.take_action` slices them
(`envs/_base_task.py:1587-1594`):

    left arm joints    actions[:, 0:6]
    left gripper       actions[:, 6]
    right arm joints   actions[:, 7:13]
    right gripper      actions[:, 13]

`get_obs` publishes joint positions in the same layout as `joint_action["vector"]`
(`_base_task.py:485-495`). It publishes each arm's end-effector position, rotation
and gripper value as `endpose` (`_base_task.py:473-484`). SAPIEN orders that
rotation `(w, x, y, z)`, so the runner must reorder it to `RotationFormat.QUATERNION`
before publishing.
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
        ee_pose=EEObservationSpec(
            rotation=RotationFormat.QUATERNION,
            arm_count=2,
            gripper=GripperObservationSpec(dim=1, encoding=GripperFormat.UNSIGNED),
        ),
    ),
)
