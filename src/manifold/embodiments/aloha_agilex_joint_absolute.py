"""ALOHA AgileX (two 6-joint arms, a parallel gripper on each) in absolute joint targets.

The dual-arm robot RoboTwin 2.0 evaluates on as `aloha-agilex`. An action and the
reported state both follow the ALOHA layout that `arm_count=2` describes: the left
arm's 6 joint angles, the left gripper, the right arm's 6 joint angles, the right
gripper, 14 values in all. RoboTwin's own eval sends that vector to
`take_action(..., action_type="qpos")`, and its observation's `joint_action.vector`
reports it in the same order.

The targets are absolute: the vector is the joint positions the planner drives to,
not increments, so no `Frame` applies to the action.

The gripper value is the opening fraction RoboTwin uses for both the command and the
observation: 1 is fully open and 0 is fully closed, continuous in between, so it is
UNSIGNED on both sides.

No `ee_pose` is declared. RoboTwin reports each arm's end pose with its quaternion in
`(w, x, y, z)` order, and the policies paired with it read joint state only.
"""

from __future__ import annotations

from manifold.core.conventions import GripperFormat
from manifold.core.embodiment import (
    Embodiment,
    JointActionSpace,
    JointObservationSpec,
    Proprioception,
)
from manifold.core.names import EmbodimentName

ALOHA_AGILEX_JOINT_ABSOLUTE = Embodiment(
    name=EmbodimentName.ALOHA_AGILEX_JOINT_ABSOLUTE,
    action=JointActionSpace(
        dof=12,
        gripper=GripperFormat.UNSIGNED,
        arm_count=2,
        delta=False,
    ),
    proprioception=Proprioception(
        joint_pos=JointObservationSpec(dof=12, gripper=GripperFormat.UNSIGNED, arm_count=2),
    ),
)
