"""Rainbow Robotics RB-Y1 with two Wuji2 hands, in absolute joint targets.

A bimanual mobile humanoid: two 7-joint arms, a 2-joint neck, a 6-joint torso, and a
20-joint Wuji2 hand on each arm. An action and the reported state both list 62 joint
positions in radians: the left arm, the right arm, the neck, the 40 hand joints, then
the torso. The wheels of the mobile base are not in the vector.

The hands are joints, not grippers, so no `GripperFormat` applies. `arm_count` stays 1
because the arms, neck, hands and torso do not split into equal arms, so the vector
is a single block of 62 joints.

The targets are absolute: the vector is the joint positions that the controller drives
to, not increments, so no `Frame` applies to the action.
"""

from __future__ import annotations

from manifold.core.embodiment import (
    Embodiment,
    JointActionSpace,
    JointObservationSpec,
    Proprioception,
)
from manifold.core.names import EmbodimentName

RBY1_WUJI2_JOINT_ABSOLUTE = Embodiment(
    name=EmbodimentName.RBY1_WUJI2_JOINT_ABSOLUTE,
    action=JointActionSpace(dof=62, gripper=None, arm_count=1, delta=False),
    proprioception=Proprioception(
        joint_pos=JointObservationSpec(dof=62, gripper=None, arm_count=1),
    ),
)
