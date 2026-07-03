"""Franka Panda driven in absolute joint targets (joint-space control mode).

The action is 7 joint-angle targets plus a binary gripper. Unlike `franka_ee`,
the values are absolute, not deltas. The arm reports both its 7 joint angles (plus
the gripper scalar) and its end-effector pose as proprioception.

Why both proprio channels: a joint-space policy checkpoint is conditioned on
`observation/joint_position` (7) + `observation/gripper_position` (1), not the
end-effector pose — so the benchmark must publish joint-angle proprio or a
joint-space policy cannot pair. We keep `ee_pose` too: the runtime already has
the EE pose on hand, so it is published too, which lets an EE-space policy pair
against the same benchmark.

The joint-proprio gripper is a single scalar in [0, 1] (0 = open, 1 = closed),
declared `UNSIGNED_OPEN_LOW` to model that range and polarity.
"""

from __future__ import annotations

from manifold.core.conventions import Frame, GripperFormat, RotationFormat
from manifold.core.embodiment import (
    EEObservationSpec,
    Embodiment,
    JointActionSpace,
    JointObservationSpec,
    Proprioception,
)

FRANKA_JOINT = Embodiment(
    name="franka_joint",
    action=JointActionSpace(
        dof=7,
        gripper=GripperFormat.BINARY,
        delta=False,
    ),
    proprioception=Proprioception(
        joint_pos=JointObservationSpec(dof=7, gripper=GripperFormat.UNSIGNED_OPEN_LOW),
        # The runner sources the EE pose in the robot base_link frame (not the fixed
        # world/table frame). Declare ``Frame.BASE`` so the published contract matches
        # the data actually sent: an EE-space policy must declare ``Frame.BASE`` to
        # pair, and a WORLD-frame policy is correctly rejected by ``first_unmet``
        # rather than silently fed base-frame values.
        ee_pose=EEObservationSpec(rotation=RotationFormat.AXIS_ANGLE, frame=Frame.BASE),
    ),
)
