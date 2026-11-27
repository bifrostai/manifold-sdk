"""Two Franka Panda arms driven in end-effector deltas, the first arm then the second.

ManiLadder's dual-arm tasks place two Pandas facing each other across the table. An
action is 14 floats, 7 for each arm:

    [ first arm: EE pos delta (3) | EE rot delta (3, XYZ Euler) | gripper (1)
      | second arm: the same 7 ]

Each arm reads its deltas in its own base frame, and its gripper opens at +1.

The robot reports each arm's EE pose with its two finger joints, in the world frame.
"""

from __future__ import annotations

from manifold.core.conventions import Frame, GripperFormat, RotationFormat
from manifold.core.embodiment import (
    EEActionSpace,
    EEObservationSpec,
    Embodiment,
    GripperObservationSpec,
    Proprioception,
)
from manifold.core.names import EmbodimentName

DUAL_FRANKA_EE_DELTA = Embodiment(
    name=EmbodimentName.DUAL_FRANKA_EE_DELTA,
    action=EEActionSpace(
        rotation=RotationFormat.EULER_XYZ,
        gripper=GripperFormat.SIGNED,
        arm_count=2,
        frame=Frame.BASE,
        delta=True,
    ),
    proprioception=Proprioception(
        ee_pose=EEObservationSpec(
            rotation=RotationFormat.QUATERNION,
            gripper=GripperObservationSpec(dim=2),
            arm_count=2,
            frame=Frame.WORLD,
        ),
    ),
)
