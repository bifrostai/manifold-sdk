"""Two Franka Pandas, one with a right and one with a left Leap hand, in deltas.

ManiLadder builds this robot from `panda_leaphand` and `panda_leaphand_left`, in either
order. An action is 44 floats: the right hand's arm, then the left hand's arm, each as
`franka_leap_ee_delta` reads it:

    [ right: EE pos delta (3) | EE rot delta (3, XYZ Euler) | hand joint deltas (16)
      | left: the same 22 ]

ManiLadder splits a flat action over its arms in the order of their ids, and
`panda_leaphand` sorts before `panda_leaphand_left`, so the right hand comes first
whichever position the task gives it.

The robot reports each arm's EE pose in the world frame, then each hand's 16 joint
positions, in the same order.
"""

from __future__ import annotations

from manifold.core.conventions import Frame, RotationFormat
from manifold.core.embodiment import (
    CompositeActionSpace,
    EEActionSpace,
    EEObservationSpec,
    Embodiment,
    JointActionSpace,
    JointObservationSpec,
    Proprioception,
)
from manifold.core.names import EmbodimentName

DUAL_FRANKA_LEAP_EE_DELTA = Embodiment(
    name=EmbodimentName.DUAL_FRANKA_LEAP_EE_DELTA,
    action=CompositeActionSpace(
        groups=(
            EEActionSpace(rotation=RotationFormat.EULER_XYZ, frame=Frame.BASE, delta=True),
            JointActionSpace(dof=16, delta=True),
            EEActionSpace(rotation=RotationFormat.EULER_XYZ, frame=Frame.BASE, delta=True),
            JointActionSpace(dof=16, delta=True),
        ),
    ),
    proprioception=Proprioception(
        ee_pose=EEObservationSpec(
            rotation=RotationFormat.QUATERNION, arm_count=2, frame=Frame.WORLD
        ),
        joint_pos=JointObservationSpec(dof=32, arm_count=2),
    ),
)
