"""Franka Panda with a right Leap hand, driven in end-effector deltas and hand joint deltas.

ManiLadder calls this robot `panda_leaphand`. An action is 22 floats, read in this order:

    [ arm: EE pos delta (3) | EE rot delta (3, XYZ Euler) | hand: joint deltas (16) ]

The arm reads its deltas in its base frame. The hand's 16 joints are the Leap hand's
joints `0` to `15`, each a change of its target in [-1, 1], which the robot scales to
0.1 rad.

The robot reports the EE pose in the world frame, and the hand's 16 joint positions as
`joint_pos`.
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

FRANKA_LEAP_EE_DELTA = Embodiment(
    name=EmbodimentName.FRANKA_LEAP_EE_DELTA,
    action=CompositeActionSpace(
        groups=(
            EEActionSpace(rotation=RotationFormat.EULER_XYZ, frame=Frame.BASE, delta=True),
            JointActionSpace(dof=16, delta=True),
        ),
    ),
    proprioception=Proprioception(
        ee_pose=EEObservationSpec(rotation=RotationFormat.QUATERNION, frame=Frame.WORLD),
        joint_pos=JointObservationSpec(dof=16),
    ),
)
