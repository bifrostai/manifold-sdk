"""Two Franka Pandas, each with a right Leap hand, in end-effector and hand joint deltas.

ManiLadder builds this robot from two `panda_leaphand` arms. An action is 44 floats: the
first arm, then the second, each as `franka_leap_ee_delta` reads it. Both hands are right
hands, so this robot differs from `dual_franka_leap_ee_delta`, whose second hand is a
left hand.

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

DUAL_FRANKA_RIGHT_LEAP_EE_DELTA = Embodiment(
    name=EmbodimentName.DUAL_FRANKA_RIGHT_LEAP_EE_DELTA,
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
