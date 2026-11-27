"""Two mobile Franka Pandas, one with a left and one with a right Leap hand, as a whole body.

ManiLadder builds this robot from `mobile_panda_leaphand_left_urdf` and
`mobile_panda_leaphand_urdf`. An action is 50 floats: the left hand's robot, then the
right hand's, each as `franka_mobile_leap_whole_body` reads it. ManiLadder splits a flat
action over its robots in the order of their ids, and the left robot's id sorts first.

The robot reports each arm's EE pose in the world frame, and each hand's 16 joint
positions, in the same order. A spec holds one base pose, so neither base's pose is
reported.
"""

from __future__ import annotations

from manifold.core.conventions import Frame, RotationFormat
from manifold.core.embodiment import (
    BodyActionSpace,
    BodyPart,
    Command,
    CompositeActionSpace,
    EEActionSpace,
    EEObservationSpec,
    Embodiment,
    JointActionSpace,
    JointObservationSpec,
    Proprioception,
)
from manifold.core.names import EmbodimentName


def _robot() -> tuple[BodyActionSpace | EEActionSpace | JointActionSpace, ...]:
    return (
        BodyActionSpace(
            part=BodyPart.MOBILE_BASE, dof=3, command=Command.ABSOLUTE, frame=Frame.MOUNT
        ),
        EEActionSpace(rotation=RotationFormat.EULER_XYZ, frame=Frame.MOUNT, delta=True),
        JointActionSpace(dof=16, delta=True),
    )


DUAL_FRANKA_MOBILE_LEAP_WHOLE_BODY = Embodiment(
    name=EmbodimentName.DUAL_FRANKA_MOBILE_LEAP_WHOLE_BODY,
    action=CompositeActionSpace(groups=(*_robot(), *_robot())),
    proprioception=Proprioception(
        ee_pose=EEObservationSpec(
            rotation=RotationFormat.QUATERNION, arm_count=2, frame=Frame.WORLD
        ),
        joint_pos=JointObservationSpec(dof=32, arm_count=2),
    ),
)
