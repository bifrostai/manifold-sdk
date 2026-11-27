"""Franka Panda with a right Leap hand on a mobile base, driven as a whole body.

ManiLadder calls this robot `mobile_panda_leaphand_urdf`. An action is 25 floats, read in
this order:

    [ base: target x, y, yaw (3) | arm: EE pos delta (3) | EE rot delta (3, XYZ Euler)
      | hand: joint deltas (16) ]

The base values are the targets of the base's joints, which count from the point where
the robot is mounted in the scene. The arm reads its deltas in the frame of that point.
The hand's joints are as `franka_leap_ee_delta` reads them.

The robot reports the EE pose and the base pose in the world frame, and the hand's 16
joint positions as `joint_pos`.
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

FRANKA_MOBILE_LEAP_WHOLE_BODY = Embodiment(
    name=EmbodimentName.FRANKA_MOBILE_LEAP_WHOLE_BODY,
    action=CompositeActionSpace(
        groups=(
            BodyActionSpace(
                part=BodyPart.MOBILE_BASE, dof=3, command=Command.ABSOLUTE, frame=Frame.MOUNT
            ),
            EEActionSpace(rotation=RotationFormat.EULER_XYZ, frame=Frame.MOUNT, delta=True),
            JointActionSpace(dof=16, delta=True),
        ),
    ),
    proprioception=Proprioception(
        ee_pose=EEObservationSpec(rotation=RotationFormat.QUATERNION, frame=Frame.WORLD),
        base_pose=EEObservationSpec(rotation=RotationFormat.QUATERNION, frame=Frame.WORLD),
        joint_pos=JointObservationSpec(dof=16),
    ),
)
