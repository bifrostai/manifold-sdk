"""Franka Panda on a mobile base, driven as a whole body: base, then arm.

A Franka Panda arm on a base with three joints: x, y and yaw. ManiLadder calls this
robot `mobile_panda`. An action is 10 floats, read in this order:

    [ base: target x, y, yaw (3) | arm: EE pos delta (3)
      | EE rot delta (3, XYZ Euler) | gripper (1) ]

The base values are the targets of the base's joints, which count from the point
where the robot is mounted in the scene, so a zero target puts the base on that
point. The arm's deltas are in the world frame.

The robot reports the EE pose with its two finger joints and the base pose, both in
the world frame.
"""

from __future__ import annotations

from manifold.core.conventions import Frame, GripperFormat, RotationFormat
from manifold.core.embodiment import (
    BodyActionSpace,
    BodyPart,
    Command,
    CompositeActionSpace,
    EEActionSpace,
    EEObservationSpec,
    Embodiment,
    GripperObservationSpec,
    Proprioception,
)
from manifold.core.names import EmbodimentName

FRANKA_MOBILE_WHOLE_BODY = Embodiment(
    name=EmbodimentName.FRANKA_MOBILE_WHOLE_BODY,
    action=CompositeActionSpace(
        groups=(
            BodyActionSpace(
                part=BodyPart.MOBILE_BASE,
                dof=3,
                command=Command.ABSOLUTE,
                frame=Frame.MOUNT,
            ),
            EEActionSpace(
                rotation=RotationFormat.EULER_XYZ,
                gripper=GripperFormat.SIGNED,
                frame=Frame.WORLD,
                delta=True,
            ),
        ),
    ),
    proprioception=Proprioception(
        ee_pose=EEObservationSpec(
            rotation=RotationFormat.QUATERNION,
            gripper=GripperObservationSpec(dim=2),
            frame=Frame.WORLD,
        ),
        base_pose=EEObservationSpec(rotation=RotationFormat.QUATERNION, frame=Frame.WORLD),
    ),
)
