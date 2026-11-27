"""Two mobile Franka Pandas with grippers, driven as a whole body, first robot then second.

ManiLadder builds this robot from two `mobile_panda` robots. An action is 20 floats: the
first robot, then the second, each as `franka_mobile_whole_body` reads it:

    [ base: target x, y, yaw (3) | arm: EE pos delta (3) | EE rot delta (3, XYZ Euler)
      | gripper (1) ] twice

Each base counts from the point where its robot is mounted, and each arm reads its deltas
in the frame of that point. Each gripper opens at +1.

The robot reports each arm's EE pose with its two finger joints in the world frame. A
spec holds one base pose, so neither base's pose is reported.
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


def _robot() -> tuple[BodyActionSpace | EEActionSpace, ...]:
    return (
        BodyActionSpace(
            part=BodyPart.MOBILE_BASE, dof=3, command=Command.ABSOLUTE, frame=Frame.MOUNT
        ),
        EEActionSpace(
            rotation=RotationFormat.EULER_XYZ,
            gripper=GripperFormat.SIGNED,
            frame=Frame.MOUNT,
            delta=True,
        ),
    )


DUAL_FRANKA_MOBILE_WHOLE_BODY = Embodiment(
    name=EmbodimentName.DUAL_FRANKA_MOBILE_WHOLE_BODY,
    action=CompositeActionSpace(groups=(*_robot(), *_robot())),
    proprioception=Proprioception(
        ee_pose=EEObservationSpec(
            rotation=RotationFormat.QUATERNION,
            gripper=GripperObservationSpec(dim=2),
            arm_count=2,
            frame=Frame.WORLD,
        ),
    ),
)
