"""Mobile Franka (PandaOmron) driven as a whole body: arm, base, torso and mode.

A Franka Panda arm on an Omron holonomic base with a torso lift, under robosuite's
HybridMobileBase controller. An action is 12 floats, read in this order:

    [ arm: EE pos delta (3) | EE rot delta (3, axis-angle) | gripper (1)
      | base: velocity forward, side, yaw (3) | torso: lift delta (1) | mode (1) ]

The mode slot sets what the arm's delta is added to: its previous target when
high, its measured pose when low. RoboCasa365's gym wrapper cuts it at 0.5.

The robot reports the base-relative EE pose with its two finger joints, the arm
joints, the base pose in the world and the torso lift. The env reports both
orientations as scalar-last (x, y, z, w) quaternions (robot0_base_to_eef_quat,
robot0_base_quat); per ADR 0001, decision 7, the pairing pipeline re-encodes them
for a policy that reads another form.
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
    JointObservationSpec,
    ModeCut,
    Proprioception,
)
from manifold.core.names import EmbodimentName

PANDA_OMRON_WHOLE_BODY = Embodiment(
    name=EmbodimentName.PANDA_OMRON_WHOLE_BODY,
    action=CompositeActionSpace(
        groups=(
            # robosuite's PandaGripper closes at +1 and opens at -1.
            EEActionSpace(
                rotation=RotationFormat.AXIS_ANGLE,
                gripper=GripperFormat.SIGNED_OPEN_LOW,
                frame=Frame.BASE,
                delta=True,
            ),
            BodyActionSpace(
                part=BodyPart.MOBILE_BASE,
                dof=3,
                command=Command.VELOCITY,
                frame=Frame.BASE,
            ),
            BodyActionSpace(part=BodyPart.TORSO, dof=1, command=Command.DELTA),
        ),
        mode_slot=ModeCut.BINARY_0_5,
    ),
    proprioception=Proprioception(
        ee_pose=EEObservationSpec(
            rotation=RotationFormat.QUATERNION,
            gripper=GripperObservationSpec(dim=2),
            frame=Frame.BASE,
        ),
        joint_pos=JointObservationSpec(dof=7),
        base_pose=EEObservationSpec(rotation=RotationFormat.QUATERNION, frame=Frame.WORLD),
        torso_pos=JointObservationSpec(dof=1),
    ),
)
