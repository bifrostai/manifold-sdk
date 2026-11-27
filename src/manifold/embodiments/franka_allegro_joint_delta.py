"""Franka Panda with an Allegro hand, driven by the hand's joint deltas alone.

ManiLadder calls this robot `panda_allegro_hora`. Its one task, `SpinPenHora-v1`, holds
the arm still and drives only the hand. An action is the Allegro hand's 16 joints,
`joint_0.0` to `joint_15.0`, each a change of its target in [-1, 1], which the robot
scales to 0.1 rad.

The robot reports the EE pose in the world frame, and the hand's 16 joint positions as
`joint_pos`.
"""

from __future__ import annotations

from manifold.core.conventions import Frame, RotationFormat
from manifold.core.embodiment import (
    EEObservationSpec,
    Embodiment,
    JointActionSpace,
    JointObservationSpec,
    Proprioception,
)
from manifold.core.names import EmbodimentName

FRANKA_ALLEGRO_JOINT_DELTA = Embodiment(
    name=EmbodimentName.FRANKA_ALLEGRO_JOINT_DELTA,
    action=JointActionSpace(dof=16, delta=True),
    proprioception=Proprioception(
        ee_pose=EEObservationSpec(rotation=RotationFormat.QUATERNION, frame=Frame.WORLD),
        joint_pos=JointObservationSpec(dof=16),
    ),
)
