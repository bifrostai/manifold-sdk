"""WidowX arm driven in end-effector deltas (the SIMPLER / Bridge control mode).

A 7-D action: a 3-D position delta, a 3-D euler-XYZ rotation delta, and a signed
gripper. The arm reports its end-effector pose in the robot base frame, with
rotation as a scalar-last quaternion and the gripper closedness nested in the
ee_pose.
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

WIDOWX_EE_DELTA = Embodiment(
    name="widowx_ee_delta",
    action=EEActionSpace(
        rotation=RotationFormat.EULER_XYZ,
        gripper=GripperFormat.SIGNED,
        delta=True,
    ),
    proprioception=Proprioception(
        ee_pose=EEObservationSpec(
            rotation=RotationFormat.QUATERNION,
            gripper=GripperObservationSpec(dim=1),
            frame=Frame.BASE,
        ),
    ),
)
