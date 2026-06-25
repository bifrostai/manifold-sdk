"""Franka Panda driven in end-effector deltas (the LIBERO control mode).

A 7-D action: a 3-D position delta, a 3-D axis-angle rotation delta, and a signed
gripper that opens on the LOW value. This is the robosuite operational-space
control LIBERO runs the arm in: the lerobot LIBERO env opens the gripper at -1 and
closes at +1, so the convention is SIGNED_OPEN_LOW (not the high-opens SIGNED).
"""

from __future__ import annotations

from manifold.core.conventions import GripperFormat, RotationFormat
from manifold.core.embodiment import (
    EEActionSpace,
    EEObservationSpec,
    Embodiment,
    GripperObservationSpec,
    Proprioception,
)

FRANKA_EE = Embodiment(
    name="franka_ee",
    action=EEActionSpace(
        rotation=RotationFormat.AXIS_ANGLE,
        gripper=GripperFormat.SIGNED_OPEN_LOW,
        delta=True,
    ),
    proprioception=Proprioception(
        # The env's TRUE native ee_pose: robosuite/LIBERO reports the end-effector
        # orientation as a scalar-last (x, y, z, w) QUATERNION, not axis-angle — so
        # the embodiment declares QUATERNION here (ADR-0001, decision 7: the embodiment records
        # the env's native form, and the pairing pipeline bridges it to whatever the
        # policy consumes). robosuite's parallel-jaw Franka reports both finger joint
        # positions, nested in the EE value (mirroring the gripper field on the action
        # side).
        ee_pose=EEObservationSpec(
            rotation=RotationFormat.QUATERNION,
            gripper=GripperObservationSpec(dim=2),
        ),
    ),
)
