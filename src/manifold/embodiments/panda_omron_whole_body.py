"""Mobile Franka (PandaOmron) driven for RoboCasa kitchen tasks.

A Franka Panda arm on an Omron holonomic base under robosuite/MuJoCo. The 12-D
action drives the whole body:

    [ EE pos delta (3) | EE rot delta (3, axis-angle) | gripper (1)
      | base motion (4) | control mode (1) ]

The leading 7 floats are exactly the FRANKA_EE_DELTA end-effector-delta action; the
trailing 5 are 4 mobile-base DOFs plus a control-mode selector, all driven (the
base is not pinned) since the kitchen tasks involve whole-body motion.

Modeled as a `UnifiedActionSpace(width=12)` with the FRANKA_EE_DELTA arm action as its
payload rather than an extended `EEActionSpace`: the EE layout fixes the gripper as
the last element of each step (indexed via `ee_step_layout`), so appending base DOFs
would push it off the end. The trailing 5 dims are real action, not padding, so a
pairing matches the identical UnifiedActionSpace directly (no `UnifiedSliceAdapter`,
which would discard them).

Proprioception is the base-relative EE pose with the 2-D parallel-jaw gripper qpos
nested in it, mirroring FRANKA_EE_DELTA. The env reports the EE orientation as a
scalar-last (x, y, z, w) QUATERNION natively (robocasa/robosuite's
robot0_base_to_eef_quat), so the embodiment declares QUATERNION here — the
rotation re-encode to whatever a policy consumes moved off the runner and onto the
pairing pipeline per ADR-0001, decision 7. The mobile base pose is part of the RLDX state but
has no proprioception channel; the runner ships it in the flat obs dict.
"""

from __future__ import annotations

from manifold.core.conventions import Frame, GripperFormat, RotationFormat
from manifold.core.embodiment import (
    EEActionSpace,
    EEObservationSpec,
    Embodiment,
    GripperObservationSpec,
    Proprioception,
    UnifiedActionSpace,
)

# The gripper is SIGNED ([-1, 1], +1 = closed for the PandaOmron), the polarity
# the env applies — note this differs from FRANKA_EE_DELTA's open-low convention.
PANDA_OMRON_WHOLE_BODY = Embodiment(
    name="panda_omron_whole_body",
    action=UnifiedActionSpace(
        width=12,
        payload=EEActionSpace(
            rotation=RotationFormat.AXIS_ANGLE,
            gripper=GripperFormat.SIGNED,
            frame=Frame.BASE,
            delta=True,
        ),
    ),
    proprioception=Proprioception(
        # The base-relative EE pose, with the 2-D parallel-jaw finger qpos nested in
        # the EE value (mirroring FRANKA_EE_DELTA). The env reports the orientation as a
        # scalar-last (x, y, z, w) QUATERNION (robot0_base_to_eef_quat), so the
        # embodiment declares QUATERNION here (ADR-0001, decision 7: the embodiment records the
        # env's TRUE native form; the pairing pipeline bridges it to whatever the
        # policy consumes). The pose is genuinely base-relative, so frame stays BASE.
        ee_pose=EEObservationSpec(
            rotation=RotationFormat.QUATERNION,
            gripper=GripperObservationSpec(dim=2),
            frame=Frame.BASE,
        ),
    ),
)
