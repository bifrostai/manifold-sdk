"""DROID (Franka Panda + Robotiq 2F-85) in absolute joint targets (RoboLab jointpos).

RoboLab's default benchmark embodiment, per `robolab/robots/README.md`: a Franka
Panda arm with a Robotiq 2F-85 gripper, tagged `benchmark-default`, with high PD
gains and gravity disabled on the arm. That README lists DROID and a stock Franka
Panda as *separate* robots — the stock entry keeps the factory finger gripper and
reports "EE frame pose and finger joint positions", which is what `franka_ee_delta`
models — so this is its own embodiment rather than a reuse of either Franka entry.

The README gives DROID three action configs, and this models the first, which the
suite's standard registration binds (`registrations/droid/`, jointpos):

    DroidJointPositionActionCfg   7 arm joint targets + binary gripper       8
    DroidIKActionCfg              absolute EE pose (x, y, z, qw, qx, qy, qz)  8
    DroidRelIKActionCfg           relative EE pose (dx, dy, dz, d-euler)      7

The targets are absolute, not deltas: `JointPositionActionCfg(use_default_offset=False)`
commands the angle itself, and the config's own docstring reads "Joint-space arm +
gripper actions; no Cartesian frame" — so no `Frame` applies to the action.

The gripper is BINARY_OPEN_LOW, the README's convention for every binary gripper
RoboLab ships: "a scalar per gripper, `> 0.5` closes, `<= 0.5` opens".

Proprioception is the 7 arm joint angles and one gripper scalar (clipped to [0, 1],
with Gaussian observation noise). Note the polarity: the README calls this a gripper
*open* fraction, but `gripper_pos` is `finger_joint / (pi / 4)` where the open
command is `finger_joint = 0.0` and the close command is `pi / 4`, and its own
docstring reads "Returns gripper position as 0 for open and 1 for closed". The code
governs, so it is declared UNSIGNED_OPEN_LOW.

`ee_pose` carries the rotated control frame (`eef_*`), not the gripper mount flange
(`ee_*`). RoboLab publishes both, and at the pinned commit they are co-located —
observed identical to four decimals — differing only in orientation, so the choice
is a rotation convention rather than a position one. `eef_frame` is the one upstream
treats as canonical (its own IK demo converts `eef_frame` targets into `base_link`
actions), and it is declared on the robot config rather than per task, so every task
in the suite reports it.

Two conventions the runner reconciles rather than this declaration:

The README fixes quaternion order globally — "Quaternions are `(w, x, y, z)`" —
while `RotationFormat.QUATERNION` is scalar-last `(x, y, z, w)`. The runner reorders
before the value goes on the wire, as the SIMPLER and Arena runners do for the same
Isaac convention, so the declaration stays scalar-last and matches the wire. A
scalar-first `RotationFormat` variant is rejected: the wire carries one order,
converted at the edge.

`Frame.WORLD` because that is what the pinned commit reports: `eef_pos` subtracts
`env_origins` from a world position and `eef_quat` is `target_quat_w` outright, so
the pose is the environment's own world frame, not the robot root. Upstream has
since moved `ee_pos`/`ee_quat` to the robot-root frame, which would be `Frame.BASE`
— re-probe when `ROBOLAB_REF` moves. A policy requiring base-frame values pairs
through `FrameRebaseAdapter`; one fed base-frame values under a WORLD label would
fail silently, which is why the frame is named from the code rather than assumed.
"""

from __future__ import annotations

from manifold.core.conventions import Frame, GripperFormat, RotationFormat
from manifold.core.embodiment import (
    EEObservationSpec,
    Embodiment,
    JointActionSpace,
    JointObservationSpec,
    Proprioception,
)

DROID_JOINT_ABSOLUTE = Embodiment(
    name="droid_joint_absolute",
    action=JointActionSpace(
        dof=7,
        gripper=GripperFormat.BINARY_OPEN_LOW,
        delta=False,
    ),
    proprioception=Proprioception(
        joint_pos=JointObservationSpec(dof=7, gripper=GripperFormat.UNSIGNED_OPEN_LOW),
        # No `gripper` here: the finger state is already the eighth value of
        # `joint_pos`, and RoboLab reports it once rather than nesting it in the EE
        # value the way robosuite's parallel-jaw Franka does.
        ee_pose=EEObservationSpec(rotation=RotationFormat.QUATERNION, frame=Frame.WORLD),
    ),
)
