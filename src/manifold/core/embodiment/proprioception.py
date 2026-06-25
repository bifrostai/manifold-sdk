"""What a robot reports about its own state: end-effector pose or joint angles."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from manifold.core.conventions import Frame, GripperFormat, RotationFormat, ee_step_layout
from manifold.core.embodiment.spec import ValueSpec


class GripperObservationSpec(ValueSpec):
    """The gripper finger joint state nested in an end-effector observation.

    One value carries `dim` finger joint positions (qpos) — 2 for a parallel-jaw
    gripper (the two opposing fingers), as robosuite's Franka publishes.

    This is observed gripper *proprioception*: where the fingers actually are.
    It is distinct from the action-side `GripperFormat` (the polarity and range of
    a *commanded* gripper value), and mirrors it structurally: just as the gripper
    is a field of `EEActionSpace`, this is the gripper field of `EEObservationSpec`.

    `encoding` is optional and defaults to None — the long-standing case where the
    value is raw finger qpos with no declared polarity/range, untouched at the seam.
    When a benchmark instead reports the grasp as a single scalar in one of the
    `GripperFormat` ranges (e.g. SIMPLER's `1 - closedness` openness in [0, 1]), it
    declares that encoding here, and `ObservedGripperAdapter` can bridge it to the
    encoding the policy was trained to read. Setting `encoding` does not change
    `expected_length`; it annotates what the `dim` values mean.
    """

    dim: int = Field(default=2, ge=1)
    encoding: GripperFormat | None = None

    def expected_length(self) -> int:
        """Floats in one gripper value: the `dim` finger joint positions."""
        return self.dim


class EEObservationSpec(ValueSpec):
    """The end-effector state the robot reports.

    One value carries, in order: 3 floats for Cartesian position (xyz, meters),
    the rotation (3, 4, or 6 floats per `rotation`), and the gripper finger joint
    positions if `gripper` is set (`gripper.dim` floats, e.g. 2 for a parallel-jaw).
    """

    rotation: RotationFormat
    gripper: GripperObservationSpec | None = None
    frame: Frame = Frame.WORLD

    def expected_length(self) -> int:
        """Floats in one EE value: position, rotation, then optional gripper qpos."""
        pose_len = sum(ee_step_layout(self.rotation, None))
        return pose_len + (self.gripper.dim if self.gripper is not None else 0)


class JointObservationSpec(ValueSpec):
    """The joint angles the robot reports: `dof` angles, then an optional gripper."""

    dof: int = Field(ge=1)
    gripper: GripperFormat | None = None

    def expected_length(self) -> int:
        """Floats in one joint value: the joint angles, then an optional gripper."""
        return self.dof + (1 if self.gripper else 0)


class Proprioception(BaseModel):
    """What the robot reports about its own state each step.

    Set `ee_pose`, `joint_pos`, or any combination. All may be left unset for a
    robot that reports no proprioception. Gripper finger state, when reported, is
    nested in the end-effector value (`EEObservationSpec.gripper`), mirroring the
    action side where the gripper is a field of `EEActionSpace`.
    """

    model_config = ConfigDict(frozen=True)

    ee_pose: EEObservationSpec | None = None
    joint_pos: JointObservationSpec | None = None

    def first_unmet(self, available: Proprioception) -> str | None:
        """The first component this needs that `available` does not match, or None.

        Used to check that a benchmark publishes the proprioception a policy
        consumes: `self` is what the policy requires, `available` is what is offered.
        The `ee_pose` comparison includes its nested gripper, so a policy that
        needs gripper qpos is unmet against a benchmark whose ee_pose lacks it.
        """
        if self.ee_pose is not None and available.ee_pose != self.ee_pose:
            return "ee_pose"
        if self.joint_pos is not None and available.joint_pos != self.joint_pos:
            return "joint_pos"
        return None


__all__ = [
    "EEObservationSpec",
    "GripperObservationSpec",
    "JointObservationSpec",
    "Proprioception",
]
