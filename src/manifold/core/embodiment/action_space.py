"""The action spaces an embodiment can be driven in.

A discriminated union on `kind`. The field set already distinguishes the
variants; the explicit discriminator makes parsing unambiguous. A spec carries
one step unless `chunk_size > 1`, in which case it carries that many consecutive
steps stacked on a leading axis (one action chunk per inference call).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from manifold.core.conventions import (
    Frame,
    GripperFormat,
    RotationFormat,
    check_dof_divides_across_arms,
    ee_step_layout,
)
from manifold.core.embodiment.spec import ValueSpec


class EEActionSpace(ValueSpec):
    """An end-effector action: a pose target, or a pose increment when `delta`.

    Each arm contributes, in order: 3 floats for position (xyz, meters), the rotation
    (per `rotation`), and 1 gripper float if `gripper` is set. A step is `arm_count`
    of those groups back to back, since every arm carries its own pose. Delta
    commands are common in VLA-style policies (RT-1, OpenVLA).

    `gripper` is what each arm ends in, so there are `arm_count` grippers when it is
    set and none otherwise; the count is never declared on its own. One gripper per
    arm is the model -- an arm carrying several is not describable here.
    """

    kind: Literal["ee"] = "ee"
    rotation: RotationFormat
    gripper: GripperFormat | None = None
    arm_count: int = Field(default=1, ge=1)
    frame: Frame = Frame.WORLD
    delta: bool = False
    chunk_size: int = Field(default=1, ge=1)

    def per_arm_length(self) -> int:
        """Floats one arm contributes to a step: position, rotation, its gripper if set."""
        return sum(ee_step_layout(self.rotation, self.gripper))

    def per_step_length(self) -> int:
        """Floats in one chunk step: every arm's position, rotation and gripper."""
        return self.per_arm_length() * self.arm_count

    def expected_length(self) -> int:
        """Floats across one emitted action, all chunk steps included."""
        return self.per_step_length() * self.chunk_size


class JointActionSpace(ValueSpec):
    """A joint action: target joint angles, or per-joint increments when `delta`.

    One step carries `dof` joint values in radians across every arm, each arm's
    joints followed by its gripper float if `gripper` is set. Absolute target angles
    are common in ACT- and ALOHA-style policies.

    `arm_count` is how many arms share those joints. Unlike on `EEActionSpace`, `dof`
    is the total across arms, not a per-arm figure -- which is what it already meant
    before a robot could have two -- so each arm has `dof // arm_count` joints. At 1
    that is the existing layout, `dof` joints then one gripper, and nothing already
    declared changes meaning.

    At 2 it is the ALOHA convention, which is what a bimanual robot actually sends:
    `[left joints (6), left gripper, right joints (6), right gripper]`, grippers at
    index 6 and 13. That is where openpi's ALOHA policy expects them, its own comment
    reading "state is [left_arm_joint_angles, left_arm_gripper,
    right_arm_joint_angles, right_arm_gripper]". Declaring trailing grippers instead
    would have meant every runner and every policy permuting into a layout nothing in
    the ecosystem uses.

    `gripper` is what each arm ends in, so there are `arm_count` grippers when it is
    set. One gripper per arm and arms of equal size are the model: a 7+6 robot is not
    describable here, and declares one arm.
    """

    kind: Literal["joint"] = "joint"
    dof: int = Field(ge=1)
    gripper: GripperFormat | None = None
    arm_count: int = Field(default=1, ge=1)
    delta: bool = False
    chunk_size: int = Field(default=1, ge=1)

    def per_arm_length(self) -> int:
        """Floats one arm contributes to a step: its joints, then its gripper if set."""
        return self.dof // self.arm_count + (1 if self.gripper else 0)

    def per_step_length(self) -> int:
        """Floats in one chunk step: every joint, then one gripper per arm if set."""
        return self.dof + (self.arm_count if self.gripper else 0)

    @model_validator(mode="after")
    def _check_dof_divides_across_arms(self) -> JointActionSpace:
        """Refuse a `dof` that cannot split into `arm_count` equal arms."""
        check_dof_divides_across_arms(self.dof, self.arm_count)
        return self

    def expected_length(self) -> int:
        """Floats across one emitted action, all chunk steps included."""
        return self.per_step_length() * self.chunk_size


class UnifiedActionSpace(ValueSpec):
    """A fixed-width padded action buffer a multi-embodiment base model emits.

    The policy emits a `width`-wide vector per step; one embodiment's real action
    occupies the leading slots and the rest is zero padding (pi0.5 = 32, RDT-1B =
    128). `payload` is the real action a slice adapter extracts and that is then
    matched against a benchmark. Leave it None for an uncommitted base model,
    which has neither embodiment semantics nor a link to a benchmark.
    """

    kind: Literal["unified"] = "unified"
    width: int = Field(ge=1)
    payload: Annotated[EEActionSpace | JointActionSpace, Field(discriminator="kind")] | None = None

    def expected_length(self) -> int:
        """Floats across one emitted buffer, all chunk steps included."""
        chunk = self.payload.chunk_size if self.payload is not None else 1
        return self.width * chunk


ActionSpace = Annotated[
    EEActionSpace | JointActionSpace | UnifiedActionSpace,
    Field(discriminator="kind"),
]


__all__ = [
    "ActionSpace",
    "EEActionSpace",
    "JointActionSpace",
    "UnifiedActionSpace",
]
