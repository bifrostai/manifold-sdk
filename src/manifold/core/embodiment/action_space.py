"""The action spaces an embodiment can be driven in.

A discriminated union on `kind`. The field set already distinguishes the
variants; the explicit discriminator makes parsing unambiguous. A spec carries
one step unless `chunk_size > 1`, in which case it carries that many consecutive
steps stacked on a leading axis (one action chunk per inference call).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from manifold.core.conventions import Frame, GripperFormat, RotationFormat, ee_step_layout
from manifold.core.embodiment.spec import ValueSpec


class EEActionSpace(ValueSpec):
    """An end-effector action: a pose target, or a pose increment when `delta`.

    One step carries, in order: 3 floats for position (xyz, meters), the rotation
    (per `rotation`), and 1 gripper float if `gripper` is set. Delta commands are
    common in VLA-style policies (RT-1, OpenVLA).
    """

    kind: Literal["ee"] = "ee"
    rotation: RotationFormat
    gripper: GripperFormat | None = None
    frame: Frame = Frame.WORLD
    delta: bool = False
    chunk_size: int = Field(default=1, ge=1)

    def per_step_length(self) -> int:
        """Floats in one chunk step: position, rotation, and the gripper if set."""
        return sum(ee_step_layout(self.rotation, self.gripper))

    def expected_length(self) -> int:
        """Floats across one emitted action, all chunk steps included."""
        return self.per_step_length() * self.chunk_size


class JointActionSpace(ValueSpec):
    """A joint action: target joint angles, or per-joint increments when `delta`.

    One step carries `dof` joint values in radians, optionally followed by 1
    gripper float. Absolute target angles are common in ACT- and ALOHA-style
    policies.
    """

    kind: Literal["joint"] = "joint"
    dof: int = Field(ge=1)
    gripper: GripperFormat | None = None
    delta: bool = False
    chunk_size: int = Field(default=1, ge=1)

    def per_step_length(self) -> int:
        """Floats in one chunk step: the joint values and the gripper if set."""
        return self.dof + (1 if self.gripper else 0)

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
