"""The action spaces an embodiment can be driven in.

A discriminated union on `kind`. The field set already distinguishes the
variants; the explicit discriminator makes parsing unambiguous. A spec describes
a single action, which holds the values that the robot executes in one environment
step. `PolicySignature` declares how many actions a model predicts in a call
(`chunk_size`) and how many of them run (`execution_steps`).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from manifold.core.conventions import Frame, GripperFormat, RotationFormat, ee_step_layout
from manifold.core.embodiment.spec import ValueSpec


def _reject_chunk_size(data: Any) -> Any:
    """Drop a legacy `chunk_size` of 1; raise for a larger one.

    Older benchmark images still send `chunk_size: 1` in their HELLO, so a value of
    1 parses and is discarded. A larger value used to mean an action chunk, which
    the signature now declares; silently dropping it would serve the wrong shape.
    """
    if isinstance(data, dict) and "chunk_size" in data:
        chunk_size = data["chunk_size"]
        if chunk_size != 1:
            raise ValueError(
                f"`chunk_size` moved to `PolicySignature` (got chunk_size={chunk_size!r} "
                "on the action space); declare `PolicySignature(chunk_size=..., "
                "execution_steps=...)`; the action space describes a single action"
            )
        data = {k: v for k, v in data.items() if k != "chunk_size"}
    return data


class EEActionSpace(ValueSpec):
    """An end-effector action: a pose target, or a pose increment when `delta`.

    An action carries, in order: 3 floats for position (xyz, meters), the rotation
    (per `rotation`), and 1 gripper float if `gripper` is set. Delta commands are
    common in VLA-style policies (RT-1, OpenVLA).
    """

    kind: Literal["ee"] = "ee"
    rotation: RotationFormat
    gripper: GripperFormat | None = None
    frame: Frame = Frame.WORLD
    delta: bool = False

    _reject_chunk_size = model_validator(mode="before")(_reject_chunk_size)

    def expected_length(self) -> int:
        """Floats in a single action: position, rotation, and the gripper if set."""
        return sum(ee_step_layout(self.rotation, self.gripper))


class JointActionSpace(ValueSpec):
    """A joint action: target joint angles, or per-joint increments when `delta`.

    An action carries `dof` joint values in radians, optionally followed by 1
    gripper float. Absolute target angles are common in ACT- and ALOHA-style
    policies.
    """

    kind: Literal["joint"] = "joint"
    dof: int = Field(ge=1)
    gripper: GripperFormat | None = None
    delta: bool = False

    _reject_chunk_size = model_validator(mode="before")(_reject_chunk_size)

    def expected_length(self) -> int:
        """Floats in a single action: the joint values and the gripper if set."""
        return self.dof + (1 if self.gripper else 0)


class UnifiedActionSpace(ValueSpec):
    """A fixed-width padded action buffer a multi-embodiment base model emits.

    The policy emits a `width`-wide vector for each action. One embodiment's action
    occupies the leading slots and the rest is zero padding (pi0.5 = 32, RDT-1B =
    128). `payload` is the real action a slice adapter extracts and that is then
    matched against a benchmark. Leave it None for an uncommitted base model,
    which has neither embodiment semantics nor a link to a benchmark.
    """

    kind: Literal["unified"] = "unified"
    width: int = Field(ge=1)
    payload: Annotated[EEActionSpace | JointActionSpace, Field(discriminator="kind")] | None = None

    def expected_length(self) -> int:
        """Floats in a single action: the full padded width."""
        return self.width


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
