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

    Each arm contributes, in order: 3 floats for position (xyz, meters), the rotation
    (per `rotation`), and 1 gripper float if `gripper` is set. An action holds
    `arm_count` of those groups back to back, because each arm has its own pose.
    Delta commands are common in VLA-style policies (RT-1, OpenVLA).

    The `gripper` field describes the gripper that ends each arm. When `gripper` is
    set, the robot has `arm_count` grippers. When it is unset, the robot has none. No
    field counts the grippers, so no second number can contradict `arm_count`.

    This spec allows a single gripper on an arm. It cannot describe an arm that
    carries two.
    """

    kind: Literal["ee"] = "ee"
    rotation: RotationFormat
    gripper: GripperFormat | None = None
    arm_count: int = Field(default=1, ge=1)
    frame: Frame = Frame.WORLD
    delta: bool = False

    _reject_chunk_size = model_validator(mode="before")(_reject_chunk_size)

    def per_arm_length(self) -> int:
        """Floats one arm contributes: position, rotation, its gripper if set."""
        return sum(ee_step_layout(self.rotation, self.gripper))

    def expected_length(self) -> int:
        """Floats in a single action: every arm's position, rotation and gripper."""
        return self.per_arm_length() * self.arm_count


class JointActionSpace(ValueSpec):
    """A joint action: target joint angles, or per-joint increments when `delta`.

    An action carries `dof` joint values in radians across the arms. Each arm's
    joints come first, followed by that arm's gripper float if `gripper` is set.
    Absolute target angles are common in ACT- and ALOHA-style policies.

    `arm_count` is the number of arms that share those joints. On `EEActionSpace` the
    layout describes a single arm. Here `dof` counts the joints of the whole robot,
    which is what `dof` meant before a robot could have two arms. Each arm therefore
    has `dof // arm_count` joints.

    At `arm_count=1` the layout is unchanged: `dof` joints, then a gripper. Nothing
    that was already declared changes meaning.

    At `arm_count=2` the layout follows the ALOHA convention, which is what a
    bimanual robot sends: `[left joints (6), left gripper, right joints (6), right
    gripper]`, with grippers at index 6 and 13. openpi's ALOHA policy reads them at
    those indices, and its comment reads "state is [left_arm_joint_angles,
    left_arm_gripper, right_arm_joint_angles, right_arm_gripper]". If the grippers
    were declared at the end instead, runners and policies would have to permute into
    a layout that no other project uses.

    The `gripper` field describes the gripper that ends each arm, so a robot with
    `gripper` set has `arm_count` of them. This spec allows a single gripper on an
    arm, and arms of equal size. A robot with a 7-joint arm and a 6-joint arm cannot
    be described here, and declares a single arm.
    """

    kind: Literal["joint"] = "joint"
    dof: int = Field(ge=1)
    gripper: GripperFormat | None = None
    arm_count: int = Field(default=1, ge=1)
    delta: bool = False

    _reject_chunk_size = model_validator(mode="before")(_reject_chunk_size)

    def per_arm_length(self) -> int:
        """Floats one arm contributes: its joints, then its gripper if set."""
        return self.dof // self.arm_count + (1 if self.gripper else 0)

    def expected_length(self) -> int:
        """Floats in a single action: every joint, then one gripper per arm if set."""
        return self.dof + (self.arm_count if self.gripper else 0)

    @model_validator(mode="after")
    def _check_dof_divides_across_arms(self) -> JointActionSpace:
        """Refuse a `dof` that cannot split into `arm_count` equal arms."""
        if self.dof % self.arm_count:
            raise ValueError(f"dof={self.dof} does not divide into {self.arm_count} equal arms")
        return self


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
