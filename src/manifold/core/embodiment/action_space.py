"""The action spaces an embodiment can be driven in.

A discriminated union on `kind`. The field set already distinguishes the
variants; the explicit discriminator makes parsing unambiguous. A spec describes
a single action, which holds the values that the robot executes in one environment
step. `PolicySignature` declares how many actions a model predicts in a call
(`chunk_size`) and how many of them run (`execution_steps`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from manifold.core.conventions import Frame, GripperFormat, RotationFormat, ee_step_layout
from manifold.core.embodiment.spec import ValueSpec
from manifold.lib.compat import StrEnum, assert_never


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


class Command(StrEnum):
    """What a body group's values command."""

    ABSOLUTE = "absolute"
    DELTA = "delta"
    VELOCITY = "velocity"


class BodyPart(StrEnum):
    """The part of a mobile robot that a body group drives."""

    TORSO = "torso"  # A torso is the joints between a base and the arms, such as a lift.
    MOBILE_BASE = "mobile_base"  # A base is holonomic (forward, side, yaw) or differential.


class BodyActionSpace(ValueSpec):
    """A torso or mobile base: `dof` values of one `command`.

    `frame` is the frame that a base moves in, and a torso, which moves along its own
    joints, has none.
    """

    kind: Literal["body"] = "body"
    part: BodyPart
    dof: int = Field(ge=1)
    command: Command
    frame: Frame | None = None

    _reject_chunk_size = model_validator(mode="before")(_reject_chunk_size)

    @model_validator(mode="after")
    def _check_frame_fits_part(self) -> BodyActionSpace:
        """Require a frame for a mobile base and refuse one for a torso."""
        if self.part is BodyPart.MOBILE_BASE and self.frame is None:
            raise ValueError("a mobile base needs the frame that it moves in")
        if self.part is BodyPart.TORSO and self.frame is not None:
            raise ValueError("a torso moves along its joints and takes no frame")
        return self

    def expected_length(self) -> int:
        """Floats in one command: one per degree of freedom."""
        return self.dof


class ModeCut(StrEnum):
    """Where a mode slot's value cuts between its low and high side.

    robosuite's HybridMobileBase controller reads the raw value against 0
    (`BINARY_0`); RoboCasa365's gym wrapper snaps it at 0.5 first (`BINARY_0_5`).
    There the mode slot sets what each arm's delta is added to: the arm's previous target on
    the high side, its measured pose on the low side, while every part of the body
    still acts every step. -1 is on the low side of both cuts, and 1 on the high
    side. A `CompositeActionSpace` declares one, and `mode_is_high` reads an action
    against it.
    """

    BINARY_0 = "binary_0"  # The value is high when it is > 0.
    BINARY_0_5 = "binary_0.5"  # The value is high when it is >= 0.5.


# One group of a whole-body action: an arm, a torso or a mobile base.
ActionGroup = Annotated[
    EEActionSpace | JointActionSpace | BodyActionSpace,
    Field(discriminator="kind"),
]


class CompositeActionSpace(ValueSpec):
    """A whole-body action: each group's values back to back, then a mode slot if any.

    Arms reuse `EEActionSpace` and `JointActionSpace`, each with its own gripper and
    `arm_count`. `BodyActionSpace` covers the rest of a mobile robot: its torso and
    its base. The groups appear in the order that the robot reads them in.
    """

    kind: Literal["composite"] = "composite"
    groups: tuple[ActionGroup, ...] = Field(min_length=1)
    mode_slot: ModeCut | None = None

    _reject_chunk_size = model_validator(mode="before")(_reject_chunk_size)

    def expected_length(self) -> int:
        """Floats in a single action: every group, then the mode slot if declared."""
        width = sum(group.expected_length() for group in self.groups)
        return width + (1 if self.mode_slot is not None else 0)

    def group_kinds(self) -> list[str]:
        """Each group's kind, or its part for a body group: `ee`, `joint`, `torso`."""
        return [
            group.part.value if isinstance(group, BodyActionSpace) else group.kind
            for group in self.groups
        ]

    def group_names(self) -> list[str]:
        """Each group's kind, numbered when a kind appears more than once."""
        kinds = self.group_kinds()
        return [
            f"{kind}{kinds[:index].count(kind)}" if kinds.count(kind) > 1 else kind
            for index, kind in enumerate(kinds)
        ]

    def mode_is_high(self, action: Sequence[float]) -> bool:
        """True when `action`'s mode slot is on the high side of the declared cut."""
        if self.mode_slot is None:
            raise ValueError("this whole-body action declares no mode slot")
        self.validate_value(action)
        value = float(action[self.mode_index()])
        if self.mode_slot is ModeCut.BINARY_0:
            return value > 0.0
        if self.mode_slot is ModeCut.BINARY_0_5:
            return value >= 0.5
        return assert_never(self.mode_slot)

    def mode_index(self) -> int:
        """Where the mode slot sits in an action: after the last group."""
        if self.mode_slot is None:
            raise ValueError("this whole-body action declares no mode slot")
        return sum(group.expected_length() for group in self.groups)

    def group_slices(self) -> list[tuple[str, int, int]]:
        """Each group's name and [start, stop) in the action."""
        out, start = [], 0
        for name, group in zip(self.group_names(), self.groups, strict=True):
            stop = start + group.expected_length()
            out.append((name, start, stop))
            start = stop
        return out

    def slices(self) -> list[tuple[str, int, int]]:
        """Every slot range in the action in order: the groups, then the mode slot's."""
        out = self.group_slices()
        if self.mode_slot is not None:
            mode = self.mode_index()
            out.append(("mode_slot", mode, mode + 1))
        return out


ActionSpace = Annotated[
    EEActionSpace | JointActionSpace | UnifiedActionSpace | CompositeActionSpace,
    Field(discriminator="kind"),
]


__all__ = [
    "ActionGroup",
    "ActionSpace",
    "BodyActionSpace",
    "BodyPart",
    "Command",
    "CompositeActionSpace",
    "EEActionSpace",
    "JointActionSpace",
    "ModeCut",
    "UnifiedActionSpace",
]
