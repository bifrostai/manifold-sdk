"""What a robot reports about its own state: end-effector pose, joints, base, torso, extras."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from manifold.core.conventions import (
    Frame,
    GripperFormat,
    RotationFormat,
    ee_step_layout,
)
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

    Each arm contributes, in order: 3 floats for Cartesian position (xyz, meters),
    the rotation (3, 4, or 6 floats per `rotation`), and that arm's finger joint
    positions if `gripper` is set (`gripper.dim` floats, for example 2 for a
    parallel-jaw gripper). A value holds `arm_count` of those groups back to back.

    `arm_count` counts the arms. When `gripper` is set, each arm has a gripper.
    `gripper.dim` counts the fingers on a gripper. Those are three separate numbers,
    and none of them stands in for another. A bimanual robot with parallel-jaw
    grippers therefore reports 2 x 2 = 4 finger values, which come from two grippers
    of two fingers.
    """

    rotation: RotationFormat
    gripper: GripperObservationSpec | None = None
    arm_count: int = Field(default=1, ge=1)
    frame: Frame = Frame.WORLD

    def per_arm_length(self) -> int:
        """Floats one arm contributes: position, rotation, then its gripper's fingers."""
        pose_len = sum(ee_step_layout(self.rotation, None))
        return pose_len + (self.gripper.dim if self.gripper is not None else 0)

    def expected_length(self) -> int:
        """Floats in one EE value: every arm's position, rotation and finger qpos."""
        return self.per_arm_length() * self.arm_count


class JointObservationSpec(ValueSpec):
    """The joint angles the robot reports: `dof` angles across `arm_count` arms.

    The layout matches `JointActionSpace`. Each arm's `dof // arm_count` joints are
    followed by that arm's gripper, if `gripper` is set. The robot reports its state
    in the same layout that it accepts commands in.
    """

    dof: int = Field(ge=1)
    gripper: GripperFormat | None = None
    arm_count: int = Field(default=1, ge=1)

    def per_arm_length(self) -> int:
        """Floats one arm contributes: its joints, then its gripper if set."""
        return self.dof // self.arm_count + (1 if self.gripper else 0)

    def expected_length(self) -> int:
        """Floats in one joint value: every joint, then one gripper per arm if set."""
        return self.dof + (self.arm_count if self.gripper else 0)

    @model_validator(mode="after")
    def _check_dof_divides_across_arms(self) -> JointObservationSpec:
        """Refuse a `dof` that cannot split into `arm_count` equal arms."""
        if self.dof % self.arm_count:
            raise ValueError(f"dof={self.dof} does not divide into {self.arm_count} equal arms")
        return self


class BasePoseObservationSpec(ValueSpec):
    """The pose of a mobile robot's base.

    One value is 3 floats of position (xyz, meters), then the rotation (3, 4, or 6
    floats per `rotation`). There is one base, so there is no `arm_count`. `frame`
    is the frame the pose is expressed in, usually the world.
    """

    rotation: RotationFormat
    frame: Frame = Frame.WORLD

    def expected_length(self) -> int:
        """Floats in one base pose: the position, then the rotation."""
        return sum(ee_step_layout(self.rotation, None))


class ExtraObservationSpec(ValueSpec):
    """A state value no typed field holds, such as finger velocities: `dim` floats.

    `name` is the `Observation.state` key it travels under. Matching compares `name`
    and `dim`; `meaning` describes the values for a reader. A value a typed field can
    hold, such as a base pose, belongs in that field.
    """

    name: str = Field(min_length=1)
    dim: int = Field(ge=1)
    meaning: str = Field(min_length=1)

    def expected_length(self) -> int:
        """Floats in one value: `dim`."""
        return self.dim


class Proprioception(BaseModel):
    """What the robot reports about its own state each step.

    Set `ee_pose`, `joint_pos`, `base_pose`, `torso_pos`, or any combination. All may
    be left unset for a robot that reports no proprioception. `torso_pos` holds the
    joints between a mobile base and its arms, such as a lift, with no gripper. Each
    field's name is the key its value travels under in `Observation.state`. Gripper
    finger state, when reported, is nested in the end-effector value
    (`EEObservationSpec.gripper`), mirroring the action side where the gripper is a
    field of `EEActionSpace`. `extras` lists, by name, any other value the robot
    reports.
    """

    model_config = ConfigDict(frozen=True)

    ee_pose: EEObservationSpec | None = None
    joint_pos: JointObservationSpec | None = None
    base_pose: BasePoseObservationSpec | None = None
    torso_pos: JointObservationSpec | None = None
    extras: tuple[ExtraObservationSpec, ...] = ()

    @classmethod
    def spec_fields(cls) -> tuple[str, ...]:
        """The typed fields, each holding one spec or None: every field but `extras`."""
        return tuple(name for name in cls.model_fields if name != "extras")

    def extra(self, name: str) -> ExtraObservationSpec | None:
        """The extra named `name`, or None if this declares none."""
        return next((extra for extra in self.extras if extra.name == name), None)

    @classmethod
    def pose_fields(cls) -> tuple[str, ...]:
        """The fields whose spec is a pose, a position then a rotation."""
        return ("ee_pose", "base_pose")

    @model_validator(mode="after")
    def _check_torso_is_one_chain(self) -> Proprioception:
        """Refuse a torso with a gripper or more than one arm: it holds joints only."""
        torso = self.torso_pos
        if torso is not None and (torso.gripper is not None or torso.arm_count != 1):
            raise ValueError("torso_pos holds the torso joints only: no gripper, one chain")
        return self

    @model_validator(mode="after")
    def _check_extras_are_named_once(self) -> Proprioception:
        """Refuse an extra named twice, or named after a typed field."""
        names = [extra.name for extra in self.extras]
        for name in names:
            if names.count(name) > 1:
                raise ValueError(f"extra {name!r} is declared twice")
            if name in self.spec_fields():
                raise ValueError(f"extra {name!r} is a typed field: declare it there")
        return self

    def first_unmet(self, available: Proprioception) -> str | None:
        """The first component this needs that `available` does not match, or None.

        Used to check that a benchmark publishes the proprioception a policy
        consumes: `self` is what the policy requires, `available` is what is offered.
        The `ee_pose` comparison includes its nested gripper, so a policy that
        needs gripper qpos is unmet against a benchmark whose ee_pose lacks it. An
        extra is met by an extra of the same name and width.
        """
        for name in self.spec_fields():
            needed = getattr(self, name)
            if needed is not None and getattr(available, name) != needed:
                return name
        for extra in self.extras:
            offered = available.extra(extra.name)
            if offered is None or offered.dim != extra.dim:
                return f"extras.{extra.name}"
        return None


__all__ = [
    "BasePoseObservationSpec",
    "EEObservationSpec",
    "ExtraObservationSpec",
    "GripperObservationSpec",
    "JointObservationSpec",
    "Proprioception",
]
