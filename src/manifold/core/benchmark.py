"""A task suite a policy is run against.

A Benchmark is an embodiment placed in a task, plus the sensors it publishes. It
defines what a policy receives each step (proprioception from the embodiment, the
published sensors, and an instruction) and the canonical action space the policy
must drive (the embodiment's `action`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from manifold.core.conventions import GripperFormat
from manifold.core.embodiment import (
    ActionSpace,
    CompositeActionSpace,
    EEActionSpace,
    Embodiment,
    JointActionSpace,
    UnifiedActionSpace,
)
from manifold.core.names import CameraName
from manifold.core.observation_space import ObservationSpace
from manifold.core.sensor import Camera
from manifold.lib.compat import assert_never

if TYPE_CHECKING:
    from collections.abc import Mapping

# The form of a CUDA architecture name: `sm_` for native code or `compute_` for PTX, the
# compute capability without its dot, and an optional `a` or `f` feature suffix.
_CUDA_ARCHITECTURE_PATTERN = r"^(sm|compute)_[1-9][0-9]{1,2}[af]?$"


class ActionRange(BaseModel):
    """The values that a benchmark accepts in one action, with one entry for each value.

    `low` and `high` bound each value. `scale` is how far one unit of each value moves
    the robot, in metres for a position and in radians for a rotation, for a benchmark
    that scales the action that it receives. A value without a unit, such as a gripper
    command, has a scale of 1. `scale` is None when the benchmark reads each value in
    metres and radians already.

    For example, robosuite's OSC controller clips each value to -1 to 1, and one unit
    moves the arm 0.05 m or turns it 0.5 rad.
    """

    model_config = ConfigDict(frozen=True)

    low: tuple[float, ...]
    high: tuple[float, ...]
    scale: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def _check_entries(self) -> ActionRange:
        """Require one entry for each value, and each low bound at or below its high bound."""
        if len(self.low) != len(self.high):
            raise ValueError(f"low has {len(self.low)} values and high has {len(self.high)}")
        if self.scale is not None and len(self.scale) != len(self.low):
            raise ValueError(f"scale has {len(self.scale)} values and low has {len(self.low)}")
        above = [
            index for index, (lo, hi) in enumerate(zip(self.low, self.high, strict=True)) if lo > hi
        ]
        if above:
            raise ValueError(f"low is above high at the values {above}")
        return self


class Benchmark(BaseModel):
    """An embodiment in a task, with the sensors it publishes.

    `instruction` records whether the benchmark publishes a language instruction.
    A policy that needs one is incompatible with a benchmark that publishes none.

    `action_range` declares the values that the embodiment's action accepts, so that a
    policy author can scale the policy's output to them. It is None when the benchmark
    declares no range. Pairing does not compare it.

    `cuda_architectures` lists the GPU architectures that the benchmark's compiled CUDA
    code is built for, in the form that `nvcc --gpu-architecture` accepts. Native code,
    such as `sm_86`, runs on a GPU of the same major compute capability and the same or a
    higher minor one. PTX, such as `compute_120`, runs on a GPU of that compute capability
    or a higher one. A name with an `a` suffix runs only on that exact compute capability,
    and one with an `f` suffix only on GPUs of the same family. It is None when the
    benchmark does not declare the list. Pairing does not compare it.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    embodiment: Embodiment
    sensors: list[Camera] = Field(default_factory=list)
    instruction: bool = True
    action_range: ActionRange | None = None
    cuda_architectures: (
        tuple[Annotated[str, StringConstraints(pattern=_CUDA_ARCHITECTURE_PATTERN)], ...] | None
    ) = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _check_cuda_architectures_are_unique(self) -> Benchmark:
        """Require each CUDA architecture to appear once."""
        names = self.cuda_architectures or ()
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"cuda_architectures lists {repeated} more than once")
        return self

    @model_validator(mode="after")
    def _check_action_range_length(self) -> Benchmark:
        """Require the action range to have one entry for each value of the action."""
        expected = self.embodiment.action.expected_length()
        if self.action_range is not None and len(self.action_range.low) != expected:
            raise ValueError(
                f"action_range has {len(self.action_range.low)} values and the action "
                f"has {expected}"
            )
        return self

    @classmethod
    def from_received(cls, payload: Mapping[str, Any]) -> Benchmark:
        """The benchmark a peer sent, without the cameras this SDK does not list.

        A peer built against a newer SDK may publish a camera named after a viewpoint
        this one does not list. No policy on this SDK can read that camera, so it is
        dropped rather than failing the parse. An unlisted embodiment still fails: a
        robot this SDK does not know cannot be one its policy was built for.
        """
        listed = {name.value for name in CameraName}
        sensors = [sensor for sensor in payload.get("sensors", []) if sensor.get("name") in listed]
        return cls.model_validate({**payload, "sensors": sensors})

    @property
    def observation_space(self) -> ObservationSpace:
        """The composite observation the benchmark publishes.

        Derived from the embodiment's proprioception, the published sensors, and
        whether an instruction is published — the single channel-aggregate a check
        folds the observation chain over. The underlying fields stay the source of
        truth; this is a view, not duplicate storage.
        """
        return ObservationSpace(
            proprioception=self.embodiment.proprioception,
            cameras=tuple(self.sensors),
            instruction=self.instruction,
        )

    @property
    def action_space(self) -> ActionSpace:
        """The canonical action space the benchmark expects a policy to drive.

        A view onto the embodiment's `action`, exposed alongside
        `observation_space` so a benchmark presents the same `(observation_space,
        action_space)` interface a `PolicySignature` declares — the two are what
        `check_compatibility` pairs.
        """
        return self.embodiment.action

    @property
    def gripper_format(self) -> GripperFormat:
        """The benchmark's gripper format, for a gripper adapter target.

        "Read a benchmark's gripper convention" is a general query a gripper
        adapter makes when it builds its chain, so it lives here on `Benchmark`
        rather than in any one caller. The action space is a union (EE / joint /
        unified / composite); this checks it exhaustively and returns whichever
        variant's `gripper` applies, raising if the benchmark has no gripper to
        target rather than silently mis-typing it. A composite returns the gripper
        that its arm groups share, and raises if they do not share one.

        Raises:
            TypeError: If the action space has no gripper (an EE or joint space
                with `gripper=None`, a unified space with no payload or a payload
                with `gripper=None`, or a composite with no arm gripper), or if a
                composite's arm groups declare different grippers.
        """
        action = self.action_space
        if isinstance(action, (EEActionSpace, JointActionSpace)):
            gripper = action.gripper
        elif isinstance(action, UnifiedActionSpace):
            gripper = action.payload.gripper if action.payload is not None else None
        elif isinstance(action, CompositeActionSpace):
            grippers = [
                group.gripper
                for group in action.groups
                if isinstance(group, (EEActionSpace, JointActionSpace))
            ]
            # An arm group without a gripper does not share the others' gripper.
            if len(set(grippers)) > 1:
                raise TypeError(
                    f"{self!r} has arm groups that do not share one gripper: {grippers}"
                )
            gripper = grippers[0] if grippers else None
        else:
            assert_never(action)
        if gripper is None:
            raise TypeError(f"{self!r} has no gripper action to target")
        return gripper


__all__ = [
    "ActionRange",
    "Benchmark",
]
