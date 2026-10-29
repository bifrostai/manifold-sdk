"""A task suite a policy is run against.

A Benchmark is an embodiment placed in a task, plus the sensors it publishes. It
defines what a policy receives each step (proprioception from the embodiment, the
published sensors, and an instruction) and the canonical action space the policy
must drive (the embodiment's `action`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
from manifold.lib.compat import StrEnum, assert_never

if TYPE_CHECKING:
    from collections.abc import Mapping

# The image encodings carried in an `__image__` wrapper's `format` field.
ImageFormat = Literal["raw", "jpeg", "png"]


class GpuArchitecture(StrEnum):
    """A GPU architecture that CUDA code is compiled for.

    The values are the names that `nvcc` accepts in CUDA 12.9 and 13.4: `sm_` for native
    code or `compute_` for PTX, then a compute capability without its dot, then an
    optional `a` or `f` suffix.
    """

    SM_50 = "sm_50"
    SM_52 = "sm_52"
    SM_53 = "sm_53"
    SM_60 = "sm_60"
    SM_61 = "sm_61"
    SM_62 = "sm_62"
    SM_70 = "sm_70"
    SM_72 = "sm_72"
    SM_75 = "sm_75"
    SM_80 = "sm_80"
    SM_86 = "sm_86"
    SM_87 = "sm_87"
    SM_88 = "sm_88"
    SM_89 = "sm_89"
    SM_90 = "sm_90"
    SM_90A = "sm_90a"
    SM_100 = "sm_100"
    SM_100A = "sm_100a"
    SM_100F = "sm_100f"
    SM_101 = "sm_101"
    SM_101A = "sm_101a"
    SM_101F = "sm_101f"
    SM_103 = "sm_103"
    SM_103A = "sm_103a"
    SM_103F = "sm_103f"
    SM_107 = "sm_107"
    SM_107A = "sm_107a"
    SM_107F = "sm_107f"
    SM_110 = "sm_110"
    SM_110A = "sm_110a"
    SM_110F = "sm_110f"
    SM_120 = "sm_120"
    SM_120A = "sm_120a"
    SM_120F = "sm_120f"
    SM_121 = "sm_121"
    SM_121A = "sm_121a"
    SM_121F = "sm_121f"
    COMPUTE_50 = "compute_50"
    COMPUTE_52 = "compute_52"
    COMPUTE_53 = "compute_53"
    COMPUTE_60 = "compute_60"
    COMPUTE_61 = "compute_61"
    COMPUTE_62 = "compute_62"
    COMPUTE_70 = "compute_70"
    COMPUTE_72 = "compute_72"
    COMPUTE_75 = "compute_75"
    COMPUTE_80 = "compute_80"
    COMPUTE_86 = "compute_86"
    COMPUTE_87 = "compute_87"
    COMPUTE_88 = "compute_88"
    COMPUTE_89 = "compute_89"
    COMPUTE_90 = "compute_90"
    COMPUTE_90A = "compute_90a"
    COMPUTE_100 = "compute_100"
    COMPUTE_100A = "compute_100a"
    COMPUTE_100F = "compute_100f"
    COMPUTE_101 = "compute_101"
    COMPUTE_101A = "compute_101a"
    COMPUTE_101F = "compute_101f"
    COMPUTE_103 = "compute_103"
    COMPUTE_103A = "compute_103a"
    COMPUTE_103F = "compute_103f"
    COMPUTE_107 = "compute_107"
    COMPUTE_107A = "compute_107a"
    COMPUTE_107F = "compute_107f"
    COMPUTE_110 = "compute_110"
    COMPUTE_110A = "compute_110a"
    COMPUTE_110F = "compute_110f"
    COMPUTE_120 = "compute_120"
    COMPUTE_120A = "compute_120a"
    COMPUTE_120F = "compute_120f"
    COMPUTE_121 = "compute_121"
    COMPUTE_121A = "compute_121a"
    COMPUTE_121F = "compute_121f"


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

    `step_seconds`, `max_steps` and `image_format` state how the benchmark runs, so that
    a policy author can match them: the simulated time between two steps, the step limit
    of an episode, and the encoding of each camera image on its way to the policy. A
    benchmark's config states them under `run`, and the config copies them here. They
    are None when the benchmark does not state them, and `max_steps` is None when each
    task has its own limit or upstream ends episodes itself. Pairing does not compare
    them.

    `gpu_architectures` lists the GPU architectures that the benchmark's compiled CUDA
    code is built for, in the form that `nvcc --gpu-architecture` accepts. Native code,
    such as `sm_86`, runs on a GPU of the same major compute capability and the same or a
    higher minor one. PTX, such as `compute_120`, runs on a GPU of that compute capability
    or a higher one. A name with an `a` suffix runs only on that exact compute capability,
    and one with an `f` suffix only on GPUs of the same family. It is None when the
    benchmark does not declare the list. The benchmark does not send it to the policy.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    embodiment: Embodiment
    sensors: list[Camera] = Field(default_factory=list)
    instruction: bool = True
    action_range: ActionRange | None = None
    step_seconds: float | None = Field(default=None, gt=0)
    max_steps: int | None = Field(default=None, ge=1)
    image_format: ImageFormat | None = None
    gpu_architectures: tuple[GpuArchitecture, ...] | None = Field(default=None, min_length=1)

    @field_validator("gpu_architectures")
    @classmethod
    def _drop_repeated_gpu_architectures(
        cls, gpu_architectures: tuple[GpuArchitecture, ...] | None
    ) -> tuple[GpuArchitecture, ...] | None:
        """Keep the first of each repeated GPU architecture."""
        return tuple(dict.fromkeys(gpu_architectures)) if gpu_architectures is not None else None

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
