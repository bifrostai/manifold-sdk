"""A task suite a policy is run against.

A Benchmark is an embodiment placed in a task, plus the sensors it publishes. It
defines what a policy receives each step (proprioception from the embodiment, the
published sensors, and an instruction) and the canonical action space the policy
must drive (the embodiment's `action`).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from manifold.core.conventions import GripperFormat
from manifold.core.embodiment import (
    ActionSpace,
    EEActionSpace,
    Embodiment,
    JointActionSpace,
    UnifiedActionSpace,
)
from manifold.core.observation_space import ObservationSpace
from manifold.core.sensor import Camera
from manifold.lib.compat import assert_never


class Benchmark(BaseModel):
    """An embodiment in a task, with the sensors it publishes.

    `instruction` records whether the benchmark publishes a language instruction.
    A policy that needs one is incompatible with a benchmark that publishes none.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    embodiment: Embodiment
    sensors: list[Camera] = Field(default_factory=list)
    instruction: bool = True

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
        unified); this checks it exhaustively and returns whichever variant's
        `gripper` applies, raising if the benchmark has no gripper to target
        rather than silently mis-typing it.

        Raises:
            TypeError: If the action space has no gripper — an EE or joint space
                with `gripper=None`, or a unified space with no payload or a
                payload with `gripper=None`.
        """
        action = self.action_space
        if isinstance(action, (EEActionSpace, JointActionSpace)):
            gripper = action.gripper
        elif isinstance(action, UnifiedActionSpace):
            gripper = action.payload.gripper if action.payload is not None else None
        else:
            assert_never(action)
        if gripper is None:
            raise TypeError(f"{self!r} has no gripper action to target")
        return gripper


__all__ = ["Benchmark"]
