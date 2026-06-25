"""A robot's canonical action space and what it reports about itself.

An Embodiment owns one canonical action space (the single representation it is
defined in) plus its proprioception (the state it reports each step: end-effector
pose, joint angles, or both). It is reusable across benchmarks. The cameras and
task signals a benchmark adds live in `sensor.py` and `benchmark.py`.

The action space is one typed spec carrying its conventions: rotation format,
gripper polarity, frame, and delta-vs-absolute. Two action spaces are compatible
when equal, or when an adapter bridges them (see `adapter.py`, `check.py`).

The pieces live in submodules — `spec` (the shared value-spec base),
`proprioception`, and `action_space` — and are re-exported here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from manifold.core.embodiment.action_space import (
    ActionSpace,
    EEActionSpace,
    JointActionSpace,
    UnifiedActionSpace,
)
from manifold.core.embodiment.proprioception import (
    EEObservationSpec,
    GripperObservationSpec,
    JointObservationSpec,
    Proprioception,
)
from manifold.core.embodiment.spec import ValueSpec


class Embodiment(BaseModel):
    """A robot: its canonical action space and its proprioception.

    Reusable across benchmarks. A benchmark references an embodiment and adds
    sensors and task signals; a policy declares the action space it was trained
    to drive, which a check compares against this embodiment's `action`.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    action: ActionSpace
    proprioception: Proprioception


__all__ = [
    "ActionSpace",
    "EEActionSpace",
    "EEObservationSpec",
    "Embodiment",
    "GripperObservationSpec",
    "JointActionSpace",
    "JointObservationSpec",
    "Proprioception",
    "UnifiedActionSpace",
    "ValueSpec",
]
