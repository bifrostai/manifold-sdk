"""The values that pass between a benchmark and a policy each step.

Observation is what the benchmark sends; Action is what the policy sends back.
They are plain frozen dataclasses, not pydantic models: they are built and
discarded every step, so they do not validate per step. The specs that
describe their shapes live in the pydantic models elsewhere in `core/`, and a
check reads those.

The observation is decomposed, not a single flat bag: proprioception under
`state` keyed by role ("ee_pose", "joint_pos"), exteroception under `sensors`
keyed by sensor name ("agentview", "wrist"), and the instruction on its own. This
mirrors the primitives: the embodiment owns the state, the benchmark owns the
sensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# `eq=False`: these hold numpy arrays, so the generated __eq__ would return an
# array, not a bool, and break `if obs == other`. Identity equality is correct
# for per-step values that are built and discarded.
@dataclass(frozen=True, eq=False)
class Observation:
    """One observation from the benchmark.

    `state` holds proprioception by role, `sensors` holds camera arrays by sensor
    name, and `instruction` is the language task instruction if the benchmark
    publishes one.
    """

    state: dict[str, np.ndarray] = field(default_factory=dict)
    sensors: dict[str, np.ndarray] = field(default_factory=dict)
    instruction: str | None = None

    @classmethod
    def example(cls) -> Observation:
        """An empty observation, for warming up inference before a run."""
        return cls()


@dataclass(frozen=True, eq=False)
class Action:
    """One action the policy returns, laid out per the embodiment's action space.

    `values` is a flat (or chunk-leading) array whose length matches the
    benchmark's action spec. The runtime packs it onto the wire; the benchmark
    unpacks it per its own spec.
    """

    values: np.ndarray

    @classmethod
    def from_array(cls, values: np.ndarray | list[float]) -> Action:
        """Build an Action from a sequence, stored as float32."""
        return cls(values=np.asarray(values, dtype=np.float32))


__all__ = ["Action", "Observation"]
