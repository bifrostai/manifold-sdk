"""What a policy declares about itself, so a pairing can be checked.

A PolicySignature is the policy side of a pairing: the action space it was trained
to emit and the composite observation it consumes (proprioception, the cameras
with their conventions, and whether it needs a language instruction).
`check_compatibility` compares this against a benchmark before any rollout.

This declares the signature only; the serving harness lives in
`manifold.recipes.serving`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from manifold.core.embodiment import ActionSpace, Proprioception
from manifold.core.observation_space import ObservationSpace
from manifold.core.sensor import Camera


class PolicySignature(BaseModel):
    """The policy side of a pairing.

    `action_space` is the space the policy natively emits; it is compatible with
    a benchmark when it equals the benchmark embodiment's canonical action space
    or a supplied adapter bridges it. The consumed observation is declared as its
    channels — the proprioception, the cameras (each a typed `Camera` carrying the
    conventions the policy consumes: orientation, channel_order, shape, dtype,
    modality), and whether it needs an instruction; the benchmark must publish a
    form that supplies every consumed channel, or an observation adapter bridges
    it. `observation_space` composes those channels into the same shape a
    `Benchmark` publishes, so matching stays at the channel seam.

    This models only the *encoding* the policy consumes and emits — shape and
    conventions. **Normalization** (the per-field statistics the policy was
    trained with) is not modeled here: it is policy-internal, so the
    serving wrapper must normalize inputs and un-normalize outputs with the stats
    the checkpoint expects. A mismatch — notably the right scheme but the wrong
    stats — is not caught by `check_compatibility` and fails silently. See
    ADR-0001.
    """

    model_config = ConfigDict(frozen=True)

    action_space: ActionSpace
    proprioception: Proprioception = Field(default_factory=Proprioception)
    cameras: list[Camera] = Field(default_factory=list)
    instruction: bool = True

    @property
    def observation_space(self) -> ObservationSpace:
        """The composite observation this policy consumes.

        A view over the channel fields, mirroring `Benchmark.observation_space`:
        the fields stay the source of truth, this is not stored.
        """
        return ObservationSpace(
            proprioception=self.proprioception,
            cameras=tuple(self.cameras),
            instruction=self.instruction,
        )


__all__ = ["PolicySignature"]
