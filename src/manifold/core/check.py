"""Judge whether a policy can run a benchmark, from their specs, before any rollout.

`check_compatibility` compares a PolicySignature against a Benchmark, given a
pipeline, and returns a Report: whether they are compatible, and if not, why. It
proves two things:

1. Observation: folding the pipeline's observation chain over the benchmark's
   published observation spec yields a form that supplies every channel the
   policy consumes — proprioception, each camera (with matching conventions),
   and an instruction if required. This single check subsumes the former
   separate sensor and instruction checks.
2. Action: folding the pipeline's action chain over the policy's action space
   reaches the benchmark embodiment's canonical action space.

The pipeline is supplied by the caller; `check_compatibility` does not decide
pairing strategy and does not search. A pass means the declared contracts are
compatible. It does not promise the policy performs well: whether the policy is
in-distribution for this benchmark is empirical, and only a rollout can show it.
`check_compatibility` reads specs; it runs nothing. To confirm a pairing by
running synthetic data through the pipeline, see `verify`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel

from manifold.core.adapter import ActionAdapter, ObservationAdapter
from manifold.core.benchmark import Benchmark
from manifold.core.observation_space import ObservationSpace
from manifold.core.pipeline import Pipeline
from manifold.core.policy import PolicySignature
from manifold.lib.compat import StrEnum


def _check_action(
    policy_action: BaseModel,
    bench_action: BaseModel,
    chain: Sequence[ActionAdapter],
) -> str | None:
    current = policy_action
    for adapter in chain:
        if type(current) is not adapter.from_spec or not adapter.applies(current):
            return f"action: {type(adapter).__name__} cannot transform {type(current).__name__}"
        current = adapter.produce(current)
    if current == bench_action:
        return None
    return (
        f"action space mismatch: the pipeline yields {type(current).__name__}"
        f"({_field_summary(current)}), benchmark expects {type(bench_action).__name__}"
        f"({_field_summary(bench_action)})"
    )


def _check_observation(
    bench_obs: ObservationSpace,
    policy_consumes: ObservationSpace,
    chain: Sequence[ObservationAdapter],
) -> str | None:
    current: BaseModel = bench_obs
    for adapter in chain:
        if type(current) is not adapter.from_spec or not adapter.applies(current):
            return (
                f"observation: {type(adapter).__name__} cannot transform {type(current).__name__}"
            )
        current = adapter.produce(current)
    if not isinstance(current, ObservationSpace):
        return (
            f"observation: the pipeline produced {type(current).__name__}, "
            f"expected ObservationSpace"
        )
    unmet = policy_consumes.first_unmet(current)
    if unmet is not None:
        return f"observation: {unmet}"
    return None


def _field_summary(spec: BaseModel) -> str:
    return ", ".join(f"{name}={value!r}" for name, value in spec.model_dump(mode="json").items())


class Compatibility(StrEnum):
    """How a policy relates to a benchmark."""

    COMPATIBLE = "compatible"  # forms already match; no pipeline needed.
    COMPATIBLE_VIA_PIPELINE = "compatible_via_pipeline"  # the supplied pipeline bridges them.
    INCOMPATIBLE = "incompatible"  # the supplied pipeline does not bridge the gap.


@dataclass(frozen=True)
class Report:
    """The result of a compatibility check.

    `reasons` lists every mismatch found, and is non-empty exactly when `status`
    is INCOMPATIBLE. `lossless` is set exactly when a pipeline was needed
    (COMPATIBLE_VIA_PIPELINE), and says whether every adapter in it is a
    lossless representation conversion (safe to apply automatically) or at least
    one is lossy (a caller opted in by supplying it). `ok` is True unless the
    pairing is incompatible.
    """

    status: Compatibility
    reasons: list[str] = field(default_factory=list)
    lossless: bool | None = None

    def __post_init__(self) -> None:
        if (self.status is Compatibility.INCOMPATIBLE) != bool(self.reasons):
            raise ValueError("reasons must be non-empty exactly when status is incompatible")
        if (self.status is Compatibility.COMPATIBLE_VIA_PIPELINE) != (self.lossless is not None):
            raise ValueError("lossless must be set exactly when status is compatible_via_pipeline")

    @property
    def ok(self) -> bool:
        """True unless the pairing is incompatible."""
        return self.status is not Compatibility.INCOMPATIBLE


def check_compatibility(
    policy: PolicySignature,
    benchmark: Benchmark,
    pipeline: Pipeline | None = None,
) -> Report:
    """Compare a policy against a benchmark, given a pipeline, and report compatibility."""
    if pipeline is None:
        pipeline = Pipeline()
    candidate_reasons = (
        _check_observation(
            benchmark.observation_space,
            policy.observation_space,
            pipeline.observation,
        ),
        _check_action(policy.action_space, benchmark.embodiment.action, pipeline.action),
    )
    reasons = [reason for reason in candidate_reasons if reason is not None]
    if reasons:
        return Report(status=Compatibility.INCOMPATIBLE, reasons=reasons)
    transforms = (*pipeline.observation, *pipeline.action)
    if not transforms:
        return Report(status=Compatibility.COMPATIBLE)
    return Report(
        status=Compatibility.COMPATIBLE_VIA_PIPELINE,
        lossless=all(transform.lossless for transform in transforms),
    )


__all__ = ["Compatibility", "Report", "check_compatibility"]
