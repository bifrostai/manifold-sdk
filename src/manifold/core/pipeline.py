"""A pipeline: the ordered transform chains that bridge a pairing.

A pairing is a given policy against a given benchmark. A pipeline is what makes
it work: an ordered chain of `ObservationAdapter`s that carry the benchmark's
observation toward the policy's form, and an ordered chain of `ActionAdapter`s
that carry the policy's action toward the benchmark's form. Either chain may be
empty when that side already matches.

A pipeline is a plain value a caller assembles (or a resolver in `recipes/`
builds). `check_compatibility` validates it; `verify` runs data through it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from manifold.core.adapter import ActionAdapter, Adapter, ObservationAdapter
from manifold.core.state import DEFAULT_LANE, LaneKey, PipelineState
from manifold.core.values import Action

if TYPE_CHECKING:
    from pydantic import BaseModel

    from manifold.core.native_layout import NativeLayout
    from manifold.core.observation_space import ObservationSpace
    from manifold.core.values import Observation


class _StatefulAdapter(Protocol):
    """The shape of a stateful adapter's `adapt`: it accepts the `state` keyword.

    `Pipeline._adapt` views a stateful adapter through this protocol to pass its
    per-episode slice (the `state_key is not None` check is the runtime guarantee
    the adapter accepts the keyword). See `core/adapter` for the mechanism.
    """

    def adapt(self, data: Any, /, *, source: BaseModel, state: dict[str, Any]) -> Any: ...


class Pack(Protocol):
    """The observation-side packing adapter's surface the pipeline drives.

    Typed structurally (not by importing `adapters/packing`) so `core/pipeline`
    does not depend on the adapter catalog. `produce` yields the model's native
    layout spec; `adapt` produces the model's native tensor dict.
    """

    def produce(self, source: BaseModel) -> BaseModel:
        """Return the model's native layout spec derived from `source`."""
        ...

    def adapt(self, observation: Any, /, *, source: BaseModel) -> dict[str, Any]:
        """Pack `observation` into the model's native tensor dict."""
        ...


class Unpack(Protocol):
    """The action-side packing adapter's surface the pipeline drives.

    Reads one step of a raw model-output chunk into an SDK `Action`. `layout` is the
    output `NativeLayout` the adapter reads — exposed so a caller can fold the unpack
    phase without separately threading the layout it already declared (the serve loop
    does exactly this).
    """

    layout: NativeLayout

    def adapt(self, raw_output: Any, /, *, source: BaseModel, step: int) -> Action:
        """Read step `step` of `raw_output` into an `Action`."""
        ...


class StatefulStateMissing(RuntimeError):
    """A stateful adapter (`state_key` set; see `core/adapter`) ran with no `PipelineState`.

    A hard error, not a no-op: running with `state=None` would silently drop the
    history or fall back to a shared buffer — the cross-episode leak the ADR records.
    A missed state object must be loud, not a silent contamination.
    """

    def __init__(self, adapter: Adapter, key: str) -> None:
        super().__init__(
            f"{type(adapter).__name__} is stateful (state_key={key!r}): "
            f"apply_* was called without a PipelineState; pass one to run it."
        )
        self.adapter = adapter
        self.key = key


class NoPackingPhase(RuntimeError):
    """`apply_pack`/`apply_unpack` was called on a pipeline with no packing phase.

    The packing phase is optional, so calling an apply method for an unset phase is
    a caller bug (it should gate on `pipeline.pack`/`pipeline.unpack`); a loud error
    rather than a silent no-op handing back an empty dict.
    """

    def __init__(self, method: str, field: str) -> None:
        super().__init__(
            f"Pipeline.{method} needs a {field} adapter, but this pipeline has none "
            f"(pipeline.{field} is None); set it or gate on it before calling."
        )


@dataclass(frozen=True)
class Pipeline:
    """The observation and action transform chains for a pairing.

    `observation` carries the benchmark's published form toward the policy's;
    `action` carries the policy's emitted action toward the benchmark's. Stored as
    tuples so the pipeline stays immutable.

    `pack` and `unpack` are the optional native-packing phase — phase two, where
    the seam crosses from the channel-typed `Observation` to the model's native
    tensor dict and back (ADR-0001, "The pipeline reaches the model"). When both are
    None the pipeline behaves identically to one without them. They are separate from
    the convention chains: matching (`check_compatibility`) stays at the channel seam
    and never folds packing in; only `verify` and serving exercise the packing phase.
    """

    observation: Sequence[ObservationAdapter] = ()
    action: Sequence[ActionAdapter] = ()
    pack: Pack | None = None
    unpack: Unpack | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation", tuple(self.observation))
        object.__setattr__(self, "action", tuple(self.action))
        # Both-or-neither: the packing phase is a round trip. `_step` gates on
        # `pack and unpack` and falls through to `session.infer` when only one is
        # set — and the no-packing session raises NotImplementedError. `verify` is
        # also misleadingly lenient (it probes `pack or unpack`), so a one-sided
        # pipeline can pass the gate yet die at serve time. Reject it at construction
        # so the contradiction surfaces where the pipeline is built, not at runtime.
        if (self.pack is None) != (self.unpack is None):
            present, missing = ("pack", "unpack") if self.pack is not None else ("unpack", "pack")
            raise ValueError(
                f"Pipeline has a {present} adapter but no {missing}: the native-packing "
                f"phase is a round trip and must be set both-or-neither. Set {missing} too, "
                f"or clear {present}."
            )

    def apply_observation(
        self,
        observation: Observation,
        *,
        source: ObservationSpace,
        state: PipelineState | None = None,
        lane: LaneKey = DEFAULT_LANE,
    ) -> Observation:
        """Fold the observation chain over a live observation, returning the result.

        The walk threads the layout spec through the chain (starting at `source`,
        advancing by `adapter.produce(spec)`), so each `adapt` sees the spec its
        input is laid out under. A pure transform: it assumes the pipeline was
        already validated (e.g. via `check_compatibility`) and performs no validation.

        State threading follows the `state_key` contract (see `core/adapter`): a
        stateless chain ignores `state`; a stateful adapter gets its `(lane, key)`
        slice, and a missing `state` raises `StatefulStateMissing`. `source` is the
        observation spec the benchmark publishes; `lane` selects which rollout's
        state to use. Returns the observation in the form the policy consumes.
        """
        spec: BaseModel = source
        for adapter in self.observation:
            observation = self._adapt(adapter, observation, source=spec, state=state, lane=lane)
            spec = adapter.produce(spec)
        return observation

    def apply_action(
        self,
        action: Action,
        *,
        source: BaseModel,
        state: PipelineState | None = None,
        lane: LaneKey = DEFAULT_LANE,
    ) -> Action:
        """Fold the action chain over a live action, returning the result.

        The action-side twin of `apply_observation`: same spec-threading walk, same
        no-validation contract, same `state_key` state threading. `source` is the
        policy's action space. Returns the action in the form the benchmark expects.
        """
        values = action.values
        spec: BaseModel = source
        for adapter in self.action:
            values = self._adapt(adapter, values, source=spec, state=state, lane=lane)
            spec = adapter.produce(spec)
        return Action.from_array(values)

    def apply_pack(
        self, observation: Observation, *, source: ObservationSpace
    ) -> dict[str, np.ndarray]:
        """Cross the seam: pack a (policy-form) observation into the model tensor dict.

        Phase two of the observation side. `source` is the spec the observation is
        laid out under at the phase boundary — the policy's consumed form (the spec
        the observation chain produced). No validation, mirroring `apply_observation`.
        """
        if self.pack is None:
            raise NoPackingPhase("apply_pack", "pack")
        return self.pack.adapt(observation, source=source)

    def apply_unpack(
        self, raw_output: Any, *, source: BaseModel | None = None, step: int = 0
    ) -> Action:
        """Cross the seam back: read step `step` of the model's raw output into an Action.

        Phase two of the action side, run before the convention `action` chain. The
        `Action` it produces is laid out under the policy's action space; the
        convention adapters then carry it toward the benchmark. `source` defaults to
        the layout the unpack adapter declared, so the serve loop need not re-thread it.
        """
        if self.unpack is None:
            raise NoPackingPhase("apply_unpack", "unpack")
        if source is None:
            source = self.unpack.layout
        return self.unpack.adapt(raw_output, source=source, step=step)

    @staticmethod
    def _adapt(
        adapter: Adapter,
        data: Any,
        *,
        source: BaseModel,
        state: PipelineState | None,
        lane: LaneKey,
    ) -> Any:
        """Dispatch one adapter on its `state_key`: stateless or stateful path.

        See `core/adapter` for the contract; a missing `PipelineState` for a
        stateful adapter raises `StatefulStateMissing` rather than running without
        history.
        """
        key = adapter.state_key
        if key is None:
            return adapter.adapt(data, source=source)
        if state is None:
            raise StatefulStateMissing(adapter, key)
        slot = state.slice_for(lane=lane, key=key)
        # View through the stateful protocol to pass the slice (the base
        # `Adapter.adapt` does not declare the keyword; the state_key check guarantees
        # this adapter accepts it).
        return cast("_StatefulAdapter", adapter).adapt(data, source=source, state=slot)


__all__ = ["NoPackingPhase", "Pack", "Pipeline", "StatefulStateMissing", "Unpack"]
