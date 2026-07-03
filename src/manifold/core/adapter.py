"""Typed adapters on both sides of the policy-benchmark seam.

The seam is bidirectional. An `ObservationAdapter` is a pre-processor: it carries
the benchmark's published observation form toward the form the policy consumes.
An `ActionAdapter` is a post-processor: it carries the policy's emitted action
toward the form the benchmark expects. A `Pipeline` (see `pipeline.py`) is an
ordered chain of each.

An adapter is **unary and typed**, which is what lets a chain be checked before
it runs. It declares the spec type it accepts (`from_spec`) and the type it
produces (`to_spec`); `applies` says whether it handles a concrete spec;
`produce` returns the concrete spec that results; `adapt` maps the data. Folding
`produce` along a chain yields each intermediate spec, so `check_compatibility`
can prove the chain reaches the benchmark's form from the policy's — from specs
alone, no rollout. Concrete adapters may assume `applies(source)` is True
before `produce` or `adapt` is called; the caller (the check, or a resolver)
guarantees it.

An adapter is a plain value. Core does not hold a registry and registers nothing
at import time: a caller assembles the adapters it selects into a pipeline (or a
resolver in `recipes/` does). The adapter contract is the public surface third
parties implement.

The three required class attributes (`from_spec`, `to_spec`, `lossless`) are
declared as bare `ClassVar` annotations without defaults, and a concrete adapter
sets each as a class attribute. This is a documented convention, not a statically
enforced one: `ty` does not flag a subclass that leaves a bare `ClassVar` unset,
so an adapter that omits one is caught at runtime on first access (an
`AttributeError`) rather than at definition time. We accept that first-use
failure in exchange for keeping the contract a plain declaration with no
metaclass or `__init_subclass__` machinery.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel


class Adapter(ABC):
    """A typed, unary spec adapter: the shared shape of both adapter kinds.

    `from_spec`/`to_spec` are the spec classes it bridges, matched by exact type.
    `lossless` is True for a representation conversion (safe to apply
    automatically) and False for a lossy one (a caller opts in).
    """

    from_spec: ClassVar[type[BaseModel]]
    to_spec: ClassVar[type[BaseModel]]
    lossless: ClassVar[bool]

    # The state slice this adapter touches, or None for a stateless one (the
    # default — every existing adapter is stateless and unchanged). A label, not
    # a permission model: it identifies the per-`(lane, key)` bag `PipelineState`
    # passes to the adapter, and `Pipeline.apply_*` dispatches on it. None ⇒ the
    # adapter keeps the stateless `adapt(self, x, *, source)` signature; a set label
    # ⇒ the adapter takes an extra `state` bag: `adapt(self, x, *, source, state)`.
    # Collisions on the same label are intentional sharing the author owns. Unlike
    # the three required attributes above, this one has a default, so it is
    # optional — it never breaks an existing subclass; only a stateful adapter
    # overrides it.
    state_key: ClassVar[str | None] = None

    @abstractmethod
    def applies(self, source: BaseModel) -> bool:
        """Whether this adapter applies to `source` (a concrete `from_spec`).

        Must return False once `source` is already in the transformed form, so an
        adapter fires at most once per chain — the chain's termination invariant.
        """

    @abstractmethod
    def produce(self, source: BaseModel) -> BaseModel:
        """The concrete `to_spec` spec that results from transforming `source`."""

    @abstractmethod
    def adapt(self, data: Any, /, *, source: BaseModel) -> Any:
        """Map a value laid out under `source` to one under `produce(source)`.

        The value is positional-only so a subclass may name it `values` (action) or
        `observation` (observation). A stateless adapter implements exactly this
        signature. A stateful adapter (one declaring a `state_key`) adds an optional
        `state` keyword — its per-episode bag, which `Pipeline.apply_*` supplies.
        Adding an optional keyword is a compatible override; `Pipeline._adapt` passes
        `state` only to a stateful adapter, so a stateless one never sees it.
        """


class ActionAdapter(Adapter, ABC):
    """One step on the action side: policy action toward benchmark action.

    `from_spec`/`to_spec` are action-space specs (e.g. `EEActionSpace`).
    """


class ObservationAdapter(Adapter, ABC):
    """One step on the observation side: benchmark obs form toward policy obs form.

    `from_spec`/`to_spec` are the composite `ObservationSpace`. An adapter
    rewrites one channel (proprioception or a named camera) and passes the rest
    through, so a chain of single-channel adapters composes over the aggregate.
    """


__all__ = [
    "ActionAdapter",
    "Adapter",
    "ObservationAdapter",
]
