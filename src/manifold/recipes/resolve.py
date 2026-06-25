"""Find a pipeline that bridges a pairing, by searching a supplied pool.

`resolve` is a strategy: given the adapters a caller supplies, it searches for an
observation chain and an action chain that carry the benchmark's form to the
policy's and the policy's action to the benchmark's. It prefers lossless chains
(it tries lossless-only first) and shortest (breadth-first), and returns a
`Pipeline` that `check_compatibility` will accept, or None if the pool cannot
bridge the gap.

It searches only the pool it is given — there is no registry. Each adapter is a
node-to-node edge over specs: from a spec it applies to, `produce` yields the
next spec. Specs are compared and de-duplicated by value (they are frozen), so
the search terminates.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from typing import TypeVar

from pydantic import BaseModel

from manifold.core.adapter import ActionAdapter, ObservationAdapter
from manifold.core.benchmark import Benchmark
from manifold.core.observation_space import ObservationSpace
from manifold.core.pipeline import Pipeline
from manifold.core.policy import PolicySignature

_Adapter = TypeVar("_Adapter", ActionAdapter, ObservationAdapter)


def resolve(
    policy: PolicySignature,
    benchmark: Benchmark,
    adapters: Sequence[ActionAdapter | ObservationAdapter],
) -> Pipeline | None:
    """Search `adapters` for a pipeline bridging the pairing, or return None."""
    # Stateful adapters are excluded from the search pools: they carry per-episode
    # runtime state a resolver cannot reason about, so they are author-inserted
    # into a hand-built pipeline, never auto-discovered (ADR-0001). They remain
    # fully checkable by `check_compatibility` / `verify` on that hand-built chain.
    discoverable = [a for a in adapters if a.state_key is None]
    action_pool = [a for a in discoverable if isinstance(a, ActionAdapter)]
    observation_pool = [a for a in discoverable if isinstance(a, ObservationAdapter)]

    bench_action = benchmark.embodiment.action
    action_chain = _search_lossless_first(
        policy.action_space, action_pool, lambda spec: spec == bench_action
    )
    if action_chain is None:
        return None

    observation_chain = _search_lossless_first(
        benchmark.observation_space,
        observation_pool,
        lambda spec: _satisfies(spec, policy.observation_space),
    )
    if observation_chain is None:
        return None

    return Pipeline(observation=observation_chain, action=action_chain)


def _search_lossless_first(
    start: BaseModel,
    adapters: Sequence[_Adapter],
    reached: Callable[[BaseModel], bool],
) -> list[_Adapter] | None:
    lossless = [adapter for adapter in adapters if adapter.lossless]
    found = _search(start, lossless, reached)
    if found is not None:
        return found
    return _search(start, adapters, reached)


def _search(
    start: BaseModel,
    adapters: Sequence[_Adapter],
    reached: Callable[[BaseModel], bool],
) -> list[_Adapter] | None:
    """Breadth-first search over specs; the shortest chain of adapters, or None."""
    queue: deque[tuple[BaseModel, list[_Adapter]]] = deque([(start, [])])
    # Dedup by value fingerprint, not object identity: specs need not be hashable,
    # and equal specs reached by different chains collapse to one node.
    seen = {_fingerprint(start)}
    while queue:
        spec, path = queue.popleft()
        if reached(spec):
            return path
        for adapter in adapters:
            if type(spec) is not adapter.from_spec or not adapter.applies(spec):
                continue
            nxt = adapter.produce(spec)
            fingerprint = _fingerprint(nxt)
            if fingerprint not in seen:
                seen.add(fingerprint)
                queue.append((nxt, [*path, adapter]))
    return None


def _fingerprint(spec: BaseModel) -> str:
    return spec.model_dump_json()


def _satisfies(current: BaseModel, wanted: ObservationSpace) -> bool:
    return isinstance(current, ObservationSpace) and wanted.first_unmet(current) is None


__all__ = ["resolve"]
