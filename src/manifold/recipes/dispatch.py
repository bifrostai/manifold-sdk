"""Build the `Callable[[Benchmark], Pipeline]` matchmaker that `serve` consumes."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from manifold.core.benchmark import Benchmark
from manifold.core.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

_Profile = TypeVar("_Profile")


def multi_pairing_pipeline(
    pairings: Iterable[tuple[Benchmark, Pipeline]],
) -> Callable[[Benchmark], Pipeline]:
    """Build a `Callable[[Benchmark], Pipeline]` dispatching by advertised benchmark.

    Given the `(benchmark, pipeline)` values that one server mounts, return the
    matchmaker `serve` invokes per connection: it reads the benchmark the runner
    advertised on HELLO and returns the pipeline that bridges it.

    The dispatch keys on the FULLER `(benchmark.name, benchmark.embodiment.name)`
    identity rather than `benchmark.name` alone. Keying on the name alone silently
    picks the wrong pipeline when two benchmarks share a name, or a name is reused
    for a variant on a different embodiment; keying on the pair means distinct
    embodiments for the same task name are never confused. The returned callable
    raises a clear `ValueError` for any advertised benchmark no pairing covers, and
    `multi_pairing_pipeline` itself raises `ValueError` up front if two pairings
    collide on the same `(name, embodiment)` key (an ambiguous mount the dispatcher
    could not resolve deterministically).

    `pairings` is the `(benchmark, pipeline)` values this server mounts; each
    benchmark's `(name, embodiment.name)` must be unique across the set. The returned
    callable maps an advertised `Benchmark` to its `Pipeline`, raising `ValueError`
    for an unknown benchmark. `multi_pairing_pipeline` itself raises `ValueError` if
    `pairings` is empty, or if two pairings collide on the same `(name,
    embodiment.name)` key.
    """
    by_identity: dict[tuple[str, str], Pipeline] = {}
    for benchmark, pipeline in pairings:
        key = (benchmark.name, benchmark.embodiment.name)
        if key in by_identity:
            raise ValueError(
                f"Two pairings mount the same (benchmark, embodiment) "
                f"{key[0]!r}/{key[1]!r}; the dispatcher could not resolve which "
                "pipeline to use. Mount each (benchmark, embodiment) at most once."
            )
        by_identity[key] = pipeline

    if not by_identity:
        raise ValueError("multi_pairing_pipeline needs at least one (benchmark, pipeline) pair")

    def dispatch(benchmark: Benchmark) -> Pipeline:
        key = (benchmark.name, benchmark.embodiment.name)
        pipeline = by_identity.get(key)
        if pipeline is None:
            known = ", ".join(f"{n!r}/{e!r}" for n, e in sorted(by_identity))
            raise ValueError(
                f"Runner advertised benchmark {benchmark.name!r} "
                f"(embodiment {benchmark.embodiment.name!r}) but no pairing covers "
                f"it. Known (benchmark, embodiment) pairs for this server: {known}"
            )
        return pipeline

    return dispatch


def assert_shared_profile(profiles: Iterable[_Profile]) -> _Profile:
    """Assert every profile is the same object identity; return it.

    Several pairings may share one running server only when they load the same
    checkpoint — the one loaded model the server serves. So the caller anchors on
    the first profile and this guard asserts the rest are identical by object
    identity (the contract / checkpoint is one shared value), raising a clear error
    if any differs. Returns the shared profile so the caller builds the endpoint
    from it.

    Compare with identity (`is`) rather than equality: profiles are the concrete
    descriptor objects a pairing exports, and two pairings sharing a server are
    expected to import and reference the *same* module-level profile constant, not
    two equal copies. A mismatch means two distinct checkpoints, which need two
    servers.

    `profiles` is the profile each mounted pairing exports, in order; it must be
    non-empty and all entries must be the same object. Returns the single shared
    profile, and raises `ValueError` if `profiles` is empty or any profile differs
    from the first.
    """
    items = list(profiles)
    if not items:
        raise ValueError("assert_shared_profile needs at least one profile")
    anchor = items[0]
    for profile in items[1:]:
        if profile is not anchor:
            raise ValueError(
                "All pairings mounted on one server must share the same profile "
                "(same checkpoint / contract): one loaded model serves them all. "
                "Run separate servers for different policies."
            )
    return anchor


__all__ = [
    "assert_shared_profile",
    "multi_pairing_pipeline",
]
