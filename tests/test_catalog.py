import pytest

from manifold.benchmarks import ALL
from manifold.core import Benchmark, Compatibility, PolicySignature, check_compatibility


@pytest.mark.parametrize("benchmark", ALL, ids=lambda benchmark: benchmark.name)
def test_a_native_policy_is_compatible_with_each_shipped_benchmark(benchmark: Benchmark) -> None:
    # A policy that declares exactly what the benchmark publishes must pair,
    # with no adapter — this is the coherence check on the catalog as it grows.
    native = PolicySignature(
        action_space=benchmark.embodiment.action,
        proprioception=benchmark.observation_space.proprioception,
        cameras=list(benchmark.observation_space.cameras),
        instruction=benchmark.observation_space.instruction,
    )
    assert check_compatibility(native, benchmark).status is Compatibility.COMPATIBLE
