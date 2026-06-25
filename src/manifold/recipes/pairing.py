"""The pairing contract: a (profile, benchmark, pipeline) triple read off a module.

A pairing is the unit a server mounts — a policy `PolicyProfile`, the benchmark it
bridges to, and the adapter `Pipeline` between them. The matchmaker
(`multi_pairing_pipeline`), the shared-profile guard (`assert_shared_profile`), and
the launcher (`launch_server`) all operate on this triple, so it lives here in the
library rather than in any consumer. `read_pairing` reads the triple off an
already-imported module that exports module-level `PROFILE`/`BENCHMARK`/`PIPELINE` —
the authoring convention for a "pairing module" — and is namespace-agnostic: the
caller imports the module however it likes (a launcher de-duplicating its own bundled
pairings, or a third party launching a policy module by import path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    from manifold.core.benchmark import Benchmark
    from manifold.core.pipeline import Pipeline
    from manifold.recipes.serving import PolicyProfile

_REQUIRED = ("PROFILE", "BENCHMARK", "PIPELINE")


@dataclass(frozen=True)
class Pairing:
    """A policy profile, the benchmark it bridges to, and the adapter pipeline between.

    The unit a server mounts: `profile` loads into a `PolicyEndpoint`, `benchmark` is
    what the runner advertises, and `pipeline` bridges the two. Naming it as a typed
    value lets the launcher and the matchmaker work against `.profile`/`.benchmark`/
    `.pipeline` rather than raw, unchecked module attributes.
    """

    profile: PolicyProfile
    benchmark: Benchmark
    pipeline: Pipeline


def read_pairing(module: ModuleType) -> Pairing:
    """Read a `Pairing` off a module exporting PROFILE / BENCHMARK / PIPELINE.

    The authoring convention for a "pairing module": three module-level names —
    `PROFILE` (the `PolicyProfile`), `BENCHMARK` (the catalog `Benchmark`), and
    `PIPELINE` (the adapter `Pipeline`). Namespace-agnostic — the caller imports the
    module however it likes — so the SDK fixes the contract without owning anyone's
    module layout.

    Raises:
        AttributeError: Naming every missing name, if the module does not export all
            of PROFILE / BENCHMARK / PIPELINE — so a malformed module fails with one
            clear message rather than an opaque attribute error later.
    """
    missing = [attr for attr in _REQUIRED if not hasattr(module, attr)]
    if missing:
        raise AttributeError(
            f"{module.__name__} is missing {', '.join(missing)} — a pairing module "
            "must export module-level PROFILE, BENCHMARK, and PIPELINE."
        )
    return Pairing(module.PROFILE, module.BENCHMARK, module.PIPELINE)


__all__ = ["Pairing", "read_pairing"]
