"""Running a benchmark from a config file and a short hooks file.

Shared code runs the same lifecycle for every benchmark: a Client claims work from
the task queue, runs episodes on a Service, asks the policy for actions and reports
the results. A benchmark supplies a config file (the facts and the spec) and a
hooks file (the upstream calls). `python -m manifold.benchmark <config>` runs it.

This package re-exports the spec, `manifold.core.benchmark.Benchmark`, and the
values that a hooks file returns, so a hooks file imports from this package alone.
It also exports the types of the task server and the episode record. Tests and
`manifold.recipes.evaluate` import them from here.
"""

from __future__ import annotations

from manifold.benchmark._service import ResetResult, StepResult
from manifold.benchmark._worker import WorkerEpisodeRecord, WorkerSession, WorkerTask
from manifold.benchmark.config import (
    BenchmarkBatchedServiceConfig,
    BenchmarkConfig,
    BenchmarkEnvConfig,
    BenchmarkReplayConfig,
    BenchmarkRunConfig,
    BenchmarkSingleServiceConfig,
    BenchmarkUpstreamConfig,
    BenchmarkVersionConfig,
    load_config,
)
from manifold.benchmark.hooks import (
    OPTIONAL_HOOKS,
    BatchedServiceHooks,
    HooksError,
    SingleServiceHooks,
    load_hooks,
)
from manifold.core.benchmark import Benchmark
from manifold.core.values import Action, Observation

__all__ = [
    "OPTIONAL_HOOKS",
    "Action",
    "BatchedServiceHooks",
    "Benchmark",
    "BenchmarkBatchedServiceConfig",
    "BenchmarkConfig",
    "BenchmarkEnvConfig",
    "BenchmarkReplayConfig",
    "BenchmarkRunConfig",
    "BenchmarkSingleServiceConfig",
    "BenchmarkUpstreamConfig",
    "BenchmarkVersionConfig",
    "HooksError",
    "Observation",
    "ResetResult",
    "SingleServiceHooks",
    "StepResult",
    "WorkerEpisodeRecord",
    "WorkerSession",
    "WorkerTask",
    "load_config",
    "load_hooks",
]
