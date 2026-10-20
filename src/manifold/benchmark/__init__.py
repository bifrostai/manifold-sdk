"""Shared code that runs a benchmark from a config file and a hooks file.

The config file states the benchmark's values and its spec. The hooks file calls
upstream. Shared code runs the lifecycle steps around the hooks. A Client claims tasks
from the task queue, runs their episodes on a Service, asks the policy for actions and
reports the results. `python -m manifold.benchmark <config>` runs a benchmark.

A hooks file imports the config, `Benchmark` and the values that hooks return from
this package. Tests and `manifold.recipes.evaluate` import the types for the task
server and the episode record from here.
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
