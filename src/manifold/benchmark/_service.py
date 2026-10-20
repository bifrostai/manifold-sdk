"""The shared Services that run a benchmark's lifecycle steps around its hooks.

A Service starts the envs for a task, starts episodes on them, steps them with the
policy's actions and unloads them. The Client calls it; the benchmark's hooks file
fills in the upstream calls. Two types cover the benchmarks:

- `single`: a simulator for each env. `on_reset` and `on_step` move one env each.
- `batched`: a simulator that holds all envs. `on_step` moves all of them at once.

Keys of `episodes`, `actions` and the returned dicts are env ids, from 0 to
`num_envs - 1`. The Client resets an env only when it is not in an episode, and
sends actions only for envs that are in one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from manifold.benchmark.config import (
    BenchmarkBatchedServiceConfig,
    BenchmarkSingleServiceConfig,
)
from manifold.benchmark.hooks import BatchedServiceHooks, SingleServiceHooks
from manifold.lib.compat import assert_never

if TYPE_CHECKING:
    from collections.abc import Mapping

    from manifold.benchmark._worker import WorkerTask
    from manifold.benchmark.config import BenchmarkConfig
    from manifold.benchmark.simulators import Simulator
    from manifold.core.values import Action, Observation
    from manifold.replay import ManifoldEvent


@dataclass(frozen=True)
class StepResult:
    """The outcome of stepping the environment with one action.

    `observation` is the next observation the runner ships back to the policy;
    `success` marks the task solved this step; `done` marks the episode otherwise
    finished (truncated, terminated without success). Either flag ends the
    episode; only `success` counts toward the tally.

    `events` holds what upstream reports after this step, as Manifold events: each
    goal check as a fact whose `is_active` says whether it holds, such as
    `OnEvent(object="bowl", reference="plate", is_active=True)`, and each message for a
    reader as a `NoteEvent`. A replay records a fact when its `is_active` changes, and
    any other event as it comes. A benchmark that reports nothing passes an empty
    tuple.
    """

    observation: Observation
    success: bool
    done: bool
    events: tuple[ManifoldEvent, ...]


@dataclass(frozen=True)
class ResetResult:
    """The outcome of resetting the environment for one episode.

    `observation` is the episode's first observation, which the runner sends to the
    policy. `task_id` is the benchmark's own identifier for the episode's task. It
    goes onto the episode record and each live view frame, and never to the policy.

    `attributes` labels the episode for result tables, such as
    `{"category": "camera", "difficulty": "3"}`. They go into the replay's
    `EpisodeStartedEvent` and onto the episode record. A benchmark without labels passes
    an empty dict. `events` holds what upstream reports at the start of the
    episode, as `StepResult.events` does after each step.

    `max_steps` is the step limit of the episode's task, for a benchmark whose tasks
    have different limits, such as 280 steps for one suite and 520 for another. Such
    a benchmark sets `config.run.max_steps` to null, and the Client refuses a limit
    here while the config sets one. A benchmark whose tasks share the config's limit
    passes None.
    """

    observation: Observation
    task_id: str
    attributes: dict[str, str]
    events: tuple[ManifoldEvent, ...]
    max_steps: int | None


class Service(ABC):
    """Runs the lifecycle steps for one benchmark: start, load, reset, step, unload and stop."""

    hooks: SingleServiceHooks | BatchedServiceHooks

    def __init__(
        self, config: BenchmarkConfig, hooks: SingleServiceHooks | BatchedServiceHooks
    ) -> None:
        self.config = config
        self.hooks = hooks
        # Reads the replay's scene and physics, or is None for a benchmark without one.
        self.simulator: Simulator | None = None
        self.num_envs = config.env.num_envs
        self.reset_together = (
            isinstance(config.service, BenchmarkBatchedServiceConfig)
            and config.service.reset_together
        )

    def start(self) -> None:
        """Start the simulator, once for each container, then the hooks' `on_start`."""
        if self.simulator is not None:
            self.simulator.start()
        self.hooks.on_start(self.config)

    @abstractmethod
    def load(self, task: WorkerTask) -> None:
        """Load the env(s) for `task`."""

    @abstractmethod
    def reset(self, episodes: Mapping[int, int]) -> dict[int, ResetResult]:
        """Start episode `episodes[env_id]` on each env id in `episodes`."""

    @abstractmethod
    def step(self, actions: Mapping[int, Action]) -> dict[int, StepResult]:
        """Step each env in `actions` with its action."""

    @abstractmethod
    def unload(self) -> None:
        """Unload the env(s) of the task."""

    def stop(self) -> None:
        """Run the hooks' `on_stop`, then stop the simulator, once for each container."""
        self.hooks.on_stop(self.config)
        if self.simulator is not None:
            self.simulator.stop()


class SingleService(Service):
    """A simulator for each env; the hooks' `on_reset` and `on_step` move one env each."""

    hooks: SingleServiceHooks

    def __init__(self, config: BenchmarkConfig, hooks: SingleServiceHooks) -> None:
        super().__init__(config, hooks)
        self._envs: list[Any] = []

    def load(self, task: WorkerTask) -> None:
        self._task = task
        self._envs = [
            self.hooks.on_load(self.config, task.task_config) for _ in range(self.num_envs)
        ]
        if self.simulator is not None:
            for env_id, env in enumerate(self._envs):
                self.simulator.attach(env, [env_id])

    def reset(self, episodes: Mapping[int, int]) -> dict[int, ResetResult]:
        return {
            env_id: self.hooks.on_reset(
                self.config, self._envs[env_id], self._task.task_config, episode_idx
            )
            for env_id, episode_idx in episodes.items()
        }

    def step(self, actions: Mapping[int, Action]) -> dict[int, StepResult]:
        return {
            env_id: self.hooks.on_step(
                self.config, self._envs[env_id], self._task.task_config, action
            )
            for env_id, action in actions.items()
        }

    def unload(self) -> None:
        for env in self._envs:
            self.hooks.on_unload(self.config, env)
        self._envs = []


class BatchedService(Service):
    """A simulator that holds all envs; a single `on_step` call moves all of them."""

    hooks: BatchedServiceHooks

    def __init__(self, config: BenchmarkConfig, hooks: BatchedServiceHooks) -> None:
        super().__init__(config, hooks)
        self._env: Any = None

    def load(self, task: WorkerTask) -> None:
        self._task = task
        self._env = self.hooks.on_load(self.config, task.task_config)
        if self.simulator is not None:
            self.simulator.attach(self._env, list(range(self.num_envs)))

    def reset(self, episodes: Mapping[int, int]) -> dict[int, ResetResult]:
        if self.reset_together and sorted(episodes) != list(range(len(episodes))):
            raise ValueError(
                "a Service that resets all envs together resets env ids 0..n-1 at once"
            )
        return {
            env_id: self.hooks.on_reset(
                self.config, self._env, env_id, self._task.task_config, episodes[env_id]
            )
            for env_id in sorted(episodes)
        }

    def step(self, actions: Mapping[int, Action]) -> dict[int, StepResult]:
        env_ids = sorted(actions)
        results = self.hooks.on_step(
            self.config,
            self._env,
            self._task.task_config,
            env_ids,
            [actions[env_id] for env_id in env_ids],
        )
        return dict(zip(env_ids, results, strict=True))

    def unload(self) -> None:
        if self._env is not None:
            self.hooks.on_unload(self.config, self._env)
        self._env = None


def make_service(config: BenchmarkConfig, hooks: object) -> Service:
    """Build the Service that `config.service` names, around a checked hooks module.

    `load_hooks` has already checked `hooks` against the Service type's Protocol.
    """
    service = config.service
    if isinstance(service, BenchmarkSingleServiceConfig):
        return SingleService(config, cast("SingleServiceHooks", hooks))
    if isinstance(service, BenchmarkBatchedServiceConfig):
        return BatchedService(config, cast("BatchedServiceHooks", hooks))
    return assert_never(service)


__all__ = [
    "BatchedService",
    "Service",
    "SingleService",
    "make_service",
]
