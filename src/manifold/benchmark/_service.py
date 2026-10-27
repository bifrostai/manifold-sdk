"""The shared Services that run a benchmark's lifecycle steps around its hooks.

A Service loads the envs for a task, starts episodes on them, steps them with the
policy's actions and unloads them. The Client calls the Service, and the Service calls
the hooks. There are two types of Service:

- `single`: a simulator for each env. `on_reset` and `on_step` act on a single env.
- `batched`: a simulator that contains all envs. `on_step` steps all of them in a
  single call.

The keys of `episodes`, `actions` and the returned dicts are env ids, from 0 to
`num_envs - 1`. The Client resets an env only between episodes, and sends actions
only for envs in an episode.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, cast

from packaging.specifiers import SpecifierSet

from manifold.benchmark.config import (
    BenchmarkBatchedServiceConfig,
    BenchmarkSingleServiceConfig,
)
from manifold.benchmark.hooks import BatchedServiceHooks, SingleServiceHooks
from manifold.benchmark.simulators import make_simulator
from manifold.lib.compat import assert_never

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, MutableMapping
    from logging import Logger, LoggerAdapter

    from manifold.benchmark._worker import WorkerTask
    from manifold.benchmark.config import BenchmarkConfig
    from manifold.benchmark.simulators import Simulator
    from manifold.core.values import Action, Observation
    from manifold.replay import ManifoldEvent

_logger = logging.getLogger(__name__)

# The hooks' loggers write to this logger. `main` shows its messages at INFO.
_hooks_logger = logging.getLogger("manifold.benchmark.hooks")


class _HooksLogger(logging.LoggerAdapter[logging.Logger]):
    """A logger for the hooks. It writes `prefix` before each message."""

    def __init__(self, prefix: str) -> None:
        super().__init__(_hooks_logger, {})
        self._prefix = prefix

    def process(
        self, msg: object, kwargs: MutableMapping[str, Any]
    ) -> tuple[object, MutableMapping[str, Any]]:
        return f"{self._prefix}{msg}", kwargs


@dataclass(frozen=True)
class StepResult:
    """The result of a step of an env with an action.

    `observation` is the next observation, which the Client sends to the policy.
    `success` is true when the step solved the task. `done` is true when the episode
    ended for another reason, such as a truncation or a termination without success.
    Either flag ends the episode, and the episode record stores `success`.

    `events` contains what upstream reports after the step, as Manifold events. Each
    goal check is a fact, such as `OnEvent(object="bowl", reference="plate",
    is_active=True)`. Its `is_active` states whether the fact is true. Each message for
    a reader is a `NoteEvent`. A replay records a fact when its `is_active` changes, and
    records other events when they arrive. A benchmark that reports nothing passes an
    empty tuple.
    """

    observation: Observation
    success: bool
    done: bool
    events: tuple[ManifoldEvent, ...]


@dataclass(frozen=True)
class ResetResult:
    """The result of a reset of an env for an episode.

    `observation` is the episode's first observation, which the Client sends to the
    policy. `instruction` is the episode's language instruction, or None for a
    benchmark without instructions. The Client puts the instruction on each observation
    of the episode, so the hooks leave it out of their observations.

    `task_id` is the benchmark's own identifier for the episode's task. The Client
    writes it to the episode record and to each live view frame. The policy does not
    receive it.

    `attributes` labels the episode for result tables, such as
    `{"category": "camera", "difficulty": "3"}`. The replay's `EpisodeStartedEvent` and
    the episode record contain them. A benchmark without labels passes an empty dict.
    `events` contains what upstream reports at the start of the episode, in the same
    form as `StepResult.events`.

    `max_steps` is the step limit of the episode's task, for a benchmark whose tasks
    have different limits, such as 280 steps in a suite and 520 in another. Such a
    benchmark sets `config.run.max_steps` to null. The Client raises an error if both
    the hooks and the config set a limit. A benchmark whose tasks use the config's
    limit passes None.
    """

    observation: Observation
    instruction: str | None
    task_id: str
    attributes: dict[str, str]
    events: tuple[ManifoldEvent, ...]
    max_steps: int | None


def _log_unsupported_versions(simulator: Simulator) -> None:
    """Log a warning for each installed package outside `simulator`'s supported versions.

    The function skips the packages that are not installed.
    """
    for package, supported in simulator.supported_versions.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            continue
        if not SpecifierSet(supported).contains(installed, prereleases=True):
            _logger.warning(
                "%s %s is outside the versions that %s supports, %s",
                package,
                installed,
                type(simulator).__name__,
                supported,
            )


class Service(ABC):
    """A base class that runs a benchmark's lifecycle steps around its hooks.

    The steps are start, load, reset, step, unload and stop.
    """

    hooks: SingleServiceHooks | BatchedServiceHooks

    def __init__(
        self, config: BenchmarkConfig, hooks: SingleServiceHooks | BatchedServiceHooks
    ) -> None:
        self.config = config
        self.hooks = hooks
        # The simulator that the replay reads the scene and physics from. It is None for a
        # config without a simulator.
        self.simulator = make_simulator(config)
        self.num_envs = config.env.num_envs
        self.reset_together = (
            isinstance(config.service, BenchmarkBatchedServiceConfig)
            and config.service.reset_together
        )
        # The `env_key` of the loaded envs. It is None when the Service unloads the envs
        # after the task.
        self._env_key: Hashable | None = None
        # `on_start` and `on_stop` get the container's logger. `on_load` and `on_unload`
        # get the logger of the last task that loaded envs.
        self._container_logger = _HooksLogger("")
        self._task_logger = self._container_logger

    def start(self) -> None:
        """Start the simulator, then run the hooks' `on_start`.

        Before the simulator starts, `start` logs a warning for each installed package
        outside the versions that the simulator supports.
        """
        if self.simulator is not None:
            _log_unsupported_versions(self.simulator)
            self.simulator.start()
        self.hooks.on_start(self.config, self._container_logger)

    def load(self, task: WorkerTask) -> None:
        """Load the envs for `task`, or reuse the loaded envs if `env_key` returns the same key."""
        self._task = task
        task_logger = _HooksLogger(f"task {task.task_id}: ")
        env_key = self.hooks.env_key(self.config, task.task_config, task_logger)
        if env_key is not None and env_key == self._env_key:
            return
        self._unload_envs()
        self._task_logger = task_logger
        self._load_envs(task)
        self._env_key = env_key

    def unload(self) -> None:
        """Unload the envs of the task, unless `env_key` returned a key for them."""
        if self._env_key is None:
            self._unload_envs()

    @abstractmethod
    def reset(
        self, episodes: Mapping[int, int], loggers: Mapping[int, LoggerAdapter[Logger]]
    ) -> dict[int, ResetResult]:
        """Start episode `episodes[env_id]` on each env id in `episodes`.

        `loggers` contains the logger of each env's new episode, by env id.
        """

    @abstractmethod
    def step(
        self, actions: Mapping[int, Action], loggers: Mapping[int, LoggerAdapter[Logger]]
    ) -> dict[int, StepResult]:
        """Step each env in `actions` with its action.

        `loggers` contains the logger of each env's episode, by env id.
        """

    @abstractmethod
    def _load_envs(self, task: WorkerTask) -> None:
        """Run the hooks' `on_load` for `task`, and attach the simulator."""

    def _run_simulator_post_step(self) -> None:
        """Run the simulator's `post_step`, after the hooks reset or step any env."""
        if self.simulator is not None:
            self.simulator.post_step()

    @abstractmethod
    def _unload_envs(self) -> None:
        """Run the hooks' `on_unload` for each loaded env."""

    def stop(self) -> None:
        """Unload the loaded envs, run the hooks' `on_stop`, then stop the simulator."""
        try:
            self._unload_envs()
            self._env_key = None
            self.hooks.on_stop(self.config, self._container_logger)
        finally:
            if self.simulator is not None:
                self.simulator.stop()


class SingleService(Service):
    """A Service with a simulator for each env.

    Each call to `on_reset` or `on_step` acts on a single env.
    """

    hooks: SingleServiceHooks

    def __init__(self, config: BenchmarkConfig, hooks: SingleServiceHooks) -> None:
        super().__init__(config, hooks)
        self._envs: list[Any] = []

    def _load_envs(self, task: WorkerTask) -> None:
        self._envs = [
            self.hooks.on_load(self.config, task.task_config, self._task_logger)
            for _ in range(self.num_envs)
        ]
        if self.simulator is not None:
            for env_id, env in enumerate(self._envs):
                self.simulator.attach(env, [env_id])

    def reset(
        self, episodes: Mapping[int, int], loggers: Mapping[int, LoggerAdapter[Logger]]
    ) -> dict[int, ResetResult]:
        results = {
            env_id: self.hooks.on_reset(
                self.config,
                self._envs[env_id],
                self._task.task_config,
                episode_idx,
                loggers[env_id],
            )
            for env_id, episode_idx in episodes.items()
        }
        self._run_simulator_post_step()
        return results

    def step(
        self, actions: Mapping[int, Action], loggers: Mapping[int, LoggerAdapter[Logger]]
    ) -> dict[int, StepResult]:
        results = {
            env_id: self.hooks.on_step(
                self.config, self._envs[env_id], self._task.task_config, action, loggers[env_id]
            )
            for env_id, action in actions.items()
        }
        self._run_simulator_post_step()
        return results

    def _unload_envs(self) -> None:
        for env in self._envs:
            self.hooks.on_unload(self.config, env, self._task_logger)
        self._envs = []


class BatchedService(Service):
    """A Service with a simulator that contains all envs.

    A single call to `on_step` steps all envs that have actions.
    """

    hooks: BatchedServiceHooks

    def __init__(self, config: BenchmarkConfig, hooks: BatchedServiceHooks) -> None:
        super().__init__(config, hooks)
        self._env: Any = None

    def _load_envs(self, task: WorkerTask) -> None:
        self._env = self.hooks.on_load(self.config, task.task_config, self._task_logger)
        if self.simulator is not None:
            self.simulator.attach(self._env, list(range(self.num_envs)))

    def reset(
        self, episodes: Mapping[int, int], loggers: Mapping[int, LoggerAdapter[Logger]]
    ) -> dict[int, ResetResult]:
        if self.reset_together and sorted(episodes) != list(range(len(episodes))):
            raise ValueError(
                "a Service that resets all envs together resets env ids 0..n-1 at once"
            )
        results = {
            env_id: self.hooks.on_reset(
                self.config,
                self._env,
                env_id,
                self._task.task_config,
                episodes[env_id],
                loggers[env_id],
            )
            for env_id in sorted(episodes)
        }
        self._run_simulator_post_step()
        return results

    def step(
        self, actions: Mapping[int, Action], loggers: Mapping[int, LoggerAdapter[Logger]]
    ) -> dict[int, StepResult]:
        env_ids = sorted(actions)
        results = self.hooks.on_step(
            self.config,
            self._env,
            self._task.task_config,
            env_ids,
            [actions[env_id] for env_id in env_ids],
            [loggers[env_id] for env_id in env_ids],
        )
        self._run_simulator_post_step()
        return dict(zip(env_ids, results, strict=True))

    def _unload_envs(self) -> None:
        if self._env is not None:
            self.hooks.on_unload(self.config, self._env, self._task_logger)
        self._env = None


def make_service(config: BenchmarkConfig, hooks: object) -> Service:
    """Build the Service that `config.service` names.

    `hooks` is a module that `load_hooks` checked against the Protocol of the Service
    type.
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
