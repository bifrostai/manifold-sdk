"""The hooks of a benchmark's hooks file, and the check that the file defines them.

A hooks file is a Python module with a function for each hook. The Protocols in this
module state the arguments and the return value of each hook, for each type of
Service. A hooks file defines the hooks as plain functions without `self`.

Each hook takes the config first and reads its values from it, such as
`config.extras`. Each hook takes a logger last and writes its messages through it.
`on_start` and `on_stop` get the container's logger. `env_key`, `on_load` and
`on_unload` get the task's logger. `on_reset` and `on_step` get the episode's logger.

`load_hooks` imports `hooks.py` from the config's directory and checks it against the
Protocol of its Service. It imports the file before the simulator starts, and
`--check` imports it without a simulator. Some packages load only in a running
simulator, such as Isaac Lab after Isaac Sim launches. A hooks file imports such a
package inside each hook that uses it.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    import logging
    from collections.abc import Hashable, Sequence
    from pathlib import Path
    from types import ModuleType

    from manifold.benchmark._service import ResetResult, StepResult
    from manifold.benchmark.config import BenchmarkConfig
    from manifold.core.values import Action


# `load_hooks` imports the hooks file with this name from the config's directory.
_HOOKS_FILE = "hooks.py"

# The hooks that a hooks file can leave out. `load_hooks` sets each missing hook to a
# function that does nothing. The Service then calls all hooks in the same way.
OPTIONAL_HOOKS = ("on_start", "env_key", "on_unload", "on_stop")

# The type of the env that `on_load` returns. The other hooks of the task get this env.
_EnvT = TypeVar("_EnvT")


class SingleServiceHooks(Protocol[_EnvT]):
    """The hooks of a single Service. A single Service has a simulator for each env."""

    def on_start(
        self, config: BenchmarkConfig, logger: logging.LoggerAdapter[logging.Logger]
    ) -> None:
        """Start the simulator, once for each container. Optional."""
        ...

    def env_key(
        self,
        config: BenchmarkConfig,
        task_config: dict[str, Any],
        logger: logging.LoggerAdapter[logging.Logger],
    ) -> Hashable | None:
        """Return what identifies the env that `on_load` builds for the task. Optional.

        If `env_key` returns the same value for the next task, the Service reuses the env
        and skips `on_unload` and `on_load`. If `env_key` returns None, the Service
        builds a new env for each task.
        """
        ...

    def on_load(
        self,
        config: BenchmarkConfig,
        task_config: dict[str, Any],
        logger: logging.LoggerAdapter[logging.Logger],
    ) -> _EnvT:
        """Load the task, and return the env."""
        ...

    def on_reset(
        self,
        config: BenchmarkConfig,
        env: _EnvT,
        task_config: dict[str, Any],
        episode_idx: int,
        logger: logging.LoggerAdapter[logging.Logger],
    ) -> ResetResult:
        """Start episode `episode_idx` on `env`. `logger` is the episode's logger."""
        ...

    def on_step(
        self,
        config: BenchmarkConfig,
        env: _EnvT,
        task_config: dict[str, Any],
        action: Action,
        logger: logging.LoggerAdapter[logging.Logger],
    ) -> StepResult:
        """Step `env` with `action`. `logger` is the episode's logger."""
        ...

    def on_unload(
        self, config: BenchmarkConfig, env: _EnvT, logger: logging.LoggerAdapter[logging.Logger]
    ) -> None:
        """Unload the task's env. Optional."""
        ...

    def on_stop(
        self, config: BenchmarkConfig, logger: logging.LoggerAdapter[logging.Logger]
    ) -> None:
        """Stop the simulator, once for each container. Optional.

        The Service calls this hook even when `on_start` or the simulator's start
        raised.
        """
        ...


class BatchedServiceHooks(Protocol[_EnvT]):
    """The hooks of a batched Service.

    A batched Service has a simulator that contains all envs.
    """

    def on_start(
        self, config: BenchmarkConfig, logger: logging.LoggerAdapter[logging.Logger]
    ) -> None:
        """Start the simulator, once for each container. Optional."""
        ...

    def env_key(
        self,
        config: BenchmarkConfig,
        task_config: dict[str, Any],
        logger: logging.LoggerAdapter[logging.Logger],
    ) -> Hashable | None:
        """Return what identifies the env that `on_load` builds for the task. Optional.

        If `env_key` returns the same value for the next task, the Service reuses the env
        and skips `on_unload` and `on_load`. If `env_key` returns None, the Service
        builds a new env for each task.
        """
        ...

    def on_load(
        self,
        config: BenchmarkConfig,
        task_config: dict[str, Any],
        logger: logging.LoggerAdapter[logging.Logger],
    ) -> _EnvT:
        """Load the task, and return the batched env with all `num_envs` envs."""
        ...

    def on_reset(
        self,
        config: BenchmarkConfig,
        env: _EnvT,
        env_id: int,
        task_config: dict[str, Any],
        episode_idx: int,
        logger: logging.LoggerAdapter[logging.Logger],
    ) -> ResetResult:
        """Start episode `episode_idx` on env `env_id`. `logger` is the episode's logger."""
        ...

    def on_step(
        self,
        config: BenchmarkConfig,
        env: _EnvT,
        task_config: dict[str, Any],
        env_ids: Sequence[int],
        actions: Sequence[Action],
        loggers: Sequence[logging.LoggerAdapter[logging.Logger]],
    ) -> list[StepResult]:
        """Step the envs in `env_ids` with `actions`, in a single call.

        `actions`, `loggers` and the returned results follow the order of `env_ids`.
        `loggers` contains the logger of each env's episode.
        """
        ...

    def on_unload(
        self, config: BenchmarkConfig, env: _EnvT, logger: logging.LoggerAdapter[logging.Logger]
    ) -> None:
        """Unload the task's env. Optional."""
        ...

    def on_stop(
        self, config: BenchmarkConfig, logger: logging.LoggerAdapter[logging.Logger]
    ) -> None:
        """Stop the simulator, once for each container. Optional.

        The Service calls this hook even when `on_start` or the simulator's start
        raised.
        """
        ...


# The Protocol of each Service type, by the config's `service.type`.
HOOKS: dict[str, type] = {
    "single": SingleServiceHooks,
    "batched": BatchedServiceHooks,
}


class HooksError(ValueError):
    """Raised when a hooks file lacks a hook, or a hook takes the wrong arguments."""


def _hook_parameters(protocol: type) -> dict[str, list[str]]:
    """The argument names of each hook in `protocol`, without `self`."""
    return {
        name: list(inspect.signature(member).parameters)[1:]
        for name, member in vars(protocol).items()
        if inspect.isfunction(member) and not name.startswith("_")
    }


def _check_hooks(hooks: Any, service_type: str) -> None:
    """Check that a hooks module defines the hooks of `service_type`, with their arguments.

    Raises:
        HooksError: If a required hook is missing, or a hook's argument names differ
            from its Protocol's.
    """
    for name, expected in _hook_parameters(HOOKS[service_type]).items():
        function = getattr(hooks, name, None)
        if function is None:
            if name not in OPTIONAL_HOOKS:
                raise HooksError(f"the hooks file of a {service_type} Service must define {name}")
            continue
        found = list(inspect.signature(function).parameters)
        if found != expected:
            raise HooksError(f"{name} must take {tuple(expected)}; it takes {tuple(found)}")


def _do_nothing(*_args: object) -> None:
    """Take the place of an optional hook that a hooks file leaves out."""


def _fill_optional_hooks(hooks: Any) -> None:
    """Set each optional hook that `hooks` lacks to a hook that does nothing."""
    for name in OPTIONAL_HOOKS:
        if getattr(hooks, name, None) is None:
            setattr(hooks, name, _do_nothing)


def load_hooks(config: BenchmarkConfig, root: Path) -> ModuleType:
    """Import `hooks.py` from `root`, the config's directory, and check it.

    Each optional hook that the file leaves out becomes a function that does nothing.

    Raises:
        HooksError: As `_check_hooks` describes, or if the file cannot be imported.
    """
    path = root / _HOOKS_FILE
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    if spec is None or spec.loader is None:
        raise HooksError(f"cannot import the hooks file {path}")
    hooks = importlib.util.module_from_spec(spec)
    # A dataclass looks up its module in `sys.modules` while Python runs the file.
    sys.modules[spec.name] = hooks
    spec.loader.exec_module(hooks)
    _check_hooks(hooks, config.service.type)
    _fill_optional_hooks(hooks)
    return hooks


__all__ = [
    "HOOKS",
    "OPTIONAL_HOOKS",
    "BatchedServiceHooks",
    "HooksError",
    "SingleServiceHooks",
    "load_hooks",
]
