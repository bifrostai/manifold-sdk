"""The hooks that a benchmark's hooks file defines, and the check that it defines them.

A hooks file is a Python module with a function for each hook. The Protocols below
state each hook's arguments and return value for each type of Service. A hooks file
implements them as plain functions, without `self`. Each hook gets the config first,
so a hook reads its values from the config, such as `config.extras`.
`load_hooks` imports `hooks.py`, next to the config, and checks it against the
Protocol for its Service.

`load_hooks` imports the file before the simulator starts, and `--check` imports it
without starting the simulator at all. So a hooks file imports a package that needs a
running simulator inside the hook that uses it, such as Isaac Lab, which must load
after Isaac Sim launches.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from types import ModuleType

    from manifold.benchmark._service import ResetResult, StepResult
    from manifold.benchmark.config import BenchmarkConfig
    from manifold.core.values import Action


# `load_hooks` imports the hooks file with this name from the config's folder.
_HOOKS_FILE = "hooks.py"

# The hooks that a hooks file may leave out. `load_hooks` fills in each missing one
# with a hook that does nothing, so the Service calls every hook the same way.
OPTIONAL_HOOKS = ("on_start", "on_unload", "on_stop")

# The env that `on_load` returns, which the other hooks of the task receive.
_EnvT = TypeVar("_EnvT")


class SingleServiceHooks(Protocol[_EnvT]):
    """The hooks of a single Service, which has a simulator for each env."""

    def on_start(self, config: BenchmarkConfig) -> None:
        """Start the simulator, once for each container. Optional."""
        ...

    def on_load(self, config: BenchmarkConfig, task_config: dict[str, Any]) -> _EnvT:
        """Load the task, and return the env."""
        ...

    def on_reset(
        self, config: BenchmarkConfig, env: _EnvT, task_config: dict[str, Any], episode_idx: int
    ) -> ResetResult:
        """Start episode `episode_idx` on `env`."""
        ...

    def on_step(
        self, config: BenchmarkConfig, env: _EnvT, task_config: dict[str, Any], action: Action
    ) -> StepResult:
        """Step `env` with `action`."""
        ...

    def on_unload(self, config: BenchmarkConfig, env: _EnvT) -> None:
        """Unload the task's env. Optional."""
        ...

    def on_stop(self, config: BenchmarkConfig) -> None:
        """Stop the simulator, once for each container. Optional."""
        ...


class BatchedServiceHooks(Protocol[_EnvT]):
    """The hooks of a batched Service, which has a simulator that holds all envs."""

    def on_start(self, config: BenchmarkConfig) -> None:
        """Start the simulator, once for each container. Optional."""
        ...

    def on_load(self, config: BenchmarkConfig, task_config: dict[str, Any]) -> _EnvT:
        """Load the task, and return the simulator that holds all envs."""
        ...

    def on_reset(
        self,
        config: BenchmarkConfig,
        env: _EnvT,
        env_id: int,
        task_config: dict[str, Any],
        episode_idx: int,
    ) -> ResetResult:
        """Start episode `episode_idx` on env `env_id`."""
        ...

    def on_step(
        self,
        config: BenchmarkConfig,
        env: _EnvT,
        task_config: dict[str, Any],
        env_ids: Sequence[int],
        actions: Sequence[Action],
    ) -> list[StepResult]:
        """Step the envs in `env_ids` with `actions`, in one call."""
        ...

    def on_unload(self, config: BenchmarkConfig, env: _EnvT) -> None:
        """Unload the task's simulator. Optional."""
        ...

    def on_stop(self, config: BenchmarkConfig) -> None:
        """Stop the simulator, once for each container. Optional."""
        ...


# The Protocol that a hooks file follows, for each type of Service.
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
    """Import `hooks.py` from `root`, the config's folder, and check it.

    Each optional hook that the file leaves out becomes a hook that does nothing.

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
