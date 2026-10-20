"""Shared code for each simulator, such as MuJoCo or Isaac Sim, in a module for each.

A `Simulator` runs a simulator's process, and reads from the simulator what a
replay needs: the scene's bodies and meshes, each body's pose, each camera's pose,
the cameras that the policy does not see, and the simulator's own scalars, such as
joint torques and contact forces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from manifold.replay import ReplayPose, ReplaySceneBody, RerunScalar


class Simulator(ABC):
    """Run one simulator's process, and read the simulator for the replay.

    The Service calls `start` before the hooks' `on_start`, `attach` after each
    `on_load`, and `stop` after `on_stop`. `start` and `stop` run the simulator's
    process, which a benchmark needs whether or not it writes a replay.

    The methods named `replay_*` read the simulator for the replay, and only the
    Client's replay recorder calls them. `attach` gives them the env to read.

    To add a simulator, follow the rules in ADR 0011 under "Simulator classes":

    1. Add a module with `Benchmark<Simulator>SimulatorConfig` and
       `<Simulator>Simulator`. If the replay methods read values once for each
       episode, also add `_<Simulator>EpisodeCache`.
    2. Implement the methods of this class. Add no other public method.
    3. Make each other method private. If it uses only its arguments, make it a
       static method. If it reads `self`, make it an instance method.
    4. If a private method returns a value, name it for that value, such as
       `_body_name`. If it returns nothing, name it with a verb for what it does,
       such as `_log_skipped_geoms`.
    """

    @abstractmethod
    def start(self) -> None:
        """Start the simulator's process, once for each container, before any hook runs."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the simulator's process, once for each container, after the last hook."""

    @abstractmethod
    def attach(self, env: Any, env_ids: Sequence[int]) -> None:
        """Read env ids `env_ids` from `env` until the next `attach`.

        `env` is an env that `on_load` returned. A single Service attaches each env
        with its own id. A batched Service attaches its one env with every env id.
        """

    @abstractmethod
    def replay_bodies(self, env_id: int) -> tuple[ReplaySceneBody, ...]:
        """Each body of env `env_id`'s scene with its meshes, read at the start of an episode."""

    @abstractmethod
    def replay_poses(self, env_id: int) -> dict[str, ReplayPose]:
        """A dict from each scene body's name to its pose in env `env_id` now."""

    @abstractmethod
    def replay_extrinsics(self, env_id: int) -> dict[str, ReplayPose]:
        """Each posed camera's camera-to-world pose in env `env_id` now, in OpenCV axes."""

    @abstractmethod
    def replay_images(self, env_id: int) -> dict[str, np.ndarray]:
        """A dict from each extra camera's name to its frame in env `env_id`.

        An extra camera is in the scene but absent from the spec.
        """

    @abstractmethod
    def replay_scalars(self, env_id: int, ee_position: np.ndarray | None) -> list[RerunScalar]:
        """Env `env_id`'s scalars now: the same names, in the same order, at every step.

        `ee_position` is the end effector's position from the observation, which
        the distances to objects are measured from, or None when the observation
        holds none.
        """


__all__ = [
    "Simulator",
]
