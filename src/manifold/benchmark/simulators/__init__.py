"""The `Simulator` base class and the helpers that the simulator modules share.

Each simulator, such as MuJoCo or Isaac Sim, has a module in this package. A
`Simulator` runs a simulator's process. It also reads the values of a replay from
the simulator: the scene's bodies and meshes, each body's pose, each camera's pose,
the extra cameras, and scalars such as joint torques and contact forces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from manifold.lib.compat import assert_never
from manifold.replay import ReplayMesh, ReplayScenePart

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from manifold.benchmark.config import BenchmarkConfig
    from manifold.replay import ReplayPose, ReplaySceneBody, RerunScalar


# The replay draws a scene part with more vertices than this as its bounding box, such
# as a scanned table. A very large mesh would make each replay log large.
BOX_VERTICES = 50_000

# The name of a fixture that the task does not name, such as the table or the walls
# around it. Failure analysis reads a touch of it as a touch of the table.
TABLE = "table"

# The parts of the robot that a contact names.
ARM = "arm"
GRIPPER = "gripper"


def bounding_box_part(parts: list[ReplayScenePart]) -> ReplayScenePart:
    """A box part that bounds `parts`, in the colour of the first part.

    The vertices of each part must be in the body's frame, with the part at the
    body's origin.
    """
    vertices = np.concatenate([np.asarray(part.mesh.vertices, dtype=np.float64) for part in parts])
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    return ReplayScenePart(
        mesh=ReplayMesh.box(np.maximum((high - low) / 2.0, 1e-4).tolist()),
        position=((low + high) / 2.0).astype(np.float32),
        color=parts[0].color,
    )


class Simulator(ABC):
    """A base class that runs a simulator's process and reads the simulator for the replay.

    The Service calls `start` before the hooks' `on_start`, `attach` after each
    `on_load`, and `stop` after `on_stop`. A benchmark needs the process whether or
    not it writes a replay.

    Only the Client's replay recorder calls the `replay_*` methods. They read the env
    that `attach` gave.

    To add a simulator:

    1. Add a module with `Benchmark<Simulator>SimulatorConfig` and
       `<Simulator>Simulator`. If the replay methods read values once for each
       episode, also add `_<Simulator>EpisodeCache`.
    2. Add the config to the type of `BenchmarkConfig.simulator`, and add a branch
       to `make_simulator`.
    3. Implement the methods and the property of this class. The only other public
       method can be a static method that the hooks call to set config values on an
       env before the env is built.
    4. Name and shape the private methods by the rules in ADR 0011, under
       "Simulator classes".
    """

    @property
    @abstractmethod
    def supported_versions(self) -> dict[str, str]:
        """The version range of each Python package that this simulator reads.

        Each key is a package name, and each value is a range in pip's syntax, such as
        `">=3.2,<3.9"`. A range covers the versions that benchmark images have run.
        The Service logs a warning for an installed version outside its range.
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
        with its own id. A batched Service attaches its single env with all env ids.
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
    def replay_contacts(self, env_id: int) -> frozenset[tuple[str, str]]:
        """Each robot part that touches a scene body in env `env_id` now, with the body.

        A part is `ARM` or `GRIPPER`, and a body is named as the scene names it. The
        gripper's contacts with fixtures are included, and the arm's contacts with
        fixtures and objects. The hooks report the gripper touching an object.
        """

    @abstractmethod
    def replay_scalars(self, env_id: int, ee_position: np.ndarray | None) -> list[RerunScalar]:
        """Env `env_id`'s scalars now: the same names, in the same order, at every step.

        `ee_position` is the end effector's position in the observation, or None if
        the observation has no end effector pose. The distances to objects start at
        this position.
        """


def make_simulator(config: BenchmarkConfig) -> Simulator | None:
    """The simulator for `config.simulator`, or None for a config without one."""
    from manifold.benchmark.simulators.isaacsim import (
        BenchmarkIsaacSimSimulatorConfig,
        IsaacSimSimulator,
    )
    from manifold.benchmark.simulators.mujoco import BenchmarkMujocoSimulatorConfig, MujocoSimulator
    from manifold.benchmark.simulators.sapien import BenchmarkSapienSimulatorConfig, SapienSimulator

    simulator = config.simulator
    if simulator is None:
        return None
    if isinstance(simulator, BenchmarkMujocoSimulatorConfig):
        return MujocoSimulator(simulator)
    if isinstance(simulator, BenchmarkIsaacSimSimulatorConfig):
        return IsaacSimSimulator(simulator, config)
    if isinstance(simulator, BenchmarkSapienSimulatorConfig):
        return SapienSimulator(simulator)
    return assert_never(simulator)


__all__ = [
    "ARM",
    "BOX_VERTICES",
    "GRIPPER",
    "TABLE",
    "Simulator",
    "bounding_box_part",
    "make_simulator",
]
