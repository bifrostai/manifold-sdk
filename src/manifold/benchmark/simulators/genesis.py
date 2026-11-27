"""The Genesis simulator, which reads the scene, poses, touches and scalars of a Genesis scene.

The replay scene contains each link of the scene's rigid entities that has a visual
geom. Genesis gives each part of a link as a visual geom, with its vertices in the
geom's frame and the geom's pose in the link's frame. The scene leaves out planes,
such as the floor. Genesis orders quaternions `wxyz`, and this module returns them in
`xyzw` order.

Genesis entities have no names, so a task's entities are named by the attributes of
its env that hold them: `self.cube` is `cube`, and the second entity of the list
`self._blocks` is `blocks_1`. A link of an entity with several links is named
`<entity>_<link>`, and each link of the robot by its own name, such as `left_finger`.
A scene with several robots, such as two arms, names each link of the second robot
`<link>_1`, of the third `<link>_2`, and so on.

The simulator reads rigid entities only. A soft body, such as Genesis' MPM, PBD or FEM
entities, is not in the scene, and the replay records no touch of it.

This module imports `genesis` inside the functions that need it, so the SDK installs
and imports without it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from manifold.benchmark.simulators import (
    BOX_VERTICES,
    GREY,
    TABLE,
    BenchmarkHandTouchLinkConfig,
    Simulator,
    bounding_box_part,
    touches,
)
from manifold.lib.rotation import rotation_matrix_from_xyzw, wxyz_to_xyzw
from manifold.replay import (
    ReplayMesh,
    ReplayPose,
    ReplaySceneBody,
    ReplaySceneBodyRole,
    ReplayScenePart,
    RerunScalar,
    RerunTab,
    rerun_scalars,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

# The smallest contact force, in newtons, that counts as a touch.
_TOUCH_NEWTONS = 0.1


@dataclass(frozen=True)
class _GenesisEpisodeCache:
    """Values that `replay_bodies` reads from the scene at the start of an episode.

    The scene stays the same during an episode, so the other replay methods use these
    values at each step instead of searching the scene again.

    `scene` is the Genesis scene that the values come from. `bodies` maps the name of
    each scene body to its link's index in the rigid solver. `parts` maps the solver
    index of each touch link to its kind, `ARM` or `GRIPPER`, and `pads` maps the
    solver index of each finger pad's link to the pad's body name. `things` maps the
    solver index of each link outside the robots that has a name to that name and to
    whether the link is an object. `robots` holds each robot's entity and the indices of
    its arm's and gripper's degrees of freedom, in the order of the config's robots.
    """

    scene: Any
    bodies: dict[str, int]
    parts: dict[int, str]
    pads: dict[int, str]
    things: dict[int, tuple[str, bool]]
    robots: tuple[tuple[Any, np.ndarray, np.ndarray], ...]


# A path step that picks the item of a list whose attribute has a value, such as
# `agents[uid=panda_leaphand_left]`.
_SELECTOR = re.compile(r"^(\w+)\[(\w+)=([^\]]+)\]$")


def _follow(start: Any, path: str) -> Any:
    """The value at `path` from `start`, one attribute for each step.

    A step that is a number indexes a list, and a step `name[key=value]` takes the item of
    the list `name` whose attribute `key` is `value`. For example, `agent.agents.1.robot`
    reads the robot of the second agent in the list `agent.agents`, and
    `agent.agents[uid=panda_leaphand].robot` the robot of the agent whose `uid` is
    `panda_leaphand`.

    Raises:
        ValueError: If a selector step matches no item, or more than one.
    """
    value = start
    for step in path.split("."):
        selector = _SELECTOR.match(step)
        if selector is None:
            value = value[int(step)] if step.isdigit() else getattr(value, step)
            continue
        name, key, wanted = selector.groups()
        matches = [item for item in getattr(value, name) if str(getattr(item, key)) == wanted]
        if len(matches) != 1:
            raise ValueError(f"{len(matches)} items of {name} have {key} {wanted}, not one")
        value = matches[0]
    return value


def _scene_link_name_of(name: str, robot_index: int) -> str:
    """The scene's name for the link `name` of the robot at `robot_index` in the config's robots."""
    return name if robot_index == 0 else f"{name}_{robot_index}"


def _scene_link_name(link: Any, robot_index: int) -> str:
    """The scene's name for a link of the robot at `robot_index` in the config's robots."""
    return _scene_link_name_of(link.name, robot_index)


class BenchmarkGenesisRobotConfig(BaseModel):
    """One robot of a Genesis scene: its path from the task's env, its joints and touch links.

    Each field means what the field of the same name on `BenchmarkGenesisSimulatorConfig`
    means.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    robot: str
    arm_joints: tuple[str, ...]
    gripper_joints: tuple[str, ...]
    touch_links: dict[str, Literal["arm", "gripper"] | BenchmarkHandTouchLinkConfig]


class BenchmarkGenesisSimulatorConfig(BaseModel):
    """The Genesis simulator's values from the benchmark's config.

    `path` is the attribute path from the env that `on_load` returns to the task's env,
    such as `gym_env.unwrapped`. The task's env holds the Genesis scene in `scene`, and
    holds each of the task's entities in an attribute. `robot` is the attribute path from
    the task's env to the robot's entity, such as `agent.robot`. A step of the path that
    is a number indexes a list, such as `agent.agents.0.robot`. `arm_joints` and
    `gripper_joints` list the names of the arm's and the gripper's joints, in the order
    that the replay plots them.

    `touch_links` maps each robot link whose touches the replay records to `arm`,
    `gripper` or a hand link, by the link's name. The replay records a `gripper` link's
    touches of fixtures only, because the hooks record the gripper's touches of the task's
    objects. A hand link, such as `{"hand": "0", "thumb": false}`, is a link of a dexterous
    hand on a robot that has no gripper command. The replay records its touches as touches
    by that hand, and records a grasp from which links of which hand touch an object.
    `gripper_joints` then lists the hand's joints.

    `other_robots` lists the robots of the scene after the first, such as the second arm
    of a robot that ManiLadder builds from two Pandas. The replay names each of their
    links with its robot's position in the list after it, from `_1`, and plots each of
    their joints with the same suffix. It is empty for a scene with one robot.
    """

    # A config states every value, so an unknown key is an error.
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["genesis"]
    path: str
    robot: str
    arm_joints: tuple[str, ...]
    gripper_joints: tuple[str, ...]
    touch_links: dict[str, Literal["arm", "gripper"] | BenchmarkHandTouchLinkConfig]
    other_robots: tuple[BenchmarkGenesisRobotConfig, ...] = ()

    @property
    def robots(self) -> tuple[BenchmarkGenesisRobotConfig, ...]:
        """Each robot of the scene: the first from the config's own fields, then the others."""
        first = BenchmarkGenesisRobotConfig(
            robot=self.robot,
            arm_joints=self.arm_joints,
            gripper_joints=self.gripper_joints,
            touch_links=self.touch_links,
        )
        return (first, *self.other_robots)

    @property
    def scene_touch_links(self) -> dict[str, str | BenchmarkHandTouchLinkConfig]:
        """Each robot's touch links, by each link's name in the replay's scene."""
        return {
            _scene_link_name_of(link, index): part
            for index, robot in enumerate(self.robots)
            for link, part in robot.touch_links.items()
        }


class GenesisSimulator(Simulator):
    """Simulator that reads the scene, poses, touches and scalars from a Genesis scene."""

    def __init__(
        self, config: BenchmarkGenesisSimulatorConfig, finger_pads: tuple[str, ...] = ()
    ) -> None:
        self._config = config
        self._finger_pads = finger_pads
        # Map each env id to the env that `on_load` returned.
        self._envs: dict[int, Any] = {}
        self._caches: dict[int, _GenesisEpisodeCache] = {}

    @property
    def supported_versions(self) -> dict[str, str]:
        """The version range of Genesis that this simulator reads."""
        return {"genesis-world": ">=0.3.6,<0.4"}

    def start(self) -> None:
        """Do nothing, because Genesis runs in the benchmark's process."""

    def stop(self) -> None:
        """Do nothing, because Genesis runs in the benchmark's process."""

    def attach(self, env: Any, env_ids: Sequence[int]) -> None:
        """Store `env` as env `env_ids[0]`.

        Each episode looks up the task's env at the config's path in `env`, because a
        framework can build a new env or a new scene between episodes.

        Raises:
            ValueError: If `env_ids` contains more than one id, because this simulator
                reads a scene built for one env.
        """
        if len(env_ids) != 1:
            raise ValueError(
                f"a Genesis scene read by the replay holds one env, not {len(env_ids)}"
            )
        self._envs[env_ids[0]] = env

    def post_step(self) -> None:
        """Keep nothing, because each replay method reads the scene when it is called."""

    def replay_bodies(self, env_id: int) -> tuple[ReplaySceneBody, ...]:
        """Each link with a visual geom, read again at the start of each episode.

        This method also finds the touch links, the finger pads and the robot's joints,
        and caches them for the episode.

        Raises:
            ValueError: If the scene was built for several envs, if a robot has no link
                named after one of its touch links, or if no robot has a link named after
                a finger pad.
        """
        import genesis as gs  # ty: ignore[unresolved-import]

        task_env = _follow(self._envs[env_id], self._config.path)
        scene = task_env.scene
        solver = scene.rigid_solver
        if solver.n_envs != 0:
            raise ValueError(f"the replay reads a scene of one env, not {solver.n_envs} envs")
        configs = self._config.robots
        robots = [_follow(task_env, config.robot) for config in configs]
        # The position of each robot's entity in the config's robots, by the entity's id.
        robot_indices = {id(robot): index for index, robot in enumerate(robots)}
        names = self._entity_names(task_env, solver.entities)
        bodies: dict[str, int] = {}
        scene_bodies = []
        things: dict[int, tuple[str, bool]] = {}
        for entity in solver.entities:
            robot_index = robot_indices.get(id(entity))
            is_robot = robot_index is not None
            entity_name = names.get(id(entity))
            for link in entity.links:
                role, label = self._link_label(link, entity_name, is_robot=is_robot)
                if label is not None:
                    things[link.idx] = (label, role is ReplaySceneBodyRole.OBJECT)
                if isinstance(entity.morph, gs.morphs.Plane) or not link.vgeoms:
                    continue
                name = (
                    _scene_link_name(link, robot_index)
                    if robot_index is not None
                    else self._body_name(entity, link, entity_name)
                )
                name = name if name not in bodies else f"{name}#{link.idx}"
                bodies[name] = link.idx
                scene_bodies.append(
                    ReplaySceneBody(
                        name=name,
                        parts=tuple(self._part(vgeom) for vgeom in link.vgeoms),
                        role=role,
                        object=label,
                    )
                )
        parts: dict[int, str] = {}
        pads: dict[int, str] = {}
        for index, (robot, config) in enumerate(zip(robots, configs, strict=True)):
            missing = sorted(set(config.touch_links) - {link.name for link in robot.links})
            if missing:
                raise ValueError(f"the robot {config.robot} has no link {', '.join(missing)}")
            parts.update(
                (link.idx, self._part_name(config.touch_links[link.name], link, index))
                for link in robot.links
                if link.name in config.touch_links
            )
            pads.update(
                (link.idx, _scene_link_name(link, index))
                for link in robot.links
                if _scene_link_name(link, index) in self._finger_pads
            )
        missing = sorted(set(self._finger_pads) - set(pads.values()))
        if missing:
            raise ValueError(f"the robot has no link {', '.join(missing)}")
        self._caches[env_id] = _GenesisEpisodeCache(
            scene=scene,
            bodies=bodies,
            parts=parts,
            pads=pads,
            things=things,
            robots=tuple(
                (robot, *self._robot_dofs(robot, config))
                for robot, config in zip(robots, configs, strict=True)
            ),
        )
        return tuple(scene_bodies)

    def replay_poses(self, env_id: int) -> dict[str, ReplayPose]:
        """Each scene body's world pose, from its link's position and orientation."""
        cache = self._caches[env_id]
        solver = cache.scene.rigid_solver
        positions = self._numpy(solver.get_links_pos())
        orientations = self._numpy(solver.get_links_quat())
        return {
            name: ReplayPose(
                position=positions[index].astype(np.float32),
                orientation=wxyz_to_xyzw(orientations[index]).astype(np.float32),
            )
            for name, index in cache.bodies.items()
        }

    def replay_extrinsics(self, env_id: int) -> dict[str, ReplayPose]:
        """An empty dict, because this simulator poses no cameras."""
        return {}

    def replay_images(self, env_id: int) -> dict[str, np.ndarray]:
        """An empty dict, because a Genesis benchmark sends the policy each camera of its replay."""
        return {}

    def replay_finger_pads(self) -> tuple[str, ...]:
        """The body names of the finger pads in the benchmark's config."""
        return self._finger_pads

    def replay_contacts(self, env_id: int) -> frozenset[tuple[str, str]]:
        """Each robot part that touches a named body at the last physics step, with the body.

        The contacts are those that the solver found on the last physics step of the
        action. A finger pad's contacts appear under the pad's body name. A touch that
        starts and ends between two actions is missed.
        """
        cache = self._caches[env_id]
        contacts = cache.scene.rigid_solver.collider.get_contacts(as_tensor=True, to_torch=False)
        forces = np.linalg.norm(
            np.asarray(contacts["force"], dtype=np.float64).reshape(-1, 3), axis=1
        )
        touching = set()
        for first, second, force in zip(
            np.asarray(contacts["link_a"]).reshape(-1).tolist(),
            np.asarray(contacts["link_b"]).reshape(-1).tolist(),
            forces.tolist(),
            strict=True,
        ):
            if force >= _TOUCH_NEWTONS:
                touching.update(touches(first, second, cache.things, cache.parts, cache.pads))
        return frozenset(touching)

    def replay_scalars(self, env_id: int, ee_position: np.ndarray | None) -> list[RerunScalar]:
        """Each robot's arm and gripper joints and torques, from its degrees of freedom.

        The torques are the forces that the robot's commands give its degrees of freedom
        in the current state, within each degree of freedom's force range. The scalars of
        each robot after the first end in the robot's suffix, such as `joints_1`.
        """
        scalars: list[RerunScalar] = []
        for index, (robot, arm_dofs, gripper_dofs) in enumerate(self._caches[env_id].robots):
            suffix = "" if index == 0 else f"_{index}"
            qpos = self._numpy(robot.get_dofs_position())
            qvel = self._numpy(robot.get_dofs_velocity())
            forces = self._numpy(robot.get_dofs_control_force())
            angles = qpos[arm_dofs]
            named = [
                ("joints", RerunTab.ROBOT, angles),
                ("joints_vel", RerunTab.ROBOT, qvel[arm_dofs]),
                ("joint_pos_cos", RerunTab.ROBOT, np.cos(angles)),
                ("joint_pos_sin", RerunTab.ROBOT, np.sin(angles)),
                ("gripper_qvel", RerunTab.ROBOT, qvel[gripper_dofs]),
                ("joint_torques", RerunTab.PHYSICS, forces[arm_dofs]),
                ("gripper_torques", RerunTab.PHYSICS, forces[gripper_dofs]),
                ("qpos_robot", RerunTab.PHYSICS, angles),
                ("qvel_robot", RerunTab.PHYSICS, qvel[arm_dofs]),
                ("qpos_gripper", RerunTab.PHYSICS, qpos[gripper_dofs]),
                ("qvel_gripper", RerunTab.PHYSICS, qvel[gripper_dofs]),
            ]
            for name, tab, values in named:
                scalars += rerun_scalars(f"{name}{suffix}", tab, values)
        return scalars

    @staticmethod
    def _part_name(part: str | BenchmarkHandTouchLinkConfig, link: Any, robot_index: int) -> str:
        """The name of a touch link in a touch: `ARM`, `GRIPPER`, or a hand link's scene name."""
        return part if isinstance(part, str) else _scene_link_name(link, robot_index)

    @staticmethod
    def _robot_dofs(
        robot: Any, config: BenchmarkGenesisRobotConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        """The robot's degrees of freedom of the config's arm joints and gripper joints.

        Each is an index into the robot's degrees of freedom, in the order of
        `get_dofs_position()`.
        """
        arm, gripper = (
            np.array(
                [dof for name in joints for dof in robot.get_joint(name).dofs_idx_local],
                dtype=np.int64,
            )
            for joints in (config.arm_joints, config.gripper_joints)
        )
        return arm, gripper

    @staticmethod
    def _entity_names(task_env: Any, entities: Sequence[Any]) -> dict[int, str]:
        """The name of each entity that an attribute of `task_env` holds, by the entity's id.

        The name is the attribute's name without leading underscores. An entity in a list
        or tuple is named by the attribute and its position, such as `blocks_1`. The first
        attribute that holds an entity names it.
        """
        ids = {id(entity) for entity in entities}
        names: dict[int, str] = {}
        for attribute, value in vars(task_env).items():
            stem = attribute.lstrip("_")
            held = (
                [(f"{stem}_{index}", item) for index, item in enumerate(value)]
                if isinstance(value, (list, tuple))
                else [(stem, value)]
            )
            for name, item in held:
                if id(item) in ids:
                    names.setdefault(id(item), name)
        return names

    @staticmethod
    def _link_label(
        link: Any, entity_name: str | None, *, is_robot: bool
    ) -> tuple[ReplaySceneBodyRole, str | None]:
        """The link's role, and the task's name for it.

        A robot link has no name for the task. A link that cannot move is a fixture under
        its entity's name, or `TABLE` for an unnamed entity. A link that can move is an
        object under its entity's name, or has no name for an unnamed entity.
        """
        if is_robot:
            return ReplaySceneBodyRole.ROBOT, None
        if link.is_fixed:
            return ReplaySceneBodyRole.FIXTURE, entity_name or TABLE
        return ReplaySceneBodyRole.OBJECT, entity_name

    @staticmethod
    def _body_name(entity: Any, link: Any, entity_name: str | None) -> str:
        """The scene's name for a link outside the robots.

        A link of a named entity with one link has the entity's name, and a link of a
        named entity with several links has the entity's name and the link's. A link of an
        unnamed entity is `entity<index>_<link>`.
        """
        stem = entity_name or f"entity{entity.idx}"
        return stem if entity_name is not None and len(entity.links) == 1 else f"{stem}_{link.name}"

    @staticmethod
    def _part(vgeom: Any) -> ReplayScenePart:
        """A visual geom as a part, posed in its link's frame and coloured by its surface.

        A geom with more than `BOX_VERTICES` vertices becomes its bounding box.
        """
        vertices = np.asarray(vgeom.init_vverts, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(vgeom.init_vfaces, dtype=np.int32).reshape(-1, 3)
        position = np.asarray(vgeom.init_pos, dtype=np.float64).reshape(3)
        orientation = wxyz_to_xyzw(np.asarray(vgeom.init_quat, dtype=np.float64))
        colour = GenesisSimulator._surface_colour(vgeom.vmesh.surface)
        if len(vertices) > BOX_VERTICES:
            rotation = rotation_matrix_from_xyzw(orientation)
            local = ReplayMesh(vertices=vertices @ rotation.T + position, faces=faces)
            return bounding_box_part([ReplayScenePart(mesh=local, color=colour)])
        return ReplayScenePart(
            mesh=ReplayMesh(vertices=vertices.astype(np.float32), faces=faces),
            position=position.astype(np.float32),
            orientation=orientation.astype(np.float32),
            color=colour,
        )

    @staticmethod
    def _surface_colour(surface: Any) -> tuple[float, float, float, float]:
        """The surface's colour, else the mean colour of its image, else grey."""
        texture = surface.get_rgba()
        colour = getattr(texture, "color", None)
        if colour is not None and len(colour) == 4:
            red, green, blue, alpha = (float(value) for value in colour)
            return red, green, blue, alpha
        image = getattr(texture, "image_array", None)
        if image is None or np.ndim(image) != 3 or np.shape(image)[-1] < 3:
            return GREY
        mean = np.asarray(image, dtype=np.float64)[..., :3].reshape(-1, 3).mean(axis=0) / 255.0
        factor = np.asarray(getattr(texture, "image_color", None) or (1.0, 1.0, 1.0))[:3]
        red, green, blue = (float(value) for value in mean * factor)
        return red, green, blue, 1.0

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        """A tensor or an array as a numpy array of floats on the host."""
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float64)


__all__ = [
    "BenchmarkGenesisRobotConfig",
    "BenchmarkGenesisSimulatorConfig",
    "GenesisSimulator",
]
