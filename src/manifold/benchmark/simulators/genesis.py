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

The simulator reads rigid entities only. A soft body, such as Genesis' MPM, PBD or FEM
entities, is not in the scene, and the replay records no touch of it.

This module imports `genesis` inside the functions that need it, so the SDK installs
and imports without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from manifold.benchmark.simulators import (
    BOX_VERTICES,
    GREY,
    TABLE,
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
    solver index of each link outside the robot that has a name to that name and to
    whether the link is an object. `robot` is the robot's entity, and `arm_dofs` and
    `gripper_dofs` index its degrees of freedom.
    """

    scene: Any
    bodies: dict[str, int]
    parts: dict[int, str]
    pads: dict[int, str]
    things: dict[int, tuple[str, bool]]
    robot: Any
    arm_dofs: np.ndarray
    gripper_dofs: np.ndarray


class BenchmarkGenesisSimulatorConfig(BaseModel):
    """The Genesis simulator's values from the benchmark's config.

    `path` is the attribute path from the env that `on_load` returns to the task's env,
    such as `gym_env.unwrapped`. The task's env holds the Genesis scene in `scene`, and
    holds each of the task's entities in an attribute. `robot` is the attribute path from
    the task's env to the robot's entity, such as `agent.robot`. `arm_joints` and
    `gripper_joints` list the names of the arm's and the gripper's joints, in the order
    that the replay plots them.

    `touch_links` maps each robot link whose touches the replay records to `arm` or
    `gripper`, by the link's name. The replay records a `gripper` link's touches of
    fixtures only, because the hooks record the gripper's touches of the task's objects.
    """

    # A config states every value, so an unknown key is an error.
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["genesis"]
    path: str
    robot: str
    arm_joints: tuple[str, ...]
    gripper_joints: tuple[str, ...]
    touch_links: dict[str, Literal["arm", "gripper"]]


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
            ValueError: If the scene was built for several envs, or if the robot has no
                link named after a configured touch link or finger pad.
        """
        import genesis as gs  # ty: ignore[unresolved-import]

        task_env = attrgetter(self._config.path)(self._envs[env_id])
        scene = task_env.scene
        solver = scene.rigid_solver
        if solver.n_envs != 0:
            raise ValueError(f"the replay reads a scene of one env, not {solver.n_envs} envs")
        robot = attrgetter(self._config.robot)(task_env)
        names = self._entity_names(task_env, solver.entities)
        bodies: dict[str, int] = {}
        scene_bodies = []
        things: dict[int, tuple[str, bool]] = {}
        for entity in solver.entities:
            is_robot = entity is robot
            entity_name = names.get(id(entity))
            for link in entity.links:
                role, label = self._link_label(link, entity_name, is_robot=is_robot)
                if label is not None:
                    things[link.idx] = (label, role is ReplaySceneBodyRole.OBJECT)
                if isinstance(entity.morph, gs.morphs.Plane) or not link.vgeoms:
                    continue
                name = self._body_name(entity, link, entity_name, is_robot=is_robot)
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
        touch_links = self._config.touch_links
        missing = sorted(
            (set(touch_links) | set(self._finger_pads)) - {link.name for link in robot.links}
        )
        if missing:
            raise ValueError(f"the robot has no link {', '.join(missing)}")
        parts: dict[int, str] = {
            link.idx: touch_links[link.name] for link in robot.links if link.name in touch_links
        }
        pads = {link.idx: link.name for link in robot.links if link.name in self._finger_pads}
        arm_dofs, gripper_dofs = self._robot_dofs(robot)
        self._caches[env_id] = _GenesisEpisodeCache(
            scene=scene,
            bodies=bodies,
            parts=parts,
            pads=pads,
            things=things,
            robot=robot,
            arm_dofs=arm_dofs,
            gripper_dofs=gripper_dofs,
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
        """The arm's and the gripper's joints and torques, from the robot's degrees of freedom.

        The torques are the forces that the robot's commands give its degrees of freedom
        in the current state, within each degree of freedom's force range.
        """
        cache = self._caches[env_id]
        robot = cache.robot
        qpos = self._numpy(robot.get_dofs_position())
        qvel = self._numpy(robot.get_dofs_velocity())
        forces = self._numpy(robot.get_dofs_control_force())
        angles = qpos[cache.arm_dofs]
        return [
            *rerun_scalars("joints", RerunTab.ROBOT, angles),
            *rerun_scalars("joints_vel", RerunTab.ROBOT, qvel[cache.arm_dofs]),
            *rerun_scalars("joint_pos_cos", RerunTab.ROBOT, np.cos(angles)),
            *rerun_scalars("joint_pos_sin", RerunTab.ROBOT, np.sin(angles)),
            *rerun_scalars("gripper_qvel", RerunTab.ROBOT, qvel[cache.gripper_dofs]),
            *rerun_scalars("joint_torques", RerunTab.PHYSICS, forces[cache.arm_dofs]),
            *rerun_scalars("gripper_torques", RerunTab.PHYSICS, forces[cache.gripper_dofs]),
            *rerun_scalars("qpos_robot", RerunTab.PHYSICS, angles),
            *rerun_scalars("qvel_robot", RerunTab.PHYSICS, qvel[cache.arm_dofs]),
            *rerun_scalars("qpos_gripper", RerunTab.PHYSICS, qpos[cache.gripper_dofs]),
            *rerun_scalars("qvel_gripper", RerunTab.PHYSICS, qvel[cache.gripper_dofs]),
        ]

    def _robot_dofs(self, robot: Any) -> tuple[np.ndarray, np.ndarray]:
        """The robot's degrees of freedom of the config's arm joints and gripper joints.

        Each is an index into the robot's degrees of freedom, in the order of
        `get_dofs_position()`.
        """
        arm, gripper = (
            np.array(
                [dof for name in joints for dof in robot.get_joint(name).dofs_idx_local],
                dtype=np.int64,
            )
            for joints in (self._config.arm_joints, self._config.gripper_joints)
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
    def _body_name(entity: Any, link: Any, entity_name: str | None, *, is_robot: bool) -> str:
        """The scene's name for a link.

        A robot link has its own name. A link of a named entity with one link has the
        entity's name, and a link of a named entity with several links has the entity's
        name and the link's. A link of an unnamed entity is `entity<index>_<link>`.
        """
        if is_robot:
            return link.name
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
    "BenchmarkGenesisSimulatorConfig",
    "GenesisSimulator",
]
