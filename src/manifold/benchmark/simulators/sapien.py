"""The SAPIEN simulator, which reads the scene, poses and scalars of SAPIEN 2 and 3.

The replay scene contains each body with a drawn shape, such as a table, an object
or a robot link. SAPIEN 3 lists these bodies as entities with a
`RenderBodyComponent`. SAPIEN 2 lists them as actors with visual bodies, and the
links of each articulation are actors too. Both versions use the same names for
articulations, joints, poses and camera matrices, so the rest of this module reads
both.

This module scales each shape's vertices and poses them in its body's frame. SAPIEN
gives a box as a unit box with its half size, a SAPIEN 2 sphere as a unit sphere
with its radius, and a mesh in its file's units with its `scale`. A shape with more
than `BOX_VERTICES` vertices is drawn as its bounding box. The scene leaves out
planes, such as the ground. SAPIEN orders quaternions `wxyz`, and this module
returns them in `xyzw` order.

The config lists the arm's and the gripper's joints by name, because a SAPIEN robot
is a single articulation, and its joints have the robot's own names. This module
imports `sapien` inside the functions that need it, so the SDK installs and imports
without it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from manifold.benchmark.simulators import (
    BOX_VERTICES,
    Simulator,
    bounding_box_part,
)
from manifold.core.conventions import CameraAxes
from manifold.lib.rotation import (
    convert_camera_axes,
    rotation_matrix_from_xyzw,
    wxyz_to_xyzw,
    xyzw_from_rotation_matrix,
)
from manifold.replay import (
    ReplayMesh,
    ReplayPose,
    ReplaySceneBody,
    ReplayScenePart,
    RerunScalar,
    RerunTab,
    rerun_scalars,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_logger = logging.getLogger(__name__)

# The colour of a part without a material.
_GREY = (0.7, 0.7, 0.7, 1.0)


@dataclass(frozen=True)
class _SapienEpisodeCache:
    """Values that `replay_bodies` reads from the scene at the start of an episode.

    The scene stays the same during an episode, so the other replay methods use these
    values at each step instead of searching the scene again.

    `bodies` maps the name of each scene body to its SAPIEN 3 entity or SAPIEN 2
    actor. `cameras` maps each spec camera that the replay poses to its SAPIEN camera,
    which has `get_model_matrix`. `articulation` is the articulation with the config's
    arm joints, or None if the scene has no such articulation. `arm_joints` and
    `gripper_joints` index the articulation's active joints.
    """

    bodies: dict[str, Any]
    cameras: dict[str, Any]
    articulation: Any
    arm_joints: np.ndarray
    gripper_joints: np.ndarray


class BenchmarkSapienSimulatorConfig(BaseModel):
    """The SAPIEN simulator's values from the benchmark's config.

    `path` is the attribute path from the env that `on_load` returns to the
    `sapien.Scene`, such as `task_env.scene`. `cameras` maps each spec camera that the
    replay poses to the name of its `RenderCameraComponent`. `arm_joints` and
    `gripper_joints` list the names of the arm's and the gripper's joints, in the
    order that the replay plots them.
    """

    # A config states every value, so an unknown key is an error.
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["sapien"]
    path: str
    cameras: dict[str, str]
    arm_joints: tuple[str, ...]
    gripper_joints: tuple[str, ...]


class SapienSimulator(Simulator):
    """Simulator that reads the scene, poses and scalars from a SAPIEN scene for each env."""

    def __init__(self, config: BenchmarkSapienSimulatorConfig) -> None:
        self._config = config
        # Map each env id to the env that `on_load` returned.
        self._envs: dict[int, Any] = {}
        self._caches: dict[int, _SapienEpisodeCache] = {}

    @property
    def supported_versions(self) -> dict[str, str]:
        """The version range of SAPIEN that this simulator reads."""
        return {"sapien": ">=2.2,<3.1"}

    def start(self) -> None:
        """Do nothing, because SAPIEN runs in the benchmark's process."""

    def stop(self) -> None:
        """Do nothing, because SAPIEN runs in the benchmark's process."""

    def attach(self, env: Any, env_ids: Sequence[int]) -> None:
        """Store `env` as env `env_ids[0]`.

        Each read looks up the scene at the config's path in `env`, because a
        framework can build a new scene for each episode.

        Raises:
            ValueError: If `env_ids` contains more than one id, because a SAPIEN
                scene contains one env.
        """
        if len(env_ids) != 1:
            raise ValueError(f"a SAPIEN scene holds one env, not {len(env_ids)}")
        self._envs[env_ids[0]] = env

    def replay_bodies(self, env_id: int) -> tuple[ReplaySceneBody, ...]:
        """Each body with a drawn shape, read again at the start of each episode.

        A SAPIEN 3 scene has `get_entities`, and a SAPIEN 2 scene does not. This
        method also finds the cameras and the robot's joints, and caches them for the
        episode.
        """
        scene = self._scene(env_id)
        if hasattr(scene, "get_entities"):
            found, cameras = self._sapien3_bodies(scene)
        else:
            found, cameras = self._sapien2_bodies(scene)
        bodies: dict[str, Any] = {}
        scene_bodies = []
        for body, scene_id, parts in found:
            name = self._body_name(body.get_name(), scene_id, bodies)
            bodies[name] = body
            scene_bodies.append(ReplaySceneBody(name=name, parts=tuple(parts)))
        articulation, arm, gripper = self._robot_joints(scene)
        self._caches[env_id] = _SapienEpisodeCache(
            bodies=bodies,
            cameras=cameras,
            articulation=articulation,
            arm_joints=arm,
            gripper_joints=gripper,
        )
        return tuple(scene_bodies)

    def replay_poses(self, env_id: int) -> dict[str, ReplayPose]:
        """Each scene body's world pose, from the pose of its entity or actor."""
        return {
            name: self._pose(body.get_pose()) for name, body in self._caches[env_id].bodies.items()
        }

    def replay_extrinsics(self, env_id: int) -> dict[str, ReplayPose]:
        """Each camera that the config maps, as its camera-to-world pose in OpenCV axes.

        SAPIEN's model matrix is the camera-to-world pose in OpenGL axes.
        """
        posed = {}
        for name, camera in self._caches[env_id].cameras.items():
            model = np.asarray(camera.get_model_matrix(), dtype=float)
            rotation = convert_camera_axes(model[:3, :3], CameraAxes.OPENGL, CameraAxes.OPENCV)
            posed[name] = ReplayPose(
                position=model[:3, 3].astype(np.float32),
                orientation=xyzw_from_rotation_matrix(rotation),
            )
        return posed

    def replay_images(self, env_id: int) -> dict[str, np.ndarray]:
        """An empty dict, because a SAPIEN benchmark sends the policy each camera of its replay."""
        return {}

    def replay_contacts(self, env_id: int) -> frozenset[tuple[str, str]]:
        """An empty set, because contacts are not implemented for this simulator."""
        return frozenset()

    def replay_scalars(self, env_id: int, ee_position: np.ndarray | None) -> list[RerunScalar]:
        """The arm's and the gripper's joints and forces, from the articulation.

        The forces are `get_qf()`, the joint forces of the last step. If the scene has
        no robot articulation, this method returns an empty list.
        """
        cache = self._caches[env_id]
        if cache.articulation is None:
            return []
        qpos = np.asarray(cache.articulation.get_qpos(), dtype=np.float64)
        qvel = np.asarray(cache.articulation.get_qvel(), dtype=np.float64)
        forces = np.asarray(cache.articulation.get_qf(), dtype=np.float64)
        angles = qpos[cache.arm_joints]
        return [
            *rerun_scalars("joints", RerunTab.ROBOT, angles),
            *rerun_scalars("joints_vel", RerunTab.ROBOT, qvel[cache.arm_joints]),
            *rerun_scalars("joint_pos_cos", RerunTab.ROBOT, np.cos(angles)),
            *rerun_scalars("joint_pos_sin", RerunTab.ROBOT, np.sin(angles)),
            *rerun_scalars("gripper_qvel", RerunTab.ROBOT, qvel[cache.gripper_joints]),
            *rerun_scalars("joint_torques", RerunTab.PHYSICS, forces[cache.arm_joints]),
            *rerun_scalars("gripper_torques", RerunTab.PHYSICS, forces[cache.gripper_joints]),
            *rerun_scalars("qpos_robot", RerunTab.PHYSICS, angles),
            *rerun_scalars("qvel_robot", RerunTab.PHYSICS, qvel[cache.arm_joints]),
            *rerun_scalars("qpos_gripper", RerunTab.PHYSICS, qpos[cache.gripper_joints]),
            *rerun_scalars("qvel_gripper", RerunTab.PHYSICS, qvel[cache.gripper_joints]),
        ]

    def _sapien3_bodies(self, scene: Any) -> tuple[list[tuple[Any, int, list]], dict[str, Any]]:
        """Each SAPIEN 3 entity with a drawn shape, its scene id and parts, and the cameras."""
        # The SDK does not depend on `sapien`, so this import is local.
        import sapien  # ty: ignore[unresolved-import]

        camera_names = {camera: spec for spec, camera in self._config.cameras.items()}
        found = []
        cameras: dict[str, Any] = {}
        for entity in scene.get_entities():
            for component in entity.get_components():
                if isinstance(component, sapien.render.RenderCameraComponent):
                    spec_camera = camera_names.get(component.get_name())
                    if spec_camera is not None:
                        cameras[spec_camera] = component
            render_body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
            if render_body is None:
                continue
            parts = [
                part
                for shape in render_body.render_shapes
                if not isinstance(shape, sapien.render.RenderShapePlane)
                for part in self._sapien3_shape_parts(shape)
            ]
            if parts:
                found.append((entity, entity.per_scene_id, parts))
        return found, cameras

    def _sapien2_bodies(self, scene: Any) -> tuple[list[tuple[Any, int, list]], dict[str, Any]]:
        """Each SAPIEN 2 actor or link with a drawn shape, its id and parts, and the cameras."""
        camera_names = {camera: spec for spec, camera in self._config.cameras.items()}
        cameras = {
            camera_names[camera.get_name()]: camera
            for camera in scene.get_cameras()
            if camera.get_name() in camera_names
        }
        links = [
            link
            for articulation in scene.get_all_articulations()
            for link in articulation.get_links()
        ]
        found = []
        for actor in [*scene.get_all_actors(), *links]:
            parts = [
                part
                for visual_body in actor.get_visual_bodies()
                if visual_body.type != "plane"
                for part in self._sapien2_body_parts(visual_body)
            ]
            if parts:
                found.append((actor, actor.get_id(), parts))
        return found, cameras

    def _scene(self, env_id: int) -> Any:
        """The current SAPIEN scene of env `env_id`, at the config's path from its env."""
        return attrgetter(self._config.path)(self._envs[env_id])

    def _robot_joints(self, scene: Any) -> tuple[Any, np.ndarray, np.ndarray]:
        """The articulation with the config's arm joints, and its arm and gripper joints.

        Each joint is an index into the articulation's active joints, in the order of
        `get_qpos()`. A config name matches the active joint with that name, or else
        the single active joint whose name ends with it. Some frameworks prefix each
        joint's name with the scene and the robot, such as `scene-0-panda_panda_joint1`
        for `panda_joint1`. This method logs each joint that the articulation lacks
        and leaves it out.
        """
        configured = (*self._config.arm_joints, *self._config.gripper_joints)
        for articulation in scene.get_all_articulations():
            names = [joint.get_name() for joint in articulation.get_active_joints()]
            index: dict[str, int] = {}
            for name in configured:
                ends_with_name = [
                    position for position, found in enumerate(names) if found.endswith(name)
                ]
                if name in names:
                    index[name] = names.index(name)
                elif len(ends_with_name) == 1:
                    index[name] = ends_with_name[0]
            if not set(self._config.arm_joints) & set(index):
                continue
            missing = [name for name in configured if name not in index]
            if missing:
                _logger.warning("the articulation has no joints named %s", missing)
            return (
                articulation,
                np.array([index[n] for n in self._config.arm_joints if n in index], dtype=int),
                np.array([index[n] for n in self._config.gripper_joints if n in index], dtype=int),
            )
        _logger.warning("no articulation has the arm joints %s", list(self._config.arm_joints))
        return None, np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    @staticmethod
    def _posed_parts(
        meshes: list[tuple[Any, Any, Any]], size: Any, local: Any
    ) -> list[ReplayScenePart]:
        """Parts from `(vertices, faces, material)` meshes, scaled by `size` and posed by `local`.

        `local` is the shape's pose in its body's frame. If `size` is None, the
        vertices are at full size. A part without a material is grey. A shape with
        more than `BOX_VERTICES` vertices becomes its bounding box.
        """
        scale = np.asarray(size if size is not None else (1.0, 1.0, 1.0), dtype=np.float64)
        rotation = rotation_matrix_from_xyzw(wxyz_to_xyzw(np.asarray(local.q, dtype=np.float64)))
        offset = np.asarray(local.p, dtype=np.float64)
        parts = []
        for vertices, faces, material in meshes:
            scaled = np.asarray(vertices, dtype=np.float64) * scale
            colour = _GREY
            if material is not None:
                red, green, blue, alpha = (float(c) for c in material.base_color)
                colour = (red, green, blue, alpha)
            parts.append(
                ReplayScenePart(
                    mesh=ReplayMesh(
                        vertices=(scaled @ rotation.T + offset).astype(np.float32),
                        faces=np.asarray(faces, dtype=np.int32).reshape(-1, 3),
                    ),
                    color=colour,
                )
            )
        if sum(len(part.mesh.vertices) for part in parts) > BOX_VERTICES:
            return [bounding_box_part(parts)]
        return parts

    @classmethod
    def _sapien3_shape_parts(cls, shape: Any) -> list[ReplayScenePart]:
        """A SAPIEN 3 render shape's parts, scaled by its half size or its scale."""
        size = getattr(shape, "half_size", None)
        if size is None:
            size = getattr(shape, "scale", None)
        meshes = [
            (part.get_vertices(), part.get_triangles(), part.material) for part in shape.get_parts()
        ]
        return cls._posed_parts(meshes, size, shape.local_pose)

    @classmethod
    def _sapien2_body_parts(cls, visual_body: Any) -> list[ReplayScenePart]:
        """A SAPIEN 2 visual body's parts, scaled for its type.

        A box is a unit box with `half_lengths`, a sphere is a unit sphere with
        `radius`, and a mesh has its `scale`. Other types, such as a capsule, have
        vertices at full size.
        """
        if visual_body.type == "box":
            size = visual_body.half_lengths
        elif visual_body.type == "sphere":
            size = (visual_body.radius,) * 3
        elif visual_body.type == "mesh":
            size = visual_body.scale
        else:
            size = None
        meshes = [
            (shape.mesh.vertices, shape.mesh.indices, shape.material)
            for shape in visual_body.get_render_shapes()
        ]
        return cls._posed_parts(meshes, size, visual_body.local_pose)

    @staticmethod
    def _body_name(name: str, scene_id: int, taken: dict[str, Any]) -> str:
        """The body's name, with its scene id added if another body has the name.

        An unnamed body gets the name `entity<id>`. The replay stores poses by body
        name, so a second body with the same name would replace the first.
        """
        name = name or f"entity{scene_id}"
        return name if name not in taken else f"{name}#{scene_id}"

    @staticmethod
    def _pose(pose: Any) -> ReplayPose:
        """A SAPIEN pose as a `ReplayPose`, with its quaternion reordered to `xyzw`."""
        return ReplayPose(
            position=np.asarray(pose.p, dtype=np.float32),
            orientation=wxyz_to_xyzw(np.asarray(pose.q, dtype=np.float64)).astype(np.float32),
        )


__all__ = [
    "BenchmarkSapienSimulatorConfig",
    "SapienSimulator",
]
