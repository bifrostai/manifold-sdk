"""The simulator for MuJoCo: the scene, poses and scalars of a MuJoCo model.

The scene has the geoms of the groups in the config's `geom_groups`, grouped by
body. Every body with such a geom is in the scene, the arm's links and the
gripper's fingers included. robosuite renders groups 1 and 2 and keeps its
collision geoms in group 0, so a benchmark on robosuite lists `[1, 2]`. A geom's
group decides whether it is drawn, and its collision flags do not, because some
drawn geoms also collide.

The config's prefixes sort the model's joints and bodies into the arm, the gripper,
the rest of the robot, and the objects. robosuite names them `robot0_`, `gripper0_`
and `mount0_`.

The handle is the one that the benchmark's framework holds, such as robosuite's
`MjSim`, with `model` and `data`. This module imports `mujoco` inside the functions
that need it, so the SDK installs and imports without it; a MuJoCo benchmark's
image has it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from manifold.benchmark.simulators import Simulator, follow
from manifold.core.conventions import CameraAxes, RotationFormat
from manifold.lib.rotation import convert_camera_axes, from_matrix, wxyz_to_xyzw
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


@dataclass(frozen=True)
class _MujocoEpisodeCache:
    """What `replay_bodies` found in the model at the start of an episode, for each later step.

    The model does not change during an episode, so `replay_poses` and `replay_scalars`
    read these fields instead of searching the model again at each step.

    `body_ids` lists the ids of the bodies that have a drawn geom, in the order of
    the scene's bodies.

    The joint fields are integer arrays of indices into MuJoCo's flat state arrays.
    An index array selects a part's values in a single read, such as
    `data.qpos[cache.arm_qpos]` for the arm's joint angles:

    - `arm_qpos`, `gripper_qpos` and `object_qpos` index `data.qpos`, which has
      the joints' positions.
    - `arm_dof`, `gripper_dof` and `object_dof` index `data.qvel`, `data.qacc`
      and `data.qfrc_actuator`, which have the joints' velocities, accelerations
      and forces. MuJoCo calls each of these values a degree of freedom, or dof.

    `objects` maps an object's body name to its body id. An object is a body with a
    joint that is not part of the robot. `contacts` maps the name of a body whose
    contact force is plotted to its body id.
    """

    body_ids: tuple[int, ...]
    arm_qpos: np.ndarray
    arm_dof: np.ndarray
    gripper_qpos: np.ndarray
    gripper_dof: np.ndarray
    object_qpos: np.ndarray
    object_dof: np.ndarray
    objects: dict[str, int]
    contacts: dict[str, int]


class BenchmarkMujocoSimulatorConfig(BaseModel):
    """Where a MuJoCo simulator is, and the names of the robot's parts in its model.

    `path` is the attribute path from the env that `on_load` returns to the
    simulator's handle, which has `model` and `data`, such as `env.sim` for robosuite.
    `cameras` maps each spec camera that the replay poses to the model's camera
    name. `geom_groups` are the geom groups that the cameras render. The prefixes
    sort the model's joints and bodies: the arm's joints, the gripper's joints,
    the bodies that are part of the robot rather than an object, and the bodies
    whose contact force the replay plots.
    """

    # A config spells out every value, so it takes no key that it does not list.
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["mujoco"]
    path: str
    cameras: dict[str, str]
    geom_groups: tuple[int, ...]
    arm_prefix: str
    gripper_prefix: str
    robot_prefixes: tuple[str, ...]
    contact_prefixes: tuple[str, ...]


class MujocoSimulator(Simulator):
    """Read the scene, poses and scalars of MuJoCo simulators, one handle for each env."""

    def __init__(self, config: BenchmarkMujocoSimulatorConfig) -> None:
        self._config = config
        # The env that `on_load` returned, for each env id. A framework can build a new
        # handle on each reset, so each read follows the config's path from the env.
        self._envs: dict[int, Any] = {}
        self._caches: dict[int, _MujocoEpisodeCache] = {}

    def start(self) -> None:
        """MuJoCo runs in the benchmark's own process, so there is nothing to start."""

    def stop(self) -> None:
        """MuJoCo runs in the benchmark's own process, so there is nothing to stop."""

    def attach(self, env: Any, env_ids: Sequence[int]) -> None:
        """Read env `env_ids[0]` from the MuJoCo handle at the config's path in `env`.

        The handle is looked up at each read, because a framework can replace it,
        such as robosuite, which builds a new `MjSim` on each reset.

        Raises:
            ValueError: If `env_ids` holds more than one id, because a MuJoCo
                handle holds one env.
        """
        if len(env_ids) != 1:
            raise ValueError(f"a MuJoCo handle holds one env, not {len(env_ids)}")
        self._envs[env_ids[0]] = env

    def replay_bodies(self, env_id: int) -> tuple[ReplaySceneBody, ...]:
        """Each body with a drawn geom, read again for each episode.

        A task can lay its objects out again on each reset, so the scene, the
        joints, the objects and the contact bodies are read at the start of each
        episode.
        """
        sim = self._sim(env_id)
        model = sim.model
        drawn = frozenset(self._config.geom_groups)
        parts: dict[int, list[ReplayScenePart]] = {}
        skipped: dict[int, int] = {}
        means: dict[int, tuple[float, float, float] | None] = {}
        for geom_id in range(model.ngeom):
            if int(model.geom_group[geom_id]) not in drawn:
                continue
            mesh = self._geom_mesh(model, geom_id)
            if mesh is None:
                kind = int(model.geom_type[geom_id])
                skipped[kind] = skipped.get(kind, 0) + 1
                continue
            parts.setdefault(int(model.geom_bodyid[geom_id]), []).append(
                ReplayScenePart(
                    mesh=mesh,
                    position=np.asarray(model.geom_pos[geom_id], dtype=np.float32),
                    orientation=wxyz_to_xyzw(model.geom_quat[geom_id]).astype(np.float32),
                    color=self._geom_colour(model, geom_id, means),
                )
            )
        self._log_skipped_geoms(skipped)
        qpos, dof = self._joint_indices(model)
        cache = _MujocoEpisodeCache(
            body_ids=tuple(sorted(parts)),
            arm_qpos=qpos["arm"],
            arm_dof=dof["arm"],
            gripper_qpos=qpos["gripper"],
            gripper_dof=dof["gripper"],
            object_qpos=qpos["objects"],
            object_dof=dof["objects"],
            objects=self._object_bodies(model),
            contacts={
                name: body_id
                for body_id in range(int(model.nbody))
                if (name := self._body_name(model, body_id)).startswith(
                    self._config.contact_prefixes
                )
            },
        )
        self._caches[env_id] = cache
        return tuple(
            ReplaySceneBody(name=self._body_name(model, body_id), parts=tuple(parts[body_id]))
            for body_id in cache.body_ids
        )

    def replay_poses(self, env_id: int) -> dict[str, ReplayPose]:
        """Each scene body's world pose now, from `data.xpos` and `data.xquat`.

        MuJoCo's `MjData`, dm_control's physics and robosuite's wrapper all have these
        two names.
        """
        sim = self._sim(env_id)
        data = sim.data
        return {
            self._body_name(sim.model, body_id): ReplayPose(
                position=np.asarray(data.xpos[body_id], dtype=np.float32),
                orientation=wxyz_to_xyzw(data.xquat[body_id]).astype(np.float32),
            )
            for body_id in self._caches[env_id].body_ids
        }

    def replay_extrinsics(self, env_id: int) -> dict[str, ReplayPose]:
        """Each camera that the config maps, as its camera-to-world pose in OpenCV axes."""
        sim = self._sim(env_id)
        model, data = sim.model, sim.data
        posed = {}
        for name, camera in self._config.cameras.items():
            camera_id = int(model.camera(camera).id)
            rotation = convert_camera_axes(
                np.asarray(data.cam_xmat[camera_id], dtype=float),
                CameraAxes.OPENGL,
                CameraAxes.OPENCV,
            )
            posed[name] = ReplayPose(
                position=np.asarray(data.cam_xpos[camera_id], dtype=np.float32),
                orientation=np.asarray(
                    from_matrix(rotation, RotationFormat.QUATERNION), dtype=np.float32
                ),
            )
        return posed

    def replay_images(self, env_id: int) -> dict[str, np.ndarray]:
        """A MuJoCo benchmark sends every camera that its replay shows to the policy."""
        return {}

    def replay_scalars(self, env_id: int, ee_position: np.ndarray | None) -> list[RerunScalar]:
        """The arm's joints, the objects' distances, and the forces of the physics.

        The torques of the arm and the gripper are `qfrc_actuator` at their joints'
        velocity addresses. `ee_force` and `ee_torque` are the first six values of
        `sensordata`, which robosuite fills from the wrist's force-torque sensor.
        """
        data = self._sim(env_id).data
        cache = self._caches[env_id]
        qpos = np.asarray(data.qpos, dtype=np.float64)
        qvel = np.asarray(data.qvel, dtype=np.float64)
        qacc = np.asarray(data.qacc, dtype=np.float64)
        actuator = np.asarray(data.qfrc_actuator, dtype=np.float64)
        angles = qpos[cache.arm_qpos]
        scalars = [
            *rerun_scalars("joints", RerunTab.ROBOT, angles),
            *rerun_scalars("joints_vel", RerunTab.ROBOT, qvel[cache.arm_dof]),
            *rerun_scalars("joints_acc", RerunTab.ROBOT, qacc[cache.arm_dof]),
            *rerun_scalars("joint_pos_cos", RerunTab.ROBOT, np.cos(angles)),
            *rerun_scalars("joint_pos_sin", RerunTab.ROBOT, np.sin(angles)),
            *rerun_scalars("gripper_qvel", RerunTab.ROBOT, qvel[cache.gripper_dof]),
        ]
        positions = data.xpos
        for name, body_id in cache.objects.items():
            distance = 0.0
            if ee_position is not None:
                offset = np.asarray(positions[body_id], dtype=np.float64) - ee_position[:3]
                distance = float(np.linalg.norm(offset))
            scalars.append(RerunScalar(f"object_dist/{name}", RerunTab.TASK, distance, None))
        external = np.asarray(data.cfrc_ext, dtype=np.float64)
        for name, body_id in cache.contacts.items():
            scalars.append(
                RerunScalar(
                    f"contacts/{name}",
                    RerunTab.PHYSICS,
                    float(np.linalg.norm(external[body_id])),
                    None,
                )
            )
        sensors = np.asarray(getattr(data, "sensordata", ()), dtype=np.float64).reshape(-1)
        scalars += [
            *rerun_scalars("ee_force", RerunTab.PHYSICS, sensors[:3]),
            *rerun_scalars("ee_torque", RerunTab.PHYSICS, sensors[3:6]),
            *rerun_scalars("joint_torques", RerunTab.PHYSICS, actuator[cache.arm_dof]),
            *rerun_scalars("gripper_torques", RerunTab.PHYSICS, actuator[cache.gripper_dof]),
            *rerun_scalars(
                "actuator_force", RerunTab.PHYSICS, np.asarray(data.actuator_force).reshape(-1)
            ),
            *rerun_scalars("ctrl", RerunTab.PHYSICS, np.asarray(data.ctrl).reshape(-1)),
            *rerun_scalars("qpos_robot", RerunTab.PHYSICS, angles),
            *rerun_scalars("qvel_robot", RerunTab.PHYSICS, qvel[cache.arm_dof]),
            *rerun_scalars("qpos_gripper", RerunTab.PHYSICS, qpos[cache.gripper_qpos]),
            *rerun_scalars("qvel_gripper", RerunTab.PHYSICS, qvel[cache.gripper_dof]),
            *rerun_scalars("qpos_objects", RerunTab.PHYSICS, qpos[cache.object_qpos]),
            *rerun_scalars("qvel_objects", RerunTab.PHYSICS, qvel[cache.object_dof]),
        ]
        return scalars

    def _sim(self, env_id: int) -> Any:
        """Env `env_id`'s MuJoCo handle now, at the config's path from its env."""
        return follow(self._envs[env_id], self._config.path)

    def _joint_indices(self, model: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """The indices of each part's joint values, for the arm, the gripper and the objects.

        It returns two dicts with the keys `arm`, `gripper` and `objects`: the first
        has indices into `qpos`, and the second has indices into the dof arrays.
        MuJoCo stores the joints' values one after another, so a joint's values run
        from its address to the next joint's address, or to the end for the last.
        """
        count = int(model.njnt)
        qpos = [*(int(address) for address in model.jnt_qposadr[:count]), int(model.nq)]
        dof = [*(int(address) for address in model.jnt_dofadr[:count]), int(model.nv)]
        found: dict[str, tuple[list[int], list[int]]] = {
            "arm": ([], []),
            "gripper": ([], []),
            "objects": ([], []),
        }
        for joint_id in range(count):
            name = self._joint_name(model, joint_id)
            if name.startswith(self._config.arm_prefix):
                part = "arm"
            elif name.startswith(self._config.gripper_prefix):
                part = "gripper"
            else:
                part = "objects"
            found[part][0].extend(range(qpos[joint_id], qpos[joint_id + 1]))
            found[part][1].extend(range(dof[joint_id], dof[joint_id + 1]))
        return (
            {part: np.array(indices[0], dtype=np.int64) for part, indices in found.items()},
            {part: np.array(indices[1], dtype=np.int64) for part, indices in found.items()},
        )

    def _object_bodies(self, model: Any) -> dict[str, int]:
        """Each object's body name and body id: the bodies with a joint that are not the robot."""
        counts = np.asarray(model.body_jntnum, dtype=np.int64).reshape(-1)
        return {
            name: body_id
            for body_id in range(int(model.nbody))
            if int(counts[body_id]) > 0
            and not (name := self._body_name(model, body_id)).startswith(
                self._config.robot_prefixes
            )
        }

    @staticmethod
    def _log_skipped_geoms(skipped: dict[int, int]) -> None:
        """Log the rendered geoms that the scene leaves out, so a missing object is visible."""
        # The SDK does not depend on `mujoco`. A MuJoCo benchmark image installs it.
        import mujoco  # ty: ignore[unresolved-import]

        if len(skipped) == 0:
            return
        described = ", ".join(
            f"{count} {mujoco.mjtGeom(kind).name}" for kind, count in sorted(skipped.items())
        )
        _logger.info("replay scene omits untessellated rendered geoms: %s", described)

    @staticmethod
    def _geom_mesh(model: Any, geom_id: int) -> ReplayMesh | None:
        """The geom's triangles, or None for a geom type that the scene skips.

        The scene draws boxes, cylinders and meshes. It leaves out planes and height
        fields, and the curved primitives are not turned into triangles yet.
        """
        # The SDK does not depend on `mujoco`. A MuJoCo benchmark image installs it.
        import mujoco  # ty: ignore[unresolved-import]

        kind = int(model.geom_type[geom_id])
        size = np.asarray(model.geom_size[geom_id], dtype=np.float32)
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            return ReplayMesh.box([float(value) for value in size[:3]])
        if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return ReplayMesh.cylinder(radius=float(size[0]), half_length=float(size[1]))
        if kind != mujoco.mjtGeom.mjGEOM_MESH:
            return None
        mesh_id = int(model.geom_dataid[geom_id])
        vertex_start = int(model.mesh_vertadr[mesh_id])
        face_start = int(model.mesh_faceadr[mesh_id])
        vertices = np.asarray(
            model.mesh_vert[vertex_start : vertex_start + int(model.mesh_vertnum[mesh_id])]
        ).reshape(-1, 3)
        faces = np.asarray(
            model.mesh_face[face_start : face_start + int(model.mesh_facenum[mesh_id])]
        ).reshape(-1, 3)
        return ReplayMesh(vertices=vertices, faces=faces.astype(np.uint32))

    @staticmethod
    def _texture_mean(model: Any, texture_id: int) -> tuple[float, float, float] | None:
        """The texture's mean RGB colour, or None when its pixels are not available.

        MuJoCo 3.2.3 and later store every texture in `tex_data`, from `tex_adr`.
        MuJoCo 2.3.7 has no `tex_data` or `tex_nchannel`, and robosuite 1.4.0 allows
        it. Without `tex_data`, this returns None, so the geom takes its material's
        colour.
        """
        data = getattr(model, "tex_data", None)
        if data is None:
            return None
        channels = int(model.tex_nchannel[texture_id])
        count = int(model.tex_width[texture_id]) * int(model.tex_height[texture_id]) * channels
        start = int(model.tex_adr[texture_id])
        pixels = np.asarray(data).reshape(-1)[start : start + count]
        if count == 0 or pixels.size < count:
            return None
        mean = pixels.reshape(-1, channels)[:, :3].astype(np.float32).mean(axis=0) / 255.0
        return float(mean[0]), float(mean[1]), float(mean[2])

    @staticmethod
    def _geom_colour(
        model: Any, geom_id: int, means: dict[int, tuple[float, float, float] | None]
    ) -> tuple[float, float, float, float]:
        """The geom's colour: its texture's mean colour, else its material's, else its own.

        The geom's own alpha is kept in every case. `means` holds each texture's mean
        colour once it is read.
        """
        rgba = np.asarray(model.geom_rgba[geom_id], dtype=np.float32).reshape(-1)
        alpha = float(rgba[3])
        material_id = int(model.geom_matid[geom_id])
        if material_id < 0:
            return float(rgba[0]), float(rgba[1]), float(rgba[2]), alpha
        texture_id = int(np.asarray(model.mat_texid[material_id]).reshape(-1).max())
        if texture_id >= 0:
            if texture_id not in means:
                means[texture_id] = MujocoSimulator._texture_mean(model, texture_id)
            mean = means[texture_id]
            if mean is not None:
                return mean[0], mean[1], mean[2], alpha
        material = np.asarray(model.mat_rgba[material_id], dtype=np.float32).reshape(-1)
        return float(material[0]), float(material[1]), float(material[2]), alpha

    @staticmethod
    def _body_name(model: Any, body_id: int) -> str:
        """The model's name for the body, or `body<id>` for an unnamed body.

        The replay stores poses by body name, so a second body with the same name
        would replace the first. An unnamed body therefore gets its id.

        `model.body(id)` is part of MuJoCo's own `MjModel` from 2.3.7 onwards, and
        robosuite's model wrapper passes it through. robosuite's `body_id2name` is
        absent from MuJoCo itself and from dm_control. mujoco-py has `body_id2name`
        but no `model.body(id)`, so mujoco-py needs other lookups.
        """
        return model.body(body_id).name or f"body{body_id}"

    @staticmethod
    def _joint_name(model: Any, joint_id: int) -> str:
        """The model's name for a joint, or an empty string for an unnamed joint."""
        return model.joint(joint_id).name


__all__ = [
    "BenchmarkMujocoSimulatorConfig",
    "MujocoSimulator",
]
