"""The MuJoCo simulator, which reads the scene, poses and scalars of a MuJoCo model.

The replay scene contains the geoms in the groups that the config's `geom_groups`
lists, grouped by body. The arm's links and the gripper's fingers are bodies too.
The group of a geom determines whether the cameras draw it. Its collision flags do
not determine this, because some drawn geoms also collide. robosuite draws groups 1
and 2 and puts its collision geoms in group 0, so a robosuite benchmark lists
`[1, 2]`.

The config's prefixes sort the model's joints and bodies into the arm, the gripper,
the rest of the robot, and the objects. robosuite uses the prefixes `robot0_`,
`gripper0_` and `mount0_`.

The simulator reads the model through a handle, such as robosuite's `MjSim`. A
handle is an object with `model` and `data`. This module imports `mujoco` inside
the functions that need it, so the SDK installs and imports without it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from manifold.benchmark.simulators import (
    Simulator,
)
from manifold.core.conventions import CameraAxes
from manifold.lib.rotation import convert_camera_axes, wxyz_to_xyzw, xyzw_from_rotation_matrix
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
    """Values that `replay_bodies` reads from the model at the start of an episode.

    The model stays the same during an episode, so the other replay methods use these
    values at each step instead of searching the model again.

    `body_ids` contains the ids of the bodies with a drawn geom, in the order of the
    scene's bodies.

    The joint fields are integer arrays of indices into MuJoCo's flat state arrays.
    Such an array selects a part's values in a single read, such as
    `data.qpos[cache.arm_qpos]` for the arm's joint angles:

    - `arm_qpos`, `gripper_qpos` and `object_qpos` index `data.qpos`, which contains
      the joints' positions.
    - `arm_dof`, `gripper_dof` and `object_dof` index `data.qvel`, `data.qacc`
      and `data.qfrc_actuator`, which contain the joints' velocities, accelerations
      and forces. MuJoCo calls each of these values a degree of freedom, or dof.

    `objects` maps the name of each object's body to the body's id. An object is a
    body with a joint outside the robot. `contacts` maps the name of each body whose
    contact force the replay plots to the body's id.
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
    """The MuJoCo simulator's values from the benchmark's config.

    `path` is the attribute path from the env that `on_load` returns to the MuJoCo
    handle, such as `sim` for robosuite. `cameras` maps each spec camera that the
    replay poses to the camera's name in the model. `geom_groups` lists the geom
    groups that the cameras draw. `arm_prefix` and `gripper_prefix` select the arm's
    and the gripper's joints. `robot_prefixes` selects the bodies of the robot, and
    `contact_prefixes` selects the bodies whose contact force the replay plots.
    """

    # A config states every value, so an unknown key is an error.
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
    """Simulator that reads the scene, poses and scalars from a MuJoCo handle for each env."""

    def __init__(self, config: BenchmarkMujocoSimulatorConfig) -> None:
        self._config = config
        # Map each env id to the env that `on_load` returned.
        self._envs: dict[int, Any] = {}
        self._caches: dict[int, _MujocoEpisodeCache] = {}

    @property
    def supported_versions(self) -> dict[str, str]:
        """The version ranges of MuJoCo and robosuite that this simulator reads."""
        return {"mujoco": ">=3.2,<3.9", "robosuite": ">=1.4,<1.6"}

    def start(self) -> None:
        """Do nothing, because MuJoCo runs in the benchmark's process."""

    def stop(self) -> None:
        """Do nothing, because MuJoCo runs in the benchmark's process."""

    def attach(self, env: Any, env_ids: Sequence[int]) -> None:
        """Store `env` as env `env_ids[0]`.

        Each read looks up the handle at the config's path in `env`, because a
        framework can replace the handle. robosuite builds a new `MjSim` on each reset.

        Raises:
            ValueError: If `env_ids` contains more than one id, because a MuJoCo
                handle contains one env.
        """
        if len(env_ids) != 1:
            raise ValueError(f"a MuJoCo handle holds one env, not {len(env_ids)}")
        self._envs[env_ids[0]] = env

    def replay_bodies(self, env_id: int) -> tuple[ReplaySceneBody, ...]:
        """Each body with a drawn geom, read again at the start of each episode.

        A task can lay out its objects again on each reset. This method therefore
        also reads the joints, the objects and the contact bodies again, and caches
        them for the episode.
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
        """Each scene body's world pose, from `data.xpos` and `data.xquat`.

        MuJoCo's `MjData`, dm_control's physics and robosuite's wrapper all have these
        two attributes.
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
        """Each camera that the config maps, as its camera-to-world pose in OpenCV axes.

        `cam_xpos` and `cam_xmat` give the camera-to-world pose in OpenGL axes.
        """
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
                orientation=xyzw_from_rotation_matrix(rotation),
            )
        return posed

    def replay_images(self, env_id: int) -> dict[str, np.ndarray]:
        """An empty dict, because a MuJoCo benchmark sends the policy each camera of its replay."""
        return {}

    def replay_scalars(self, env_id: int, ee_position: np.ndarray | None) -> list[RerunScalar]:
        """The arm's and the gripper's joints, the distance to each object, and the forces.

        The arm's and the gripper's torques are `qfrc_actuator` at their joints' dof
        indices. `ee_force` and `ee_torque` are the first six values of `sensordata`,
        which robosuite fills from the wrist's force-torque sensor.
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
        """The current MuJoCo handle of env `env_id`, at the config's path from its env."""
        return attrgetter(self._config.path)(self._envs[env_id])

    def _joint_indices(self, model: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """The indices of the joint values of the arm, the gripper and the objects.

        Both dicts have the keys `arm`, `gripper` and `objects`. The first dict
        contains indices into `qpos`, and the second contains indices into the dof
        arrays. MuJoCo stores the joints' values one after another, so a joint's
        values run from its address to the next joint's address. The values of the
        last joint run to the end.
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
        """Each object's body name and id, for the bodies with a joint outside the robot."""
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
        """Log the drawn geoms that the scene leaves out, so a missing object shows in the log."""
        # The SDK does not depend on `mujoco`, so this import is local.
        import mujoco  # ty: ignore[unresolved-import]

        if len(skipped) == 0:
            return
        described = ", ".join(
            f"{count} {mujoco.mjtGeom(kind).name}" for kind, count in sorted(skipped.items())
        )
        _logger.info("replay scene omits untessellated rendered geoms: %s", described)

    @staticmethod
    def _geom_mesh(model: Any, geom_id: int) -> ReplayMesh | None:
        """The geom's triangles, or None for a geom type that the scene leaves out.

        The scene draws boxes, cylinders and meshes. It leaves out the other types,
        such as planes, height fields and spheres.
        """
        # The SDK does not depend on `mujoco`, so this import is local.
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
        """The texture's mean RGB colour, or None if its pixels are not available.

        MuJoCo 3.2.3 and later store each texture's pixels in `tex_data`, from
        `tex_adr`. Older versions have no `tex_data`, so `_geom_colour` uses the
        material's colour instead.
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

        The geom's own alpha applies in each case. `means` caches the mean colour of
        each texture after the first read.
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
        would replace the first. An unnamed body therefore gets its id in its name.

        This method uses `model.body(id)`, because MuJoCo's `MjModel` has it from 2.3.7
        and robosuite's model wrapper passes it through. robosuite's `body_id2name` is
        absent from MuJoCo and dm_control.
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
